#!/usr/bin/env python3
"""Oracle shot-cost audit for fixed N2 frames and full-covariance GLS-DQG."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np  # noqa: E402
from scipy.linalg import cho_factor, cho_solve  # noqa: E402


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ROOT / "code"
VENDOR = CODE / "vendor"
for path in (CODE, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constrained_shadow import WeightedFitStatistics, solve_dqg_sdp  # noqa: E402
from mcpdft_derandomization import matrix_to_variable_vector  # noqa: E402
from mcpdft_selector import _symmetric_d2_variables  # noqa: E402
from run_c2_ftpbe_shot_reallocation import AcquisitionOracle  # noqa: E402
from run_c2_oracle_subset_reallocation import (  # noqa: E402
    _hamiltonian_gradient,
    _raw_ftpbe_gradient,
)
from run_hybrid_complex_shadow_ablation import _pair_indicators  # noqa: E402
from run_n2_equilibrium_random_seed_trajectories import (  # noqa: E402
    DEFAULT_SEEDS,
    _build_references,
    _candidate_design,
    _derived_seeds,
    _solver_args,
)
from run_safe_mcpdft_derandomization import _initial_result  # noqa: E402
from run_sweep import _atomic_json, _atomic_npz  # noqa: E402


FRAME_COUNT = 25
MAX_SHOTS_PER_FRAME = 10_000
PILOT_SHOTS = 500
ALLOCATION_CHUNK = 250
BUDGETS = (25_000, 50_000, 75_000, 100_000, 150_000, 200_000, 250_000)
SHRINKAGE = 0.05
RELATIVE_FLOOR = 1e-6
ENERGY_TARGET_MEH = 1.6
D2_GUARD_RATIO = 1.25
METHOD_WEIGHTS = {
    "uniform": None,
    "h_only": np.asarray((0.20, 0.80, 0.00)),
    "ftpbe_task": np.asarray((0.15, 0.15, 0.70)),
}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=ROOT / "sweeps" / "n2_equilibrium_random_m1_30_five_seeds",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "sweeps" / "n2_ftpbe_shot_cost_oracle",
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--bond-length", type=float, default=1.10)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--solver-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args(argv)


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _frame_block(
    vectors: np.ndarray, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    block = np.real(np.conjugate(vectors[:, rows]) * vectors[:, cols])
    block[:, rows != cols] *= 2.0
    return np.asarray(block, dtype=float)


def _regularized_single_covariance(
    outcomes: np.ndarray,
) -> np.ndarray:
    covariance = np.cov(outcomes, rowvar=False, ddof=1)
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(np.maximum(np.diag(covariance), 0.0))
    regularized = (1.0 - SHRINKAGE) * covariance + SHRINKAGE * diagonal
    scale = max(float(np.max(np.diag(diagonal))), 1e-8)
    return regularized + RELATIVE_FLOOR * scale * np.eye(len(covariance))


def _exact_single_covariance(
    determinant_probabilities: np.ndarray, indicators: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = determinant_probabilities @ indicators
    second = (indicators.T * determinant_probabilities) @ indicators
    covariance = second - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = np.diag(np.maximum(np.diag(covariance), 0.0))
    regularized = (1.0 - SHRINKAGE) * covariance + SHRINKAGE * diagonal
    scale = max(float(np.max(np.diag(diagonal))), 1e-8)
    regularized += RELATIVE_FLOOR * scale * np.eye(len(covariance))
    return np.asarray(mean), regularized


def _inverse(matrix: np.ndarray) -> np.ndarray:
    factor = cho_factor(0.5 * (matrix + matrix.T), lower=True, check_finite=False)
    return cho_solve(factor, np.eye(len(matrix)), check_finite=False)


def _risks(
    covariance: np.ndarray,
    d2_weights: np.ndarray,
    hamiltonian: np.ndarray,
    ftpbe: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            float(d2_weights @ np.diag(covariance)),
            float(hamiltonian @ covariance @ hamiltonian),
            float(ftpbe @ covariance @ ftpbe),
        ]
    )


def _allocations(
    updates: Sequence[np.ndarray],
    factors: Sequence[np.ndarray],
    ridge: float,
    normalizers: np.ndarray,
    d2_weights: np.ndarray,
    hamiltonian: np.ndarray,
    ftpbe: np.ndarray,
) -> dict[str, dict[int, np.ndarray]]:
    allocations: dict[str, dict[int, np.ndarray]] = {}
    for method, weights in METHOD_WEIGHTS.items():
        counts = np.full(FRAME_COUNT, PILOT_SHOTS, dtype=int)
        precision = ridge * np.eye(updates[0].shape[0])
        for index, update in enumerate(updates):
            precision += counts[index] * update
        by_budget: dict[int, np.ndarray] = {}
        for budget in BUDGETS:
            while int(np.sum(counts)) < budget:
                available = [
                    index
                    for index in range(FRAME_COUNT)
                    if counts[index] + ALLOCATION_CHUNK <= MAX_SHOTS_PER_FRAME
                ]
                if not available:
                    raise RuntimeError("No frame can accept the remaining shot budget.")
                if method == "uniform":
                    index = min(available, key=lambda item: (counts[item], item))
                else:
                    covariance = _inverse(precision)

                    def benefit(item: int) -> float:
                        projected = factors[item] @ covariance
                        reductions = np.asarray(
                            [
                                float(
                                    np.sum(
                                        np.square(projected)
                                        * d2_weights[None, :]
                                    )
                                ),
                                float(np.sum(np.square(projected @ hamiltonian))),
                                float(np.sum(np.square(projected @ ftpbe))),
                            ]
                        )
                        return float(weights @ (reductions / normalizers))

                    index = max(
                        available,
                        key=lambda item: (
                            benefit(item),
                            -counts[item],
                            -item,
                        ),
                    )
                counts[index] += ALLOCATION_CHUNK
                precision += ALLOCATION_CHUNK * updates[index]
            by_budget[int(budget)] = counts.copy()
        allocations[method] = by_budget
    return allocations


def _full_covariance_statistics(
    blocks: Sequence[np.ndarray],
    frame_outcomes: Sequence[np.ndarray],
    counts: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> WeightedFitStatistics:
    augmented = []
    for block, master, count in zip(blocks, frame_outcomes, counts):
        outcomes = np.asarray(master[: int(count)], dtype=float)
        mean = np.mean(outcomes, axis=0)
        covariance_single = _regularized_single_covariance(outcomes)
        covariance_mean = covariance_single / int(count)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_mean)
        scale = max(float(np.max(eigenvalues)), 1.0 / int(count))
        eigenvalues = np.maximum(eigenvalues, RELATIVE_FLOOR * scale)
        whitener = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
        augmented.append(
            whitener @ np.column_stack((block, -mean))
        )
    matrix = np.vstack(augmented)
    factor = np.linalg.qr(matrix, mode="reduced")[1]
    return WeightedFitStatistics(
        rows=np.asarray(rows),
        cols=np.asarray(cols),
        residual_factor=factor / np.sqrt(matrix.shape[0]),
        observation_count=matrix.shape[0],
    )


def _score_result(
    d2: np.ndarray,
    gamma: np.ndarray,
    exact: Any,
    objective: Any,
    exact_ftpbe: float,
    plateau_d2: float,
) -> dict[str, Any]:
    hamiltonian = float(
        np.sum(exact.one_body * gamma)
        + np.sum(exact.two_body * d2)
        + exact.nuclear_energy
    )
    ftpbe = float(objective.evaluate(d2, gamma, gradient=False).total_energy)
    d2_error = float(np.linalg.norm(d2 - exact.exact_d2))
    h_error = 1000.0 * abs(hamiltonian - exact.exact_energy)
    f_error = 1000.0 * abs(ftpbe - exact_ftpbe)
    passed = bool(
        d2_error <= D2_GUARD_RATIO * plateau_d2
        and h_error <= ENERGY_TARGET_MEH
        and f_error <= ENERGY_TARGET_MEH
    )
    return {
        "d2_frobenius_error": d2_error,
        "d2_ratio_to_random_m25": d2_error / plateau_d2,
        "hamiltonian_error_meh": h_error,
        "ftpbe_error_meh": f_error,
        "joint_target_pass": passed,
    }


def _run_seed(args: argparse.Namespace, seed: int) -> None:
    output = args.output_dir / f"seed_{seed}"
    analysis_path = output / "analysis.json"
    if analysis_path.is_file() and not args.overwrite:
        print(f"[seed={seed}] complete checkpoint", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    selection, exact, objective, exact_ftpbe = _build_references(args)
    random_payload = json.loads(
        (args.trajectory_dir / f"seed_{seed}" / "analysis.json").read_text(
            encoding="utf-8"
        )
    )
    plateau_row = next(row for row in random_payload["rows"] if int(row["m"]) == 25)
    plateau_d2 = float(plateau_row["d2_frobenius_error"])
    real_seed, complex_seed, sampling_seed = _derived_seeds(seed)
    rotations, pair_vectors, is_complex = _candidate_design(
        exact,
        FRAME_COUNT,
        4,
        real_seed,
        complex_seed,
    )
    oracle = AcquisitionOracle(
        exact, rotations, pair_vectors, MAX_SHOTS_PER_FRAME, sampling_seed
    )
    records = [oracle.sample(index, 0) for index in range(FRAME_COUNT)]
    frame_outcomes = tuple(
        _pair_indicators(np.asarray(record["occupations"]), exact.pairs).astype(float)
        for record in records
    )
    variable_rows, variable_cols = _symmetric_d2_variables(exact)
    variables = matrix_to_variable_vector(
        exact.exact_d2, variable_rows, variable_cols
    )
    blocks = []
    updates = []
    factors = []
    for index in range(FRAME_COUNT):
        start = index * len(exact.pairs)
        stop = start + len(exact.pairs)
        block = _frame_block(
            pair_vectors[start:stop], variable_rows, variable_cols
        )
        probabilities, covariance = _exact_single_covariance(
            oracle._frame_probabilities(index), oracle.indicators
        )
        if not np.allclose(probabilities, block @ variables, atol=2e-10):
            raise RuntimeError("Exact frame probabilities disagree with the D2 map.")
        blocks.append(block)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        whitener = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
        factor = whitener @ block
        factors.append(factor)
        updates.append(factor.T @ factor)
    d2_weights = np.where(variable_rows == variable_cols, 1.0, 2.0)
    hamiltonian = _hamiltonian_gradient(exact, variable_rows, variable_cols)
    ftpbe = _raw_ftpbe_gradient(
        objective,
        exact,
        exact.exact_d2,
        exact.exact_gamma,
        variable_rows,
        variable_cols,
    )
    full_information = sum(
        MAX_SHOTS_PER_FRAME * update for update in updates
    )
    ridge = max(
        1e-3 * float(np.trace(full_information)) / len(variable_rows), 1e-12
    )
    full_covariance = _inverse(
        ridge * np.eye(len(variable_rows)) + full_information
    )
    normalizers = _risks(
        full_covariance, d2_weights, hamiltonian, ftpbe
    )
    allocation_paths = _allocations(
        updates,
        factors,
        ridge,
        normalizers,
        d2_weights,
        hamiltonian,
        ftpbe,
    )
    solver_args = _solver_args(args)
    initial = _initial_result(solver_args, selection)
    rows: list[dict[str, Any]] = []
    for method in METHOD_WEIGHTS:
        warm_d2 = np.asarray(initial.d2)
        warm_gamma = np.asarray(initial.gamma)
        for budget in BUDGETS:
            checkpoint = output / "checkpoints" / f"{method}_{budget}.npz"
            metadata_path = checkpoint.with_suffix(".json")
            if checkpoint.is_file() and metadata_path.is_file() and not args.overwrite:
                with np.load(checkpoint, allow_pickle=False) as archive:
                    warm_d2 = np.asarray(archive["d2"])
                    warm_gamma = np.asarray(archive["gamma"])
                row = json.loads(metadata_path.read_text(encoding="utf-8"))
                rows.append(row)
                print(f"[seed={seed} {method} B={budget}] checkpoint", flush=True)
                continue
            counts = allocation_paths[method][budget]
            statistics = _full_covariance_statistics(
                blocks,
                frame_outcomes,
                counts,
                variable_rows,
                variable_cols,
            )
            print(f"[seed={seed} {method} B={budget}] solving", flush=True)
            started = time.perf_counter()
            result = solve_dqg_sdp(
                selection,
                solver=args.solver,
                tolerance=args.solver_tolerance,
                max_iterations=args.max_iterations,
                solver_threads=args.solver_threads,
                positivity_conditions="DQG",
                symmetry_blocked_psd=True,
                weighted_shadow_fit=True,
                weighted_fit_statistics=statistics,
                initial_d2=warm_d2,
                initial_gamma=warm_gamma,
            )
            warm_d2 = np.asarray(result.d2)
            warm_gamma = np.asarray(result.gamma)
            metrics = _score_result(
                warm_d2,
                warm_gamma,
                exact,
                objective,
                exact_ftpbe,
                plateau_d2,
            )
            row = {
                "seed": seed,
                "method": method,
                "total_shots": int(budget),
                "minimum_frame_shots": int(np.min(counts)),
                "maximum_frame_shots": int(np.max(counts)),
                "effective_topped_frames": int(np.count_nonzero(counts > PILOT_SHOTS)),
                "complex_shots": int(np.sum(counts[is_complex])),
                "shot_counts": ",".join(map(str, counts)),
                "solver_seconds": time.perf_counter() - started,
                "status": str(result.status),
                "fit_status": str(result.fit_status),
                **metrics,
            }
            rows.append(row)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            _atomic_npz(checkpoint, d2=warm_d2, gamma=warm_gamma)
            _atomic_json(metadata_path, row)
            print(
                f"[seed={seed} {method} B={budget}] D2={metrics['d2_frobenius_error']:.5f}, "
                f"H={metrics['hamiltonian_error_meh']:.3f}, "
                f"F={metrics['ftpbe_error_meh']:.3f}, "
                f"pass={metrics['joint_target_pass']}",
                flush=True,
            )
    payload = {
        "configuration": {
            "system": "N2",
            "bond_length_angstrom": args.bond_length,
            "frame_count": FRAME_COUNT,
            "frame_pool": "paired random every-fourth-complex prefix",
            "pilot_shots_per_frame": PILOT_SHOTS,
            "allocation_chunk": ALLOCATION_CHUNK,
            "maximum_shots_per_frame": MAX_SHOTS_PER_FRAME,
            "budgets": list(BUDGETS),
            "covariance": "exact allocation covariance; empirical shrinkage GLS reconstruction",
            "shrinkage": SHRINKAGE,
            "oracle_exact_ftpbe_tangent": True,
            "oracle_exact_probabilities": True,
            "cross_fitting_included": False,
        },
        "rows": rows,
    }
    _atomic_csv(output / "results.csv", rows)
    _atomic_json(analysis_path, payload)


def _aggregate(args: argparse.Namespace, seeds: Sequence[int]) -> None:
    rows = []
    for seed in seeds:
        path = args.output_dir / f"seed_{seed}" / "analysis.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed seed analysis: {path}")
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    summary = {}
    for method in METHOD_WEIGHTS:
        per_seed = {}
        for seed in seeds:
            hits = sorted(
                int(row["total_shots"])
                for row in rows
                if row["method"] == method
                and int(row["seed"]) == int(seed)
                and bool(row["joint_target_pass"])
            )
            per_seed[str(seed)] = hits[0] if hits else None
        finite = [value for value in per_seed.values() if value is not None]
        summary[method] = {
            "first_hit_shots_by_seed": per_seed,
            "hit_count": len(finite),
            "median_first_hit_shots": (
                float(np.median(finite)) if finite else None
            ),
        }
    payload = {
        "configuration": {"seeds": list(seeds), "budgets": list(BUDGETS)},
        "summary": summary,
        "rows": rows,
    }
    _atomic_csv(args.output_dir / "oracle_results.csv", rows)
    _atomic_json(args.output_dir / "summary.json", payload)
    print(json.dumps(summary, indent=2), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    args.trajectory_dir = args.trajectory_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    if args.aggregate_only:
        _aggregate(args, seeds)
        return
    for seed in seeds:
        _run_seed(args, int(seed))
    _aggregate(args, seeds)


if __name__ == "__main__":
    main()
