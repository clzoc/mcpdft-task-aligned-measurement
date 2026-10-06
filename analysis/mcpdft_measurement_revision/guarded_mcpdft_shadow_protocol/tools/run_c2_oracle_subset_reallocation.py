#!/usr/bin/env python3
"""Oracle subset and saved-shot reallocation audit for cadence-4 C2 shadows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
from scipy.linalg import solve


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CODE = PROJECT / "code"
VENDOR = CODE / "vendor"
for path in (CODE, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constrained_shadow import (  # noqa: E402
    PairVectorDesign,
    ShadowData,
    _wilson_bounds,
)
from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    LeakGuardShadowData,
    acquire_shadow_bases,
    build_c2_reference,
    build_c2_selection_reference,
    contraction_gamma_gradient_vector,
    load_blind_shadow_npz,
    matrix_to_variable_vector,
    predicted_inverse_variances,
    shadow_design_blocks,
    symmetric_matrix_gradient_vector,
)
from mcpdft_selector import FtPBEEnergyObjective, _symmetric_d2_variables  # noqa: E402
from ridge_selector import weighted_prediction_rmse  # noqa: E402
from run_c2_ftpbe_shot_reallocation import AcquisitionOracle  # noqa: E402
from run_safe_mcpdft_derandomization import (  # noqa: E402
    RECONSTRUCTION_ENERGY,
    _initial_result,
    _solve_shadow_dqg,
)
from run_sweep import _atomic_json, _atomic_npz  # noqa: E402


SIZE_GRID = {
    10: (3, 4, 5, 6, 7, 8, 9),
    20: (6, 8, 10, 12, 14, 16, 18, 19),
    30: (10, 12, 15, 18, 20, 22, 24, 26, 28, 29),
    40: (12, 16, 20, 24, 26, 28, 30, 32, 34, 36, 38, 39),
}
RIDGE_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument(
        "--shadow-archive",
        type=Path,
        help="Optional longer cadence-4 archive; defaults to baseline-dir/hybrid_shadows.npz.",
    )
    parser.add_argument("--frozen-mo-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--counts", default="10,20,30")
    parser.add_argument("--formal-candidates-per-size", type=int, default=1)
    parser.add_argument("--search-seed", type=int, default=2718281)
    parser.add_argument("--reallocation-seed", type=int, default=57721)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--solver-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _parse_counts(value: str) -> tuple[int, ...]:
    counts = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    if not counts or any(count not in SIZE_GRID for count in counts):
        raise ValueError(f"counts must be drawn from {tuple(SIZE_GRID)}.")
    return counts


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _matrix_from_variables(
    variables: np.ndarray, rows: np.ndarray, cols: np.ndarray, dimension: int
) -> np.ndarray:
    matrix = np.zeros((dimension, dimension), dtype=float)
    matrix[rows, cols] = variables
    matrix[cols, rows] = variables
    return matrix


def _raw_ftpbe_gradient(
    objective: FtPBEEnergyObjective,
    reference: Any,
    d2: np.ndarray,
    gamma: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    evaluation = objective.evaluate(d2, gamma, gradient=True)
    gradient = symmetric_matrix_gradient_vector(evaluation.d2_gradient, rows, cols)
    gradient += contraction_gamma_gradient_vector(
        evaluation.gamma_gradient,
        reference.pairs,
        rows,
        cols,
        reference.n_electrons,
    )
    return gradient


def _hamiltonian_gradient(
    reference: Any, rows: np.ndarray, cols: np.ndarray
) -> np.ndarray:
    gradient = symmetric_matrix_gradient_vector(reference.two_body, rows, cols)
    gradient += contraction_gamma_gradient_vector(
        reference.one_body,
        reference.pairs,
        rows,
        cols,
        reference.n_electrons,
    )
    return gradient


def _frame_statistics(
    blocks: Sequence[np.ndarray], shadows: ShadowData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_per_frame = len(shadows.values) // shadows.n_shadows
    information = []
    right_hand_sides = []
    noise_rms = []
    for index, block in enumerate(blocks):
        start = index * rows_per_frame
        stop = start + rows_per_frame
        hits = np.asarray(shadows.hits[start:stop], dtype=float)
        probabilities = (hits + 0.5) / (shadows.shots_per_basis + 1.0)
        variances = np.maximum(
            probabilities * (1.0 - probabilities) / shadows.shots_per_basis,
            1e-12,
        )
        inverse_variances = 1.0 / variances
        weighted = np.sqrt(inverse_variances)[:, None] * block
        information.append(weighted.T @ weighted)
        right_hand_sides.append(
            block.T @ (inverse_variances * shadows.values[start:stop])
        )
        noise_rms.append(
            np.sqrt(
                np.mean(
                    np.square(
                        (shadows.values[start:stop] - shadows.exact_values[start:stop])
                        / np.sqrt(variances)
                    )
                )
            )
        )
    return (
        np.asarray(information),
        np.asarray(right_hand_sides),
        np.asarray(noise_rms),
    )


class OracleSurrogate:
    def __init__(
        self,
        information: np.ndarray,
        right_hand_sides: np.ndarray,
        ridge: float,
        prior: np.ndarray,
        exact: np.ndarray,
        off_diagonal: np.ndarray,
        hamiltonian_gradient: np.ndarray,
        ftpbe_gradient: np.ndarray,
        scales: np.ndarray,
    ):
        self.information = information
        self.right_hand_sides = right_hand_sides
        self.ridge = float(ridge)
        self.prior = np.asarray(prior)
        self.exact = np.asarray(exact)
        self.off_diagonal = np.asarray(off_diagonal, dtype=bool)
        self.hamiltonian_gradient = np.asarray(hamiltonian_gradient)
        self.ftpbe_gradient = np.asarray(ftpbe_gradient)
        self.scales = np.asarray(scales)
        self.eye = np.eye(len(prior))
        self.cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}

    def evaluate(self, subset: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(sorted(int(index) for index in subset))
        if key not in self.cache:
            matrix = self.ridge * self.eye + np.sum(self.information[list(key)], axis=0)
            rhs = self.ridge * self.prior + np.sum(
                self.right_hand_sides[list(key)], axis=0
            )
            variables = solve(
                matrix,
                rhs,
                assume_a="pos",
                check_finite=False,
                overwrite_a=False,
                overwrite_b=False,
            )
            delta = variables - self.exact
            d2_error = float(
                np.sqrt(
                    np.sum(np.square(delta[~self.off_diagonal]))
                    + 2.0 * np.sum(np.square(delta[self.off_diagonal]))
                )
            )
            errors = np.asarray(
                [
                    d2_error,
                    abs(float(self.hamiltonian_gradient @ delta)),
                    abs(float(self.ftpbe_gradient @ delta)),
                ]
            )
            self.cache[key] = variables, errors
        variables, errors = self.cache[key]
        return variables.copy(), errors.copy()

    def ratios(self, subset: Sequence[int]) -> np.ndarray:
        return self.evaluate(subset)[1] / self.scales


def _search_weights(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    fixed = (
        np.asarray((1.0, 1.0, 1.0)) / 3.0,
        np.asarray((0.6, 0.2, 0.2)),
        np.asarray((0.2, 0.6, 0.2)),
        np.asarray((0.2, 0.2, 0.6)),
        np.asarray((0.45, 0.45, 0.1)),
        np.asarray((0.45, 0.1, 0.45)),
        np.asarray((0.1, 0.45, 0.45)),
    )
    random_weights = tuple(rng.dirichlet(np.ones(3)) for _ in range(9))
    return fixed + random_weights


def _scalar_score(ratios: np.ndarray, weights: np.ndarray) -> float:
    clipped = np.maximum(np.asarray(ratios), 1e-12)
    return float(np.sum(weights * np.log(clipped)) + 0.2 * np.max(clipped))


def _forward_path(
    maximum: int, surrogate: OracleSurrogate, weights: np.ndarray
) -> dict[int, tuple[int, ...]]:
    selected: list[int] = []
    path: dict[int, tuple[int, ...]] = {}
    remaining = set(range(maximum))
    while remaining:
        chosen = min(
            remaining,
            key=lambda index: (
                _scalar_score(surrogate.ratios(selected + [index]), weights),
                index,
            ),
        )
        selected.append(chosen)
        remaining.remove(chosen)
        path[len(selected)] = tuple(sorted(selected))
    return path


def _backward_path(
    maximum: int, surrogate: OracleSurrogate, weights: np.ndarray
) -> dict[int, tuple[int, ...]]:
    selected = list(range(maximum))
    path = {maximum: tuple(selected)}
    while len(selected) > 1:
        removed = min(
            selected,
            key=lambda index: (
                _scalar_score(
                    surrogate.ratios([item for item in selected if item != index]),
                    weights,
                ),
                index,
            ),
        )
        selected.remove(removed)
        path[len(selected)] = tuple(selected)
    return path


def _candidate_subsets(
    maximum: int,
    surrogate: OracleSurrogate,
    noise_rms: np.ndarray,
    rng: np.random.Generator,
) -> dict[int, list[tuple[int, ...]]]:
    candidates: dict[int, set[tuple[int, ...]]] = {
        size: set() for size in SIZE_GRID[maximum]
    }
    for weights in _search_weights(rng):
        for path_builder in (_forward_path, _backward_path):
            path = path_builder(maximum, surrogate, weights)
            for size in candidates:
                candidates[size].add(path[size])
    noise_order = np.argsort(noise_rms[:maximum])
    for size in candidates:
        candidates[size].add(tuple(sorted(int(index) for index in noise_order[:size])))
        candidates[size].add(tuple(range(size)))
    return {
        size: sorted(
            subsets,
            key=lambda subset: (
                np.max(surrogate.ratios(subset)),
                np.mean(surrogate.ratios(subset)),
                subset,
            ),
        )
        for size, subsets in candidates.items()
    }


def _score_dqg(
    result: Any,
    exact: Any,
    objective: FtPBEEnergyObjective,
    exact_ftpbe: float,
    acquired: ShadowData,
) -> dict[str, Any]:
    d2 = np.asarray(result.d2)
    gamma = np.asarray(result.gamma)
    ftpbe = float(objective.evaluate(d2, gamma, gradient=False).total_energy)
    hamiltonian = float(
        np.sum(exact.one_body * gamma)
        + np.sum(exact.two_body * d2)
        + exact.nuclear_energy
    )
    return {
        "d2_frobenius_error": float(np.linalg.norm(d2 - exact.exact_d2)),
        "hamiltonian_error_meh": 1000.0 * abs(hamiltonian - exact.exact_energy),
        "ftpbe_error_meh": 1000.0 * abs(ftpbe - exact_ftpbe),
        "hamiltonian_energy_eh": hamiltonian,
        "ftpbe_energy_eh": ftpbe,
        "weighted_fit_rmse": float(weighted_prediction_rmse(acquired, d2)),
    }


def _shadow_from_records(
    rotations: Sequence[np.ndarray],
    pair_vectors: Sequence[np.ndarray],
    hits: Sequence[np.ndarray],
    shots_per_basis: int,
) -> ShadowData:
    stacked_hits = np.concatenate([np.asarray(item, dtype=int) for item in hits])
    values = stacked_hits / shots_per_basis
    lower, upper = _wilson_bounds(stacked_hits, shots_per_basis, 2.0)
    vectors = np.vstack(pair_vectors)
    return ShadowData(
        rotations=tuple(np.asarray(rotation) for rotation in rotations),
        pair_vectors=vectors,
        design=PairVectorDesign(vectors),
        values=values,
        lower_bounds=lower,
        upper_bounds=upper,
        hits=stacked_hits,
        shots_per_basis=shots_per_basis,
        exact_values=np.full_like(values, np.nan),
        exact_constraints=False,
        occupations=None,
    )


def _reallocated_shadows(
    raw: ShadowData,
    subset: Sequence[int],
    counts: dict[int, int],
    oracle: AcquisitionOracle,
) -> ShadowData:
    rows_per_frame = len(raw.values) // raw.n_shadows
    rotations: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    hits: list[np.ndarray] = []
    for index in subset:
        start = index * rows_per_frame
        stop = start + rows_per_frame
        rotations.append(np.asarray(raw.rotations[index]))
        vectors.append(np.asarray(raw.pair_vectors[start:stop]))
        hits.append(np.asarray(raw.hits[start:stop]))
        for repeat in range(counts[int(index)] - 1):
            record = oracle.sample(int(index), repeat)
            rotations.append(np.asarray(record["rotation"]))
            vectors.append(np.asarray(record["pair_vectors"]))
            hits.append(np.asarray(record["hits"]))
    return _shadow_from_records(rotations, vectors, hits, raw.shots_per_basis)


def _equal_allocation(subset: Sequence[int], total_blocks: int) -> dict[int, int]:
    counts = {int(index): 1 for index in subset}
    for position in range(total_blocks - len(subset)):
        counts[int(subset[position % len(subset)])] += 1
    return counts


def _oracle_information_allocation(
    blocks: Sequence[np.ndarray],
    subset: Sequence[int],
    total_blocks: int,
    exact_variables: np.ndarray,
    hamiltonian_gradient: np.ndarray,
    ftpbe_gradient: np.ndarray,
    shots_per_basis: int,
) -> dict[int, int]:
    variable_count = blocks[0].shape[1]
    updates = {}
    for index in subset:
        inverse_variances = predicted_inverse_variances(
            blocks[int(index)],
            exact_variables,
            shots_per_basis,
            probability_floor=0.01,
        )
        weighted = np.sqrt(inverse_variances)[:, None] * blocks[int(index)]
        updates[int(index)] = weighted.T @ weighted
    scale = max(float(np.trace(next(iter(updates.values()))) / variable_count), 1e-12)
    information = 1e-3 * scale * np.eye(variable_count)
    counts = {int(index): 1 for index in subset}
    for index in subset:
        information += updates[int(index)]

    def variances(matrix: np.ndarray) -> np.ndarray:
        inverse = np.linalg.inv(matrix)
        return np.asarray(
            [
                np.trace(inverse),
                hamiltonian_gradient @ inverse @ hamiltonian_gradient,
                ftpbe_gradient @ inverse @ ftpbe_gradient,
            ]
        )

    for _ in range(total_blocks - len(subset)):
        current = variances(information)
        selected = min(
            subset,
            key=lambda index: (
                np.mean(variances(information + updates[int(index)]) / current),
                counts[int(index)],
                int(index),
            ),
        )
        counts[int(selected)] += 1
        information += updates[int(selected)]
    return counts


def _report(
    baselines: dict[int, dict[str, Any]],
    formal_rows: Sequence[dict[str, Any]],
    reallocation_rows: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# C2 cadence-4 complex-shadow oracle subset audit",
        "",
        "Exact RDM information is intentionally used to screen subsets and allocate "
        "saved shots. This is an oracle upper-bound diagnostic, not a deployable "
        "reference-free selector.",
        "",
        "## Full-prefix baselines",
        "",
        "| M | D2 error | H error (mEh) | ftPBE error (mEh) |",
        "|---:|---:|---:|---:|",
    ]
    for maximum, row in baselines.items():
        lines.append(
            f"| {maximum} | {row['d2_frobenius_error']:.6f} | "
            f"{row['hamiltonian_error_meh']:.3f} | {row['ftpbe_error_meh']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Validated subsets",
            "",
            "| M | k | subset (1-based) | max ratio | D2 | H (mEh) | ftPBE (mEh) |",
            "|---:|---:|:---|---:|---:|---:|---:|",
        ]
    )
    for row in formal_rows:
        lines.append(
            f"| {row['maximum']} | {row['subset_size']} | "
            f"{row['subset_one_based']} | {row['maximum_error_ratio']:.3f} | "
            f"{row['d2_frobenius_error']:.6f} | "
            f"{row['hamiltonian_error_meh']:.3f} | {row['ftpbe_error_meh']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Saved-shot reallocation",
            "",
            "| M | k | allocation | counts (1-based:blocks) | D2 | H (mEh) | ftPBE (mEh) |",
            "|---:|---:|:---|:---|---:|---:|---:|",
        ]
    )
    for row in reallocation_rows:
        lines.append(
            f"| {row['maximum']} | {row['subset_size']} | {row['allocation']} | "
            f"{row['block_counts_one_based']} | {row['d2_frobenius_error']:.6f} | "
            f"{row['hamiltonian_error_meh']:.3f} | {row['ftpbe_error_meh']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    counts = _parse_counts(args.counts)
    if args.formal_candidates_per_size < 1:
        raise ValueError("formal-candidates-per-size must be positive.")
    args.baseline_dir = args.baseline_dir.resolve()
    args.shadow_archive = (
        args.shadow_archive.resolve()
        if args.shadow_archive is not None
        else args.baseline_dir / "hybrid_shadows.npz"
    )
    args.frozen_mo_npz = args.frozen_mo_npz.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_reallocation: dict[tuple[int, str], dict[str, Any]] = {}
    prior_analysis_path = args.output_dir / "analysis.json"
    if prior_analysis_path.is_file() and not args.overwrite:
        prior_payload = json.loads(prior_analysis_path.read_text(encoding="utf-8"))
        prior_reallocation = {
            (int(row["maximum"]), str(row["allocation"])): row
            for row in prior_payload.get("reallocation_rows", [])
        }

    baseline_payload = json.loads(
        (args.baseline_dir / "analysis.json").read_text(encoding="utf-8")
    )
    baseline_rows = {int(row["shadows"]): row for row in baseline_payload["rows"]}
    raw = load_blind_shadow_npz(args.shadow_archive, load_design=False)
    if max(counts) > raw.n_shadows:
        raise ValueError("The cadence-4 shadow archive is shorter than counts.")
    with np.load(args.shadow_archive, allow_pickle=False) as archive:
        exact_values = np.asarray(archive["exact_values"], dtype=float)
        is_complex = np.asarray(archive["is_complex_basis"], dtype=bool)
    raw_with_exact = ShadowData(
        rotations=raw.rotations,
        pair_vectors=raw.pair_vectors,
        design=raw.design,
        values=raw.values,
        lower_bounds=raw.lower_bounds,
        upper_bounds=raw.upper_bounds,
        hits=raw.hits,
        shots_per_basis=raw.shots_per_basis,
        exact_values=exact_values,
        exact_constraints=False,
        occupations=None,
    )
    with np.load(args.frozen_mo_npz, allow_pickle=False) as archive:
        frozen_mo = np.asarray(archive["mo_coeff"], dtype=float)

    selection_base = build_c2_selection_reference(
        1.25, basis="cc-pvtz", frozen_mo_coeff=frozen_mo
    )
    selection = LeakGuardReference(selection_base)
    exact = build_c2_reference(1.25, basis="cc-pvtz", frozen_mo_coeff=frozen_mo)
    objective = FtPBEEnergyObjective(selection, grid_level=args.grid_level)
    exact_ftpbe = float(
        objective.evaluate(exact.exact_d2, exact.exact_gamma, gradient=False).total_energy
    )
    variable_rows, variable_cols = _symmetric_d2_variables(selection)
    blocks = shadow_design_blocks(
        LeakGuardShadowData.from_shadow_data(raw),
        len(selection.pairs),
        variable_rows,
        variable_cols,
    )
    frame_information, frame_rhs, noise_rms = _frame_statistics(
        blocks, raw_with_exact
    )
    exact_variables = matrix_to_variable_vector(
        exact.exact_d2, variable_rows, variable_cols
    )
    hamiltonian_gradient = _hamiltonian_gradient(
        selection, variable_rows, variable_cols
    )
    ftpbe_gradient = _raw_ftpbe_gradient(
        objective,
        selection,
        exact.exact_d2,
        exact.exact_gamma,
        variable_rows,
        variable_cols,
    )
    solver_args = SimpleNamespace(
        solver=args.solver.upper(),
        solver_tolerance=args.solver_tolerance,
        max_iterations=args.max_iterations,
        solver_threads=args.solver_threads,
        verbose_solver=False,
        weighted_fit_rmse_cap=None,
        reconstruction_objective=RECONSTRUCTION_ENERGY,
        ridge_grid=(0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0),
        ridge_folds=5,
        fallback_ridge=1e-2,
        symmetry_blocked_psd=True,
    )
    initial = _initial_result(solver_args, selection)
    prior_variables = matrix_to_variable_vector(
        initial.d2, variable_rows, variable_cols
    )
    off_diagonal = variable_rows != variable_cols
    information_scale = max(
        float(np.trace(frame_information[0]) / len(variable_rows)), 1e-12
    )
    rng = np.random.default_rng(args.search_seed)

    baselines: dict[int, dict[str, Any]] = {}
    screening_payload: dict[str, Any] = {}
    formal_rows: list[dict[str, Any]] = []
    for maximum in counts:
        baseline_source = baseline_rows.get(maximum)
        baseline_checkpoint = (
            args.baseline_dir / "checkpoints" / f"shadows_{maximum:04d}.npz"
        )
        generated_checkpoint = (
            args.output_dir / "checkpoints" / f"m{maximum:02d}_full.npz"
        )
        generated_json = generated_checkpoint.with_suffix(".json")
        if baseline_source is not None and baseline_checkpoint.is_file():
            baseline = {
                "d2_frobenius_error": float(
                    baseline_source["hybrid_d2_frobenius_error"]
                ),
                "hamiltonian_error_meh": 1000.0
                * float(baseline_source["hybrid_hamiltonian_error_eh"]),
                "ftpbe_error_meh": 1000.0
                * float(baseline_source["hybrid_ftpbe_error_eh"]),
            }
            with np.load(baseline_checkpoint, allow_pickle=False) as archive:
                full_d2 = np.asarray(archive["d2"])
                full_gamma = np.asarray(archive["gamma"])
        elif (
            generated_checkpoint.is_file()
            and generated_json.is_file()
            and not args.overwrite
        ):
            generated = json.loads(generated_json.read_text(encoding="utf-8"))
            baseline = {
                key: generated[key]
                for key in (
                    "d2_frobenius_error",
                    "hamiltonian_error_meh",
                    "ftpbe_error_meh",
                )
            }
            with np.load(generated_checkpoint, allow_pickle=False) as archive:
                full_d2 = np.asarray(archive["d2"])
                full_gamma = np.asarray(archive["gamma"])
            print(f"[M={maximum:02d}] full-prefix checkpoint", flush=True)
        else:
            acquired_full = acquire_shadow_bases(raw, tuple(range(maximum)))
            lower_counts = [
                count
                for count in baseline_rows
                if count < maximum
                and (
                    args.baseline_dir
                    / "checkpoints"
                    / f"shadows_{count:04d}.npz"
                ).is_file()
            ]
            if lower_counts:
                warm_count = max(lower_counts)
                with np.load(
                    args.baseline_dir
                    / "checkpoints"
                    / f"shadows_{warm_count:04d}.npz",
                    allow_pickle=False,
                ) as archive:
                    warm_d2 = np.asarray(archive["d2"])
                    warm_gamma = np.asarray(archive["gamma"])
            else:
                warm_d2 = np.asarray(initial.d2)
                warm_gamma = np.asarray(initial.gamma)
            print(f"[M={maximum:02d}] solving full-prefix baseline", flush=True)
            result, reconstruction = _solve_shadow_dqg(
                solver_args,
                selection,
                acquired_full,
                warm_d2,
                warm_gamma,
            )
            full_d2 = np.asarray(result.d2)
            full_gamma = np.asarray(result.gamma)
            generated = {
                **_score_dqg(
                    result, exact, objective, exact_ftpbe, acquired_full
                ),
                "status": str(result.status),
                "fit_status": str(result.fit_status),
                **reconstruction,
            }
            baseline = {
                key: generated[key]
                for key in (
                    "d2_frobenius_error",
                    "hamiltonian_error_meh",
                    "ftpbe_error_meh",
                )
            }
            _atomic_npz(
                generated_checkpoint, d2=full_d2, gamma=full_gamma
            )
            _atomic_json(generated_json, generated)
            print(
                f"[M={maximum:02d}] full D2={baseline['d2_frobenius_error']:.5f}, "
                f"H={baseline['hamiltonian_error_meh']:.3f}, "
                f"ftPBE={baseline['ftpbe_error_meh']:.3f} mEh",
                flush=True,
            )
        baselines[maximum] = baseline
        full_variables = matrix_to_variable_vector(
            full_d2, variable_rows, variable_cols
        )
        calibrated = []
        for fraction in RIDGE_GRID:
            ridge = fraction * information_scale
            matrix = ridge * np.eye(len(variable_rows)) + np.sum(
                frame_information[:maximum], axis=0
            )
            rhs = ridge * prior_variables + np.sum(frame_rhs[:maximum], axis=0)
            variables = solve(matrix, rhs, assume_a="pos", check_finite=False)
            delta = variables - full_variables
            calibration_error = float(
                np.sqrt(
                    np.sum(np.square(delta[~off_diagonal]))
                    + 2.0 * np.sum(np.square(delta[off_diagonal]))
                )
            )
            calibrated.append((calibration_error, fraction, ridge))
        _, ridge_fraction, ridge = min(calibrated)
        scales = np.asarray(
            [
                baseline["d2_frobenius_error"],
                baseline["hamiltonian_error_meh"] / 1000.0,
                baseline["ftpbe_error_meh"] / 1000.0,
            ]
        )
        surrogate = OracleSurrogate(
            frame_information[:maximum],
            frame_rhs[:maximum],
            ridge,
            prior_variables,
            exact_variables,
            off_diagonal,
            hamiltonian_gradient,
            ftpbe_gradient,
            scales,
        )
        candidates = _candidate_subsets(
            maximum, surrogate, noise_rms[:maximum], rng
        )
        screening_payload[str(maximum)] = {
            "ridge_fraction": ridge_fraction,
            "ridge": ridge,
            "candidate_sizes": {
                str(size): [
                    {
                        "indices_zero_based": list(subset),
                        "indices_one_based": [index + 1 for index in subset],
                        "surrogate_errors": surrogate.evaluate(subset)[1].tolist(),
                        "surrogate_ratios": surrogate.ratios(subset).tolist(),
                    }
                    for subset in subsets[: args.formal_candidates_per_size]
                ]
                for size, subsets in candidates.items()
            },
        }
        if args.screen_only:
            continue
        for size in SIZE_GRID[maximum]:
            for rank, subset in enumerate(
                candidates[size][: args.formal_candidates_per_size], start=1
            ):
                checkpoint = (
                    args.output_dir
                    / "checkpoints"
                    / f"m{maximum:02d}_k{size:02d}_r{rank}.npz"
                )
                checkpoint_json = checkpoint.with_suffix(".json")
                if checkpoint.is_file() and checkpoint_json.is_file() and not args.overwrite:
                    row = json.loads(checkpoint_json.read_text(encoding="utf-8"))
                    formal_rows.append(row)
                    print(
                        f"[M={maximum:02d}, k={size:02d}, r={rank}] checkpoint",
                        flush=True,
                    )
                    continue
                acquired = acquire_shadow_bases(raw, subset)
                print(
                    f"[M={maximum:02d}, k={size:02d}, r={rank}] solving "
                    f"subset={','.join(str(index + 1) for index in subset)}",
                    flush=True,
                )
                started = time.perf_counter()
                result, reconstruction = _solve_shadow_dqg(
                    solver_args,
                    selection,
                    acquired,
                    full_d2,
                    full_gamma,
                )
                elapsed = time.perf_counter() - started
                metrics = _score_dqg(
                    result, exact, objective, exact_ftpbe, acquired
                )
                ratios = np.asarray(
                    [
                        metrics["d2_frobenius_error"]
                        / baseline["d2_frobenius_error"],
                        metrics["hamiltonian_error_meh"]
                        / baseline["hamiltonian_error_meh"],
                        metrics["ftpbe_error_meh"] / baseline["ftpbe_error_meh"],
                    ]
                )
                row = {
                    "maximum": maximum,
                    "subset_size": size,
                    "candidate_rank": rank,
                    "subset_zero_based": ",".join(map(str, subset)),
                    "subset_one_based": ",".join(str(index + 1) for index in subset),
                    "complex_frames": int(np.count_nonzero(is_complex[list(subset)])),
                    "total_shots": size * raw.shots_per_basis,
                    "d2_error_ratio": float(ratios[0]),
                    "hamiltonian_error_ratio": float(ratios[1]),
                    "ftpbe_error_ratio": float(ratios[2]),
                    "maximum_error_ratio": float(np.max(ratios)),
                    "solver_seconds": elapsed,
                    "status": str(result.status),
                    "fit_status": str(result.fit_status),
                    **metrics,
                    **reconstruction,
                }
                formal_rows.append(row)
                _atomic_npz(
                    checkpoint,
                    d2=np.asarray(result.d2),
                    gamma=np.asarray(result.gamma),
                )
                _atomic_json(checkpoint_json, row)
                print(
                    f"[M={maximum:02d}, k={size:02d}] max-ratio="
                    f"{row['maximum_error_ratio']:.3f}, D2="
                    f"{row['d2_frobenius_error']:.5f}, H="
                    f"{row['hamiltonian_error_meh']:.3f}, ftPBE="
                    f"{row['ftpbe_error_meh']:.3f} mEh",
                    flush=True,
                )

    _atomic_json(args.output_dir / "screening.json", screening_payload)
    if args.screen_only:
        print(f"Screening results: {args.output_dir / 'screening.json'}", flush=True)
        return
    _write_csv(args.output_dir / "validated_subsets.csv", formal_rows)

    selected_subsets: dict[int, dict[str, Any]] = {}
    for maximum in counts:
        available = [row for row in formal_rows if row["maximum"] == maximum]
        close_110 = [row for row in available if row["maximum_error_ratio"] <= 1.10]
        close_125 = [row for row in available if row["maximum_error_ratio"] <= 1.25]
        pool = close_110 or close_125 or available
        selected_subsets[maximum] = min(
            pool,
            key=lambda row: (
                row["subset_size"] if close_110 or close_125 else row["maximum_error_ratio"],
                row["maximum_error_ratio"],
                row["subset_one_based"],
            ),
        )

    reallocation_oracle = AcquisitionOracle(
        exact,
        raw.rotations,
        raw.pair_vectors,
        raw.shots_per_basis,
        args.reallocation_seed,
    )
    reallocation_rows: list[dict[str, Any]] = []
    for maximum, selected_row in selected_subsets.items():
        subset = tuple(
            int(item) for item in selected_row["subset_zero_based"].split(",")
        )
        checkpoint = (
            args.output_dir
            / "checkpoints"
            / f"m{maximum:02d}_k{len(subset):02d}_r{selected_row['candidate_rank']}.npz"
        )
        with np.load(checkpoint, allow_pickle=False) as archive:
            subset_d2 = np.asarray(archive["d2"])
            subset_gamma = np.asarray(archive["gamma"])
        allocations = {
            "equal": _equal_allocation(subset, maximum),
            "oracle_information": _oracle_information_allocation(
                blocks,
                subset,
                maximum,
                exact_variables,
                hamiltonian_gradient,
                ftpbe_gradient,
                raw.shots_per_basis,
            ),
        }
        for name, block_counts in allocations.items():
            reallocation_checkpoint = (
                args.output_dir
                / "checkpoints"
                / f"m{maximum:02d}_reallocated_{name}.npz"
            )
            prior_key = (maximum, name)
            if (
                reallocation_checkpoint.is_file()
                and prior_key in prior_reallocation
                and not args.overwrite
            ):
                reallocation_rows.append(prior_reallocation[prior_key])
                print(
                    f"[M={maximum:02d}, {name}] reallocation checkpoint",
                    flush=True,
                )
                continue
            acquired = _reallocated_shadows(
                raw, subset, block_counts, reallocation_oracle
            )
            print(
                f"[M={maximum:02d}, k={len(subset):02d}] solving {name} "
                "saved-shot allocation",
                flush=True,
            )
            started = time.perf_counter()
            result, reconstruction = _solve_shadow_dqg(
                solver_args,
                selection,
                acquired,
                subset_d2,
                subset_gamma,
            )
            elapsed = time.perf_counter() - started
            metrics = _score_dqg(result, exact, objective, exact_ftpbe, acquired)
            row = {
                "maximum": maximum,
                "subset_size": len(subset),
                "subset_one_based": ",".join(str(index + 1) for index in subset),
                "allocation": name,
                "block_counts_zero_based": ",".join(
                    f"{index}:{block_counts[index]}" for index in subset
                ),
                "block_counts_one_based": ",".join(
                    f"{index + 1}:{block_counts[index]}" for index in subset
                ),
                "distinct_frames": len(subset),
                "total_blocks": sum(block_counts.values()),
                "total_shots": sum(block_counts.values()) * raw.shots_per_basis,
                "solver_seconds": elapsed,
                "status": str(result.status),
                "fit_status": str(result.fit_status),
                **metrics,
                **reconstruction,
            }
            reallocation_rows.append(row)
            _atomic_npz(
                reallocation_checkpoint,
                d2=np.asarray(result.d2),
                gamma=np.asarray(result.gamma),
            )
            print(
                f"[M={maximum:02d}, {name}] D2={row['d2_frobenius_error']:.5f}, "
                f"H={row['hamiltonian_error_meh']:.3f}, "
                f"ftPBE={row['ftpbe_error_meh']:.3f} mEh",
                flush=True,
            )

    configuration = {
        "baseline_dir": str(args.baseline_dir),
        "shadow_archive": str(args.shadow_archive),
        "frozen_mo_npz": str(args.frozen_mo_npz),
        "counts": list(counts),
        "shots_per_block": raw.shots_per_basis,
        "candidate_frame_pattern": "every fourth frame complex (1-based 4,8,12,...)",
        "oracle_scope": (
            "exact-assisted multiobjective subset screening, posthoc subset choice, "
            "and exact-information saved-shot allocation"
        ),
        "close_threshold_primary": 1.10,
        "close_threshold_fallback": 1.25,
        "size_grid": {str(key): list(value) for key, value in SIZE_GRID.items()},
        "formal_candidates_per_size": args.formal_candidates_per_size,
        "search_seed": args.search_seed,
        "reallocation_seed": args.reallocation_seed,
        "solver": args.solver.upper(),
        "solver_tolerance": args.solver_tolerance,
        "solver_threads": args.solver_threads,
    }
    payload = {
        "configuration": configuration,
        "baselines": baselines,
        "selected_subsets": selected_subsets,
        "validated_subsets": formal_rows,
        "reallocation_rows": reallocation_rows,
    }
    _atomic_json(args.output_dir / "configuration.json", configuration)
    _atomic_json(args.output_dir / "analysis.json", payload)
    _write_csv(args.output_dir / "reallocation.csv", reallocation_rows)
    (args.output_dir / "C2_ORACLE_SUBSET_REALLOCATION.md").write_text(
        _report(baselines, formal_rows, reallocation_rows), encoding="utf-8"
    )
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
