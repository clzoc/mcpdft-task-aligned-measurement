#!/usr/bin/env python3
"""Pilot-adaptive shot allocation for random real+complex N2 CST frames.

The online policy never reads an exact RDM.  A state-independent random frame
pool is frozen first, every frame receives a pilot, and the pilot occupations
are used to estimate full within-frame covariances and a current Eq. (11) DQG
solution.  Production-shot allocations are then frozen by minimizing the worst
normalized D2/H/ftPBE linearized risk.  Pilot shots are not reused in the
formal production reconstruction, which avoids optional-stopping reuse of the
same fluctuations that selected the allocation.

The exact state appears only in the simulator that returns occupations and in
posthoc scoring against the reachable exact-mean Eq. (11) solution D-dagger.
All formal reconstructions use the original nuclear-norm objective, DQG, and
MOSEK; weighted GLS and consistency projection are deliberately absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/n2-random-adaptive-allocation")

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.linalg import qr  # noqa: E402


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ROOT / "code"
VENDOR = CODE / "vendor"
for path in (CODE, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constrained_shadow import (  # noqa: E402
    ShadowData,
    build_n2_reference,
    quadratic_design,
    random_orthogonal_rotations,
    shadow_pair_vectors,
    solve_dqg_sdp,
)
from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    build_n2_selection_reference,
)
from mcpdft_selector import FtPBEEnergyObjective, _symmetric_d2_variables  # noqa: E402
from run_c2_ftpbe_shot_reallocation import AcquisitionOracle  # noqa: E402
from run_c2_oracle_subset_reallocation import (  # noqa: E402
    _hamiltonian_gradient,
    _raw_ftpbe_gradient,
)
from run_n2_ftpbe_shot_cost_oracle import (  # noqa: E402
    RELATIVE_FLOOR,
    SHRINKAGE,
    _frame_block,
    _regularized_single_covariance,
)
from run_n2_joint_shadow_norm_conic_oracle import (  # noqa: E402
    ConstraintBasis,
    _atomic_json,
    _atomic_npz,
    _finite_shadows,
    _parse_integers,
    _sample_outcomes,
)
from run_n2_paper_nuclear_hybrid_trajectories import (  # noqa: E402
    DEFAULT_NOISE_SEEDS,
    _blind_shadows,
    _random_unitaries,
)


DEFAULT_OUTPUT = ROOT / "sweeps" / "n2_random_real_complex_adaptive_allocation"
DEFAULT_BUDGETS = (30_000, 60_000, 120_000, 180_000, 240_000, 300_000)
METHODS = ("uniform", "pilot_adaptive")


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--shot-seeds", type=_parse_integers, default=DEFAULT_NOISE_SEEDS
    )
    parser.add_argument("--budgets", type=_parse_integers, default=DEFAULT_BUDGETS)
    parser.add_argument("--frame-count", type=int, default=30)
    parser.add_argument("--complex-fraction", type=float, default=0.5)
    parser.add_argument("--real-frame-seed", type=int, default=20260716)
    parser.add_argument("--complex-frame-seed", type=int, default=271828)
    parser.add_argument("--placement-seed", type=int, default=314159)
    parser.add_argument("--pilot-shots-per-frame", type=int, default=500)
    parser.add_argument("--production-minimum", type=int, default=500)
    parser.add_argument("--allocation-chunk", type=int, default=500)
    parser.add_argument("--bond-length", type=float, default=1.10)
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--active-electrons", type=int, default=6)
    parser.add_argument("--active-orbitals", type=int, default=6)
    parser.add_argument("--d2-capacity", type=float, default=0.03)
    parser.add_argument("--energy-capacity-meh", type=float, default=1.6)
    parser.add_argument("--d2-hardware-floor", type=float, default=0.0)
    parser.add_argument("--hamiltonian-hardware-floor-meh", type=float, default=0.0)
    parser.add_argument("--ftpbe-hardware-floor-meh", type=float, default=0.0)
    parser.add_argument("--stability-window", type=int, default=2)
    parser.add_argument("--nuclear-weight", type=float, default=1.0)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--solver-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args(argv)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _random_mixed_frames(
    n_orbitals: int,
    count: int,
    complex_fraction: float,
    real_seed: int,
    complex_seed: int,
    placement_seed: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Draw a frozen random mixture of Haar O(n) and Haar U(n) frames."""

    if count < 2:
        raise ValueError("At least two random frames are required.")
    if not 0.0 < complex_fraction < 1.0:
        raise ValueError("complex-fraction must lie strictly between zero and one.")
    complex_count = int(round(count * complex_fraction))
    complex_count = min(max(complex_count, 1), count - 1)
    real_count = count - complex_count
    real = iter(random_orthogonal_rotations(n_orbitals, real_count, real_seed))
    complex_frames = iter(_random_unitaries(n_orbitals, complex_count, complex_seed))
    flags = np.asarray([False] * real_count + [True] * complex_count, dtype=bool)
    np.random.default_rng(placement_seed).shuffle(flags)
    rotations = tuple(
        np.asarray(next(complex_frames) if flag else next(real)) for flag in flags
    )
    return rotations, flags


def _build_random_design(
    reference: Any, rotations: Sequence[np.ndarray]
) -> tuple[np.ndarray, Any, tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Build state-independent measurement maps for the frozen frame pool."""

    rows, cols = _symmetric_d2_variables(reference)
    pair_vectors = shadow_pair_vectors(
        rotations, reference.n_spatial_orbitals, reference.pairs
    )
    design = quadratic_design(pair_vectors)
    rows_per_frame = len(reference.pairs)
    blocks = tuple(
        _frame_block(
            pair_vectors[index * rows_per_frame : (index + 1) * rows_per_frame],
            rows,
            cols,
        )
        for index in range(len(rotations))
    )
    return np.asarray(pair_vectors), design, blocks, rows, cols


def _pilot_constraint_basis(
    name: str,
    rotations: Sequence[np.ndarray],
    pair_vectors: np.ndarray,
    design: Any,
    blocks: Sequence[np.ndarray],
    pilot_outcomes: Sequence[np.ndarray],
    dimension: int,
) -> ConstraintBasis:
    """Choose a full-rank literal-row basis using pilot covariance only."""

    covariances = tuple(
        _regularized_single_covariance(np.asarray(outcomes, dtype=float))
        for outcomes in pilot_outcomes
    )
    diagonal_variances = tuple(
        np.maximum(np.diag(covariance), 1e-12) for covariance in covariances
    )
    stacked = np.vstack(blocks)
    scales = np.sqrt(np.concatenate(diagonal_variances))
    weighted = stacked / scales[:, None]
    _, triangular, pivots = qr(weighted.T, mode="economic", pivoting=True)
    tolerance = abs(triangular[0, 0]) * max(weighted.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(np.abs(np.diag(triangular)) > tolerance))
    if rank < dimension:
        raise RuntimeError(
            f"The random frame pool has rank {rank}, below required {dimension}."
        )
    rows_per_frame = blocks[0].shape[0]
    chosen = np.asarray(pivots[:dimension], dtype=int)
    mapping = sorted(
        (int(index // rows_per_frame), int(index % rows_per_frame)) for index in chosen
    )
    global_rows = np.asarray(
        [frame * rows_per_frame + local for frame, local in mapping], dtype=int
    )
    rows_by_frame = []
    selected_blocks = []
    selected_covariances = []
    for frame in range(len(rotations)):
        local = np.asarray(
            [local for selected_frame, local in mapping if selected_frame == frame],
            dtype=int,
        )
        rows_by_frame.append(local)
        selected_blocks.append(np.asarray(blocks[frame])[local])
        selected_covariances.append(
            covariances[frame][np.ix_(local, local)] if len(local) else np.empty((0, 0))
        )
    singular = np.linalg.svd(stacked[global_rows], compute_uv=False)
    provisional = ConstraintBasis(
        name=name,
        rotations=tuple(np.asarray(rotation) for rotation in rotations),
        pair_vectors=np.asarray(pair_vectors),
        design=design,
        global_rows=global_rows,
        rows_by_frame=tuple(rows_by_frame),
        blocks=tuple(selected_blocks),
        covariances=tuple(selected_covariances),
        raw_condition=float(singular[0] / singular[-1]),
        fisher_condition=np.nan,
        rank=dimension,
    )
    information = _information(
        provisional,
        np.asarray([1 if len(local) else 0 for local in rows_by_frame]),
    )
    eigenvalues = np.linalg.eigvalsh(information)
    return ConstraintBasis(
        **{
            **provisional.__dict__,
            "fisher_condition": float(eigenvalues[-1] / eigenvalues[0]),
        }
    )


def _information(basis: ConstraintBasis, counts: np.ndarray) -> np.ndarray:
    dimension = basis.rank
    information = np.zeros((dimension, dimension), dtype=float)
    for frame in basis.active_frames:
        covariance = basis.covariances[frame]
        block = basis.blocks[frame]
        information += int(counts[frame]) * (
            block.T @ np.linalg.solve(covariance, block)
        )
    return 0.5 * (information + information.T)


def _inverse_information(information: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (information + information.T))
    floor = max(float(eigenvalues[-1]) * 1e-12, 1e-14)
    return (eigenvectors / np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _updated_covariance(
    covariance: np.ndarray,
    block: np.ndarray,
    observation_covariance: np.ndarray,
    added_shots: int,
) -> np.ndarray:
    projected = block @ covariance
    small = observation_covariance / added_shots + projected @ block.T
    updated = covariance - projected.T @ np.linalg.solve(small, projected)
    return 0.5 * (updated + updated.T)


@dataclass(frozen=True)
class TargetModel:
    d2_weights: np.ndarray
    hamiltonian: np.ndarray
    ftpbe_modes: np.ndarray
    capacities_squared: np.ndarray
    hardware_floors_squared: np.ndarray


def _risks(covariance: np.ndarray, targets: TargetModel) -> np.ndarray:
    ftpbe_variances = np.einsum(
        "ki,ij,kj->k",
        targets.ftpbe_modes,
        covariance,
        targets.ftpbe_modes,
        optimize=True,
    )
    return np.asarray(
        [
            float(targets.d2_weights @ np.diag(covariance)),
            float(targets.hamiltonian @ covariance @ targets.hamiltonian),
            float(np.max(ftpbe_variances)),
        ]
    )


def _risk_score(risks: np.ndarray, targets: TargetModel) -> tuple[float, float]:
    total = risks + targets.hardware_floors_squared
    ratios = total / targets.capacities_squared
    return float(np.max(ratios)), float(np.mean(ratios))


def _equal_counts(
    frame_count: int, active: Sequence[int], budget: int, quantum: int
) -> np.ndarray:
    active = tuple(int(index) for index in active)
    minimum = len(active) * quantum
    if budget < minimum or budget % quantum:
        raise ValueError(
            "Uniform production budget is incompatible with active frames."
        )
    counts = np.zeros(frame_count, dtype=int)
    blocks = budget // quantum
    for offset in range(blocks):
        counts[active[offset % len(active)]] += quantum
    return counts


def _allocation_paths(
    basis: ConstraintBasis,
    production_budgets: Sequence[int],
    minimum: int,
    chunk: int,
    targets: TargetModel,
) -> tuple[dict[str, dict[int, np.ndarray]], list[dict[str, Any]]]:
    active = basis.active_frames
    adaptive = np.zeros(len(basis.rotations), dtype=int)
    adaptive[list(active)] = minimum
    covariance = _inverse_information(_information(basis, adaptive))
    paths: dict[str, dict[int, np.ndarray]] = {method: {} for method in METHODS}
    diagnostics: list[dict[str, Any]] = []
    for budget in production_budgets:
        if budget < int(np.sum(adaptive)) or budget % chunk:
            raise ValueError("A production budget is below the exploration minimum.")
        while int(np.sum(adaptive)) < budget:
            candidates = []
            for frame in active:
                trial = _updated_covariance(
                    covariance,
                    basis.blocks[frame],
                    basis.covariances[frame],
                    chunk,
                )
                risks = _risks(trial, targets)
                candidates.append(
                    (
                        _risk_score(risks, targets),
                        int(adaptive[frame]),
                        frame,
                        trial,
                    )
                )
            _, _, selected, covariance = min(
                candidates,
                key=lambda item: (item[0][0], item[0][1], item[1], item[2]),
            )
            adaptive[selected] += chunk
        paths["pilot_adaptive"][int(budget)] = adaptive.copy()
        uniform = _equal_counts(len(basis.rotations), active, int(budget), chunk)
        paths["uniform"][int(budget)] = uniform
        for method, counts in (
            ("uniform", uniform),
            ("pilot_adaptive", adaptive),
        ):
            method_covariance = _inverse_information(_information(basis, counts))
            risks = _risks(method_covariance, targets)
            score = _risk_score(risks, targets)
            diagnostics.append(
                {
                    "method": method,
                    "production_shots": int(budget),
                    "predicted_max_capacity_ratio": score[0],
                    "predicted_mean_capacity_ratio": score[1],
                    "predicted_d2_rms": float(np.sqrt(risks[0])),
                    "predicted_hamiltonian_rms_meh": float(1000.0 * np.sqrt(risks[1])),
                    "predicted_ftpbe_rms_meh": float(1000.0 * np.sqrt(risks[2])),
                    "shot_counts": ",".join(map(str, counts)),
                }
            )
    return paths, diagnostics


def _solve_eq11(
    args: argparse.Namespace,
    selection: Any,
    shadows: ShadowData,
    warm_d2: np.ndarray,
    warm_gamma: np.ndarray,
    warm_corrected: np.ndarray,
) -> Any:
    return solve_dqg_sdp(
        LeakGuardReference(selection),
        shadow_data=_blind_shadows(shadows),
        shadow_error_weight=args.nuclear_weight,
        solver=args.solver,
        tolerance=args.solver_tolerance,
        max_iterations=args.max_iterations,
        solver_threads=args.solver_threads,
        positivity_conditions="DQG",
        symmetry_blocked_psd=True,
        selection_objective="energy",
        initial_d2=warm_d2,
        initial_gamma=warm_gamma,
        initial_corrected_d2=warm_corrected,
    )


def _energy_values(
    reference: Any, objective: Any, d2: np.ndarray, gamma: np.ndarray
) -> tuple[float, float]:
    hamiltonian = float(
        np.sum(reference.one_body * gamma)
        + np.sum(reference.two_body * d2)
        + reference.nuclear_energy
    )
    ftpbe = float(objective.evaluate(d2, gamma, gradient=False).total_energy)
    return hamiltonian, ftpbe


def _comparison_metrics(
    d2: np.ndarray,
    gamma: np.ndarray,
    reference_d2: np.ndarray,
    reference_h: float,
    reference_f: float,
    selection: Any,
    objective: Any,
) -> dict[str, float]:
    hamiltonian, ftpbe = _energy_values(selection, objective, d2, gamma)
    return {
        "d2_error": float(np.linalg.norm(d2 - reference_d2)),
        "signed_hamiltonian_error_meh": float(1000.0 * (hamiltonian - reference_h)),
        "hamiltonian_error_meh": float(1000.0 * abs(hamiltonian - reference_h)),
        "signed_ftpbe_error_meh": float(1000.0 * (ftpbe - reference_f)),
        "ftpbe_error_meh": float(1000.0 * abs(ftpbe - reference_f)),
    }


def _exact_mean_shadows(basis: ConstraintBasis, d2: np.ndarray) -> ShadowData:
    values = np.asarray(
        basis.design[basis.global_rows] @ np.asarray(d2).ravel(order="C")
    ).ravel()
    return ShadowData(
        rotations=tuple(basis.rotations[index] for index in basis.active_frames),
        pair_vectors=basis.pair_vectors[basis.global_rows],
        design=basis.design[basis.global_rows],
        values=values,
        lower_bounds=values.copy(),
        upper_bounds=values.copy(),
        hits=np.full(len(values), -1, dtype=int),
        shots_per_basis=0,
        exact_values=np.full(len(values), np.nan),
        exact_constraints=False,
        occupations=None,
    )


def _validate_args(args: argparse.Namespace) -> None:
    args.solver = args.solver.upper()
    if args.solver != "MOSEK":
        raise ValueError("This runner fixes the formal SDP solver to MOSEK.")
    if args.frame_count < 2 or args.pilot_shots_per_frame < 2:
        raise ValueError("Frame count and pilot shots must be at least two.")
    if args.production_minimum < 1 or args.allocation_chunk < 1:
        raise ValueError("Production minimum and allocation chunk must be positive.")
    if args.production_minimum % args.allocation_chunk:
        raise ValueError("production-minimum must align with allocation-chunk.")
    if args.nuclear_weight <= 0.0 or args.d2_capacity <= 0.0:
        raise ValueError("Nuclear weight and D2 capacity must be positive.")
    if args.energy_capacity_meh <= 0.0:
        raise ValueError("Energy capacity must be positive.")
    if args.stability_window < 1:
        raise ValueError("stability-window must be positive.")
    capacities = np.asarray(
        [args.d2_capacity, args.energy_capacity_meh, args.energy_capacity_meh]
    )
    floors = np.asarray(
        [
            args.d2_hardware_floor,
            args.hamiltonian_hardware_floor_meh,
            args.ftpbe_hardware_floor_meh,
        ]
    )
    if np.any(floors < 0.0) or np.any(floors >= capacities):
        raise ValueError("Every hardware floor must be nonnegative and below capacity.")


def _run_seed(
    args: argparse.Namespace,
    seed: int,
    selection: Any,
    exact: Any,
    objective: Any,
    rotations: Sequence[np.ndarray],
    is_complex: np.ndarray,
    pair_vectors: np.ndarray,
    design: Any,
    blocks: Sequence[np.ndarray],
    variable_rows: np.ndarray,
    variable_cols: np.ndarray,
    baseline: Any,
) -> None:
    output = args.output_dir / f"shot_seed_{seed}"
    analysis_path = output / "analysis.json"
    if analysis_path.is_file() and not args.overwrite:
        print(f"[shot_seed={seed}] complete checkpoint", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    oracle = AcquisitionOracle(exact, rotations, pair_vectors, 1, seed)
    pilot_counts = np.full(args.frame_count, args.pilot_shots_per_frame, dtype=int)
    pilot_outcomes = _sample_outcomes(oracle, seed + 100_003, pilot_counts)
    basis = _pilot_constraint_basis(
        "random_real_complex_pilot_basis",
        rotations,
        pair_vectors,
        design,
        blocks,
        pilot_outcomes,
        len(variable_rows),
    )
    pilot_shadows = _finite_shadows(basis, pilot_outcomes, pilot_counts)
    print(f"[shot_seed={seed}] solving blind pilot Eq. (11)", flush=True)
    pilot = _solve_eq11(
        args,
        selection,
        pilot_shadows,
        np.asarray(baseline.d2),
        np.asarray(baseline.gamma),
        np.asarray(baseline.d2),
    )
    hamiltonian = _hamiltonian_gradient(selection, variable_rows, variable_cols)
    ftpbe_pilot = _raw_ftpbe_gradient(
        objective,
        selection,
        np.asarray(pilot.d2),
        np.asarray(pilot.gamma),
        variable_rows,
        variable_cols,
    )
    ftpbe_baseline = _raw_ftpbe_gradient(
        objective,
        selection,
        np.asarray(baseline.d2),
        np.asarray(baseline.gamma),
        variable_rows,
        variable_cols,
    )
    targets = TargetModel(
        d2_weights=np.where(variable_rows == variable_cols, 1.0, 2.0),
        hamiltonian=hamiltonian,
        ftpbe_modes=np.vstack((ftpbe_pilot, ftpbe_baseline)),
        capacities_squared=np.square(
            np.asarray(
                [
                    args.d2_capacity,
                    args.energy_capacity_meh / 1000.0,
                    args.energy_capacity_meh / 1000.0,
                ]
            )
        ),
        hardware_floors_squared=np.square(
            np.asarray(
                [
                    args.d2_hardware_floor,
                    args.hamiltonian_hardware_floor_meh / 1000.0,
                    args.ftpbe_hardware_floor_meh / 1000.0,
                ]
            )
        ),
    )
    pilot_total = args.frame_count * args.pilot_shots_per_frame
    production_budgets = tuple(int(budget - pilot_total) for budget in args.budgets)
    paths, diagnostics = _allocation_paths(
        basis,
        production_budgets,
        args.production_minimum,
        args.allocation_chunk,
        targets,
    )
    _write_csv(output / "allocation_diagnostics.csv", diagnostics)
    _atomic_json(
        output / "blind_policy.json",
        {
            "shot_seed": seed,
            "frame_types": ["complex" if flag else "real" for flag in is_complex],
            "active_frames": [index + 1 for index in basis.active_frames],
            "rows_per_frame": [len(rows) for rows in basis.rows_by_frame],
            "raw_condition": basis.raw_condition,
            "pilot_fisher_condition": basis.fisher_condition,
            "pilot_shots": pilot_total,
            "production_paths": {
                method: {
                    str(pilot_total + budget): paths[method][budget].tolist()
                    for budget in production_budgets
                }
                for method in METHODS
            },
            "online_exact_state_access": False,
            "pilot_reused_in_formal_reconstruction": False,
        },
    )

    # D-dagger is constructed only after the blind allocation path is frozen.
    reachable = _solve_eq11(
        args,
        selection,
        _exact_mean_shadows(basis, exact.exact_d2),
        np.asarray(baseline.d2),
        np.asarray(baseline.gamma),
        np.asarray(baseline.d2),
    )
    reachable_h, reachable_f = _energy_values(
        selection, objective, np.asarray(reachable.d2), np.asarray(reachable.gamma)
    )
    exact_h, exact_f = _energy_values(
        selection, objective, exact.exact_d2, exact.exact_gamma
    )
    maximum_counts = np.max(
        np.vstack(
            [
                paths[method][budget]
                for method in METHODS
                for budget in production_budgets
            ]
        ),
        axis=0,
    )
    production_outcomes = _sample_outcomes(oracle, seed + 200_003, maximum_counts)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        warm_d2 = np.asarray(pilot.d2)
        warm_gamma = np.asarray(pilot.gamma)
        warm_corrected = np.asarray(pilot.corrected_d2)
        previous: tuple[np.ndarray, float, float] | None = None
        stability_flags: list[bool] = []
        for total_budget, production_budget in zip(args.budgets, production_budgets):
            counts = paths[method][production_budget]
            shadows = _finite_shadows(basis, production_outcomes, counts)
            print(
                f"[shot_seed={seed} {method} B={total_budget}] solving Eq. (11)",
                flush=True,
            )
            started = time.perf_counter()
            result = _solve_eq11(
                args,
                selection,
                shadows,
                warm_d2,
                warm_gamma,
                warm_corrected,
            )
            warm_d2 = np.asarray(result.d2)
            warm_gamma = np.asarray(result.gamma)
            warm_corrected = np.asarray(result.corrected_d2)
            reachable_metrics = _comparison_metrics(
                warm_d2,
                warm_gamma,
                np.asarray(reachable.d2),
                reachable_h,
                reachable_f,
                selection,
                objective,
            )
            exact_metrics = _comparison_metrics(
                warm_d2,
                warm_gamma,
                np.asarray(exact.exact_d2),
                exact_h,
                exact_f,
                selection,
                objective,
            )
            current_h, current_f = _energy_values(
                selection, objective, warm_d2, warm_gamma
            )
            if previous is None:
                stability_d2 = None
                stability_h = None
                stability_f = None
            else:
                stability_d2 = float(np.linalg.norm(warm_d2 - previous[0]))
                stability_h = float(1000.0 * abs(current_h - previous[1]))
                stability_f = float(1000.0 * abs(current_f - previous[2]))
            previous = (warm_d2.copy(), current_h, current_f)
            diagnostic = next(
                row
                for row in diagnostics
                if row["method"] == method
                and int(row["production_shots"]) == production_budget
            )
            stability_pass = bool(
                stability_d2 is not None
                and stability_d2 <= args.d2_capacity
                and stability_h <= args.energy_capacity_meh
                and stability_f <= args.energy_capacity_meh
                and diagnostic["predicted_max_capacity_ratio"] <= 1.0
            )
            stability_flags.append(stability_pass)
            online_stop = bool(
                len(stability_flags) >= args.stability_window
                and all(stability_flags[-args.stability_window :])
            )
            row = {
                "shot_seed": seed,
                "method": method,
                "total_shots": int(total_budget),
                "pilot_shots": int(pilot_total),
                "production_shots": int(production_budget),
                "minimum_production_frame_shots": int(
                    np.min(counts[list(basis.active_frames)])
                ),
                "maximum_production_frame_shots": int(
                    np.max(counts[list(basis.active_frames)])
                ),
                "complex_production_shots": int(np.sum(counts[is_complex])),
                "shot_counts": ",".join(map(str, counts)),
                "solver_seconds": time.perf_counter() - started,
                "status": str(result.status),
                "fit_status": str(result.fit_status),
                "d2_error_to_reachable": reachable_metrics["d2_error"],
                "signed_hamiltonian_error_to_reachable_meh": reachable_metrics[
                    "signed_hamiltonian_error_meh"
                ],
                "hamiltonian_error_to_reachable_meh": reachable_metrics[
                    "hamiltonian_error_meh"
                ],
                "signed_ftpbe_error_to_reachable_meh": reachable_metrics[
                    "signed_ftpbe_error_meh"
                ],
                "ftpbe_error_to_reachable_meh": reachable_metrics["ftpbe_error_meh"],
                "d2_error_to_exact": exact_metrics["d2_error"],
                "hamiltonian_error_to_exact_meh": exact_metrics[
                    "hamiltonian_error_meh"
                ],
                "ftpbe_error_to_exact_meh": exact_metrics["ftpbe_error_meh"],
                "d2_change_from_previous": stability_d2,
                "hamiltonian_change_from_previous_meh": stability_h,
                "ftpbe_change_from_previous_meh": stability_f,
                "online_stability_pass": stability_pass,
                "online_stop_recommended": online_stop,
                "reachable_capacity_pass": bool(
                    reachable_metrics["d2_error"] <= args.d2_capacity
                    and reachable_metrics["hamiltonian_error_meh"]
                    <= args.energy_capacity_meh
                    and reachable_metrics["ftpbe_error_meh"] <= args.energy_capacity_meh
                ),
                **{
                    key: value
                    for key, value in diagnostic.items()
                    if key.startswith("predicted_")
                },
            }
            rows.append(row)
            label = f"{method}_{total_budget}"
            _atomic_npz(
                output / "checkpoints" / f"{label}.npz",
                d2=warm_d2,
                gamma=warm_gamma,
                corrected_d2=warm_corrected,
            )
            _atomic_json(output / "checkpoints" / f"{label}.json", row)
            print(
                f"[shot_seed={seed} {method} B={total_budget}] "
                f"D2dag={reachable_metrics['d2_error']:.4f}, "
                f"Hdag={reachable_metrics['hamiltonian_error_meh']:.3f}, "
                f"Fdag={reachable_metrics['ftpbe_error_meh']:.3f} mEh",
                flush=True,
            )
    _write_csv(output / "results.csv", rows)
    _atomic_json(
        analysis_path,
        {
            "configuration": _configuration(args, is_complex),
            "reachable_reference": {
                "definition": "exact-mean Eq. (11) solution after blind path freeze",
                "d2_error_to_exact": float(
                    np.linalg.norm(np.asarray(reachable.d2) - exact.exact_d2)
                ),
                "hamiltonian_error_to_exact_meh": float(
                    1000.0 * abs(reachable_h - exact_h)
                ),
                "ftpbe_error_to_exact_meh": float(1000.0 * abs(reachable_f - exact_f)),
            },
            "rows": rows,
        },
    )


def _configuration(args: argparse.Namespace, is_complex: np.ndarray) -> dict[str, Any]:
    return {
        "system": "N2",
        "bond_length_angstrom": args.bond_length,
        "basis": args.basis,
        "active_space": [args.active_electrons, args.active_orbitals],
        "frame_pool": "state-independent random Haar O(n)+U(n)",
        "frame_count": args.frame_count,
        "real_frames": int(np.count_nonzero(~is_complex)),
        "complex_frames": int(np.count_nonzero(is_complex)),
        "manifold_optimization_used": False,
        "pilot_shots_per_frame": args.pilot_shots_per_frame,
        "pilot_reused_in_formal_reconstruction": False,
        "budgets_include_pilot": True,
        "budgets": list(args.budgets),
        "d2_capacity": args.d2_capacity,
        "energy_capacity_meh": args.energy_capacity_meh,
        "hardware_floors": {
            "d2": args.d2_hardware_floor,
            "hamiltonian_meh": args.hamiltonian_hardware_floor_meh,
            "ftpbe_meh": args.ftpbe_hardware_floor_meh,
        },
        "online_stop_rule": (
            f"predicted capacity ratio <= 1 and D2/H/ftPBE changes remain within "
            f"capacity for {args.stability_window} consecutive budget points"
        ),
        "allocation_objective": (
            "minimize worst normalized pilot full-covariance D2 A-risk, known "
            "Hamiltonian c-risk, and robust baseline/pilot ftPBE tangent c-risk"
        ),
        "formal_reconstruction": "paper Eq. (11) nuclear norm with DQG",
        "solver": "MOSEK",
        "weighted_gls_used": False,
        "consistency_projection_used": False,
        "online_exact_state_access": False,
        "reachable_reference_is_posthoc": True,
        "pilot_covariance_shrinkage": SHRINKAGE,
        "pilot_covariance_relative_floor": RELATIVE_FLOOR,
    }


def _plot_reachable_curves(
    args: argparse.Namespace, rows: Sequence[dict[str, Any]]
) -> None:
    metrics = (
        ("d2_error_to_reachable", "2-RDM error to reachable", args.d2_capacity),
        (
            "hamiltonian_error_to_reachable_meh",
            "Hamiltonian error to reachable (mEh)",
            args.energy_capacity_meh,
        ),
        (
            "ftpbe_error_to_reachable_meh",
            "ftPBE error to reachable (mEh)",
            args.energy_capacity_meh,
        ),
    )
    colors = {"uniform": "#4C78A8", "pilot_adaptive": "#C23B22"}
    labels = {"uniform": "uniform", "pilot_adaptive": "pilot-adaptive"}
    figure, axes = plt.subplots(3, 1, figsize=(8.0, 9.0), sharex=True)
    for axis, (key, ylabel, threshold) in zip(axes, metrics):
        for method in METHODS:
            for seed in args.shot_seeds:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["method"] == method and int(row["shot_seed"]) == seed
                    ),
                    key=lambda row: int(row["total_shots"]),
                )
                axis.plot(
                    [int(row["total_shots"]) for row in selected],
                    [float(row[key]) for row in selected],
                    color=colors[method],
                    alpha=0.22,
                    linewidth=1.0,
                )
            medians = []
            for budget in args.budgets:
                medians.append(
                    float(
                        np.median(
                            [
                                row[key]
                                for row in rows
                                if row["method"] == method
                                and int(row["total_shots"]) == budget
                            ]
                        )
                    )
                )
            axis.plot(
                args.budgets,
                medians,
                color=colors[method],
                linewidth=2.4,
                marker="o",
                markersize=4,
                label=labels[method],
            )
        axis.axhline(threshold, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Total shots (pilot included)")
    figure.suptitle("N2 random real+complex CST: allocation errors")
    figure.tight_layout()
    figure.savefig(args.output_dir / "reachable_error_curves.png", dpi=220)
    plt.close(figure)


def _plot_reachable_boxplots(
    args: argparse.Namespace, rows: Sequence[dict[str, Any]]
) -> None:
    metrics = (
        ("d2_error_to_reachable", "2-RDM error to reachable", args.d2_capacity),
        (
            "hamiltonian_error_to_reachable_meh",
            "Hamiltonian error to reachable (mEh)",
            args.energy_capacity_meh,
        ),
        (
            "ftpbe_error_to_reachable_meh",
            "ftPBE error to reachable (mEh)",
            args.energy_capacity_meh,
        ),
    )
    colors = {"uniform": "#4C78A8", "pilot_adaptive": "#C23B22"}
    centers = np.arange(len(args.budgets), dtype=float)
    offsets = {"uniform": -0.18, "pilot_adaptive": 0.18}
    figure, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    for axis, (key, ylabel, threshold) in zip(axes, metrics):
        for method in METHODS:
            values = [
                [
                    float(row[key])
                    for row in rows
                    if row["method"] == method and int(row["total_shots"]) == budget
                ]
                for budget in args.budgets
            ]
            artists = axis.boxplot(
                values,
                positions=centers + offsets[method],
                widths=0.30,
                patch_artist=True,
                manage_ticks=False,
                showfliers=True,
            )
            for box in artists["boxes"]:
                box.set_facecolor(colors[method])
                box.set_alpha(0.55)
            for median in artists["medians"]:
                median.set_color("#111111")
        axis.axhline(threshold, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(centers, [f"{budget // 1000}k" for budget in args.budgets])
    axes[-1].set_xlabel("Total shots (pilot included)")
    handles = [
        plt.Line2D([0], [0], color=colors[method], linewidth=8, alpha=0.55)
        for method in METHODS
    ]
    axes[0].legend(handles, ("uniform", "pilot-adaptive"), frameon=False, ncol=2)
    figure.suptitle("N2 random real+complex CST: allocation distributions")
    figure.tight_layout()
    figure.savefig(args.output_dir / "reachable_error_boxplots.png", dpi=220)
    plt.close(figure)


def _aggregate(args: argparse.Namespace) -> None:
    rows = []
    reachable = []
    for seed in args.shot_seeds:
        path = args.output_dir / f"shot_seed_{seed}" / "analysis.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        reachable.append({"shot_seed": seed, **payload["reachable_reference"]})
    summaries = []
    for method in METHODS:
        for budget in args.budgets:
            selected = [
                row
                for row in rows
                if row["method"] == method and int(row["total_shots"]) == budget
            ]
            summaries.append(
                {
                    "method": method,
                    "total_shots": budget,
                    "median_d2_error_to_reachable": float(
                        np.median([row["d2_error_to_reachable"] for row in selected])
                    ),
                    "median_hamiltonian_error_to_reachable_meh": float(
                        np.median(
                            [
                                row["hamiltonian_error_to_reachable_meh"]
                                for row in selected
                            ]
                        )
                    ),
                    "median_ftpbe_error_to_reachable_meh": float(
                        np.median(
                            [row["ftpbe_error_to_reachable_meh"] for row in selected]
                        )
                    ),
                    "capacity_hit_rate": float(
                        np.mean([row["reachable_capacity_pass"] for row in selected])
                    ),
                }
            )
    stopping = []
    for method in METHODS:
        by_seed = {}
        oracle_by_seed = {}
        for seed in args.shot_seeds:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method and int(row["shot_seed"]) == seed
                ),
                key=lambda row: int(row["total_shots"]),
            )
            online_hits = [
                int(row["total_shots"])
                for row in selected
                if row["online_stop_recommended"]
            ]
            oracle_hits = [
                int(row["total_shots"])
                for row in selected
                if row["reachable_capacity_pass"]
            ]
            by_seed[str(seed)] = online_hits[0] if online_hits else None
            oracle_by_seed[str(seed)] = oracle_hits[0] if oracle_hits else None
        stopping.append(
            {
                "method": method,
                "online_first_stop_by_seed": by_seed,
                "posthoc_first_reachable_hit_by_seed": oracle_by_seed,
            }
        )
    _write_csv(args.output_dir / "all_results.csv", rows)
    _write_csv(args.output_dir / "aggregate_by_budget.csv", summaries)
    _plot_reachable_curves(args, rows)
    _plot_reachable_boxplots(args, rows)
    _atomic_json(
        args.output_dir / "summary.json",
        {
            "summaries": summaries,
            "stopping": stopping,
            "reachable_references": reachable,
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    _validate_args(args)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        _aggregate(args)
        return
    print(
        "Building the selection-only N2 Hamiltonian and random frame pool...",
        flush=True,
    )
    selection = build_n2_selection_reference(
        args.bond_length,
        basis=args.basis,
        active_electrons=args.active_electrons,
        active_orbitals=args.active_orbitals,
    )
    objective = FtPBEEnergyObjective(selection, grid_level=args.grid_level)
    rotations, is_complex = _random_mixed_frames(
        selection.n_spatial_orbitals,
        args.frame_count,
        args.complex_fraction,
        args.real_frame_seed,
        args.complex_frame_seed,
        args.placement_seed,
    )
    pair_vectors, design, blocks, variable_rows, variable_cols = _build_random_design(
        selection, rotations
    )
    _atomic_json(
        args.output_dir / "configuration.json", _configuration(args, is_complex)
    )
    _atomic_npz(
        args.output_dir / "random_frame_pool.npz",
        rotations=np.asarray(rotations),
        is_complex=is_complex,
        pair_vectors=pair_vectors,
    )
    print("Solving the shadow-free DQG initialization with MOSEK...", flush=True)
    baseline = solve_dqg_sdp(
        LeakGuardReference(selection),
        solver=args.solver,
        tolerance=args.solver_tolerance,
        max_iterations=args.max_iterations,
        solver_threads=args.solver_threads,
        positivity_conditions="DQG",
        symmetry_blocked_psd=True,
    )
    # From this point the exact object is confined to acquisition and posthoc scoring.
    exact = build_n2_reference(
        args.bond_length,
        basis=args.basis,
        active_electrons=args.active_electrons,
        active_orbitals=args.active_orbitals,
    )
    np.testing.assert_allclose(selection.one_body, exact.one_body, atol=2e-9)
    np.testing.assert_allclose(selection.two_body, exact.two_body, atol=2e-9)
    for seed in args.shot_seeds:
        _run_seed(
            args,
            int(seed),
            selection,
            exact,
            objective,
            rotations,
            is_complex,
            pair_vectors,
            design,
            blocks,
            variable_rows,
            variable_cols,
            baseline,
        )
    _aggregate(args)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
