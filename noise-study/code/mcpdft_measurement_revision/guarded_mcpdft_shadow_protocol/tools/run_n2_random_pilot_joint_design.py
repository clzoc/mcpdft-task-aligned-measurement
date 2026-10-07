#!/usr/bin/env python3
"""Fair-cost pilot frame selection and shot allocation for N2 CST.

Three deployable protocols are compared at the same *total* quantum-shot cost:

* ``uniform30`` uses the frozen random real+complex pool without a pilot, so
  every shot is a production shot.
* ``pilot_allocate30`` pays to pilot every candidate, keeps the full pool, and
  allocates the remaining production shots using pilot statistics.
* ``pilot_joint`` pays the same pilot cost, selects a variable-size subset of
  random frames, and allocates the remaining production shots over that subset.

Frame selection uses a D-optimal coverage guard, a local DQG-lineality gain
guard, and D2/H/ftPBE target risks.  It does not use an exact state.  Pilot data
are discarded from the formal reconstruction to keep selection data separate
from the original Eq. (11) production equalities.  All formal SDPs use DQG,
the paper nuclear-norm correction, and MOSEK, with no weighted GLS or
consistency projection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/n2-random-pilot-joint-design")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import qr  # noqa: E402


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ROOT / "code"
VENDOR = CODE / "vendor"
for path in (CODE, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constrained_shadow import (  # noqa: E402
    build_n2_reference,
    quadratic_design,
    solve_dqg_sdp,
)
from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    build_n2_selection_reference,
)
from mcpdft_selector import FtPBEEnergyObjective  # noqa: E402
from run_c2_ftpbe_shot_reallocation import AcquisitionOracle  # noqa: E402
from run_c2_oracle_subset_reallocation import (  # noqa: E402
    _hamiltonian_gradient,
    _raw_ftpbe_gradient,
)
from run_n2_joint_shadow_norm_conic_oracle import (  # noqa: E402
    ConstraintBasis,
    _atomic_json,
    _atomic_npz,
    _dqg_lineality_basis,
    _finite_shadows,
    _parse_integers,
    _sample_outcomes,
)
from run_n2_paper_nuclear_hybrid_trajectories import (  # noqa: E402
    DEFAULT_NOISE_SEEDS,
)
from run_n2_random_adaptive_allocation import (  # noqa: E402
    TargetModel,
    _allocation_paths,
    _build_random_design,
    _comparison_metrics,
    _energy_values,
    _exact_mean_shadows,
    _information,
    _inverse_information,
    _random_mixed_frames,
    _regularized_single_covariance,
    _risk_score,
    _risks,
    _solve_eq11,
)


DEFAULT_OUTPUT = ROOT / "sweeps" / "n2_random_pilot_joint_design"
DEFAULT_BUDGETS = (30_000, 60_000, 120_000, 180_000, 240_000, 300_000)
DEFAULT_SIZES = (5, 8, 10, 12, 14, 18, 24, 30)
METHODS = ("uniform30", "pilot_allocate30", "pilot_joint")
COLORS = {
    "uniform30": "#4C78A8",
    "pilot_allocate30": "#E09F3E",
    "pilot_joint": "#C23B22",
}
LABELS = {
    "uniform30": "uniform30",
    "pilot_allocate30": "pilot allocate30",
    "pilot_joint": "pilot joint",
}


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--shot-seeds", type=_parse_integers, default=DEFAULT_NOISE_SEEDS
    )
    parser.add_argument("--budgets", type=_parse_integers, default=DEFAULT_BUDGETS)
    parser.add_argument(
        "--candidate-sizes", type=_parse_integers, default=DEFAULT_SIZES
    )
    parser.add_argument("--frame-count", type=int, default=30)
    parser.add_argument("--complex-fraction", type=float, default=0.5)
    parser.add_argument("--real-frame-seed", type=int, default=20260716)
    parser.add_argument("--complex-frame-seed", type=int, default=271828)
    parser.add_argument("--placement-seed", type=int, default=314159)
    parser.add_argument("--pilot-shots-per-frame", type=int, default=500)
    parser.add_argument("--production-minimum", type=int, default=500)
    parser.add_argument("--allocation-chunk", type=int, default=500)
    parser.add_argument("--d-optimal-guard-fraction", type=float, default=0.05)
    parser.add_argument("--conic-guard-fraction", type=float, default=0.10)
    parser.add_argument("--design-ridge-fraction", type=float, default=1e-6)
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


@dataclass(frozen=True)
class ProtocolDesign:
    name: str
    basis: ConstraintBasis
    pool_indices: tuple[int, ...]
    paths: dict[int, np.ndarray]
    pilot_shots: int
    selection_diagnostics: dict[str, Any]


def _constraint_basis_from_covariances(
    name: str,
    pool_indices: Sequence[int],
    rotations: Sequence[np.ndarray],
    pair_vectors: np.ndarray,
    blocks: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray],
    rows_per_frame: int,
    dimension: int,
    max_constraint_rows: int | None = None,
) -> ProtocolDesign:
    """Select a full-rank literal Eq. (11) row basis for a frame subset.

    When *max_constraint_rows* is provided and smaller than *dimension*, the
    top *max_constraint_rows* QR-pivot rows are kept instead.  This drops
    high-frequency constraints to keep the downstream SDP tractable for
    large active spaces.
    """

    selected = tuple(int(index) for index in pool_indices)
    local_rotations = tuple(np.asarray(rotations[index]) for index in selected)
    local_vectors = np.vstack(
        [
            pair_vectors[index * rows_per_frame : (index + 1) * rows_per_frame]
            for index in selected
        ]
    )
    local_blocks = tuple(np.asarray(blocks[index]) for index in selected)
    local_covariances = tuple(np.asarray(covariances[index]) for index in selected)
    stacked = np.vstack(local_blocks)
    scales = np.sqrt(
        np.concatenate(
            [np.maximum(np.diag(covariance), 1e-12) for covariance in local_covariances]
        )
    )
    weighted = stacked / scales[:, None]
    _, triangular, pivots = qr(weighted.T, mode="economic", pivoting=True)
    tolerance = abs(triangular[0, 0]) * max(weighted.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(np.abs(np.diag(triangular)) > tolerance))
    if rank < dimension:
        raise RuntimeError(f"{name} rank {rank} is below required {dimension}.")
    chosen_count = min(dimension, max_constraint_rows) if max_constraint_rows is not None else dimension
    chosen = np.asarray(pivots[:chosen_count], dtype=int)
    mapping = sorted(
        (int(index // rows_per_frame), int(index % rows_per_frame)) for index in chosen
    )
    global_rows = np.asarray(
        [frame * rows_per_frame + local for frame, local in mapping], dtype=int
    )
    rows_by_frame = []
    selected_blocks = []
    selected_covariances = []
    for local_frame in range(len(selected)):
        local_rows = np.asarray(
            [row for frame, row in mapping if frame == local_frame], dtype=int
        )
        rows_by_frame.append(local_rows)
        selected_blocks.append(local_blocks[local_frame][local_rows])
        selected_covariances.append(
            local_covariances[local_frame][np.ix_(local_rows, local_rows)]
            if len(local_rows)
            else np.empty((0, 0))
        )
    singular = np.linalg.svd(stacked[global_rows], compute_uv=False)
    basis = ConstraintBasis(
        name=name,
        rotations=local_rotations,
        pair_vectors=local_vectors,
        design=quadratic_design(local_vectors),
        global_rows=global_rows,
        rows_by_frame=tuple(rows_by_frame),
        blocks=tuple(selected_blocks),
        covariances=tuple(selected_covariances),
        raw_condition=float(singular[0] / singular[-1]),
        fisher_condition=np.nan,
        rank=dimension,
    )
    unit_counts = np.asarray(
        [1 if len(local) else 0 for local in basis.rows_by_frame], dtype=int
    )
    eigenvalues = np.linalg.eigvalsh(_information(basis, unit_counts))
    basis = replace(basis, fisher_condition=float(eigenvalues[-1] / eigenvalues[0]))
    return ProtocolDesign(
        name=name,
        basis=basis,
        pool_indices=selected,
        paths={},
        pilot_shots=0,
        selection_diagnostics={},
    )


def _target_model(
    args: argparse.Namespace,
    variable_rows: np.ndarray,
    variable_cols: np.ndarray,
    hamiltonian: np.ndarray,
    ftpbe_modes: np.ndarray,
) -> TargetModel:
    return TargetModel(
        d2_weights=np.where(variable_rows == variable_cols, 1.0, 2.0),
        hamiltonian=np.asarray(hamiltonian),
        ftpbe_modes=np.asarray(ftpbe_modes),
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


def _restricted_gain_per_total_shot(
    information: np.ndarray, lineality: np.ndarray, total_shots: int
) -> float | None:
    if lineality.shape[1] == 0:
        return None
    restricted = lineality.T @ information @ lineality
    minimum = float(np.linalg.eigvalsh(restricted)[0])
    return minimum / max(int(total_shots), 1)


def _greedy_frame_order(
    args: argparse.Namespace,
    blocks: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray],
    targets: TargetModel,
    lineality: np.ndarray,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    """Order random candidates using coverage/conic guards and target risk."""

    updates = tuple(
        block.T @ np.linalg.solve(covariance, block)
        for block, covariance in zip(blocks, covariances)
    )
    dimension = updates[0].shape[0]
    full = sum(updates, np.zeros((dimension, dimension)))
    ridge = max(args.design_ridge_fraction * float(np.trace(full)) / dimension, 1e-12)
    measurement_information = np.zeros_like(full)
    regularized = ridge * np.eye(dimension)
    selected = []
    remaining = set(range(len(blocks)))
    diagnostics = []
    while remaining:
        base_sign, base_logdet = np.linalg.slogdet(regularized)
        if base_sign <= 0:
            raise np.linalg.LinAlgError("Regularized design is not positive definite.")
        candidates = []
        for index in sorted(remaining):
            trial_measurement = measurement_information + updates[index]
            trial = regularized + updates[index]
            sign, logdet = np.linalg.slogdet(trial)
            if sign <= 0:
                raise np.linalg.LinAlgError(
                    "Candidate design is not positive definite."
                )
            covariance = _inverse_information(trial)
            risks = _risks(covariance, targets)
            target_score = _risk_score(risks, targets)
            conic_gain = _restricted_gain_per_total_shot(
                trial_measurement, lineality, len(selected) + 1
            )
            candidates.append(
                {
                    "index": index,
                    "target_max": target_score[0],
                    "target_mean": target_score[1],
                    "d_optimal_gain": float(logdet - base_logdet),
                    "conic_gain": conic_gain,
                    "trial": trial,
                    "trial_measurement": trial_measurement,
                }
            )
        maximum_d = max(item["d_optimal_gain"] for item in candidates)
        maximum_c = max(
            (float(item["conic_gain"] or 0.0) for item in candidates), default=0.0
        )
        eligible = [
            item
            for item in candidates
            if item["d_optimal_gain"]
            >= args.d_optimal_guard_fraction * maximum_d - 1e-12
            and (
                maximum_c <= 0.0
                or float(item["conic_gain"] or 0.0)
                >= args.conic_guard_fraction * maximum_c - 1e-12
            )
        ]
        choice = min(
            eligible,
            key=lambda item: (
                item["target_max"],
                item["target_mean"],
                -float(item["conic_gain"] or 0.0),
                -item["d_optimal_gain"],
                item["index"],
            ),
        )
        selected.append(int(choice["index"]))
        remaining.remove(int(choice["index"]))
        regularized = np.asarray(choice["trial"])
        measurement_information = np.asarray(choice["trial_measurement"])
        diagnostics.append(
            {
                "order": len(selected),
                "pool_index": int(choice["index"]),
                "target_max": float(choice["target_max"]),
                "target_mean": float(choice["target_mean"]),
                "d_optimal_gain": float(choice["d_optimal_gain"]),
                "restricted_gain_per_frame_shot": choice["conic_gain"],
            }
        )
    return tuple(selected), diagnostics


def _attach_adaptive_paths(
    args: argparse.Namespace,
    design: ProtocolDesign,
    production_budgets: Sequence[int],
    targets: TargetModel,
    pilot_total: int,
) -> tuple[ProtocolDesign, list[dict[str, Any]]]:
    paths, diagnostics = _allocation_paths(
        design.basis,
        production_budgets,
        args.production_minimum,
        args.allocation_chunk,
        targets,
    )
    adaptive = paths["pilot_adaptive"]
    design = replace(design, paths=adaptive, pilot_shots=pilot_total)
    selected_diagnostics = [
        {**row, "method": design.name}
        for row in diagnostics
        if row["method"] == "pilot_adaptive"
    ]
    return design, selected_diagnostics


def _screen_joint_prefixes(
    args: argparse.Namespace,
    ordering: Sequence[int],
    rotations: Sequence[np.ndarray],
    pair_vectors: np.ndarray,
    blocks: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray],
    rows_per_frame: int,
    dimension: int,
    production_budgets: Sequence[int],
    targets: TargetModel,
    lineality: np.ndarray,
    pilot_total: int,
) -> tuple[ProtocolDesign, list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    allocation_rows = []
    for size in args.candidate_sizes:
        if size > len(ordering):
            continue
        try:
            design = _constraint_basis_from_covariances(
                f"pilot_joint_k{size}",
                ordering[:size],
                rotations,
                pair_vectors,
                blocks,
                covariances,
                rows_per_frame,
                dimension,
                max_constraint_rows=getattr(args, "max_constraint_rows", None),
            )
            design, diagnostics = _attach_adaptive_paths(
                args, design, production_budgets, targets, pilot_total
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            candidates.append(
                {
                    "size": size,
                    "full_rank": False,
                    "reason": str(error),
                }
            )
            continue
        allocation_rows.extend(diagnostics)
        endpoint = diagnostics[-1]
        hit = next(
            (
                pilot_total + int(row["production_shots"])
                for row in diagnostics
                if float(row["predicted_max_capacity_ratio"]) <= 1.0
            ),
            None,
        )
        endpoint_counts = design.paths[int(production_budgets[-1])]
        endpoint_information = _information(design.basis, endpoint_counts)
        restricted_gain = _restricted_gain_per_total_shot(
            endpoint_information, lineality, int(np.sum(endpoint_counts))
        )
        eigenvalues = np.linalg.eigvalsh(endpoint_information)
        row = {
            "size": size,
            "active_frames": len(design.basis.active_frames),
            "full_rank": True,
            "predicted_first_hit_total_shots": hit,
            "endpoint_max_capacity_ratio": float(
                endpoint["predicted_max_capacity_ratio"]
            ),
            "endpoint_mean_capacity_ratio": float(
                endpoint["predicted_mean_capacity_ratio"]
            ),
            "restricted_gain_per_total_shot": restricted_gain,
            "ordinary_information_condition": float(eigenvalues[-1] / eigenvalues[0]),
            "raw_row_condition": design.basis.raw_condition,
            "pilot_fisher_condition": design.basis.fisher_condition,
        }
        candidates.append(row)
        design = replace(design, selection_diagnostics=row)
        candidates[-1]["_design"] = design
    valid = [row for row in candidates if row.get("full_rank")]
    if not valid:
        raise RuntimeError("No random pilot prefix produced a full-rank design.")
    maximum_gain = max(
        float(row["restricted_gain_per_total_shot"] or 0.0) for row in valid
    )
    eligible = [
        row
        for row in valid
        if maximum_gain <= 0.0
        or float(row["restricted_gain_per_total_shot"] or 0.0)
        >= args.conic_guard_fraction * maximum_gain - 1e-12
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["predicted_first_hit_total_shots"] is None,
            row["predicted_first_hit_total_shots"] or 10**18,
            row["endpoint_max_capacity_ratio"],
            -float(row["restricted_gain_per_total_shot"] or 0.0),
            row["size"],
        ),
    )
    chosen = selected.pop("_design")
    public_rows = []
    for row in candidates:
        public_rows.append(
            {key: value for key, value in row.items() if key != "_design"}
        )
    return chosen, public_rows, allocation_rows


def _equal_counts(
    frame_count: int, active: Sequence[int], budget: int, quantum: int
) -> np.ndarray:
    active = tuple(int(index) for index in active)
    if budget < len(active) * quantum or budget % quantum:
        raise ValueError("Uniform budget cannot fund its active frame basis.")
    counts = np.zeros(frame_count, dtype=int)
    for block in range(budget // quantum):
        counts[active[block % len(active)]] += quantum
    return counts


def _pool_counts(
    design: ProtocolDesign, local_counts: np.ndarray, pool_size: int
) -> np.ndarray:
    counts = np.zeros(pool_size, dtype=int)
    for local, pool in enumerate(design.pool_indices):
        counts[pool] = int(local_counts[local])
    return counts


def _local_outcomes(
    design: ProtocolDesign, pool_outcomes: Sequence[np.ndarray]
) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(pool_outcomes[index]) for index in design.pool_indices)


def _online_capacity_ratio(
    args: argparse.Namespace,
    design: ProtocolDesign,
    outcomes: Sequence[np.ndarray],
    counts: np.ndarray,
    selection: Any,
    objective: Any,
    result: Any,
    baseline: Any,
    variable_rows: np.ndarray,
    variable_cols: np.ndarray,
) -> float:
    covariances = []
    for frame, local_rows in enumerate(design.basis.rows_by_frame):
        if not len(local_rows):
            covariances.append(np.empty((0, 0)))
            continue
        sample = np.asarray(outcomes[frame][: int(counts[frame])], dtype=float)
        full = _regularized_single_covariance(sample)
        covariances.append(full[np.ix_(local_rows, local_rows)])
    empirical = replace(design.basis, covariances=tuple(covariances))
    covariance = _inverse_information(_information(empirical, counts))
    hamiltonian = _hamiltonian_gradient(selection, variable_rows, variable_cols)
    current_gradient = _raw_ftpbe_gradient(
        objective,
        selection,
        np.asarray(result.d2),
        np.asarray(result.gamma),
        variable_rows,
        variable_cols,
    )
    baseline_gradient = _raw_ftpbe_gradient(
        objective,
        selection,
        np.asarray(baseline.d2),
        np.asarray(baseline.gamma),
        variable_rows,
        variable_cols,
    )
    targets = _target_model(
        args,
        variable_rows,
        variable_cols,
        hamiltonian,
        np.vstack((current_gradient, baseline_gradient)),
    )
    return _risk_score(_risks(covariance, targets), targets)[0]


def _validate_args(args: argparse.Namespace) -> None:
    args.solver = args.solver.upper()
    if args.solver != "MOSEK":
        raise ValueError("This runner fixes the formal solver to MOSEK.")
    if args.frame_count < 5 or args.pilot_shots_per_frame < 2:
        raise ValueError("At least five frames and two pilot shots are required.")
    if any(size < 1 or size > args.frame_count for size in args.candidate_sizes):
        raise ValueError("candidate-sizes must lie inside the random pool.")
    if args.frame_count not in args.candidate_sizes:
        raise ValueError("candidate-sizes must include frame-count.")
    if args.production_minimum < 2 or args.allocation_chunk < 1:
        raise ValueError("Production minimum must be at least two.")
    if args.production_minimum % args.allocation_chunk:
        raise ValueError("production-minimum must align with allocation-chunk.")
    if not 0.0 < args.d_optimal_guard_fraction <= 1.0:
        raise ValueError("Invalid D-optimal guard fraction.")
    if not 0.0 < args.conic_guard_fraction <= 1.0:
        raise ValueError("Invalid conic guard fraction.")
    if args.design_ridge_fraction <= 0.0 or args.stability_window < 1:
        raise ValueError("Design ridge and stability window must be positive.")
    pilot_total = args.frame_count * args.pilot_shots_per_frame
    if min(args.budgets) <= pilot_total:
        raise ValueError("Every total budget must exceed the full-pool pilot cost.")
    if any(budget % args.allocation_chunk for budget in args.budgets):
        raise ValueError("Budgets must align with allocation-chunk.")
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
    if (
        np.any(capacities <= 0.0)
        or np.any(floors < 0.0)
        or np.any(floors >= capacities)
    ):
        raise ValueError("Capacities must be positive and exceed hardware floors.")


def _configuration(args: argparse.Namespace, is_complex: np.ndarray) -> dict[str, Any]:
    pilot_total = args.frame_count * args.pilot_shots_per_frame
    system = str(getattr(args, "system", "N2"))
    return {
        "system": system,
        "bond_length_angstrom": args.bond_length,
        "basis": args.basis,
        "active_space": [args.active_electrons, args.active_orbitals],
        "frame_pool": "state-independent random Haar O(n)+U(n)",
        "frame_count": args.frame_count,
        "real_frames": int(np.count_nonzero(~is_complex)),
        "complex_frames": int(np.count_nonzero(is_complex)),
        "candidate_sizes": list(args.candidate_sizes),
        "budgets": list(args.budgets),
        "pilot_cost_total_shots": pilot_total,
        "cost_accounting": {
            "uniform30": "total = production; no pilot charged",
            "pilot_allocate30": "total = all-candidate pilot + production",
            "pilot_joint": "total = all-candidate pilot + selected-frame production",
        },
        "pilot_reused_in_formal_reconstruction": False,
        "frame_selection": (
            "pilot full-covariance target risk with D-optimal coverage and local "
            "DQG-lineality restricted-gain guards"
        ),
        "ordinary_condition_role": "diagnostic only",
        "restricted_gain_scope": (
            "local DQG active-lineality proxy at the pilot solution; not a full "
            "nuclear-norm critical-cone certificate"
        ),
        "d2_capacity": args.d2_capacity,
        "energy_capacity_meh": args.energy_capacity_meh,
        "formal_reconstruction": "paper Eq. (11) nuclear norm with DQG",
        "solver": "MOSEK",
        "weighted_gls_used": False,
        "consistency_projection_used": False,
        "online_exact_state_access": False,
        "common_reachable_reference": (
            "exact-mean Eq. (11) solution for the frozen full random pool; posthoc only"
        ),
    }


def _run_seed(
    args: argparse.Namespace,
    seed: int,
    selection: Any,
    exact: Any,
    objective: Any,
    rotations: Sequence[np.ndarray],
    is_complex: np.ndarray,
    pair_vectors: np.ndarray,
    blocks: Sequence[np.ndarray],
    variable_rows: np.ndarray,
    variable_cols: np.ndarray,
    structural: ProtocolDesign,
    reachable: Any,
    baseline: Any,
) -> None:
    output = args.output_dir / f"shot_seed_{seed}"
    analysis_path = output / "analysis.json"
    if analysis_path.is_file() and not args.overwrite:
        print(f"[shot_seed={seed}] complete checkpoint", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    oracle = AcquisitionOracle(exact, rotations, pair_vectors, 1, seed)
    rows_per_frame = len(selection.pairs)
    pilot_total = args.frame_count * args.pilot_shots_per_frame
    production_budgets = tuple(int(budget - pilot_total) for budget in args.budgets)
    pilot_counts = np.full(args.frame_count, args.pilot_shots_per_frame, dtype=int)
    pilot_outcomes = _sample_outcomes(oracle, seed + 100_003, pilot_counts)
    pilot_covariances = tuple(
        _regularized_single_covariance(np.asarray(outcomes, dtype=float))
        for outcomes in pilot_outcomes
    )
    _max_cr = getattr(args, "max_constraint_rows", None)
    full_pilot = _constraint_basis_from_covariances(
        "pilot_allocate30",
        tuple(range(args.frame_count)),
        rotations,
        pair_vectors,
        blocks,
        pilot_covariances,
        rows_per_frame,
        len(variable_rows),
        max_constraint_rows=_max_cr,
    )
    pilot_shadows = _finite_shadows(full_pilot.basis, pilot_outcomes, pilot_counts)
    print(f"[shot_seed={seed}] solving blind full-pool pilot Eq. (11)", flush=True)
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
    targets = _target_model(
        args,
        variable_rows,
        variable_cols,
        hamiltonian,
        np.vstack((ftpbe_pilot, ftpbe_baseline)),
    )
    local_reference = SimpleNamespace(
        **vars(selection),
        n_electrons=selection.n_electrons,
        exact_d2=np.asarray(pilot.d2),
        exact_gamma=np.asarray(pilot.gamma),
    )
    lineality, active_diagnostics = _dqg_lineality_basis(
        local_reference, variable_rows, variable_cols
    )
    ordering, ordering_rows = _greedy_frame_order(
        args, blocks, pilot_covariances, targets, lineality
    )
    full_pilot, full_diagnostics = _attach_adaptive_paths(
        args, full_pilot, production_budgets, targets, pilot_total
    )
    joint, size_rows, joint_diagnostics = _screen_joint_prefixes(
        args,
        ordering,
        rotations,
        pair_vectors,
        blocks,
        pilot_covariances,
        rows_per_frame,
        len(variable_rows),
        production_budgets,
        targets,
        lineality,
        pilot_total,
    )
    uniform_paths = {
        int(budget): _equal_counts(
            len(structural.basis.rotations),
            structural.basis.active_frames,
            int(budget),
            args.allocation_chunk,
        )
        for budget in args.budgets
    }
    uniform = replace(structural, paths=uniform_paths, pilot_shots=0)
    designs = {
        "uniform30": uniform,
        "pilot_allocate30": full_pilot,
        "pilot_joint": joint,
    }
    _write_csv(output / "frame_ordering.csv", ordering_rows)
    _write_csv(output / "joint_size_screen.csv", size_rows)
    _write_csv(
        output / "allocation_diagnostics.csv", full_diagnostics + joint_diagnostics
    )
    _atomic_json(
        output / "blind_policy.json",
        {
            "shot_seed": seed,
            "frame_types": ["complex" if flag else "real" for flag in is_complex],
            "frame_order_pool_indices_zero_based": list(ordering),
            "selected_joint_size": len(joint.pool_indices),
            "selected_joint_pool_indices_zero_based": list(joint.pool_indices),
            "selected_joint_active_frames": len(joint.basis.active_frames),
            "active_dqg_diagnostics": active_diagnostics,
            "pilot_total_shots": pilot_total,
            "online_exact_state_access": False,
            "pilot_reused_in_formal_reconstruction": False,
            "paths": {
                method: {
                    str(total): design.paths[
                        total if method == "uniform30" else total - pilot_total
                    ].tolist()
                    for total in args.budgets
                }
                for method, design in designs.items()
            },
        },
    )
    maximum_pool_counts = np.zeros(args.frame_count, dtype=int)
    for method, design in designs.items():
        for total in args.budgets:
            path_key = total if method == "uniform30" else total - pilot_total
            maximum_pool_counts = np.maximum(
                maximum_pool_counts,
                _pool_counts(design, design.paths[path_key], args.frame_count),
            )
    production_outcomes = _sample_outcomes(oracle, seed + 200_003, maximum_pool_counts)
    reachable_h, reachable_f = _energy_values(
        selection, objective, np.asarray(reachable.d2), np.asarray(reachable.gamma)
    )
    exact_h, exact_f = _energy_values(
        selection, objective, exact.exact_d2, exact.exact_gamma
    )
    rows = []
    for method, design in designs.items():
        warm_d2 = np.asarray(pilot.d2 if method != "uniform30" else baseline.d2)
        warm_gamma = np.asarray(
            pilot.gamma if method != "uniform30" else baseline.gamma
        )
        warm_corrected = np.asarray(
            pilot.corrected_d2 if method != "uniform30" else baseline.d2
        )
        local_outcomes = _local_outcomes(design, production_outcomes)
        previous: tuple[np.ndarray, float, float] | None = None
        stability_flags = []
        for total in args.budgets:
            path_key = total if method == "uniform30" else total - pilot_total
            counts = design.paths[path_key]
            label = f"{method}_{total}"
            checkpoint = output / "checkpoints" / f"{label}.npz"
            metadata_path = output / "checkpoints" / f"{label}.json"
            if checkpoint.is_file() and metadata_path.is_file() and not args.overwrite:
                with np.load(checkpoint, allow_pickle=False) as archive:
                    warm_d2 = np.asarray(archive["d2"])
                    warm_gamma = np.asarray(archive["gamma"])
                    warm_corrected = np.asarray(archive["corrected_d2"])
                row = json.loads(metadata_path.read_text(encoding="utf-8"))
                rows.append(row)
                current_h, current_f = _energy_values(
                    selection, objective, warm_d2, warm_gamma
                )
                previous = (warm_d2.copy(), current_h, current_f)
                stability_flags.append(bool(row["online_stability_pass"]))
                print(
                    f"[seed={seed} {method} total={total}] checkpoint",
                    flush=True,
                )
                continue
            shadows = _finite_shadows(design.basis, local_outcomes, counts)
            print(f"[seed={seed} {method} total={total}] solving Eq. (11)", flush=True)
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
            current_h, current_f = _energy_values(
                selection, objective, warm_d2, warm_gamma
            )
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
            if previous is None:
                changes = (None, None, None)
            else:
                changes = (
                    float(np.linalg.norm(warm_d2 - previous[0])),
                    float(1000.0 * abs(current_h - previous[1])),
                    float(1000.0 * abs(current_f - previous[2])),
                )
            previous = (warm_d2.copy(), current_h, current_f)
            online_ratio = _online_capacity_ratio(
                args,
                design,
                local_outcomes,
                counts,
                selection,
                objective,
                result,
                baseline,
                variable_rows,
                variable_cols,
            )
            stability_pass = bool(
                changes[0] is not None
                and changes[0] <= args.d2_capacity
                and changes[1] <= args.energy_capacity_meh
                and changes[2] <= args.energy_capacity_meh
                and online_ratio <= 1.0
            )
            stability_flags.append(stability_pass)
            online_stop = bool(
                len(stability_flags) >= args.stability_window
                and all(stability_flags[-args.stability_window :])
            )
            production_shots = int(np.sum(counts))
            row = {
                "shot_seed": seed,
                "method": method,
                "total_shots": int(total),
                "pilot_shots": int(design.pilot_shots),
                "production_shots": production_shots,
                "cost_identity_holds": bool(
                    int(design.pilot_shots) + production_shots == int(total)
                ),
                "candidate_frames_piloted": (
                    args.frame_count if design.pilot_shots else 0
                ),
                "selected_frames": len(design.pool_indices),
                "active_production_frames": len(design.basis.active_frames),
                "selected_pool_indices_zero_based": ",".join(
                    map(str, design.pool_indices)
                ),
                "complex_production_shots": int(
                    np.sum(_pool_counts(design, counts, args.frame_count)[is_complex])
                ),
                "minimum_active_frame_shots": int(
                    np.min(counts[list(design.basis.active_frames)])
                ),
                "maximum_active_frame_shots": int(
                    np.max(counts[list(design.basis.active_frames)])
                ),
                "shot_counts": ",".join(map(str, counts)),
                "online_predicted_max_capacity_ratio": online_ratio,
                "d2_change_from_previous": changes[0],
                "hamiltonian_change_from_previous_meh": changes[1],
                "ftpbe_change_from_previous_meh": changes[2],
                "online_stability_pass": stability_pass,
                "online_stop_recommended": online_stop,
                "d2_error_to_reachable": reachable_metrics["d2_error"],
                "hamiltonian_error_to_reachable_meh": reachable_metrics[
                    "hamiltonian_error_meh"
                ],
                "ftpbe_error_to_reachable_meh": reachable_metrics["ftpbe_error_meh"],
                "signed_hamiltonian_error_to_reachable_meh": reachable_metrics[
                    "signed_hamiltonian_error_meh"
                ],
                "signed_ftpbe_error_to_reachable_meh": reachable_metrics[
                    "signed_ftpbe_error_meh"
                ],
                "d2_error_to_exact": exact_metrics["d2_error"],
                "hamiltonian_error_to_exact_meh": exact_metrics[
                    "hamiltonian_error_meh"
                ],
                "ftpbe_error_to_exact_meh": exact_metrics["ftpbe_error_meh"],
                "reachable_capacity_pass": bool(
                    reachable_metrics["d2_error"] <= args.d2_capacity
                    and reachable_metrics["hamiltonian_error_meh"]
                    <= args.energy_capacity_meh
                    and reachable_metrics["ftpbe_error_meh"] <= args.energy_capacity_meh
                ),
                "energy_capacity_pass": bool(
                    reachable_metrics["hamiltonian_error_meh"]
                    <= args.energy_capacity_meh
                    and reachable_metrics["ftpbe_error_meh"] <= args.energy_capacity_meh
                ),
                "solver_seconds": time.perf_counter() - started,
                "status": str(result.status),
                "fit_status": str(result.fit_status),
            }
            rows.append(row)
            _atomic_npz(
                checkpoint,
                d2=warm_d2,
                gamma=warm_gamma,
                corrected_d2=warm_corrected,
            )
            _atomic_json(metadata_path, row)
            print(
                f"[seed={seed} {method} total={total}] "
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
            "selected_joint": joint.selection_diagnostics,
            "rows": rows,
        },
    )


def _plot_curves(args: argparse.Namespace, rows: Sequence[dict[str, Any]]) -> None:
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
    figure, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)
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
                    [row["total_shots"] for row in selected],
                    [row[key] for row in selected],
                    color=COLORS[method],
                    alpha=0.18,
                    linewidth=1.0,
                )
            medians = [
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
                for budget in args.budgets
            ]
            axis.plot(
                args.budgets,
                medians,
                color=COLORS[method],
                linewidth=2.2,
                marker="o",
                markersize=4,
                label=LABELS[method],
            )
        axis.axhline(threshold, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Total quantum shots (pilot included)")
    figure.suptitle(f"{getattr(args, 'system', 'N2')} random-frame pilot joint design")
    figure.tight_layout()
    figure.savefig(args.output_dir / "reachable_error_curves.png", dpi=220)
    plt.close(figure)


def _plot_boxplots(args: argparse.Namespace, rows: Sequence[dict[str, Any]]) -> None:
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
    centers = np.arange(len(args.budgets), dtype=float)
    offsets = {"uniform30": -0.25, "pilot_allocate30": 0.0, "pilot_joint": 0.25}
    figure, axes = plt.subplots(3, 1, figsize=(10.0, 9.0), sharex=True)
    for axis, (key, ylabel, threshold) in zip(axes, metrics):
        for method in METHODS:
            values = [
                [
                    row[key]
                    for row in rows
                    if row["method"] == method and int(row["total_shots"]) == budget
                ]
                for budget in args.budgets
            ]
            artists = axis.boxplot(
                values,
                positions=centers + offsets[method],
                widths=0.22,
                patch_artist=True,
                manage_ticks=False,
            )
            for box in artists["boxes"]:
                box.set_facecolor(COLORS[method])
                box.set_alpha(0.55)
            for median in artists["medians"]:
                median.set_color("#111111")
        axis.axhline(threshold, color="#555555", linestyle="--", linewidth=1.0)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(centers, [f"{budget // 1000}k" for budget in args.budgets])
    axes[-1].set_xlabel("Total quantum shots (pilot included)")
    handles = [
        plt.Line2D([0], [0], color=COLORS[method], linewidth=8, alpha=0.55)
        for method in METHODS
    ]
    axes[0].legend(
        handles, [LABELS[method] for method in METHODS], frameon=False, ncol=3
    )
    figure.suptitle(
        f"{getattr(args, 'system', 'N2')} random-frame pilot joint design distributions"
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "reachable_error_boxplots.png", dpi=220)
    plt.close(figure)


def _aggregate(args: argparse.Namespace) -> None:
    rows = []
    selected_joint = []
    for seed in args.shot_seeds:
        path = args.output_dir / f"shot_seed_{seed}" / "analysis.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
        selected_joint.append({"shot_seed": seed, **payload["selected_joint"]})
    summaries = []
    stopping = []
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
                    "median_selected_frames": float(
                        np.median([row["selected_frames"] for row in selected])
                    ),
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
                    "energy_hit_rate": float(
                        np.mean(
                            [
                                float(row["hamiltonian_error_to_reachable_meh"])
                                <= args.energy_capacity_meh
                                and float(row["ftpbe_error_to_reachable_meh"])
                                <= args.energy_capacity_meh
                                for row in selected
                            ]
                        )
                    ),
                }
            )
        online_by_seed = {}
        posthoc_by_seed = {}
        energy_by_seed = {}
        for seed in args.shot_seeds:
            trajectory = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method and int(row["shot_seed"]) == seed
                ),
                key=lambda row: int(row["total_shots"]),
            )
            online_hits = [
                row["total_shots"]
                for row in trajectory
                if row["online_stop_recommended"]
            ]
            posthoc_hits = [
                row["total_shots"]
                for row in trajectory
                if row["reachable_capacity_pass"]
            ]
            energy_hits = [
                row["total_shots"]
                for row in trajectory
                if float(row["hamiltonian_error_to_reachable_meh"])
                <= args.energy_capacity_meh
                and float(row["ftpbe_error_to_reachable_meh"])
                <= args.energy_capacity_meh
            ]
            online_by_seed[str(seed)] = online_hits[0] if online_hits else None
            posthoc_by_seed[str(seed)] = posthoc_hits[0] if posthoc_hits else None
            energy_by_seed[str(seed)] = energy_hits[0] if energy_hits else None
        finite_energy = [
            value for value in energy_by_seed.values() if value is not None
        ]
        finite_all_three = [
            value for value in posthoc_by_seed.values() if value is not None
        ]
        stopping.append(
            {
                "method": method,
                "online_first_stop_by_seed": online_by_seed,
                "posthoc_first_energy_hit_by_seed": energy_by_seed,
                "posthoc_first_capacity_hit_by_seed": posthoc_by_seed,
                "energy_hit_fraction": len(finite_energy) / len(args.shot_seeds),
                "median_first_energy_hit_total_shots": (
                    float(np.median(finite_energy)) if finite_energy else None
                ),
                "all_three_hit_fraction": len(finite_all_three) / len(args.shot_seeds),
                "median_first_all_three_hit_total_shots": (
                    float(np.median(finite_all_three)) if finite_all_three else None
                ),
            }
        )
    _write_csv(args.output_dir / "all_results.csv", rows)
    _write_csv(args.output_dir / "aggregate_by_budget.csv", summaries)
    _plot_curves(args, rows)
    _plot_boxplots(args, rows)
    _atomic_json(
        args.output_dir / "summary.json",
        {
            "summaries": summaries,
            "stopping": stopping,
            "selected_joint_by_seed": selected_joint,
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
    print("Building selection-only N2 data and frozen random frame pool...", flush=True)
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
    pair_vectors, _, blocks, variable_rows, variable_cols = _build_random_design(
        selection, rotations
    )
    structural_covariances = tuple(
        np.eye(len(selection.pairs), dtype=float) for _ in rotations
    )
    structural = _constraint_basis_from_covariances(
        "uniform30",
        tuple(range(args.frame_count)),
        rotations,
        pair_vectors,
        blocks,
        structural_covariances,
        len(selection.pairs),
        len(variable_rows),
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
    print("Solving shadow-free DQG initialization with MOSEK...", flush=True)
    baseline = solve_dqg_sdp(
        LeakGuardReference(selection),
        solver=args.solver,
        tolerance=args.solver_tolerance,
        max_iterations=args.max_iterations,
        solver_threads=args.solver_threads,
        positivity_conditions="DQG",
        symmetry_blocked_psd=True,
    )
    exact = build_n2_reference(
        args.bond_length,
        basis=args.basis,
        active_electrons=args.active_electrons,
        active_orbitals=args.active_orbitals,
    )
    np.testing.assert_allclose(selection.one_body, exact.one_body, atol=2e-9)
    np.testing.assert_allclose(selection.two_body, exact.two_body, atol=2e-9)
    # Common posthoc D-dagger.  It is never passed into frame selection/allocation.
    reachable = _solve_eq11(
        args,
        selection,
        _exact_mean_shadows(structural.basis, exact.exact_d2),
        np.asarray(baseline.d2),
        np.asarray(baseline.gamma),
        np.asarray(baseline.d2),
    )
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
            blocks,
            variable_rows,
            variable_cols,
            structural,
            reachable,
            baseline,
        )
    _aggregate(args)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
