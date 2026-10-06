#!/usr/bin/env python3
"""Joint frame/shot oracle for N2 constrained shadow tomography.

Frame prefixes come from the existing exact-information manifold design.  For
each prefix, a pivoted basis of literal pair-occupation constraints removes
stochastic left-null relations, so independent finite-shot means are feasible
for the public Eq. (11) model without a consistency projection.  Frames and
shots are ranked by D2/H/ftPBE effective shadow risks; DQG restricted gain and
an empirical nuclear-weight recovery margin are reported as conic diagnostics.
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/n2-joint-shadow-conic-oracle")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import null_space, qr  # noqa: E402


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ROOT / "code"
VENDOR = CODE / "vendor"
for path in (CODE, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constrained_shadow import (  # noqa: E402
    ShadowData,
    contract_one_rdm,
    dqg_matrices,
    exact_shadows,
    solve_dqg_sdp,
)
from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    shadow_design_blocks,
)
from mcpdft_selector import _symmetric_d2_variables  # noqa: E402
from run_c2_ftpbe_shot_reallocation import AcquisitionOracle  # noqa: E402
from run_c2_oracle_subset_reallocation import (  # noqa: E402
    _hamiltonian_gradient,
    _raw_ftpbe_gradient,
)
from run_n2_ftpbe_shot_cost_oracle import _exact_single_covariance  # noqa: E402
from run_n2_paper_nuclear_hybrid_trajectories import (  # noqa: E402
    DEFAULT_NOISE_SEEDS,
    _blind_shadows,
    _build_references,
    mixed_orbital_rotations,
)


DEFAULT_OUTPUT = (
    ROOT
    / "sweeps"
    / "n2_paper_nuclear_hybrid_m1_30_five_seeds"
    / "joint_shadow_norm_conic_oracle"
)
DEFAULT_DESIGN = ROOT / "sweeps" / "n2_equilibrium_manifold_derandomization_oracle"
DEFAULT_BUDGETS = (30_000, 60_000, 120_000, 180_000, 240_000, 300_000)
DEFAULT_SIZES = (5, 8, 10, 12, 14)
FORMAL_METHODS = ("uniform30_rank_basis", "joint_shadow_norm")
REPORT_METHODS = FORMAL_METHODS + ("posthoc_library_oracle",)
NUCLEAR_WEIGHT_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
METRICS = (
    ("d2_frobenius_error", "2-RDM Frobenius error"),
    ("hamiltonian_error_meh", "Hamiltonian error (mEh)"),
    ("ftpbe_error_meh", "ftPBE error (mEh)"),
)


def _parse_integers(value: str) -> tuple[int, ...]:
    result: set[int] = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(part) for part in item.split("-", 1))
            if first > last:
                raise argparse.ArgumentTypeError("Integer ranges must be increasing.")
            result.update(range(first, last + 1))
        else:
            result.add(int(item))
    if not result:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return tuple(sorted(result))


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifold-design-dir", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument(
        "--shot-seeds", type=_parse_integers, default=DEFAULT_NOISE_SEEDS
    )
    parser.add_argument("--budgets", type=_parse_integers, default=DEFAULT_BUDGETS)
    parser.add_argument("--sizes", type=_parse_integers, default=DEFAULT_SIZES)
    parser.add_argument("--bond-length", type=float, default=1.10)
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--active-electrons", type=int, default=6)
    parser.add_argument("--active-orbitals", type=int, default=6)
    parser.add_argument("--d2-target", type=float, default=0.03)
    parser.add_argument("--energy-target-meh", type=float, default=1.6)
    parser.add_argument("--allocation-minimum", type=int, default=500)
    parser.add_argument("--allocation-chunk", type=int, default=500)
    parser.add_argument("--nuclear-weight", type=float, default=1.0)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--solver-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--design-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args(argv)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


@dataclass(frozen=True)
class ConstraintBasis:
    name: str
    rotations: tuple[np.ndarray, ...]
    pair_vectors: np.ndarray
    design: Any
    global_rows: np.ndarray
    rows_by_frame: tuple[np.ndarray, ...]
    blocks: tuple[np.ndarray, ...]
    covariances: tuple[np.ndarray, ...]
    raw_condition: float
    fisher_condition: float
    rank: int

    @property
    def active_frames(self) -> tuple[int, ...]:
        return tuple(
            index for index, rows in enumerate(self.rows_by_frame) if len(rows)
        )


def _risk_components(
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


def _target_score(args: argparse.Namespace, risks: np.ndarray) -> tuple[float, float]:
    ratios = risks / np.asarray(
        [
            args.d2_target**2,
            (args.energy_target_meh / 1000.0) ** 2,
            (args.energy_target_meh / 1000.0) ** 2,
        ]
    )
    return float(np.max(ratios)), float(np.mean(ratios))


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


def _equal_counts(frame_count: int, budget: int, quantum: int = 1) -> np.ndarray:
    if budget < frame_count * quantum:
        raise ValueError("Budget cannot fund every active frame.")
    counts = np.full(
        frame_count, (budget // frame_count // quantum) * quantum, dtype=int
    )
    remaining = budget - int(np.sum(counts))
    for index in range(remaining // quantum):
        counts[index % frame_count] += quantum
    if int(np.sum(counts)) != budget:
        raise ValueError("Budget does not align with allocation quantum.")
    return counts


def _information(basis: ConstraintBasis, counts: np.ndarray) -> np.ndarray:
    dimension = basis.blocks[0].shape[1]
    information = np.zeros((dimension, dimension))
    for index in basis.active_frames:
        block = basis.blocks[index]
        inverse = np.linalg.inv(basis.covariances[index])
        information += int(counts[index]) * (block.T @ inverse @ block)
    return 0.5 * (information + information.T)


def _allocation_path(
    args: argparse.Namespace,
    basis: ConstraintBasis,
    d2_weights: np.ndarray,
    hamiltonian: np.ndarray,
    ftpbe: np.ndarray,
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    active = basis.active_frames
    counts = np.zeros(len(basis.rotations), dtype=int)
    counts[list(active)] = args.allocation_minimum
    information = _information(basis, counts)
    covariance = np.linalg.inv(information)
    allocations: dict[int, np.ndarray] = {}
    diagnostics = []
    for budget in args.budgets:
        if budget < int(np.sum(counts)):
            raise ValueError("Budget is below the joint-design minimum allocation.")
        while int(np.sum(counts)) < budget:
            candidates = []
            for index in active:
                trial = _updated_covariance(
                    covariance,
                    basis.blocks[index],
                    basis.covariances[index],
                    args.allocation_chunk,
                )
                risks = _risk_components(trial, d2_weights, hamiltonian, ftpbe)
                worst_direction = float(
                    np.max(
                        np.linalg.eigvalsh(
                            np.sqrt(d2_weights)[:, None]
                            * trial
                            * np.sqrt(d2_weights)[None, :]
                        )
                    )
                )
                candidates.append(
                    (
                        _target_score(args, risks),
                        worst_direction,
                        int(counts[index]),
                        index,
                        trial,
                    )
                )
            _, _, _, selected, covariance = min(
                candidates,
                key=lambda item: (item[0][0], item[0][1], item[1], item[2], item[3]),
            )
            counts[selected] += args.allocation_chunk
        risks = _risk_components(covariance, d2_weights, hamiltonian, ftpbe)
        allocations[int(budget)] = counts.copy()
        diagnostics.append(
            {
                "basis": basis.name,
                "total_shots": int(budget),
                "target_max_variance_ratio": _target_score(args, risks)[0],
                "target_mean_variance_ratio": _target_score(args, risks)[1],
                "d2_predicted_rms": float(np.sqrt(risks[0])),
                "hamiltonian_predicted_rms_meh": float(1000.0 * np.sqrt(risks[1])),
                "ftpbe_predicted_rms_meh": float(1000.0 * np.sqrt(risks[2])),
                "d2_effective_shadow_norm_squared": float(budget * risks[0]),
                "hamiltonian_effective_shadow_norm_squared": float(budget * risks[1]),
                "ftpbe_effective_shadow_norm_squared": float(budget * risks[2]),
                "minimum_frame_shots": int(np.min(counts[list(active)])),
                "maximum_frame_shots": int(np.max(counts[list(active)])),
                "shot_counts": ",".join(map(str, counts)),
            }
        )
    return allocations, diagnostics


def _constraint_basis(
    name: str,
    rotations: Sequence[np.ndarray],
    exact: Any,
    rows: np.ndarray,
    cols: np.ndarray,
) -> ConstraintBasis:
    exact_data = exact_shadows(exact, rotations)
    blocks = shadow_design_blocks(exact_data, len(exact.pairs), rows, cols)
    oracle = AcquisitionOracle(
        exact,
        rotations,
        np.asarray(exact_data.pair_vectors),
        1,
        0,
    )
    covariances = []
    diagonal_variances = []
    for index in range(len(rotations)):
        _, covariance = _exact_single_covariance(
            oracle._frame_probabilities(index), oracle.indicators
        )
        covariances.append(np.asarray(covariance))
        diagonal_variances.append(np.maximum(np.diag(covariance), 1e-12))
    stacked = np.vstack(blocks)
    scales = np.sqrt(np.concatenate(diagonal_variances))
    weighted = stacked / scales[:, None]
    _, triangular, pivots = qr(weighted.T, mode="economic", pivoting=True)
    tolerance = abs(triangular[0, 0]) * max(weighted.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(np.abs(np.diag(triangular)) > tolerance))
    dimension = len(rows)
    if rank < dimension:
        raise RuntimeError(f"{name} has rank {rank}, below required {dimension}.")
    chosen = np.asarray(pivots[:dimension], dtype=int)
    rows_per_frame = len(exact.pairs)
    mapping = sorted(
        (
            (int(index // rows_per_frame), int(index % rows_per_frame))
            for index in chosen
        )
    )
    global_rows = np.asarray(
        [frame * rows_per_frame + local for frame, local in mapping], dtype=int
    )
    selected_blocks = []
    selected_covariances = []
    rows_by_frame = []
    for frame in range(len(rotations)):
        local = np.asarray(
            [row for selected_frame, row in mapping if selected_frame == frame],
            dtype=int,
        )
        rows_by_frame.append(local)
        selected_blocks.append(np.asarray(blocks[frame])[local])
        selected_covariances.append(
            np.asarray(covariances[frame])[np.ix_(local, local)]
            if len(local)
            else np.empty((0, 0))
        )
    selected_matrix = stacked[global_rows]
    singular = np.linalg.svd(selected_matrix, compute_uv=False)
    reference_counts = np.zeros(len(rotations), dtype=int)
    reference_counts[
        list(index for index, local in enumerate(rows_by_frame) if len(local))
    ] = 1
    fisher = _information(
        ConstraintBasis(
            name=name,
            rotations=tuple(np.asarray(item) for item in rotations),
            pair_vectors=np.asarray(exact_data.pair_vectors),
            design=exact_data.design,
            global_rows=global_rows,
            rows_by_frame=tuple(rows_by_frame),
            blocks=tuple(selected_blocks),
            covariances=tuple(selected_covariances),
            raw_condition=1.0,
            fisher_condition=1.0,
            rank=dimension,
        ),
        reference_counts,
    )
    fisher_eigenvalues = np.linalg.eigvalsh(fisher)
    return ConstraintBasis(
        name=name,
        rotations=tuple(np.asarray(item) for item in rotations),
        pair_vectors=np.asarray(exact_data.pair_vectors),
        design=exact_data.design,
        global_rows=global_rows,
        rows_by_frame=tuple(rows_by_frame),
        blocks=tuple(selected_blocks),
        covariances=tuple(selected_covariances),
        raw_condition=float(singular[0] / singular[-1]),
        fisher_condition=float(fisher_eigenvalues[-1] / fisher_eigenvalues[0]),
        rank=dimension,
    )


def _dqg_lineality_basis(
    exact: Any, rows: np.ndarray, cols: np.ndarray, active_tolerance: float = 1e-8
) -> tuple[np.ndarray, dict[str, Any]]:
    exact_d, exact_q, exact_g = dqg_matrices(
        exact.exact_d2, exact.exact_gamma, exact.pairs
    )
    matrices = (exact_d, exact_q, exact_g)
    active_vectors = []
    spectra = {}
    for label, matrix in zip(("D", "Q", "G"), matrices):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        active = eigenvectors[:, eigenvalues <= active_tolerance]
        positive = eigenvalues[eigenvalues > active_tolerance]
        active_vectors.append(active)
        spectra[label] = {
            "active_eigenvalues": int(active.shape[1]),
            "minimum_eigenvalue": float(eigenvalues[0]),
            "minimum_positive_eigenvalue": float(np.min(positive))
            if len(positive)
            else None,
            "positive_spectral_condition": (
                float(np.max(positive) / np.min(positive)) if len(positive) else None
            ),
        }
    derivative_blocks = [[] for _ in matrices]
    for row, col in zip(rows, cols):
        perturbation = np.zeros_like(exact.exact_d2)
        perturbation[row, col] = 1.0
        perturbation[col, row] = 1.0
        if row == col:
            perturbation[row, col] = 1.0
        gamma_delta = contract_one_rdm(
            perturbation,
            exact.n_spin_orbitals,
            exact.n_electrons,
            exact.pairs,
        )
        trial = dqg_matrices(
            exact.exact_d2 + perturbation,
            exact.exact_gamma + gamma_delta,
            exact.pairs,
        )
        for index in range(3):
            derivative_blocks[index].append(trial[index] - matrices[index])
    constraints = []
    for active, derivatives in zip(active_vectors, derivative_blocks):
        if active.shape[1] == 0:
            continue
        upper = np.triu_indices(active.shape[1])
        constraints.append(
            np.column_stack(
                [(active.T @ derivative @ active)[upper] for derivative in derivatives]
            ).T
        )
    constraint_matrix = (
        np.vstack([item.T for item in constraints])
        if constraints
        else np.empty((0, len(rows)))
    )
    lineality = null_space(constraint_matrix)
    return lineality, {
        "active_tolerance": active_tolerance,
        "linearized_active_constraint_rank": int(
            np.linalg.matrix_rank(constraint_matrix)
        ),
        "lineality_dimension": int(lineality.shape[1]),
        "spectra": spectra,
    }


def _conic_gain(basis: ConstraintBasis, lineality: np.ndarray) -> dict[str, Any]:
    information = _information(
        basis,
        np.asarray([1 if len(rows) else 0 for rows in basis.rows_by_frame]),
    )
    eigenvalues = np.linalg.eigvalsh(information)
    if lineality.shape[1]:
        restricted = lineality.T @ information @ lineality
        restricted_eigenvalues = np.linalg.eigvalsh(restricted)
        restricted_minimum = float(restricted_eigenvalues[0])
        restricted_condition = float(
            restricted_eigenvalues[-1] / restricted_eigenvalues[0]
        )
    else:
        restricted_minimum = None
        restricted_condition = None
    return {
        "information_minimum_eigenvalue_per_shot": float(eigenvalues[0]),
        "information_condition_per_shot": float(eigenvalues[-1] / eigenvalues[0]),
        "dqg_lineality_minimum_gain_per_shot": restricted_minimum,
        "dqg_lineality_condition": restricted_condition,
    }


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "system": "N2",
        "bond_length_angstrom": args.bond_length,
        "basis": args.basis,
        "active_space": [args.active_electrons, args.active_orbitals],
        "shot_seeds": list(args.shot_seeds),
        "budgets": list(args.budgets),
        "candidate_sizes": list(args.sizes),
        "d2_target": args.d2_target,
        "energy_target_meh": args.energy_target_meh,
        "allocation_minimum": args.allocation_minimum,
        "allocation_chunk": args.allocation_chunk,
        "nuclear_weight": args.nuclear_weight,
        "nuclear_weight_margin_grid": list(NUCLEAR_WEIGHT_GRID),
        "solver": args.solver,
        "solver_threads": args.solver_threads,
        "formal_reconstruction": "paper Eq. (11) nuclear norm with DQG",
        "formal_weighted_gls_used": False,
        "consistency_projection_used": False,
        "constraint_compression": (
            "pivoted basis of literal pair-occupation rows with full row rank; "
            "stochastic left-null dimension is zero"
        ),
        "primary_design_criterion": (
            "D2 A-optimal and Hamiltonian/ftPBE c-optimal effective shadow risks"
        ),
        "condition_role": "DQG restricted-gain guard and diagnostic, not primary score",
        "oracle_information": (
            "exact single-shot covariances, exact Hamiltonian gradient, exact ftPBE tangent"
        ),
        "oracle_is_deployable": False,
    }


def _save_design(
    args: argparse.Namespace,
    uniform: ConstraintBasis,
    joint: ConstraintBasis,
    joint_size_rows: Sequence[dict[str, Any]],
    allocation_rows: Sequence[dict[str, Any]],
    allocations: dict[int, np.ndarray],
    conic: dict[str, Any],
) -> None:
    _write_csv(args.output_dir / "joint_size_screen.csv", joint_size_rows)
    _write_csv(args.output_dir / "allocation_diagnostics.csv", allocation_rows)
    _atomic_npz(
        args.output_dir / "selected_design.npz",
        joint_rotations=np.asarray(joint.rotations),
        joint_global_rows=joint.global_rows,
        uniform_rotations=np.asarray(uniform.rotations),
        uniform_global_rows=uniform.global_rows,
        budgets=np.asarray(args.budgets),
        joint_counts=np.asarray([allocations[budget] for budget in args.budgets]),
    )
    _atomic_json(
        args.output_dir / "selected_design.json",
        {
            "selected_joint_size": len(joint.rotations),
            "joint_active_frames": [index + 1 for index in joint.active_frames],
            "joint_rows_per_frame": [len(rows) for rows in joint.rows_by_frame],
            "uniform_rows_per_frame": [len(rows) for rows in uniform.rows_by_frame],
            "joint_raw_condition": joint.raw_condition,
            "joint_fisher_condition": joint.fisher_condition,
            "uniform_raw_condition": uniform.raw_condition,
            "uniform_fisher_condition": uniform.fisher_condition,
            "conic_diagnostics": conic,
        },
    )


def _build_designs(
    args: argparse.Namespace,
    exact: Any,
    objective: Any,
) -> tuple[ConstraintBasis, ConstraintBasis, dict[int, np.ndarray], dict[str, Any]]:
    with np.load(
        args.manifold_design_dir / "design.npz", allow_pickle=False
    ) as archive:
        manifold = tuple(np.asarray(item) for item in archive["manifold_refined"])
    rows, cols = _symmetric_d2_variables(exact)
    d2_weights = np.where(rows == cols, 1.0, 2.0)
    hamiltonian = _hamiltonian_gradient(exact, rows, cols)
    ftpbe = _raw_ftpbe_gradient(
        objective,
        exact,
        exact.exact_d2,
        exact.exact_gamma,
        rows,
        cols,
    )
    lineality, active_diagnostics = _dqg_lineality_basis(exact, rows, cols)
    size_rows = []
    candidates = []
    for size in args.sizes:
        try:
            basis = _constraint_basis(
                f"manifold_refined_k{size}", manifold[:size], exact, rows, cols
            )
        except RuntimeError:
            size_rows.append(
                {
                    "size": size,
                    "full_rank": False,
                    "target_max_variance_ratio_at_max_budget": None,
                }
            )
            continue
        allocations, allocation_rows = _allocation_path(
            args, basis, d2_weights, hamiltonian, ftpbe
        )
        endpoint = allocation_rows[-1]
        gain = _conic_gain(basis, lineality)
        row = {
            "size": size,
            "full_rank": True,
            "active_frames": len(basis.active_frames),
            "raw_condition": basis.raw_condition,
            "fisher_condition": basis.fisher_condition,
            "target_max_variance_ratio_at_max_budget": endpoint[
                "target_max_variance_ratio"
            ],
            "d2_predicted_rms_at_max_budget": endpoint["d2_predicted_rms"],
            "hamiltonian_predicted_rms_meh_at_max_budget": endpoint[
                "hamiltonian_predicted_rms_meh"
            ],
            "ftpbe_predicted_rms_meh_at_max_budget": endpoint[
                "ftpbe_predicted_rms_meh"
            ],
            **gain,
        }
        size_rows.append(row)
        candidates.append((row, basis, allocations, allocation_rows))
        print(
            f"[design k={size}] score={row['target_max_variance_ratio_at_max_budget']:.3f}, "
            f"D2={row['d2_predicted_rms_at_max_budget']:.4f}, "
            f"H={row['hamiltonian_predicted_rms_meh_at_max_budget']:.3f}, "
            f"F={row['ftpbe_predicted_rms_meh_at_max_budget']:.3f} mEh",
            flush=True,
        )
    if not candidates:
        raise RuntimeError("No manifold prefix supplied a full-rank constraint basis.")
    selected_row, joint, allocations, allocation_rows = min(
        candidates,
        key=lambda item: (
            item[0]["target_max_variance_ratio_at_max_budget"],
            -float(item[0]["dqg_lineality_minimum_gain_per_shot"] or 0.0),
            item[0]["size"],
        ),
    )
    current_rotations, _ = mixed_orbital_rotations(
        exact.n_spatial_orbitals, 30, 4, 20260716, 271828
    )
    uniform = _constraint_basis(
        "uniform30_rank_basis", current_rotations, exact, rows, cols
    )
    conic = {
        "active_dqg": active_diagnostics,
        "selected_joint": _conic_gain(joint, lineality),
        "uniform30": _conic_gain(uniform, lineality),
        "selected_size_screen_row": selected_row,
    }
    _save_design(args, uniform, joint, size_rows, allocation_rows, allocations, conic)
    return (
        uniform,
        joint,
        allocations,
        {
            "rows": rows,
            "cols": cols,
            "d2_weights": d2_weights,
            "hamiltonian": hamiltonian,
            "ftpbe": ftpbe,
            "conic": conic,
        },
    )


def _sample_outcomes(
    oracle: AcquisitionOracle, seed: int, maximum_counts: np.ndarray
) -> tuple[np.ndarray, ...]:
    outcomes = []
    for index, count in enumerate(maximum_counts):
        if count < 1:
            outcomes.append(np.empty((0, len(oracle.reference.pairs)), dtype=np.uint8))
            continue
        probabilities = oracle._frame_probabilities(index)
        rng = np.random.default_rng([int(seed), index, 20260716])
        sampled = rng.choice(len(probabilities), size=int(count), p=probabilities)
        outcomes.append(np.asarray(oracle.indicators[sampled], dtype=np.uint8))
    return tuple(outcomes)


def _finite_shadows(
    basis: ConstraintBasis,
    outcomes: Sequence[np.ndarray],
    counts: np.ndarray,
) -> ShadowData:
    values = []
    for frame, local_rows in enumerate(basis.rows_by_frame):
        if not len(local_rows):
            continue
        values.extend(
            np.mean(outcomes[frame][: int(counts[frame])], axis=0)[local_rows]
        )
    values_array = np.asarray(values, dtype=float)
    active_rotations = tuple(basis.rotations[index] for index in basis.active_frames)
    return ShadowData(
        rotations=active_rotations,
        pair_vectors=basis.pair_vectors[basis.global_rows],
        design=basis.design[basis.global_rows],
        values=values_array,
        lower_bounds=values_array.copy(),
        upper_bounds=values_array.copy(),
        hits=np.full(len(values_array), -1, dtype=int),
        shots_per_basis=0,
        exact_values=np.full(len(values_array), np.nan),
        exact_constraints=False,
        occupations=None,
    )


def _score_result(
    args: argparse.Namespace,
    result: Any,
    exact: Any,
    objective: Any,
    exact_ftpbe: float,
) -> dict[str, Any]:
    d2 = np.asarray(result.d2, dtype=float)
    gamma = np.asarray(result.gamma, dtype=float)
    hamiltonian = float(
        np.sum(exact.one_body * gamma)
        + np.sum(exact.two_body * d2)
        + exact.nuclear_energy
    )
    ftpbe = float(objective.evaluate(d2, gamma, gradient=False).total_energy)
    d2_error = float(np.linalg.norm(d2 - exact.exact_d2))
    signed_h = 1000.0 * (hamiltonian - exact.exact_energy)
    signed_f = 1000.0 * (ftpbe - exact_ftpbe)
    energy_pass = bool(
        abs(signed_h) <= args.energy_target_meh
        and abs(signed_f) <= args.energy_target_meh
    )
    correction = np.asarray(result.corrected_d2) - d2
    correction_singular = np.linalg.svd(correction, compute_uv=False)
    nonzero = correction_singular[correction_singular > 1e-8]
    return {
        "d2_frobenius_error": d2_error,
        "signed_hamiltonian_error_meh": signed_h,
        "hamiltonian_error_meh": abs(signed_h),
        "signed_ftpbe_error_meh": signed_f,
        "ftpbe_error_meh": abs(signed_f),
        "shadow_error_trace": float(result.shadow_error_trace),
        "correction_rank_1e8": int(len(nonzero)),
        "correction_spectral_condition_1e8": (
            float(nonzero[0] / nonzero[-1]) if len(nonzero) else None
        ),
        "energy_target_pass": energy_pass,
        "three_target_pass": bool(energy_pass and d2_error <= args.d2_target),
    }


def _exact_basis_shadows(basis: ConstraintBasis, exact: Any) -> ShadowData:
    values = np.asarray(
        basis.design[basis.global_rows] @ exact.exact_d2.ravel(order="C")
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


def _nuclear_weight_margin_scan(
    args: argparse.Namespace,
    basis: ConstraintBasis,
    selection: Any,
    exact: Any,
    objective: Any,
    exact_ftpbe: float,
    baseline: Any,
) -> list[dict[str, Any]]:
    shadows = _exact_basis_shadows(basis, exact)
    rows = []
    for weight in NUCLEAR_WEIGHT_GRID:
        label = str(weight).replace(".", "p")
        checkpoint = args.output_dir / "weight_scan" / f"w_{label}.npz"
        metadata_path = checkpoint.with_suffix(".json")
        if checkpoint.is_file() and metadata_path.is_file() and not args.overwrite:
            row = json.loads(metadata_path.read_text(encoding="utf-8"))
            rows.append(row)
            continue
        started = time.perf_counter()
        result = solve_dqg_sdp(
            LeakGuardReference(selection),
            shadow_data=_blind_shadows(shadows),
            shadow_error_weight=weight,
            solver=args.solver,
            tolerance=args.solver_tolerance,
            max_iterations=args.max_iterations,
            solver_threads=args.solver_threads,
            positivity_conditions="DQG",
            symmetry_blocked_psd=True,
            selection_objective="energy",
            initial_d2=np.asarray(baseline.d2),
            initial_gamma=np.asarray(baseline.gamma),
            initial_corrected_d2=np.asarray(exact.exact_d2),
        )
        metrics = _score_result(args, result, exact, objective, exact_ftpbe)
        row = {
            "nuclear_weight": weight,
            "solver_seconds": time.perf_counter() - started,
            "status": str(result.status),
            "fit_status": str(result.fit_status),
            "exact_recovery_1e6": bool(
                metrics["d2_frobenius_error"] <= 1e-6
                and metrics["hamiltonian_error_meh"] <= 1e-3
                and metrics["ftpbe_error_meh"] <= 1e-3
            ),
            **metrics,
        }
        _atomic_npz(
            checkpoint,
            d2=np.asarray(result.d2),
            gamma=np.asarray(result.gamma),
            corrected_d2=np.asarray(result.corrected_d2),
        )
        _atomic_json(metadata_path, row)
        rows.append(row)
        print(
            f"[weight scan w={weight:g}] D2={metrics['d2_frobenius_error']:.3e}, "
            f"H={metrics['signed_hamiltonian_error_meh']:+.3e}, "
            f"F={metrics['signed_ftpbe_error_meh']:+.3e} mEh, "
            f"exact={row['exact_recovery_1e6']}",
            flush=True,
        )
    _write_csv(args.output_dir / "nuclear_weight_margin_scan.csv", rows)
    passing = [row["nuclear_weight"] for row in rows if row["exact_recovery_1e6"]]
    _atomic_json(
        args.output_dir / "nuclear_weight_margin_scan.json",
        {
            "minimum_tested_exact_recovery_weight": min(passing) if passing else None,
            "primary_finite_shot_weight": args.nuclear_weight,
            "rows": rows,
        },
    )
    return rows


def _solve_case(
    args: argparse.Namespace,
    output: Path,
    method: str,
    budget: int,
    basis: ConstraintBasis,
    outcomes: Sequence[np.ndarray],
    counts: np.ndarray,
    selection: Any,
    exact: Any,
    objective: Any,
    exact_ftpbe: float,
    warm_d2: np.ndarray,
    warm_gamma: np.ndarray,
    warm_corrected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint = output / "checkpoints" / f"{method}_{budget}.npz"
    metadata_path = checkpoint.with_suffix(".json")
    if checkpoint.is_file() and metadata_path.is_file() and not args.overwrite:
        with np.load(checkpoint, allow_pickle=False) as archive:
            d2 = np.asarray(archive["d2"])
            gamma = np.asarray(archive["gamma"])
            corrected = np.asarray(archive["corrected_d2"])
        return (
            d2,
            gamma,
            corrected,
            json.loads(metadata_path.read_text(encoding="utf-8")),
        )
    shadows = _finite_shadows(basis, outcomes, counts)
    started = time.perf_counter()
    result = solve_dqg_sdp(
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
    metrics = _score_result(args, result, exact, objective, exact_ftpbe)
    row = {
        "shot_seed": int(args.current_shot_seed),
        "allocation": method,
        "constraint_basis": basis.name,
        "total_shots": int(budget),
        "active_frames": len(basis.active_frames),
        "minimum_frame_shots": int(np.min(counts[list(basis.active_frames)])),
        "maximum_frame_shots": int(np.max(counts[list(basis.active_frames)])),
        "shot_counts": ",".join(map(str, counts)),
        "solver_seconds": time.perf_counter() - started,
        "status": str(result.status),
        "fit_status": str(result.fit_status),
        **metrics,
    }
    d2 = np.asarray(result.d2)
    gamma = np.asarray(result.gamma)
    corrected = np.asarray(result.corrected_d2)
    _atomic_npz(checkpoint, d2=d2, gamma=gamma, corrected_d2=corrected)
    _atomic_json(metadata_path, row)
    print(
        f"[{output.name} {method} B={budget}] D2={metrics['d2_frobenius_error']:.5f}, "
        f"H={metrics['signed_hamiltonian_error_meh']:+.3f}, "
        f"F={metrics['signed_ftpbe_error_meh']:+.3f} mEh, "
        f"pass={metrics['three_target_pass']}, {row['solver_seconds']:.1f}s",
        flush=True,
    )
    return d2, gamma, corrected, row


def _run_seed(
    args: argparse.Namespace,
    seed: int,
    configuration: dict[str, Any],
    uniform: ConstraintBasis,
    joint: ConstraintBasis,
    joint_allocations: dict[int, np.ndarray],
    selection: Any,
    exact: Any,
    objective: Any,
    exact_ftpbe: float,
    baseline: Any,
) -> list[dict[str, Any]]:
    output = args.output_dir / f"shot_seed_{seed}"
    analysis_path = output / "analysis.json"
    if analysis_path.is_file() and not args.overwrite:
        rows = json.loads(analysis_path.read_text(encoding="utf-8")).get("rows", [])
        expected = {
            (method, budget) for method in FORMAL_METHODS for budget in args.budgets
        }
        observed = {(row["allocation"], int(row["total_shots"])) for row in rows}
        if observed == expected:
            print(f"[shot_seed={seed}] complete checkpoint", flush=True)
            return list(rows)
    output.mkdir(parents=True, exist_ok=True)
    args.current_shot_seed = int(seed)
    uniform_counts = {
        budget: _equal_counts(len(uniform.rotations), budget, args.allocation_chunk)
        for budget in args.budgets
    }
    maximum_uniform = np.max(np.asarray(list(uniform_counts.values())), axis=0)
    maximum_joint = np.max(np.asarray(list(joint_allocations.values())), axis=0)
    uniform_oracle = AcquisitionOracle(
        exact, uniform.rotations, uniform.pair_vectors, 1, 0
    )
    joint_oracle = AcquisitionOracle(exact, joint.rotations, joint.pair_vectors, 1, 0)
    uniform_outcomes = _sample_outcomes(uniform_oracle, seed, maximum_uniform)
    joint_outcomes = _sample_outcomes(joint_oracle, seed, maximum_joint)
    cases = {
        "uniform30_rank_basis": (uniform, uniform_outcomes, uniform_counts),
        "joint_shadow_norm": (joint, joint_outcomes, joint_allocations),
    }
    rows = []
    for method in FORMAL_METHODS:
        basis, outcomes, allocations = cases[method]
        warm_d2 = np.asarray(baseline.d2)
        warm_gamma = np.asarray(baseline.gamma)
        warm_corrected = warm_d2.copy()
        for budget in args.budgets:
            warm_d2, warm_gamma, warm_corrected, row = _solve_case(
                args,
                output,
                method,
                budget,
                basis,
                outcomes,
                allocations[budget],
                selection,
                exact,
                objective,
                exact_ftpbe,
                warm_d2,
                warm_gamma,
                warm_corrected,
            )
            rows.append(row)
    _write_csv(output / "results.csv", rows)
    _atomic_json(
        analysis_path,
        {"configuration": {**configuration, "shot_seed": seed}, "rows": rows},
    )
    return rows


def _posthoc_rows(
    args: argparse.Namespace, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    for seed in args.shot_seeds:
        for budget in args.budgets:
            candidates = [
                row
                for row in rows
                if int(row["shot_seed"]) == seed
                and int(row["total_shots"]) == budget
                and row["allocation"] in FORMAL_METHODS
            ]
            winner = min(
                candidates,
                key=lambda row: max(
                    float(row["d2_frobenius_error"]) / args.d2_target,
                    float(row["hamiltonian_error_meh"]) / args.energy_target_meh,
                    float(row["ftpbe_error_meh"]) / args.energy_target_meh,
                ),
            )
            result.append(
                {
                    **winner,
                    "source_allocation": winner["allocation"],
                    "allocation": "posthoc_library_oracle",
                }
            )
    return result


def _aggregate_rows(
    args: argparse.Namespace, rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    summaries = []
    for method in REPORT_METHODS:
        for budget in args.budgets:
            selected = [
                row
                for row in rows
                if row["allocation"] == method and int(row["total_shots"]) == budget
            ]
            summary: dict[str, Any] = {
                "allocation": method,
                "total_shots": budget,
                "replicates": len(selected),
                "energy_target_hit_rate": float(
                    np.mean([row["energy_target_pass"] for row in selected])
                ),
                "three_target_hit_rate": float(
                    np.mean([row["three_target_pass"] for row in selected])
                ),
            }
            for metric, _ in METRICS:
                values = np.asarray([float(row[metric]) for row in selected])
                summary.update(
                    {
                        f"{metric}_mean": float(np.mean(values)),
                        f"{metric}_median": float(np.median(values)),
                        f"{metric}_q25": float(np.quantile(values, 0.25)),
                        f"{metric}_q75": float(np.quantile(values, 0.75)),
                        f"{metric}_min": float(np.min(values)),
                        f"{metric}_max": float(np.max(values)),
                    }
                )
            summaries.append(summary)
    return summaries


def _shots_to_target(
    args: argparse.Namespace, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    result = {}
    for method in REPORT_METHODS:
        energy = {}
        three = {}
        for seed in args.shot_seeds:
            selected = [
                row
                for row in rows
                if row["allocation"] == method and int(row["shot_seed"]) == seed
            ]
            energy_hits = sorted(
                int(row["total_shots"]) for row in selected if row["energy_target_pass"]
            )
            three_hits = sorted(
                int(row["total_shots"]) for row in selected if row["three_target_pass"]
            )
            energy[str(seed)] = energy_hits[0] if energy_hits else None
            three[str(seed)] = three_hits[0] if three_hits else None
        finite_energy = [value for value in energy.values() if value is not None]
        finite_three = [value for value in three.values() if value is not None]
        result[method] = {
            "first_energy_hit_by_seed": energy,
            "first_three_metric_hit_by_seed": three,
            "median_first_energy_hit": (
                float(np.median(finite_energy)) if finite_energy else None
            ),
            "median_first_three_metric_hit": (
                float(np.median(finite_three)) if finite_three else None
            ),
            "energy_hit_fraction": len(finite_energy) / len(args.shot_seeds),
            "three_metric_hit_fraction": len(finite_three) / len(args.shot_seeds),
        }
    return result


def _plot_errors(
    args: argparse.Namespace,
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
) -> None:
    colors = {
        "uniform30_rank_basis": "#1769AA",
        "joint_shadow_norm": "#C23B22",
        "posthoc_library_oracle": "#228B5A",
    }
    labels = {
        "uniform30_rank_basis": "uniform 30-frame rank basis",
        "joint_shadow_norm": "joint shadow-norm design",
        "posthoc_library_oracle": "posthoc best of two",
    }
    thresholds = {
        "d2_frobenius_error": args.d2_target,
        "hamiltonian_error_meh": args.energy_target_meh,
        "ftpbe_error_meh": args.energy_target_meh,
    }
    figure, axes = plt.subplots(3, 1, figsize=(9.4, 10.4), sharex=True)
    for axis, (metric, ylabel) in zip(axes, METRICS):
        for method in REPORT_METHODS:
            selected_summary = [row for row in summaries if row["allocation"] == method]
            budgets = np.asarray([row["total_shots"] for row in selected_summary])
            median = np.asarray([row[f"{metric}_median"] for row in selected_summary])
            q25 = np.asarray([row[f"{metric}_q25"] for row in selected_summary])
            q75 = np.asarray([row[f"{metric}_q75"] for row in selected_summary])
            axis.fill_between(budgets, q25, q75, color=colors[method], alpha=0.13)
            axis.plot(
                budgets,
                median,
                color=colors[method],
                marker="o",
                linewidth=1.8,
                label=labels[method],
            )
            for seed_index, seed in enumerate(args.shot_seeds):
                selected = sorted(
                    (
                        row
                        for row in rows
                        if row["allocation"] == method and int(row["shot_seed"]) == seed
                    ),
                    key=lambda row: int(row["total_shots"]),
                )
                offset = (seed_index - 2) * 500.0
                axis.scatter(
                    [int(row["total_shots"]) + offset for row in selected],
                    [float(row[metric]) for row in selected],
                    color=colors[method],
                    s=11,
                    alpha=0.5,
                    edgecolors="none",
                )
        axis.axhline(thresholds[metric], color="#444444", linestyle="--", linewidth=0.9)
        axis.set_ylabel(ylabel)
        axis.grid(True, color="#E1E1E1", linewidth=0.55)
    axes[-1].set_xlabel("Total shots")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    figure.suptitle(
        "N2 constrained shadows: joint frame/shot oracle",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(args.output_dir / "joint_oracle_errors.png", dpi=220)
    plt.close(figure)


def _plot_design(args: argparse.Namespace) -> None:
    rows = []
    with (args.output_dir / "joint_size_screen.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows.extend(csv.DictReader(handle))
    valid = [row for row in rows if row["full_rank"] == "True"]
    sizes = np.asarray([int(row["size"]) for row in valid])
    figure, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True)
    axes[0].plot(
        sizes,
        [float(row["target_max_variance_ratio_at_max_budget"]) for row in valid],
        color="#C23B22",
        marker="o",
    )
    axes[0].set_ylabel("Maximum normalized shadow variance")
    axes[1].plot(
        sizes,
        [float(row["dqg_lineality_minimum_gain_per_shot"]) for row in valid],
        color="#1769AA",
        marker="o",
    )
    axes[1].set_ylabel("DQG restricted gain per shot")
    axes[1].set_xlabel("Selected manifold frames")
    for axis in axes:
        axis.grid(True, color="#E1E1E1", linewidth=0.55)
    figure.tight_layout()
    figure.savefig(args.output_dir / "joint_design_diagnostics.png", dpi=220)
    plt.close(figure)


def _plot_allocations(args: argparse.Namespace) -> None:
    with np.load(
        args.output_dir / "selected_design.npz", allow_pickle=False
    ) as archive:
        budgets = np.asarray(archive["budgets"], dtype=int)
        counts = np.asarray(archive["joint_counts"], dtype=float)
    fractions = counts / budgets[:, None]
    figure, axis = plt.subplots(figsize=(11.0, 4.8))
    image = axis.imshow(
        fractions,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
    )
    axis.set_yticks(
        np.arange(len(budgets)), [f"{budget // 1000}k" for budget in budgets]
    )
    axis.set_xticks(np.arange(counts.shape[1]), np.arange(1, counts.shape[1] + 1))
    axis.set_xlabel("Selected manifold frame")
    axis.set_ylabel("Total shot budget")
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Fraction of total shots")
    figure.tight_layout()
    figure.savefig(args.output_dir / "joint_shot_allocations.png", dpi=220)
    plt.close(figure)


def _plot_hit_rates(
    args: argparse.Namespace, summaries: Sequence[dict[str, Any]]
) -> None:
    colors = {
        "uniform30_rank_basis": "#1769AA",
        "joint_shadow_norm": "#C23B22",
        "posthoc_library_oracle": "#228B5A",
    }
    figure, axes = plt.subplots(2, 1, figsize=(8.8, 6.8), sharex=True)
    for method in REPORT_METHODS:
        selected = [row for row in summaries if row["allocation"] == method]
        budgets = [row["total_shots"] for row in selected]
        axes[0].plot(
            budgets,
            [row["energy_target_hit_rate"] for row in selected],
            color=colors[method],
            marker="o",
            linewidth=1.7,
            label=method.replace("_", " "),
        )
        axes[1].plot(
            budgets,
            [row["three_target_hit_rate"] for row in selected],
            color=colors[method],
            marker="o",
            linewidth=1.7,
        )
    axes[0].set_ylabel("Energy-target hit rate")
    axes[1].set_ylabel("Three-metric hit rate")
    axes[1].set_xlabel("Total shots")
    for axis in axes:
        axis.set_ylim(-0.04, 1.04)
        axis.grid(True, color="#E1E1E1", linewidth=0.55)
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "joint_target_hit_rates.png", dpi=220)
    plt.close(figure)


def _plot_weight_margin(args: argparse.Namespace) -> None:
    payload = json.loads(
        (args.output_dir / "nuclear_weight_margin_scan.json").read_text(
            encoding="utf-8"
        )
    )
    rows = payload["rows"]
    weights = [float(row["nuclear_weight"]) for row in rows]
    metrics = (
        ("d2_frobenius_error", "2-RDM Frobenius error"),
        ("hamiltonian_error_meh", "Hamiltonian error (mEh)"),
        ("ftpbe_error_meh", "ftPBE error (mEh)"),
    )
    figure, axes = plt.subplots(3, 1, figsize=(8.2, 8.8), sharex=True)
    for axis, (metric, label) in zip(axes, metrics):
        axis.plot(
            weights,
            [float(row[metric]) for row in rows],
            color="#C23B22",
            marker="o",
            linewidth=1.7,
        )
        axis.set_ylabel(label)
        axis.grid(True, color="#E1E1E1", linewidth=0.55)
    axes[-1].set_xlabel("Nuclear error weight w")
    figure.tight_layout()
    figure.savefig(args.output_dir / "nuclear_weight_margin.png", dpi=220)
    plt.close(figure)


def _report(
    args: argparse.Namespace,
    summaries: Sequence[dict[str, Any]],
    shots_to_target: dict[str, Any],
) -> str:
    design = json.loads(
        (args.output_dir / "selected_design.json").read_text(encoding="utf-8")
    )
    lines = [
        "# N2 joint shadow-norm/conic oracle",
        "",
        "The primary design score is the effective shadow variance for D2, "
        "Hamiltonian, and the exact ftPBE tangent. Ordinary condition numbers are "
        "diagnostics only. A DQG active-lineality restricted gain is used as a conic "
        "guard. Literal row-basis compression removes stochastic left-null relations, "
        "so the formal Eq. (11) SDPs use independent finite-shot means directly.",
        "",
        f"Selected manifold prefix: `{design['selected_joint_size']}` frames; "
        f"active after row pivoting: `{len(design['joint_active_frames'])}`.",
        "",
        "## Fixed-budget medians",
        "",
        "| allocation | shots | D2 | H (mEh) | ftPBE (mEh) | energy hit | all-three hit |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['allocation']} | {row['total_shots']} | "
            f"{row['d2_frobenius_error_median']:.6f} | "
            f"{row['hamiltonian_error_meh_median']:.3f} | "
            f"{row['ftpbe_error_meh_median']:.3f} | "
            f"{row['energy_target_hit_rate']:.0%} | "
            f"{row['three_target_hit_rate']:.0%} |"
        )
    lines.extend(["", "## First-hit budgets", ""])
    for method in REPORT_METHODS:
        item = shots_to_target[method]
        lines.append(
            f"- `{method}`: median energy first hit "
            f"{item['median_first_energy_hit']}; median all-three first hit "
            f"{item['median_first_three_metric_hit']}."
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Frame selection, exact single-shot covariances, exact target gradients, "
            "and posthoc choice use oracle information. The formal reconstruction uses "
            "MOSEK, DQG, nuclear weight w=1, no weighted GLS, and no consistency "
            "projection. The reported DQG lineality gain is a local conic proxy, not a "
            "complete RIP or dual-certificate proof for the composite SDP.",
            "",
        ]
    )
    return "\n".join(lines)


def _aggregate(args: argparse.Namespace, configuration: dict[str, Any]) -> None:
    rows = []
    for seed in args.shot_seeds:
        path = args.output_dir / f"shot_seed_{seed}" / "analysis.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed seed result: {path}")
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    rows.extend(_posthoc_rows(args, rows))
    summaries = _aggregate_rows(args, rows)
    shots_to_target = _shots_to_target(args, rows)
    _write_csv(args.output_dir / "all_results.csv", rows)
    _write_csv(args.output_dir / "fixed_budget_summary.csv", summaries)
    _atomic_json(
        args.output_dir / "analysis.json",
        {
            "configuration": configuration,
            "shots_to_target": shots_to_target,
            "fixed_budget_summary": summaries,
            "rows": rows,
        },
    )
    _plot_errors(args, rows, summaries)
    _plot_design(args)
    _plot_allocations(args)
    _plot_hit_rates(args, summaries)
    _plot_weight_margin(args)
    (args.output_dir / "JOINT_SHADOW_CONIC_FINDINGS.md").write_text(
        _report(args, summaries, shots_to_target), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    args.output_dir = args.output_dir.resolve()
    args.manifold_design_dir = args.manifold_design_dir.resolve()
    args.solver = args.solver.upper()
    if args.solver != "MOSEK":
        raise ValueError("This formal diagnostic is frozen to MOSEK.")
    if any(budget % args.allocation_chunk for budget in args.budgets):
        raise ValueError("Budgets must align with allocation-chunk.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = _configuration(args)
    _atomic_json(args.output_dir / "configuration.json", configuration)
    selection, exact, objective, exact_ftpbe = _build_references(args)
    if args.aggregate_only:
        _aggregate(args, configuration)
        return
    print("Building shadow-norm and DQG conic design diagnostics...", flush=True)
    uniform, joint, allocations, _ = _build_designs(args, exact, objective)
    if args.design_only:
        print(f"Design results: {args.output_dir}", flush=True)
        return
    print("Solving the shared shadow-free DQG baseline with MOSEK...", flush=True)
    baseline = solve_dqg_sdp(
        LeakGuardReference(selection),
        solver=args.solver,
        tolerance=args.solver_tolerance,
        max_iterations=args.max_iterations,
        solver_threads=args.solver_threads,
        positivity_conditions="DQG",
        symmetry_blocked_psd=True,
    )
    _nuclear_weight_margin_scan(
        args,
        joint,
        selection,
        exact,
        objective,
        exact_ftpbe,
        baseline,
    )
    for seed in args.shot_seeds:
        _run_seed(
            args,
            seed,
            configuration,
            uniform,
            joint,
            allocations,
            selection,
            exact,
            objective,
            exact_ftpbe,
            baseline,
        )
    _aggregate(args, configuration)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
