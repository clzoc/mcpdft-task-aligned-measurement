#!/usr/bin/env python3
"""Run the fair-cost random-frame pilot joint protocol for C2 CAS(8,8)."""

from __future__ import annotations

import argparse
import os
import sys
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/c2-random-pilot-joint-design")

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
    exterior_square,
    quadratic_design,
    random_orthogonal_rotations,
    solve_dqg_sdp,
)
from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    build_c2_reference,
    build_c2_selection_reference,
)
from mcpdft_selector import FtPBEEnergyObjective, _symmetric_d2_variables  # noqa: E402
from run_n2_joint_shadow_norm_conic_oracle import (  # noqa: E402
    _atomic_json,
    _atomic_npz,
    _parse_integers,
)
from run_n2_paper_nuclear_hybrid_trajectories import (  # noqa: E402
    DEFAULT_NOISE_SEEDS,
    _random_unitaries,
)
import run_n2_random_pilot_joint_design as joint  # noqa: E402
from run_n2_random_adaptive_allocation import (  # noqa: E402
    _exact_mean_shadows,
    _random_mixed_frames,
    _solve_eq11,
)
from run_n2_ftpbe_shot_cost_oracle import _frame_block  # noqa: E402


DEFAULT_OUTPUT = ROOT / "sweeps" / "c2_random_pilot_joint_design"
DEFAULT_BUDGETS = (30_000, 60_000, 120_000, 180_000, 240_000, 300_000)
DEFAULT_SIZES = (8, 10, 12, 14, 18, 24, 30)


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
    parser.add_argument("--bond-length", type=float, default=1.25)
    parser.add_argument("--basis", default="cc-pvtz")
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
    args = parser.parse_args(argv)
    args.system = "C2"
    args.active_electrons = 8
    args.active_orbitals = 8
    return args


def _observable_coordinate_chart(
    blocks: Sequence[np.ndarray],
    variable_rows: np.ndarray,
    variable_cols: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray, np.ndarray, int]:
    """Choose independent observable variable columns for the C2 risk model."""

    stacked = np.vstack(blocks)
    _, triangular, pivots = qr(stacked, mode="economic", pivoting=True)
    tolerance = abs(triangular[0, 0]) * max(stacked.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(np.abs(np.diag(triangular)) > tolerance))
    columns = np.sort(np.asarray(pivots[:rank], dtype=int))
    reduced = tuple(np.asarray(block)[:, columns] for block in blocks)
    return (
        reduced,
        np.asarray(variable_rows)[columns],
        np.asarray(variable_cols)[columns],
        columns,
        rank,
    )


def _c2_random_frames(
    n_orbitals: int,
    count: int,
    complex_fraction: float,
    real_seed: int,
    complex_seed: int,
    placement_seed: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Draw same-spin frames plus one independent alpha/beta complex frame."""

    if count < 3:
        raise ValueError("C2 needs at least three frames for the mixed frame family.")
    target_complex = min(max(int(round(count * complex_fraction)), 1), count - 1)
    base_count = count - 1
    base_complex = target_complex - 1
    base_fraction = base_complex / base_count
    if base_complex == 0:
        base = random_orthogonal_rotations(n_orbitals, base_count, real_seed)
        base_is_complex = np.zeros(base_count, dtype=bool)
    else:
        base, base_is_complex = _random_mixed_frames(
            n_orbitals,
            base_count,
            base_fraction,
            real_seed,
            complex_seed,
            placement_seed + 1,
        )
    alpha = _random_unitaries(n_orbitals, 1, complex_seed + 101)[0]
    beta = _random_unitaries(n_orbitals, 1, complex_seed + 102)[0]
    insertion = int(np.random.default_rng(placement_seed).integers(0, count))
    rotations = []
    is_complex = []
    is_spin_asymmetric = []
    base_index = 0
    for index in range(count):
        if index == insertion:
            rotations.append(np.stack((alpha, beta)))
            is_complex.append(True)
            is_spin_asymmetric.append(True)
        else:
            frame = np.asarray(base[base_index])
            rotations.append(np.stack((frame, frame)))
            is_complex.append(bool(base_is_complex[base_index]))
            is_spin_asymmetric.append(False)
            base_index += 1
    return (
        tuple(rotations),
        np.asarray(is_complex, dtype=bool),
        np.asarray(is_spin_asymmetric, dtype=bool),
    )


def _c2_random_design(
    reference: Any, rotations: Sequence[np.ndarray]
) -> tuple[np.ndarray, Any, tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Build pair vectors for independent alpha/beta orbital rotations."""

    rows, cols = _symmetric_d2_variables(reference)
    vectors = []
    blocks = []
    n_spatial = reference.n_spatial_orbitals
    zero = np.zeros((n_spatial, n_spatial), dtype=complex)
    for rotation in rotations:
        alpha, beta = np.asarray(rotation)
        spin_rotation = np.block([[alpha, zero], [zero, beta]])
        frame_vectors = exterior_square(spin_rotation, reference.pairs).T
        vectors.append(frame_vectors)
        blocks.append(_frame_block(frame_vectors, rows, cols))
    pair_vectors = np.vstack(vectors)
    return pair_vectors, quadratic_design(pair_vectors), tuple(blocks), rows, cols


def _configuration(
    args: argparse.Namespace,
    is_complex: np.ndarray,
    full_dimension: int,
    observable_rank: int,
    observable_columns: np.ndarray,
    is_spin_asymmetric: np.ndarray,
) -> dict[str, Any]:
    configuration = joint._configuration(args, is_complex)
    configuration.update(
        {
            "allocation_coordinate_model": (
                "pivoted independent variable chart of the random-frame observable "
                "row space; formal Eq. (11) retains the full D2"
            ),
            "full_symmetric_variable_dimension": int(full_dimension),
            "observable_variable_rank": int(observable_rank),
            "structural_null_dimension": int(full_dimension - observable_rank),
            "observable_variable_columns_zero_based": observable_columns.tolist(),
            "frozen_mo_for_acquisition": True,
            "spin_asymmetric_frames": int(np.count_nonzero(is_spin_asymmetric)),
            "spin_asymmetric_indices_one_based": (
                np.flatnonzero(is_spin_asymmetric).astype(int) + 1
            ).tolist(),
            "spin_asymmetric_definition": (
                "independent complex Haar U_alpha and U_beta; no alpha/beta mixing"
            ),
        }
    )
    return configuration


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    joint._validate_args(args)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        joint._aggregate(args)
        return

    print("Building selection-only C2 CAS(8,8) data and random frames...", flush=True)
    selection = build_c2_selection_reference(args.bond_length, basis=args.basis)
    frozen_mo = np.asarray(selection.mean_field.mo_coeff, dtype=float)
    objective = FtPBEEnergyObjective(selection, grid_level=args.grid_level)
    rotations, is_complex, is_spin_asymmetric = _c2_random_frames(
        selection.n_spatial_orbitals,
        args.frame_count,
        args.complex_fraction,
        args.real_frame_seed,
        args.complex_frame_seed,
        args.placement_seed,
    )
    (
        pair_vectors,
        _,
        full_blocks,
        full_variable_rows,
        full_variable_cols,
    ) = _c2_random_design(selection, rotations)
    (
        blocks,
        variable_rows,
        variable_cols,
        observable_columns,
        observable_rank,
    ) = _observable_coordinate_chart(
        full_blocks, full_variable_rows, full_variable_cols
    )
    structural_covariances = tuple(
        np.eye(len(selection.pairs), dtype=float) for _ in rotations
    )
    structural = joint._constraint_basis_from_covariances(
        "uniform30",
        tuple(range(args.frame_count)),
        rotations,
        pair_vectors,
        blocks,
        structural_covariances,
        len(selection.pairs),
        observable_rank,
    )
    configuration = _configuration(
        args,
        is_complex,
        len(full_variable_rows),
        observable_rank,
        observable_columns,
        is_spin_asymmetric,
    )
    _atomic_json(args.output_dir / "configuration.json", configuration)
    _atomic_npz(
        args.output_dir / "random_frame_pool.npz",
        rotations=np.asarray(rotations),
        is_complex=is_complex,
        pair_vectors=pair_vectors,
        observable_variable_columns=observable_columns,
        is_spin_asymmetric=is_spin_asymmetric,
    )
    print(
        f"C2 observable chart: {observable_rank}/{len(full_variable_rows)} variables.",
        flush=True,
    )
    print("Solving shadow-free C2 DQG initialization with MOSEK...", flush=True)
    baseline = solve_dqg_sdp(
        LeakGuardReference(selection),
        solver=args.solver,
        tolerance=args.solver_tolerance,
        max_iterations=args.max_iterations,
        solver_threads=args.solver_threads,
        positivity_conditions="DQG",
        symmetry_blocked_psd=True,
    )
    exact = build_c2_reference(
        args.bond_length,
        basis=args.basis,
        frozen_mo_coeff=frozen_mo,
    )
    np.testing.assert_allclose(selection.one_body, exact.one_body, atol=2e-9)
    np.testing.assert_allclose(selection.two_body, exact.two_body, atol=2e-9)
    reachable = _solve_eq11(
        args,
        selection,
        _exact_mean_shadows(structural.basis, exact.exact_d2),
        np.asarray(baseline.d2),
        np.asarray(baseline.gamma),
        np.asarray(baseline.d2),
    )
    for seed in args.shot_seeds:
        joint._run_seed(
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
    joint._aggregate(args)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
