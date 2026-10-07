#!/usr/bin/env python3
"""Run and plot five N2 equilibrium random-shadow prefix trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/n2-equilibrium-random-seed-trajectories")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ROOT / "code"
VENDOR = CODE / "vendor"
for path in (CODE, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constrained_shadow import (  # noqa: E402
    build_n2_reference,
    random_orthogonal_rotations,
    shadow_pair_vectors,
)
from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    build_n2_selection_reference,
)
from mcpdft_selector import FtPBEEnergyObjective  # noqa: E402
from run_c2_ftpbe_shot_reallocation import (  # noqa: E402
    AcquisitionOracle,
    _records_to_shadows,
)
from run_c2_m50_probe_rule_validation import _solve_blind  # noqa: E402
from run_hybrid_complex_shadow_ablation import _random_unitaries  # noqa: E402
from run_safe_mcpdft_derandomization import (  # noqa: E402
    RECONSTRUCTION_ENERGY,
    _initial_result,
)
from run_sweep import _atomic_json  # noqa: E402


DEFAULT_SEEDS = (20260811, 20260812, 20260813, 20260814, 20260815)
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "sweeps" / "n2_equilibrium_random_m1_30_five_seeds",
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--bond-length", type=float, default=1.10)
    parser.add_argument("--frame-count", type=int, default=30)
    parser.add_argument("--shots-per-frame", type=int, default=10_000)
    parser.add_argument("--complex-cadence", type=int, default=4)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--solver-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args(argv)


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _solver_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
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


def _derived_seeds(seed: int) -> tuple[int, int, int]:
    children = np.random.SeedSequence(seed).spawn(3)
    return tuple(int(child.generate_state(1)[0]) for child in children)  # type: ignore[return-value]


def _candidate_design(
    exact: Any,
    frame_count: int,
    cadence: int,
    real_seed: int,
    complex_seed: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    rotations = list(
        random_orthogonal_rotations(
            exact.n_spatial_orbitals, frame_count, real_seed
        )
    )
    is_complex = np.zeros(frame_count, dtype=bool)
    indices = tuple(range(cadence - 1, frame_count, cadence))
    complex_rotations = _random_unitaries(
        exact.n_spatial_orbitals, len(indices), complex_seed
    )
    for index, rotation in zip(indices, complex_rotations):
        rotations[index] = np.asarray(rotation)
        is_complex[index] = True
    rotation_tuple = tuple(np.asarray(rotation) for rotation in rotations)
    pair_vectors = shadow_pair_vectors(
        rotation_tuple, exact.n_spatial_orbitals, exact.pairs
    )
    return rotation_tuple, pair_vectors, is_complex


def _build_references(args: argparse.Namespace) -> tuple[Any, Any, Any, float]:
    selection_base = build_n2_selection_reference(
        args.bond_length,
        basis="cc-pvdz",
        active_electrons=6,
        active_orbitals=6,
    )
    exact = build_n2_reference(
        args.bond_length,
        basis="cc-pvdz",
        active_electrons=6,
        active_orbitals=6,
    )
    if not (
        np.allclose(selection_base.one_body, exact.one_body, atol=2e-9)
        and np.allclose(selection_base.two_body, exact.two_body, atol=2e-9)
        and np.isclose(selection_base.nuclear_energy, exact.nuclear_energy, atol=2e-9)
    ):
        raise RuntimeError("Selection and exact N2 Hamiltonians do not match.")
    selection = LeakGuardReference(selection_base)
    objective = FtPBEEnergyObjective(selection, grid_level=args.grid_level)
    exact_ftpbe = float(
        objective.evaluate(exact.exact_d2, exact.exact_gamma, gradient=False).total_energy
    )
    return selection, exact, objective, exact_ftpbe


def _run_seed(args: argparse.Namespace, seed: int) -> None:
    seed_dir = args.output_dir / f"seed_{seed}"
    analysis_path = seed_dir / "analysis.json"
    if analysis_path.is_file() and not args.overwrite:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        if len(payload.get("rows", ())) == args.frame_count:
            print(f"[seed={seed}] complete checkpoint", flush=True)
            return
    seed_dir.mkdir(parents=True, exist_ok=True)
    selection, exact, objective, exact_ftpbe = _build_references(args)
    real_seed, complex_seed, sampling_seed = _derived_seeds(seed)
    rotations, pair_vectors, is_complex = _candidate_design(
        exact,
        args.frame_count,
        args.complex_cadence,
        real_seed,
        complex_seed,
    )
    oracle = AcquisitionOracle(
        exact, rotations, pair_vectors, args.shots_per_frame, sampling_seed
    )
    print(f"[seed={seed}] sampling {args.frame_count} frames", flush=True)
    records = [oracle.sample(index, 0) for index in range(args.frame_count)]
    solver_args = _solver_args(args)
    print(f"[seed={seed}] solving no-data DQG", flush=True)
    initial = _initial_result(solver_args, selection)
    warm_d2 = np.asarray(initial.d2)
    warm_gamma = np.asarray(initial.gamma)
    rows: list[dict[str, Any]] = []
    for count in range(1, args.frame_count + 1):
        acquired = _records_to_shadows(records[:count], args.shots_per_frame)
        d2, gamma, metadata = _solve_blind(
            f"m_{count:02d}",
            acquired,
            solver_args,
            selection,
            warm_d2,
            warm_gamma,
            seed_dir,
            args.overwrite,
        )
        warm_d2 = np.asarray(d2)
        warm_gamma = np.asarray(gamma)
        hamiltonian = float(
            np.sum(exact.one_body * warm_gamma)
            + np.sum(exact.two_body * warm_d2)
            + exact.nuclear_energy
        )
        ftpbe = float(
            objective.evaluate(warm_d2, warm_gamma, gradient=False).total_energy
        )
        row = {
            "trajectory_seed": seed,
            "m": count,
            "total_shots": count * args.shots_per_frame,
            "complex_frames": int(np.count_nonzero(is_complex[:count])),
            "d2_frobenius_error": float(np.linalg.norm(warm_d2 - exact.exact_d2)),
            "hamiltonian_error_meh": 1000.0
            * float(abs(hamiltonian - exact.exact_energy)),
            "ftpbe_error_meh": 1000.0 * float(abs(ftpbe - exact_ftpbe)),
            "solver_seconds": float(metadata["solver_seconds"]),
            "status": str(metadata["status"]),
            "fit_status": str(metadata["fit_status"]),
        }
        rows.append(row)
        print(
            f"[seed={seed} m={count:02d}] D2={row['d2_frobenius_error']:.6f}, "
            f"H={row['hamiltonian_error_meh']:.3f}, "
            f"ftPBE={row['ftpbe_error_meh']:.3f} mEh",
            flush=True,
        )
    configuration = {
        "system": "N2",
        "bond_length_angstrom": args.bond_length,
        "basis": "cc-pvdz",
        "active_space": [6, 6],
        "trajectory_seed": seed,
        "derived_real_frame_seed": real_seed,
        "derived_complex_frame_seed": complex_seed,
        "derived_sampling_seed": sampling_seed,
        "frame_count": args.frame_count,
        "shots_per_frame": args.shots_per_frame,
        "complex_cadence": args.complex_cadence,
        "complex_indices_one_based": [
            int(index) + 1 for index in np.flatnonzero(is_complex)
        ],
        "solver": args.solver.upper(),
        "solver_tolerance": args.solver_tolerance,
        "solver_threads": args.solver_threads,
        "positivity": "DQG",
        "grid_level": args.grid_level,
        "seed_scope": "real frames, complex frames, and finite-shot bitstrings",
        "exact_use": "simulated acquisition and posthoc scoring only",
    }
    payload = {"configuration": configuration, "rows": rows}
    _atomic_csv(seed_dir / "trajectory.csv", rows)
    _atomic_json(seed_dir / "analysis.json", payload)


def _plot_metric(
    axis: Any,
    rows_by_seed: dict[int, list[dict[str, Any]]],
    key: str,
    ylabel: str,
    colors: dict[int, str],
    show_legend: bool,
) -> None:
    for seed, rows in rows_by_seed.items():
        axis.plot(
            [int(row["m"]) for row in rows],
            [float(row[key]) for row in rows],
            color=colors[seed],
            marker="o",
            markersize=2.8,
            linewidth=1.35,
            label=f"seed {seed}",
        )
    axis.set_yscale("log")
    axis.set_xlim(1, 30)
    axis.set_xticks((1, 5, 10, 15, 20, 25, 30))
    axis.set_xlabel("Number of shadow frames, m")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="major", color="#D9D9D9", linewidth=0.7)
    axis.grid(True, which="minor", color="#EEEEEE", linewidth=0.45)
    if key.endswith("_error_meh"):
        axis.axhline(1.6, color="#4D4D4D", linestyle="--", linewidth=1.0)
    if show_legend:
        axis.legend(frameon=False, fontsize=8, ncol=1)


def _aggregate(args: argparse.Namespace, seeds: Sequence[int]) -> None:
    rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in seeds:
        path = args.output_dir / f"seed_{seed}" / "analysis.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed trajectory: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = list(payload["rows"])
        if [int(row["m"]) for row in rows] != list(
            range(1, args.frame_count + 1)
        ):
            raise RuntimeError(f"Seed {seed} does not contain m=1..{args.frame_count}.")
        rows_by_seed[int(seed)] = rows
    all_rows = [row for rows in rows_by_seed.values() for row in rows]
    _atomic_csv(args.output_dir / "trajectories.csv", all_rows)
    colors = {seed: COLORS[index] for index, seed in enumerate(seeds)}
    metrics = (
        ("d2_frobenius_error", "D2 Frobenius error", "d2_error"),
        ("hamiltonian_error_meh", "Hamiltonian error (mEh)", "hamiltonian_error"),
        ("ftpbe_error_meh", "ftPBE error (mEh)", "ftpbe_error"),
    )
    for key, ylabel, stem in metrics:
        figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        _plot_metric(axis, rows_by_seed, key, ylabel, colors, True)
        axis.set_title("N2 at R = 1.10 Angstrom")
        figure.savefig(args.output_dir / f"n2_equilibrium_{stem}.png", dpi=220)
        plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    figure.subplots_adjust(
        left=0.055, right=0.99, bottom=0.16, top=0.78, wspace=0.25
    )
    for axis, (key, ylabel, _) in zip(axes, metrics):
        _plot_metric(axis, rows_by_seed, key, ylabel, colors, False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=len(seeds),
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "N2 random-shadow prefix errors at R = 1.10 Angstrom "
        "(10,000 shots/frame; every 4th frame complex)",
        fontsize=11,
        y=0.98,
    )
    figure.savefig(args.output_dir / "n2_equilibrium_three_error_trajectories.png", dpi=220)
    plt.close(figure)
    endpoint = {
        str(seed): next(row for row in rows if int(row["m"]) == args.frame_count)
        for seed, rows in rows_by_seed.items()
    }
    summary = {
        "configuration": {
            "system": "N2",
            "bond_length_angstrom": args.bond_length,
            "basis": "cc-pvdz",
            "active_space": [6, 6],
            "trajectory_seeds": list(seeds),
            "frame_count": args.frame_count,
            "shots_per_frame": args.shots_per_frame,
            "complex_cadence": args.complex_cadence,
            "seed_scope": "real frames, complex frames, and finite-shot bitstrings",
        },
        "m30_by_seed": endpoint,
        "color_by_seed": {str(seed): colors[seed] for seed in seeds},
    }
    _atomic_json(args.output_dir / "summary.json", summary)
    print(f"Wrote five-seed plots to {args.output_dir}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    args.output_dir = args.output_dir.resolve()
    seeds = tuple(args.seed) if args.seed else DEFAULT_SEEDS
    if len(set(seeds)) != len(seeds):
        raise ValueError("Trajectory seeds must be unique.")
    if args.frame_count != 30:
        raise ValueError("This comparison is fixed to m=1..30.")
    if args.shots_per_frame < 1 or args.complex_cadence < 2:
        raise ValueError("Shots must be positive and complex cadence at least two.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        for seed in seeds:
            _run_seed(args, int(seed))
    if not args.no_aggregate:
        aggregate_seeds = DEFAULT_SEEDS if args.aggregate_only else seeds
        _aggregate(args, aggregate_seeds)


if __name__ == "__main__":
    main()
