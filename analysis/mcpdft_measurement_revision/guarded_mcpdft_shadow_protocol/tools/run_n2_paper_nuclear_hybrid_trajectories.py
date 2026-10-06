#!/usr/bin/env python3
"""Run the paper nuclear-norm constrained-shadow model on mixed N2 frames.

This runner follows Eq. (11) of arXiv:2511.09717 rather than the repository's
raw-frequency weighted-fit extensions. One globally Gaussian-perturbed 2-RDM
is generated per noise seed. Every real/complex shadow in that seed is an
exact projection of the same noisy matrix, and every m-point result uses the
nested prefix of a single frozen frame sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/n2-paper-nuclear-hybrid")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CODE = ROOT / "code"
VENDOR = CODE / "vendor"
for path in (CODE, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constrained_shadow import (  # noqa: E402
    ShadowData,
    dqg_matrices,
    exact_shadows,
    gaussian_noisy_shadows,
    random_orthogonal_rotations,
    solve_dqg_sdp,
    subset_shadows,
)
from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    build_n2_selection_reference,
    shadow_design_blocks,
)
from mcpdft_selector import FtPBEEnergyObjective, _symmetric_d2_variables  # noqa: E402


DEFAULT_NOISE_SEEDS = tuple(range(20260811, 20260816))
DEFAULT_OUTPUT = ROOT / "sweeps" / "n2_paper_nuclear_hybrid_m1_30_five_seeds"
AGGREGATE_METRICS = (
    "d2_frobenius_error",
    "hamiltonian_error_meh",
    "ftpbe_error_meh",
    "signed_hamiltonian_error_meh",
    "signed_ftpbe_error_meh",
    "shadow_error_trace",
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bond-length", type=float, default=1.10)
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--active-electrons", type=int, default=6)
    parser.add_argument("--active-orbitals", type=int, default=6)
    parser.add_argument("--frame-count", type=int, default=30)
    parser.add_argument("--shots-per-frame", type=int, default=1000)
    parser.add_argument("--complex-cadence", type=int, default=4)
    parser.add_argument("--real-frame-seed", type=int, default=20260716)
    parser.add_argument("--complex-frame-seed", type=int, default=271828)
    parser.add_argument("--noise-seed", action="append", type=int)
    parser.add_argument("--nuclear-weight", type=float, default=1.0)
    parser.add_argument(
        "--probability-model",
        choices=("natural-pair-geometric", "max-variance"),
        default="natural-pair-geometric",
    )
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--solver-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="Run only the noiseless shadow trajectory, then aggregate all results.",
    )
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


def _random_unitaries(n_orbitals: int, count: int, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    rotations = []
    for _ in range(count):
        matrix = (
            rng.normal(size=(n_orbitals, n_orbitals))
            + 1j * rng.normal(size=(n_orbitals, n_orbitals))
        ) / math.sqrt(2.0)
        q_matrix, r_matrix = np.linalg.qr(matrix)
        diagonal = np.diag(r_matrix)
        phases = np.where(np.abs(diagonal) > 0.0, diagonal / np.abs(diagonal), 1.0)
        rotations.append(q_matrix @ np.diag(phases))
    return tuple(rotations)


def mixed_orbital_rotations(
    n_orbitals: int,
    count: int,
    cadence: int,
    real_seed: int,
    complex_seed: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Return a frozen real sequence with every cadence-th frame complex."""

    if n_orbitals < 1 or count < 1:
        raise ValueError("Orbital and frame counts must be positive.")
    if cadence < 2:
        raise ValueError("complex cadence must be at least two.")
    rotations = list(random_orthogonal_rotations(n_orbitals, count, real_seed))
    complex_indices = tuple(
        index for index in range(count) if (index + 1) % cadence == 0
    )
    complex_rotations = _random_unitaries(
        n_orbitals, len(complex_indices), complex_seed
    )
    is_complex = np.zeros(count, dtype=bool)
    for index, rotation in zip(complex_indices, complex_rotations):
        rotations[index] = rotation
        is_complex[index] = True
    return tuple(np.asarray(rotation) for rotation in rotations), is_complex


def _blind_shadows(shadows: ShadowData) -> ShadowData:
    """Remove posthoc exact responses before the SDP receives the data."""

    return ShadowData(
        rotations=shadows.rotations,
        pair_vectors=shadows.pair_vectors,
        design=shadows.design,
        values=np.asarray(shadows.values, dtype=float),
        lower_bounds=np.asarray(shadows.lower_bounds, dtype=float),
        upper_bounds=np.asarray(shadows.upper_bounds, dtype=float),
        hits=np.asarray(shadows.hits, dtype=int),
        shots_per_basis=shadows.shots_per_basis,
        exact_values=np.full(len(shadows.values), np.nan),
        exact_constraints=False,
        occupations=None,
    )


def _design_diagnostics(shadows: ShadowData, reference: Any) -> dict[str, Any]:
    rows, cols = _symmetric_d2_variables(reference)
    matrix = np.vstack(shadow_design_blocks(shadows, len(reference.pairs), rows, cols))
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = singular[0] * max(matrix.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > tolerance))
    return {
        "design_variable_count": int(len(rows)),
        "design_rank": rank,
        "design_nullity": int(len(rows) - rank),
        "design_nonzero_condition_number": (
            float(singular[0] / singular[rank - 1]) if rank else None
        ),
    }


def _rdm_energy(d2: np.ndarray, gamma: np.ndarray, reference: Any) -> float:
    return float(
        np.sum(reference.one_body * gamma)
        + np.sum(reference.two_body * d2)
        + reference.nuclear_energy
    )


def _score(
    count: int,
    noise_seed: int | str,
    is_complex: np.ndarray,
    shadows: ShadowData,
    noisy_d2: np.ndarray,
    result: Any,
    exact: Any,
    objective: FtPBEEnergyObjective,
    exact_ftpbe: float,
    nuclear_weight: float,
    seconds: float,
) -> dict[str, Any]:
    d2 = np.asarray(result.d2, dtype=float)
    gamma = np.asarray(result.gamma, dtype=float)
    corrected = np.asarray(result.corrected_d2, dtype=float)
    hamiltonian = _rdm_energy(d2, gamma, exact)
    ftpbe = float(objective.evaluate(d2, gamma, gradient=False).total_energy)
    signed_h = 1000.0 * (hamiltonian - exact.exact_energy)
    signed_ftpbe = 1000.0 * (ftpbe - exact_ftpbe)
    corrected_fit = shadows.predict(corrected)
    d2_matrix, q2_matrix, g2_matrix = dqg_matrices(d2, gamma, exact.pairs)
    return {
        "noise_seed": noise_seed,
        "m": count,
        "total_nominal_shots": count * shadows.shots_per_basis,
        "complex_frames": int(np.count_nonzero(is_complex[:count])),
        "complex_frame_indices_one_based": ",".join(
            str(index + 1) for index in np.flatnonzero(is_complex[:count])
        ),
        "status": str(result.status),
        "fit_status": str(result.fit_status),
        "solver_seconds": seconds,
        "d2_frobenius_error": float(np.linalg.norm(d2 - exact.exact_d2)),
        "gamma_frobenius_error": float(np.linalg.norm(gamma - exact.exact_gamma)),
        "physical_to_noisy_d2_frobenius": float(np.linalg.norm(d2 - noisy_d2)),
        "corrected_to_noisy_d2_frobenius": float(np.linalg.norm(corrected - noisy_d2)),
        "corrected_shadow_rmse": float(
            np.sqrt(np.mean((corrected_fit - shadows.values) ** 2))
        ),
        "hamiltonian_energy_eh": hamiltonian,
        "signed_hamiltonian_error_meh": signed_h,
        "hamiltonian_error_meh": abs(signed_h),
        "ftpbe_energy_eh": ftpbe,
        "signed_ftpbe_error_meh": signed_ftpbe,
        "ftpbe_error_meh": abs(signed_ftpbe),
        "shadow_error_trace": float(result.shadow_error_trace),
        "paper_penalized_objective_eh": float(
            result.energy + nuclear_weight * result.shadow_error_trace
        ),
        "min_eigenvalue_d": float(np.linalg.eigvalsh(d2_matrix)[0]),
        "min_eigenvalue_q": float(np.linalg.eigvalsh(q2_matrix)[0]),
        "min_eigenvalue_g": float(np.linalg.eigvalsh(g2_matrix)[0]),
        **_design_diagnostics(shadows, exact),
    }


def _configuration(args: argparse.Namespace, seeds: Sequence[int]) -> dict[str, Any]:
    return {
        "system": "N2",
        "bond_length_angstrom": args.bond_length,
        "basis": args.basis,
        "active_space": [args.active_electrons, args.active_orbitals],
        "frame_count": args.frame_count,
        "shots_per_frame": args.shots_per_frame,
        "noise_seeds": list(seeds),
        "real_frame_seed": args.real_frame_seed,
        "complex_frame_seed": args.complex_frame_seed,
        "complex_cadence": args.complex_cadence,
        "nuclear_weight": args.nuclear_weight,
        "probability_model": args.probability_model,
        "solver": args.solver,
        "solver_tolerance": args.solver_tolerance,
        "solver_threads": args.solver_threads,
        "max_iterations": args.max_iterations,
        "positivity": "DQG",
        "symmetry_blocked_psd": True,
        "grid_level": args.grid_level,
        "measurement_model": (
            "one global Gaussian noisy 2-RDM per seed; every frame is an exact "
            "projection of that same matrix"
        ),
        "optimization": "E[D] + w Tr(E_positive + E_negative)",
        "weighted_gls_used": False,
        "wilson_intervals_used": False,
        "frame_scope": (
            "state-independent mixed spatial rotations; every cadence-th frame "
            "is complex Haar and the remainder are real Haar-orthogonal"
        ),
        "known_reproduction_assumption": (
            "The unavailable supplement does not define the four-index mapping "
            "from natural occupations to Gaussian p; natural-pair-geometric is "
            "the recorded default reconstruction assumption."
        ),
        "upstream_repository": "https://github.com/damazz/ConstrainedShadowTomography",
        "upstream_commit": "190341d",
        "upstream_maple_file": "sv2RDM_example.mpl",
        "paper_equation": 11,
        "supplement_present_in_upstream_or_arxiv_source": False,
    }


def _build_references(args: argparse.Namespace) -> tuple[Any, Any, Any, float]:
    selection = build_n2_selection_reference(
        args.bond_length,
        basis=args.basis,
        active_electrons=args.active_electrons,
        active_orbitals=args.active_orbitals,
    )
    from constrained_shadow import build_n2_reference

    exact = build_n2_reference(
        args.bond_length,
        basis=args.basis,
        active_electrons=args.active_electrons,
        active_orbitals=args.active_orbitals,
    )
    np.testing.assert_allclose(selection.one_body, exact.one_body, atol=2e-9)
    np.testing.assert_allclose(selection.two_body, exact.two_body, atol=2e-9)
    objective = FtPBEEnergyObjective(selection, grid_level=args.grid_level)
    exact_ftpbe = float(
        objective.evaluate(
            exact.exact_d2, exact.exact_gamma, gradient=False
        ).total_energy
    )
    return selection, exact, objective, exact_ftpbe


def _run_seed(
    args: argparse.Namespace,
    noise_seed: int,
    configuration: dict[str, Any],
    rotations: Sequence[np.ndarray],
    is_complex: np.ndarray,
    selection: Any,
    exact: Any,
    objective: FtPBEEnergyObjective,
    exact_ftpbe: float,
    baseline: Any,
) -> list[dict[str, Any]]:
    seed_dir = args.output_dir / f"seed_{noise_seed}"
    analysis_path = seed_dir / "analysis.json"
    if analysis_path.is_file() and not args.overwrite:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        rows = list(payload.get("rows", ()))
        if [int(row["m"]) for row in rows] == list(range(1, args.frame_count + 1)):
            print(f"[seed={noise_seed}] complete checkpoint", flush=True)
            return rows

    all_shadows, noisy_d2, noise_diagnostics = gaussian_noisy_shadows(
        exact,
        rotations,
        args.shots_per_frame,
        noise_seed,
        probability_model=args.probability_model,
    )
    seed_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        seed_dir / "paper_input.npz",
        rotations=np.asarray(rotations),
        is_complex=is_complex,
        noisy_d2=noisy_d2,
        shadow_values=np.asarray(all_shadows.values),
    )

    current_d2 = np.asarray(baseline.d2, dtype=float)
    current_gamma = np.asarray(baseline.gamma, dtype=float)
    current_corrected = current_d2.copy()
    rows: list[dict[str, Any]] = []
    for count in range(1, args.frame_count + 1):
        checkpoint_npz = seed_dir / "checkpoints" / f"m_{count:02d}.npz"
        checkpoint_json = checkpoint_npz.with_suffix(".json")
        if (
            checkpoint_npz.is_file()
            and checkpoint_json.is_file()
            and not args.overwrite
        ):
            with np.load(checkpoint_npz, allow_pickle=False) as arrays:
                current_d2 = np.asarray(arrays["d2"], dtype=float)
                current_gamma = np.asarray(arrays["gamma"], dtype=float)
                current_corrected = np.asarray(arrays["corrected_d2"], dtype=float)
            rows.append(json.loads(checkpoint_json.read_text(encoding="utf-8")))
            print(f"[seed={noise_seed} m={count:02d}] checkpoint", flush=True)
            continue

        scored_shadows = subset_shadows(all_shadows, count)
        solver_shadows = _blind_shadows(scored_shadows)
        started = time.perf_counter()
        result = solve_dqg_sdp(
            LeakGuardReference(selection),
            shadow_data=solver_shadows,
            shadow_error_weight=args.nuclear_weight,
            solver=args.solver,
            tolerance=args.solver_tolerance,
            max_iterations=args.max_iterations,
            solver_threads=args.solver_threads,
            positivity_conditions="DQG",
            symmetry_blocked_psd=True,
            selection_objective="energy",
            initial_d2=current_d2,
            initial_gamma=current_gamma,
            initial_corrected_d2=current_corrected,
        )
        elapsed = time.perf_counter() - started
        row = _score(
            count,
            noise_seed,
            is_complex,
            scored_shadows,
            noisy_d2,
            result,
            exact,
            objective,
            exact_ftpbe,
            args.nuclear_weight,
            elapsed,
        )
        rows.append(row)
        current_d2 = np.asarray(result.d2, dtype=float)
        current_gamma = np.asarray(result.gamma, dtype=float)
        current_corrected = np.asarray(result.corrected_d2, dtype=float)
        _atomic_npz(
            checkpoint_npz,
            d2=current_d2,
            gamma=current_gamma,
            corrected_d2=current_corrected,
        )
        _atomic_json(checkpoint_json, row)
        print(
            f"[seed={noise_seed} m={count:02d}] "
            f"D2={row['d2_frobenius_error']:.6f}, "
            f"H={row['signed_hamiltonian_error_meh']:+.3f}, "
            f"ftPBE={row['signed_ftpbe_error_meh']:+.3f} mEh, "
            f"traceE={row['shadow_error_trace']:.5f}, {elapsed:.1f}s",
            flush=True,
        )

    _write_csv(seed_dir / "trajectory.csv", rows)
    _atomic_json(
        analysis_path,
        {
            "configuration": {**configuration, "noise_seed": noise_seed},
            "noise_diagnostics": noise_diagnostics,
            "rows": rows,
        },
    )
    return rows


def _run_exact(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    rotations: Sequence[np.ndarray],
    is_complex: np.ndarray,
    selection: Any,
    exact: Any,
    objective: FtPBEEnergyObjective,
    exact_ftpbe: float,
    baseline: Any,
) -> list[dict[str, Any]]:
    output = args.output_dir / "exact_shadow"
    analysis_path = output / "analysis.json"
    if analysis_path.is_file() and not args.overwrite:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        rows = list(payload.get("rows", ()))
        if [int(row["m"]) for row in rows] == list(range(1, args.frame_count + 1)):
            print("[exact] complete checkpoint", flush=True)
            return rows

    all_shadows = exact_shadows(exact, rotations)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        output / "paper_input.npz",
        rotations=np.asarray(rotations),
        is_complex=is_complex,
        shadow_values=np.asarray(all_shadows.values),
    )

    current_d2 = np.asarray(baseline.d2, dtype=float)
    current_gamma = np.asarray(baseline.gamma, dtype=float)
    current_corrected = current_d2.copy()
    rows: list[dict[str, Any]] = []
    for count in range(1, args.frame_count + 1):
        checkpoint_npz = output / "checkpoints" / f"m_{count:02d}.npz"
        checkpoint_json = checkpoint_npz.with_suffix(".json")
        if (
            checkpoint_npz.is_file()
            and checkpoint_json.is_file()
            and not args.overwrite
        ):
            with np.load(checkpoint_npz, allow_pickle=False) as arrays:
                current_d2 = np.asarray(arrays["d2"], dtype=float)
                current_gamma = np.asarray(arrays["gamma"], dtype=float)
                current_corrected = np.asarray(arrays["corrected_d2"], dtype=float)
            rows.append(json.loads(checkpoint_json.read_text(encoding="utf-8")))
            print(f"[exact m={count:02d}] checkpoint", flush=True)
            continue

        scored_shadows = subset_shadows(all_shadows, count)
        solver_shadows = _blind_shadows(scored_shadows)
        started = time.perf_counter()
        result = solve_dqg_sdp(
            LeakGuardReference(selection),
            shadow_data=solver_shadows,
            shadow_error_weight=args.nuclear_weight,
            solver=args.solver,
            tolerance=args.solver_tolerance,
            max_iterations=args.max_iterations,
            solver_threads=args.solver_threads,
            positivity_conditions="DQG",
            symmetry_blocked_psd=True,
            selection_objective="energy",
            initial_d2=current_d2,
            initial_gamma=current_gamma,
            initial_corrected_d2=current_corrected,
        )
        elapsed = time.perf_counter() - started
        row = _score(
            count,
            "exact",
            is_complex,
            scored_shadows,
            exact.exact_d2,
            result,
            exact,
            objective,
            exact_ftpbe,
            args.nuclear_weight,
            elapsed,
        )
        row["trajectory_type"] = "exact_shadow"
        rows.append(row)
        current_d2 = np.asarray(result.d2, dtype=float)
        current_gamma = np.asarray(result.gamma, dtype=float)
        current_corrected = np.asarray(result.corrected_d2, dtype=float)
        _atomic_npz(
            checkpoint_npz,
            d2=current_d2,
            gamma=current_gamma,
            corrected_d2=current_corrected,
        )
        _atomic_json(checkpoint_json, row)
        print(
            f"[exact m={count:02d}] "
            f"D2={row['d2_frobenius_error']:.6f}, "
            f"H={row['signed_hamiltonian_error_meh']:+.3f}, "
            f"ftPBE={row['signed_ftpbe_error_meh']:+.3f} mEh, "
            f"traceE={row['shadow_error_trace']:.5f}, {elapsed:.1f}s",
            flush=True,
        )

    _write_csv(output / "trajectory.csv", rows)
    _atomic_json(
        analysis_path,
        {
            "configuration": {
                **configuration,
                "measurement_model": "exact noiseless shadow projections",
                "shots_per_frame": 0,
                "trajectory_type": "exact_shadow",
            },
            "rows": rows,
        },
    )
    return rows


def _aggregate_rows(
    configuration: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    seeds = tuple(int(seed) for seed in configuration["noise_seeds"])
    aggregate = []
    for count in range(1, int(configuration["frame_count"]) + 1):
        selected = [row for row in rows if int(row["m"]) == count]
        if len(selected) != len(seeds):
            raise RuntimeError(
                f"m={count} has {len(selected)} rows for {len(seeds)} seeds."
            )
        summary: dict[str, Any] = {
            "m": count,
            "seed_count": len(selected),
            "complex_frames": int(selected[0]["complex_frames"]),
            "total_nominal_shots": int(selected[0]["total_nominal_shots"]),
        }
        for metric in AGGREGATE_METRICS:
            values = np.asarray([float(row[metric]) for row in selected])
            standard_deviation = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            critical = (
                float(student_t.ppf(0.975, len(values) - 1)) if len(values) > 1 else 0.0
            )
            half_width = critical * standard_deviation / math.sqrt(len(values))
            summary.update(
                {
                    f"{metric}_mean": float(np.mean(values)),
                    f"{metric}_median": float(np.median(values)),
                    f"{metric}_std": standard_deviation,
                    f"{metric}_ci95_low": float(np.mean(values) - half_width),
                    f"{metric}_ci95_high": float(np.mean(values) + half_width),
                    f"{metric}_q25": float(np.quantile(values, 0.25)),
                    f"{metric}_q75": float(np.quantile(values, 0.75)),
                    f"{metric}_min": float(np.min(values)),
                    f"{metric}_max": float(np.max(values)),
                    f"{metric}_p90": float(np.quantile(values, 0.90)),
                }
            )
        aggregate.append(summary)
    return aggregate


def _plot_absolute(
    output: Path,
    rows: Sequence[dict[str, Any]],
    aggregate: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    exact_rows: Sequence[dict[str, Any]],
) -> None:
    metrics = (
        ("d2_frobenius_error", "2-RDM Frobenius error"),
        ("hamiltonian_error_meh", "Hamiltonian error (mEh)"),
        ("ftpbe_error_meh", "ftPBE error (mEh)"),
    )
    figure, axes = plt.subplots(3, 1, figsize=(9.2, 11.0), sharex=True)
    for axis, (metric, label) in zip(axes, metrics):
        for seed in seeds:
            selected = [row for row in rows if int(row["noise_seed"]) == seed]
            axis.plot(
                [int(row["m"]) for row in selected],
                [max(float(row[metric]), 1e-9) for row in selected],
                color="#7A7A7A",
                alpha=0.42,
                linewidth=0.9,
            )
        counts = np.asarray([int(row["m"]) for row in aggregate])
        mean = np.asarray([float(row[f"{metric}_mean"]) for row in aggregate])
        low = np.asarray([float(row[f"{metric}_ci95_low"]) for row in aggregate])
        high = np.asarray([float(row[f"{metric}_ci95_high"]) for row in aggregate])
        axis.fill_between(
            counts,
            np.maximum(low, 1e-9),
            np.maximum(high, 1e-9),
            color="#5B8DB8",
            alpha=0.22,
            label="95% t interval",
        )
        axis.plot(counts, mean, color="#1769AA", linewidth=2.0, label="seed mean")
        if exact_rows:
            axis.plot(
                [int(row["m"]) for row in exact_rows],
                [max(float(row[metric]), 1e-9) for row in exact_rows],
                color="#C23B22",
                linewidth=1.7,
                marker="o",
                markersize=2.5,
                label="exact shadow",
            )
        axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.grid(True, which="both", color="#E1E1E1", linewidth=0.55)
    axes[-1].set_xlabel("Number of shadow frames m")
    axes[0].legend(frameon=False, ncol=2)
    figure.suptitle(
        "N2 paper nuclear-norm constrained shadows: mixed real/complex frames",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(output / "finite_shot_trajectories.png", dpi=220)
    figure.savefig(output / "three_error_curves.png", dpi=220)
    plt.close(figure)


def _plot_boxplots(
    output: Path,
    rows: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    frame_count: int,
    exact_rows: Sequence[dict[str, Any]],
) -> None:
    metrics = (
        ("d2_frobenius_error", "2-RDM Frobenius error"),
        ("hamiltonian_error_meh", "Hamiltonian error (mEh)"),
        ("ftpbe_error_meh", "ftPBE error (mEh)"),
    )
    positions = np.arange(1, frame_count + 1)
    offsets = np.linspace(-0.16, 0.16, len(seeds))
    seed_colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(seeds)))
    figure, axes = plt.subplots(3, 1, figsize=(13.0, 11.4), sharex=True)
    for axis, (metric, label) in zip(axes, metrics):
        distributions = [
            [max(float(row[metric]), 1e-9) for row in rows if int(row["m"]) == count]
            for count in positions
        ]
        axis.boxplot(
            distributions,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#1F1F1F", "linewidth": 1.15},
            boxprops={"facecolor": "#9CC7E8", "edgecolor": "#4C84AE", "alpha": 0.78},
            whiskerprops={"color": "#4C84AE", "linewidth": 0.9},
            capprops={"color": "#4C84AE", "linewidth": 0.9},
        )
        for offset, color, seed in zip(offsets, seed_colors, seeds):
            selected = sorted(
                (row for row in rows if int(row["noise_seed"]) == seed),
                key=lambda row: int(row["m"]),
            )
            axis.scatter(
                positions + offset,
                [max(float(row[metric]), 1e-9) for row in selected],
                s=14,
                color=color,
                edgecolors="white",
                linewidths=0.35,
                zorder=3,
                label=str(seed),
            )
        if exact_rows:
            axis.plot(
                positions,
                [max(float(row[metric]), 1e-9) for row in exact_rows],
                color="#C23B22",
                linewidth=1.55,
                marker="o",
                markersize=2.5,
                zorder=4,
                label="exact shadow",
            )
        axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.grid(True, which="both", axis="y", color="#E1E1E1", linewidth=0.55)
    axes[-1].set_xlabel("Number of shadow frames m")
    axes[-1].set_xticks([1, 5, 10, 15, 20, 25, 30])
    axes[0].legend(
        title="trajectory", frameon=False, ncol=3, fontsize=8, title_fontsize=8
    )
    figure.suptitle(
        "N2 paper nuclear-norm constrained shadows: error distributions",
        fontsize=12,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(output / "three_error_boxplots.png", dpi=220)
    plt.close(figure)


def _plot_signed(
    output: Path,
    rows: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    exact_rows: Sequence[dict[str, Any]],
) -> None:
    energy_metrics = (
        ("signed_hamiltonian_error_meh", "Signed Hamiltonian error (mEh)"),
        ("signed_ftpbe_error_meh", "Signed ftPBE error (mEh)"),
    )
    figure, axes = plt.subplots(3, 1, figsize=(9.2, 10.2), sharex=True)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(seeds)))

    for color, seed in zip(colors, seeds):
        selected = [row for row in rows if int(row["noise_seed"]) == seed]
        axes[0].plot(
            [int(row["m"]) for row in selected],
            [float(row["d2_frobenius_error"]) for row in selected],
            color=color,
            linewidth=1.25,
            label=str(seed),
        )
    if exact_rows:
        axes[0].plot(
            [int(row["m"]) for row in exact_rows],
            [float(row["d2_frobenius_error"]) for row in exact_rows],
            color="#C23B22",
            linewidth=1.7,
            marker="o",
            markersize=2.5,
            label="exact shadow",
        )
    axes[0].set_ylabel("2-RDM Frobenius error")
    axes[0].grid(True, color="#E1E1E1", linewidth=0.55)

    for axis, (metric, label) in zip(axes[1:], energy_metrics):
        for color, seed in zip(colors, seeds):
            selected = [row for row in rows if int(row["noise_seed"]) == seed]
            axis.plot(
                [int(row["m"]) for row in selected],
                [float(row[metric]) for row in selected],
                color=color,
                linewidth=1.25,
                label=str(seed),
            )
        if exact_rows:
            axis.plot(
                [int(row["m"]) for row in exact_rows],
                [float(row[metric]) for row in exact_rows],
                color="#C23B22",
                linewidth=1.7,
                marker="o",
                markersize=2.5,
                label="exact shadow",
            )
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_ylabel(label)
        axis.grid(True, color="#E1E1E1", linewidth=0.55)
    axes[-1].set_xlabel("Number of shadow frames m")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "signed_energy_trajectories.png", dpi=220)
    plt.close(figure)


def _report(
    configuration: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    aggregate: Sequence[dict[str, Any]],
    exact_rows: Sequence[dict[str, Any]],
) -> str:
    endpoint = aggregate[-1]
    lines = [
        "# N2 paper nuclear-norm mixed-shadow trajectories",
        "",
        "## Protocol",
        "",
        "- Primary SDP: `E[D] + w Tr(E_positive + E_negative)` with DQG.",
        "- No weighted GLS, Wilson interval, ridge, or lexicographic fit stage.",
        "- One global Gaussian noisy 2-RDM is reused by all prefixes in each seed.",
        f"- Frames: {configuration['frame_count']} total; every "
        f"{configuration['complex_cadence']}-th frame is complex Haar.",
        f"- Nominal shots/frame: {configuration['shots_per_frame']}; "
        f"nuclear weight: {configuration['nuclear_weight']}.",
        "",
        "## Endpoint by seed",
        "",
        "| seed | D2 error | signed H (mEh) | signed ftPBE (mEh) | Tr(E1+E2) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if int(row["m"]) != int(configuration["frame_count"]):
            continue
        lines.append(
            f"| {row['noise_seed']} | {row['d2_frobenius_error']:.8f} | "
            f"{row['signed_hamiltonian_error_meh']:+.6f} | "
            f"{row['signed_ftpbe_error_meh']:+.6f} | "
            f"{row['shadow_error_trace']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## Endpoint aggregate",
            "",
            "| metric | mean | median | p90 | min..max |",
            "|:---|---:|---:|---:|:---|",
        ]
    )
    for metric, label in (
        ("d2_frobenius_error", "D2 Frobenius"),
        ("hamiltonian_error_meh", "absolute H (mEh)"),
        ("ftpbe_error_meh", "absolute ftPBE (mEh)"),
    ):
        lines.append(
            f"| {label} | {endpoint[f'{metric}_mean']:.8f} | "
            f"{endpoint[f'{metric}_median']:.8f} | "
            f"{endpoint[f'{metric}_p90']:.8f} | "
            f"{endpoint[f'{metric}_min']:.8f}..{endpoint[f'{metric}_max']:.8f} |"
        )
    if exact_rows:
        exact_endpoint = exact_rows[-1]
        lines.extend(
            [
                "",
                "## Exact-shadow endpoint",
                "",
                "| D2 error | signed H (mEh) | signed ftPBE (mEh) | Tr(E1+E2) |",
                "|---:|---:|---:|---:|",
                f"| {exact_endpoint['d2_frobenius_error']:.8f} | "
                f"{exact_endpoint['signed_hamiltonian_error_meh']:+.6f} | "
                f"{exact_endpoint['signed_ftpbe_error_meh']:+.6f} | "
                f"{exact_endpoint['shadow_error_trace']:.8f} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Reproduction boundary",
            "",
            "The public repository and arXiv source omit the cited supplement. "
            "The mapping from four-index 2-RDM entries to the natural-occupation "
            "probability used in the Gaussian variance is therefore not uniquely "
            "specified. This run records and uses `natural-pair-geometric`. The "
            "complex-frame cadence is an explicit phase-completeness extension; "
            "the public Maple generator itself samples real orthogonal matrices.",
            "",
        ]
    )
    return "\n".join(lines)


def _aggregate(
    args: argparse.Namespace,
    configuration: dict[str, Any],
    seeds: Sequence[int],
) -> None:
    rows = []
    for seed in seeds:
        path = args.output_dir / f"seed_{seed}" / "analysis.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed seed result: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    aggregate = _aggregate_rows(configuration, rows)
    exact_path = args.output_dir / "exact_shadow" / "analysis.json"
    exact_rows = []
    if exact_path.is_file():
        exact_rows = list(json.loads(exact_path.read_text(encoding="utf-8"))["rows"])
    _write_csv(args.output_dir / "trajectories.csv", rows)
    _write_csv(args.output_dir / "aggregate_by_m.csv", aggregate)
    _atomic_json(
        args.output_dir / "summary.json",
        {
            "configuration": configuration,
            f"m{configuration['frame_count']}": aggregate[-1],
            "exact_shadow_endpoint": exact_rows[-1] if exact_rows else None,
        },
    )
    _plot_absolute(args.output_dir, rows, aggregate, seeds, exact_rows)
    _plot_boxplots(
        args.output_dir,
        rows,
        seeds,
        int(configuration["frame_count"]),
        exact_rows,
    )
    _plot_signed(args.output_dir, rows, seeds, exact_rows)
    (args.output_dir / "PAPER_NUCLEAR_HYBRID_FINDINGS.md").write_text(
        _report(configuration, rows, aggregate, exact_rows), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    args.output_dir = args.output_dir.resolve()
    args.solver = args.solver.upper()
    if args.frame_count < 1 or args.shots_per_frame < 1:
        raise ValueError("frame-count and shots-per-frame must be positive.")
    if args.nuclear_weight <= 0.0:
        raise ValueError("nuclear-weight must be positive.")
    seeds = tuple(args.noise_seed or DEFAULT_NOISE_SEEDS)
    if len(set(seeds)) != len(seeds):
        raise ValueError("noise seeds must be unique.")
    if args.aggregate_only and args.exact_only:
        raise ValueError("--aggregate-only and --exact-only are mutually exclusive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = _configuration(args, seeds)
    _atomic_json(args.output_dir / "configuration.json", configuration)
    if args.aggregate_only:
        _aggregate(args, configuration, seeds)
        return

    print("Building aligned N2 references and ftPBE grid...", flush=True)
    selection, exact, objective, exact_ftpbe = _build_references(args)
    rotations, is_complex = mixed_orbital_rotations(
        exact.n_spatial_orbitals,
        args.frame_count,
        args.complex_cadence,
        args.real_frame_seed,
        args.complex_frame_seed,
    )
    print("Solving the shadow-free DQG baseline...", flush=True)
    baseline = solve_dqg_sdp(
        LeakGuardReference(selection),
        solver=args.solver,
        tolerance=args.solver_tolerance,
        max_iterations=args.max_iterations,
        solver_threads=args.solver_threads,
        positivity_conditions="DQG",
        symmetry_blocked_psd=True,
    )
    if args.exact_only:
        _run_exact(
            args,
            configuration,
            rotations,
            is_complex,
            selection,
            exact,
            objective,
            exact_ftpbe,
            baseline,
        )
        _aggregate(args, configuration, seeds)
        print(f"Results: {args.output_dir}", flush=True)
        return
    for seed in seeds:
        _run_seed(
            args,
            seed,
            configuration,
            rotations,
            is_complex,
            selection,
            exact,
            objective,
            exact_ftpbe,
            baseline,
        )
    _aggregate(args, configuration, seeds)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
