#!/usr/bin/env python3
"""Scan raw Aer weighted-LS DQG with and without hard ftPBE-hole bands."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/raw-aer-wls-dqg-xc-hole")

import matplotlib.pyplot as plt
import numpy as np


INTEGRATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = INTEGRATION_ROOT.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
QACSE_ROOT = (
    WORKSPACE_ROOT
    / "QACSE"
    / "ConstrainedShadowTomography"
    / "python_reproduction"
)
for import_root in (QACSE_ROOT, PROJECT_ROOT):
    value = str(import_root)
    if value not in sys.path:
        sys.path.insert(0, value)

from constrained_shadow import (  # noqa: E402
    SDPResult,
    ShadowData,
    build_n2_reference,
    load_shadow_npz,
    random_orthogonal_rotations,
    reconstruction_metrics,
    sample_shadows,
    save_npz,
    solve_dqg_sdp,
    subset_shadows,
)
from xc_hole_constraints import (  # noqa: E402
    ExplicitBandConfig,
    FullGridHoleConstraint,
    HoleGridConfig,
    solve_calibrated_hard_ftpbe_target,
)


METHOD_BASELINE = "raw Aer + weighted-LS + DQG"
METHOD_HOLE = "raw Aer + weighted-LS + DQG + hard ftPBE hole"
METHOD_HOLE_CLOSEST = METHOD_HOLE + " (closest baseline D2)"
METHOD_HOLE_ORACLE = METHOD_HOLE + " (oracle closest exact D2)"
CSV_FIELDS = (
    "shadows",
    "method",
    "energy",
    "energy_error",
    "d2_normalized_frobenius_error",
    "d2_frobenius_error",
    "gamma_frobenius_error",
    "shadow_rmse",
    "exact_sampling_rmse",
    "weighted_fit_rmse",
    "weighted_fit_reference_rmse",
    "min_eigenvalue_d",
    "min_eigenvalue_q",
    "min_eigenvalue_g",
    "status",
    "fit_status",
    "seconds",
    "calibrated_relative_tolerance",
    "calibration_active_constraints",
    "hard_active_constraints",
    "hard_constraints_certified",
    "hard_final_maximum_normalized_excess",
)


def parse_shadow_counts(value: str) -> tuple[int, ...]:
    """Parse comma-separated counts or a Python-style start:stop[:step] range."""

    try:
        if ":" in value and "," not in value:
            fields = tuple(int(item.strip()) for item in value.split(":"))
            if len(fields) not in (2, 3):
                raise ValueError
            start, stop = fields[:2]
            step = fields[2] if len(fields) == 3 else 1
            counts = tuple(range(start, stop, step))
        else:
            counts = tuple(
                sorted({int(item.strip()) for item in value.split(",") if item.strip()})
            )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Use comma-separated integers or start:stop[:step]."
        ) from error
    if not counts or min(counts) < 1:
        raise argparse.ArgumentTypeError("Shadow counts must be positive integers.")
    return counts


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shadow-counts", type=parse_shadow_counts, default=tuple(range(1, 31))
    )
    parser.add_argument("--bond-length", type=float, default=1.75)
    parser.add_argument("--active-electrons", type=int, default=6)
    parser.add_argument("--active-orbitals", type=int, default=6)
    parser.add_argument("--shots-per-shadow", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--shadow-input-npz",
        type=Path,
        help="Reuse raw Aer rotations/hits from a constrained-shadow NPZ.",
    )
    parser.add_argument("--solver", default="SCS")
    parser.add_argument("--solver-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--u-min", type=float, default=0.05)
    parser.add_argument("--u-max", type=float, default=0.50)
    parser.add_argument("--u-points", type=int, default=10)
    parser.add_argument("--angular-points", type=int, default=14)
    parser.add_argument("--grid-batch-size", type=int, default=2048)
    parser.add_argument("--cuts-per-round", type=int, default=64)
    parser.add_argument("--max-exchange-rounds", type=int, default=8)
    parser.add_argument("--separation-tolerance", type=float, default=2e-4)
    parser.add_argument("--ftpbe-relative-tolerance", type=float, default=0.0)
    parser.add_argument("--calibration-margin", type=float, default=0.0)
    parser.add_argument(
        "--hole-selection",
        choices=("energy", "closest-baseline", "oracle-exact-d2"),
        default="energy",
        help=(
            "Final hard-band selector after preserving the best weighted shadow fit; "
            "oracle-exact-d2 is a posthoc upper-bound diagnostic with FCI leakage"
        ),
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INTEGRATION_ROOT / "results" / "cas66_shadows_1_30",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def make_weighted_solver(base_solver: Callable[..., SDPResult]) -> Callable[..., SDPResult]:
    """Adapt explicit-band SDP calls to the raw-frequency weighted-LS route."""

    def weighted_solver(
        reference: Any, shadow_data: ShadowData | None = None, **kwargs: Any
    ) -> SDPResult:
        kwargs["weighted_shadow_fit"] = shadow_data is not None
        return base_solver(reference, shadow_data=shadow_data, **kwargs)

    return weighted_solver


def make_closest_weighted_solver(
    base_solver: Callable[..., SDPResult], d2_target: np.ndarray
) -> Callable[..., SDPResult]:
    """Select the hard-band solution nearest a fixed baseline 2-RDM."""

    target = np.asarray(d2_target, dtype=float).copy()

    def closest_solver(
        reference: Any, shadow_data: ShadowData | None = None, **kwargs: Any
    ) -> SDPResult:
        kwargs["weighted_shadow_fit"] = shadow_data is not None
        kwargs["selection_objective"] = "closest_d2"
        kwargs["selection_d2_target"] = target
        return base_solver(reference, shadow_data=shadow_data, **kwargs)

    return closest_solver


def weighted_fit_rmse(shadows: ShadowData, d2: np.ndarray) -> float:
    """Return the Jeffreys-stabilized residual used by solve_dqg_sdp."""

    probabilities = (shadows.hits.astype(float) + 0.5) / (
        shadows.shots_per_basis + 1.0
    )
    variances = np.maximum(
        probabilities * (1.0 - probabilities) / shadows.shots_per_basis,
        1e-12,
    )
    residual = (shadows.predict(d2) - shadows.values) / np.sqrt(variances)
    return float(np.sqrt(np.mean(residual**2)))


def _grid_config(args: argparse.Namespace) -> HoleGridConfig:
    return HoleGridConfig(
        grid_level=args.grid_level,
        u_min_bohr=args.u_min,
        u_max_bohr=args.u_max,
        u_points=args.u_points,
        angular_points=args.angular_points,
        batch_size=args.grid_batch_size,
    )


def _band_config(args: argparse.Namespace) -> ExplicitBandConfig:
    return ExplicitBandConfig(
        mode="ftpbe_target",
        cuts_per_round=args.cuts_per_round,
        max_rounds=args.max_exchange_rounds,
        separation_tolerance=args.separation_tolerance,
        ftpbe_relative_tolerance=args.ftpbe_relative_tolerance,
        allow_minimax_relaxation=False,
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


class ConsoleProgress:
    def __init__(self, heartbeat_seconds: float):
        self.heartbeat_seconds = max(float(heartbeat_seconds), 1.0)
        self.last_update: dict[str, float] = {}

    def __call__(
        self,
        stage: str,
        completed: int | None,
        total: int | None,
        detail: dict[str, Any],
    ) -> None:
        now = time.monotonic()
        event = detail.get("event", "progress")
        last = self.last_update.get(stage, -math.inf)
        if event not in {"start", "complete"} and now - last < self.heartbeat_seconds:
            return
        self.last_update[stage] = now
        amount = "" if completed is None else f" {completed}/{total or '?'}"
        suffixes = []
        for name in (
            "active_constraints",
            "maximum_normalized_excess",
            "minimax_slack",
            "solver_status",
        ):
            if name in detail:
                suffixes.append(f"{name}={detail[name]}")
        suffix = "" if not suffixes else " | " + ", ".join(suffixes)
        print(f"    [{event}] {stage}{amount}{suffix}", flush=True)


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "route": (
            "raw Aer pair frequencies -> Jeffreys-weighted least squares under "
            "DQG -> configured final selection inside the preserved fit region"
        ),
        "comparison": [METHOD_BASELINE, METHOD_HOLE],
        "bond_length_angstrom": args.bond_length,
        "active_electrons": args.active_electrons,
        "active_spatial_orbitals": args.active_orbitals,
        "shadow_counts": list(args.shadow_counts),
        "shots_per_shadow": args.shots_per_shadow,
        "seed": args.seed,
        "solver": args.solver,
        "solver_tolerance": args.solver_tolerance,
        "max_iterations": args.max_iterations,
        "grid": {
            "level": args.grid_level,
            "u_min_bohr": args.u_min,
            "u_max_bohr": args.u_max,
            "u_points": args.u_points,
            "angular_points": args.angular_points,
            "batch_size": args.grid_batch_size,
        },
        "explicit_hole": {
            "mode": "ftpbe_target",
            "cuts_per_round": args.cuts_per_round,
            "max_exchange_rounds": args.max_exchange_rounds,
            "separation_tolerance": args.separation_tolerance,
            "base_relative_tolerance": args.ftpbe_relative_tolerance,
            "calibration_margin": args.calibration_margin,
            "full_grid_exchange": True,
            "final_constraints": "hard affine inequalities",
            "final_selection_objective": args.hole_selection,
            "posthoc_fci_oracle": args.hole_selection == "oracle-exact-d2",
        },
        "shadow_input_npz": (
            None if args.shadow_input_npz is None else str(args.shadow_input_npz.resolve())
        ),
    }


def _write_or_validate_configuration(
    path: Path, configuration: dict[str, Any], overwrite: bool
) -> None:
    if path.exists() and not overwrite:
        existing = json.loads(path.read_text())
        if existing != configuration:
            raise ValueError(
                f"{path.parent} contains checkpoints for a different configuration; "
                "use a new --output-dir or --overwrite."
            )
    _atomic_json(path, configuration)


def _validate_args(args: argparse.Namespace) -> None:
    if args.active_electrons < 2 or args.active_electrons % 2:
        raise ValueError("active_electrons must be a positive even number.")
    if args.active_orbitals < 1 or args.active_electrons > 2 * args.active_orbitals:
        raise ValueError("The active space cannot hold the requested electrons.")
    if args.shots_per_shadow < 1 or args.max_iterations < 1:
        raise ValueError("shots_per_shadow and max_iterations must be positive.")
    if args.solver_tolerance <= 0.0 or args.calibration_margin < 0.0:
        raise ValueError("Solver tolerance must be positive and margin nonnegative.")


def _validate_shadow_archive(
    path: Path,
    shadows: ShadowData,
    reference: Any,
    required_count: int,
    expected_shots: int,
) -> None:
    if shadows.n_shadows < required_count:
        raise ValueError(
            f"{path} has {shadows.n_shadows} shadows; {required_count} are required."
        )
    if shadows.shots_per_basis != expected_shots:
        raise ValueError(
            f"{path} uses {shadows.shots_per_basis} shots, expected {expected_shots}."
        )
    if shadows.rotations[0].shape != (
        reference.n_spatial_orbitals,
        reference.n_spatial_orbitals,
    ):
        raise ValueError(f"{path} does not match the requested active space.")
    with np.load(path) as archive:
        if "exact_d2" in archive.files and not np.allclose(
            archive["exact_d2"], reference.exact_d2, atol=2e-9
        ):
            raise ValueError(f"{path} does not match the rebuilt molecular reference.")


def _validate_rotations(shadows: ShadowData, reference: Any, count: int, seed: int) -> None:
    expected = random_orthogonal_rotations(reference.n_spatial_orbitals, count, seed)
    actual = np.asarray(shadows.rotations[:count])
    if actual.shape != np.asarray(expected).shape or not np.allclose(
        actual, expected, atol=2e-13
    ):
        raise ValueError("The raw-shadow rotations do not match the requested --seed.")


def _load_or_sample_shadows(
    args: argparse.Namespace, reference: Any, output_dir: Path
) -> tuple[ShadowData, str]:
    maximum = max(args.shadow_counts)
    cache_path = output_dir / "raw_aer_shadows.npz"
    input_path = args.shadow_input_npz
    if input_path is None and cache_path.exists() and not args.overwrite:
        input_path = cache_path
    if input_path is not None:
        shadows = load_shadow_npz(input_path)
        _validate_shadow_archive(
            input_path, shadows, reference, maximum, args.shots_per_shadow
        )
        _validate_rotations(shadows, reference, maximum, args.seed)
        return shadows, str(input_path.resolve())

    rotations = random_orthogonal_rotations(
        reference.n_spatial_orbitals, maximum, args.seed
    )
    print(
        f"Sampling {maximum} Aer bases x {args.shots_per_shadow} shots...",
        flush=True,
    )
    shadows = sample_shadows(
        reference,
        rotations,
        args.shots_per_shadow,
        args.seed,
    )
    save_npz(cache_path, reference, shadows)
    return shadows, str(cache_path.resolve())


def _method_row(
    count: int,
    method: str,
    d2: np.ndarray,
    gamma: np.ndarray,
    reference: Any,
    shadows: ShadowData,
    *,
    status: str,
    fit_status: str | None,
    seconds: float,
    weighted_fit_reference_rmse: float,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "shadows": count,
        "method": method,
        **reconstruction_metrics(d2, gamma, reference, shadows),
        "weighted_fit_rmse": weighted_fit_rmse(shadows, d2),
        "weighted_fit_reference_rmse": weighted_fit_reference_rmse,
        "status": status,
        "fit_status": fit_status,
        "seconds": float(seconds),
        **extra,
    }


def _hole_summary(explicit: Any, config: ExplicitBandConfig) -> dict[str, Any]:
    hard_certified = bool(
        explicit.hard.converged
        and explicit.hard.final_separation.maximum_normalized_excess
        <= config.separation_tolerance
    )
    return {
        "calibrated_relative_tolerance": explicit.calibrated_relative_tolerance,
        "calibration_active_constraints": explicit.calibration.active_constraints,
        "hard_active_constraints": explicit.hard.active_constraints,
        "hard_constraints_certified": hard_certified,
        "hard_final_maximum_normalized_excess": (
            explicit.hard.final_separation.maximum_normalized_excess
        ),
        "calibration": {
            "extra_tolerance": explicit.calibration_extra_tolerance,
            "converged": explicit.calibration.converged,
            "minimax_slack": explicit.calibration.minimax_slack,
            "final_maximum_normalized_excess": (
                explicit.calibration.final_separation.maximum_normalized_excess
            ),
            "rounds": len(explicit.calibration.rounds),
        },
        "hard": {
            "converged": explicit.hard.converged,
            "rounds": len(explicit.hard.rounds),
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    temporary.replace(path)


def _write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    styles = {
        METHOD_BASELINE: dict(color="#2b5d8a", marker="o", label="weighted-LS + DQG"),
        METHOD_HOLE: dict(color="#b34a3c", marker="s", label="+ hard ftPBE hole"),
        METHOD_HOLE_CLOSEST: dict(
            color="#2f7d55", marker="^", label="+ hole, closest baseline D2"
        ),
        METHOD_HOLE_ORACLE: dict(
            color="#7a4b9a", marker="D", label="+ hole, oracle exact-D2 bound"
        ),
    }
    figure, axes = plt.subplots(3, 1, figsize=(7.4, 9.0), sharex=True)
    for method, style in styles.items():
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["shadows"],
        )
        if not selected:
            continue
        counts = [row["shadows"] for row in selected]
        axes[0].semilogy(
            counts,
            [max(row["energy_error"], 1e-16) for row in selected],
            **style,
        )
        axes[1].semilogy(
            counts,
            [max(row["d2_normalized_frobenius_error"], 1e-16) for row in selected],
            **style,
        )
        axes[2].plot(
            counts,
            [row["weighted_fit_rmse"] for row in selected],
            **style,
        )
    axes[0].set_ylabel("Energy error (Eh)")
    axes[1].set_ylabel("Normalized D2 error")
    axes[2].set_ylabel("Weighted fit RMSE")
    axes[2].set_xlabel("Number of shadows")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("N2 raw Aer frequencies: weighted-LS DQG and XC-hole constraint")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _checkpoint_paths(output_dir: Path, count: int) -> tuple[Path, Path]:
    stem = output_dir / "checkpoints" / f"shadows_{count:02d}"
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def _load_checkpoint(
    output_dir: Path, count: int
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    json_path, npz_path = _checkpoint_paths(output_dir, count)
    if not json_path.exists() or not npz_path.exists():
        return None
    payload = json.loads(json_path.read_text())
    with np.load(npz_path) as arrays:
        return (
            list(payload["rows"]),
            np.asarray(arrays["baseline_d2"]),
            np.asarray(arrays["baseline_gamma"]),
            np.asarray(arrays["hole_d2"]),
            np.asarray(arrays["hole_gamma"]),
        )


def _write_checkpoint(
    output_dir: Path,
    count: int,
    rows: list[dict[str, Any]],
    baseline: SDPResult,
    explicit: Any,
) -> None:
    json_path, npz_path = _checkpoint_paths(output_dir, count)
    _atomic_npz(
        npz_path,
        baseline_d2=baseline.d2,
        baseline_gamma=baseline.gamma,
        hole_d2=explicit.d2,
        hole_gamma=explicit.gamma,
        calibration_active_keys=np.asarray(explicit.calibration.active_keys, dtype=int),
        hard_active_keys=np.asarray(explicit.hard.active_keys, dtype=int),
    )
    _atomic_json(json_path, {"shadows": count, "rows": rows})


def _comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    counts = sorted({int(row["shadows"]) for row in rows})
    for count in counts:
        methods = {row["method"]: row for row in rows if row["shadows"] == count}
        candidates = [row for method, row in methods.items() if method != METHOD_BASELINE]
        if METHOD_BASELINE not in methods or len(candidates) != 1:
            continue
        baseline = methods[METHOD_BASELINE]
        hole = candidates[0]
        output.append(
            {
                "shadows": count,
                "energy_error_ratio_hole_over_baseline": (
                    hole["energy_error"] / max(baseline["energy_error"], 1e-30)
                ),
                "d2_error_ratio_hole_over_baseline": (
                    hole["d2_normalized_frobenius_error"]
                    / max(baseline["d2_normalized_frobenius_error"], 1e-30)
                ),
                "weighted_fit_rmse_increase": (
                    hole["weighted_fit_rmse"] - baseline["weighted_fit_rmse"]
                ),
                "hole_improves_energy": hole["energy_error"] < baseline["energy_error"],
                "hole_improves_d2": (
                    hole["d2_normalized_frobenius_error"]
                    < baseline["d2_normalized_frobenius_error"]
                ),
            }
        )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    _validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = _configuration(args)
    _write_or_validate_configuration(
        output_dir / "configuration.json", configuration, args.overwrite
    )

    print(
        f"Building N2 CAS({args.active_electrons}e,{args.active_orbitals}o) reference...",
        flush=True,
    )
    reference = build_n2_reference(
        args.bond_length,
        active_electrons=args.active_electrons,
        active_orbitals=args.active_orbitals,
    )
    all_shadows, shadow_source = _load_or_sample_shadows(args, reference, output_dir)
    print(f"Raw-shadow source: {shadow_source}", flush=True)

    progress = ConsoleProgress(args.heartbeat_seconds)
    print("Building full-grid ftPBE-hole constraint...", flush=True)
    constraint = FullGridHoleConstraint(reference, _grid_config(args), progress=progress)
    band_config = _band_config(args)
    weighted_solver = make_weighted_solver(solve_dqg_sdp)

    rows: list[dict[str, Any]] = []
    prior_baseline_d2: np.ndarray | None = None
    prior_baseline_gamma: np.ndarray | None = None
    started = time.perf_counter()
    for index, count in enumerate(args.shadow_counts, start=1):
        if not args.overwrite:
            checkpoint = _load_checkpoint(output_dir, count)
            if checkpoint is not None:
                point_rows, prior_baseline_d2, prior_baseline_gamma, _, _ = checkpoint
                rows.extend(point_rows)
                print(f"[{index}/{len(args.shadow_counts)}] {count} shadows: checkpoint", flush=True)
                continue

        selected = subset_shadows(all_shadows, count)
        print(f"[{index}/{len(args.shadow_counts)}] {count} shadows: weighted-LS DQG", flush=True)
        baseline_started = time.perf_counter()
        baseline = weighted_solver(
            reference,
            shadow_data=selected,
            solver=args.solver,
            tolerance=args.solver_tolerance,
            max_iterations=args.max_iterations,
            verbose=args.verbose_solver,
            positivity_conditions="DQG",
            initial_d2=prior_baseline_d2,
            initial_gamma=prior_baseline_gamma,
        )
        baseline_seconds = time.perf_counter() - baseline_started
        fit_reference = float(
            baseline.weighted_fit_optimum_rmse
            if baseline.weighted_fit_optimum_rmse is not None
            else weighted_fit_rmse(selected, baseline.d2)
        )
        baseline_row = _method_row(
            count,
            METHOD_BASELINE,
            baseline.d2,
            baseline.gamma,
            reference,
            selected,
            status=baseline.status,
            fit_status=baseline.fit_status,
            seconds=baseline_seconds,
            weighted_fit_reference_rmse=fit_reference,
        )

        print(f"[{index}/{len(args.shadow_counts)}] {count} shadows: calibrated hard ftPBE", flush=True)
        hole_started = time.perf_counter()
        if args.hole_selection == "closest-baseline":
            hole_solver = make_closest_weighted_solver(solve_dqg_sdp, baseline.d2)
            hole_method = METHOD_HOLE_CLOSEST
        elif args.hole_selection == "oracle-exact-d2":
            hole_solver = make_closest_weighted_solver(
                solve_dqg_sdp, reference.exact_d2
            )
            hole_method = METHOD_HOLE_ORACLE
        else:
            hole_solver = weighted_solver
            hole_method = METHOD_HOLE
        explicit = solve_calibrated_hard_ftpbe_target(
            reference,
            selected,
            baseline,
            constraint,
            hole_solver,
            band_config,
            calibration_margin=args.calibration_margin,
            solver=args.solver,
            tolerance=args.solver_tolerance,
            max_iterations=args.max_iterations,
            verbose=args.verbose_solver,
            progress=progress,
        )
        hole_seconds = time.perf_counter() - hole_started
        hole_summary = _hole_summary(explicit, band_config)
        hole_row = _method_row(
            count,
            hole_method,
            explicit.d2,
            explicit.gamma,
            reference,
            selected,
            status=explicit.status,
            fit_status=(
                "weighted_least_squares + calibrated_hard_ftpbe + "
                f"{args.hole_selection}"
            ),
            seconds=hole_seconds,
            weighted_fit_reference_rmse=fit_reference,
            **{key: value for key, value in hole_summary.items() if not isinstance(value, dict)},
        )
        point_rows = [baseline_row, hole_row]
        _write_checkpoint(output_dir, count, point_rows, baseline, explicit)
        rows.extend(point_rows)
        prior_baseline_d2 = baseline.d2
        prior_baseline_gamma = baseline.gamma
        _write_csv(output_dir / "trend.csv", rows)
        _write_plot(output_dir / "trend.png", rows)
        _atomic_json(
            output_dir / "summary.json",
            {
                "configuration": configuration,
                "reference": {
                    "exact_energy": reference.exact_energy,
                    "trace_exact_d2": float(np.trace(reference.exact_d2)),
                },
                "raw_shadow_source": shadow_source,
                "rows": rows,
                "comparisons": _comparisons(rows),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        print(
            f"    baseline dE={baseline_row['energy_error']:.3e}, "
            f"D2={baseline_row['d2_normalized_frobenius_error']:.3e}; "
            f"hole dE={hole_row['energy_error']:.3e}, "
            f"D2={hole_row['d2_normalized_frobenius_error']:.3e}, "
            f"certified={hole_summary['hard_constraints_certified']}",
            flush=True,
        )
        gc.collect()

    _write_csv(output_dir / "trend.csv", rows)
    _write_plot(output_dir / "trend.png", rows)
    report = {
        "configuration": configuration,
        "reference": {
            "exact_energy": reference.exact_energy,
            "trace_exact_d2": float(np.trace(reference.exact_d2)),
        },
        "raw_shadow_source": shadow_source,
        "rows": rows,
        "comparisons": _comparisons(rows),
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_dir / "summary.json", report)
    print(f"Completed {len(args.shadow_counts)} points in {report['wall_seconds']:.1f} s.", flush=True)
    print(f"Trend data: {output_dir / 'trend.csv'}", flush=True)
    print(f"Trend plot: {output_dir / 'trend.png'}", flush=True)


if __name__ == "__main__":
    main()
