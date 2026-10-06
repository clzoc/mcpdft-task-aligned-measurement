#!/usr/bin/env python3
"""Blind M=50 validation of the learned C2 frame-retention rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    LeakGuardShadowData,
    acquire_shadow_bases,
    build_c2_reference,
    build_c2_selection_reference,
    load_blind_shadow_npz,
    matrix_to_variable_vector,
    predicted_inverse_variances,
    shadow_design_blocks,
)
from mcpdft_selector import FtPBEEnergyObjective, _symmetric_d2_variables  # noqa: E402
from run_c2_ftpbe_shot_reallocation import AcquisitionOracle  # noqa: E402
from run_c2_oracle_subset_reallocation import (  # noqa: E402
    _equal_allocation,
    _hamiltonian_gradient,
    _oracle_information_allocation,
    _raw_ftpbe_gradient,
    _reallocated_shadows,
    _score_dqg,
)
from run_safe_mcpdft_derandomization import (  # noqa: E402
    RECONSTRUCTION_ENERGY,
    _solve_shadow_dqg,
)
from run_sweep import _atomic_json, _atomic_npz  # noqa: E402


EXPECTED_EXCLUDED_ONE_BASED = (3, 5, 10, 11, 22, 42, 43, 47, 49)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--shadow-archive", type=Path, required=True)
    parser.add_argument("--frozen-mo-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probability-threshold", type=float, default=0.30)
    parser.add_argument("--d2-error-cap", type=float, default=1.25)
    parser.add_argument("--target-variance-cap", type=float, default=1.35)
    parser.add_argument("--energy-target-meh", type=float, default=1.6)
    parser.add_argument("--information-ridge-fraction", type=float, default=1e-3)
    parser.add_argument("--reallocation-seed", type=int, default=20260714)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--solver-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _load_locked_selection(
    feature_dir: Path, threshold: float
) -> tuple[tuple[int, ...], list[dict[str, Any]], dict[str, Any]]:
    analysis = json.loads((feature_dir / "analysis.json").read_text(encoding="utf-8"))
    model = analysis["summary"]["rank_feature_model"]
    feature_order = model["model_feature_order"]
    means = np.asarray(model["scaler_mean"], dtype=float)
    scales = np.asarray(model["scaler_scale"], dtype=float)
    coefficients = np.asarray(model["logistic_coefficients_raw_order"], dtype=float)
    intercept = float(model["logistic_intercept"])
    with (feature_dir / "frame_features.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["maximum"]) == 50]
    if len(rows) != 50:
        raise ValueError("The locked feature table must contain exactly 50 M=50 rows.")
    selected: list[int] = []
    for row in rows:
        vector = np.asarray([float(row[name]) for name in feature_order])
        standardized = (vector - means) / scales
        contributions = coefficients * standardized
        logit = intercept + float(np.sum(contributions))
        probability = 1.0 / (1.0 + math.exp(-logit))
        index = int(row["frame_zero_based"])
        keep = probability >= threshold or bool(int(row["is_complex"]))
        row["retention_probability"] = probability
        row["locked_kept"] = int(keep)
        row["model_logit"] = logit
        for name, contribution in zip(feature_order, contributions):
            row[f"contribution_{name}"] = float(contribution)
        if keep:
            selected.append(index)
    excluded_one_based = tuple(
        int(row["frame_one_based"]) for row in rows if not int(row["locked_kept"])
    )
    if threshold == 0.30 and excluded_one_based != EXPECTED_EXCLUDED_ONE_BASED:
        raise RuntimeError(
            "The frozen p>=0.30 selection changed: "
            f"expected {EXPECTED_EXCLUDED_ONE_BASED}, got {excluded_one_based}."
        )
    return tuple(selected), rows, model


def _information_matrix(
    blocks: Sequence[np.ndarray],
    indices: Sequence[int],
    variables: np.ndarray,
    shots_per_basis: int,
    ridge_fraction: float,
) -> np.ndarray:
    updates = []
    for index in indices:
        block = np.asarray(blocks[int(index)], dtype=float)
        inverse_variances = predicted_inverse_variances(
            block, variables, shots_per_basis, probability_floor=0.01
        )
        weighted = np.sqrt(inverse_variances)[:, None] * block
        updates.append(weighted.T @ weighted)
    variable_count = blocks[0].shape[1]
    reference_inverse_variances = predicted_inverse_variances(
        np.asarray(blocks[0], dtype=float),
        variables,
        shots_per_basis,
        probability_floor=0.01,
    )
    reference_weighted = np.sqrt(reference_inverse_variances)[:, None] * np.asarray(
        blocks[0], dtype=float
    )
    scale = max(
        float(np.trace(reference_weighted.T @ reference_weighted) / variable_count),
        1e-12,
    )
    return ridge_fraction * scale * np.eye(variable_count) + np.sum(updates, axis=0)


def _information_diagnostics(
    blocks: Sequence[np.ndarray],
    selected: Sequence[int],
    variables: np.ndarray,
    hamiltonian_gradient: np.ndarray,
    ftpbe_gradient: np.ndarray,
    shots_per_basis: int,
    ridge_fraction: float,
) -> dict[str, Any]:
    full_indices = tuple(range(len(blocks)))
    full_information = _information_matrix(
        blocks,
        full_indices,
        variables,
        shots_per_basis,
        ridge_fraction,
    )
    selected_information = _information_matrix(
        blocks,
        selected,
        variables,
        shots_per_basis,
        ridge_fraction,
    )
    full_inverse = solve(
        full_information,
        np.eye(full_information.shape[0]),
        assume_a="pos",
        check_finite=False,
    )
    selected_inverse = solve(
        selected_information,
        np.eye(selected_information.shape[0]),
        assume_a="pos",
        check_finite=False,
    )

    def target_variance(inverse: np.ndarray, gradient: np.ndarray) -> float:
        return float(gradient @ inverse @ gradient)

    full_design = np.vstack([blocks[index] for index in full_indices])
    selected_design = np.vstack([blocks[index] for index in selected])
    return {
        "full_design_rank": int(np.linalg.matrix_rank(full_design)),
        "selected_design_rank": int(np.linalg.matrix_rank(selected_design)),
        "trace_variance_ratio": float(
            np.trace(selected_inverse) / np.trace(full_inverse)
        ),
        "hamiltonian_variance_ratio": float(
            target_variance(selected_inverse, hamiltonian_gradient)
            / target_variance(full_inverse, hamiltonian_gradient)
        ),
        "ftpbe_variance_ratio": float(
            target_variance(selected_inverse, ftpbe_gradient)
            / target_variance(full_inverse, ftpbe_gradient)
        ),
    }


def _solve_case(
    name: str,
    acquired: Any,
    solver_args: Any,
    selection: Any,
    exact: Any,
    objective: FtPBEEnergyObjective,
    exact_ftpbe: float,
    warm_d2: np.ndarray,
    warm_gamma: np.ndarray,
    output_dir: Path,
    metadata: dict[str, Any],
    overwrite: bool,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    checkpoint = output_dir / "checkpoints" / f"{name}.npz"
    checkpoint_json = checkpoint.with_suffix(".json")
    if checkpoint.is_file() and checkpoint_json.is_file() and not overwrite:
        row = json.loads(checkpoint_json.read_text(encoding="utf-8"))
        with np.load(checkpoint, allow_pickle=False) as archive:
            d2 = np.asarray(archive["d2"])
            gamma = np.asarray(archive["gamma"])
        print(f"[{name}] checkpoint", flush=True)
        return row, d2, gamma
    print(f"[{name}] solving {acquired.n_shadows} measurement blocks", flush=True)
    started = time.perf_counter()
    result, reconstruction = _solve_shadow_dqg(
        solver_args, selection, acquired, warm_d2, warm_gamma
    )
    elapsed = time.perf_counter() - started
    metrics = _score_dqg(result, exact, objective, exact_ftpbe, acquired)
    row = {
        "case": name,
        "solver_seconds": elapsed,
        "status": str(result.status),
        "fit_status": str(result.fit_status),
        **metadata,
        **metrics,
        **reconstruction,
    }
    d2 = np.asarray(result.d2)
    gamma = np.asarray(result.gamma)
    _atomic_npz(checkpoint, d2=d2, gamma=gamma)
    _atomic_json(checkpoint_json, row)
    print(
        f"[{name}] D2={row['d2_frobenius_error']:.6f}, "
        f"H={row['hamiltonian_error_meh']:.3f}, "
        f"ftPBE={row['ftpbe_error_meh']:.3f} mEh",
        flush=True,
    )
    return row, d2, gamma


def _feature_comparison(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    features = (
        "standardized_residual_rms",
        "absolute_ftpbe_pull_z",
        "absolute_hamiltonian_pull_z",
        "absolute_contrast_pull_z",
        "ftpbe_variance_loss_fraction",
        "hamiltonian_variance_loss_fraction",
        "trace_variance_loss_fraction",
        "d_optimal_loss",
        "d2_pull_frobenius",
        "target_information_alignment",
    )
    kept = [row for row in rows if int(row["locked_kept"])]
    excluded = [row for row in rows if not int(row["locked_kept"])]
    return [
        {
            "feature": feature,
            "kept_median": float(np.median([float(row[feature]) for row in kept])),
            "excluded_median": float(
                np.median([float(row[feature]) for row in excluded])
            ),
        }
        for feature in features
    ]


def _excluded_reasons(
    rows: Sequence[dict[str, Any]], model: dict[str, Any]
) -> list[dict[str, Any]]:
    feature_order = model["model_feature_order"]
    reasons = []
    for row in rows:
        if int(row["locked_kept"]):
            continue
        contributions = sorted(
            ((name, float(row[f"contribution_{name}"])) for name in feature_order),
            key=lambda item: item[1],
        )
        reasons.append(
            {
                "frame_one_based": int(row["frame_one_based"]),
                "retention_probability": float(row["retention_probability"]),
                "strongest_negative_features": ",".join(
                    name for name, value in contributions[:3] if value < 0.0
                ),
                "strongest_negative_contributions": ",".join(
                    f"{value:.4f}" for _, value in contributions[:3] if value < 0.0
                ),
                "is_complex": int(row["is_complex"]),
            }
        )
    return reasons


def _report(payload: dict[str, Any]) -> str:
    configuration = payload["configuration"]
    diagnostics = payload["blind_information_diagnostics"]
    rows = payload["rows"]
    baseline = next(row for row in rows if row["case"] == "full_m50")
    lines = [
        "# C2 M=50 observable-rule validation",
        "",
        "The p>=0.30 retention rule and both guards were frozen from M=20/30/40 "
        "before exact M=50 scoring. M=50 exact RDM data enter only posthoc scores; "
        "the acquisition simulator necessarily uses the exact state to generate new "
        "finite-shot bitstrings.",
        "",
        "## Locked rule",
        "",
        f"- Retained frames: {configuration['selected_frames']} / 50",
        f"- Excluded (1-based): {configuration['excluded_one_based']}",
        f"- Full/selected design rank: {diagnostics['full_design_rank']} / "
        f"{diagnostics['selected_design_rank']}",
        f"- Trace, H, ftPBE variance ratios: "
        f"{diagnostics['trace_variance_ratio']:.4f}, "
        f"{diagnostics['hamiltonian_variance_ratio']:.4f}, "
        f"{diagnostics['ftpbe_variance_ratio']:.4f}",
        "",
        "## Exact scoring after path freeze",
        "",
        "| case | distinct frames | total shots | D2 error | D2/full | H (mEh) | ftPBE (mEh) |",
        "|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['distinct_frames']} | {row['total_shots']} | "
            f"{row['d2_frobenius_error']:.6f} | "
            f"{row['d2_frobenius_error'] / baseline['d2_frobenius_error']:.3f} | "
            f"{row['hamiltonian_error_meh']:.3f} | {row['ftpbe_error_meh']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A useful frame is not simply one with a small residual. It combines "
            "nonredundant design leverage with a statistically significant pull along "
            "the ftPBE tangent, Hamiltonian, or D2 directions. A removable frame is "
            "usually weak/redundant in those directions, or has an anomalous ftPBE-vs-H "
            "contrast after the covariance of the two targets is accounted for. This "
            "classification is pool-dependent: a complex frame can become redundant "
            "only after phase-completing information is already present.",
            "",
            "The trace/D2 guard alone is insufficient. The selected set can preserve "
            "the global inverse-information trace while losing much more precision in "
            "the two narrow energy directions, which is why explicit Hamiltonian and "
            "ftPBE tangent variance guards are part of the deployable rule.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    for name in (
        "feature_dir",
        "baseline_dir",
        "shadow_archive",
        "frozen_mo_npz",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected, feature_rows, model = _load_locked_selection(
        args.feature_dir, args.probability_threshold
    )
    raw = load_blind_shadow_npz(args.shadow_archive, load_design=False)
    if raw.n_shadows != 50:
        raise ValueError("This validation requires exactly 50 candidate frames.")
    with np.load(args.frozen_mo_npz, allow_pickle=False) as archive:
        frozen_mo = np.asarray(archive["mo_coeff"], dtype=float)

    selection = LeakGuardReference(
        build_c2_selection_reference(1.25, basis="cc-pvtz", frozen_mo_coeff=frozen_mo)
    )
    objective = FtPBEEnergyObjective(selection, grid_level=args.grid_level)
    variable_rows, variable_cols = _symmetric_d2_variables(selection)
    blocks = shadow_design_blocks(
        LeakGuardShadowData.from_shadow_data(raw),
        len(selection.pairs),
        variable_rows,
        variable_cols,
    )
    full_checkpoint = args.baseline_dir / "checkpoints" / "shadows_0050.npz"
    with np.load(full_checkpoint, allow_pickle=False) as archive:
        full_d2 = np.asarray(archive["d2"])
        full_gamma = np.asarray(archive["gamma"])
    full_variables = matrix_to_variable_vector(full_d2, variable_rows, variable_cols)
    hamiltonian_gradient = _hamiltonian_gradient(
        selection, variable_rows, variable_cols
    )
    ftpbe_gradient = _raw_ftpbe_gradient(
        objective,
        selection,
        full_d2,
        full_gamma,
        variable_rows,
        variable_cols,
    )
    diagnostics = _information_diagnostics(
        blocks,
        selected,
        full_variables,
        hamiltonian_gradient,
        ftpbe_gradient,
        raw.shots_per_basis,
        args.information_ridge_fraction,
    )
    if diagnostics["selected_design_rank"] < diagnostics["full_design_rank"]:
        raise RuntimeError("The locked subset violates the design-rank guard.")
    if diagnostics["trace_variance_ratio"] > args.d2_error_cap:
        raise RuntimeError("The locked subset violates the D2/trace guard.")
    if (
        max(
            diagnostics["hamiltonian_variance_ratio"],
            diagnostics["ftpbe_variance_ratio"],
        )
        > args.target_variance_cap
    ):
        raise RuntimeError("The locked subset violates the target-variance guard.")

    exact = build_c2_reference(1.25, basis="cc-pvtz", frozen_mo_coeff=frozen_mo)
    exact_ftpbe = float(
        objective.evaluate(
            exact.exact_d2, exact.exact_gamma, gradient=False
        ).total_energy
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

    rows: list[dict[str, Any]] = []
    full_acquired = acquire_shadow_bases(raw, tuple(range(50)))
    full_row = {
        "case": "full_m50",
        "distinct_frames": 50,
        "total_blocks": 50,
        "total_shots": 50 * raw.shots_per_basis,
        "allocation": "original",
        **_score_dqg(
            SimpleNamespace(d2=full_d2, gamma=full_gamma),
            exact,
            objective,
            exact_ftpbe,
            full_acquired,
        ),
    }
    rows.append(full_row)

    subset_acquired = acquire_shadow_bases(raw, selected)
    subset_row, subset_d2, subset_gamma = _solve_case(
        "locked_k41_original",
        subset_acquired,
        solver_args,
        selection,
        exact,
        objective,
        exact_ftpbe,
        full_d2,
        full_gamma,
        args.output_dir,
        {
            "distinct_frames": len(selected),
            "total_blocks": len(selected),
            "total_shots": len(selected) * raw.shots_per_basis,
            "allocation": "original",
        },
        args.overwrite,
    )
    rows.append(subset_row)

    subset_variables = matrix_to_variable_vector(
        subset_d2, variable_rows, variable_cols
    )
    subset_ftpbe_gradient = _raw_ftpbe_gradient(
        objective,
        selection,
        subset_d2,
        subset_gamma,
        variable_rows,
        variable_cols,
    )
    allocations = {
        "equal": _equal_allocation(selected, 50),
        "observable_information": _oracle_information_allocation(
            blocks,
            selected,
            50,
            subset_variables,
            hamiltonian_gradient,
            subset_ftpbe_gradient,
            raw.shots_per_basis,
        ),
    }
    acquisition = AcquisitionOracle(
        exact,
        raw.rotations,
        raw.pair_vectors,
        raw.shots_per_basis,
        args.reallocation_seed,
    )
    for allocation, counts in allocations.items():
        acquired = _reallocated_shadows(raw, selected, counts, acquisition)
        row, _, _ = _solve_case(
            f"locked_k41_reallocated_{allocation}",
            acquired,
            solver_args,
            selection,
            exact,
            objective,
            exact_ftpbe,
            subset_d2,
            subset_gamma,
            args.output_dir,
            {
                "distinct_frames": len(selected),
                "total_blocks": sum(counts.values()),
                "total_shots": sum(counts.values()) * raw.shots_per_basis,
                "allocation": allocation,
                "block_counts_one_based": ",".join(
                    f"{index + 1}:{counts[index]}" for index in selected
                ),
            },
            args.overwrite,
        )
        rows.append(row)

    for row in rows:
        row["d2_error_ratio_to_full"] = (
            row["d2_frobenius_error"] / full_row["d2_frobenius_error"]
        )
        row["meets_d2_cap"] = bool(row["d2_error_ratio_to_full"] <= args.d2_error_cap)
        row["meets_both_energy_targets"] = bool(
            max(row["hamiltonian_error_meh"], row["ftpbe_error_meh"])
            <= args.energy_target_meh
        )
        row["both_energies_better_than_full"] = bool(
            row["hamiltonian_error_meh"] <= full_row["hamiltonian_error_meh"]
            and row["ftpbe_error_meh"] <= full_row["ftpbe_error_meh"]
        )

    excluded_one_based = [
        int(row["frame_one_based"])
        for row in feature_rows
        if not int(row["locked_kept"])
    ]
    payload = {
        "configuration": {
            "training_prefixes": [20, 30, 40],
            "held_out_prefix": 50,
            "selection_uses_exact_m50": False,
            "selection_model_uses_prior_oracle_labels": True,
            "probability_threshold": args.probability_threshold,
            "complex_guard": "always retain complex frames",
            "d2_error_cap": args.d2_error_cap,
            "target_variance_cap": args.target_variance_cap,
            "energy_target_meh": args.energy_target_meh,
            "selected_frames": len(selected),
            "selected_one_based": [index + 1 for index in selected],
            "excluded_one_based": excluded_one_based,
            "shots_per_block": raw.shots_per_basis,
            "reallocation_seed": args.reallocation_seed,
            "observable_information_allocation": (
                "estimated D2 binomial information + Hamiltonian gradient + "
                "ftPBE tangent at locked-subset DQG solution"
            ),
            "exact_state_use": (
                "posthoc scoring and simulated acquisition of extra bitstrings only"
            ),
        },
        "blind_information_diagnostics": diagnostics,
        "feature_comparison": _feature_comparison(feature_rows),
        "excluded_reasons": _excluded_reasons(feature_rows, model),
        "rows": rows,
    }
    _write_csv(args.output_dir / "validation.csv", rows)
    _write_csv(args.output_dir / "m50_frame_scores.csv", feature_rows)
    _write_csv(args.output_dir / "excluded_reasons.csv", payload["excluded_reasons"])
    _atomic_json(args.output_dir / "analysis.json", payload)
    report = args.output_dir / "C2_M50_OBSERVABLE_RULE_VALIDATION.md"
    report.write_text(_report(payload), encoding="utf-8")
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
