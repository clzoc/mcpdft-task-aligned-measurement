#!/usr/bin/env python3
"""Probe-and-promote validation of the learned C2 M=50 retention rule."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CODE = PROJECT / "code"
VENDOR = CODE / "vendor"
for path in (CODE, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_c2_oracle_frame_rules import _prefix_features  # noqa: E402
from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    LeakGuardShadowData,
    build_c2_reference,
    build_c2_selection_reference,
    load_blind_shadow_npz,
    matrix_to_variable_vector,
    shadow_design_blocks,
)
from mcpdft_selector import FtPBEEnergyObjective, _symmetric_d2_variables  # noqa: E402
from run_c2_ftpbe_shot_reallocation import (  # noqa: E402
    AcquisitionOracle,
    _records_to_shadows,
)
from run_c2_m50_observable_rule_validation import (  # noqa: E402
    _hamiltonian_gradient,
    _information_diagnostics,
    _raw_ftpbe_gradient,
    _score_dqg,
    _write_csv,
)
from run_safe_mcpdft_derandomization import (  # noqa: E402
    RECONSTRUCTION_ENERGY,
    _solve_shadow_dqg,
)
from run_sweep import _atomic_json, _atomic_npz  # noqa: E402


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--frozen-mo-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-shots", type=int, default=1_000)
    parser.add_argument("--topup-shots", type=int, default=9_000)
    parser.add_argument("--probability-threshold", type=float, default=0.30)
    parser.add_argument("--d2-guard", type=float, default=1.25)
    parser.add_argument("--target-variance-guard", type=float, default=1.35)
    parser.add_argument("--energy-target-meh", type=float, default=1.6)
    parser.add_argument("--information-ridge-fraction", type=float, default=1e-3)
    parser.add_argument("--sampling-seed", type=int, default=20260715)
    parser.add_argument("--random-seed", type=int, default=8675309)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--solver-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip the known-biased split-block likelihood diagnostic.",
    )
    parser.add_argument(
        "--audit-guard-path",
        action="store_true",
        help="Posthoc-score intermediate subsets on the frozen guard-addition path.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _model_rows(
    rows: list[dict[str, Any]], model: dict[str, Any], threshold: float
) -> tuple[int, ...]:
    features = model["model_feature_order"]
    means = np.asarray(model["scaler_mean"], dtype=float)
    scales = np.asarray(model["scaler_scale"], dtype=float)
    coefficients = np.asarray(model["logistic_coefficients_raw_order"], dtype=float)
    intercept = float(model["logistic_intercept"])
    selected = []
    for row in rows:
        vector = np.asarray([float(row[name]) for name in features])
        logit = intercept + float(np.sum(coefficients * (vector - means) / scales))
        probability = 1.0 / (1.0 + math.exp(-logit))
        keep = probability >= threshold or bool(int(row["is_complex"]))
        row["retention_probability"] = probability
        row["threshold_or_complex_kept"] = int(keep)
        if keep:
            selected.append(int(row["frame_zero_based"]))
    return tuple(selected)


def _guard_passes(
    diagnostics: dict[str, Any], d2_guard: float, target_guard: float
) -> bool:
    return bool(
        diagnostics["selected_design_rank"] >= diagnostics["full_design_rank"]
        and diagnostics["trace_variance_ratio"] <= d2_guard
        and diagnostics["hamiltonian_variance_ratio"] <= target_guard
        and diagnostics["ftpbe_variance_ratio"] <= target_guard
    )


def _guarded_selection(
    rows: list[dict[str, Any]],
    initial: Sequence[int],
    blocks: Sequence[np.ndarray],
    variables: np.ndarray,
    hamiltonian_gradient: np.ndarray,
    ftpbe_gradient: np.ndarray,
    shots: int,
    ridge_fraction: float,
    d2_guard: float,
    target_guard: float,
) -> tuple[tuple[int, ...], dict[str, Any], list[dict[str, Any]]]:
    selected = set(int(index) for index in initial)
    history = []
    while True:
        diagnostics = _information_diagnostics(
            blocks,
            tuple(sorted(selected)),
            variables,
            hamiltonian_gradient,
            ftpbe_gradient,
            shots,
            ridge_fraction,
        )
        history.append(
            {
                "selected_count": len(selected),
                "added_frame_one_based": None,
                **diagnostics,
            }
        )
        if _guard_passes(diagnostics, d2_guard, target_guard):
            return tuple(sorted(selected)), diagnostics, history
        candidates = [
            int(row["frame_zero_based"])
            for row in rows
            if int(row["frame_zero_based"]) not in selected
        ]
        if not candidates:
            raise RuntimeError("All frames were selected before the guards passed.")
        candidate_diagnostics = []
        for index in candidates:
            trial = tuple(sorted(selected | {index}))
            trial_diagnostics = _information_diagnostics(
                blocks,
                trial,
                variables,
                hamiltonian_gradient,
                ftpbe_gradient,
                shots,
                ridge_fraction,
            )
            normalized_maximum = max(
                trial_diagnostics["trace_variance_ratio"] / d2_guard,
                trial_diagnostics["hamiltonian_variance_ratio"] / target_guard,
                trial_diagnostics["ftpbe_variance_ratio"] / target_guard,
            )
            rank_deficit = (
                trial_diagnostics["full_design_rank"]
                - trial_diagnostics["selected_design_rank"]
            )
            probability = float(rows[index]["retention_probability"])
            candidate_diagnostics.append(
                (
                    rank_deficit,
                    normalized_maximum,
                    -probability,
                    index,
                    trial_diagnostics,
                )
            )
        _, _, _, chosen, chosen_diagnostics = min(candidate_diagnostics)
        selected.add(chosen)
        history[-1]["added_frame_one_based"] = chosen + 1
        history[-1]["post_add_diagnostics"] = chosen_diagnostics


def _random_complex_guard_selection(
    size: int, is_complex: np.ndarray, seed: int
) -> tuple[int, ...]:
    complex_indices = np.flatnonzero(is_complex)
    if len(complex_indices) > size:
        raise ValueError("Requested random subset is smaller than the complex guard.")
    real_indices = np.flatnonzero(~is_complex)
    rng = np.random.default_rng(seed)
    chosen_real = rng.choice(
        real_indices, size=size - len(complex_indices), replace=False
    )
    return tuple(sorted(np.concatenate((complex_indices, chosen_real)).tolist()))


def _promoted_records(
    probe_records: Sequence[dict[str, Any]],
    selected: Sequence[int],
    oracle: AcquisitionOracle,
    topup_blocks: int,
) -> list[dict[str, Any]]:
    records = list(probe_records)
    for index in selected:
        for repeat in range(1, topup_blocks + 1):
            records.append(oracle.sample(int(index), repeat))
    return records


def _aggregate_promoted_records(
    records: Sequence[dict[str, Any]], selected: Sequence[int]
) -> list[dict[str, Any]]:
    """Combine repeated binomial counts before estimating their variances."""
    grouped: dict[int, list[dict[str, Any]]] = {int(index): [] for index in selected}
    for record in records:
        index = int(record["candidate_index"])
        if index in grouped:
            grouped[index].append(record)
    aggregated = []
    for index in selected:
        parts = grouped[int(index)]
        if not parts:
            raise RuntimeError(f"No acquisition records for frame {index}.")
        aggregated.append(
            {
                "candidate_index": int(index),
                "rotation": np.asarray(parts[0]["rotation"]),
                "pair_vectors": np.asarray(parts[0]["pair_vectors"]),
                "hits": np.sum(
                    [np.asarray(part["hits"], dtype=int) for part in parts],
                    axis=0,
                ),
                "occupations": np.concatenate(
                    [np.asarray(part["occupations"], dtype=np.uint64) for part in parts]
                ),
            }
        )
    return aggregated


def _solve_blind(
    name: str,
    acquired: Any,
    solver_args: Any,
    selection_reference: Any,
    warm_d2: np.ndarray,
    warm_gamma: np.ndarray,
    output_dir: Path,
    overwrite: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint = output_dir / "checkpoints" / f"{name}.npz"
    metadata_path = checkpoint.with_suffix(".json")
    if checkpoint.is_file() and metadata_path.is_file() and not overwrite:
        with np.load(checkpoint, allow_pickle=False) as archive:
            d2 = np.asarray(archive["d2"])
            gamma = np.asarray(archive["gamma"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"[{name}] checkpoint", flush=True)
        return d2, gamma, metadata
    print(
        f"[{name}] solving {acquired.n_shadows} x {acquired.shots_per_basis} shots",
        flush=True,
    )
    started = time.perf_counter()
    result, reconstruction = _solve_shadow_dqg(
        solver_args,
        selection_reference,
        acquired,
        warm_d2,
        warm_gamma,
    )
    metadata = {
        "case": name,
        "solver_seconds": time.perf_counter() - started,
        "status": str(result.status),
        "fit_status": str(result.fit_status),
        "weighted_fit_optimum_rmse": (
            None
            if result.weighted_fit_optimum_rmse is None
            else float(result.weighted_fit_optimum_rmse)
        ),
        "weighted_fit_final_rmse": (
            None
            if result.weighted_fit_final_rmse is None
            else float(result.weighted_fit_final_rmse)
        ),
        **reconstruction,
    }
    d2 = np.asarray(result.d2)
    gamma = np.asarray(result.gamma)
    _atomic_npz(checkpoint, d2=d2, gamma=gamma)
    _atomic_json(metadata_path, metadata)
    return d2, gamma, metadata


def _report(payload: dict[str, Any]) -> str:
    configuration = payload["configuration"]
    lines = [
        "# C2 M=50 probe-and-promote validation",
        "",
        "All 50 candidates receive a 1000-shot probe. The frozen retention model "
        "and MC-PDFT/Hamiltonian/D2 information guards are evaluated without exact "
        "M=50 RDM data, then retained frames receive 9000 additional shots. For the "
        "primary aggregated estimator, promoted probe and top-up counts are combined "
        "before binomial variances are estimated; excluded probes are screening cost "
        "and are not currently used by the uniform-shot DQG likelihood.",
        "",
        f"- Learned promotions: {configuration['learned_promotions']}",
        f"- Learned promoted frames (1-based): {configuration['learned_one_based']}",
        f"- Random-control promotions: {configuration['random_promotions']}",
        "",
        "| case | promoted | total shots | D2 error | D2/full | H (mEh) | ftPBE (mEh) | target pass |",
        "|:---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['case']} | {row['promoted_frames']} | {row['total_shots']} | "
            f"{row['d2_frobenius_error']:.6f} | {row['d2_error_ratio_to_full']:.3f} | "
            f"{row['hamiltonian_error_meh']:.3f} | {row['ftpbe_error_meh']:.3f} | "
            f"{'yes' if row['joint_target_pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The random path has the same probe cost, promotion count, top-up shots, "
            "and complex-frame guard. Therefore its difference from the learned path "
            "tests frame choice rather than budget or phase completeness.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    for name in (
        "feature_dir",
        "baseline_dir",
        "candidate_pool",
        "frozen_mo_npz",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.probe_shots < 1 or args.topup_shots % args.probe_shots:
        raise ValueError("topup-shots must be a positive multiple of probe-shots.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topup_blocks = args.topup_shots // args.probe_shots

    analysis = json.loads(
        (args.feature_dir / "analysis.json").read_text(encoding="utf-8")
    )
    model = analysis["summary"]["rank_feature_model"]
    candidate_shadows = load_blind_shadow_npz(args.candidate_pool, load_design=False)
    if candidate_shadows.n_shadows != 50:
        raise ValueError("The candidate pool must contain exactly 50 frames.")
    with np.load(args.candidate_pool, allow_pickle=False) as archive:
        is_complex = np.asarray(archive["is_complex_basis"], dtype=bool)
    with np.load(args.frozen_mo_npz, allow_pickle=False) as archive:
        frozen_mo = np.asarray(archive["mo_coeff"], dtype=float)

    selection_reference = LeakGuardReference(
        build_c2_selection_reference(1.25, basis="cc-pvtz", frozen_mo_coeff=frozen_mo)
    )
    objective = FtPBEEnergyObjective(selection_reference, grid_level=args.grid_level)
    variable_rows, variable_cols = _symmetric_d2_variables(selection_reference)
    blocks = shadow_design_blocks(
        LeakGuardShadowData.from_shadow_data(candidate_shadows),
        len(selection_reference.pairs),
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
    full_checkpoint = args.baseline_dir / "checkpoints" / "shadows_0050.npz"
    with np.load(full_checkpoint, allow_pickle=False) as archive:
        full_d2 = np.asarray(archive["d2"])
        full_gamma = np.asarray(archive["gamma"])

    acquisition_reference = build_c2_reference(
        1.25, basis="cc-pvtz", frozen_mo_coeff=frozen_mo
    )
    oracle = AcquisitionOracle(
        acquisition_reference,
        candidate_shadows.rotations,
        candidate_shadows.pair_vectors,
        args.probe_shots,
        args.sampling_seed,
    )
    print("Acquiring 50 blind probe frames...", flush=True)
    probe_records = [oracle.sample(index, 0) for index in range(50)]
    probe_shadows = _records_to_shadows(probe_records, args.probe_shots)
    probe_d2, probe_gamma, probe_metadata = _solve_blind(
        "probe_50x1000",
        probe_shadows,
        solver_args,
        selection_reference,
        full_d2,
        full_gamma,
        args.output_dir,
        args.overwrite,
    )

    probe_feature_rows = _prefix_features(
        50,
        probe_d2,
        probe_gamma,
        objective,
        selection_reference,
        probe_shadows,
        blocks,
        is_complex,
        variable_rows,
        variable_cols,
        args.information_ridge_fraction,
        None,
    )
    threshold_selected = _model_rows(
        probe_feature_rows, model, args.probability_threshold
    )
    variables = matrix_to_variable_vector(probe_d2, variable_rows, variable_cols)
    hamiltonian_gradient = _hamiltonian_gradient(
        selection_reference, variable_rows, variable_cols
    )
    ftpbe_gradient = _raw_ftpbe_gradient(
        objective,
        selection_reference,
        probe_d2,
        probe_gamma,
        variable_rows,
        variable_cols,
    )
    learned_selected, guard_diagnostics, guard_history = _guarded_selection(
        probe_feature_rows,
        threshold_selected,
        blocks,
        variables,
        hamiltonian_gradient,
        ftpbe_gradient,
        args.probe_shots,
        args.information_ridge_fraction,
        args.d2_guard,
        args.target_variance_guard,
    )
    random_selected = _random_complex_guard_selection(
        len(learned_selected), is_complex, args.random_seed
    )
    print(
        f"Promoting {len(learned_selected)} learned and random-control frames...",
        flush=True,
    )
    learned_records = _promoted_records(
        probe_records, learned_selected, oracle, topup_blocks
    )
    random_records = _promoted_records(
        probe_records, random_selected, oracle, topup_blocks
    )
    learned_shadows = None
    random_shadows = None
    learned_d2 = probe_d2
    learned_gamma = probe_gamma
    random_d2 = probe_d2
    random_gamma = probe_gamma
    learned_metadata: dict[str, Any] = {}
    random_metadata: dict[str, Any] = {}
    if not args.aggregate_only:
        learned_shadows = _records_to_shadows(learned_records, args.probe_shots)
        random_shadows = _records_to_shadows(random_records, args.probe_shots)
        learned_d2, learned_gamma, learned_metadata = _solve_blind(
            "learned_probe_promote",
            learned_shadows,
            solver_args,
            selection_reference,
            probe_d2,
            probe_gamma,
            args.output_dir,
            args.overwrite,
        )
        random_d2, random_gamma, random_metadata = _solve_blind(
            "random_complex_guard_probe_promote",
            random_shadows,
            solver_args,
            selection_reference,
            probe_d2,
            probe_gamma,
            args.output_dir,
            args.overwrite,
        )
    aggregate_shots = args.probe_shots + args.topup_shots
    learned_aggregated_records = _aggregate_promoted_records(
        learned_records, learned_selected
    )
    random_aggregated_records = _aggregate_promoted_records(
        random_records, random_selected
    )
    learned_aggregated_shadows = _records_to_shadows(
        learned_aggregated_records, aggregate_shots
    )
    random_aggregated_shadows = _records_to_shadows(
        random_aggregated_records, aggregate_shots
    )
    learned_aggregated_d2, learned_aggregated_gamma, learned_aggregated_metadata = (
        _solve_blind(
            "learned_probe_promote_aggregated",
            learned_aggregated_shadows,
            solver_args,
            selection_reference,
            learned_d2,
            learned_gamma,
            args.output_dir,
            args.overwrite,
        )
    )
    random_aggregated_d2, random_aggregated_gamma, random_aggregated_metadata = (
        _solve_blind(
            "random_complex_guard_probe_promote_aggregated",
            random_aggregated_shadows,
            solver_args,
            selection_reference,
            random_d2,
            random_gamma,
            args.output_dir,
            args.overwrite,
        )
    )
    paired_full_indices = tuple(range(50))
    paired_full_records = _promoted_records(
        probe_records, paired_full_indices, oracle, topup_blocks
    )
    paired_full_aggregated_records = _aggregate_promoted_records(
        paired_full_records, paired_full_indices
    )
    paired_full_shadows = _records_to_shadows(
        paired_full_aggregated_records, aggregate_shots
    )
    paired_full_d2, paired_full_gamma, paired_full_metadata = _solve_blind(
        "paired_full_50x10000_aggregated",
        paired_full_shadows,
        solver_args,
        selection_reference,
        full_d2,
        full_gamma,
        args.output_dir,
        args.overwrite,
    )
    guard_path_cases: list[tuple[str, np.ndarray, np.ndarray, Any, dict[str, Any]]] = []
    if args.audit_guard_path:
        path_selected = set(int(index) for index in threshold_selected)
        path_warm_d2 = probe_d2
        path_warm_gamma = probe_gamma
        for history_row in guard_history:
            added = history_row["added_frame_one_based"]
            if added is None:
                continue
            path_selected.add(int(added) - 1)
            if len(path_selected) >= len(learned_selected):
                continue
            path_tuple = tuple(sorted(path_selected))
            path_records = _promoted_records(
                probe_records, path_tuple, oracle, topup_blocks
            )
            path_aggregated_records = _aggregate_promoted_records(
                path_records, path_tuple
            )
            path_shadows = _records_to_shadows(path_aggregated_records, aggregate_shots)
            name = f"learned_guard_path_k{len(path_tuple):02d}_aggregated"
            path_d2, path_gamma, path_metadata = _solve_blind(
                name,
                path_shadows,
                solver_args,
                selection_reference,
                path_warm_d2,
                path_warm_gamma,
                args.output_dir,
                args.overwrite,
            )
            path_warm_d2 = path_d2
            path_warm_gamma = path_gamma
            guard_path_cases.append(
                (
                    name,
                    path_d2,
                    path_gamma,
                    path_shadows,
                    {
                        **path_metadata,
                        "promoted_frames": len(path_tuple),
                        "total_shots": len(path_records) * args.probe_shots,
                        "likelihood_shots": (
                            len(path_aggregated_records) * aggregate_shots
                        ),
                        "unused_probe_shots": (
                            (50 - len(path_tuple)) * args.probe_shots
                        ),
                        "count_handling": (
                            "aggregate repeated hits before variance estimation"
                        ),
                        "posthoc_guard_path_audit": True,
                    },
                )
            )

    exact_ftpbe = float(
        objective.evaluate(
            acquisition_reference.exact_d2,
            acquisition_reference.exact_gamma,
            gradient=False,
        ).total_energy
    )
    cases = [
        (
            "archived_full_m50_different_seed",
            full_d2,
            full_gamma,
            candidate_shadows,
            {"promoted_frames": 50, "total_shots": 500_000},
        ),
        (
            "paired_full_50x10000_aggregated",
            paired_full_d2,
            paired_full_gamma,
            paired_full_shadows,
            {
                **paired_full_metadata,
                "promoted_frames": 50,
                "total_shots": 500_000,
                "likelihood_shots": 500_000,
                "count_handling": (
                    "aggregate repeated hits before variance estimation"
                ),
            },
        ),
        (
            "probe_50x1000",
            probe_d2,
            probe_gamma,
            probe_shadows,
            {
                **probe_metadata,
                "promoted_frames": 0,
                "total_shots": 50 * args.probe_shots,
            },
        ),
    ]
    if not args.aggregate_only:
        cases.extend(
            [
                (
                    "learned_probe_promote",
                    learned_d2,
                    learned_gamma,
                    learned_shadows,
                    {
                        **learned_metadata,
                        "promoted_frames": len(learned_selected),
                        "total_shots": len(learned_records) * args.probe_shots,
                    },
                ),
                (
                    "random_complex_guard_probe_promote",
                    random_d2,
                    random_gamma,
                    random_shadows,
                    {
                        **random_metadata,
                        "promoted_frames": len(random_selected),
                        "total_shots": len(random_records) * args.probe_shots,
                    },
                ),
            ]
        )
    cases.extend(
        [
            (
                "learned_probe_promote_aggregated",
                learned_aggregated_d2,
                learned_aggregated_gamma,
                learned_aggregated_shadows,
                {
                    **learned_aggregated_metadata,
                    "promoted_frames": len(learned_selected),
                    "total_shots": len(learned_records) * args.probe_shots,
                    "likelihood_shots": (
                        len(learned_aggregated_records) * aggregate_shots
                    ),
                    "unused_probe_shots": (
                        (50 - len(learned_selected)) * args.probe_shots
                    ),
                    "count_handling": "aggregate repeated hits before variance estimation",
                },
            ),
            (
                "random_complex_guard_probe_promote_aggregated",
                random_aggregated_d2,
                random_aggregated_gamma,
                random_aggregated_shadows,
                {
                    **random_aggregated_metadata,
                    "promoted_frames": len(random_selected),
                    "total_shots": len(random_records) * args.probe_shots,
                    "likelihood_shots": (
                        len(random_aggregated_records) * aggregate_shots
                    ),
                    "unused_probe_shots": (
                        (50 - len(random_selected)) * args.probe_shots
                    ),
                    "count_handling": "aggregate repeated hits before variance estimation",
                },
            ),
        ]
    )
    cases.extend(guard_path_cases)
    rows = []
    full_error = float(np.linalg.norm(paired_full_d2 - acquisition_reference.exact_d2))
    for name, d2, gamma, acquired, metadata in cases:
        metrics = _score_dqg(
            SimpleNamespace(d2=d2, gamma=gamma),
            acquisition_reference,
            objective,
            exact_ftpbe,
            acquired,
        )
        d2_ratio = metrics["d2_frobenius_error"] / full_error
        rows.append(
            {
                **metadata,
                "case": name,
                **metrics,
                "d2_error_ratio_to_full": d2_ratio,
                "joint_target_pass": bool(
                    d2_ratio <= args.d2_guard
                    and max(
                        metrics["hamiltonian_error_meh"],
                        metrics["ftpbe_error_meh"],
                    )
                    <= args.energy_target_meh
                ),
            }
        )
    payload = {
        "configuration": {
            "probe_shots": args.probe_shots,
            "topup_shots": args.topup_shots,
            "probe_frames": 50,
            "probability_threshold": args.probability_threshold,
            "threshold_selected": len(threshold_selected),
            "learned_promotions": len(learned_selected),
            "learned_one_based": [index + 1 for index in learned_selected],
            "random_promotions": len(random_selected),
            "random_one_based": [index + 1 for index in random_selected],
            "complex_guard": "all complex frames promoted",
            "d2_guard": args.d2_guard,
            "target_variance_guard": args.target_variance_guard,
            "energy_target_meh": args.energy_target_meh,
            "sampling_seed": args.sampling_seed,
            "random_seed": args.random_seed,
            "promoted_probe_data_in_aggregated_likelihood": True,
            "excluded_probe_data_in_aggregated_likelihood": False,
            "aggregate_only": args.aggregate_only,
            "audit_guard_path": args.audit_guard_path,
            "selection_uses_exact_m50": False,
            "selection_model_uses_prior_oracle_labels": True,
            "exact_state_use": "bitstring simulation and posthoc scoring only",
            "primary_d2_baseline": (
                "paired full 50x10000 using the same sampled bitstrings"
            ),
        },
        "guard_diagnostics": guard_diagnostics,
        "guard_history": guard_history,
        "rows": rows,
    }
    _write_csv(args.output_dir / "probe_frame_scores.csv", probe_feature_rows)
    _write_csv(args.output_dir / "validation.csv", rows)
    _atomic_json(args.output_dir / "analysis.json", payload)
    (args.output_dir / "C2_M50_PROBE_RULE_VALIDATION.md").write_text(
        _report(payload), encoding="utf-8"
    )
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
