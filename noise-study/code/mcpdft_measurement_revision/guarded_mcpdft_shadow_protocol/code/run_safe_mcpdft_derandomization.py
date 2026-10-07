#!/usr/bin/env python3
"""Run FCI-free guarded MC-PDFT derandomization on a raw-shadow pool.

Selection reads only rotations, already acquired outcomes, the molecular
Hamiltonian, and the ftPBE functional.  Exact FCI RDMs are used only after the
selection/reconstruction path has been frozen, to produce posthoc benchmark
metrics alongside the existing random-prefix baseline.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/safe-mcpdft-derandomization")

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
QACSE_ROOT = (
    WORKSPACE_ROOT / "QACSE" / "ConstrainedShadowTomography" / "python_reproduction"
)
for path in (HERE, PROJECT_ROOT, QACSE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
LOCAL_VENDOR = HERE / "vendor"
if str(LOCAL_VENDOR) in sys.path:
    sys.path.remove(str(LOCAL_VENDOR))
sys.path.insert(0, str(LOCAL_VENDOR))

from constrained_shadow import (  # noqa: E402
    AffineRDMObjective,
    PairVectorDesign,
    build_n2_reference,
    rdm_energy,
    solve_dqg_sdp,
)
from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    LeakGuardShadowData,
    acquire_shadow_bases,
    build_c2_reference,
    build_c2_selection_reference,
    build_n2_selection_reference,
    build_weighted_fit_statistics,
    contraction_gamma_gradient_vector,
    guarded_hamiltonian_mcpdft_choice,
    guarded_mcpdft_choice,
    guarded_target_subspace_choice,
    load_blind_shadow_npz,
    matrix_to_variable_vector,
    remove_parallel_component,
    shadow_design_blocks,
    symmetric_matrix_gradient_vector,
    validate_shadow_archive_structure,
)
from mcpdft_selector import FtPBEEnergyObjective, _symmetric_d2_variables  # noqa: E402
from ridge_selector import (  # noqa: E402
    cross_validate_ridge,
    weighted_prediction_rmse,
    weighted_ridge_target,
)
from run_sweep import (  # noqa: E402
    METHOD_BASELINE,
    _atomic_json,
    _atomic_npz,
)


METHOD_RANDOM = "random prefix weighted-LS + DQG"
METHOD_DESIGN = "FCI-free D-optimal derandomization + DQG"
METHOD_MCPDFT = "FCI-free guarded MC-PDFT derandomization + DQG"
POLICY_DESIGN = "design_only"
POLICY_MCPDFT = "guarded_mcpdft"
POLICY_RANDOM = "random_prefix"
RECONSTRUCTION_ENERGY = "energy"
RECONSTRUCTION_RIDGE = "cv_ridge_closest"
RECONSTRUCTION_WEIGHTED_LS = "physical_weighted_ls"
FORBIDDEN_SELECTION_ROW_FIELDS = {
    "d2_frobenius_error",
    "d2_normalized_frobenius_error",
    "energy_error",
    "exact_sampling_rmse",
    "gamma_frobenius_error",
    "joint_target_loss_posthoc",
    "meets_both_targets_posthoc",
    "weighted_fit_reference_rmse",
}


def _parse_policies(value: str) -> tuple[str, ...]:
    allowed = {POLICY_DESIGN, POLICY_MCPDFT}
    result = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    invalid = sorted(set(result).difference(allowed))
    if not result or invalid:
        raise argparse.ArgumentTypeError(
            f"Policies must be drawn from {sorted(allowed)}; invalid={invalid}."
        )
    return result


def _parse_float_grid(value: str) -> tuple[float, ...]:
    try:
        grid = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated floats.") from error
    if not grid or min(grid) < 0.0:
        raise argparse.ArgumentTypeError(
            "Ridge values must be nonnegative."
        )
    return grid


def _parse_orbital_signs(value: str) -> tuple[float, ...]:
    try:
        signs = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected comma-separated +/-1 values.") from error
    if not signs or any(abs(sign) != 1.0 for sign in signs):
        raise argparse.ArgumentTypeError("Orbital signs must all be +1 or -1.")
    return signs


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("n2", "c2"), default="n2")
    parser.add_argument("--basis")
    parser.add_argument("--bond-length", type=float)
    parser.add_argument("--active-electrons", type=int)
    parser.add_argument("--active-orbitals", type=int)
    parser.add_argument(
        "--active-orbital-signs",
        type=_parse_orbital_signs,
        help="Optional active-MO gauge matching a pre-generated shadow archive.",
    )
    parser.add_argument("--shots-per-shadow", type=int, default=1000)
    parser.add_argument(
        "--shadow-input-npz",
        type=Path,
        default=HERE / "results" / "cas66_shadows_1_30" / "raw_aer_shadows.npz",
    )
    parser.add_argument(
        "--frozen-mo-npz",
        type=Path,
        help=(
            "C2 archive containing the full generation-time MO coefficient "
            "matrix under the mo_coeff key."
        ),
    )
    parser.add_argument(
        "--original-results-dir",
        type=Path,
        default=HERE / "results" / "cas66_shadows_1_30",
    )
    parser.add_argument(
        "--random-baseline",
        choices=("auto", "native", "existing", "none"),
        default="auto",
        help=(
            "Run the random prefix through the same reconstruction path, or reuse "
            "the historical N2 baseline. Auto selects native for C2."
        ),
    )
    parser.add_argument("--max-selected", type=int, default=18)
    parser.add_argument(
        "--policies",
        type=_parse_policies,
        default=(POLICY_DESIGN, POLICY_MCPDFT),
    )
    parser.add_argument("--design-guard-fraction", type=float, default=0.8)
    parser.add_argument("--hamiltonian-guard-fraction", type=float, default=0.0)
    parser.add_argument(
        "--mcpdft-target",
        choices=("ontop_d2", "total_contracted", "field_response"),
        default="ontop_d2",
    )
    parser.add_argument("--field-response-modes", type=int, default=24)
    parser.add_argument("--field-response-capture", type=float, default=0.999)
    parser.add_argument("--information-ridge-fraction", type=float, default=1e-3)
    parser.add_argument("--probability-floor", type=float, default=0.01)
    parser.add_argument("--remove-hamiltonian-component", action="store_true", default=True)
    parser.add_argument("--keep-hamiltonian-component", action="store_false", dest="remove_hamiltonian_component")
    parser.add_argument("--solver", default="SCS")
    parser.add_argument("--solver-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=3000)
    parser.add_argument(
        "--symmetry-blocked-psd",
        action="store_true",
        help="Use point-group/spin blocks for the D/Q/G positive-semidefinite cones.",
    )
    parser.add_argument(
        "--solver-threads",
        type=int,
        help="Optional solver thread limit (used by MOSEK).",
    )
    parser.add_argument(
        "--weighted-fit-rmse-cap",
        type=float,
        help=(
            "Predeclared Jeffreys-weighted discrepancy radius. When supplied, "
            "DQG is solved once instead of first solving a weighted-LS SDP."
        ),
    )
    parser.add_argument(
        "--reconstruction-objective",
        choices=(
            RECONSTRUCTION_ENERGY,
            RECONSTRUCTION_RIDGE,
            RECONSTRUCTION_WEIGHTED_LS,
        ),
        default=RECONSTRUCTION_ENERGY,
        help=(
            "Final selector inside the DQG/discrepancy region. The CV-ridge "
            "target uses acquired shadow data only."
        ),
    )
    parser.add_argument(
        "--ridge-grid",
        type=_parse_float_grid,
        default=(0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0),
    )
    parser.add_argument("--ridge-folds", type=int, default=5)
    parser.add_argument("--fallback-ridge", type=float, default=1e-2)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--polyak-window", type=int, default=4)
    parser.add_argument("--energy-target", type=float, default=1e-3)
    parser.add_argument("--d2-target", type=float, default=6e-3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "results" / "safe_mcpdft_derandomization_cas66",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Freeze the measurement path without constructing an FCI benchmark.",
    )
    return parser.parse_args(argv)


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": "FCI-free guarded MC-PDFT c-optimal shadow design",
        "selection_inputs": [
            "candidate rotations/design matrices",
            "already acquired raw frequencies",
            "current DQG reconstruction",
            "Hamiltonian integrals",
            "ftPBE total-energy gradient",
        ],
        "selection_excludes": [
            "exact FCI D2",
            "exact FCI gamma",
            "candidate-basis raw outcomes before acquisition",
            "posthoc energy/D2 errors",
        ],
        "selection_reference": (
            "RHF orbitals and active-space Hamiltonian only; no CASCI/FCI solve"
        ),
        "system": args.system.upper(),
        "basis": args.basis,
        "active_space_projection": (
            "frozen-core effective Hamiltonian and active-space 1-/2-RDM; "
            "no full AO-space RDM reconstruction"
        ),
        "archive_exact_arrays_loaded_for_selection": False,
        "polyak_average_predeclared": True,
        "posthoc_fci_metrics_only": True,
        "shadow_input_npz": str(args.shadow_input_npz.resolve()),
        "frozen_mo_npz": (
            str(args.frozen_mo_npz.resolve())
            if args.frozen_mo_npz is not None
            else None
        ),
        "random_baseline": args.random_baseline,
        "original_results_dir": (
            str(args.original_results_dir.resolve())
            if args.random_baseline == "existing"
            else None
        ),
        "active_space": [args.active_electrons, args.active_orbitals],
        "active_orbital_signs": (
            list(args.active_orbital_signs)
            if args.active_orbital_signs is not None
            else None
        ),
        "shots_per_shadow": args.shots_per_shadow,
        "max_selected": args.max_selected,
        "policies": list(args.policies),
        "design_guard_fraction": args.design_guard_fraction,
        "hamiltonian_guard_fraction": args.hamiltonian_guard_fraction,
        "mcpdft_target": args.mcpdft_target,
        "field_response_modes": args.field_response_modes,
        "field_response_capture": args.field_response_capture,
        "information_ridge_fraction": args.information_ridge_fraction,
        "probability_floor": args.probability_floor,
        "remove_hamiltonian_component": args.remove_hamiltonian_component,
        "solver": args.solver,
        "solver_tolerance": args.solver_tolerance,
        "max_iterations": args.max_iterations,
        "solver_threads": args.solver_threads,
        "symmetry_blocked_psd": args.symmetry_blocked_psd,
        "weighted_fit_rmse_cap": args.weighted_fit_rmse_cap,
        "dqg_reconstruction": (
            f"{args.reconstruction_objective} inside "
            + (
                "a single-stage weighted discrepancy cap"
                if args.weighted_fit_rmse_cap is not None
                else "the two-stage optimum weighted-fit region"
            )
        ),
        "reconstruction_objective": args.reconstruction_objective,
        "ridge_grid": list(args.ridge_grid),
        "ridge_folds": args.ridge_folds,
        "fallback_ridge": args.fallback_ridge,
        "grid_level": args.grid_level,
        "polyak_window": args.polyak_window,
        "targets_posthoc": {
            "energy_error_eh": args.energy_target,
            "normalized_d2_error": args.d2_target,
        },
    }


def _validate_selection_rows_blind(
    selection_rows: dict[str, list[dict[str, Any]]],
) -> None:
    """Reject truth-derived fields before persisting the frozen path."""

    for policy, rows in selection_rows.items():
        for row in rows:
            forbidden = {
                key
                for key in row
                if key in FORBIDDEN_SELECTION_ROW_FIELDS
                or key.startswith("exact_")
                or "posthoc" in key
            }
            if forbidden:
                names = ", ".join(sorted(forbidden))
                raise RuntimeError(
                    f"Selection row {policy}[m={row.get('shadows')}] contains "
                    f"truth-derived fields: {names}."
                )


def _initial_result(args: argparse.Namespace, reference: Any) -> Any:
    return solve_dqg_sdp(
        reference,
        solver=args.solver,
        tolerance=args.solver_tolerance,
        max_iterations=args.max_iterations,
        solver_threads=getattr(args, "solver_threads", None),
        verbose=args.verbose_solver,
        positivity_conditions="DQG",
        symmetry_blocked_psd=bool(
            getattr(args, "symmetry_blocked_psd", False)
        ),
    )


def _solve_shadow_dqg(
    args: argparse.Namespace,
    reference: Any,
    acquired: Any,
    initial_d2: np.ndarray,
    initial_gamma: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """Reconstruct one prefix using the predeclared common fit rule."""

    kwargs: dict[str, Any] = {}
    if args.weighted_fit_rmse_cap is not None:
        kwargs["weighted_fit_rmse_cap"] = args.weighted_fit_rmse_cap
    diagnostics: dict[str, Any] = {
        "reconstruction_objective": args.reconstruction_objective,
    }
    compact_fit = isinstance(acquired.design, PairVectorDesign)
    if compact_fit and args.reconstruction_objective == RECONSTRUCTION_WEIGHTED_LS:
        raise ValueError(
            "Implicit pair-vector designs do not support the explicit affine "
            "weighted-LS reconstruction objective."
        )
    if args.reconstruction_objective == RECONSTRUCTION_RIDGE:
        if acquired.n_shadows >= 2:
            cross_validation = cross_validate_ridge(
                reference,
                acquired,
                args.ridge_grid,
                folds=args.ridge_folds,
            )
            ridge = cross_validation.selected_ridge
            diagnostics["ridge_cv_rmse"] = float(
                np.sqrt(cross_validation.mean_scores[ridge])
            )
            diagnostics["ridge_cv_folds"] = cross_validation.folds
        else:
            ridge = args.fallback_ridge
            diagnostics["ridge_cv_rmse"] = None
            diagnostics["ridge_cv_folds"] = 0
        target = weighted_ridge_target(reference, acquired, ridge)
        kwargs["selection_objective"] = "closest_d2"
        kwargs["selection_d2_target"] = target.d2
        diagnostics.update(
            {
                "selected_ridge": ridge,
                "ridge_target_weighted_fit_rmse": target.weighted_fit_rmse,
                "ridge_target_effective_ridge": target.effective_ridge,
            }
        )
    elif args.reconstruction_objective == RECONSTRUCTION_WEIGHTED_LS:
        smoothed = (acquired.hits.astype(float) + 0.5) / (
            acquired.shots_per_basis + 1.0
        )
        standard_errors = np.sqrt(
            np.maximum(
                smoothed * (1.0 - smoothed) / acquired.shots_per_basis,
                1e-12,
            )
        )
        kwargs["selection_objective"] = "affine_least_squares"
        kwargs["affine_objective"] = AffineRDMObjective(
            d2_map=acquired.design,
            offset=-np.asarray(acquired.values, dtype=float),
            scales=standard_errors,
            name="jeffreys_weighted_shadow_fit",
        )
    shadow_data = acquired
    if compact_fit:
        variable_rows, variable_cols = _symmetric_d2_variables(reference)
        statistics = build_weighted_fit_statistics(
            acquired, variable_rows, variable_cols
        )
        kwargs["weighted_fit_statistics"] = statistics
        diagnostics.update(
            {
                "weighted_fit_representation": "chunked_tsqr",
                "weighted_fit_variables": len(variable_rows),
                "weighted_fit_factor_shape": list(
                    statistics.residual_factor.shape
                ),
            }
        )
        shadow_data = None
    result = solve_dqg_sdp(
        reference,
        shadow_data=shadow_data,
        solver=args.solver,
        tolerance=args.solver_tolerance,
        max_iterations=args.max_iterations,
        solver_threads=getattr(args, "solver_threads", None),
        verbose=args.verbose_solver,
        positivity_conditions="DQG",
        symmetry_blocked_psd=bool(
            getattr(args, "symmetry_blocked_psd", False)
        ),
        weighted_shadow_fit=True,
        initial_d2=initial_d2,
        initial_gamma=initial_gamma,
        **kwargs,
    )
    return result, diagnostics


def _checkpoint_paths(output: Path, policy: str, count: int) -> tuple[Path, Path]:
    stem = output / "checkpoints" / policy / f"shadows_{count:02d}"
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def _load_checkpoint(
    output: Path, policy: str, count: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray] | None:
    json_path, npz_path = _checkpoint_paths(output, policy, count)
    if not json_path.exists() or not npz_path.exists():
        return None
    row = json.loads(json_path.read_text())
    with np.load(npz_path) as archive:
        return row, np.asarray(archive["d2"]), np.asarray(archive["gamma"])


def _write_checkpoint(
    output: Path,
    policy: str,
    row: dict[str, Any],
    d2: np.ndarray,
    gamma: np.ndarray,
) -> None:
    json_path, npz_path = _checkpoint_paths(output, policy, int(row["shadows"]))
    _atomic_json(json_path, row)
    _atomic_npz(npz_path, d2=d2, gamma=gamma)


def _row(
    args: argparse.Namespace,
    policy: str,
    count: int,
    selected_indices: tuple[int, ...],
    choice: Any,
    result: Any,
    selected_shadows: Any,
    objective: FtPBEEnergyObjective,
    *,
    seconds: float,
    reconstruction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation = objective.evaluate(result.d2, result.gamma, gradient=False)
    return {
        "policy": policy,
        "method": METHOD_DESIGN if policy == POLICY_DESIGN else METHOD_MCPDFT,
        "shadows": count,
        "selected_index": choice.selected_index,
        "selected_indices": ",".join(map(str, selected_indices)),
        "design_only_index_at_selection": choice.design_only_index,
        "hamiltonian_only_index_at_selection": getattr(
            choice, "hamiltonian_only_index", None
        ),
        "selected_d_optimal_gain": choice.selected.d_optimal_gain,
        "maximum_d_optimal_gain": choice.maximum_d_optimal_gain,
        "selected_design_gain_fraction": (
            choice.selected.d_optimal_gain / choice.maximum_d_optimal_gain
            if choice.maximum_d_optimal_gain > 0.0
            else 1.0
        ),
        "selected_mcpdft_variance_reduction": (
            choice.selected.mcpdft_variance_reduction
        ),
        "design_only_mcpdft_variance_reduction": (
            choice.design_only.mcpdft_variance_reduction
        ),
        "mcpdft_gain_over_design": (
            choice.selected.mcpdft_variance_reduction
            - choice.design_only.mcpdft_variance_reduction
        ),
        "selected_mcpdft_fractional_reduction": (
            choice.selected.mcpdft_fractional_reduction
        ),
        "selected_hamiltonian_variance_reduction": getattr(
            choice.selected, "hamiltonian_variance_reduction", None
        ),
        "maximum_hamiltonian_variance_reduction": getattr(
            choice, "maximum_hamiltonian_variance_reduction", None
        ),
        "selected_hamiltonian_gain_fraction": (
            choice.selected.hamiltonian_variance_reduction
            / choice.maximum_hamiltonian_variance_reduction
            if getattr(choice, "maximum_hamiltonian_variance_reduction", 0.0) > 0.0
            else None
        ),
        "selected_hamiltonian_fractional_reduction": getattr(
            choice.selected, "hamiltonian_fractional_reduction", None
        ),
        "hamiltonian_only_mcpdft_variance_reduction": getattr(
            getattr(choice, "hamiltonian_only", None),
            "mcpdft_variance_reduction",
            None,
        ),
        "mcpdft_gain_over_hamiltonian_safe": (
            choice.selected.mcpdft_variance_reduction
            - getattr(
                getattr(choice, "hamiltonian_only", None),
                "mcpdft_variance_reduction",
                choice.selected.mcpdft_variance_reduction,
            )
        ),
        "current_hamiltonian_standard_error": getattr(
            choice, "current_hamiltonian_standard_error", None
        ),
        "selected_hamiltonian_standard_error": getattr(
            choice, "selected_hamiltonian_standard_error", None
        ),
        "current_mcpdft_standard_error": choice.current_mcpdft_standard_error,
        "selected_mcpdft_standard_error": choice.selected_mcpdft_standard_error,
        "current_d2_trace_standard_error": getattr(
            choice, "current_d2_trace_standard_error", None
        ),
        "selected_d2_trace_standard_error": getattr(
            choice, "selected_d2_trace_standard_error", None
        ),
        "weighted_fit_rmse": weighted_prediction_rmse(selected_shadows, result.d2),
        "ftpbe_total_energy": evaluation.total_energy,
        "ftpbe_minimum_density": evaluation.minimum_density,
        "ftpbe_minimum_on_top_pair_density": (
            evaluation.minimum_on_top_pair_density
        ),
        "energy": result.energy,
        "status": result.status,
        "fit_status": result.fit_status,
        "seconds": seconds,
        **(reconstruction or {}),
    }


def _random_row(
    args: argparse.Namespace,
    count: int,
    result: Any,
    selected_shadows: Any,
    objective: FtPBEEnergyObjective,
    *,
    seconds: float,
    reconstruction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evaluation = objective.evaluate(result.d2, result.gamma, gradient=False)
    fit = (
        f"weighted discrepancy-cap {args.weighted_fit_rmse_cap:g}"
        if args.weighted_fit_rmse_cap is not None
        else "weighted-LS"
    )
    return {
        "policy": POLICY_RANDOM,
        "method": f"random prefix {fit} + DQG",
        "shadows": count,
        "selected_index": count - 1,
        "selected_indices": ",".join(map(str, range(count))),
        "weighted_fit_rmse": weighted_prediction_rmse(
            selected_shadows, result.d2
        ),
        "ftpbe_total_energy": evaluation.total_energy,
        "ftpbe_minimum_density": evaluation.minimum_density,
        "ftpbe_minimum_on_top_pair_density": (
            evaluation.minimum_on_top_pair_density
        ),
        "energy": result.energy,
        "status": result.status,
        "fit_status": result.fit_status,
        "seconds": seconds,
        **(reconstruction or {}),
    }


def _run_random_policy(
    args: argparse.Namespace,
    reference: Any,
    all_shadows: Any,
    objective: FtPBEEnergyObjective,
    initial: Any,
) -> list[dict[str, Any]]:
    """Run the random prefix with exactly the same DQG fit rule."""

    current_d2 = initial.d2
    current_gamma = initial.gamma
    output_rows: list[dict[str, Any]] = []
    for count in range(1, args.max_selected + 1):
        if not args.overwrite:
            checkpoint = _load_checkpoint(args.output_dir, POLICY_RANDOM, count)
            if checkpoint is not None:
                row, current_d2, current_gamma = checkpoint
                output_rows.append(row)
                print(
                    f"[{POLICY_RANDOM}] m={count:02d}: checkpoint, "
                    f"E_H={row['energy']:.9f}, "
                    f"E_ftPBE={row['ftpbe_total_energy']:.9f}",
                    flush=True,
                )
                continue

        acquired = acquire_shadow_bases(all_shadows, tuple(range(count)))
        print(
            f"[{POLICY_RANDOM}] m={count:02d}: add={count - 1:02d}",
            flush=True,
        )
        started = time.perf_counter()
        result, reconstruction = _solve_shadow_dqg(
            args, reference, acquired, current_d2, current_gamma
        )
        seconds = time.perf_counter() - started
        row = _random_row(
            args,
            count,
            result,
            acquired,
            objective,
            seconds=seconds,
            reconstruction=reconstruction,
        )
        _write_checkpoint(
            args.output_dir, POLICY_RANDOM, row, result.d2, result.gamma
        )
        output_rows.append(row)
        current_d2 = result.d2
        current_gamma = result.gamma
        print(
            f"[{POLICY_RANDOM}] m={count:02d}: E_H={row['energy']:.9f}, "
            f"E_ftPBE={row['ftpbe_total_energy']:.9f}, "
            f"fit={row['weighted_fit_rmse']:.3f}",
            flush=True,
        )
        gc.collect()
    return output_rows


def _run_policy(
    args: argparse.Namespace,
    policy: str,
    reference: Any,
    all_shadows: Any,
    blocks: tuple[np.ndarray, ...],
    rows: np.ndarray,
    cols: np.ndarray,
    objective: FtPBEEnergyObjective,
    initial: Any,
    field_modes: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    current_d2 = initial.d2
    current_gamma = initial.gamma
    selected: tuple[int, ...] = ()
    output_rows: list[dict[str, Any]] = []

    for count in range(1, args.max_selected + 1):
        if not args.overwrite:
            checkpoint = _load_checkpoint(args.output_dir, policy, count)
            if checkpoint is not None:
                row, current_d2, current_gamma = checkpoint
                selected = tuple(int(item) for item in row["selected_indices"].split(","))
                output_rows.append(row)
                print(
                    f"[{policy}] m={count:02d}: checkpoint, "
                    f"E_H={row['energy']:.9f}, "
                    f"E_ftPBE={row['ftpbe_total_energy']:.9f}",
                    flush=True,
                )
                continue

        variables = matrix_to_variable_vector(current_d2, rows, cols)
        if args.mcpdft_target == "field_response" and policy != POLICY_DESIGN:
            if field_modes is None:
                raise RuntimeError("The frozen ftPBE field modes were not built.")
            choice = guarded_target_subspace_choice(
                blocks,
                selected,
                variables,
                field_modes,
                args.shots_per_shadow,
                design_guard_fraction=args.design_guard_fraction,
                ridge_fraction=args.information_ridge_fraction,
                probability_floor=args.probability_floor,
            )
        else:
            evaluation = objective.evaluate(
                current_d2, current_gamma, gradient=True
            )
            gradient = symmetric_matrix_gradient_vector(
                evaluation.d2_gradient, rows, cols
            )
            if args.mcpdft_target == "total_contracted":
                gradient += contraction_gamma_gradient_vector(
                    evaluation.gamma_gradient,
                    reference.pairs,
                    rows,
                    cols,
                    reference.n_electrons,
                )
            hamiltonian_gradient = symmetric_matrix_gradient_vector(
                reference.two_body, rows, cols
            )
            hamiltonian_gradient += contraction_gamma_gradient_vector(
                reference.one_body,
                reference.pairs,
                rows,
                cols,
                reference.n_electrons,
            )
            if args.remove_hamiltonian_component:
                gradient = remove_parallel_component(
                    gradient, hamiltonian_gradient
                )

        if policy == POLICY_DESIGN:
            choice = guarded_mcpdft_choice(
                blocks,
                selected,
                variables,
                gradient,
                args.shots_per_shadow,
                design_guard_fraction=1.0,
                ridge_fraction=args.information_ridge_fraction,
                probability_floor=args.probability_floor,
            )
        elif args.mcpdft_target == "field_response":
            pass
        elif args.hamiltonian_guard_fraction > 0.0:
            choice = guarded_hamiltonian_mcpdft_choice(
                blocks,
                selected,
                variables,
                hamiltonian_gradient,
                gradient,
                args.shots_per_shadow,
                design_guard_fraction=args.design_guard_fraction,
                hamiltonian_guard_fraction=args.hamiltonian_guard_fraction,
                ridge_fraction=args.information_ridge_fraction,
                probability_floor=args.probability_floor,
            )
        else:
            choice = guarded_mcpdft_choice(
                blocks,
                selected,
                variables,
                gradient,
                args.shots_per_shadow,
                design_guard_fraction=args.design_guard_fraction,
                ridge_fraction=args.information_ridge_fraction,
                probability_floor=args.probability_floor,
            )
        new_selected = selected + (choice.selected_index,)
        acquired = acquire_shadow_bases(all_shadows, new_selected)
        print(
            f"[{policy}] m={count:02d}: add={choice.selected_index:02d}, "
            f"D-gain={choice.selected.d_optimal_gain:.3f}, "
            f"ftPBE gain={choice.selected.mcpdft_fractional_reduction:.3%}",
            flush=True,
        )
        started = time.perf_counter()
        result, reconstruction = _solve_shadow_dqg(
            args, reference, acquired, current_d2, current_gamma
        )
        seconds = time.perf_counter() - started
        row = _row(
            args,
            policy,
            count,
            new_selected,
            choice,
            result,
            acquired,
            objective,
            seconds=seconds,
            reconstruction=reconstruction,
        )
        _write_checkpoint(
            args.output_dir, policy, row, result.d2, result.gamma
        )
        output_rows.append(row)
        selected = new_selected
        current_d2 = result.d2
        current_gamma = result.gamma
        print(
            f"[{policy}] m={count:02d}: E_H={row['energy']:.9f}, "
            f"E_ftPBE={row['ftpbe_total_energy']:.9f}, "
            f"fit={row['weighted_fit_rmse']:.3f}",
            flush=True,
        )
        gc.collect()
    return output_rows


def _load_random_rows_posthoc(
    args: argparse.Namespace,
    objective: FtPBEEnergyObjective,
    exact_ftpbe: float,
) -> list[dict[str, Any]]:
    summary = json.loads((args.original_results_dir / "summary.json").read_text())
    full = [
        row
        for row in summary["rows"]
        if row["method"] == METHOD_BASELINE
        and int(row["shadows"]) <= args.max_selected
    ]
    full.sort(key=lambda row: int(row["shadows"]))
    rows = []
    for source in full:
        count = int(source["shadows"])
        checkpoint = (
            args.original_results_dir / "checkpoints" / f"shadows_{count:02d}.npz"
        )
        with np.load(checkpoint) as archive:
            d2 = np.asarray(archive["baseline_d2"])
            gamma = np.asarray(archive["baseline_gamma"])
        evaluation = objective.evaluate(d2, gamma, gradient=False)
        rows.append(
            {
                **source,
                "policy": "random_prefix",
                "method": METHOD_RANDOM,
                "selected_index": count - 1,
                "selected_indices": ",".join(map(str, range(count))),
                "ftpbe_total_energy": evaluation.total_energy,
                "ftpbe_error_vs_exact_rdm_posthoc": (
                    evaluation.total_energy - exact_ftpbe
                ),
                "joint_target_loss_posthoc": max(
                    source["energy_error"] / args.energy_target,
                    source["d2_normalized_frobenius_error"] / args.d2_target,
                ),
                "meets_both_targets_posthoc": (
                    source["energy_error"] <= args.energy_target
                    and source["d2_normalized_frobenius_error"] <= args.d2_target
                ),
            }
        )
    return rows


def _posthoc_metrics(
    args: argparse.Namespace,
    row: dict[str, Any],
    d2: np.ndarray,
    gamma: np.ndarray,
    reference: Any,
    exact_ftpbe: float,
) -> dict[str, Any]:
    energy = rdm_energy(
        d2,
        gamma,
        reference.one_body,
        reference.two_body,
        reference.nuclear_energy,
    )
    energy_error = abs(energy - reference.exact_energy)
    d2_error = float(
        np.linalg.norm(d2 - reference.exact_d2) / np.trace(reference.exact_d2)
    )
    gamma_error = float(np.linalg.norm(gamma - reference.exact_gamma))
    ftpbe_error = float(row["ftpbe_total_energy"] - exact_ftpbe)
    return {
        **row,
        "energy": energy,
        "energy_error": energy_error,
        "d2_normalized_frobenius_error": d2_error,
        "d2_frobenius_error": float(np.linalg.norm(d2 - reference.exact_d2)),
        "gamma_frobenius_error": gamma_error,
        "ftpbe_error_vs_exact_rdm_posthoc": ftpbe_error,
        "joint_target_loss_posthoc": max(
            energy_error / args.energy_target,
            d2_error / args.d2_target,
        ),
        "meets_both_targets_posthoc": (
            energy_error <= args.energy_target and d2_error <= args.d2_target
        ),
    }


def _blind_evaluate_policy_rows(
    args: argparse.Namespace,
    policy: str,
    rows: list[dict[str, Any]],
    reference: Any,
    exact_ftpbe: float,
) -> list[dict[str, Any]]:
    evaluated = []
    for row in rows:
        _, npz_path = _checkpoint_paths(
            args.output_dir, policy, int(row["shadows"])
        )
        with np.load(npz_path) as archive:
            evaluated.append(
                _posthoc_metrics(
                    args,
                    row,
                    np.asarray(archive["d2"]),
                    np.asarray(archive["gamma"]),
                    reference,
                    exact_ftpbe,
                )
            )
    return evaluated


def _polyak_rows(
    args: argparse.Namespace,
    policy: str,
    selection_rows: list[dict[str, Any]],
    reference: Any,
    exact_ftpbe: float,
    objective: FtPBEEnergyObjective,
    all_shadows: Any,
    d2_values: Sequence[np.ndarray],
    gamma_values: Sequence[np.ndarray],
    method: str,
) -> list[dict[str, Any]]:
    """Evaluate a predeclared rolling convex average of physical DQG iterates."""

    window = int(args.polyak_window)
    if window < 2:
        return []
    if not len(selection_rows) == len(d2_values) == len(gamma_values):
        raise ValueError("Polyak rows and RDM values must align.")
    indices = [int(row["selected_index"]) for row in selection_rows]
    averaged = []
    for stop in range(window, len(selection_rows) + 1):
        start = stop - window
        d2 = np.mean(d2_values[start:stop], axis=0)
        gamma = np.mean(gamma_values[start:stop], axis=0)
        selected_shadows = acquire_shadow_bases(all_shadows, indices[:stop])
        evaluation = objective.evaluate(d2, gamma, gradient=False)
        row = {
            "policy": f"{policy}_polyak{window}",
            "method": f"{method} + rolling Polyak-{window}",
            "shadows": stop,
            "selected_index": indices[stop - 1],
            "selected_indices": ",".join(map(str, indices[:stop])),
            "polyak_window": window,
            "weighted_fit_rmse": weighted_prediction_rmse(selected_shadows, d2),
            "ftpbe_total_energy": evaluation.total_energy,
            "ftpbe_minimum_density": evaluation.minimum_density,
            "ftpbe_minimum_on_top_pair_density": (
                evaluation.minimum_on_top_pair_density
            ),
        }
        if stop < len(selection_rows):
            validation = acquire_shadow_bases(all_shadows, (indices[stop],))
            row["next_shadow_prequential_rmse"] = weighted_prediction_rmse(
                validation, d2
            )
            row["next_shadow_index"] = indices[stop]
        averaged.append(
            _posthoc_metrics(args, row, d2, gamma, reference, exact_ftpbe)
        )
    return averaged


def _prequential_diagnostics(
    all_shadows: Any,
    rows: list[dict[str, Any]],
    d2_values: list[np.ndarray],
) -> dict[str, Any]:
    """One-step-ahead validation using only outcomes acquired later in time."""

    if len(rows) != len(d2_values):
        raise ValueError("Prequential rows and RDM values must align.")
    weighted_squared_errors = []
    cumulative_rmse = []
    for index in range(len(rows) - 1):
        next_shadow = int(rows[index + 1]["selected_index"])
        validation = acquire_shadow_bases(all_shadows, (next_shadow,))
        probabilities = (validation.hits.astype(float) + 0.5) / (
            validation.shots_per_basis + 1.0
        )
        variances = np.maximum(
            probabilities * (1.0 - probabilities) / validation.shots_per_basis,
            1e-12,
        )
        residual = (validation.predict(d2_values[index]) - validation.values) / np.sqrt(
            variances
        )
        score = float(np.mean(residual**2))
        weighted_squared_errors.append(
            {
                "model_shadows": int(rows[index]["shadows"]),
                "validation_shadow": next_shadow,
                "weighted_mean_squared_error": score,
                "weighted_rmse": float(np.sqrt(score)),
            }
        )
        cumulative_rmse.append(
            float(
                np.sqrt(
                    np.mean(
                        [
                            item["weighted_mean_squared_error"]
                            for item in weighted_squared_errors
                        ]
                    )
                )
            )
        )
    best_index = int(np.argmin(cumulative_rmse)) if cumulative_rmse else None
    return {
        "one_step_scores": weighted_squared_errors,
        "cumulative_weighted_rmse": cumulative_rmse,
        "best_cumulative_model_shadow": (
            weighted_squared_errors[best_index]["model_shadows"]
            if best_index is not None
            else None
        ),
        "best_cumulative_weighted_rmse": (
            cumulative_rmse[best_index] if best_index is not None else None
        ),
        "scope": (
            "Each acquired basis validates the preceding model before that "
            "basis outcome entered reconstruction; FCI-free, online-valid."
        ),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["shadows"])
    both = [row for row in ordered if row["meets_both_targets_posthoc"]]
    best = min(ordered, key=lambda row: row["joint_target_loss_posthoc"])
    best_d2 = min(ordered, key=lambda row: row["d2_normalized_frobenius_error"])
    best_ftpbe = min(
        ordered, key=lambda row: abs(row["ftpbe_error_vs_exact_rdm_posthoc"])
    )
    return {
        "first_both_targets_shadow": both[0]["shadows"] if both else None,
        "best_joint_target_loss": best["joint_target_loss_posthoc"],
        "best_joint_target_shadow": best["shadows"],
        "best_d2_error": best_d2["d2_normalized_frobenius_error"],
        "best_d2_shadow": best_d2["shadows"],
        "best_absolute_ftpbe_error_posthoc": abs(
            best_ftpbe["ftpbe_error_vs_exact_rdm_posthoc"]
        ),
        "best_ftpbe_shadow": best_ftpbe["shadows"],
        "selected_indices": ordered[-1]["selected_indices"],
    }


def _equivalent_budget_summary(
    rows: list[dict[str, Any]],
    baseline_budget: int,
    polyak_window: int,
) -> dict[str, Any]:
    """Compare first budgets that dominate one random-baseline accuracy point."""

    result: dict[str, Any] = {}
    for label, suffix in (
        ("raw", ""),
        (f"rolling_polyak{polyak_window}", f"_polyak{polyak_window}"),
    ):
        baseline_policy = f"{POLICY_RANDOM}{suffix}"
        baseline_rows = sorted(
            (row for row in rows if row["policy"] == baseline_policy),
            key=lambda row: int(row["shadows"]),
        )
        target = next(
            (
                row
                for row in baseline_rows
                if int(row["shadows"]) == int(baseline_budget)
            ),
            None,
        )
        if target is None:
            continue
        target_energy = float(target["energy_error"])
        target_d2 = float(target["d2_normalized_frobenius_error"])
        policies: dict[str, Any] = {}
        for policy in (POLICY_RANDOM, POLICY_DESIGN, POLICY_MCPDFT):
            candidate_policy = f"{policy}{suffix}"
            candidates = sorted(
                (row for row in rows if row["policy"] == candidate_policy),
                key=lambda row: int(row["shadows"]),
            )
            hit = next(
                (
                    row
                    for row in candidates
                    if float(row["energy_error"]) <= target_energy
                    and float(row["d2_normalized_frobenius_error"]) <= target_d2
                ),
                None,
            )
            policies[policy] = (
                None
                if hit is None
                else {
                    "first_equivalent_shadow": int(hit["shadows"]),
                    "energy_error": float(hit["energy_error"]),
                    "d2_normalized_frobenius_error": float(
                        hit["d2_normalized_frobenius_error"]
                    ),
                }
            )
        baseline_hit = policies[POLICY_RANDOM]
        baseline_first = (
            None
            if baseline_hit is None
            else int(baseline_hit["first_equivalent_shadow"])
        )
        if baseline_first is not None:
            for value in policies.values():
                if value is not None:
                    value["saving_vs_baseline_fraction"] = float(
                        1.0 - value["first_equivalent_shadow"] / baseline_first
                    )
        result[label] = {
            "nominal_baseline_shadow": int(baseline_budget),
            "baseline_first_equivalent_shadow": baseline_first,
            "target_energy_error": target_energy,
            "target_d2_normalized_frobenius_error": target_d2,
            "policies": policies,
        }
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    leading = [
        "policy",
        "shadows",
        "selected_index",
        "selected_indices",
        "energy_error",
        "d2_normalized_frobenius_error",
        "ftpbe_error_vs_exact_rdm_posthoc",
        "weighted_fit_rmse",
    ]
    fields = leading + sorted(
        set().union(*(row.keys() for row in rows)).difference(leading)
    )
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _plot(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    styles = {
        POLICY_RANDOM: dict(color="#555555", marker="o", label="random prefix"),
        f"{POLICY_RANDOM}_polyak{args.polyak_window}": dict(
            color="#777777",
            marker="x",
            linestyle="--",
            label=f"random + Polyak-{args.polyak_window}",
        ),
        POLICY_DESIGN: dict(color="#2b6cb0", marker="s", label="D-optimal"),
        f"{POLICY_DESIGN}_polyak{args.polyak_window}": dict(
            color="#4299e1",
            marker="P",
            linestyle="--",
            label=f"D-optimal + Polyak-{args.polyak_window}",
        ),
        POLICY_MCPDFT: dict(
            color="#c05621", marker="^", label="guarded MC-PDFT"
        ),
        f"{POLICY_MCPDFT}_polyak{args.polyak_window}": dict(
            color="#2f855a", marker="D", label=f"MC-PDFT + Polyak-{args.polyak_window}"
        ),
    }
    figure, axes = plt.subplots(3, 1, figsize=(8.2, 9.4), sharex=True)
    for policy, style in styles.items():
        selected = sorted(
            (row for row in rows if row["policy"] == policy),
            key=lambda row: row["shadows"],
        )
        if not selected:
            continue
        counts = [row["shadows"] for row in selected]
        axes[0].semilogy(
            counts, [max(row["energy_error"], 1e-16) for row in selected], **style
        )
        axes[1].semilogy(
            counts,
            [max(row["d2_normalized_frobenius_error"], 1e-16) for row in selected],
            **style,
        )
        axes[2].semilogy(
            counts,
            [max(abs(row["ftpbe_error_vs_exact_rdm_posthoc"]), 1e-16) for row in selected],
            **style,
        )
    axes[0].axhline(args.energy_target, color="#805ad5", linestyle="--")
    axes[1].axhline(args.d2_target, color="#805ad5", linestyle="--")
    axes[0].set_ylabel("Hamiltonian error (Eh)")
    axes[1].set_ylabel("Normalized 2-RDM error")
    axes[2].set_ylabel("Posthoc |ftPBE error| (Eh)")
    axes[2].set_xlabel("Selected shadows")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle(
        f"{args.system.upper()} FCI-free guarded MC-PDFT shadow derandomization"
    )
    figure.tight_layout()
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=190)
    plt.close(figure)
    os.replace(temporary, path)


def _write_report(
    path: Path,
    configuration: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    equivalent_budget: dict[str, Any],
) -> None:
    matched_budget = min(10, int(configuration["max_selected"]))
    matched_policies = (
        f"{POLICY_RANDOM}_polyak{configuration['polyak_window']}",
        f"{POLICY_DESIGN}_polyak{configuration['polyak_window']}",
        f"{POLICY_MCPDFT}_polyak{configuration['polyak_window']}",
    )
    matched_rows = {
        policy: next(
            (
                item
                for item in rows
                if item["policy"] == policy
                and int(item["shadows"]) == matched_budget
            ),
            None,
        )
        for policy in matched_policies
    }
    lines = [
        f"# {configuration['system']} 无 FCI 泄漏的 guarded MC-PDFT derandomization",
        "",
        "## 泄漏边界",
        "",
        "- 选择阶段只构建 RHF 轨道与 active-space Hamiltonian；不运行 CASCI/FCI，selection reference 中也不存在任何 `exact_*` 字段。",
        "- raw archive 由 blind loader 读取；其中的 `exact_d2`、`exact_gamma`、`exact_values` 不加载，内部 exact response 被 NaN 占位。",
        "- shadow 选择只使用 candidate rotations、已采集 raw frequencies、当前 DQG reconstruction、Hamiltonian integrals 与 ftPBE gradient。",
        "- 未采集 candidate 的 hits/values、exact FCI D2/gamma、Hamiltonian/D2 posthoc errors 均不进入选择或重构。",
        "- exact FCI 只在完整选择路径冻结后计算本报告的 benchmark metrics。",
        "",
        "## MC-PDFT 的安全融入",
        "",
        f"- 候选首先满足至少 {100.0 * configuration['design_guard_fraction']:.0f}% 最大 D-optimal logdet gain 的 guard。",
        "- guard 内选择使 ftPBE tangent posterior variance 下降最大的 rotation；因此其预测 MC-PDFT gain 不低于纯 D-optimal 候选。",
        "- ftPBE 不作为宽 DQG feasible set 的能量极小化 selector，不直接移动 2-RDM，避免此前的非变分下冲。",
        f"- rolling Polyak-{configuration['polyak_window']} 是预先固定的最近迭代凸平均；三路消融使用同一平均器，且 DQG 可行域的凸性保持物理约束。",
        "",
        "## 等价精度预算（主终点）",
        "",
        "目标精度取 random baseline 在参考预算的实际 Hamiltonian 与 2-RDM 误差；方法必须同时不劣于这两个值。为避免 baseline 非单调造成虚假节省，分母使用 baseline 自己首次达到该精度的预算。",
        "",
        "| estimator | target baseline m | baseline first | D-optimal first | D-optimal saving | guarded first | guarded saving |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, value in equivalent_budget.items():
        policies = value["policies"]
        design = policies.get(POLICY_DESIGN)
        guarded = policies.get(POLICY_MCPDFT)
        design_first = None if design is None else design["first_equivalent_shadow"]
        guarded_first = (
            None if guarded is None else guarded["first_equivalent_shadow"]
        )
        design_saving = (
            ""
            if design is None
            else f"{100.0 * design['saving_vs_baseline_fraction']:.1f}%"
        )
        guarded_saving = (
            ""
            if guarded is None
            else f"{100.0 * guarded['saving_vs_baseline_fraction']:.1f}%"
        )
        lines.append(
            f"| {label} | {value['nominal_baseline_shadow']} | "
            f"{value['baseline_first_equivalent_shadow']} | "
            f"{design_first} | {design_saving} | "
            f"{guarded_first} | {guarded_saving} |"
        )
    lines.extend(
        [
            "",
            "## 同预算消融",
            "",
            f"以下三路都使用同一个 rolling Polyak-{configuration['polyak_window']}，因此差别只来自 shadow 选择策略：",
            "",
            "| policy | m | Hamiltonian error (mEh) | normalized D2 error | meets joint target |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy in matched_policies:
        row = matched_rows[policy]
        if row is not None:
            lines.append(
                f"| {policy} | {matched_budget} | "
                f"{1000.0 * row['energy_error']:.3f} | "
                f"{row['d2_normalized_frobenius_error']:.3e} | "
                f"{row['meets_both_targets_posthoc']} |"
            )
    guarded_row = matched_rows[
        f"{POLICY_MCPDFT}_polyak{configuration['polyak_window']}"
    ]
    design_row = matched_rows[
        f"{POLICY_DESIGN}_polyak{configuration['polyak_window']}"
    ]
    if guarded_row is not None and design_row is not None:
        energy_better = guarded_row["energy_error"] < design_row["energy_error"]
        d2_better = (
            guarded_row["d2_normalized_frobenius_error"]
            < design_row["d2_normalized_frobenius_error"]
        )
        if energy_better and d2_better:
            blind_statement = (
                "matched-budget blind energy 与 D2 也都优于纯 D-optimal。"
            )
        elif energy_better or d2_better:
            improved = "energy" if energy_better else "D2"
            worsened = "D2" if energy_better else "energy"
            blind_statement = (
                f"matched-budget blind {improved} 改善，但 {worsened} 未改善，"
                "因此 MC-PDFT 的整体正向贡献尚未成立。"
            )
        else:
            blind_statement = (
                "matched-budget blind energy 与 D2 均未优于纯 D-optimal，"
                "因此这里只能确认设计层面的正向贡献。"
            )
    else:
        blind_statement = "matched-budget blind 对照不完整。"
    lines.extend(
        [
            "",
            "设计层面，纯 D-optimal candidate 始终位于 guard 内，MC-PDFT "
            "选择的预测 tangent-variance reduction 不低于它；"
            + blind_statement,
            "",
            "## 10 + 1 审计口径",
            "",
        ]
    )
    target_policy = f"{POLICY_MCPDFT}_polyak{configuration['polyak_window']}"
    target_row = next(
        (
            item
            for item in rows
            if item["policy"] == target_policy
            and int(item["shadows"]) == matched_budget
        ),
        None,
    )
    random_first = summaries.get(POLICY_RANDOM, {}).get(
        "first_both_targets_shadow"
    )
    if target_row is not None:
        target_status = (
            "联合达标" if target_row["meets_both_targets_posthoc"] else "未联合达标"
        )
        lines.append(
            f"- 前 {matched_budget} 条路径为 `{target_row['selected_indices']}`；blind posthoc 为 "
            f"`{1000.0 * target_row['energy_error']:.3f} mEh / "
            f"{target_row['d2_normalized_frobenius_error']:.3e}`，{target_status}。"
        )
        if "next_shadow_prequential_rmse" in target_row:
            lines.append(
                f"- 第 {matched_budget + 1} 条 index "
                f"`{target_row['next_shadow_index']}` 只审计尚未见过它的 "
                f"m={matched_budget} 平均解；one-step-ahead RMSE 为 "
                f"`{target_row['next_shadow_prequential_rmse']:.3f}`。"
            )
    if random_first is not None:
        audit_budget = matched_budget + 1
        saving = 100.0 * (1.0 - audit_budget / float(random_first))
        lines.append(
            f"- 若把独立审计也计入预算，是 `{audit_budget} vs {random_first}`，"
            f"相对标准随机前缀少约 `{saving:.0f}%`；不计审计时重构本身使用 "
            f"{matched_budget} 条。"
        )
    lines.extend(
        [
            "- prequential RMSE 只证明 held-out shadow 的统计相容性，不能在未知真值下单独证明完整 D2 Frobenius 精度。",
            "",
            "## Posthoc 结果",
            "",
            "| policy | first both | best joint (m) | best D2 (m) | best |ftPBE error| (m) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for policy, summary in summaries.items():
        lines.append(
            f"| {policy} | {summary['first_both_targets_shadow']} | "
            f"{summary['best_joint_target_loss']:.3f} ({summary['best_joint_target_shadow']}) | "
            f"{summary['best_d2_error']:.3e} ({summary['best_d2_shadow']}) | "
            f"{summary['best_absolute_ftpbe_error_posthoc']:.3e} "
            f"({summary['best_ftpbe_shadow']}) |"
        )
    lines.extend(
        [
            "",
            "## 逐点",
            "",
            "| policy | m | add | energy | D2 | ftPBE error | fit | D-gain ratio | ftPBE gain over D-opt |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["policy"], item["shadows"])):
        ratio = row.get("selected_design_gain_fraction")
        gain = row.get("mcpdft_gain_over_design")
        lines.append(
            f"| {row['policy']} | {row['shadows']} | {row['selected_index']} | "
            f"{row['energy_error']:.3e} | "
            f"{row['d2_normalized_frobenius_error']:.3e} | "
            f"{row['ftpbe_error_vs_exact_rdm_posthoc']:+.3e} | "
            f"{row['weighted_fit_rmse']:.3f} | "
            f"{'' if ratio is None else f'{ratio:.3f}'} | "
            f"{'' if gain is None else f'{gain:.3e}'} |"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    if args.basis is None:
        args.basis = "cc-pvdz" if args.system == "n2" else "cc-pvtz"
    if args.bond_length is None:
        args.bond_length = 1.75 if args.system == "n2" else 1.25
    if args.active_electrons is None:
        args.active_electrons = 6 if args.system == "n2" else 8
    if args.active_orbitals is None:
        args.active_orbitals = 6 if args.system == "n2" else 8
    if args.random_baseline == "auto":
        args.random_baseline = "native" if args.system == "c2" else "existing"
    if args.system == "c2" and (
        args.active_electrons != 8 or args.active_orbitals != 8
    ):
        raise ValueError("The C2 protocol is fixed to valence AVAS CAS(8e,8o).")
    if args.system == "c2" and args.active_orbital_signs is not None:
        raise ValueError("Explicit active-orbital signs are only supported for N2.")
    if args.system == "n2" and args.frozen_mo_npz is not None:
        raise ValueError("Frozen MO archives are only supported for C2.")
    if (
        args.active_orbital_signs is not None
        and len(args.active_orbital_signs) != args.active_orbitals
    ):
        raise ValueError("active-orbital-signs must match active-orbitals.")
    args.shadow_input_npz = args.shadow_input_npz.resolve()
    if args.frozen_mo_npz is not None:
        args.frozen_mo_npz = args.frozen_mo_npz.resolve()
    args.original_results_dir = args.original_results_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 < args.design_guard_fraction <= 1.0:
        raise ValueError("design-guard-fraction must lie in (0, 1].")
    if not 0.0 <= args.hamiltonian_guard_fraction <= 1.0:
        raise ValueError("hamiltonian-guard-fraction must lie in [0, 1].")
    if (
        args.weighted_fit_rmse_cap is not None
        and args.weighted_fit_rmse_cap <= 0.0
    ):
        raise ValueError("weighted-fit-rmse-cap must be positive.")
    frozen_mo_coeff = None
    if args.frozen_mo_npz is not None:
        with np.load(args.frozen_mo_npz, allow_pickle=False) as archive:
            if "mo_coeff" not in archive.files:
                raise ValueError(
                    f"Frozen MO archive {args.frozen_mo_npz} has no mo_coeff array."
                )
            frozen_mo_coeff = np.array(archive["mo_coeff"], copy=True)

    configuration = _configuration(args)
    configuration_path = args.output_dir / "configuration.json"
    if configuration_path.exists() and not args.overwrite:
        existing = json.loads(configuration_path.read_text())
        for name in (
            "polyak_window",
            "selection_reference",
            "archive_exact_arrays_loaded_for_selection",
            "polyak_average_predeclared",
            "random_baseline",
            "weighted_fit_rmse_cap",
            "dqg_reconstruction",
            "reconstruction_objective",
            "ridge_grid",
            "ridge_folds",
            "fallback_ridge",
        ):
            existing.setdefault(name, configuration[name])
        if existing != configuration:
            raise ValueError("Output directory contains a different configuration.")
    _atomic_json(configuration_path, configuration)

    if args.system == "c2":
        selection_base = build_c2_selection_reference(
            args.bond_length,
            basis=args.basis,
            frozen_mo_coeff=frozen_mo_coeff,
        )
    else:
        selection_base = build_n2_selection_reference(
            args.bond_length,
            basis=args.basis,
            active_electrons=args.active_electrons,
            active_orbitals=args.active_orbitals,
            active_orbital_signs=args.active_orbital_signs,
        )
    all_shadows = load_blind_shadow_npz(args.shadow_input_npz, load_design=False)
    validate_shadow_archive_structure(
        args.shadow_input_npz,
        all_shadows,
        selection_base,
        args.max_selected,
        args.shots_per_shadow,
    )
    if args.max_selected > all_shadows.n_shadows:
        raise ValueError("max-selected exceeds the candidate pool.")

    selection_reference = LeakGuardReference(selection_base)
    objective = FtPBEEnergyObjective(
        selection_reference, grid_level=args.grid_level
    )
    variable_rows, variable_cols = _symmetric_d2_variables(selection_reference)
    selection_shadows = LeakGuardShadowData.from_shadow_data(all_shadows)
    blocks = shadow_design_blocks(
        selection_shadows,
        len(selection_reference.pairs),
        variable_rows,
        variable_cols,
    )
    print("Solving the common no-data DQG initialization...", flush=True)
    initial = _initial_result(args, selection_reference)
    field_modes = None
    if args.mcpdft_target == "field_response":
        print("Building frozen ftPBE field-response modes...", flush=True)
        field_modes = objective.field_response_modes(
            initial.d2,
            initial.gamma,
            max_modes=args.field_response_modes,
            capture_fraction=args.field_response_capture,
        )
        print(f"Frozen ftPBE response modes: {len(field_modes)}", flush=True)

    selection_rows: dict[str, list[dict[str, Any]]] = {}
    if args.random_baseline == "native":
        print(f"Running {POLICY_RANDOM}...", flush=True)
        selection_rows[POLICY_RANDOM] = _run_random_policy(
            args,
            selection_reference,
            all_shadows,
            objective,
            initial,
        )
        _validate_selection_rows_blind(selection_rows)
        _atomic_json(
            args.output_dir / "selection_only.json",
            {"configuration": configuration, "rows": selection_rows},
        )
    for policy in args.policies:
        print(f"Running {policy}...", flush=True)
        selection_rows[policy] = _run_policy(
            args,
            policy,
            selection_reference,
            all_shadows,
            blocks,
            variable_rows,
            variable_cols,
            objective,
            initial,
            field_modes,
        )
        _validate_selection_rows_blind(selection_rows)
        _atomic_json(
            args.output_dir / "selection_only.json",
            {"configuration": configuration, "rows": selection_rows},
        )

    if args.selection_only:
        print(
            "Selection paths frozen; --selection-only skips all FCI construction.",
            flush=True,
        )
        return

    print("Selection paths frozen; starting posthoc blind evaluation...", flush=True)
    if args.system == "c2":
        reference = build_c2_reference(
            args.bond_length,
            basis=args.basis,
            frozen_mo_coeff=frozen_mo_coeff,
        )
    else:
        reference = build_n2_reference(
            args.bond_length,
            basis=args.basis,
            active_electrons=args.active_electrons,
            active_orbitals=args.active_orbitals,
            active_orbital_signs=args.active_orbital_signs,
        )
    if not (
        np.allclose(selection_base.one_body, reference.one_body, atol=2e-10)
        and np.allclose(selection_base.two_body, reference.two_body, atol=2e-10)
        and np.isclose(
            selection_base.nuclear_energy, reference.nuclear_energy, atol=2e-10
        )
    ):
        raise RuntimeError(
            "The frozen selection Hamiltonian differs from the blind FCI reference."
        )
    exact_ftpbe = objective.evaluate(
        reference.exact_d2, reference.exact_gamma, gradient=False
    ).total_energy
    if args.random_baseline == "native":
        random_rows = _blind_evaluate_policy_rows(
            args,
            POLICY_RANDOM,
            selection_rows[POLICY_RANDOM],
            reference,
            exact_ftpbe,
        )
    elif args.random_baseline == "existing":
        random_rows = _load_random_rows_posthoc(args, objective, exact_ftpbe)
    else:
        random_rows = []
    all_rows = list(random_rows)
    random_d2_values = []
    random_gamma_values = []
    for row in random_rows:
        if args.random_baseline == "native":
            _, checkpoint = _checkpoint_paths(
                args.output_dir, POLICY_RANDOM, int(row["shadows"])
            )
            d2_key = "d2"
            gamma_key = "gamma"
        else:
            checkpoint = (
                args.original_results_dir
                / "checkpoints"
                / f"shadows_{int(row['shadows']):02d}.npz"
            )
            d2_key = "baseline_d2"
            gamma_key = "baseline_gamma"
        with np.load(checkpoint) as archive:
            random_d2_values.append(np.asarray(archive[d2_key]))
            random_gamma_values.append(np.asarray(archive[gamma_key]))
    if random_rows:
        all_rows.extend(
            _polyak_rows(
                args,
                POLICY_RANDOM,
                random_rows,
                reference,
                exact_ftpbe,
                objective,
                all_shadows,
                random_d2_values,
                random_gamma_values,
                METHOD_RANDOM,
            )
        )
    prequential: dict[str, Any] = {}
    if args.random_baseline == "native":
        prequential[POLICY_RANDOM] = _prequential_diagnostics(
            all_shadows,
            selection_rows[POLICY_RANDOM],
            random_d2_values,
        )
    for policy in args.policies:
        d2_values = []
        gamma_values = []
        for row in selection_rows[policy]:
            _, npz_path = _checkpoint_paths(
                args.output_dir, policy, int(row["shadows"])
            )
            with np.load(npz_path) as archive:
                d2_values.append(np.asarray(archive["d2"]))
                gamma_values.append(np.asarray(archive["gamma"]))
        prequential[policy] = _prequential_diagnostics(
            all_shadows, selection_rows[policy], d2_values
        )
        all_rows.extend(
            _blind_evaluate_policy_rows(
                args,
                policy,
                selection_rows[policy],
                reference,
                exact_ftpbe,
            )
        )
        all_rows.extend(
            _polyak_rows(
                args,
                policy,
                selection_rows[policy],
                reference,
                exact_ftpbe,
                objective,
                all_shadows,
                d2_values,
                gamma_values,
                METHOD_DESIGN if policy == POLICY_DESIGN else METHOD_MCPDFT,
            )
        )

    base_policies = [
        *([POLICY_RANDOM] if random_rows else []),
        *args.policies,
    ]
    summary_policies = []
    for policy in base_policies:
        summary_policies.append(policy)
        polyak_policy = f"{policy}_polyak{args.polyak_window}"
        if any(row["policy"] == polyak_policy for row in all_rows):
            summary_policies.append(polyak_policy)
    summaries = {
        policy: _summary([row for row in all_rows if row["policy"] == policy])
        for policy in summary_policies
    }
    equivalent_budget = _equivalent_budget_summary(
        all_rows,
        min(10, args.max_selected),
        args.polyak_window,
    )
    payload = {
        "configuration": configuration,
        "exact_ftpbe_total_energy_posthoc": exact_ftpbe,
        "prequential_fci_free": prequential,
        "summaries": summaries,
        "equivalent_budget_primary": equivalent_budget,
        "rows": all_rows,
    }
    _atomic_json(args.output_dir / "selection_summary.json", payload)
    _write_csv(args.output_dir / "selection_trends.csv", all_rows)
    _plot(args.output_dir / "selection_trends.png", all_rows, args)
    _write_report(
        args.output_dir / "SAFE_MCPDFT_DERANDOMIZATION_FINDINGS.md",
        configuration,
        summaries,
        all_rows,
        equivalent_budget,
    )
    print(
        f"Report: {args.output_dir / 'SAFE_MCPDFT_DERANDOMIZATION_FINDINGS.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
