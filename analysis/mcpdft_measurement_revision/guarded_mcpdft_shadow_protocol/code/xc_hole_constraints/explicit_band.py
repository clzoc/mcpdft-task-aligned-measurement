"""Explicit full-grid XC-hole inequalities solved by constraint generation.

The full tensor-product grid contains 102,400 spherical-average constraints at
the default settings. Materializing every dense RDM row would require several
gigabytes, so this module uses a standard exchange method: every round scans
the complete grid, adds the most violated affine rows to the SDP, and stops
only after complete separation scans certify the omitted rows. A final scan
then evaluates all rows, including the active SDP cuts.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from constrained_shadow import LinearRDMConstraints

from .full_grid_hole import (
    FrozenHoleModel,
    FullGridHoleConstraint,
    ProgressCallback,
    _features2,
    _report_progress,
)


BandMode = Literal["tpbe_ftpbe_band", "ftpbe_relaxed", "ftpbe_target"]


@dataclass(frozen=True)
class ExplicitBandConfig:
    """Numerical settings for full-grid affine constraint generation."""

    mode: BandMode = "ftpbe_target"
    cuts_per_round: int = 64
    max_rounds: int = 8
    separation_tolerance: float = 2e-4
    linear_lexicographic_tolerance: float = 1e-5
    scale_floor_fraction: float = 1e-3
    ftpbe_relaxation: float = 2.0
    ftpbe_relative_tolerance: float = 0.0
    allow_minimax_relaxation: bool = False

    def validate(self) -> None:
        if self.mode not in {
            "tpbe_ftpbe_band",
            "ftpbe_relaxed",
            "ftpbe_target",
        }:
            raise ValueError(f"Unsupported explicit-hole mode: {self.mode}")
        if self.cuts_per_round < 1 or self.max_rounds < 1:
            raise ValueError("cuts_per_round and max_rounds must be positive.")
        if self.separation_tolerance <= 0.0:
            raise ValueError("separation_tolerance must be positive.")
        if self.linear_lexicographic_tolerance < 0.0:
            raise ValueError("linear_lexicographic_tolerance must be nonnegative.")
        if self.scale_floor_fraction <= 0.0:
            raise ValueError("scale_floor_fraction must be positive.")
        if self.ftpbe_relaxation < 1.0:
            raise ValueError("ftpbe_relaxation must be at least one.")
        if self.ftpbe_relative_tolerance < 0.0:
            raise ValueError("ftpbe_relative_tolerance must be nonnegative.")


@dataclass(frozen=True)
class BandSeparation:
    maximum_normalized_excess: float
    maximum_absolute_excess: float
    violation_fraction: float
    evaluated_constraint_points: int
    evaluated_pair_points: int
    cuts: tuple[tuple[int, int], ...]
    cut_excesses: tuple[float, ...]

    def metrics(self) -> dict[str, float | int]:
        return {
            "maximum_normalized_excess": self.maximum_normalized_excess,
            "maximum_absolute_excess": self.maximum_absolute_excess,
            "violation_fraction": self.violation_fraction,
            "evaluated_constraint_points": self.evaluated_constraint_points,
            "evaluated_pair_points": self.evaluated_pair_points,
        }


@dataclass(frozen=True)
class ExplicitBandRound:
    iteration: int
    active_constraints: int
    added_constraints: int
    minimax_slack: float
    separation: BandSeparation
    solver_status: str
    active_max_absolute_violation: float
    active_max_normalized_violation: float
    seconds: float


@dataclass(frozen=True)
class ExplicitBandResult:
    d2: np.ndarray
    gamma: np.ndarray
    energy: float
    status: str
    mode: str
    converged: bool
    active_constraints: int
    active_keys: tuple[tuple[int, int], ...]
    minimax_slack: float
    rounds: tuple[ExplicitBandRound, ...]
    initial_separation: BandSeparation
    final_separation: BandSeparation


@dataclass(frozen=True)
class CalibratedHardBandResult:
    """Feasibility calibration followed by a fixed hard-inequality solve."""

    d2: np.ndarray
    gamma: np.ndarray
    energy: float
    status: str
    base_relative_tolerance: float
    calibration_extra_tolerance: float
    calibrated_relative_tolerance: float
    calibration_margin: float
    calibration: ExplicitBandResult
    hard: ExplicitBandResult


def _model_bounds(
    constraint: FullGridHoleConstraint,
    model: FrozenHoleModel,
    selection: slice | np.ndarray,
    radius: float,
    gaussian_pair: np.ndarray,
    config: ExplicitBandConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    variants = constraint._model_cumulants(model, selection, radius)
    if config.mode == "tpbe_ftpbe_band":
        values = gaussian_pair[None, :] + variants
        lower = np.min(values, axis=0)
        upper = np.max(values, axis=0)
    elif config.mode == "ftpbe_relaxed":
        # _model_cumulants orders tPBE cusp/Gaussian then ftPBE cusp/Gaussian.
        ftpbe = gaussian_pair[None, :] + variants[2:4]
        center = np.mean(ftpbe, axis=0)
        half_width = 0.5 * np.ptp(ftpbe, axis=0) * config.ftpbe_relaxation
        lower = center - half_width
        upper = center + half_width
    else:
        # A single ftPBE completion is the target. Its tolerance is an explicit
        # relative model-error allowance, not a width inferred from disagreement
        # with tPBE or from the finite-basis zero-slope branch.
        center = gaussian_pair + variants[2]
        local_pair_scale = np.maximum(
            np.maximum(np.abs(center), model.rho[selection] ** 2),
            1e-10,
        )
        half_width = config.ftpbe_relative_tolerance * local_pair_scale
        return (
            center - half_width,
            center + half_width,
            local_pair_scale,
        )
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    local_pair_scale = np.maximum(
        np.maximum(np.abs(center), model.rho[selection] ** 2),
        1e-10,
    )
    scales = np.maximum(
        half_width,
        config.scale_floor_fraction * local_pair_scale,
    )
    return lower, upper, scales


def _spherical_batch(
    constraint: FullGridHoleConstraint,
    d2_kernel: np.ndarray,
    gamma: np.ndarray,
    model: FrozenHoleModel,
    selection: slice | np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = constraint.coordinates[selection]
    rho_x = model.rho[selection]
    pair_value = np.zeros(len(coordinates), dtype=float)
    gaussian_pair = np.zeros(len(coordinates), dtype=float)
    for direction, angular_weight in zip(
        constraint.directions, constraint.angular_weights
    ):
        displaced = coordinates + radius * direction[None, :]
        directional_pair, aux = constraint._pair_density_batch(
            d2_kernel, gamma, selection, displaced
        )
        rho_active_y = constraint._active_density(
            aux["active_y"], model.anchor_gamma
        )
        spin_transition_squared = np.zeros(len(coordinates), dtype=float)
        for spin in (0, 1):
            active_transition = constraint._spin_active_transition_density(
                aux["active_x"],
                aux["active_y"],
                model.anchor_gamma,
                spin,
            )
            spin_transition_squared += (
                aux["gamma_core_xy"] + active_transition
            ) ** 2
        weight = float(angular_weight)
        pair_value += weight * directional_pair
        gaussian_pair += weight * (
            rho_x * (rho_active_y + aux["rho_core_y"])
            - spin_transition_squared
        )
    return pair_value, gaussian_pair


def separate_full_grid(
    constraint: FullGridHoleConstraint,
    d2: np.ndarray,
    gamma: np.ndarray,
    model: FrozenHoleModel,
    config: ExplicitBandConfig,
    *,
    excluded: set[tuple[int, int]] | None = None,
    progress: ProgressCallback | None = None,
    progress_stage: str = "explicit band: full-grid separation",
) -> BandSeparation:
    """Scan every configured (r, |u|, direction) point and return top cuts."""

    config.validate()
    excluded_keys = set() if excluded is None else excluded
    kernel = constraint.active_pair_kernel(d2)
    batch_size = max(1, int(constraint.config.batch_size))
    central_batches = math.ceil(len(constraint.coordinates) / batch_size)
    total_batches = central_batches * len(constraint.radial_values)
    completed_batches = 0
    evaluated_constraints = 0
    violation_count = 0
    maximum_normalized = 0.0
    maximum_absolute = 0.0
    heap: list[tuple[float, int, int]] = []

    _report_progress(
        progress,
        progress_stage,
        0,
        total_batches,
        event="start",
        unit="batches",
        mode=config.mode,
        central_grid_points=len(constraint.coordinates),
        radial_points=len(constraint.radial_values),
        angular_points=len(constraint.directions),
        full_pair_points=constraint.pair_point_count,
    )
    for start in range(0, len(constraint.coordinates), batch_size):
        stop = min(start + batch_size, len(constraint.coordinates))
        selection = slice(start, stop)
        for radial_index, radius in enumerate(constraint.radial_values):
            pair_value, gaussian_pair = _spherical_batch(
                constraint,
                kernel,
                gamma,
                model,
                selection,
                float(radius),
            )
            lower, upper, scales = _model_bounds(
                constraint,
                model,
                selection,
                float(radius),
                gaussian_pair,
                config,
            )
            absolute_excess = np.maximum(
                np.maximum(lower - pair_value, pair_value - upper), 0.0
            )
            normalized_excess = absolute_excess / scales
            eligible = np.ones(len(pair_value), dtype=bool)
            if excluded_keys:
                for local in range(len(pair_value)):
                    if (start + local, radial_index) in excluded_keys:
                        eligible[local] = False
            eligible_normalized = normalized_excess[eligible]
            eligible_absolute = absolute_excess[eligible]
            maximum_normalized = max(
                maximum_normalized,
                float(np.max(eligible_normalized, initial=0.0)),
            )
            maximum_absolute = max(
                maximum_absolute,
                float(np.max(eligible_absolute, initial=0.0)),
            )
            violation_count += int(np.count_nonzero(eligible_absolute > 0.0))
            evaluated_constraints += len(pair_value)

            for local in np.flatnonzero(normalized_excess > 0.0):
                key = (start + int(local), radial_index)
                if key in excluded_keys:
                    continue
                entry = (float(normalized_excess[local]), key[0], key[1])
                if len(heap) < config.cuts_per_round:
                    heapq.heappush(heap, entry)
                elif entry[0] > heap[0][0]:
                    heapq.heapreplace(heap, entry)

            completed_batches += 1
            if completed_batches < total_batches:
                _report_progress(
                    progress,
                    progress_stage,
                    completed_batches,
                    total_batches,
                    central_points_completed=stop,
                    radial_index=radial_index + 1,
                    maximum_normalized_excess=maximum_normalized,
                )

    ordered = sorted(heap, reverse=True)
    result = BandSeparation(
        maximum_normalized_excess=float(maximum_normalized),
        maximum_absolute_excess=float(maximum_absolute),
        violation_fraction=float(
            violation_count
            / max(evaluated_constraints - len(excluded_keys), 1)
        ),
        evaluated_constraint_points=int(evaluated_constraints),
        evaluated_pair_points=int(
            evaluated_constraints * len(constraint.directions)
        ),
        cuts=tuple((central, radial) for _, central, radial in ordered),
        cut_excesses=tuple(value for value, _, _ in ordered),
    )
    _report_progress(
        progress,
        progress_stage,
        total_batches,
        total_batches,
        event="complete",
        maximum_normalized_excess=result.maximum_normalized_excess,
        maximum_absolute_excess=result.maximum_absolute_excess,
        violation_fraction=result.violation_fraction,
        proposed_cuts=len(result.cuts),
    )
    return result


def _affine_rows_for_radius(
    constraint: FullGridHoleConstraint,
    model: FrozenHoleModel,
    central_indices: np.ndarray,
    radial_index: int,
    config: ExplicitBandConfig,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    radius = float(constraint.radial_values[radial_index])
    coordinates = constraint.coordinates[central_indices]
    active_x = constraint.active_values[central_indices]
    core_x = constraint.core_values[central_indices]
    rho_core_x = constraint.core_density[central_indices]
    features_x = constraint.active_features[central_indices]

    average_features_y = np.zeros_like(features_x)
    average_active_yy = np.zeros(
        (len(central_indices), constraint.n_spatial, constraint.n_spatial),
        dtype=float,
    )
    average_rho_core_y = np.zeros(len(central_indices), dtype=float)
    average_gamma_core_squared = np.zeros(len(central_indices), dtype=float)
    average_core_transition_y = np.zeros(
        (len(central_indices), constraint.n_spatial), dtype=float
    )
    gaussian_pair = np.zeros(len(central_indices), dtype=float)
    rho_x = model.rho[central_indices]

    for direction, angular_weight in zip(
        constraint.directions, constraint.angular_weights
    ):
        displaced = coordinates + radius * direction[None, :]
        ao_y = constraint.molecule.eval_gto("GTOval_sph", displaced)
        active_y = np.asarray(ao_y @ constraint.active_coeff, dtype=float)
        core_y = np.asarray(ao_y @ constraint.core_coeff, dtype=float)
        features_y = _features2(active_y, constraint.spatial_pairs)
        rho_core_y = 2.0 * np.einsum(
            "gi,gi->g", core_y, core_y, optimize=True
        )
        gamma_core_xy = np.einsum("gi,gi->g", core_x, core_y, optimize=True)
        rho_active_y = constraint._active_density(active_y, model.anchor_gamma)
        spin_transition_squared = np.zeros(len(central_indices), dtype=float)
        for spin in (0, 1):
            active_transition = constraint._spin_active_transition_density(
                active_x, active_y, model.anchor_gamma, spin
            )
            spin_transition_squared += (gamma_core_xy + active_transition) ** 2
        weight = float(angular_weight)
        average_features_y += weight * features_y
        average_active_yy += weight * np.einsum(
            "gi,gj->gij", active_y, active_y, optimize=True
        )
        average_rho_core_y += weight * rho_core_y
        average_gamma_core_squared += weight * gamma_core_xy**2
        average_core_transition_y += (
            weight * gamma_core_xy[:, None] * active_y
        )
        gaussian_pair += weight * (
            rho_x * (rho_active_y + rho_core_y) - spin_transition_squared
        )

    kernel_rows = np.einsum(
        "gi,gj->gij", features_x, average_features_y, optimize=True
    ).reshape((len(central_indices), -1), order="C")
    d2_map = sparse.csr_matrix(kernel_rows) @ constraint.pair_map

    active_xx = np.einsum(
        "gi,gj->gij", active_x, active_x, optimize=True
    )
    gamma_block = rho_core_x[:, None, None] * average_active_yy
    gamma_block += average_rho_core_y[:, None, None] * active_xx
    gamma_block -= 2.0 * np.einsum(
        "gi,gj->gij", active_x, average_core_transition_y, optimize=True
    )
    gamma_rows = np.zeros(
        (len(central_indices), constraint.n_modes, constraint.n_modes),
        dtype=float,
    )
    gamma_rows[:, : constraint.n_spatial, : constraint.n_spatial] = gamma_block
    gamma_rows[:, constraint.n_spatial :, constraint.n_spatial :] = gamma_block
    gamma_map = sparse.csr_matrix(
        gamma_rows.reshape((len(central_indices), -1), order="C")
    )

    core_constant = (
        rho_core_x * average_rho_core_y - 2.0 * average_gamma_core_squared
    )
    lower, upper, scales = _model_bounds(
        constraint,
        model,
        central_indices,
        radius,
        gaussian_pair,
        config,
    )
    return d2_map, gamma_map, lower - core_constant, upper - core_constant, scales


def build_linear_constraints(
    constraint: FullGridHoleConstraint,
    model: FrozenHoleModel,
    keys: Sequence[tuple[int, int]],
    config: ExplicitBandConfig,
) -> "LinearRDMConstraints":
    """Materialize affine rows only for the active exchange set."""

    from constrained_shadow import LinearRDMConstraints

    if not keys:
        raise ValueError("At least one explicit hole constraint key is required.")
    positions: dict[tuple[int, int], int] = {key: index for index, key in enumerate(keys)}
    d2_parts: list[tuple[np.ndarray, sparse.csr_matrix]] = []
    gamma_parts: list[tuple[np.ndarray, sparse.csr_matrix]] = []
    lower = np.empty(len(keys), dtype=float)
    upper = np.empty(len(keys), dtype=float)
    scales = np.empty(len(keys), dtype=float)
    for radial_index in sorted({radial for _, radial in keys}):
        radial_keys = [key for key in keys if key[1] == radial_index]
        output_rows = np.asarray([positions[key] for key in radial_keys], dtype=int)
        central = np.asarray([key[0] for key in radial_keys], dtype=int)
        d_map, g_map, lo, hi, row_scales = _affine_rows_for_radius(
            constraint, model, central, radial_index, config
        )
        d2_parts.append((output_rows, d_map))
        gamma_parts.append((output_rows, g_map))
        lower[output_rows] = lo
        upper[output_rows] = hi
        scales[output_rows] = row_scales

    def stack_at_positions(
        parts: list[tuple[np.ndarray, sparse.csr_matrix]], width: int
    ) -> sparse.csr_matrix:
        rows: list[sparse.csr_matrix | None] = [None] * len(keys)
        for output_rows, matrix in parts:
            for local, output in enumerate(output_rows):
                rows[int(output)] = matrix.getrow(local)
        if any(row is None for row in rows):
            raise RuntimeError("Failed to materialize every active affine row.")
        return sparse.vstack(rows, format="csr")

    return LinearRDMConstraints(
        d2_map=stack_at_positions(
            d2_parts, constraint.n_pair_basis * constraint.n_pair_basis
        ),
        gamma_map=stack_at_positions(
            gamma_parts, constraint.n_modes * constraint.n_modes
        ),
        lower_bounds=lower,
        upper_bounds=upper,
        scales=scales,
        allow_minimax_relaxation=config.allow_minimax_relaxation,
        name=config.mode,
    )


def solve_explicit_full_grid_band(
    reference: Any,
    shadow_data: Any,
    initial_result: Any,
    constraint: FullGridHoleConstraint,
    solve_sdp: Any,
    config: ExplicitBandConfig,
    *,
    solver: str = "SCS",
    tolerance: float = 1e-4,
    max_iterations: int = 50_000,
    verbose: bool = False,
    progress: ProgressCallback | None = None,
    frozen_model: FrozenHoleModel | None = None,
    initial_keys: Sequence[tuple[int, int]] = (),
    progress_prefix: str = "",
    linear_fit_only: bool = False,
) -> ExplicitBandResult:
    """Solve a frozen-model explicit band by full-grid constraint generation."""

    import time

    config.validate()
    model = (
        constraint.freeze_model(initial_result.d2, initial_result.gamma)
        if frozen_model is None
        else frozen_model
    )
    current = initial_result
    active_keys: list[tuple[int, int]] = []
    active_set: set[tuple[int, int]] = set()
    initial_separation = separate_full_grid(
        constraint,
        current.d2,
        current.gamma,
        model,
        config,
        progress=progress,
        progress_stage=(
            f"{progress_prefix}{config.mode}: initial full-grid separation"
        ),
    )
    seed_keys = list(dict.fromkeys(initial_keys))
    pending = seed_keys if seed_keys else list(initial_separation.cuts)
    rounds: list[ExplicitBandRound] = []
    final_separation = initial_separation
    minimax_slack = 0.0
    converged = (
        not seed_keys
        and initial_separation.maximum_normalized_excess
        <= config.separation_tolerance
    )

    for iteration in range(config.max_rounds):
        if converged or not pending:
            break
        new_keys = [key for key in pending if key not in active_set]
        if not new_keys:
            break
        active_keys.extend(new_keys)
        active_set.update(new_keys)
        linear = build_linear_constraints(constraint, model, active_keys, config)
        stage = (
            f"{progress_prefix}{config.mode}: exchange SDP round {iteration + 1}"
        )
        _report_progress(
            progress,
            stage,
            None,
            None,
            event="start",
            active_constraints=len(active_keys),
            added_constraints=len(new_keys),
        )
        started = time.perf_counter()
        candidate = solve_sdp(
            reference,
            shadow_data=shadow_data,
            solver=solver,
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=verbose,
            positivity_conditions="DQG",
            linear_constraints=linear,
            linear_lexicographic_tolerance=config.linear_lexicographic_tolerance,
            linear_fit_only=linear_fit_only,
            initial_d2=current.d2,
            initial_gamma=current.gamma,
        )
        seconds = time.perf_counter() - started
        minimax_slack = float(candidate.linear_max_slack or 0.0)
        _report_progress(
            progress,
            stage,
            None,
            None,
            event="complete",
            solver_status=str(candidate.status),
            linear_fit_status=candidate.linear_fit_status,
            minimax_slack=minimax_slack,
            active_max_absolute_violation=(
                candidate.linear_max_absolute_violation
            ),
            active_max_normalized_violation=(
                candidate.linear_max_normalized_violation
            ),
            seconds=seconds,
        )
        current = candidate
        final_separation = separate_full_grid(
            constraint,
            current.d2,
            current.gamma,
            model,
            config,
            excluded=active_set,
            progress=progress,
            progress_stage=(
                f"{progress_prefix}{config.mode}: full-grid separation round "
                f"{iteration + 1}"
            ),
        )
        threshold = minimax_slack + config.separation_tolerance
        converged = final_separation.maximum_normalized_excess <= threshold
        pending = [
            key
            for key, excess in zip(
                final_separation.cuts, final_separation.cut_excesses
            )
            if excess > threshold and key not in active_set
        ]
        rounds.append(
            ExplicitBandRound(
                iteration=iteration,
                active_constraints=len(active_keys),
                added_constraints=len(new_keys),
                minimax_slack=minimax_slack,
                separation=final_separation,
                solver_status=str(candidate.status),
                active_max_absolute_violation=float(
                    candidate.linear_max_absolute_violation or 0.0
                ),
                active_max_normalized_violation=float(
                    candidate.linear_max_normalized_violation or 0.0
                ),
                seconds=seconds,
            )
        )

    if active_keys:
        final_separation = separate_full_grid(
            constraint,
            current.d2,
            current.gamma,
            model,
            config,
            progress=progress,
            progress_stage=(
                f"{progress_prefix}{config.mode}: final all-row certification"
            ),
        )
        threshold = minimax_slack + config.separation_tolerance
        converged = final_separation.maximum_normalized_excess <= threshold

    return ExplicitBandResult(
        d2=np.asarray(current.d2),
        gamma=np.asarray(current.gamma),
        energy=float(current.energy),
        status=str(current.status),
        mode=config.mode,
        converged=bool(converged),
        active_constraints=len(active_keys),
        active_keys=tuple(active_keys),
        minimax_slack=float(minimax_slack),
        rounds=tuple(rounds),
        initial_separation=initial_separation,
        final_separation=final_separation,
    )


def solve_calibrated_hard_ftpbe_target(
    reference: Any,
    shadow_data: Any,
    initial_result: Any,
    constraint: FullGridHoleConstraint,
    solve_sdp: Any,
    config: ExplicitBandConfig,
    *,
    calibration_margin: float = 0.0,
    solver: str = "SCS",
    tolerance: float = 1e-4,
    max_iterations: int = 50_000,
    verbose: bool = False,
    progress: ProgressCallback | None = None,
) -> CalibratedHardBandResult:
    """Calibrate one ftPBE tolerance without FCI, then impose it as hard rows.

    The first exchange solve finds the smallest uniform additional relative
    tolerance compatible with DQG and the supplied shadows. The second solve
    freezes that scalar into every ftPBE target interval and uses no slack
    variable. Both stages scan the complete configured real-space grid.
    """

    config.validate()
    if config.mode != "ftpbe_target":
        raise ValueError("Calibrated hard solving requires mode='ftpbe_target'.")
    if calibration_margin < 0.0:
        raise ValueError("calibration_margin must be nonnegative.")

    model = constraint.freeze_model(initial_result.d2, initial_result.gamma)
    calibration_config = replace(config, allow_minimax_relaxation=True)
    calibration = solve_explicit_full_grid_band(
        reference,
        shadow_data,
        initial_result,
        constraint,
        solve_sdp,
        calibration_config,
        solver=solver,
        tolerance=tolerance,
        max_iterations=max_iterations,
        verbose=verbose,
        progress=progress,
        frozen_model=model,
        progress_prefix="calibration: ",
        linear_fit_only=True,
    )
    extra = max(
        calibration.minimax_slack,
        calibration.final_separation.maximum_normalized_excess,
    )
    calibrated = config.ftpbe_relative_tolerance + extra + calibration_margin
    hard_config = replace(
        config,
        ftpbe_relative_tolerance=calibrated,
        allow_minimax_relaxation=False,
    )
    hard = solve_explicit_full_grid_band(
        reference,
        shadow_data,
        calibration,
        constraint,
        solve_sdp,
        hard_config,
        solver=solver,
        tolerance=tolerance,
        max_iterations=max_iterations,
        verbose=verbose,
        progress=progress,
        frozen_model=model,
        initial_keys=calibration.active_keys,
        progress_prefix="hard solve: ",
    )
    return CalibratedHardBandResult(
        d2=hard.d2,
        gamma=hard.gamma,
        energy=hard.energy,
        status=hard.status,
        base_relative_tolerance=float(config.ftpbe_relative_tolerance),
        calibration_extra_tolerance=float(extra),
        calibrated_relative_tolerance=float(calibrated),
        calibration_margin=float(calibration_margin),
        calibration=calibration,
        hard=hard,
    )
