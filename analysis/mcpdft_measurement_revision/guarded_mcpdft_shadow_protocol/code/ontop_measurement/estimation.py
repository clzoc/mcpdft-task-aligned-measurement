"""Joint estimators for density and on-top quartic-form measurements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .polynomial import exponent_table, features, monomial_normalization


@dataclass(frozen=True)
class FrameSummary:
    """Sufficient marginal statistics retained from one orbital frame."""

    mean: np.ndarray
    covariance: np.ndarray
    counts: np.ndarray
    shots: int


def normalize_directions(directions: np.ndarray) -> np.ndarray:
    """Normalize orbital directions before interpreting occupations as probabilities."""
    array = np.asarray(directions, dtype=float)
    if array.ndim != 2:
        raise ValueError("directions must have shape (M,N)")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1e-14):
        raise ValueError("probability constraints require nonzero directions")
    return array / norms[:, None]


def summarize_bitstrings(bits: np.ndarray) -> FrameSummary:
    """Summarize joint alpha/beta readout from one frame.

    The input shape is ``(shots, 2, n_modes)``.  The output mean is ordered as
    all pair bits followed by all spin-summed occupation bits.  Its covariance
    is the covariance of that sample mean, including every within-frame and
    rho/Pi cross-covariance.
    """
    sample = np.asarray(bits, dtype=np.int8)
    if sample.ndim != 3 or sample.shape[1] != 2 or sample.shape[0] < 2:
        raise ValueError("bits must have shape (shots>=2, 2, n_modes)")
    if np.any((sample != 0) & (sample != 1)):
        raise ValueError("bitstrings must contain only zero and one")

    alpha = sample[:, 0, :]
    beta = sample[:, 1, :]
    pair = alpha * beta
    total = alpha + beta
    outcomes = np.concatenate([pair, total], axis=1).astype(float)
    covariance = np.cov(outcomes, rowvar=False, ddof=1) / sample.shape[0]

    counts = np.empty((sample.shape[2], 4), dtype=np.int64)
    counts[:, 0] = np.sum((alpha == 1) & (beta == 1), axis=0)
    counts[:, 1] = np.sum((alpha == 1) & (beta == 0), axis=0)
    counts[:, 2] = np.sum((alpha == 0) & (beta == 1), axis=0)
    counts[:, 3] = np.sum((alpha == 0) & (beta == 0), axis=0)
    return FrameSummary(
        mean=outcomes.mean(axis=0),
        covariance=np.asarray(covariance, dtype=float),
        counts=counts,
        shots=int(sample.shape[0]),
    )


def joint_design_block(frame: np.ndarray) -> np.ndarray:
    """Linear design for pair means followed by total-occupation means."""
    directions = np.asarray(frame, dtype=float)
    x4 = features(directions, 4)
    x2 = features(directions, 2)
    zeros42 = np.zeros((len(directions), x2.shape[1]))
    zeros24 = np.zeros((len(directions), x4.shape[1]))
    return np.block([[x4, zeros42], [zeros24, x2]])


def whiten_frame_data(
    frames: np.ndarray,
    summaries: list[FrameSummary],
    shrinkage: float = 0.05,
    relative_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a stable GLS system with empirical within-frame covariance."""
    array = np.asarray(frames, dtype=float)
    if len(array) != len(summaries):
        raise ValueError("one summary is required for every frame")
    if not 0.0 <= shrinkage <= 1.0 or relative_floor <= 0.0:
        raise ValueError("invalid covariance regularization")

    whitened_design = []
    whitened_means = []
    for frame, summary in zip(array, summaries):
        design = joint_design_block(frame)
        if summary.mean.shape != (design.shape[0],):
            raise ValueError("summary dimension does not match its frame")
        covariance = 0.5 * (summary.covariance + summary.covariance.T)
        diagonal = np.diag(np.maximum(np.diag(covariance), 0.0))
        regularized = (1.0 - shrinkage) * covariance + shrinkage * diagonal
        eigenvalues, eigenvectors = np.linalg.eigh(regularized)
        scale = max(float(np.max(eigenvalues)), 1.0 / summary.shots)
        eigenvalues = np.maximum(eigenvalues, relative_floor * scale)
        whitener = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
        whitened_design.append(whitener @ design)
        whitened_means.append(whitener @ summary.mean)
    return np.vstack(whitened_design), np.concatenate(whitened_means)


def fit_joint_gls(
    frames: np.ndarray,
    summaries: list[FrameSummary],
    ridge: float = 1e-10,
    shrinkage: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit quartic and quadratic coefficients with covariance-aware GLS."""
    design, values = whiten_frame_data(frames, summaries, shrinkage=shrinkage)
    if ridge > 0.0:
        identity = np.sqrt(ridge) * np.eye(design.shape[1])
        design_fit = np.vstack([design, identity])
        values_fit = np.concatenate([values, np.zeros(design.shape[1])])
    else:
        design_fit, values_fit = design, values
    coefficients, residuals, rank, singular = np.linalg.lstsq(design_fit, values_fit, rcond=None)
    n4 = len(exponent_table(np.asarray(frames).shape[-1], 4))
    diagnostics = {
        "rank": int(rank),
        "columns": int(design.shape[1]),
        "weighted_residual_norm": float(np.linalg.norm(design @ coefficients - values)),
        "condition": float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf"),
    }
    return coefficients[:n4], coefficients[n4:], diagnostics


def spin_balanced_probabilities(
    directions: np.ndarray,
    quartic_coefficients: np.ndarray,
    quadratic_coefficients: np.ndarray,
) -> np.ndarray:
    """Return [p11,p10,p01,p00] for every direction in a spin-balanced state."""
    direction_array = normalize_directions(directions)
    pair = features(direction_array, 4) @ np.asarray(quartic_coefficients)
    total = features(direction_array, 2) @ np.asarray(quadratic_coefficients)
    single = 0.5 * total - pair
    empty = 1.0 - total + pair
    return np.stack([pair, single, single, empty], axis=-1)


def occupation_probabilities(
    directions: np.ndarray,
    quartic_coefficients: np.ndarray,
    quadratic_coefficients: np.ndarray,
) -> np.ndarray:
    """Return [double, single-total, empty] without assuming spin balance.

    The measured pair probability and spin-summed occupation determine these
    three probabilities even when the alpha and beta single-occupation
    probabilities differ.  This is the strongest physicality condition that
    can be imposed without measuring the spin difference.
    """
    direction_array = normalize_directions(directions)
    pair = features(direction_array, 4) @ np.asarray(quartic_coefficients)
    total = features(direction_array, 2) @ np.asarray(quadratic_coefficients)
    single_total = total - 2.0 * pair
    empty = 1.0 - total + pair
    return np.stack([pair, single_total, empty], axis=-1)


def physicality_violations(
    directions: np.ndarray,
    quartic_coefficients: np.ndarray,
    quadratic_coefficients: np.ndarray,
    tolerance: float = 1e-10,
) -> dict:
    probabilities = occupation_probabilities(
        directions, quartic_coefficients, quadratic_coefficients
    )
    return {
        "minimum_probability": float(np.min(probabilities)),
        "maximum_probability": float(np.max(probabilities)),
        "negative_fraction": float(np.mean(probabilities < -tolerance)),
        "normalization_error": float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))),
    }


def _isotropic_initial(n_modes: int, n_electrons: float | None) -> np.ndarray:
    total = 1.0 if n_electrons is None else float(n_electrons) / n_modes
    total = float(np.clip(total, 0.05, 1.95))
    lower = max(1e-4, total - 1.0 + 1e-4)
    upper = total / 2.0 - 1e-4
    pair = float(np.clip(total * total / 4.0, lower, upper))

    exp4 = exponent_table(n_modes, 4)
    exp2 = exponent_table(n_modes, 2)
    c4 = np.zeros(len(exp4))
    c2 = np.zeros(len(exp2))
    for index, exponent in enumerate(exp2):
        if np.count_nonzero(exponent == 2) == 1:
            c2[index] = total / monomial_normalization(n_modes, 2)[index]
    for index, exponent in enumerate(exp4):
        if np.count_nonzero(exponent == 4) == 1:
            c4[index] = pair / monomial_normalization(n_modes, 4)[index]
        elif np.count_nonzero(exponent == 2) == 2:
            c4[index] = 2.0 * pair / monomial_normalization(n_modes, 4)[index]
    return np.concatenate([c4, c2])


def _physical_linear_constraint(directions: np.ndarray) -> LinearConstraint:
    unit = normalize_directions(directions)
    x4 = features(unit, 4)
    x2 = features(unit, 2)
    z42 = np.zeros((len(directions), x2.shape[1]))
    z24 = np.zeros((len(directions), x4.shape[1]))
    matrix = np.vstack(
        [
            np.hstack([x4, z42]),
            np.hstack([z24, x2]),
            np.hstack([-x4, 0.5 * x2]),
            np.hstack([x4, -x2]),
        ]
    )
    count = len(directions)
    lower = np.concatenate(
        [np.zeros(count), np.zeros(count), np.zeros(count), -np.ones(count)]
    )
    upper = np.concatenate(
        [
            np.full(count, np.inf),
            np.full(count, 2.0),
            np.full(count, np.inf),
            np.full(count, np.inf),
        ]
    )
    return LinearConstraint(matrix, lower, upper)


def fit_joint_constrained_gls(
    frames: np.ndarray,
    summaries: list[FrameSummary],
    validation_directions: np.ndarray | None = None,
    n_electrons: float | None = None,
    ridge: float = 1e-10,
    shrinkage: float = 0.05,
    maxiter: int = 1000,
    initial_coefficients: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Joint GLS with spin-balanced marginal physicality constraints.

    The constraints enforce p11, p10, p01, and p00 >= 0 on all supplied
    directions.  They are linear in the polynomial coefficients.  When the
    active electron number is supplied, trace(gamma) is imposed exactly.
    """
    frame_array = np.asarray(frames, dtype=float)
    design, values = whiten_frame_data(frame_array, summaries, shrinkage=shrinkage)
    n_modes = frame_array.shape[-1]
    n4 = len(exponent_table(n_modes, 4))
    n2 = len(exponent_table(n_modes, 2))
    directions = (
        frame_array.reshape(-1, n_modes)
        if validation_directions is None
        else np.asarray(validation_directions, dtype=float)
    )
    constraints: list[LinearConstraint] = [_physical_linear_constraint(directions)]

    if n_electrons is not None:
        trace_row = np.zeros(n4 + n2)
        for index, exponent in enumerate(exponent_table(n_modes, 2)):
            if np.count_nonzero(exponent == 2) == 1:
                trace_row[n4 + index] = monomial_normalization(n_modes, 2)[index]
        constraints.append(LinearConstraint(trace_row[None, :], n_electrons, n_electrons))

    objective_scale = max(
        1.0,
        float(np.linalg.norm(design, ord="fro") ** 2 / max(1, design.shape[1])),
    )

    def objective(coefficients: np.ndarray) -> float:
        residual = design @ coefficients - values
        return 0.5 * float(
            residual @ residual + ridge * coefficients @ coefficients
        ) / objective_scale

    def gradient(coefficients: np.ndarray) -> np.ndarray:
        return (
            design.T @ (design @ coefficients - values) + ridge * coefficients
        ) / objective_scale

    initial = (
        _isotropic_initial(n_modes, n_electrons)
        if initial_coefficients is None
        else np.asarray(initial_coefficients, dtype=float)
    )
    if initial.shape != (n4 + n2,):
        raise ValueError("initial_coefficients has an incompatible dimension")
    result = minimize(
        objective,
        initial,
        jac=gradient,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-9, "maxiter": maxiter, "disp": False},
    )
    if not result.success and initial_coefficients is not None:
        result = minimize(
            objective,
            _isotropic_initial(n_modes, n_electrons),
            jac=gradient,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-9, "maxiter": 2 * maxiter, "disp": False},
        )
    if not result.success:
        raise RuntimeError(f"constrained GLS failed: {result.message}")

    diagnostics = {
        "success": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        **physicality_violations(directions, result.x[:n4], result.x[n4:]),
    }
    return result.x[:n4], result.x[n4:], diagnostics


def fit_joint_adaptive_constrained_gls(
    frames: np.ndarray,
    summaries: list[FrameSummary],
    validation_directions: np.ndarray,
    n_electrons: float | None = None,
    ridge: float = 1e-10,
    shrinkage: float = 0.05,
    tolerance: float = 1e-8,
    batch_size: int = 32,
    max_rounds: int = 12,
    maxiter: int = 1000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Constrained GLS with cutting-plane physicality over a large grid.

    Starting from all measured frame directions, the fit repeatedly adds the
    worst violating validation directions.  The final coefficients therefore
    define mutually consistent field values and spatial gradients; no
    pointwise clipping is used after reconstruction.
    """
    if tolerance <= 0.0 or batch_size < 1 or max_rounds < 1:
        raise ValueError("adaptive constraint controls must be positive")

    frame_array = np.asarray(frames, dtype=float)
    validation = normalize_directions(validation_directions)
    active = normalize_directions(frame_array.reshape(-1, frame_array.shape[-1]))
    selected: set[int] = set()
    raw4, raw2, raw_diagnostics = fit_joint_gls(
        frame_array, summaries, ridge=ridge, shrinkage=shrinkage
    )
    raw_probabilities = occupation_probabilities(validation, raw4, raw2)
    raw_minimum = float(np.min(raw_probabilities))
    trace = 0.0
    for index, exponent in enumerate(exponent_table(frame_array.shape[-1], 2)):
        if np.count_nonzero(exponent == 2) == 1:
            trace += monomial_normalization(frame_array.shape[-1], 2)[index] * raw2[index]
    trace_error = 0.0 if n_electrons is None else abs(trace - n_electrons)
    if raw_minimum >= -tolerance and trace_error <= tolerance:
        return raw4, raw2, {
            **raw_diagnostics,
            "success": True,
            "constraint_rounds": 0,
            "added_validation_directions": 0,
            "minimum_validation_probability": raw_minimum,
            "particle_number_error": trace_error,
            "constraint_history": [],
        }

    initial = None
    history = []

    for round_index in range(max_rounds):
        quartic, quadratic, diagnostics = fit_joint_constrained_gls(
            frame_array,
            summaries,
            validation_directions=active,
            n_electrons=n_electrons,
            ridge=ridge,
            shrinkage=shrinkage,
            maxiter=maxiter,
            initial_coefficients=initial,
        )
        probabilities = occupation_probabilities(validation, quartic, quadratic)
        minima = np.min(probabilities, axis=1)
        worst = float(np.min(minima))
        violating = np.flatnonzero(minima < -tolerance)
        history.append(
            {
                "round": round_index + 1,
                "active_directions": int(len(active)),
                "violating_validation_directions": int(len(violating)),
                "minimum_validation_probability": worst,
            }
        )
        if len(violating) == 0:
            return quartic, quadratic, {
                **diagnostics,
                "constraint_rounds": round_index + 1,
                "added_validation_directions": len(selected),
                "minimum_validation_probability": worst,
                "constraint_history": history,
            }

        ordered = violating[np.argsort(minima[violating])]
        additions = [int(index) for index in ordered if int(index) not in selected][
            :batch_size
        ]
        if not additions:
            raise RuntimeError("adaptive constraints stalled with unresolved violations")
        selected.update(additions)
        active = np.vstack([active, validation[additions]])
        initial = np.concatenate([quartic, quadratic])

    final_probabilities = occupation_probabilities(validation, quartic, quadratic)
    worst = float(np.min(final_probabilities))
    if worst < -tolerance:
        raise RuntimeError(
            "adaptive constrained GLS did not satisfy the validation pool: "
            f"minimum probability {worst:.3e}"
        )
    return quartic, quadratic, {
        **diagnostics,
        "constraint_rounds": max_rounds,
        "added_validation_directions": len(selected),
        "minimum_validation_probability": worst,
        "constraint_history": history,
    }
