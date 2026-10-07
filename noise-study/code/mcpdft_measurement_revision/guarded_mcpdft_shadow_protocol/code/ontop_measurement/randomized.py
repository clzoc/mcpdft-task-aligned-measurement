"""Random single-particle-basis baseline for contracted polynomial tomography."""

from __future__ import annotations

from functools import lru_cache
from math import prod

import numpy as np

from .estimation import FrameSummary
from .polynomial import exponent_table, features, monomial_normalization


def _odd_double_factorial(value: int) -> int:
    if value <= 0:
        return 1
    return prod(range(value, 0, -2))


def sphere_monomial_moment(exponents: np.ndarray, n_modes: int) -> float:
    """Exact monomial moment for a uniform real unit vector in R^N."""
    powers = np.asarray(exponents, dtype=int)
    if powers.ndim != 1 or len(powers) != n_modes or np.any(powers < 0):
        raise ValueError("invalid sphere-moment exponents")
    if np.any(powers % 2):
        return 0.0
    half_degree = int(np.sum(powers) // 2)
    numerator = prod(_odd_double_factorial(int(power) - 1) for power in powers)
    denominator = prod(n_modes + 2 * index for index in range(half_degree))
    return float(numerator / denominator) if denominator else 1.0


@lru_cache(maxsize=None)
def sphere_feature_gram(n_modes: int, degree: int) -> np.ndarray:
    """Return E[f_d(u) f_d(u)^T] for u uniform on the real unit sphere."""
    exponents = exponent_table(n_modes, degree)
    scales = monomial_normalization(n_modes, degree)
    gram = np.empty((len(exponents), len(exponents)), dtype=float)
    for i, alpha in enumerate(exponents):
        for j in range(i, len(exponents)):
            value = (
                scales[i]
                * scales[j]
                * sphere_monomial_moment(alpha + exponents[j], n_modes)
            )
            gram[i, j] = gram[j, i] = value
    return gram


def randomized_frame_inverse(
    frames: np.ndarray,
    summaries: list[FrameSummary],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Invert Haar-frame occupation moments for the visible quartic/quadratic tensors.

    This inverse retains the N same-mode polynomial values from every
    execution. Averaged over a Haar frame, each direction is uniform on the
    sphere, so the exact feature Gram matrix gives an unbiased linear inverse.
    Cross-mode occupation products in the same bitstrings are outside this
    observation model. Repeating shots per random frame is allowed, but
    reduces only outcome noise; the number of independent random bases remains
    a separate resource.
    """
    frame_array = np.asarray(frames, dtype=float)
    if frame_array.ndim != 3 or frame_array.shape[1] != frame_array.shape[2]:
        raise ValueError("frames must have shape (F,N,N)")
    if len(frame_array) != len(summaries):
        raise ValueError("one summary is required for each random frame")
    n_modes = frame_array.shape[-1]
    score4 = np.zeros(len(exponent_table(n_modes, 4)))
    score2 = np.zeros(len(exponent_table(n_modes, 2)))
    total_executions = 0
    for frame, summary in zip(frame_array, summaries):
        if summary.mean.shape != (2 * n_modes,):
            raise ValueError("summary dimension does not match the frame")
        weight = summary.shots
        score4 += weight * features(frame, 4).T @ summary.mean[:n_modes]
        score2 += weight * features(frame, 2).T @ summary.mean[n_modes:]
        total_executions += weight
    denominator = total_executions * n_modes
    score4 /= denominator
    score2 /= denominator

    gram4 = sphere_feature_gram(n_modes, 4)
    gram2 = sphere_feature_gram(n_modes, 2)
    quartic = np.linalg.solve(gram4, score4)
    quadratic = np.linalg.solve(gram2, score2)
    return quartic, quadratic, {
        "random_frames": len(frame_array),
        "total_executions": int(total_executions),
        "shots_per_random_frame_min": min(summary.shots for summary in summaries),
        "shots_per_random_frame_max": max(summary.shots for summary in summaries),
        "quartic_inverse_condition": float(np.linalg.cond(gram4)),
        "quadratic_inverse_condition": float(np.linalg.cond(gram2)),
    }
