"""Particle-number-conserving orbital frames and measurement design."""

from __future__ import annotations

from math import ceil, comb

import numpy as np

from .polynomial import features


def haar_frame(n_modes: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a Haar-random real special-orthogonal frame (directions are rows)."""
    raw = rng.standard_normal((n_modes, n_modes))
    q, r = np.linalg.qr(raw)
    q *= np.where(np.diag(r) < 0.0, -1.0, 1.0)
    if np.linalg.det(q) < 0.0:
        q[-1] *= -1.0
    return q


def complete_frame(direction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Complete one normalized target direction to a real orthonormal frame."""
    first = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(first)
    if norm <= 0.0:
        raise ValueError("direction must be nonzero")
    first = first / norm

    rows = [first]
    while len(rows) < len(first):
        trial = rng.standard_normal(len(first))
        for row in rows:
            trial -= np.dot(row, trial) * row
        trial_norm = np.linalg.norm(trial)
        if trial_norm > 1e-12:
            rows.append(trial / trial_norm)
    frame = np.asarray(rows)
    if np.linalg.det(frame) < 0.0:
        frame[-1] *= -1.0
    return frame


def frames_from_direction_blocks(
    directions: np.ndarray,
    n_modes: int | None = None,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Build deterministic, orbital-gauge-covariant frames from target blocks.

    Each consecutive block supplies the preferred directions for one frame.
    Modified Gram-Schmidt is completed with the direction having the largest
    residual norm from the full pool.  Because selection uses only inner
    products, a common active-orbital rotation simply right-rotates every
    returned frame.  No coordinate-fixed random completion is introduced.
    """
    targets = np.asarray(directions, dtype=float)
    if targets.ndim != 2:
        raise ValueError("directions must have shape (M,N)")
    modes = targets.shape[1] if n_modes is None else int(n_modes)
    if targets.shape[1] != modes or modes < 1 or tolerance <= 0.0:
        raise ValueError("invalid direction block dimensions or tolerance")
    norms = np.linalg.norm(targets, axis=1)
    if np.any(norms <= tolerance):
        raise ValueError("all target directions must be nonzero")
    unit = targets / norms[:, None]

    frames = []
    for start in range(0, len(unit), modes):
        preferred = list(range(start, min(start + modes, len(unit))))
        rows = []
        used: set[int] = set()

        def residual(index: int) -> np.ndarray:
            vector = unit[index].copy()
            for row in rows:
                vector -= np.dot(row, vector) * row
            return vector

        for index in preferred:
            vector = residual(index)
            norm = np.linalg.norm(vector)
            if norm > tolerance:
                rows.append(vector / norm)
                used.add(index)

        while len(rows) < modes:
            candidates = [index for index in range(len(unit)) if index not in used]
            if not candidates:
                raise np.linalg.LinAlgError("direction pool cannot span a complete frame")
            residuals = [residual(index) for index in candidates]
            residual_norms = np.asarray([np.linalg.norm(vector) for vector in residuals])
            choice = int(np.argmax(residual_norms))
            if residual_norms[choice] <= tolerance:
                raise np.linalg.LinAlgError("direction pool is rank deficient")
            rows.append(residuals[choice] / residual_norms[choice])
            used.add(candidates[choice])

        frame = np.asarray(rows)
        # The even-degree observables are sign-blind, but a positive
        # determinant is convenient for circuit decompositions.
        if np.linalg.det(frame) < 0.0:
            frame[-1] *= -1.0
        frames.append(frame)
    return np.asarray(frames)


def frame_feature_matrix(frames: np.ndarray, degree: int = 4) -> np.ndarray:
    """Stack the N direct polynomial probes supplied by every frame."""
    array = np.asarray(frames, dtype=float)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError("frames must have shape (F,N,N)")
    return features(array.reshape(-1, array.shape[-1]), degree)


def minimum_frame_count(n_modes: int, degree: int = 4) -> int:
    """Counting lower bound when each frame supplies N direct probes."""
    return ceil(comb(n_modes + degree - 1, degree) / n_modes)


def identifiability_metrics(frames: np.ndarray, degree: int = 4, tol: float = 1e-10) -> dict:
    """Rank and conditioning diagnostics for a frame design."""
    matrix = frame_feature_matrix(frames, degree)
    singular = np.linalg.svd(matrix, compute_uv=False)
    threshold = tol * singular[0] if len(singular) else 0.0
    rank = int(np.count_nonzero(singular > threshold))
    condition = float(singular[0] / singular[rank - 1]) if rank else float("inf")
    return {
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "rank": rank,
        "condition": condition,
        "sigma_min_identifiable": float(singular[rank - 1]) if rank else 0.0,
        "sigma_max": float(singular[0]) if len(singular) else 0.0,
    }


def greedy_d_optimal_frames(
    candidates: np.ndarray,
    n_select: int,
    degree: int = 4,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Select whole frames with a greedy regularized D-optimal criterion.

    The matrix determinant lemma reduces every candidate score to an N by N
    determinant, and Woodbury updates avoid repeated K by K inversions.
    """
    frames = np.asarray(candidates, dtype=float)
    if frames.ndim != 3 or frames.shape[1] != frames.shape[2]:
        raise ValueError("candidates must have shape (C,N,N)")
    if not 1 <= n_select <= len(frames):
        raise ValueError("n_select must be between one and the candidate count")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")

    blocks = np.asarray([features(frame, degree) for frame in frames])
    n_features = blocks.shape[-1]
    inverse_information = np.eye(n_features) / ridge
    remaining = list(range(len(frames)))
    selected = []

    for _ in range(n_select):
        best_index = None
        best_score = -np.inf
        for index in remaining:
            block = blocks[index]
            small = np.eye(block.shape[0]) + block @ inverse_information @ block.T
            sign, score = np.linalg.slogdet(small)
            if sign > 0 and score > best_score:
                best_score = float(score)
                best_index = index
        if best_index is None:
            raise np.linalg.LinAlgError("no positive-definite D-optimal update found")

        block = blocks[best_index]
        small = np.eye(block.shape[0]) + block @ inverse_information @ block.T
        gain = inverse_information @ block.T
        inverse_information -= gain @ np.linalg.solve(small, gain.T)
        inverse_information = 0.5 * (inverse_information + inverse_information.T)
        selected.append(best_index)
        remaining.remove(best_index)

    indices = np.asarray(selected, dtype=int)
    return frames[indices], indices


def greedy_joint_d_optimal_frames(
    candidates: np.ndarray,
    n_select: int,
    quartic_weight: float = 1.0,
    quadratic_weight: float = 1.0,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """D-optimal whole-frame design for quartic and quadratic forms together."""
    frames = np.asarray(candidates, dtype=float)
    if frames.ndim != 3 or frames.shape[1] != frames.shape[2]:
        raise ValueError("candidates must have shape (C,N,N)")
    if not 1 <= n_select <= len(frames):
        raise ValueError("n_select must be between one and the candidate count")
    if min(quartic_weight, quadratic_weight, ridge) <= 0.0:
        raise ValueError("weights and ridge must be positive")

    block4 = np.asarray([np.sqrt(quartic_weight) * features(frame, 4) for frame in frames])
    block2 = np.asarray([np.sqrt(quadratic_weight) * features(frame, 2) for frame in frames])
    inverse4 = np.eye(block4.shape[-1]) / ridge
    inverse2 = np.eye(block2.shape[-1]) / ridge
    remaining = list(range(len(frames)))
    selected = []

    for _ in range(n_select):
        best_index = None
        best_score = -np.inf
        for index in remaining:
            small4 = np.eye(frames.shape[1]) + block4[index] @ inverse4 @ block4[index].T
            small2 = np.eye(frames.shape[1]) + block2[index] @ inverse2 @ block2[index].T
            sign4, score4 = np.linalg.slogdet(small4)
            sign2, score2 = np.linalg.slogdet(small2)
            score = score4 + score2
            if sign4 > 0 and sign2 > 0 and score > best_score:
                best_score = float(score)
                best_index = index
        if best_index is None:
            raise np.linalg.LinAlgError("no positive-definite joint D-optimal update found")

        for block, inverse in ((block4[best_index], inverse4), (block2[best_index], inverse2)):
            small = np.eye(block.shape[0]) + block @ inverse @ block.T
            gain = inverse @ block.T
            inverse -= gain @ np.linalg.solve(small, gain.T)
            inverse[:] = 0.5 * (inverse + inverse.T)
        selected.append(best_index)
        remaining.remove(best_index)

    indices = np.asarray(selected, dtype=int)
    return frames[indices], indices


def greedy_guarded_target_frames(
    candidates: np.ndarray,
    n_select: int,
    quartic_gradient: np.ndarray,
    quadratic_gradient: np.ndarray,
    n_guard: int | None = None,
    ridge: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine an identifiability guard with target-variance frame selection.

    The first ``n_guard`` frames are selected by joint D-optimality.  Remaining
    frames greedily reduce the linearized variance of the supplied total-energy
    gradients.  This avoids the rank collapse of a pure c-optimal design.
    """
    frames = np.asarray(candidates, dtype=float)
    if frames.ndim != 3 or frames.shape[1] != frames.shape[2]:
        raise ValueError("candidates must have shape (C,N,N)")
    if not 1 <= n_select <= len(frames):
        raise ValueError("n_select must be between one and the candidate count")
    if ridge <= 0.0:
        raise ValueError("ridge must be positive")

    gradient4 = np.asarray(quartic_gradient, dtype=float)
    gradient2 = np.asarray(quadratic_gradient, dtype=float)
    expected4 = features(frames[0], 4).shape[1]
    expected2 = features(frames[0], 2).shape[1]
    if gradient4.shape != (expected4,) or gradient2.shape != (expected2,):
        raise ValueError("target gradients have incompatible dimensions")

    lower_bound = minimum_frame_count(frames.shape[1], degree=4)
    if n_guard is None:
        n_guard = max(lower_bound, int(np.ceil(0.6 * n_select)))
    if not lower_bound <= n_guard <= n_select:
        raise ValueError("n_guard must ensure quartic identifiability and not exceed n_select")

    guarded, guarded_indices = greedy_joint_d_optimal_frames(
        frames, n_guard, ridge=ridge
    )
    blocks4 = np.asarray([features(frame, 4) for frame in frames])
    blocks2 = np.asarray([features(frame, 2) for frame in frames])
    information4 = ridge * np.eye(expected4)
    information2 = ridge * np.eye(expected2)
    for index in guarded_indices:
        information4 += blocks4[index].T @ blocks4[index]
        information2 += blocks2[index].T @ blocks2[index]
    inverse4 = np.linalg.inv(information4)
    inverse2 = np.linalg.inv(information2)

    selected = list(map(int, guarded_indices))
    remaining = [index for index in range(len(frames)) if index not in set(selected)]
    while len(selected) < n_select:
        best_index = None
        best_reduction = -np.inf
        for index in remaining:
            reduction = 0.0
            for block, inverse, gradient in (
                (blocks4[index], inverse4, gradient4),
                (blocks2[index], inverse2, gradient2),
            ):
                small = np.eye(block.shape[0]) + block @ inverse @ block.T
                projected = block @ inverse @ gradient
                reduction += float(projected @ np.linalg.solve(small, projected))
            if reduction > best_reduction:
                best_reduction = reduction
                best_index = index
        if best_index is None:
            raise np.linalg.LinAlgError("guarded target design could not select a frame")

        for block, inverse in (
            (blocks4[best_index], inverse4),
            (blocks2[best_index], inverse2),
        ):
            small = np.eye(block.shape[0]) + block @ inverse @ block.T
            gain = inverse @ block.T
            inverse -= gain @ np.linalg.solve(small, gain.T)
            inverse[:] = 0.5 * (inverse + inverse.T)
        selected.append(best_index)
        remaining.remove(best_index)

    indices = np.asarray(selected, dtype=int)
    return frames[indices], indices
