"""Data-only weighted-ridge targets and shadow-basis cross-validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from constrained_shadow import PairVectorDesign, ShadowData


@dataclass(frozen=True)
class RidgeTarget:
    d2: np.ndarray
    ridge: float
    effective_ridge: float
    weighted_fit_rmse: float
    lsqr_stop_code: int
    lsqr_iterations: int
    condition_estimate: float


@dataclass(frozen=True)
class RidgeCrossValidation:
    selected_ridge: float
    mean_scores: dict[float, float]
    fold_scores: dict[float, tuple[float, ...]]
    folds: int


def select_shadow_bases(
    shadow_data: ShadowData, indices: Sequence[int]
) -> ShadowData:
    """Select complete random-orbital bases without splitting correlated rows."""

    selected = tuple(int(index) for index in indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Shadow-basis indices must be nonempty and unique.")
    if min(selected) < 0 or max(selected) >= shadow_data.n_shadows:
        raise ValueError("Shadow-basis index is out of range.")
    rows_per_shadow = len(shadow_data.values) // shadow_data.n_shadows
    rows = np.concatenate(
        [
            np.arange(index * rows_per_shadow, (index + 1) * rows_per_shadow)
            for index in selected
        ]
    )
    pair_vectors = np.asarray(shadow_data.pair_vectors[rows])
    design = (
        PairVectorDesign(pair_vectors)
        if isinstance(shadow_data.design, PairVectorDesign)
        else shadow_data.design[rows]
    )
    return ShadowData(
        rotations=tuple(shadow_data.rotations[index] for index in selected),
        pair_vectors=pair_vectors,
        design=design,
        values=shadow_data.values[rows],
        lower_bounds=shadow_data.lower_bounds[rows],
        upper_bounds=shadow_data.upper_bounds[rows],
        hits=shadow_data.hits[rows],
        shots_per_basis=shadow_data.shots_per_basis,
        exact_values=shadow_data.exact_values[rows],
        exact_constraints=shadow_data.exact_constraints,
        occupations=(
            None
            if shadow_data.occupations is None
            else np.asarray(shadow_data.occupations, dtype=np.uint64)[list(selected)]
        ),
    )


def _inverse_standard_errors(shadows: ShadowData) -> np.ndarray:
    probabilities = (shadows.hits.astype(float) + 0.5) / (
        shadows.shots_per_basis + 1.0
    )
    variances = np.maximum(
        probabilities * (1.0 - probabilities) / shadows.shots_per_basis,
        1e-12,
    )
    return 1.0 / np.sqrt(variances)


def weighted_prediction_rmse(shadows: ShadowData, d2: np.ndarray) -> float:
    residual = _inverse_standard_errors(shadows) * (
        shadows.predict(d2) - shadows.values
    )
    return float(np.sqrt(np.mean(residual**2)))


def _symmetric_variables(reference: Any) -> tuple[np.ndarray, np.ndarray]:
    n_spatial = int(reference.n_spatial_orbitals)
    pairs = reference.pairs
    alpha_counts = np.asarray(
        [int(left < n_spatial) + int(right < n_spatial) for left, right in pairs]
    )
    spin_irreps = reference.orbital_irreps + reference.orbital_irreps
    pair_irreps = np.asarray(
        [spin_irreps[left] ^ spin_irreps[right] for left, right in pairs]
    )
    variables = [
        (row, col)
        for row in range(len(pairs))
        for col in range(row, len(pairs))
        if alpha_counts[row] == alpha_counts[col]
        and pair_irreps[row] == pair_irreps[col]
    ]
    return (
        np.asarray([row for row, _ in variables], dtype=int),
        np.asarray([col for _, col in variables], dtype=int),
    )


def weighted_ridge_target(
    reference: Any,
    shadows: ShadowData,
    ridge: float,
    *,
    tolerance: float = 1e-8,
    max_iterations: int | None = None,
) -> RidgeTarget:
    """Fit a symmetric spin/irrep-blocked D2 target from raw frequencies."""

    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    n_pairs = len(reference.pairs)
    rows, cols = _symmetric_variables(reference)
    off_diagonal = rows != cols
    if isinstance(shadows.design, PairVectorDesign):
        vectors = np.asarray(shadows.pair_vectors)
        reduced = np.real(
            np.conjugate(vectors[:, rows]) * vectors[:, cols]
        )
        reduced[:, off_diagonal] *= 2.0
        design = sparse.csr_matrix(reduced)
    else:
        primary = rows * n_pairs + cols
        secondary = cols * n_pairs + rows
        design = shadows.design[:, primary].tocsr()
        design = design + shadows.design[:, secondary] @ sparse.diags(
            off_diagonal.astype(float), format="csr"
        )
    inverse_errors = _inverse_standard_errors(shadows)
    weighted_design = sparse.diags(inverse_errors, format="csr") @ design
    weighted_values = inverse_errors * shadows.values
    information_scale = float(
        np.dot(weighted_design.data, weighted_design.data) / max(len(rows), 1)
    )
    effective_ridge = float(ridge) * max(information_scale, 1e-16)
    iteration_limit = (
        max_iterations
        if max_iterations is not None
        else max(2000, 4 * len(rows))
    )
    solution = sparse_linalg.lsqr(
        weighted_design,
        weighted_values,
        damp=math.sqrt(effective_ridge),
        atol=tolerance,
        btol=tolerance,
        iter_lim=iteration_limit,
    )
    coefficients = np.asarray(solution[0], dtype=float)
    d2 = np.zeros((n_pairs, n_pairs), dtype=float)
    d2[rows, cols] = coefficients
    d2[cols, rows] = coefficients
    return RidgeTarget(
        d2=d2,
        ridge=float(ridge),
        effective_ridge=effective_ridge,
        weighted_fit_rmse=weighted_prediction_rmse(shadows, d2),
        lsqr_stop_code=int(solution[1]),
        lsqr_iterations=int(solution[2]),
        condition_estimate=float(solution[6]),
    )


def cross_validate_ridge(
    reference: Any,
    shadows: ShadowData,
    ridge_grid: Sequence[float],
    *,
    folds: int = 5,
) -> RidgeCrossValidation:
    """Choose ridge by held-out complete-shadow weighted prediction error."""

    ridges = tuple(float(value) for value in ridge_grid)
    if not ridges or min(ridges) < 0.0:
        raise ValueError("ridge_grid must contain nonnegative values.")
    if shadows.n_shadows < 2:
        raise ValueError("At least two shadows are required for cross-validation.")
    fold_count = min(max(int(folds), 2), shadows.n_shadows)
    assignments = np.arange(shadows.n_shadows) % fold_count
    fold_scores: dict[float, tuple[float, ...]] = {}
    mean_scores: dict[float, float] = {}
    for ridge in ridges:
        scores = []
        for fold in range(fold_count):
            validation_indices = np.flatnonzero(assignments == fold)
            training_indices = np.flatnonzero(assignments != fold)
            training = select_shadow_bases(shadows, training_indices)
            validation = select_shadow_bases(shadows, validation_indices)
            target = weighted_ridge_target(reference, training, ridge)
            scores.append(weighted_prediction_rmse(validation, target.d2) ** 2)
        fold_scores[ridge] = tuple(float(value) for value in scores)
        mean_scores[ridge] = float(np.mean(scores))
    selected = min(ridges, key=lambda value: (mean_scores[value], value))
    return RidgeCrossValidation(
        selected_ridge=float(selected),
        mean_scores=mean_scores,
        fold_scores=fold_scores,
        folds=fold_count,
    )
