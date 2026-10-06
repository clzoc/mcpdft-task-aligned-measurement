"""Homogeneous-polynomial representation of the MC-PDFT grid fields.

For real active orbitals, rho is a quadratic form in the orbital-value vector
and the active on-top pair density is a homogeneous quartic form.  Only the
fully symmetric part of the spin-free 2-RDM is visible to the latter.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations_with_replacement, permutations
from math import factorial

import numpy as np


@lru_cache(maxsize=None)
def exponent_table(n_modes: int, degree: int) -> np.ndarray:
    """Return monomial exponents in combinations-with-replacement order."""
    if n_modes < 1:
        raise ValueError("n_modes must be positive")
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    rows = []
    for indices in combinations_with_replacement(range(n_modes), degree):
        row = np.zeros(n_modes, dtype=np.int16)
        for index in indices:
            row[index] += 1
        rows.append(row)
    return np.asarray(rows, dtype=np.int16)


@lru_cache(maxsize=None)
def monomial_normalization(n_modes: int, degree: int) -> np.ndarray:
    """Normalization making symmetric-tensor features rotation-orthonormal."""
    exps = exponent_table(n_modes, degree)
    return np.sqrt(
        np.asarray(
            [factorial(degree) / np.prod([factorial(int(a)) for a in row]) for row in exps],
            dtype=float,
        )
    )


def _raw_features(values: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    out = np.ones(values.shape[:-1] + (len(exponents),), dtype=float)
    for mode in range(values.shape[-1]):
        out *= values[..., mode, None] ** exponents[:, mode]
    return out


def features(
    x: np.ndarray,
    degree: int,
    exponents: np.ndarray | None = None,
    normalized: bool = True,
) -> np.ndarray:
    """Evaluate symmetric-tensor features at one or more vectors.

    With the default normalization, ``sum_k features(x,d)[k]**2`` equals
    ``(x @ x)**d`` and orbital rotations act orthogonally on feature space.
    """
    values = np.asarray(x, dtype=float)
    if values.ndim < 1:
        raise ValueError("x must have a mode axis")
    exps = exponent_table(values.shape[-1], degree) if exponents is None else np.asarray(exponents)
    if exps.ndim != 2 or exps.shape[1] != values.shape[-1]:
        raise ValueError("exponents and x have incompatible mode dimensions")

    out = _raw_features(values, exps)
    if normalized:
        if exponents is None:
            scale = monomial_normalization(values.shape[-1], degree)
        else:
            scale = np.sqrt(
                np.asarray(
                    [
                        factorial(degree)
                        / np.prod([factorial(int(a)) for a in row])
                        for row in exps
                    ],
                    dtype=float,
                )
            )
        out *= scale
    return out


def feature_jacobian(
    x: np.ndarray,
    degree: int,
    exponents: np.ndarray | None = None,
    normalized: bool = True,
) -> np.ndarray:
    """Return d feature_k / d x_p with shape ``x.shape[:-1] + (K, N)``."""
    values = np.asarray(x, dtype=float)
    exps = exponent_table(values.shape[-1], degree) if exponents is None else np.asarray(exponents)
    jac = np.zeros(values.shape[:-1] + (len(exps), values.shape[-1]), dtype=float)
    for mode in range(values.shape[-1]):
        active = exps[:, mode] > 0
        if not np.any(active):
            continue
        reduced = exps[active].copy()
        prefactor = reduced[:, mode].astype(float)
        reduced[:, mode] -= 1
        derivative = _raw_features(values, reduced) * prefactor
        if normalized:
            derivative *= monomial_normalization(values.shape[-1], degree)[active]
        jac[..., active, mode] = derivative
    return jac


def evaluate_polynomial(
    x: np.ndarray,
    coefficients: np.ndarray,
    degree: int,
    normalized: bool = True,
) -> np.ndarray:
    """Evaluate a homogeneous polynomial stored in symmetric-monomial form."""
    coeff = np.asarray(coefficients, dtype=float)
    return features(x, degree, normalized=normalized) @ coeff


def evaluate_with_spatial_gradient(
    orbital_values: np.ndarray,
    orbital_gradients: np.ndarray,
    coefficients: np.ndarray,
    degree: int,
    normalized: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a field and its Cartesian gradient by the chain rule.

    ``orbital_values`` has shape ``(G, N)`` and ``orbital_gradients`` has
    shape ``(G, D, N)``.  The returned gradient has shape ``(G, D)``.
    """
    phi = np.asarray(orbital_values, dtype=float)
    dphi = np.asarray(orbital_gradients, dtype=float)
    coeff = np.asarray(coefficients, dtype=float)
    if phi.ndim != 2 or dphi.ndim != 3:
        raise ValueError("expected orbital values (G,N) and gradients (G,D,N)")
    if dphi.shape[0] != phi.shape[0] or dphi.shape[2] != phi.shape[1]:
        raise ValueError("orbital value and gradient shapes are incompatible")

    basis = features(phi, degree, normalized=normalized)
    jac = feature_jacobian(phi, degree, normalized=normalized)
    value = basis @ coeff
    gradient = np.einsum("gkn,gdn,k->gd", jac, dphi, coeff, optimize=True)
    return value, gradient


def coefficients_from_tensor_contraction(
    tensor: np.ndarray,
    prefactor: float = 1.0,
    normalized: bool = True,
) -> np.ndarray:
    """Project a tensor contraction onto its fully symmetric polynomial part.

    For a rank-d tensor ``T``, this returns coefficients ``c`` satisfying

        prefactor * T[p,...] x[p] ... x[...] == features(x, d) @ c

    for every real vector x.  No symmetry of ``tensor`` is assumed.
    """
    array = np.asarray(tensor)
    degree = array.ndim
    if degree < 1 or any(size != array.shape[0] for size in array.shape):
        raise ValueError("tensor must have equal dimensions on every axis")

    coefficients = []
    for indices in combinations_with_replacement(range(array.shape[0]), degree):
        unique_permutations = set(permutations(indices))
        total = sum(array[index] for index in unique_permutations)
        coefficients.append(prefactor * total)
    result = np.real_if_close(np.asarray(coefficients))
    if np.iscomplexobj(result):
        raise ValueError("complex contractions require a Hermitian feature representation")
    result = np.asarray(result, dtype=float)
    if normalized:
        result = result / monomial_normalization(array.shape[0], degree)
    return result


def to_plain_monomial_coefficients(
    coefficients: np.ndarray,
    n_modes: int,
    degree: int,
) -> np.ndarray:
    """Convert normalized symmetric-tensor coefficients to plain monomials."""
    return np.asarray(coefficients, dtype=float) * monomial_normalization(n_modes, degree)
