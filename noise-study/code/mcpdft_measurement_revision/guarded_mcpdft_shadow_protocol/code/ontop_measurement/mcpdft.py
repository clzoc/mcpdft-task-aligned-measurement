"""MC-PDFT field reconstruction from contracted polynomial coefficients."""

from __future__ import annotations

import numpy as np

from .polynomial import evaluate_with_spatial_gradient, to_plain_monomial_coefficients


def reconstruct_active_fields(
    orbital_values: np.ndarray,
    orbital_gradients: np.ndarray,
    quartic_coefficients: np.ndarray,
    quadratic_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return active rho, grad(rho), Pi, and grad(Pi) from one coefficient pair."""
    rho, grad_rho = evaluate_with_spatial_gradient(
        orbital_values, orbital_gradients, quadratic_coefficients, 2
    )
    pair, grad_pair = evaluate_with_spatial_gradient(
        orbital_values, orbital_gradients, quartic_coefficients, 4
    )
    return rho, grad_rho, pair, grad_pair


def assemble_frozen_core_fields(
    rho_active: np.ndarray,
    grad_rho_active: np.ndarray,
    pair_active: np.ndarray,
    grad_pair_active: np.ndarray,
    rho_core: np.ndarray,
    grad_rho_core: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble total density and on-top pair density, including gradients.

    For a doubly occupied frozen core,

        Pi = Pi_a + rho_c rho_a / 2 + rho_c^2 / 4.

    The gradient is the exact product-rule derivative of this expression.
    """
    rho_a = np.asarray(rho_active, dtype=float)
    grad_rho_a = np.asarray(grad_rho_active, dtype=float)
    pair_a = np.asarray(pair_active, dtype=float)
    grad_pair_a = np.asarray(grad_pair_active, dtype=float)
    rho_c = np.asarray(rho_core, dtype=float)
    grad_rho_c = np.asarray(grad_rho_core, dtype=float)
    if grad_rho_a.shape != grad_pair_a.shape or grad_rho_a.shape != grad_rho_c.shape:
        raise ValueError("all spatial gradients must have the same shape")
    if grad_rho_a.shape[0] != len(rho_a) or any(
        len(value) != len(rho_a) for value in (pair_a, rho_c)
    ):
        raise ValueError("field values and spatial gradients are incompatible")

    rho = rho_c + rho_a
    grad_rho = grad_rho_c + grad_rho_a
    pair = pair_a + 0.5 * rho_c * rho_a + 0.25 * rho_c**2
    grad_pair = (
        grad_pair_a
        + 0.5
        * (
            rho_a[:, None] * grad_rho_c
            + rho_c[:, None] * grad_rho_a
        )
        + 0.5 * rho_c[:, None] * grad_rho_c
    )
    return rho, grad_rho, pair, grad_pair


def density_matrix_from_quadratic_coefficients(
    coefficients: np.ndarray, n_modes: int
) -> np.ndarray:
    """Recover the real symmetric spin-summed 1-RDM from normalized coefficients."""
    plain = to_plain_monomial_coefficients(coefficients, n_modes, 2)
    density = np.zeros((n_modes, n_modes), dtype=float)
    index = 0
    for p in range(n_modes):
        for q in range(p, n_modes):
            if p == q:
                density[p, p] = plain[index]
            else:
                density[p, q] = density[q, p] = 0.5 * plain[index]
            index += 1
    return density


def pyscf_gga_arrays(
    rho: np.ndarray,
    grad_rho: np.ndarray,
    pair: np.ndarray,
    grad_pair: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert value/gradient arrays to PySCF's (4,G) convention."""
    rho4 = np.vstack([np.asarray(rho)[None, :], np.asarray(grad_rho).T])
    pair4 = np.vstack([np.asarray(pair)[None, :], np.asarray(grad_pair).T])
    return rho4, pair4


def evaluate_translated_ontop_energy(
    functional,
    rho: np.ndarray,
    grad_rho: np.ndarray,
    pair: np.ndarray,
    grad_pair: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Evaluate a PySCF translated or fully translated on-top functional."""
    rho4, pair4 = pyscf_gga_arrays(rho, grad_rho, pair, grad_pair)
    rho_rows = 1 if functional.dens_deriv == 0 else 4
    pair_rows = 1 if functional.Pi_deriv == 0 else 4
    spin_density = np.stack([0.5 * rho4[:, :], 0.5 * rho4[:, :]])
    energy_density, _, _ = functional.eval_ot(
        spin_density[:, :rho_rows],
        pair4[:pair_rows],
        dderiv=0,
        weights=np.asarray(weights),
    )
    return float(np.asarray(weights) @ energy_density)


def non_ontop_energy(
    nuclear_repulsion: float,
    hcore: np.ndarray,
    density_ao: np.ndarray,
    coulomb_ao: np.ndarray,
) -> float:
    """Return V_nn + Tr(hD) + Tr(D J[D])/2, the non-on-top PDFT energy."""
    return float(
        nuclear_repulsion
        + np.einsum("uv,uv->", hcore, density_ao)
        + 0.5 * np.einsum("uv,uv->", density_ao, coulomb_ao)
    )

