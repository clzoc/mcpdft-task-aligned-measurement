"""Cancellation-preserving protection variants (exploratory).

The production soft protection penalizes whitened residuals of the raw density
and on-top maps.  On systems whose DQG anchor error cancels strongly in the
energy (C2), finite-shot noise entering those maps can rotate the error into a
non-cancelling decomposition and worsen the energies even when the physical
norms improve.

Two interventions are defined here; both keep the same DQG + nuclear-norm
problem and only change the affine protection target:

``response``
    Replace the raw physical maps by the frozen ftPBE local field-response
    modes (``FtPBEEnergyObjective.field_response_modes``): protect only the
    directions the functional actually reads (rho, grad rho, Pi, grad Pi).

``shrink``
    Keep the production ``density``/``density_on_top`` grouping but soft
    threshold the whitened measurement deviation at one standard error
    (``sign(v) * max(|v| - 1, 0)``).  Noise-dominated directions are left at
    the anchor, i.e. the anchor's cancellation is not perturbed; only
    statistically significant corrections are applied.

``response_shrink`` combines both.

Only the affine objective changes; ``original`` uses
``run_n2_random_pilot_joint_design._solve_eq11``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import eigh

import common as pc
from experiment import _blind_shadows, j

KAPPA = 0.01
SHRINKAGE_SIGMAS = 1.0
RESPONSE_MODES = 24


def response_map(c: Any, baseline: Any) -> np.ndarray:
    """Frozen ftPBE field-response metric in the lift-space coordinates."""

    modes = c.objective.field_response_modes(
        baseline.d2,
        baseline.gamma,
        max_modes=RESPONSE_MODES,
        capture_fraction=0.999,
    )
    return np.asarray(modes) @ c.lift


def physical_groups(c: Any, covariance: np.ndarray) -> list[np.ndarray]:
    """Production grouping: density space plus the covariance-orthogonalized
    on-top complement (identical to ``experiment.protected_solve``)."""

    g, r = c.blind.density, c.blind.contact
    reg = np.linalg.solve(g @ covariance @ g.T, g @ covariance @ r.T).T
    return [g, r - reg @ g]


def solve_variant(c: Any, baseline: Any, shadows: Any, theta0: np.ndarray,
                  covariance: np.ndarray, zhat: np.ndarray, variant: str) -> Any:
    if variant == "original":
        return j._solve_eq11(c.args, c.sel, shadows, baseline.d2,
                             baseline.gamma, baseline.d2)
    if variant == "protected":
        groups = physical_groups(c, covariance)
        shrink = False
    elif variant == "response":
        groups = [response_map(c, baseline)]
        shrink = False
    elif variant == "shrink":
        groups = physical_groups(c, covariance)
        shrink = True
    elif variant == "response_shrink":
        groups = [response_map(c, baseline)]
        shrink = True
    else:
        raise ValueError(variant)

    factors, deviations = [], []
    for w in groups:
        eig, u = eigh(w @ covariance @ w.T)
        assert eig[0] > 0.0
        whitened = (u / np.sqrt(eig)).T @ w
        value = whitened @ zhat
        if shrink:
            value = np.sign(value) * np.maximum(
                np.abs(value) - SHRINKAGE_SIGMAS, 0.0
            )
        scale = np.sqrt(KAPPA / len(w))
        factors.append(scale * whitened)
        deviations.append(scale * value)
    factor = np.vstack(factors)
    deviation = np.concatenate(deviations)
    l = factor @ c.geo["Z"].T * c.geo["scale"][None, :]
    target = l @ theta0 + deviation
    affine = pc.affine_map(l, c.geo, len(c.sel.pairs), -target)
    return j.solve_dqg_sdp(
        j.LeakGuardReference(c.sel),
        shadow_data=_blind_shadows(shadows),
        affine_objective=affine,
        selection_objective="affine_least_squares",
        additional_d2_objective=c.sel.two_body,
        additional_gamma_objective=c.sel.one_body,
        shadow_error_weight=c.args.nuclear_weight,
        solver=c.args.solver,
        tolerance=c.args.solver_tolerance,
        max_iterations=c.args.max_iterations,
        solver_threads=1,
        positivity_conditions="DQG",
        symmetry_blocked_psd=True,
        initial_d2=baseline.d2,
        initial_gamma=baseline.gamma,
        initial_corrected_d2=baseline.d2,
    )
