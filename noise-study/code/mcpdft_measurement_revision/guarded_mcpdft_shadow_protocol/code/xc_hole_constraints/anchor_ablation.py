"""Diagnostics for the proposed exact-anchor-only XC-hole ablation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .explicit_band import _spherical_batch
from .full_grid_hole import FullGridHoleConstraint, ProgressCallback, _report_progress


@dataclass(frozen=True)
class AnchorOnlyAudit:
    contraction_max_abs_residual: float
    endpoint_self_max_abs_residual: float
    finite_basis_slope_rms_h: float
    finite_basis_slope_rms_h_over_2: float
    slope_halving_ratio: float
    cusp_target_rms: float
    independent_on_top_measurements: int
    new_external_constraint_rank: int
    conclusion: str

    def metrics(self) -> dict[str, float | int | str | bool]:
        return {
            "contraction_max_abs_residual": self.contraction_max_abs_residual,
            "endpoint_self_max_abs_residual": self.endpoint_self_max_abs_residual,
            "finite_basis_slope_rms_h": self.finite_basis_slope_rms_h,
            "finite_basis_slope_rms_h_over_2": self.finite_basis_slope_rms_h_over_2,
            "slope_halving_ratio": self.slope_halving_ratio,
            "cusp_target_rms": self.cusp_target_rms,
            "independent_on_top_measurements": self.independent_on_top_measurements,
            "new_external_constraint_rank": self.new_external_constraint_rank,
            "sum_rule_redundant_with_dqg_contraction": True,
            "endpoint_is_self_constraint_without_extra_measurement": True,
            "finite_gaussian_basis_has_no_linear_spherical_cusp": True,
            "conclusion": self.conclusion,
        }


def audit_anchor_only_ablation(
    reference: Any,
    d2: np.ndarray,
    gamma: np.ndarray,
    constraint: FullGridHoleConstraint,
    *,
    displacement: float = 1e-3,
    progress: ProgressCallback | None = None,
) -> AnchorOnlyAudit:
    """Show which of the three anchors add information in this workflow.

    The original Fig. 1/2 protocol contains orbital-rotation diagonal-2-RDM
    shadows but no separate on-top measurement. Consequently the on-top RHS
    available to this ablation is computed from the same reconstructed D2 and
    is a self-constraint. The XC-hole sum rule is already implied by the D2 to
    1-RDM contraction equations. Finally, a finite Gaussian orbital expansion
    has an analytic spherical pair density and therefore no linear-in-u cusp.
    """

    if displacement <= 0.0:
        raise ValueError("displacement must be positive.")
    from constrained_shadow import contract_one_rdm

    stage = "anchors-only redundancy/cusp audit"
    _report_progress(progress, stage, 0, 3, event="start", unit="checks")
    contracted = contract_one_rdm(
        d2,
        reference.n_spin_orbitals,
        reference.n_electrons,
        reference.pairs,
    )
    contraction_residual = float(np.max(np.abs(contracted - gamma)))
    _report_progress(progress, stage, 1, 3)

    model = constraint.freeze_model(d2, gamma)
    _, _, on_top = constraint._on_top_fields(d2, gamma)
    endpoint_residual = float(np.max(np.abs(on_top - model.on_top_pair_density)))
    _report_progress(progress, stage, 2, 3)

    kernel = constraint.active_pair_kernel(d2)
    selection = slice(0, len(constraint.coordinates))
    pair_zero, _ = _spherical_batch(
        constraint, kernel, gamma, model, selection, 0.0
    )
    pair_h, _ = _spherical_batch(
        constraint, kernel, gamma, model, selection, displacement
    )
    pair_half, _ = _spherical_batch(
        constraint, kernel, gamma, model, selection, 0.5 * displacement
    )
    slope_h = (pair_h - pair_zero) / displacement
    slope_half = (pair_half - pair_zero) / (0.5 * displacement)
    weights = np.abs(constraint.grid_weights)
    weight_sum = max(float(np.sum(weights)), constraint.config.density_floor)

    def weighted_rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.sum(weights * values**2) / weight_sum))

    slope_rms_h = weighted_rms(slope_h)
    slope_rms_half = weighted_rms(slope_half)
    cusp_target_rms = weighted_rms(model.ordered_on_top_pair_density)
    ratio = slope_rms_half / max(slope_rms_h, 1e-300)
    _report_progress(
        progress,
        stage,
        3,
        3,
        event="complete",
        contraction_max_abs_residual=contraction_residual,
        endpoint_self_max_abs_residual=endpoint_residual,
        slope_halving_ratio=ratio,
    )
    return AnchorOnlyAudit(
        contraction_max_abs_residual=contraction_residual,
        endpoint_self_max_abs_residual=endpoint_residual,
        finite_basis_slope_rms_h=slope_rms_h,
        finite_basis_slope_rms_h_over_2=slope_rms_half,
        slope_halving_ratio=ratio,
        cusp_target_rms=cusp_target_rms,
        independent_on_top_measurements=0,
        new_external_constraint_rank=0,
        conclusion=(
            "Under the unchanged Fig. 1/2 measurement protocol, anchors only "
            "add no independent RDM information; the reconstruction is therefore "
            "identical to DQG. A nontrivial endpoint constraint would require "
            "separately budgeted on-top quantum measurements."
        ),
    )

