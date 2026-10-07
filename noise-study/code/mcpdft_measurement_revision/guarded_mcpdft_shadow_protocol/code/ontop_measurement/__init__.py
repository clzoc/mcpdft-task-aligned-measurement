"""Targeted measurement utilities for MC-PDFT grid ingredients."""

from .frames import (
    complete_frame,
    frame_feature_matrix,
    frames_from_direction_blocks,
    greedy_d_optimal_frames,
    greedy_guarded_target_frames,
    greedy_joint_d_optimal_frames,
    haar_frame,
    identifiability_metrics,
    minimum_frame_count,
)
from .estimation import (
    FrameSummary,
    fit_joint_adaptive_constrained_gls,
    fit_joint_constrained_gls,
    fit_joint_gls,
    normalize_directions,
    occupation_probabilities,
    physicality_violations,
    spin_balanced_probabilities,
    summarize_bitstrings,
)
from .polynomial import (
    coefficients_from_tensor_contraction,
    evaluate_polynomial,
    evaluate_with_spatial_gradient,
    exponent_table,
    feature_jacobian,
    features,
    monomial_normalization,
    to_plain_monomial_coefficients,
)
from .resources import MeasurementResources, resource_estimate
from .mcpdft import (
    assemble_frozen_core_fields,
    density_matrix_from_quadratic_coefficients,
    evaluate_translated_ontop_energy,
    non_ontop_energy,
    pyscf_gga_arrays,
    reconstruct_active_fields,
)
from .sampling import FCIFrameSampler
from .randomized import (
    randomized_frame_inverse,
    sphere_feature_gram,
    sphere_monomial_moment,
)
from .circuits import fci_statevector, orbital_rotation_circuit, statevector_frame_moments

__all__ = [
    "MeasurementResources",
    "FrameSummary",
    "FCIFrameSampler",
    "assemble_frozen_core_fields",
    "coefficients_from_tensor_contraction",
    "complete_frame",
    "density_matrix_from_quadratic_coefficients",
    "evaluate_polynomial",
    "evaluate_with_spatial_gradient",
    "evaluate_translated_ontop_energy",
    "exponent_table",
    "feature_jacobian",
    "features",
    "fci_statevector",
    "fit_joint_constrained_gls",
    "fit_joint_adaptive_constrained_gls",
    "fit_joint_gls",
    "frame_feature_matrix",
    "frames_from_direction_blocks",
    "greedy_d_optimal_frames",
    "greedy_guarded_target_frames",
    "greedy_joint_d_optimal_frames",
    "haar_frame",
    "identifiability_metrics",
    "minimum_frame_count",
    "monomial_normalization",
    "normalize_directions",
    "non_ontop_energy",
    "occupation_probabilities",
    "orbital_rotation_circuit",
    "physicality_violations",
    "pyscf_gga_arrays",
    "reconstruct_active_fields",
    "randomized_frame_inverse",
    "resource_estimate",
    "spin_balanced_probabilities",
    "sphere_feature_gram",
    "sphere_monomial_moment",
    "statevector_frame_moments",
    "summarize_bitstrings",
    "to_plain_monomial_coefficients",
]
