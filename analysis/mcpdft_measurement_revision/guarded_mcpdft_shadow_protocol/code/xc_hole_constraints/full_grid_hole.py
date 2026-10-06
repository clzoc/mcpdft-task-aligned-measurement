"""Full-grid short-range pair-density constraints for constrained shadows.

The implementation keeps every point in the configured PySCF molecular grid
and every point in the radial/angular displacement product grid. It does not
select points by density, gradients, or residual size.

PySCF exposes translated and fully translated on-top functionals but not their
underlying real-space PBE hole. The model used here is therefore deliberately
named an anchor-based local-hole completion: the frozen 1-RDM supplies its
nonlocal Fermi hole exactly, tPBE and ftPBE supply two local decay scales, and a
cumulant completion enforces the on-top endpoint, the exact finite-basis hole
sum rule, and the opposite-spin Coulomb cusp. A second zero-slope branch
represents the analytic behavior of the finite Gaussian orbital basis. The
envelope of these four curves is the model band; tBLYP is not used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from typing import Any, Callable, Sequence

import numpy as np
from scipy import sparse


ProgressCallback = Callable[[str, int | None, int | None, dict[str, Any]], None]


def _report_progress(
    progress: ProgressCallback | None,
    stage: str,
    completed: int | None,
    total: int | None,
    *,
    event: str = "progress",
    **detail: Any,
) -> None:
    if progress is not None:
        progress(stage, completed, total, {"event": event, **detail})


@dataclass(frozen=True)
class HoleGridConfig:
    """Deterministic full tensor-product grid used by the hole audit."""

    grid_level: int = 1
    u_min_bohr: float = 0.05
    u_max_bohr: float = 0.50
    u_points: int = 10
    angular_points: int = 14
    batch_size: int = 2048
    density_floor: float = 1e-14

    def radial_grid(self) -> np.ndarray:
        if self.u_points < 1:
            raise ValueError("u_points must be positive.")
        if self.u_min_bohr <= 0.0 or self.u_max_bohr < self.u_min_bohr:
            raise ValueError("Expected 0 < u_min_bohr <= u_max_bohr.")
        return np.linspace(self.u_min_bohr, self.u_max_bohr, self.u_points)


@dataclass(frozen=True)
class FrozenHoleModel:
    """Right-hand side frozen at one outer reconstruction iterate."""

    anchor_d2: np.ndarray
    anchor_gamma: np.ndarray
    rho: np.ndarray
    on_top_pair_density: np.ndarray
    ordered_on_top_pair_density: np.ndarray
    ratio: np.ndarray
    zeta_tpbe: np.ndarray
    zeta_ftpbe: np.ndarray
    cumulant_endpoint: np.ndarray
    cumulant_sum_target: np.ndarray
    decay_tpbe: np.ndarray
    decay_ftpbe: np.ndarray
    ratio_clipped_count: int


@dataclass(frozen=True)
class HoleAudit:
    """Full-grid band residual and, optionally, its linear RDM gradients."""

    loss: float
    weighted_violation_rms: float
    weighted_model_half_width_rms: float
    violation_fraction: float
    maximum_absolute_violation: float
    evaluated_pair_points: int
    evaluated_constraint_points: int
    central_grid_points: int
    radial_points: int
    angular_points: int
    density_scale: float
    endpoint_max_error: float
    cusp_max_error: float
    correlation_sum_rule_max_error: float
    d2_gradient: np.ndarray | None = None
    gamma_gradient: np.ndarray | None = None

    def metrics(self) -> dict[str, float | int]:
        return {
            "hole_loss": self.loss,
            "hole_weighted_violation_rms": self.weighted_violation_rms,
            "hole_weighted_model_half_width_rms": self.weighted_model_half_width_rms,
            "hole_violation_fraction": self.violation_fraction,
            "hole_maximum_absolute_violation": self.maximum_absolute_violation,
            "hole_pair_points": self.evaluated_pair_points,
            "hole_constraint_points": self.evaluated_constraint_points,
            "hole_central_grid_points": self.central_grid_points,
            "hole_radial_points": self.radial_points,
            "hole_angular_points": self.angular_points,
            "hole_density_scale": self.density_scale,
            "hole_endpoint_max_error": self.endpoint_max_error,
            "hole_cusp_max_error": self.cusp_max_error,
            "hole_correlation_sum_rule_max_error": self.correlation_sum_rule_max_error,
        }


@dataclass(frozen=True)
class HoleCorrectionStep:
    iteration: int
    accepted: bool
    before: HoleAudit
    candidate: HoleAudit
    gradient_scale: float
    d2_step_norm: float
    gamma_step_norm: float
    energy_before: float
    energy_candidate: float


@dataclass(frozen=True)
class HoleCorrectionResult:
    d2: np.ndarray
    gamma: np.ndarray
    energy: float
    status: str
    steps: tuple[HoleCorrectionStep, ...]
    final_audit: HoleAudit
    exact_reference_audit: HoleAudit | None


def _pair_lookup(
    left: int,
    right: int,
    pair_to_index: dict[tuple[int, int], int],
) -> tuple[int | None, int]:
    if left == right:
        return None, 0
    if left < right:
        return pair_to_index[(left, right)], 1
    return pair_to_index[(right, left)], -1


def _symmetric_pair_data(n_orbitals: int) -> tuple[tuple[tuple[int, int], ...], np.ndarray]:
    pairs = tuple(combinations_with_replacement(range(n_orbitals), 2))
    normalization = np.asarray(
        [1.0 if first == second else math.sqrt(2.0) for first, second in pairs]
    )
    return pairs, normalization


def _features2(values: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    output = np.empty((len(array), len(pairs)), dtype=float)
    root_two = math.sqrt(2.0)
    for index, (first, second) in enumerate(pairs):
        output[:, index] = array[:, first] * array[:, second]
        if first != second:
            output[:, index] *= root_two
    return output


def _spatial_pair_map(reference: Any) -> sparse.csr_matrix:
    """Map pair-basis spin-orbital D to a symmetric spatial pair kernel."""

    n_spatial = int(reference.n_spatial_orbitals)
    spatial_pairs, normalization = _symmetric_pair_data(n_spatial)
    spatial_lookup = {pair: index for index, pair in enumerate(spatial_pairs)}
    pair_lookup = {pair: index for index, pair in enumerate(reference.pairs)}
    n_pairs = len(reference.pairs)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for p in range(n_spatial):
        for q in range(n_spatial):
            first_pair = (p, q) if p <= q else (q, p)
            first_index = spatial_lookup[first_pair]
            for r in range(n_spatial):
                for s in range(n_spatial):
                    second_pair = (r, s) if r <= s else (s, r)
                    second_index = spatial_lookup[second_pair]
                    output = first_index * len(spatial_pairs) + second_index
                    coefficient_scale = 1.0 / (
                        normalization[first_index] * normalization[second_index]
                    )
                    for first_spin in (0, 1):
                        for second_spin in (0, 1):
                            creation_first = p + first_spin * n_spatial
                            creation_second = r + second_spin * n_spatial
                            annihilation_first = q + first_spin * n_spatial
                            annihilation_second = s + second_spin * n_spatial
                            row, row_sign = _pair_lookup(
                                creation_first, creation_second, pair_lookup
                            )
                            col, col_sign = _pair_lookup(
                                annihilation_first, annihilation_second, pair_lookup
                            )
                            if row is None or col is None:
                                continue
                            rows.append(output)
                            cols.append(row * n_pairs + col)
                            data.append(coefficient_scale * row_sign * col_sign)
    dimension = len(spatial_pairs)
    return sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(dimension * dimension, n_pairs * n_pairs),
    ).tocsr()


def _translation_zeta(functional: Any, ratio: np.ndarray) -> np.ndarray:
    return np.asarray(functional.get_zeta(ratio[None, :], fn_deriv=0)[0], dtype=float)


class FullGridHoleConstraint:
    """Evaluate and differentiate the full-grid local-hole band."""

    model_names = ("tPBE", "ftPBE")

    def __init__(
        self,
        reference: Any,
        config: HoleGridConfig,
        *,
        progress: ProgressCallback | None = None,
    ):
        from pyscf import dft
        from pyscf.dft import gen_grid
        from pyscf.mcpdft.otfnal import get_transfnal

        if reference.molecule is None or reference.active_mo_coeff is None:
            raise ValueError("The molecular reference must retain its PySCF orbitals.")
        self.reference = reference
        self.config = config
        self.molecule = reference.molecule
        self.n_spatial = int(reference.n_spatial_orbitals)
        self.n_modes = int(reference.n_spin_orbitals)
        self.n_pair_basis = len(reference.pairs)
        self.spatial_pairs, _ = _symmetric_pair_data(self.n_spatial)
        _report_progress(
            progress,
            "build pair-density map",
            None,
            None,
            event="start",
        )
        self.pair_map = _spatial_pair_map(reference)
        _report_progress(
            progress,
            "build pair-density map",
            None,
            None,
            event="complete",
            nonzero_elements=int(self.pair_map.nnz),
        )

        _report_progress(
            progress,
            "build molecular quadrature grid",
            None,
            None,
            event="start",
            grid_level=int(config.grid_level),
        )
        grids = dft.gen_grid.Grids(self.molecule)
        grids.level = int(config.grid_level)
        grids.build()
        self.coordinates = np.asarray(grids.coords, dtype=float)
        self.grid_weights = np.asarray(grids.weights, dtype=float)
        _report_progress(
            progress,
            "build molecular quadrature grid",
            len(self.coordinates),
            len(self.coordinates),
            event="complete",
            unit="points",
            grid_level=int(config.grid_level),
        )

        _report_progress(
            progress,
            "evaluate central-grid orbitals",
            0,
            len(self.coordinates),
            event="start",
            unit="points",
        )
        ao = self.molecule.eval_gto("GTOval_sph", self.coordinates)
        self.active_coeff = np.asarray(reference.active_mo_coeff, dtype=float)
        self.active_values = np.asarray(ao @ self.active_coeff, dtype=float)
        mean_field_coeff = np.asarray(reference.mean_field.mo_coeff, dtype=float)
        n_core = int(reference.n_core_orbitals)
        self.core_coeff = mean_field_coeff[:, :n_core]
        self.core_values = np.asarray(ao @ self.core_coeff, dtype=float)
        self.active_features = _features2(self.active_values, self.spatial_pairs)
        self.core_density = 2.0 * np.einsum(
            "gi,gi->g", self.core_values, self.core_values, optimize=True
        )
        _report_progress(
            progress,
            "evaluate central-grid orbitals",
            len(self.coordinates),
            len(self.coordinates),
            event="complete",
            unit="points",
        )

        _report_progress(
            progress,
            "initialize tPBE/ftPBE hole models",
            None,
            None,
            event="start",
        )
        angular = np.asarray(gen_grid.MakeAngularGrid(config.angular_points), dtype=float)
        self.directions = angular[:, :3]
        self.angular_weights = angular[:, 3]
        self.radial_values = config.radial_grid()
        self.tpbe = get_transfnal(self.molecule, "tPBE")
        self.ftpbe = get_transfnal(self.molecule, "ftPBE")
        _report_progress(
            progress,
            "initialize tPBE/ftPBE hole models",
            None,
            None,
            event="complete",
            central_grid_points=len(self.coordinates),
            radial_points=len(self.radial_values),
            angular_points=len(self.directions),
            full_pair_points=self.pair_point_count,
        )

    @property
    def pair_point_count(self) -> int:
        return (
            len(self.coordinates) * len(self.radial_values) * len(self.directions)
        )

    def active_pair_kernel(self, d2: np.ndarray) -> np.ndarray:
        vector = self.pair_map @ np.asarray(d2, dtype=float).ravel(order="C")
        dimension = len(self.spatial_pairs)
        return np.asarray(vector).reshape((dimension, dimension), order="C")

    def pair_map_adjoint(self, kernel_gradient: np.ndarray) -> np.ndarray:
        vector = self.pair_map.T @ np.asarray(kernel_gradient).ravel(order="C")
        return np.asarray(vector).reshape(
            (self.n_pair_basis, self.n_pair_basis), order="C"
        )

    def _active_density(self, values: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        result = np.zeros(len(values), dtype=float)
        for spin in (0, 1):
            block = gamma[
                spin * self.n_spatial : (spin + 1) * self.n_spatial,
                spin * self.n_spatial : (spin + 1) * self.n_spatial,
            ]
            result += np.einsum("gi,ij,gj->g", values, block, values, optimize=True)
        return result

    def _active_transition_density(
        self, left: np.ndarray, right: np.ndarray, gamma: np.ndarray
    ) -> np.ndarray:
        result = np.zeros(len(left), dtype=float)
        for spin in (0, 1):
            block = gamma[
                spin * self.n_spatial : (spin + 1) * self.n_spatial,
                spin * self.n_spatial : (spin + 1) * self.n_spatial,
            ]
            result += np.einsum("gi,ij,gj->g", left, block, right, optimize=True)
        return result

    def _spin_active_density(
        self, values: np.ndarray, gamma: np.ndarray, spin: int
    ) -> np.ndarray:
        block = gamma[
            spin * self.n_spatial : (spin + 1) * self.n_spatial,
            spin * self.n_spatial : (spin + 1) * self.n_spatial,
        ]
        return np.einsum("gi,ij,gj->g", values, block, values, optimize=True)

    def _spin_active_transition_density(
        self,
        left: np.ndarray,
        right: np.ndarray,
        gamma: np.ndarray,
        spin: int,
    ) -> np.ndarray:
        block = gamma[
            spin * self.n_spatial : (spin + 1) * self.n_spatial,
            spin * self.n_spatial : (spin + 1) * self.n_spatial,
        ]
        return np.einsum("gi,ij,gj->g", left, block, right, optimize=True)

    def _on_top_fields(
        self, d2: np.ndarray, gamma: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kernel = self.active_pair_kernel(d2)
        active_pair = np.einsum(
            "gi,ij,gj->g",
            self.active_features,
            kernel,
            self.active_features,
            optimize=True,
        )
        active_density = self._active_density(self.active_values, gamma)
        active_spin_transition = active_density
        core_spin_transition = 0.5 * self.core_density
        core_pair = self.core_density**2 - 2.0 * core_spin_transition**2
        cross_pair = 2.0 * self.core_density * active_density
        cross_pair -= 2.0 * core_spin_transition * active_spin_transition
        ordered_pair = active_pair + cross_pair + core_pair
        total_density = active_density + self.core_density
        return total_density, ordered_pair, 0.5 * ordered_pair

    def freeze_model(self, d2: np.ndarray, gamma: np.ndarray) -> FrozenHoleModel:
        rho, ordered_pair, pair = self._on_top_fields(d2, gamma)
        rho_squared = rho**2
        active = rho_squared > self.config.density_floor
        ratio = np.ones_like(rho)
        ratio[active] = 2.0 * ordered_pair[active] / rho_squared[active]
        clipped = np.maximum(ratio, 0.0)
        zeta_t = _translation_zeta(self.tpbe, clipped)
        zeta_ft = _translation_zeta(self.ftpbe, clipped)
        core_spin_density = 0.5 * self.core_density
        spin_densities = []
        spin_gamma_squared = []
        for spin in (0, 1):
            block = gamma[
                spin * self.n_spatial : (spin + 1) * self.n_spatial,
                spin * self.n_spatial : (spin + 1) * self.n_spatial,
            ]
            active_spin = np.einsum(
                "gi,ij,gj->g",
                self.active_values,
                block,
                self.active_values,
                optimize=True,
            )
            active_squared = np.einsum(
                "gi,ij,gj->g",
                self.active_values,
                block @ block,
                self.active_values,
                optimize=True,
            )
            spin_densities.append(core_spin_density + active_spin)
            spin_gamma_squared.append(core_spin_density + active_squared)
        gaussian_on_top = rho**2 - sum(value**2 for value in spin_densities)
        cumulant_endpoint = ordered_pair - gaussian_on_top
        cumulant_sum_target = -rho + sum(spin_gamma_squared)

        def decay(zeta: np.ndarray) -> np.ndarray:
            up = 0.5 * rho * (1.0 + zeta)
            down = 0.5 * rho * (1.0 - zeta)
            k_up = np.cbrt(6.0 * np.pi**2 * np.maximum(up, 0.0))
            k_down = np.cbrt(6.0 * np.pi**2 * np.maximum(down, 0.0))
            return np.maximum(0.5 * (k_up + k_down), 0.25)

        return FrozenHoleModel(
            anchor_d2=np.asarray(d2).copy(),
            anchor_gamma=np.asarray(gamma).copy(),
            rho=rho,
            on_top_pair_density=pair,
            ordered_on_top_pair_density=ordered_pair,
            ratio=ratio,
            zeta_tpbe=zeta_t,
            zeta_ftpbe=zeta_ft,
            cumulant_endpoint=cumulant_endpoint,
            cumulant_sum_target=cumulant_sum_target,
            decay_tpbe=decay(zeta_t),
            decay_ftpbe=decay(zeta_ft),
            ratio_clipped_count=int(np.count_nonzero(ratio < 0.0)),
        )

    def _model_cumulants(
        self,
        model: FrozenHoleModel,
        central_indices: slice,
        radius: float,
    ) -> np.ndarray:
        endpoint = model.cumulant_endpoint[central_indices]
        sum_target = model.cumulant_sum_target[central_indices]
        cusp_slope = model.ordered_on_top_pair_density[central_indices]
        variants: list[np.ndarray] = []
        for decay in (
            model.decay_tpbe[central_indices],
            model.decay_ftpbe[central_indices],
        ):
            for target_slope in (cusp_slope, np.zeros_like(cusp_slope)):
                linear = target_slope + decay * endpoint
                quadratic = (
                    sum_target / (4.0 * np.pi)
                    - 2.0 * endpoint / decay**3
                    - 6.0 * linear / decay**4
                ) * decay**5 / 24.0
                correlation = np.exp(-decay * radius) * (
                    endpoint + linear * radius + quadratic * radius**2
                )
                variants.append(correlation)
        return np.asarray(variants)

    def _pair_density_batch(
        self,
        d2_kernel: np.ndarray,
        gamma: np.ndarray,
        central_indices: slice,
        displaced_coordinates: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        ao_displaced = self.molecule.eval_gto("GTOval_sph", displaced_coordinates)
        active_y = np.asarray(ao_displaced @ self.active_coeff, dtype=float)
        core_y = np.asarray(ao_displaced @ self.core_coeff, dtype=float)
        features_y = _features2(active_y, self.spatial_pairs)
        features_x = self.active_features[central_indices]
        active_pair = np.einsum(
            "gi,ij,gj->g", features_x, d2_kernel, features_y, optimize=True
        )

        active_x = self.active_values[central_indices]
        core_x = self.core_values[central_indices]
        rho_core_x = self.core_density[central_indices]
        rho_core_y = 2.0 * np.einsum("gi,gi->g", core_y, core_y, optimize=True)
        rho_active_x = self._active_density(active_x, gamma)
        rho_active_y = self._active_density(active_y, gamma)
        gamma_active_xy = self._active_transition_density(active_x, active_y, gamma)
        gamma_core_xy = np.einsum("gi,gi->g", core_x, core_y, optimize=True)

        core_pair = rho_core_x * rho_core_y - 2.0 * gamma_core_xy**2
        cross_pair = rho_core_x * rho_active_y + rho_active_x * rho_core_y
        cross_pair -= 2.0 * gamma_core_xy * gamma_active_xy
        total = active_pair + cross_pair + core_pair
        aux = {
            "active_x": active_x,
            "active_y": active_y,
            "features_x": features_x,
            "features_y": features_y,
            "core_y": core_y,
            "rho_core_x": rho_core_x,
            "rho_core_y": rho_core_y,
            "gamma_core_xy": gamma_core_xy,
        }
        return total, aux

    def audit(
        self,
        d2: np.ndarray,
        gamma: np.ndarray,
        model: FrozenHoleModel | None = None,
        *,
        gradients: bool = False,
        progress: ProgressCallback | None = None,
        progress_stage: str = "full-grid hole audit",
    ) -> HoleAudit:
        frozen = self.freeze_model(d2, gamma) if model is None else model
        kernel = self.active_pair_kernel(d2)
        central_weight = np.abs(self.grid_weights)
        radial_weight = 1.0 / self.radial_values
        total_weight = central_weight.sum() * radial_weight.sum()
        density_scale = math.sqrt(
            float(
                np.sum(central_weight * frozen.ordered_on_top_pair_density**2)
                / max(central_weight.sum(), self.config.density_floor)
            )
        )
        density_scale = max(density_scale, 1e-10)

        loss_numerator = 0.0
        spread_numerator = 0.0
        maximum_violation = 0.0
        violation_count = 0
        evaluated = 0
        evaluated_constraints = 0
        kernel_gradient = np.zeros_like(kernel) if gradients else None
        gamma_gradient = (
            np.zeros((self.n_modes, self.n_modes), dtype=float) if gradients else None
        )

        batch_size = max(1, int(self.config.batch_size))
        central_batches = math.ceil(len(self.coordinates) / batch_size)
        total_batches = central_batches * len(self.radial_values)
        completed_batches = 0
        _report_progress(
            progress,
            progress_stage,
            0,
            total_batches,
            event="start",
            unit="batches",
            gradients=bool(gradients),
            central_grid_points=len(self.coordinates),
            radial_points=len(self.radial_values),
            angular_points=len(self.directions),
            full_pair_points=self.pair_point_count,
        )
        for start in range(0, len(self.coordinates), batch_size):
            stop = min(start + batch_size, len(self.coordinates))
            selection = slice(start, stop)
            coordinates = self.coordinates[selection]
            rho_x = frozen.rho[selection]
            for radial_index, radius in enumerate(self.radial_values):
                cumulants = self._model_cumulants(
                    frozen, selection, float(radius)
                )
                pair_value = np.zeros(stop - start, dtype=float)
                gaussian_pair = np.zeros(stop - start, dtype=float)
                average_features_y = np.zeros(
                    (stop - start, len(self.spatial_pairs)), dtype=float
                )
                average_active_yy = np.zeros(
                    (stop - start, self.n_spatial, self.n_spatial), dtype=float
                )
                average_rho_core_y = np.zeros(stop - start, dtype=float)
                average_core_transition_y = np.zeros(
                    (stop - start, self.n_spatial), dtype=float
                )
                last_aux: dict[str, np.ndarray] | None = None
                for direction, angular_weight in zip(
                    self.directions, self.angular_weights
                ):
                    displaced = coordinates + radius * direction[None, :]
                    directional_pair, aux = self._pair_density_batch(
                        kernel, gamma, selection, displaced
                    )
                    rho_active_y = self._active_density(
                        aux["active_y"], frozen.anchor_gamma
                    )
                    rho_core_y = aux["rho_core_y"]
                    weight = float(angular_weight)
                    pair_value += weight * directional_pair
                    spin_transition_squared = np.zeros(stop - start, dtype=float)
                    for spin in (0, 1):
                        active_transition = self._spin_active_transition_density(
                            aux["active_x"],
                            aux["active_y"],
                            frozen.anchor_gamma,
                            spin,
                        )
                        spin_transition_squared += (
                            aux["gamma_core_xy"] + active_transition
                        ) ** 2
                    gaussian_pair += weight * (
                        rho_x * (rho_active_y + rho_core_y)
                        - spin_transition_squared
                    )
                    average_features_y += weight * aux["features_y"]
                    average_active_yy += weight * np.einsum(
                        "gi,gj->gij", aux["active_y"], aux["active_y"], optimize=True
                    )
                    average_rho_core_y += weight * rho_core_y
                    average_core_transition_y += (
                        weight
                        * aux["gamma_core_xy"][:, None]
                        * aux["active_y"]
                    )
                    last_aux = aux

                model_values = gaussian_pair[None, :] + cumulants
                lower = np.min(model_values, axis=0)
                upper = np.max(model_values, axis=0)
                center = 0.5 * (lower + upper)
                half_width = 0.5 * (upper - lower)
                difference = pair_value - center
                signed_violation = np.sign(difference) * np.maximum(
                    np.abs(difference) - half_width, 0.0
                )
                point_weight = (
                    central_weight[selection] * radial_weight[radial_index]
                )
                loss_numerator += float(
                    np.sum(point_weight * signed_violation**2)
                )
                spread_numerator += float(np.sum(point_weight * half_width**2))
                maximum_violation = max(
                    maximum_violation,
                    float(np.max(np.abs(signed_violation), initial=0.0)),
                )
                violation_count += int(np.count_nonzero(signed_violation))
                evaluated += len(pair_value) * len(self.directions)
                evaluated_constraints += len(pair_value)

                if gradients:
                    if last_aux is None:
                        raise RuntimeError("The angular quadrature is empty.")
                    derivative = (
                        2.0
                        * point_weight
                        * signed_violation
                        / (total_weight * density_scale**2)
                    )
                    kernel_gradient += last_aux["features_x"].T @ (
                        derivative[:, None] * average_features_y
                    )
                    active_x = last_aux["active_x"]
                    block = np.einsum(
                        "g,g,gij->ij",
                        derivative,
                        last_aux["rho_core_x"],
                        average_active_yy,
                        optimize=True,
                    )
                    block += np.einsum(
                        "g,g,gi,gj->ij",
                        derivative,
                        average_rho_core_y,
                        active_x,
                        active_x,
                        optimize=True,
                    )
                    block -= np.einsum(
                        "g,gi,gj->ij",
                        2.0 * derivative,
                        active_x,
                        average_core_transition_y,
                        optimize=True,
                    )
                    block = 0.5 * (block + block.T)
                    gamma_gradient[: self.n_spatial, : self.n_spatial] += block
                    gamma_gradient[self.n_spatial :, self.n_spatial :] += block

                completed_batches += 1
                if completed_batches < total_batches:
                    _report_progress(
                        progress,
                        progress_stage,
                        completed_batches,
                        total_batches,
                        central_points_completed=stop,
                        radial_index=radial_index + 1,
                        evaluated_pair_points=evaluated,
                        evaluated_constraint_points=evaluated_constraints,
                    )

        normalized_loss = loss_numerator / (total_weight * density_scale**2)
        d2_gradient = None
        if gradients and kernel_gradient is not None:
            d2_gradient = self.pair_map_adjoint(kernel_gradient)
            d2_gradient = 0.5 * (d2_gradient + d2_gradient.T)
            gamma_gradient = 0.5 * (gamma_gradient + gamma_gradient.T)

        endpoint_errors = []
        cusp_errors = []
        sum_rule_errors = []
        for decay in (frozen.decay_tpbe, frozen.decay_ftpbe):
            endpoint = frozen.cumulant_endpoint
            endpoint_errors.append(np.max(np.abs(endpoint - frozen.cumulant_endpoint)))
            cusp_slope = frozen.ordered_on_top_pair_density
            linear = cusp_slope + decay * endpoint
            recovered_slope = linear - decay * endpoint
            cusp_errors.append(np.max(np.abs(recovered_slope - cusp_slope)))
            quadratic = (
                frozen.cumulant_sum_target / (4.0 * np.pi)
                - 2.0 * endpoint / decay**3
                - 6.0 * linear / decay**4
            ) * decay**5 / 24.0
            radial_integral = (
                2.0 * endpoint / decay**3
                + 6.0 * linear / decay**4
                + 24.0 * quadratic / decay**5
            )
            sum_rule_errors.append(
                np.max(
                    np.abs(
                        4.0 * np.pi * radial_integral
                        - frozen.cumulant_sum_target
                    )
                )
            )

        result = HoleAudit(
            loss=float(normalized_loss),
            weighted_violation_rms=float(
                math.sqrt(loss_numerator / max(total_weight, self.config.density_floor))
            ),
            weighted_model_half_width_rms=float(
                math.sqrt(spread_numerator / max(total_weight, self.config.density_floor))
            ),
            violation_fraction=float(
                violation_count / max(evaluated_constraints, 1)
            ),
            maximum_absolute_violation=float(maximum_violation),
            evaluated_pair_points=int(evaluated),
            evaluated_constraint_points=int(evaluated_constraints),
            central_grid_points=len(self.coordinates),
            radial_points=len(self.radial_values),
            angular_points=len(self.directions),
            density_scale=float(density_scale),
            endpoint_max_error=float(max(endpoint_errors, default=0.0)),
            cusp_max_error=float(max(cusp_errors, default=0.0)),
            correlation_sum_rule_max_error=float(max(sum_rule_errors, default=0.0)),
            d2_gradient=d2_gradient,
            gamma_gradient=gamma_gradient,
        )
        _report_progress(
            progress,
            progress_stage,
            total_batches,
            total_batches,
            event="complete",
            hole_loss=result.loss,
            violation_fraction=result.violation_fraction,
            evaluated_pair_points=result.evaluated_pair_points,
            evaluated_constraint_points=result.evaluated_constraint_points,
        )
        return result


def _objective_norm(reference: Any) -> float:
    return float(
        math.sqrt(
            np.linalg.norm(reference.two_body) ** 2
            + np.linalg.norm(reference.one_body) ** 2
        )
    )


def correct_with_full_grid_hole(
    reference: Any,
    shadow_data: Any,
    initial_result: Any,
    constraint: FullGridHoleConstraint,
    solve_sdp: Any,
    *,
    n_shadows: int,
    iterations: int = 2,
    hole_strength: float = 1.0,
    solver: str = "SCS",
    tolerance: float = 1e-4,
    max_iterations: int = 50_000,
    verbose: bool = False,
    progress: ProgressCallback | None = None,
) -> HoleCorrectionResult:
    """Run the audit/re-solve loop while retaining all DQG/shadow constraints."""

    current = initial_result
    steps: list[HoleCorrectionStep] = []
    prior_decay = 1.0 / math.sqrt(max(1, int(n_shadows)))
    for iteration in range(max(0, int(iterations))):
        model = constraint.freeze_model(current.d2, current.gamma)
        before = constraint.audit(
            current.d2,
            current.gamma,
            model,
            gradients=True,
            progress=progress,
            progress_stage=f"hole iteration {iteration + 1}: gradient audit",
        )
        if before.d2_gradient is None or before.gamma_gradient is None:
            raise RuntimeError("The full-grid audit did not return gradients.")
        gradient_norm = math.sqrt(
            np.linalg.norm(before.d2_gradient) ** 2
            + np.linalg.norm(before.gamma_gradient) ** 2
        )
        if gradient_norm <= 1e-14 or before.loss <= 1e-16:
            break
        gradient_scale = (
            float(hole_strength)
            * prior_decay
            * _objective_norm(reference)
            / gradient_norm
        )
        solve_stage = f"hole iteration {iteration + 1}: DQG re-solve"
        _report_progress(
            progress,
            solve_stage,
            None,
            None,
            event="start",
            solver=solver,
            solver_tolerance=float(tolerance),
            max_iterations=int(max_iterations),
        )
        candidate = solve_sdp(
            reference,
            shadow_data=shadow_data,
            solver=solver,
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=verbose,
            positivity_conditions="DQG",
            additional_d2_objective=gradient_scale * before.d2_gradient,
            additional_gamma_objective=gradient_scale * before.gamma_gradient,
        )
        _report_progress(
            progress,
            solve_stage,
            None,
            None,
            event="complete",
            solver_status=str(candidate.status),
        )
        candidate_audit = constraint.audit(
            candidate.d2,
            candidate.gamma,
            model,
            gradients=False,
            progress=progress,
            progress_stage=f"hole iteration {iteration + 1}: candidate audit",
        )
        accepted = candidate_audit.loss < before.loss * (1.0 - 1e-6)
        steps.append(
            HoleCorrectionStep(
                iteration=iteration,
                accepted=accepted,
                before=before,
                candidate=candidate_audit,
                gradient_scale=float(gradient_scale),
                d2_step_norm=float(np.linalg.norm(candidate.d2 - current.d2)),
                gamma_step_norm=float(np.linalg.norm(candidate.gamma - current.gamma)),
                energy_before=float(current.energy),
                energy_candidate=float(candidate.energy),
            )
        )
        if not accepted:
            break
        current = candidate

    final_model = constraint.freeze_model(current.d2, current.gamma)
    final_audit = constraint.audit(
        current.d2,
        current.gamma,
        final_model,
        gradients=False,
        progress=progress,
        progress_stage="hole result: final audit",
    )
    exact_reference_audit = None
    if getattr(reference, "exact_d2", None) is not None:
        exact_reference_audit = constraint.audit(
            reference.exact_d2,
            reference.exact_gamma,
            final_model,
            gradients=False,
            progress=progress,
            progress_stage="hole result: exact-reference posthoc audit",
        )
    return HoleCorrectionResult(
        d2=np.asarray(current.d2),
        gamma=np.asarray(current.gamma),
        energy=float(current.energy),
        status=str(current.status),
        steps=tuple(steps),
        final_audit=final_audit,
        exact_reference_audit=exact_reference_audit,
    )
