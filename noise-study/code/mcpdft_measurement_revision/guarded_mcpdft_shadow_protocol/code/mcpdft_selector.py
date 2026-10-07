"""MC-PDFT ftPBE tangent objectives for constrained-shadow SDPs."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ontop_measurement.mcpdft import (  # noqa: E402
    assemble_frozen_core_fields,
    non_ontop_energy,
    pyscf_gga_arrays,
)
from ontop_measurement.polynomial import (  # noqa: E402
    coefficients_from_tensor_contraction,
    feature_jacobian,
    features,
)


@dataclass(frozen=True)
class FtPBEEvaluation:
    total_energy: float
    non_ontop_energy: float
    on_top_energy: float
    d2_gradient: np.ndarray | None
    gamma_gradient: np.ndarray | None
    minimum_density: float
    minimum_on_top_pair_density: float


@dataclass(frozen=True)
class FtPBEFrankWolfeStep:
    iteration: int
    energy_before: float
    linear_oracle_energy: float
    linearized_gap: float
    selected_step: float
    energy_after: float
    oracle_status: str


@dataclass(frozen=True)
class FtPBEFrankWolfeResult:
    d2: np.ndarray
    gamma: np.ndarray
    total_energy: float
    status: str
    steps: tuple[FtPBEFrankWolfeStep, ...]


def _pair_element(
    d2: np.ndarray,
    first: int,
    second: int,
    third: int,
    fourth: int,
    pair_lookup: dict[tuple[int, int], int],
) -> float:
    if first == second or third == fourth:
        return 0.0
    row_sign = 1 if first < second else -1
    col_sign = 1 if third < fourth else -1
    row = pair_lookup[(min(first, second), max(first, second))]
    col = pair_lookup[(min(third, fourth), max(third, fourth))]
    return row_sign * col_sign * float(d2[row, col])


def pair_basis_to_spatial_casdm2(
    d2: np.ndarray,
    n_spatial: int,
    pairs: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Convert a spin-orbital pair-basis 2-RDM to PySCF's spatial 2-RDM."""

    pair_lookup = {pair: index for index, pair in enumerate(pairs)}
    spatial = np.zeros((n_spatial,) * 4, dtype=float)
    for p in range(n_spatial):
        for q in range(n_spatial):
            for r in range(n_spatial):
                for s in range(n_spatial):
                    value = 0.0
                    for first_spin in (0, 1):
                        for second_spin in (0, 1):
                            value += _pair_element(
                                d2,
                                p + first_spin * n_spatial,
                                r + second_spin * n_spatial,
                                q + first_spin * n_spatial,
                                s + second_spin * n_spatial,
                                pair_lookup,
                            )
                    spatial[p, q, r, s] = value
    return spatial


def _symmetric_d2_variables(reference: Any) -> tuple[np.ndarray, np.ndarray]:
    n_spatial = int(reference.n_spatial_orbitals)
    alpha_counts = np.asarray(
        [
            int(left < n_spatial) + int(right < n_spatial)
            for left, right in reference.pairs
        ]
    )
    spin_irreps = reference.orbital_irreps + reference.orbital_irreps
    pair_irreps = np.asarray(
        [spin_irreps[left] ^ spin_irreps[right] for left, right in reference.pairs]
    )
    variables = [
        (row, col)
        for row in range(len(reference.pairs))
        for col in range(row, len(reference.pairs))
        if alpha_counts[row] == alpha_counts[col]
        and pair_irreps[row] == pair_irreps[col]
    ]
    return (
        np.asarray([row for row, _ in variables], dtype=int),
        np.asarray([col for _, col in variables], dtype=int),
    )


class FtPBEEnergyObjective:
    """Evaluate MC-PDFT ftPBE energy and its RDM tangent on a fixed MO grid."""

    def __init__(self, reference: Any, *, grid_level: int = 1):
        from pyscf import dft
        from pyscf.mcpdft.otfnal import get_transfnal

        if reference.molecule is None or reference.active_mo_coeff is None:
            raise ValueError("The molecular reference must retain PySCF orbitals.")
        self.reference = reference
        self.n_spatial = int(reference.n_spatial_orbitals)
        self.n_pairs = len(reference.pairs)
        self.rows, self.cols = _symmetric_d2_variables(reference)

        grids = dft.gen_grid.Grids(reference.molecule)
        grids.level = int(grid_level)
        grids.build()
        self.coordinates = np.asarray(grids.coords, dtype=float)
        self.weights = np.asarray(grids.weights, dtype=float)

        ao = reference.molecule.eval_gto("GTOval_sph_deriv1", self.coordinates)
        active_coeff = np.asarray(reference.active_mo_coeff, dtype=float)
        core_coeff = np.asarray(reference.mean_field.mo_coeff, dtype=float)[
            :, : int(reference.n_core_orbitals)
        ]
        self.active_coeff = active_coeff
        self.core_coeff = core_coeff
        self.phi = np.asarray(ao[0] @ active_coeff, dtype=float)
        self.dphi = np.einsum(
            "xga,ap->gxp", ao[1:4], active_coeff, optimize=True
        )
        core_values = np.asarray(ao[0] @ core_coeff, dtype=float)
        core_gradients = np.einsum(
            "xga,ap->gxp", ao[1:4], core_coeff, optimize=True
        )
        self.rho_core = 2.0 * np.einsum(
            "gi,gi->g", core_values, core_values, optimize=True
        )
        self.grad_rho_core = 4.0 * np.einsum(
            "gi,gxi->gx", core_values, core_gradients, optimize=True
        )

        self.basis2 = features(self.phi, 2)
        self.basis4 = features(self.phi, 4)
        self.grad_basis2 = np.einsum(
            "gkn,gxn->gxk",
            feature_jacobian(self.phi, 2),
            self.dphi,
            optimize=True,
        )
        self.grad_basis4 = np.einsum(
            "gkn,gxn->gxk",
            feature_jacobian(self.phi, 4),
            self.dphi,
            optimize=True,
        )
        self.functional = get_transfnal(reference.molecule, "ftPBE")
        self.d2_to_quartic = self._build_d2_to_quartic_map()
        self.d2_to_quadratic = self._build_d2_to_quadratic_map()

    def _build_d2_to_quartic_map(self) -> np.ndarray:
        mapping = np.empty((self.basis4.shape[1], len(self.rows)), dtype=float)
        for position, (row, col) in enumerate(zip(self.rows, self.cols)):
            basis = np.zeros((self.n_pairs, self.n_pairs), dtype=float)
            basis[row, col] = 1.0
            basis[col, row] = 1.0
            spatial = pair_basis_to_spatial_casdm2(
                basis, self.n_spatial, self.reference.pairs
            )
            mapping[:, position] = coefficients_from_tensor_contraction(
                spatial, prefactor=0.5
            )
        return mapping

    def _build_d2_to_quadratic_map(self) -> np.ndarray:
        """Map symmetry-reduced D2 variables to spin-summed 1-RDM polynomials."""

        from constrained_shadow import contract_one_rdm

        mapping = np.empty((self.basis2.shape[1], len(self.rows)), dtype=float)
        for position, (row, col) in enumerate(zip(self.rows, self.cols)):
            basis = np.zeros((self.n_pairs, self.n_pairs), dtype=float)
            basis[row, col] = 1.0
            basis[col, row] = 1.0
            gamma = contract_one_rdm(
                basis,
                2 * self.n_spatial,
                self.reference.n_electrons,
                self.reference.pairs,
            )
            spatial_gamma = (
                gamma[: self.n_spatial, : self.n_spatial]
                + gamma[self.n_spatial :, self.n_spatial :]
            )
            mapping[:, position] = coefficients_from_tensor_contraction(
                spatial_gamma
            )
        return mapping

    @staticmethod
    def _clip_response_potential(
        potential: np.ndarray, support: np.ndarray
    ) -> np.ndarray:
        """Suppress numerically irrelevant low-density tail singularities."""

        values = np.nan_to_num(
            np.asarray(potential, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
        )
        active = np.abs(values[..., support]).reshape(-1)
        active = active[active > 0.0]
        if len(active) == 0:
            return np.zeros_like(values)
        limit = max(float(np.quantile(active, 0.995)), 1e-12)
        return np.clip(values, -limit, limit)

    def field_response_modes(
        self,
        d2: np.ndarray,
        gamma: np.ndarray,
        *,
        max_modes: int = 24,
        capture_fraction: float = 0.999,
        density_floor_fraction: float = 1e-10,
    ) -> np.ndarray:
        """Return frozen ftPBE local-response modes for A-optimal acquisition.

        A scalar energy tangent can rotate sharply as a noisy RDM changes. This
        construction instead targets the grid-resolved local response to every
        field actually read by ftPBE: rho, grad(rho), Pi, and grad(Pi). The
        resulting Gauss-Newton metric is compressed to its dominant right
        singular modes and is frozen before adaptive acquisition begins.
        """

        if max_modes < 1:
            raise ValueError("max_modes must be positive.")
        if not 0.0 < capture_fraction <= 1.0:
            raise ValueError("capture_fraction must lie in (0, 1].")
        if density_floor_fraction <= 0.0:
            raise ValueError("density_floor_fraction must be positive.")

        quartic, quadratic, _ = self._coefficients(d2, gamma)
        rho_active = self.basis2 @ quadratic
        grad_rho_active = np.einsum(
            "gxk,k->gx", self.grad_basis2, quadratic, optimize=True
        )
        pair_active = self.basis4 @ quartic
        grad_pair_active = np.einsum(
            "gxk,k->gx", self.grad_basis4, quartic, optimize=True
        )
        fields = assemble_frozen_core_fields(
            rho_active,
            grad_rho_active,
            pair_active,
            grad_pair_active,
            self.rho_core,
            self.grad_rho_core,
        )
        rho, _, _, _ = fields
        rho4, pair4 = pyscf_gga_arrays(*fields)
        rho_rows = 1 if self.functional.dens_deriv == 0 else 4
        pair_rows = 1 if self.functional.Pi_deriv == 0 else 4
        spin_density = np.stack([0.5 * rho4, 0.5 * rho4])
        _, potentials, _ = self.functional.eval_ot(
            spin_density[:, :rho_rows],
            pair4[:pair_rows],
            dderiv=1,
            weights=self.weights,
        )
        density_floor = density_floor_fraction * max(float(np.max(rho)), 1e-30)
        support = np.asarray(rho > density_floor, dtype=bool)
        v_rho = self._clip_response_potential(potentials[0], support)
        v_pair = self._clip_response_potential(potentials[1], support)

        quadratic_response = v_rho[0, :, None] * self.basis2
        quadratic_response += (
            0.5 * self.rho_core * v_pair[0]
        )[:, None] * self.basis2
        if v_rho.shape[0] > 1:
            quadratic_response += np.einsum(
                "xg,gxk->gk", v_rho[1:4], self.grad_basis2, optimize=True
            )
        if v_pair.shape[0] > 1:
            pair_density_response = 0.5 * (
                self.grad_rho_core[:, :, None] * self.basis2[:, None, :]
                + self.rho_core[:, None, None] * self.grad_basis2
            )
            quadratic_response += np.einsum(
                "xg,gxk->gk", v_pair[1:4], pair_density_response, optimize=True
            )

        quartic_response = v_pair[0, :, None] * self.basis4
        if v_pair.shape[0] > 1:
            quartic_response += np.einsum(
                "xg,gxk->gk", v_pair[1:4], self.grad_basis4, optimize=True
            )
        response = (
            quartic_response @ self.d2_to_quartic
            + quadratic_response @ self.d2_to_quadratic
        )
        response *= np.sqrt(np.abs(self.weights))[:, None]
        response[~support] = 0.0

        gram = response.T @ response
        gram = 0.5 * (gram + gram.T)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        positive = eigenvalues > max(float(eigenvalues[-1]), 1.0) * 1e-13
        eigenvalues = eigenvalues[positive][::-1]
        eigenvectors = eigenvectors[:, positive][:, ::-1]
        if len(eigenvalues) == 0:
            raise RuntimeError("The frozen ftPBE field-response metric is zero.")
        cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        mode_count = min(
            max_modes,
            int(np.searchsorted(cumulative, capture_fraction, side="left") + 1),
        )
        return np.sqrt(eigenvalues[:mode_count])[:, None] * eigenvectors[
            :, :mode_count
        ].T

    def _coefficients(
        self, d2: np.ndarray, gamma: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        symmetric_d2 = 0.5 * (np.asarray(d2) + np.asarray(d2).T)
        variables = symmetric_d2[self.rows, self.cols]
        quartic = self.d2_to_quartic @ variables
        gamma_array = 0.5 * (np.asarray(gamma) + np.asarray(gamma).T)
        spatial_gamma = (
            gamma_array[: self.n_spatial, : self.n_spatial]
            + gamma_array[self.n_spatial :, self.n_spatial :]
        )
        quadratic = coefficients_from_tensor_contraction(spatial_gamma)
        return quartic, quadratic, spatial_gamma

    def evaluate(
        self,
        d2: np.ndarray,
        gamma: np.ndarray,
        *,
        gradient: bool = True,
    ) -> FtPBEEvaluation:
        quartic, quadratic, spatial_gamma = self._coefficients(d2, gamma)
        rho_active = self.basis2 @ quadratic
        grad_rho_active = np.einsum(
            "gxk,k->gx", self.grad_basis2, quadratic, optimize=True
        )
        pair_active = self.basis4 @ quartic
        grad_pair_active = np.einsum(
            "gxk,k->gx", self.grad_basis4, quartic, optimize=True
        )
        fields = assemble_frozen_core_fields(
            rho_active,
            grad_rho_active,
            pair_active,
            grad_pair_active,
            self.rho_core,
            self.grad_rho_core,
        )
        rho, grad_rho, pair, grad_pair = fields
        rho4, pair4 = pyscf_gga_arrays(*fields)
        rho_rows = 1 if self.functional.dens_deriv == 0 else 4
        pair_rows = 1 if self.functional.Pi_deriv == 0 else 4
        spin_density = np.stack([0.5 * rho4, 0.5 * rho4])
        derivative_order = 1 if gradient else 0
        energy_density, potentials, _ = self.functional.eval_ot(
            spin_density[:, :rho_rows],
            pair4[:pair_rows],
            dderiv=derivative_order,
            weights=self.weights,
        )
        on_top = float(self.weights @ np.asarray(energy_density))

        density_ao = (
            2.0 * self.core_coeff @ self.core_coeff.T
            + self.active_coeff @ spatial_gamma @ self.active_coeff.T
        )
        coulomb = self.reference.mean_field.get_j(dm=density_ao)
        non_ontop = non_ontop_energy(
            self.reference.molecule.energy_nuc(),
            self.reference.mean_field.get_hcore(),
            density_ao,
            coulomb,
        )
        if not gradient:
            return FtPBEEvaluation(
                total_energy=non_ontop + on_top,
                non_ontop_energy=non_ontop,
                on_top_energy=on_top,
                d2_gradient=None,
                gamma_gradient=None,
                minimum_density=float(np.min(rho)),
                minimum_on_top_pair_density=float(np.min(pair)),
            )

        v_rho = np.asarray(potentials[0], dtype=float)
        v_pair = np.asarray(potentials[1], dtype=float)
        weighted_v_rho = self.weights[None, :] * v_rho
        weighted_v_pair = self.weights[None, :] * v_pair
        quartic_gradient = self.basis4.T @ weighted_v_pair[0]
        if v_pair.shape[0] > 1:
            quartic_gradient += np.einsum(
                "gxk,xg->k",
                self.grad_basis4,
                weighted_v_pair[1:4],
                optimize=True,
            )

        quadratic_gradient = self.basis2.T @ (
            weighted_v_rho[0]
            + 0.5 * self.rho_core * weighted_v_pair[0]
        )
        if v_rho.shape[0] > 1:
            quadratic_gradient += np.einsum(
                "gxk,xg->k",
                self.grad_basis2,
                weighted_v_rho[1:4],
                optimize=True,
            )
        if v_pair.shape[0] > 1:
            pair_gradient_basis2 = 0.5 * (
                self.grad_rho_core[:, :, None] * self.basis2[:, None, :]
                + self.rho_core[:, None, None] * self.grad_basis2
            )
            quadratic_gradient += np.einsum(
                "gxk,xg->k",
                pair_gradient_basis2,
                weighted_v_pair[1:4],
                optimize=True,
            )

        fock_active = self.active_coeff.T @ (
            self.reference.mean_field.get_hcore() + coulomb
        ) @ self.active_coeff
        spatial_pairs = tuple(
            combinations_with_replacement(range(self.n_spatial), 2)
        )
        for index, (row, col) in enumerate(spatial_pairs):
            quadratic_gradient[index] += (
                fock_active[row, col]
                if row == col
                else math.sqrt(2.0) * fock_active[row, col]
            )

        variable_gradient = self.d2_to_quartic.T @ quartic_gradient
        d2_gradient = np.zeros((self.n_pairs, self.n_pairs), dtype=float)
        for value, row, col in zip(variable_gradient, self.rows, self.cols):
            if row == col:
                d2_gradient[row, col] = value
            else:
                d2_gradient[row, col] = 0.5 * value
                d2_gradient[col, row] = 0.5 * value

        spatial_gradient = np.zeros(
            (self.n_spatial, self.n_spatial), dtype=float
        )
        for value, (row, col) in zip(quadratic_gradient, spatial_pairs):
            if row == col:
                spatial_gradient[row, col] = value
            else:
                spatial_gradient[row, col] = value / math.sqrt(2.0)
                spatial_gradient[col, row] = value / math.sqrt(2.0)
        gamma_gradient = np.zeros_like(np.asarray(gamma), dtype=float)
        gamma_gradient[: self.n_spatial, : self.n_spatial] = spatial_gradient
        gamma_gradient[self.n_spatial :, self.n_spatial :] = spatial_gradient
        return FtPBEEvaluation(
            total_energy=non_ontop + on_top,
            non_ontop_energy=non_ontop,
            on_top_energy=on_top,
            d2_gradient=d2_gradient,
            gamma_gradient=gamma_gradient,
            minimum_density=float(np.min(rho)),
            minimum_on_top_pair_density=float(np.min(pair)),
        )


def make_linearized_ftpbe_solver(
    base_solver: Callable[..., Any],
    evaluation: FtPBEEvaluation,
    rmse_cap: float,
) -> Callable[..., Any]:
    """Use one ftPBE tangent as the final objective inside a discrepancy ball."""

    if evaluation.d2_gradient is None or evaluation.gamma_gradient is None:
        raise ValueError("The ftPBE evaluation must include its RDM gradient.")
    d2_gradient = np.asarray(evaluation.d2_gradient, dtype=float).copy()
    gamma_gradient = np.asarray(evaluation.gamma_gradient, dtype=float).copy()

    def solve(reference: Any, shadow_data: Any = None, **kwargs: Any) -> Any:
        kwargs["weighted_shadow_fit"] = shadow_data is not None
        kwargs["weighted_fit_rmse_cap"] = float(rmse_cap)
        kwargs["selection_objective"] = "linear_rdm"
        kwargs["additional_d2_objective"] = d2_gradient
        kwargs["additional_gamma_objective"] = gamma_gradient
        return base_solver(reference, shadow_data=shadow_data, **kwargs)

    return solve


def minimize_ftpbe_frank_wolfe(
    objective: FtPBEEnergyObjective,
    initial_result: Any,
    linear_oracle: Callable[[FtPBEEvaluation, Any, int], Any],
    *,
    max_iterations: int = 3,
    line_steps: Sequence[float] = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0),
    energy_tolerance: float = 1e-7,
) -> FtPBEFrankWolfeResult:
    """Minimize true ftPBE total energy over a convex RDM feasible region.

    ``linear_oracle`` must minimize the current ftPBE tangent over the same
    convex feasible set as ``initial_result``. Convex line search therefore
    keeps every accepted iterate feasible even though ftPBE itself is
    nonlinear and is not represented directly inside CVXPY.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    steps = tuple(sorted(set(float(value) for value in line_steps)))
    if not steps or steps[0] != 0.0 or steps[-1] != 1.0:
        raise ValueError("line_steps must include both 0 and 1.")
    if any(value < 0.0 or value > 1.0 for value in steps):
        raise ValueError("line_steps must lie in [0, 1].")
    if energy_tolerance < 0.0:
        raise ValueError("energy_tolerance must be nonnegative.")

    from types import SimpleNamespace

    current = SimpleNamespace(
        d2=np.asarray(initial_result.d2, dtype=float).copy(),
        gamma=np.asarray(initial_result.gamma, dtype=float).copy(),
        status=str(getattr(initial_result, "status", "initial")),
    )
    history: list[FtPBEFrankWolfeStep] = []
    final_status = "maximum_iterations"
    for iteration in range(1, max_iterations + 1):
        evaluation = objective.evaluate(current.d2, current.gamma, gradient=True)
        oracle = linear_oracle(evaluation, current, iteration)
        oracle_d2 = np.asarray(oracle.d2, dtype=float)
        oracle_gamma = np.asarray(oracle.gamma, dtype=float)
        d2_direction = oracle_d2 - current.d2
        gamma_direction = oracle_gamma - current.gamma
        gap = -float(
            np.sum(evaluation.d2_gradient * d2_direction)
            + np.sum(evaluation.gamma_gradient * gamma_direction)
        )
        oracle_energy = objective.evaluate(
            oracle_d2, oracle_gamma, gradient=False
        ).total_energy

        line_energies: dict[float, float] = {0.0: evaluation.total_energy}
        for step in steps[1:]:
            line_energies[step] = objective.evaluate(
                current.d2 + step * d2_direction,
                current.gamma + step * gamma_direction,
                gradient=False,
            ).total_energy
        selected_step = min(steps, key=lambda value: (line_energies[value], value))
        selected_energy = line_energies[selected_step]

        if selected_step not in {0.0, 1.0}:
            from scipy.optimize import minimize_scalar

            position = steps.index(selected_step)
            lower = steps[position - 1]
            upper = steps[position + 1]

            def line_energy(step: float) -> float:
                return objective.evaluate(
                    current.d2 + step * d2_direction,
                    current.gamma + step * gamma_direction,
                    gradient=False,
                ).total_energy

            refined = minimize_scalar(
                line_energy,
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": 1e-4, "maxiter": 24},
            )
            if refined.success and float(refined.fun) < selected_energy:
                selected_step = float(refined.x)
                selected_energy = float(refined.fun)

        if selected_energy >= evaluation.total_energy - energy_tolerance:
            selected_step = 0.0
            selected_energy = evaluation.total_energy
            final_status = "converged_no_descent"
        else:
            current = SimpleNamespace(
                d2=current.d2 + selected_step * d2_direction,
                gamma=current.gamma + selected_step * gamma_direction,
                status=str(getattr(oracle, "status", "unknown")),
            )
        history.append(
            FtPBEFrankWolfeStep(
                iteration=iteration,
                energy_before=float(evaluation.total_energy),
                linear_oracle_energy=float(oracle_energy),
                linearized_gap=float(gap),
                selected_step=float(selected_step),
                energy_after=float(selected_energy),
                oracle_status=str(getattr(oracle, "status", "unknown")),
            )
        )
        if selected_step == 0.0:
            break

    final = objective.evaluate(current.d2, current.gamma, gradient=False)
    return FtPBEFrankWolfeResult(
        d2=current.d2,
        gamma=current.gamma,
        total_energy=float(final.total_energy),
        status=final_status,
        steps=tuple(history),
    )
