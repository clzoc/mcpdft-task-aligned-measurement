#!/usr/bin/env python3
"""F range over the DQG set with tight practical constraints.

Caps relative to the CASSCF reference:
    |E_H - E_H_ref|        <= 1 mHa                  (absolute)
    |C   - C_ref|          <= 5 mHa                  (common non-on-top part)
    ||D2 - D2_ref||_F      <= 1.01% * ||D2_ref||_F   (1% cap + 1% growth)

The on-top energy is nonlinear; the common part C is an exact convex
quadratic in gamma.  Its upper band is imposed exactly, its lower band by
accumulated tangent (supporting) hyperplanes, which are sufficient for
C >= C_ref - eps_c and are refreshed at every Frank-Wolfe iterate.  The
extremes are found by Frank-Wolfe with exact bounded line search on the
true ftPBE energy.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import sys  # noqa: E402

import cvxpy as cp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parent.parent
PROTOCOL = ROOT / "mcpdft_measurement_revision" / "guarded_mcpdft_shadow_protocol"
for _path in (PROTOCOL / "code", PROTOCOL / "code" / "vendor", PROTOCOL / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from build_casscf_reference import build  # noqa: E402
from constrained_shadow import (  # noqa: E402
    _cvx_d_element,
    _dqg_linear_maps,
    _solver_options,
    dqg_matrices,
    pair_basis,
    rdm_symmetry_blocks,
)
from mcpdft_selector import FtPBEEnergyObjective  # noqa: E402
from ontop_measurement.mcpdft import non_ontop_energy  # noqa: E402

SOLVER = "MOSEK"
TOLERANCE = 1e-9
THREADS = 4
MAX_FW_ITER = 8

EPS_H = 1e-3      # 1 mHa
EPS_C = 5e-3      # 5 mHa
D2_GROWTH = 1.01  # 1% cap grown by 1%


class CommonPart:
    """Exact evaluate/gradient of C = f_non_ontop on the fixed-core grid."""

    def __init__(self, reference):
        self.reference = reference
        self.n_spatial = int(reference.n_spatial_orbitals)
        self.n_modes = int(reference.n_spin_orbitals)
        self.nuclear = float(reference.molecule.energy_nuc())
        self.hcore = reference.mean_field.get_hcore()
        self.core_coeff = np.asarray(reference.mean_field.mo_coeff)[
            :, : int(reference.n_core_orbitals)
        ]
        self.active_coeff = np.asarray(reference.active_mo_coeff)
        self.core_density = 2.0 * self.core_coeff @ self.core_coeff.T
        self.theta_pairs = [
            (i, j)
            for i in range(self.n_spatial)
            for j in range(i, self.n_spatial)
        ]

    def density(self, gamma):
        gamma = np.asarray(gamma)
        spatial = gamma[:self.n_spatial, :self.n_spatial] + gamma[
            self.n_spatial:, self.n_spatial:
        ]
        return (self.core_density
                + self.active_coeff @ spatial @ self.active_coeff.T), spatial

    def value_gradient(self, gamma):
        """C and its full 16x16 symmetric-matrix gradient in gamma."""
        density, spatial = self.density(gamma)
        coulomb = self.reference.mean_field.get_j(dm=density)
        value = non_ontop_energy(self.nuclear, self.hcore, density, coulomb)
        matrix = self.active_coeff.T @ (
            self.hcore + coulomb
        ) @ self.active_coeff
        gradient = np.zeros((self.n_modes, self.n_modes))
        gradient[:self.n_spatial, :self.n_spatial] = matrix
        gradient[self.n_spatial:, self.n_spatial:] = matrix
        return value, gradient, density, spatial

    def spatial_value_gradient(self, spatial_block):
        """C and its theta-gradient for the alpha spatial block X=gamma_aa.

        C depends on X_s = gamma_aa + gamma_bb = 2 X, so the independent
        theta coordinates of X carry dC = 2 Tr[A dX].
        """
        density = (self.core_density
                   + 2.0 * self.active_coeff @ spatial_block
                   @ self.active_coeff.T)
        coulomb = self.reference.mean_field.get_j(dm=density)
        value = non_ontop_energy(self.nuclear, self.hcore, density, coulomb)
        matrix = self.active_coeff.T @ (
            self.hcore + coulomb
        ) @ self.active_coeff
        gradient = np.empty(len(self.theta_pairs))
        for index, (i, j) in enumerate(self.theta_pairs):
            gradient[index] = (2.0 * matrix[i, i] if i == j
                               else 4.0 * matrix[i, j])
        return value, gradient, matrix

    def hessian(self, spatial_ref, step=1e-4):
        """Exact Hessian of C in the independent theta coordinates of X."""
        size = len(self.theta_pairs)
        hessian = np.zeros((size, size))
        for column, (i, j) in enumerate(self.theta_pairs):
            plus = spatial_ref.copy()
            minus = spatial_ref.copy()
            if i == j:
                plus[i, i] += step
                minus[i, i] -= step
            else:
                plus[i, j] += step
                plus[j, i] += step
                minus[i, j] -= step
                minus[j, i] -= step
            _, gp, _ = self.spatial_value_gradient(plus)
            _, gm, _ = self.spatial_value_gradient(minus)
            hessian[:, column] = (gp - gm) / (2.0 * step)
        return 0.5 * (hessian + hessian.T)


class TightDQGSolver:
    def __init__(self, reference, common, eps_d2):
        self.reference = reference
        self.common = common
        n_modes = int(reference.n_spin_orbitals)
        n_pairs = len(reference.pairs)
        n_spatial = int(reference.n_spatial_orbitals)
        n_electrons = int(reference.n_electrons)
        n_alpha, n_beta = int(reference.n_alpha), int(reference.n_beta)
        self.n_modes, self.n_pairs, self.n_spatial = n_modes, n_pairs, n_spatial
        pair_to_index = {pair: index for index, pair in enumerate(reference.pairs)}

        pair_alpha_count = np.array(
            [
                int(first < n_spatial) + int(second < n_spatial)
                for first, second in reference.pairs
            ]
        )
        spin_orbital_irreps = reference.orbital_irreps + reference.orbital_irreps
        pair_irreps = np.array(
            [spin_orbital_irreps[first] ^ spin_orbital_irreps[second]
             for first, second in reference.pairs]
        )
        psd_blocks = rdm_symmetry_blocks(reference)

        d2 = cp.Variable((n_pairs, n_pairs), symmetric=True, name="D")
        gamma = cp.Variable((n_modes, n_modes), symmetric=True, name="gamma")
        self.d2, self.gamma = d2, gamma

        constraints = [
            cp.trace(d2) == math.comb(n_electrons, 2),
            cp.trace(gamma) == n_electrons,
        ]
        for indices in psd_blocks["d2"]:
            constraints.append(d2[np.ix_(indices, indices)] >> 0)
        for indices in psd_blocks["gamma"]:
            block = gamma[np.ix_(indices, indices)]
            constraints.extend([block >> 0, np.eye(len(indices)) - block >> 0])

        for first in range(n_modes):
            for third in range(n_modes):
                contracted = sum(
                    (_cvx_d_element(d2, first, second, third, second, pair_to_index)
                     for second in range(n_modes)),
                    0.0,
                )
                constraints.append(
                    contracted == (n_electrons - 1) * gamma[first, third]
                )

        constraints.extend([
            cp.trace(gamma[:n_spatial, :n_spatial]) == n_alpha,
            cp.trace(gamma[n_spatial:, n_spatial:]) == n_beta,
            gamma[:n_spatial, n_spatial:] == 0,
        ])
        if n_alpha == n_beta:
            spin_exchange = sum(
                (
                    _cvx_d_element(d2, p, q + n_spatial, q, p + n_spatial,
                                   pair_to_index)
                    for p in range(n_spatial)
                    for q in range(n_spatial)
                ),
                0.0,
            )
            constraints.extend([
                spin_exchange == n_electrons / 2,
                gamma[:n_spatial, :n_spatial] == gamma[n_spatial:, n_spatial:],
            ])

        for first in range(n_modes):
            for third in range(first + 1, n_modes):
                if spin_orbital_irreps[first] != spin_orbital_irreps[third]:
                    constraints.append(gamma[first, third] == 0)
        for row in range(n_pairs):
            for col in range(row + 1, n_pairs):
                if (pair_alpha_count[row] != pair_alpha_count[col]
                        or pair_irreps[row] != pair_irreps[col]):
                    constraints.append(d2[row, col] == 0)

        sector_traces = {
            2: math.comb(n_alpha, 2),
            1: n_alpha * n_beta,
            0: math.comb(n_beta, 2),
        }
        for alpha_count, target in sector_traces.items():
            indices = np.flatnonzero(pair_alpha_count == alpha_count)
            constraints.append(cp.sum(cp.diag(d2)[indices]) == target)

        if n_alpha == n_beta:
            spatial_pairs = pair_basis(n_spatial)
            alpha_alpha = np.array(
                [pair_to_index[pair] for pair in spatial_pairs], dtype=int
            )
            beta_beta = np.array(
                [pair_to_index[(first + n_spatial, second + n_spatial)]
                 for first, second in spatial_pairs], dtype=int
            )
            alpha_beta = np.array(
                [pair_to_index[(first, second + n_spatial)]
                 for first in range(n_spatial)
                 for second in range(n_spatial)], dtype=int
            )
            antisymmetric = np.zeros(
                (n_spatial * n_spatial, len(spatial_pairs)), dtype=float
            )
            symmetric_pairs = tuple((p, p) for p in range(n_spatial)) + spatial_pairs
            symmetric = np.zeros(
                (n_spatial * n_spatial, len(symmetric_pairs)), dtype=float
            )
            for column, (first, second) in enumerate(spatial_pairs):
                antisymmetric[first * n_spatial + second, column] = 1.0 / math.sqrt(2.0)
                antisymmetric[second * n_spatial + first, column] = -1.0 / math.sqrt(2.0)
            for column, (first, second) in enumerate(symmetric_pairs):
                if first == second:
                    symmetric[first * n_spatial + second, column] = 1.0
                else:
                    symmetric[first * n_spatial + second, column] = 1.0 / math.sqrt(2.0)
                    symmetric[second * n_spatial + first, column] = 1.0 / math.sqrt(2.0)
            d_alpha_alpha = d2[np.ix_(alpha_alpha, alpha_alpha)]
            d_beta_beta = d2[np.ix_(beta_beta, beta_beta)]
            d_alpha_beta = d2[np.ix_(alpha_beta, alpha_beta)]
            d_triplet = antisymmetric.T @ d_alpha_beta @ antisymmetric
            constraints.extend([
                d_alpha_alpha == d_beta_beta,
                d_triplet == d_alpha_alpha,
                antisymmetric.T @ d_alpha_beta @ symmetric == 0,
            ])

        for first in range(n_modes):
            for third in range(n_modes):
                first_spin = int(first >= n_spatial)
                third_spin = int(third >= n_spatial)
                if first_spin != third_spin:
                    continue
                for summed_spin, spin_particles in ((0, n_alpha), (1, n_beta)):
                    orbitals = (
                        range(0, n_spatial)
                        if summed_spin == 0
                        else range(n_spatial, n_modes)
                    )
                    contracted = sum(
                        (
                            _cvx_d_element(d2, first, second, third, second,
                                           pair_to_index)
                            for second in orbitals
                        ),
                        0.0,
                    )
                    factor = spin_particles - int(first_spin == summed_spin)
                    constraints.append(contracted == factor * gamma[first, third])

        q_d_map, q_gamma_map, g_d_map, g_gamma_map, q_constant = _dqg_linear_maps(
            n_modes, reference.pairs
        )
        d2_vector = cp.vec(d2, order="C")
        gamma_vector = cp.vec(gamma, order="C")
        q2 = cp.reshape(
            q_d_map @ d2_vector + q_gamma_map @ gamma_vector + q_constant,
            (n_pairs, n_pairs),
            order="C",
        )
        q2_symmetric = 0.5 * (q2 + q2.T)
        for indices in psd_blocks["d2"]:
            constraints.append(q2_symmetric[np.ix_(indices, indices)] >> 0)

        g_dimension = n_modes * n_modes
        g2 = cp.reshape(
            g_d_map @ d2_vector + g_gamma_map @ gamma_vector,
            (g_dimension, g_dimension),
            order="C",
        )
        g2_symmetric = 0.5 * (g2 + g2.T)
        for indices in psd_blocks["g"]:
            constraints.append(g2_symmetric[np.ix_(indices, indices)] >> 0)

        electronic_energy = (
            cp.sum(cp.multiply(reference.one_body, gamma))
            + cp.sum(cp.multiply(reference.two_body, d2))
        )
        electronic_reference = reference.exact_energy - reference.nuclear_energy
        constraints.extend([
            electronic_energy <= electronic_reference + EPS_H,
            electronic_energy >= electronic_reference - EPS_H,
            cp.norm(d2 - reference.exact_d2, "fro") <= eps_d2,
        ])

        # ---- exact common-part C band (convex upper, tangent lower) ----
        c_ref, gradient_ref, _, spatial_sum = common.value_gradient(
            reference.exact_gamma
        )
        self.c_ref = c_ref
        spatial_ref = 0.5 * spatial_sum          # X = gamma_aa = gamma_bb
        self.theta_ref = np.array(
            [spatial_ref[i, j] for i, j in common.theta_pairs]
        )
        hessian = common.hessian(spatial_ref)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        self.hessian_factor = (np.sqrt(eigenvalues)[:, None]
                               * eigenvectors.T)
        _, self.spatial_gradient, _ = common.spatial_value_gradient(spatial_ref)
        spatial_block = gamma[:n_spatial, :n_spatial]
        theta = cp.hstack(
            [spatial_block[i, j] for (i, j) in common.theta_pairs]
        )
        self.theta = theta
        self.q = theta - self.theta_ref
        common_deviation = (
            self.spatial_gradient @ self.q
            + 0.5 * cp.sum_squares(self.hessian_factor @ self.q)
        )
        constraints.append(common_deviation <= EPS_C)

        # accumulated tangents: C(gamma_i) + <g_i, gamma-gamma_i> >= C_ref-EPS_C
        self.tangents = [(c_ref, gradient_ref, np.array(reference.exact_gamma))]
        self.base_constraints = constraints

        self.options = _solver_options(SOLVER, TOLERANCE, 20000, False, THREADS)
        self.solve_count = 0

    def add_tangent(self, gamma_point):
        value, gradient, _, _ = self.common.value_gradient(gamma_point)
        self.tangents.append((value, gradient, np.array(gamma_point)))

    def solve(self, d2_gradient, gamma_gradient):
        gD = 0.5 * (np.asarray(d2_gradient) + np.asarray(d2_gradient).T)
        gG = 0.5 * (np.asarray(gamma_gradient) + np.asarray(gamma_gradient).T)
        constraints = list(self.base_constraints)
        for value, gradient, point in self.tangents:
            tangent = cp.sum(cp.multiply(gradient, self.gamma)) - np.sum(
                gradient * point
            )
            constraints.append(tangent >= (self.c_ref - EPS_C) - value)
        objective = cp.Minimize(
            cp.sum(cp.multiply(gD, self.d2))
            + cp.sum(cp.multiply(gG, self.gamma))
        )
        problem = cp.Problem(objective, constraints)
        started = time.time()
        problem.solve(**self.options)
        self.solve_count += 1
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"SDP status {problem.status}")
        return SimpleNamespace(
            d2=np.asarray(self.d2.value),
            gamma=np.asarray(self.gamma.value),
            status=str(problem.status),
            solve_seconds=time.time() - started,
            linear_objective=float(problem.value),
        )


def refine(objective, solver, common, reference, sign, max_iter=MAX_FW_ITER):
    def energy(d2, gamma):
        return objective.evaluate(d2, gamma, gradient=False).total_energy

    current = SimpleNamespace(
        d2=np.array(reference.exact_d2), gamma=np.array(reference.exact_gamma)
    )
    best = SimpleNamespace(
        d2=current.d2.copy(), gamma=current.gamma.copy(),
        energy=sign * energy(current.d2, current.gamma),
    )
    history = []
    for iteration in range(1, max_iter + 1):
        evaluation = objective.evaluate(current.d2, current.gamma, gradient=True)
        oracle = solver.solve(sign * evaluation.d2_gradient,
                              sign * evaluation.gamma_gradient)
        direction_d2 = oracle.d2 - current.d2
        direction_gamma = oracle.gamma - current.gamma
        step_norm = np.linalg.norm(direction_d2) + np.linalg.norm(direction_gamma)

        def line_value(step):
            return sign * energy(
                current.d2 + step * direction_d2,
                current.gamma + step * direction_gamma,
            )

        base = line_value(0.0)
        selected_step = 0.0
        if step_norm > 1e-12:
            refined = minimize_scalar(
                line_value, bounds=(0.0, 1.0), method="bounded",
                options={"xatol": 1e-6, "maxiter": 40},
            )
            if refined.success and float(refined.fun) < base - 1e-10:
                selected_step = float(refined.x)
        selected_energy = line_value(selected_step)
        history.append(dict(
            iteration=iteration,
            oracle_status=oracle.status,
            oracle_seconds=oracle.solve_seconds,
            sdp_linear_objective=oracle.linear_objective,
            step=float(selected_step),
            energy_before=float(base),
            energy_after=float(selected_energy),
            n_tangents=len(solver.tangents),
        ))
        print(
            f"  sign={sign:+d} it={iteration:2d} t={selected_step:.4f} "
            f"F={sign * selected_energy:+.12f}", flush=True,
        )
        if selected_step == 0.0:
            break
        current = SimpleNamespace(
            d2=current.d2 + selected_step * direction_d2,
            gamma=current.gamma + selected_step * direction_gamma,
        )
        solver.add_tangent(current.gamma)
        if sign * selected_energy < best.energy:
            best = SimpleNamespace(
                d2=current.d2.copy(), gamma=current.gamma.copy(),
                energy=sign * selected_energy,
            )
    return best, history


def diagnose(objective, common, reference, candidate, eps_h, eps_c, eps_d2):
    d2 = np.asarray(candidate.d2)
    gamma = np.asarray(candidate.gamma)
    evaluation = objective.evaluate(d2, gamma, gradient=False)
    electronic = (np.sum(reference.one_body * gamma)
                  + np.sum(reference.two_body * d2))
    e_h = electronic + reference.nuclear_energy
    c_value, _, _, _ = common.value_gradient(gamma)
    _, q2, g2 = dqg_matrices(d2, gamma, reference.pairs)
    return dict(
        f_total=float(evaluation.total_energy),
        f_non_ontop=float(evaluation.non_ontop_energy),
        f_on_top=float(evaluation.on_top_energy),
        c_common=float(c_value),
        c_error=float(c_value - common.value_gradient(reference.exact_gamma)[0]),
        e_h=float(e_h),
        e_h_error_meh=float(1e3 * (e_h - reference.exact_energy)),
        d2_error=float(np.linalg.norm(d2 - reference.exact_d2)),
        d2_error_relative=float(
            np.linalg.norm(d2 - reference.exact_d2)
            / np.linalg.norm(reference.exact_d2)
        ),
        min_eig_d2=float(np.linalg.eigvalsh(0.5 * (d2 + d2.T))[0]),
        min_eig_q2=float(np.linalg.eigvalsh(q2)[0]),
        min_eig_g2=float(np.linalg.eigvalsh(g2)[0]),
        minimum_density=float(evaluation.minimum_density),
        minimum_on_top_pair_density=float(
            evaluation.minimum_on_top_pair_density
        ),
        caps=dict(eps_h=eps_h, eps_c=eps_c, eps_d2=eps_d2),
    )


def main():
    started = time.time()
    reference = build()[0]
    obj = FtPBEEnergyObjective(reference, grid_level=1)
    common = CommonPart(reference)
    c_ref, _, _, _ = common.value_gradient(reference.exact_gamma)
    reference_energy = obj.evaluate(
        reference.exact_d2, reference.exact_gamma, gradient=False
    )
    eps_d2 = D2_GROWTH * 0.01 * np.linalg.norm(reference.exact_d2)
    print(
        f"reference: E_H={reference.exact_energy:.12f}  "
        f"F={reference_energy.total_energy:.12f}  C={c_ref:.12f}",
        flush=True,
    )
    print(
        f"caps: |dE_H|<={1e3*EPS_H:.2f} mHa, |dC|<={1e3*EPS_C:.2f} mHa, "
        f"||dD2||_F<={eps_d2:.12f} ({100*eps_d2/np.linalg.norm(reference.exact_d2):.3f}%)",
        flush=True,
    )

    solver = TightDQGSolver(reference, common, eps_d2)
    print("SDP built; solving min F ...", flush=True)
    minimum, min_history = refine(obj, solver, common, reference, sign=+1)
    print("solving max F ...", flush=True)
    maximum, max_history = refine(obj, solver, common, reference, sign=-1)

    payload = dict(
        system="N2", r_angstrom=1.10, basis="cc-pvdz", ncas=8, nelecas=10,
        method="CASSCF", otxc="ftPBE", grid_level=1,
        constraints=dict(
            eps_h_hartree=EPS_H, eps_c_hartree=EPS_C,
            d2_growth=D2_GROWTH, eps_d2=float(eps_d2),
            d2_space="spin-orbital pair basis (120x120)",
        ),
        reference=dict(
            e_h=reference.exact_energy,
            f_total=reference_energy.total_energy,
            f_non_ontop=reference_energy.non_ontop_energy,
            f_on_top=reference_energy.on_top_energy,
            c_common=float(c_ref),
            d2_frobenius=float(np.linalg.norm(reference.exact_d2)),
        ),
        minimum=diagnose(obj, common, reference, minimum, EPS_H, EPS_C, eps_d2),
        maximum=diagnose(obj, common, reference, maximum, EPS_H, EPS_C, eps_d2),
        min_history=min_history,
        max_history=max_history,
        sdp_solves=solver.solve_count,
        total_seconds=time.time() - started,
    )
    reference_f = reference_energy.total_energy
    payload["delta_f"] = dict(
        min=float(payload["minimum"]["f_total"] - reference_f),
        max=float(payload["maximum"]["f_total"] - reference_f),
    )
    (HERE / "range_tight_results.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    np.savez_compressed(
        HERE / "range_tight_extremes.npz",
        d2_min=minimum.d2, gamma_min=minimum.gamma,
        d2_max=maximum.d2, gamma_max=maximum.gamma,
    )
    print(json.dumps({k: payload[k] for k in ("reference", "minimum",
                                              "maximum", "delta_f")},
                     indent=2))


if __name__ == "__main__":
    main()
