#!/usr/bin/env python3
"""Pilot-driven four-target weak-mode rule with a DQG-lineality guard.

Truth-free inputs only: frame designs, pilot covariances, the Hamiltonian
gradient, the fixed-anchor FTPBE gradient and the anchor DQG matrices.

The target direction is split through the 1-RDM contraction channel
``gamma[p,r] = sum_q D[pq,rq]/(N-1)``:

  common  : P1 (h_hat + f_hat)      1-RDM channel shared by H and F
  h_only  : (1 - P1) h_hat          H-specific remainder
  f_only  : (1 - P1) f_hat          F-specific remainder
  diff    : h_only - f_only         H/F disagreement direction
  sum3    : unit(common) + unit(h_only) + unit(f_only)  equal-weight mixture

For a target t the per-frame scores are

  q_f  = t^T I_f t                          I_f = A_f^T Cov_f^{-1} A_f
  qL_f = w^T (L^T I_f L)^+ w,   w = L^T t   restricted to the anchor lineality

and s_f = q_f / sum(q) + wL * qL_f / sum(qL).  Frames are ordered by s_f;
K is the smallest prefix reaching rho=80% of the total with full affine rank,
then extended until the selected subset's lineality minimum gain per frame is
at least CONIC_GUARD_FRACTION of the full-pool value.  All 30 x 500 pilot
shots are charged; unselected pilots never enter the fit.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, null_space

import gl15 as g
import mcpdft_derandomization as md
from constrained_shadow import contract_one_rdm, dqg_matrices

FRAMES = 30
PILOT = 500
RHO = .8
WEAK_FRACTION = .5
LINEALITY_TOL = 1e-8
CONIC_GUARD_FRACTION = .10
LINEALITY_WEIGHT = 1.0
LINEALITY_DIRECTION_FLOOR = 1e-3
TARGETS = ("common", "h_only", "f_only", "diff", "sum3")


def information_matrices(covariances, designs):
    infos = []
    for design, covariance in zip(designs, covariances):
        values, vectors = eigh(covariance)
        cutoff = max(values[-1] * 1e-10, 1e-12)
        positive = values > cutoff
        inverse = (vectors[:, positive] / values[positive]) @ vectors[:, positive].T
        infos.append(design.T @ inverse @ design)
    return np.asarray(infos)


def design_rank(designs):
    stacked = np.vstack(designs)
    singular = np.linalg.svd(stacked, compute_uv=False)
    cut = singular[0] * 1e-10 if singular.size else 0.0
    return int(np.count_nonzero(singular > cut))


def _subspace_projector(columns, cutoff=1e-10):
    columns = np.asarray(columns, dtype=float)
    u, s, _ = np.linalg.svd(columns, full_matrices=False)
    tol = (s[0] * cutoff) if s.size else 0.0
    keep = s > tol
    basis = u[:, keep]
    return basis @ basis.T, int(keep.sum())


def channel_projector(c):
    """Projector onto the 1-RDM contraction channel in 181-d chart space."""
    sel = c.sel
    n_modes = int(sel.n_spin_orbitals)
    rows, cols = c.geo["rows"], c.geo["cols"]
    columns = []
    for first in range(n_modes):
        for second in range(first, n_modes):
            basis = np.zeros((n_modes, n_modes))
            basis[first, second] = 1.0
            basis[second, first] = 1.0
            pull = md.contraction_gamma_gradient_vector(
                basis, sel.pairs, rows, cols, sel.n_electrons)
            columns.append(pull @ c.lift)
    return _subspace_projector(np.column_stack(columns))


def target_vectors(c):
    h_hat = np.asarray(c.h, dtype=float)
    f_hat = np.asarray(c.f, dtype=float)
    h_hat = h_hat / np.linalg.norm(h_hat)
    f_hat = f_hat / np.linalg.norm(f_hat)
    projector, rank = channel_projector(c)
    identity = np.eye(projector.shape[0])
    u_common = projector @ (h_hat + f_hat)
    u_h = (identity - projector) @ h_hat
    u_f = (identity - projector) @ f_hat

    def unit(vector):
        value = np.linalg.norm(vector)
        return vector / value if value > 0 else vector

    u_sum = unit(u_common) + unit(u_h) + unit(u_f)
    return {
        "common": u_common,
        "h_only": u_h,
        "f_only": u_f,
        "diff": u_h - u_f,
        "sum3": u_sum,
    }, {
        "channel_rank": rank,
        "channel_fraction_h": float(np.linalg.norm(projector @ h_hat)),
        "channel_fraction_f": float(np.linalg.norm(projector @ f_hat)),
        "h_f_cosine": float(h_hat @ f_hat),
        "norm_h_only": float(np.linalg.norm(u_h)),
        "norm_f_only": float(np.linalg.norm(u_f)),
        "norm_common": float(np.linalg.norm(u_common)),
        "norm_diff": float(np.linalg.norm(u_h - u_f)),
        "norm_sum3": float(np.linalg.norm(u_sum)),
    }


def _chart_perturbation(values, rows, cols, dimension):
    matrix = np.zeros((dimension, dimension))
    matrix[rows, cols] = values
    matrix[cols, rows] = values
    return matrix


def lineality(c, tolerance=LINEALITY_TOL):
    """First-order DQG lineality basis in chart coordinates, at the anchor."""
    sel, base = c.sel, c.base
    rows, cols = c.geo["rows"], c.geo["cols"]
    matrices = dqg_matrices(base.d2, base.gamma, sel.pairs)
    active_vectors, spectra = [], {}
    for label, matrix in zip(("D", "Q", "G"), matrices):
        values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
        active = vectors[:, values <= tolerance]
        active_vectors.append(active)
        positive = values[values > tolerance]
        spectra[label] = {
            "active_eigenvalues": int(active.shape[1]),
            "minimum_eigenvalue": float(values[0]),
            "minimum_positive_eigenvalue": float(positive.min()) if len(positive) else None,
        }
    derivatives = [[] for _ in matrices]
    dimension = matrices[0].shape[0]
    for index in range(c.lift.shape[1]):
        perturbation = _chart_perturbation(c.lift[:, index], rows, cols, dimension)
        gamma_delta = contract_one_rdm(perturbation, sel.n_spin_orbitals,
                                       sel.n_electrons, sel.pairs)
        trial = dqg_matrices(base.d2 + perturbation, base.gamma + gamma_delta,
                             sel.pairs)
        for block in range(3):
            derivatives[block].append(trial[block] - matrices[block])
    constraints = []
    for active, blocks in zip(active_vectors, derivatives):
        if active.shape[1] == 0:
            continue
        upper = np.triu_indices(active.shape[1])
        constraints.append(np.column_stack(
            [(active.T @ block @ active)[upper] for block in blocks]).T)
    if constraints:
        constraint_matrix = np.vstack([item.T for item in constraints])
        basis = null_space(constraint_matrix)
    else:
        basis = np.eye(len(rows))
    return basis, {
        "active_tolerance": tolerance,
        "lineality_dimension": int(basis.shape[1]),
        "constraint_rank": int(constraint_matrix.shape[0]) if constraints else 0,
        "spectra": spectra,
    }


def risk_model(designs, covariances, h, f, target_rank):
    """Rank-aware risk model (upstream build_risks when the pool is full rank)."""
    rank = design_rank(designs)
    if rank != target_rank:
        raise ValueError(f"design rank {rank} != target rank {target_rank}")
    design = np.vstack(designs)
    u, singular, vt = np.linalg.svd(design, full_matrices=False)
    cut = singular[0] * 1e-10 if singular.size else 0.0
    keep = singular > cut
    inverse = (vt.T[:, keep] / singular[keep]) @ u[:, keep].T
    energy = np.stack((h, f))
    joint, frobenius = [], []
    start = 0
    for block, covariance in zip(designs, covariances):
        back = inverse[:, start:start + len(block)]
        propagated = back @ covariance @ back.T
        local = energy @ propagated @ energy.T
        joint.append((local + local.T) / 2)
        frobenius.append(float(np.trace(propagated)))
        start += len(block)
    joint = np.asarray(joint)
    frobenius = np.asarray(frobenius)
    eigenvalues, vectors = eigh(joint.sum(0))
    if eigenvalues[0] <= eigenvalues[-1] * 1e-12:
        raise ValueError("H/F covariance is numerically rank deficient.")
    whitening = (vectors / np.sqrt(eigenvalues)).T
    whitened = np.asarray([whitening @ block @ whitening.T for block in joint])
    return dict(joint=joint, frobenius=frobenius, whitened=whitened,
                whitening=whitening, rank=rank,
                condition=float(singular[0] / singular[keep][-1]),
                gradient_cosine=float(h @ f / np.linalg.norm(h)
                                      / np.linalg.norm(f)))


def _restricted_min_gain(infos_sum, basis):
    if basis.shape[1] == 0:
        return None
    restricted = basis.T @ infos_sum @ basis
    return float(np.linalg.eigvalsh(0.5 * (restricted + restricted.T))[0])


def _restricted_floor(matrix):
    values, vectors = eigh(0.5 * (matrix + matrix.T))
    cutoff = max(values[-1] * 1e-10, 1e-12)
    positive = values > cutoff
    return vectors[:, positive], values[positive]


def build_model(c, d, stream):
    counts = np.full(FRAMES, PILOT, dtype=int)
    pilot = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream), counts)
    covariances = np.asarray([d.r.robust_covariance(c, sample) for sample in pilot])
    designs = np.asarray([block @ c.lift for block in c.blocks])
    infos = information_matrices(covariances, designs)
    vectors, channel = target_vectors(c)
    basis, lineality_diagnostics = lineality(c)
    return {
        "stream": int(stream),
        "covariances": covariances,
        "designs": designs,
        "infos": infos,
        "targets": vectors,
        "lineality": basis,
        "lineality_diagnostics": lineality_diagnostics,
        "channel_diagnostics": channel,
        "pool_rank": design_rank(designs),
        "pool_information": infos.sum(0),
    }


def select(model, c, target, budget, rho=RHO):
    infos = model["infos"]
    designs = model["designs"]
    basis = model["lineality"]
    full = model["pool_information"]
    target_rank = int(model["pool_rank"])
    values, vectors = eigh(0.5 * (full + full.T))
    if target_rank < designs.shape[1]:
        cutoff = max(values[-1] * 1e-10, 1e-12)
        identifiable = vectors[:, values > cutoff]
        count = max(1, int(identifiable.shape[1] * WEAK_FRACTION))
        weak = identifiable[:, :count]
    else:
        weak = vectors[:, :max(1, int(len(values) * WEAK_FRACTION))]
    direction = np.asarray(model["targets"][target], dtype=float)
    projected = weak @ (weak.T @ direction)
    norm = np.linalg.norm(projected)
    if norm <= 1e-12:
        raise ValueError(f"Target {target} vanishes in the weak subspace")
    t = projected / norm
    q = np.clip(np.einsum("i,fij,j->f", t, infos, t, optimize=True), 0, None)
    w = basis.T @ t
    lineality_active = (basis.shape[1] > 0
                        and np.linalg.norm(w) > LINEALITY_DIRECTION_FLOOR)
    ql = np.zeros(FRAMES)
    if lineality_active:
        for index, info in enumerate(infos):
            restricted = basis.T @ info @ basis
            positive, spectrum = _restricted_floor(restricted)
            coefficients = positive.T @ w
            ql[index] = float(np.sum(coefficients ** 2 / spectrum))
        ql = np.clip(ql, 0, None)
    score = q / max(q.sum(), 1e-30)
    if lineality_active and ql.sum() > 0:
        score = score + LINEALITY_WEIGHT * ql / ql.sum()
    order = np.argsort(-score)
    cumulative = np.cumsum(score[order])
    k = int(np.searchsorted(cumulative, rho * cumulative[-1]) + 1)
    k_floor = k
    for k in range(max(k_floor, 2), FRAMES + 1):
        if design_rank(designs[order[:k]]) >= target_rank:
            break
    else:
        raise RuntimeError(f"The pool never reaches rank {target_rank}")
    pool_gain = _restricted_min_gain(full, basis) if basis.shape[1] else None
    guard_ratio = None
    if pool_gain and pool_gain > 0:
        while k < FRAMES:
            subset_gain = _restricted_min_gain(infos[order[:k]].sum(0), basis)
            guard_ratio = subset_gain / k / (pool_gain / FRAMES)
            if guard_ratio >= CONIC_GUARD_FRACTION:
                break
            k += 1
        guard_ratio = (_restricted_min_gain(infos[order[:k]].sum(0), basis)
                       / k / (pool_gain / FRAMES))
    indices = np.asarray(order[:k], dtype=int)
    if target_rank == designs.shape[1]:
        risks = g.allocation.build_risks(designs[indices],
                                         model["covariances"][indices], c.h, c.f)
    else:
        risks = risk_model(designs[indices], model["covariances"][indices],
                           c.h, c.f, target_rank)
    measured = budget - (FRAMES - k) * PILOT
    fit_counts, allocation_diagnostic = g.allocation.allocate(
        risks, measured, np.full(k, PILOT, dtype=int), "joint")
    counts = np.full(FRAMES, PILOT, dtype=int)
    counts[indices] = fit_counts
    return {
        "target": target,
        "k": int(k),
        "k_floor": int(k_floor),
        "indices": indices.tolist(),
        "counts": counts.tolist(),
        "measured_fit_shots": int(measured),
        "captured_fraction": float(cumulative[k - 1] / cumulative[-1]),
        "scores": score.tolist(),
        "weak_information": q.tolist(),
        "lineality_information": ql.tolist(),
        "lineality_active": bool(lineality_active and ql.sum() > 0),
        "lineality_guard_ratio": None if guard_ratio is None else float(guard_ratio),
        "lineality_dimension": int(basis.shape[1]),
        "channel_diagnostics": model["channel_diagnostics"],
        "lineality_diagnostics": model["lineality_diagnostics"],
        "allocation_diagnostic": allocation_diagnostic,
    }
