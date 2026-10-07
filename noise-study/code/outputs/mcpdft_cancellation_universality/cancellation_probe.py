#!/usr/bin/env python3
"""Universality probe for the qr_nuclear energy-cancellation structure.

Question: the Eq. (11) qr_nuclear estimator (min H energy + ||DeltaD||_* subject
to DQG + QR-selected literal shadow equalities) pushes the reconstruction error
into directions where the d2-part and the contraction(gamma)-part of the energy
error cancel (CR = |total|/(|d2 part| + |gamma part|) << 1 on C2 and N2).  Is
that a C2/late-basis special case or generic across small molecules?

For each system this script runs
  probe A: exact-mean (noiseless) constraints -> qr_nuclear SDP -> structural
           DeltaD -> exact split of the H error and direct/linearized split of
           the ftPBE error into d2 and gamma parts -> CR_H, CR_F.
  probe B: one Born-sampled finite-shot stream (uniform over the selected
           frames) -> qr_nuclear and qr_gls reconstructions -> same split.
  probe C: 200 random directions in the unmeasured-row complement, rescaled to
           ||DeltaD|| from probe A -> CR distribution as the no-cancellation
           reference scale.

All library code is imported unchanged from the GMSP tree; nothing there is
modified.  Gates (see VALIDATION.json):
  1. reference energy identity |rdm_energy - CASCI| < 1e-8 Eh per system;
  2. decomposition identity |h_d2 + h_g - h_total| < 1e-6 mEh per record;
  3. N2 (1.10 Ang, cc-pVDZ, CAS(6,6)) probe B: CR_H(qr_nuclear) < 0.8 and
     CR_H(qr_gls) > CR_H(qr_nuclear).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.linalg import cholesky, qr, solve_triangular  # noqa: E402

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent.parent
GMSP = ANALYSIS / "mcpdft_measurement_revision" / "guarded_mcpdft_shadow_protocol"
for path in (GMSP / "code", GMSP / "code" / "vendor", GMSP / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constrained_shadow import (  # noqa: E402
    AffineRDMObjective,
    MolecularReference,
    ShadowData,
    _canonicalize_mo_signs,
    _canonicalize_statevector_phase,
    _hamiltonian_coefficients,
    contract_one_rdm,
    exterior_square,
    pair_basis,
    quadratic_design,
    rdm_energy,
    rdms_from_statevector,
    solve_dqg_sdp,
    spin_orbital_rotation,
)
from mcpdft_derandomization import (  # noqa: E402
    LeakGuardReference,
    build_c2_reference,
)
from mcpdft_selector import FtPBEEnergyObjective, _symmetric_d2_variables  # noqa: E402
from run_c2_ftpbe_shot_reallocation import AcquisitionOracle  # noqa: E402
from run_n2_ftpbe_shot_cost_oracle import (  # noqa: E402
    RELATIVE_FLOOR,
    SHRINKAGE,
    _exact_single_covariance,
    _frame_block,
)
from run_n2_joint_shadow_norm_conic_oracle import (  # noqa: E402
    ConstraintBasis,
    _finite_shadows,
    _sample_outcomes,
)
from run_n2_paper_nuclear_hybrid_trajectories import _random_unitaries  # noqa: E402
from run_n2_random_adaptive_allocation import (  # noqa: E402
    _energy_values,
    _exact_mean_shadows,
    _random_mixed_frames,
)


# --------------------------------------------------------------------------
# System grid
# --------------------------------------------------------------------------

SYSTEMS: tuple[dict[str, Any], ...] = (
    dict(
        name="n2_r1.10", atom="N 0 0 -0.55; N 0 0 0.55", basis="cc-pvdz",
        active_electrons=6, active_orbitals=6, ncore=4, bond_length=1.10,
        frame_count=30, builder="canonical", gate=True,
    ),
    dict(
        name="h2_r0.74", atom="H 0 0 -0.37; H 0 0 0.37", basis="sto-3g",
        active_electrons=2, active_orbitals=2, ncore=0, bond_length=0.74,
        frame_count=10, builder="canonical",
    ),
    dict(
        name="h2_r2.00", atom="H 0 0 -1.0; H 0 0 1.0", basis="sto-3g",
        active_electrons=2, active_orbitals=2, ncore=0, bond_length=2.00,
        frame_count=10, builder="canonical",
    ),
    dict(
        name="li2_r2.67", atom="Li 0 0 -1.335; Li 0 0 1.335", basis="cc-pvdz",
        active_electrons=6, active_orbitals=6, ncore=0, bond_length=2.67,
        frame_count=30, builder="canonical",
    ),
    dict(
        name="li2_r4.00", atom="Li 0 0 -2.0; Li 0 0 2.0", basis="cc-pvdz",
        active_electrons=6, active_orbitals=6, ncore=0, bond_length=4.00,
        frame_count=30, builder="canonical",
    ),
    dict(
        name="n2_r1.75", atom="N 0 0 -0.875; N 0 0 0.875", basis="cc-pvdz",
        active_electrons=6, active_orbitals=6, ncore=4, bond_length=1.75,
        frame_count=30, builder="canonical",
    ),
    dict(
        name="beh2_r1.32", atom="Be 0 0 0; H 0 0 -1.32; H 0 0 1.32",
        basis="cc-pvdz", active_electrons=4, active_orbitals=4, ncore=1,
        bond_length=1.32, frame_count=30, builder="canonical",
    ),
    dict(
        name="c2_r1.25_pvdz", atom="C 0 0 -0.625; C 0 0 0.625",
        basis="cc-pvdz", active_electrons=8, active_orbitals=8, ncore=2,
        bond_length=1.25, frame_count=30, builder="c2_avas",
    ),
    dict(
        name="c2_r1.25_pvtz", atom="C 0 0 -0.625; C 0 0 0.625",
        basis="cc-pvtz", active_electrons=8, active_orbitals=8, ncore=2,
        bond_length=1.25, frame_count=30, builder="c2_avas", optional=True,
    ),
    dict(
        name="f2_r1.41", atom="F 0 0 -0.706; F 0 0 0.706", basis="cc-pvdz",
        active_electrons=8, active_orbitals=6, ncore=5, bond_length=1.412,
        frame_count=30, builder="canonical", optional=True,
    ),
    dict(
        name="hf_r0.92", atom="H 0 0 0; F 0 0 0.917", basis="cc-pvdz",
        active_electrons=6, active_orbitals=6, ncore=2, bond_length=0.917,
        frame_count=30, builder="canonical", symmetry=None, optional=True,
    ),
    dict(
        name="cr2_r1.68", atom="Cr 0 0 -0.84; Cr 0 0 0.84", basis="cc-pvtz",
        active_electrons=12, active_orbitals=12, ncore=18, bond_length=1.68,
        frame_count=30, builder="cr2_cached", optional=True,
    ),
)

# Frozen protocol constants (N2 random-frame convention).
REAL_FRAME_SEED = 20260716
COMPLEX_FRAME_SEED = 271828
PLACEMENT_SEED = 314159
COMPLEX_FRACTION = 0.5
SHOT_SEED = 90721
RANDOM_DIRECTION_SEED = 60631
NUCLEAR_WEIGHT = 1.0
GRID_LEVEL = 1
SOLVER_KWARGS = dict(
    solver="MOSEK", tolerance=1e-8, max_iterations=3000, solver_threads=4
)
# Set from main() for systems whose nuclear-norm SDP exceeds the MOSEK memory
# envelope (Cr2): overrides solver/tolerance/iterations for qr_nuclear only.
_NUCLEAR_KWARGS_OVERRIDE: dict[str, Any] | None = None

GATE_SYSTEM = "n2_r1.10"
GATE_ENERGY_IDENTITY_EH = 1e-8
GATE_DECOMPOSITION_MEH = 1e-6
GATE_N2_CR_H_NUCLEAR = 0.8

# Cr2 CAS(12,12)/cc-pVTZ staged cache (CASSCF orbitals + exact RDMs, no CI
# vector; the CI is regenerated by one CASCI on the cached orbitals).
CR2_CACHE = (
    GMSP
    / "sweeps"
    / "cr2_equilibrium_exact_shadow"
    / "reference"
    / "cr2_ccpvtz_casscf_irrep.npz"
)
CR2_EXPECTED_ENERGY = -2086.6555946298045


def _peak_rss_gb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# --------------------------------------------------------------------------
# Reference construction (mirrors build_n2_reference, generalized)
# --------------------------------------------------------------------------

def build_canonical_cas_reference(spec: dict[str, Any]) -> MolecularReference:
    """CASCI reference on canonical RHF orbitals for a singlet linear molecule."""

    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from pyscf import ao2mo, fci, gto, mcscf, scf, symm

    active_electrons = int(spec["active_electrons"])
    active_orbitals = int(spec["active_orbitals"])
    symmetry = spec.get("symmetry", "D2h")
    molecule = gto.M(
        atom=spec["atom"],
        basis=spec["basis"],
        charge=0,
        spin=0,
        unit="Angstrom",
        symmetry=symmetry if symmetry else False,
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-10
    mean_field.max_cycle = 100
    mean_field.kernel()
    if not mean_field.converged:
        mean_field = mean_field.newton().run()
    if not mean_field.converged:
        raise RuntimeError(f"{spec['name']}: RHF did not converge.")
    mo_coeff = _canonicalize_mo_signs(mean_field.mo_coeff)
    ncore = spec["ncore"]
    if ncore is None:
        ncore = (molecule.nelectron - active_electrons) // 2
    ncore = int(ncore)
    if 2 * ncore + active_electrons != molecule.nelectron:
        raise ValueError(
            f"{spec['name']}: ncore={ncore} plus CAS({active_electrons},"
            f"{active_orbitals}) does not cover {molecule.nelectron} electrons."
        )
    mean_field.mo_coeff = mo_coeff

    casci = mcscf.CASCI(mean_field, active_orbitals, active_electrons)
    casci.ncore = ncore
    casci.kernel()
    if not casci.converged:
        raise RuntimeError(f"{spec['name']}: CASCI did not converge.")
    n_alpha = active_electrons // 2
    n_beta = active_electrons // 2
    spin_square = float(
        casci.fcisolver.spin_square(casci.ci, active_orbitals, (n_alpha, n_beta))[0]
    )
    if abs(spin_square) > 1e-7:
        raise RuntimeError(f"{spec['name']}: singlet check failed, S^2={spin_square}.")

    n_modes = 2 * active_orbitals
    alpha_strings = fci.cistring.make_strings(range(active_orbitals), n_alpha)
    beta_strings = fci.cistring.make_strings(range(active_orbitals), n_beta)
    statevector = np.zeros(1 << n_modes, dtype=complex)
    for alpha_index, alpha_string in enumerate(alpha_strings):
        for beta_index, beta_string in enumerate(beta_strings):
            basis_state = int(alpha_string) | (int(beta_string) << active_orbitals)
            statevector[basis_state] = casci.ci[alpha_index, beta_index]
    statevector /= np.linalg.norm(statevector)
    statevector = _canonicalize_statevector_phase(statevector)

    pairs = pair_basis(n_modes)
    gamma, d2 = rdms_from_statevector(statevector, n_modes, pairs)
    one_electron, core_energy = casci.get_h1eff()
    two_electron = ao2mo.restore(1, casci.get_h2eff(), active_orbitals)
    active_hamiltonian = ElectronicEnergy.from_raw_integrals(
        one_electron, two_electron
    ).second_q_op()
    one_body, two_body = _hamiltonian_coefficients(active_hamiltonian, n_modes, pairs)

    if symmetry:
        orbital_labels = symm.label_orb_symm(
            molecule,
            molecule.irrep_name,
            molecule.symm_orb,
            mean_field.mo_coeff,
            check=True,
        )
        active_labels = orbital_labels[ncore : ncore + active_orbitals]
        orbital_irreps = tuple(
            int(symm.irrep_name2id(molecule.groupname, label)) for label in active_labels
        )
    else:
        # No point-group symmetry (e.g. heteronuclear diatomics): a single
        # trivial block is a valid input -- irreps are only XOR-compared.
        orbital_irreps = tuple(0 for _ in range(active_orbitals))

    exact_energy = float(casci.e_tot)
    reconstructed_energy = rdm_energy(
        d2, gamma, one_body, two_body, float(core_energy)
    )
    identity_error = abs(exact_energy - reconstructed_energy)
    if not np.isclose(exact_energy, reconstructed_energy, atol=1e-9):
        raise RuntimeError(
            f"{spec['name']}: CASCI Hamiltonian and pair-basis 2-RDM disagree: "
            f"{exact_energy} versus {reconstructed_energy}."
        )

    reference = MolecularReference(
        statevector=statevector,
        exact_energy=exact_energy,
        nuclear_energy=float(core_energy),
        exact_gamma=gamma,
        exact_d2=d2,
        one_body=one_body,
        two_body=two_body,
        pairs=pairs,
        n_spatial_orbitals=active_orbitals,
        n_spin_orbitals=n_modes,
        n_alpha=n_alpha,
        n_beta=n_beta,
        orbital_irreps=orbital_irreps,
        molecule=molecule,
        mean_field=mean_field,
        active_mo_coeff=np.asarray(mean_field.mo_coeff[:, ncore : ncore + active_orbitals]),
        n_core_orbitals=ncore,
    )
    aux = {"rhf_energy_eh": float(mean_field.e_tot)}
    return reference, identity_error, aux


def build_cr2_cached_reference(spec: dict[str, Any]) -> MolecularReference:
    """Cr2 CAS(12,12)/cc-pVTZ reference from the staged CASSCF cache.

    The cache ships CASSCF-optimized MOs and the exact RDMs but no CI vector;
    one CASCI(12,12) on the cached orbitals regenerates the statevector used
    for Born sampling.  The cached RDMs stay authoritative; the regenerated CI
    is checked against them before use.
    """

    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from pyscf import ao2mo, fci, gto, mcscf, scf, symm

    from constrained_shadow import rdms_from_pyscf_spin_blocks

    with np.load(CR2_CACHE, allow_pickle=False) as archive:
        orbitals = np.asarray(archive["mo_coeff"], dtype=float)
        cache_gamma = np.asarray(archive["exact_gamma"], dtype=float)
        cache_d2 = np.asarray(archive["exact_d2"], dtype=float)
        cache_energy = float(np.asarray(archive["exact_energy"]).item())
    if abs(cache_energy - CR2_EXPECTED_ENERGY) > 1e-9:
        raise RuntimeError(
            f"cr2: cache energy {cache_energy} != {CR2_EXPECTED_ENERGY}."
        )

    molecule = gto.M(
        atom=spec["atom"],
        basis=spec["basis"],
        charge=0,
        spin=0,
        unit="Angstrom",
        symmetry="D2h",
        verbose=0,
        max_memory=8_000,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-10
    mean_field.max_cycle = 200
    mean_field.kernel()
    if not mean_field.converged:
        mean_field = mean_field.newton().run()
    if not mean_field.converged:
        raise RuntimeError("cr2: RHF did not converge.")
    rhf_energy = float(mean_field.e_tot)

    metric = orbitals.T @ mean_field.get_ovlp() @ orbitals
    if not np.allclose(metric, np.eye(metric.shape[0]), atol=2e-8):
        raise RuntimeError("cr2: cached MOs are not AO-metric orthonormal.")
    mean_field.mo_coeff = orbitals

    casci = mcscf.CASCI(mean_field, 12, (6, 6))
    casci.ncore = 18
    casci.fcisolver.wfnsym = "Ag"  # Cr2 ground state 1Sg+ -> Ag in D2h
    casci.fcisolver.conv_tol = 1e-12
    casci.fcisolver.max_cycle = 200
    try:
        casci.fcisolver.threads = 4
    except AttributeError:
        pass
    casci.kernel(orbitals)
    if not casci.converged:
        raise RuntimeError("cr2: CASCI did not converge.")
    # The cache itself was generated at fcisolver conv_tol=1e-9, so its RDMs
    # carry ~1e-4 wavefunction noise.  The reference state below is the
    # tighter (conv_tol=1e-12) regenerated CI: statevector, RDMs and energy
    # all come from one wavefunction, exactly like the other systems.  The
    # cache is still used to validate the state (energy to 1e-8, orbitals
    # verbatim) and the RDM-level difference is recorded.
    exact_energy = float(casci.e_tot)
    cache_energy_diff = abs(exact_energy - cache_energy)
    if cache_energy_diff > 1e-8:
        raise RuntimeError(
            f"cr2: regenerated CASCI energy {exact_energy} differs from the "
            f"cache {cache_energy} by {cache_energy_diff:.3e} Eh."
        )
    spin_square = float(casci.fcisolver.spin_square(casci.ci, 12, (6, 6))[0])
    if abs(spin_square) > 1e-7:
        raise RuntimeError(f"cr2: singlet check failed, S^2={spin_square}.")
    dm1s, dm2s = casci.fcisolver.make_rdm12s(casci.ci, 12, (6, 6))
    exact_gamma, exact_d2 = rdms_from_pyscf_spin_blocks(dm1s, dm2s, 12)
    cache_rdm_mismatch = max(
        float(np.linalg.norm(np.asarray(exact_gamma) - cache_gamma)),
        float(np.linalg.norm(np.asarray(exact_d2) - cache_d2)),
    )

    n_modes = 24
    alpha_strings = fci.cistring.make_strings(range(12), 6)
    beta_strings = fci.cistring.make_strings(range(12), 6)
    statevector = np.zeros(1 << n_modes, dtype=complex)
    for alpha_index, alpha_string in enumerate(alpha_strings):
        for beta_index, beta_string in enumerate(beta_strings):
            basis_state = int(alpha_string) | (int(beta_string) << 12)
            statevector[basis_state] = casci.ci[alpha_index, beta_index]
    statevector /= np.linalg.norm(statevector)
    statevector = _canonicalize_statevector_phase(statevector)

    pairs = pair_basis(n_modes)
    one_electron, core_energy = casci.get_h1eff(orbitals)
    two_electron = ao2mo.restore(1, casci.get_h2eff(orbitals), 12)
    active_hamiltonian = ElectronicEnergy.from_raw_integrals(
        one_electron, two_electron
    ).second_q_op()
    one_body, two_body = _hamiltonian_coefficients(active_hamiltonian, n_modes, pairs)

    orbital_labels = symm.label_orb_symm(
        molecule, molecule.irrep_name, molecule.symm_orb, orbitals, check=True,
    )
    orbital_irreps = tuple(
        int(symm.irrep_name2id(molecule.groupname, label))
        for label in orbital_labels[18:30]
    )

    identity_error = abs(
        rdm_energy(exact_d2, exact_gamma, one_body, two_body, float(core_energy))
        - exact_energy
    )
    reference = MolecularReference(
        statevector=statevector,
        exact_energy=exact_energy,
        nuclear_energy=float(core_energy),
        exact_gamma=exact_gamma,
        exact_d2=exact_d2,
        one_body=one_body,
        two_body=two_body,
        pairs=pairs,
        n_spatial_orbitals=12,
        n_spin_orbitals=n_modes,
        n_alpha=6,
        n_beta=6,
        orbital_irreps=orbital_irreps,
        molecule=molecule,
        mean_field=mean_field,
        active_mo_coeff=np.asarray(orbitals[:, 18:30]),
        n_core_orbitals=18,
    )
    aux = {
        "rhf_energy_eh": rhf_energy,
        "casci_regeneration_error_eh": cache_energy_diff,
        "ci_rdm_mismatch": cache_rdm_mismatch,
    }
    return reference, identity_error, aux


def build_reference(spec: dict[str, Any]) -> tuple[MolecularReference, float, dict]:
    if spec["builder"] == "c2_avas":
        reference = build_c2_reference(spec["bond_length"], basis=spec["basis"])
        identity_error = abs(
            rdm_energy(
                reference.exact_d2,
                reference.exact_gamma,
                reference.one_body,
                reference.two_body,
                reference.nuclear_energy,
            )
            - reference.exact_energy
        )
        aux: dict[str, Any] = {}
        try:
            aux["rhf_energy_eh"] = float(reference.mean_field.e_tot)
        except (AttributeError, TypeError):
            pass
        return reference, identity_error, aux
    if spec["builder"] == "cr2_cached":
        return build_cr2_cached_reference(spec)
    return build_canonical_cas_reference(spec)


# --------------------------------------------------------------------------
# Design: random frame pool + exact-covariance-whitened QR row selection
# --------------------------------------------------------------------------

def select_qr_basis(
    name: str,
    rotations: Sequence[np.ndarray],
    pair_vectors: np.ndarray,
    design: Any,
    blocks: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray],
    dimension: int,
) -> ConstraintBasis:
    """QR-pivot selection of `dimension` literal rows, whitened by the exact
    single-frame covariance diagonals (deterministic analogue of the pilot
    selection in run_n2_random_adaptive_allocation._pilot_constraint_basis)."""

    diagonal_variances = tuple(
        np.maximum(np.diag(covariance), 1e-12) for covariance in covariances
    )
    stacked = np.vstack(blocks)
    scales = np.sqrt(np.concatenate(diagonal_variances))
    weighted = stacked / scales[:, None]
    _, triangular, pivots = qr(weighted.T, mode="economic", pivoting=True)
    tolerance = abs(triangular[0, 0]) * max(weighted.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(np.abs(np.diag(triangular)) > tolerance))
    if rank < dimension:
        raise RuntimeError(
            f"{name}: frame pool rank {rank} below required {dimension}."
        )
    rows_per_frame = blocks[0].shape[0]
    chosen = np.asarray(pivots[:dimension], dtype=int)
    mapping = sorted(
        (int(index // rows_per_frame), int(index % rows_per_frame)) for index in chosen
    )
    global_rows = np.asarray(
        [frame * rows_per_frame + local for frame, local in mapping], dtype=int
    )
    rows_by_frame = tuple(
        np.asarray(
            [local for frame, local in mapping if frame == index], dtype=int
        )
        for index in range(len(rotations))
    )
    selected_blocks = tuple(
        np.asarray(blocks[index])[rows_by_frame[index]]
        for index in range(len(rotations))
    )
    selected_covariances = tuple(
        covariances[index][np.ix_(rows_by_frame[index], rows_by_frame[index])]
        if len(rows_by_frame[index])
        else np.empty((0, 0))
        for index in range(len(rotations))
    )
    singular = np.linalg.svd(stacked[global_rows], compute_uv=False)
    return ConstraintBasis(
        name=name,
        rotations=tuple(np.asarray(rotation) for rotation in rotations),
        pair_vectors=np.asarray(pair_vectors),
        design=design,
        global_rows=global_rows,
        rows_by_frame=rows_by_frame,
        blocks=selected_blocks,
        covariances=selected_covariances,
        raw_condition=float(singular[0] / singular[-1]),
        fisher_condition=np.nan,
        rank=dimension,
    )


def whitened_rank(blocks, covariances) -> int:
    """Rank of the covariance-whitened stacked literal design rows."""

    diagonal_variances = tuple(
        np.maximum(np.diag(covariance), 1e-12) for covariance in covariances
    )
    stacked = np.vstack(blocks)
    scales = np.sqrt(np.concatenate(diagonal_variances))
    weighted = stacked / scales[:, None]
    _, triangular, _ = qr(weighted.T, mode="economic", pivoting=True)
    tolerance = abs(triangular[0, 0]) * max(weighted.shape) * np.finfo(float).eps
    return int(np.count_nonzero(np.abs(np.diag(triangular)) > tolerance))


def frame_pair_vectors(rotation, n_spatial: int, pairs) -> np.ndarray:
    """Pair-rotation columns for one frame; (2,n,n) frames use independent
    alpha/beta blocks (the C2 spin-asymmetric frame family)."""

    array = np.asarray(rotation)
    if array.ndim == 3:
        alpha, beta = array
        zero = np.zeros((n_spatial, n_spatial), dtype=complex)
        spin_rotation = np.block([[alpha, zero], [zero, beta]])
    else:
        spin_rotation = spin_orbital_rotation(array)
    return exterior_square(spin_rotation, pairs).T


def build_design(reference, rotations):
    """Pair vectors, quadratic CSR design, and per-frame variable blocks.

    Equivalent to _build_random_design for same-spin frames; extended to
    stacked (alpha, beta) frames.
    """

    rows, cols = _symmetric_d2_variables(reference)
    vectors = []
    blocks = []
    for rotation in rotations:
        frame_vectors = frame_pair_vectors(
            rotation, reference.n_spatial_orbitals, reference.pairs
        )
        vectors.append(frame_vectors)
        blocks.append(_frame_block(frame_vectors, rows, cols))
    pair_vectors = np.vstack(vectors)
    return pair_vectors, quadratic_design(pair_vectors), tuple(blocks), rows, cols


def prepare_design(reference, spec):
    """Random frame pool, extended in deterministic chunks until the whitened
    literal rows span the symmetry-reduced variable space, plus the QR basis."""

    name = spec["name"]
    dimension = int(len(_symmetric_d2_variables(reference)[0]))
    rotations = list(
        _random_mixed_frames(
            reference.n_spatial_orbitals,
            int(spec["frame_count"]),
            COMPLEX_FRACTION,
            REAL_FRAME_SEED,
            COMPLEX_FRAME_SEED,
            PLACEMENT_SEED,
        )[0]
    )
    extension = 0
    while True:
        pair_vectors, design, blocks, var_rows, var_cols = build_design(
            reference, rotations
        )
        oracle = AcquisitionOracle(reference, tuple(rotations), pair_vectors, 1,
                                   SHOT_SEED)
        exact_covariances = tuple(
            _exact_single_covariance(
                oracle._frame_probabilities(index), oracle.indicators.astype(float)
            )[1]
            for index in range(len(rotations))
        )
        rank = whitened_rank(blocks, exact_covariances)
        if rank >= dimension:
            break
        if extension >= 12:
            raise RuntimeError(
                f"{name}: rank {rank} < {dimension} even after "
                f"{extension} spin-asymmetric frame extensions."
            )
        extension += 1
        # Same-spin diagonal frames have a structural null space in the
        # symmetry-reduced variable space for some irrep patterns (Li2, C2).
        # Independent alpha/beta Haar frames cover it -- the published C2
        # protocol ships one such frame per pool for this reason.
        alpha = _random_unitaries(reference.n_spatial_orbitals, 1,
                                  77003 + 1009 * extension)[0]
        beta = _random_unitaries(reference.n_spatial_orbitals, 1,
                                 77005 + 1009 * extension)[0]
        rotations.append(np.stack((alpha, beta)))
        print(
            f"[{name}] pool rank {rank} < {dimension}; appended "
            f"spin-asymmetric frame #{extension}",
            flush=True,
        )
    basis = select_qr_basis(
        f"{name}_exact_covariance_qr",
        rotations,
        pair_vectors,
        design,
        blocks,
        exact_covariances,
        dimension,
    )
    return {
        "rotations": tuple(rotations),
        "pair_vectors": pair_vectors,
        "design": design,
        "blocks": blocks,
        "var_rows": var_rows,
        "var_cols": var_cols,
        "oracle": oracle,
        "exact_covariances": exact_covariances,
        "basis": basis,
        "pool_extensions": extension,
    }


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def solve_qr_nuclear(reference, shadows: ShadowData, warm) -> Any:
    """Paper Eq. (11): min energy + w*||corrected - d2||_* s.t. DQG + rows."""

    kwargs = dict(SOLVER_KWARGS)
    if _NUCLEAR_KWARGS_OVERRIDE:
        kwargs.update(_NUCLEAR_KWARGS_OVERRIDE)
    return solve_dqg_sdp(
        LeakGuardReference(reference),
        shadow_data=shadows,
        shadow_error_weight=NUCLEAR_WEIGHT,
        positivity_conditions="DQG",
        symmetry_blocked_psd=True,
        selection_objective="energy",
        initial_d2=warm.d2,
        initial_gamma=warm.gamma,
        initial_corrected_d2=warm.d2,
        **kwargs,
    )


def solve_qr_gls(
    reference,
    basis: ConstraintBasis,
    outcomes: Sequence[np.ndarray],
    counts: np.ndarray,
    blocks: Sequence[np.ndarray],
    rows: np.ndarray,
    cols: np.ndarray,
    covariances: Sequence[np.ndarray],
    warm,
) -> Any:
    """Full-covariance GLS over DQG on the QR-selected rows (run_evidence
    gls_factor/fit_gls, qr_gls variant: selected literal rows only)."""

    n_pairs = len(reference.pairs)
    pieces = []
    for pool in range(len(basis.rotations)):
        count = int(counts[pool])
        selected = basis.rows_by_frame[pool]
        if count == 0 or not len(selected):
            continue
        mean = np.mean(outcomes[pool][:count], axis=0)[selected]
        covariance = covariances[pool][np.ix_(selected, selected)]
        chol = cholesky(covariance, lower=True)
        augmented = np.column_stack((blocks[pool][selected], -mean))
        pieces.append(np.sqrt(count) * solve_triangular(chol, augmented, lower=True))
    stacked = np.vstack(pieces)
    normalization = np.sqrt(float(np.sum(counts)))
    factor = qr(stacked / normalization, mode="r")[0][: min(stacked.shape)]
    n_rows, n_cols = factor[:, :-1].shape
    indices = rows * n_pairs + cols
    mapping = sparse.coo_matrix(
        (
            factor[:, :-1].ravel(),
            (np.repeat(np.arange(n_rows), n_cols), np.tile(indices, n_rows)),
        ),
        shape=(n_rows, n_pairs * n_pairs),
    ).tocsr()
    objective = AffineRDMObjective(
        d2_map=mapping,
        offset=np.asarray(factor[:, -1], dtype=float),
        name="full_covariance_gls",
    )
    return solve_dqg_sdp(
        LeakGuardReference(reference),
        affine_objective=objective,
        selection_objective="affine_least_squares",
        positivity_conditions="DQG",
        symmetry_blocked_psd=True,
        initial_d2=warm.d2,
        initial_gamma=warm.gamma,
        **SOLVER_KWARGS,
    )


# --------------------------------------------------------------------------
# Error decomposition and cancellation ratios
# --------------------------------------------------------------------------

def cancellation_ratio(part_a: float, part_b: float, total: float) -> float:
    denominator = abs(part_a) + abs(part_b)
    if denominator <= 1e-12:
        return float("nan")
    return float(abs(total) / denominator)


def decompose(reference, objective, d2_hat, gamma_hat, exact_gradients) -> dict[str, float]:
    """Split the H (exact) and ftPBE (direct + first-order) energy errors."""

    d2_star = np.asarray(reference.exact_d2)
    gamma_star = np.asarray(reference.exact_gamma)
    h_star, f_star = _energy_values(reference, objective, d2_star, gamma_star)
    h_d2, f_d2 = _energy_values(reference, objective, d2_hat, gamma_star)
    h_g, f_g = _energy_values(reference, objective, d2_star, gamma_hat)
    h_total, f_total = _energy_values(reference, objective, d2_hat, gamma_hat)
    h_d2 = 1000.0 * (h_d2 - h_star)
    h_g = 1000.0 * (h_g - h_star)
    h_total = 1000.0 * (h_total - h_star)
    f_d2 = 1000.0 * (f_d2 - f_star)
    f_g = 1000.0 * (f_g - f_star)
    f_total = 1000.0 * (f_total - f_star)

    dd = np.asarray(d2_hat) - d2_star
    dg = np.asarray(gamma_hat) - gamma_star
    g_d2, g_gamma = exact_gradients
    f_d2_lin = 1000.0 * float(np.sum(g_d2 * dd))
    f_g_lin = 1000.0 * float(np.sum(g_gamma * dg))
    f_total_lin = f_d2_lin + f_g_lin

    contracted = contract_one_rdm(
        np.asarray(d2_hat), reference.n_spin_orbitals, reference.n_electrons,
        reference.pairs,
    )
    slack = float(
        np.linalg.norm(np.asarray(gamma_hat) - contracted)
        / max(np.linalg.norm(np.asarray(gamma_hat)), 1e-14)
    )
    dd_norm = float(np.linalg.norm(dd))
    degenerate = dd_norm < 1e-6  # at/below solver accuracy: CR is 0/0 noise
    cr_h = cancellation_ratio(h_d2, h_g, h_total)
    cr_f = cancellation_ratio(f_d2, f_g, f_total)
    cr_f_lin = cancellation_ratio(f_d2_lin, f_g_lin, f_total_lin)
    if degenerate:
        cr_h = cr_f = cr_f_lin = float("nan")
    return {
        "h_d2_meh": h_d2,
        "h_gamma_meh": h_g,
        "h_total_meh": h_total,
        "h_decomposition_identity_meh": abs(h_d2 + h_g - h_total),
        "cr_h": cr_h,
        "f_d2_meh": f_d2,
        "f_gamma_meh": f_g,
        "f_total_meh": f_total,
        "f_nonlinear_residual_meh": f_total - (f_d2 + f_g),
        "cr_f": cr_f,
        "f_d2_lin_meh": f_d2_lin,
        "f_gamma_lin_meh": f_g_lin,
        "f_total_lin_meh": f_total_lin,
        "cr_f_linearized": cr_f_lin,
        "dd_norm": dd_norm,
        "dd_degenerate": bool(degenerate),
        "gamma_contraction_slack": slack,
    }


# --------------------------------------------------------------------------
# Probe C: random directions in the unmeasured complement
# --------------------------------------------------------------------------

def probe_random_directions(
    reference,
    objective,
    basis: ConstraintBasis,
    scale: float,
    n_directions: int,
    seed: int,
    exact_gradients=None,
) -> list[dict[str, float]]:
    n_pairs = len(reference.pairs)
    design_rows = np.asarray(basis.design[basis.global_rows].todense())
    q_span, _ = qr(design_rows.T, mode="economic")
    d2_star = np.asarray(reference.exact_d2)
    gamma_star = np.asarray(reference.exact_gamma)
    _, f_star = _energy_values(reference, objective, d2_star, gamma_star)
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n_directions):
        gaussian = rng.normal(size=(n_pairs, n_pairs))
        symmetric = gaussian + gaussian.T
        vector = symmetric.ravel(order="C")
        vector = vector - q_span @ (q_span.T @ vector)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            continue
        vector *= scale / norm
        dd = vector.reshape(n_pairs, n_pairs)
        dd = 0.5 * (dd + dd.T)
        dg = contract_one_rdm(
            dd, reference.n_spin_orbitals, reference.n_electrons, reference.pairs
        )
        h_d2 = 1000.0 * float(np.sum(reference.two_body * dd))
        h_g = 1000.0 * float(np.sum(reference.one_body * dg))
        h_total = h_d2 + h_g
        f_d2 = 1000.0 * (
            _energy_values(reference, objective, d2_star + dd, gamma_star)[1] - f_star
        )
        f_g = 1000.0 * (
            _energy_values(reference, objective, d2_star, gamma_star + dg)[1] - f_star
        )
        f_total = 1000.0 * (
            _energy_values(reference, objective, d2_star + dd, gamma_star + dg)[1]
            - f_star
        )
        row = {
            "direction_index": index,
            "dd_norm": float(np.linalg.norm(dd)),
            "h_d2_meh": h_d2,
            "h_gamma_meh": h_g,
            "h_total_meh": h_total,
            "cr_h": cancellation_ratio(h_d2, h_g, h_total),
            "f_d2_meh": f_d2,
            "f_gamma_meh": f_g,
            "f_total_meh": f_total,
            "f_nonlinear_residual_meh": f_total - (f_d2 + f_g),
            "cr_f": cancellation_ratio(f_d2, f_g, f_total),
        }
        if exact_gradients is not None:
            g_d2, g_gamma = exact_gradients
            f_d2_lin = 1000.0 * float(np.sum(g_d2 * dd))
            f_g_lin = 1000.0 * float(np.sum(g_gamma * dg))
            row.update(
                {
                    "f_d2_lin_meh": f_d2_lin,
                    "f_gamma_lin_meh": f_g_lin,
                    "f_total_lin_meh": f_d2_lin + f_g_lin,
                    "cr_f_linearized": cancellation_ratio(
                        f_d2_lin, f_g_lin, f_d2_lin + f_g_lin
                    ),
                }
            )
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Per-system pipeline
# --------------------------------------------------------------------------

_DECOMPOSITION_KEYS = (
    "h_d2_meh", "h_gamma_meh", "h_total_meh", "h_decomposition_identity_meh",
    "cr_h", "f_d2_meh", "f_gamma_meh", "f_total_meh",
    "f_nonlinear_residual_meh", "cr_f", "f_d2_lin_meh", "f_gamma_lin_meh",
    "f_total_lin_meh", "cr_f_linearized", "dd_norm", "gamma_contraction_slack",
)


def _failed_decomposition() -> dict[str, Any]:
    """NaN decomposition for a probe whose SDP could not be solved (the
    failure itself is a result; probe C must still run)."""

    row = {key: float("nan") for key in _DECOMPOSITION_KEYS}
    row["dd_degenerate"] = False
    return row


def run_system(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    name = spec["name"]
    started_all = time.perf_counter()
    print(f"[{name}] building CASCI reference ({spec['basis']}, "
          f"CAS({spec['active_electrons']},{spec['active_orbitals']}), "
          f"R={spec['bond_length']} A)", flush=True)
    reference, energy_identity_error, reference_aux = build_reference(spec)
    objective = FtPBEEnergyObjective(reference, grid_level=GRID_LEVEL)
    evaluation = objective.evaluate(
        reference.exact_d2, reference.exact_gamma, gradient=True
    )
    exact_gradients = (
        np.asarray(evaluation.d2_gradient),
        np.asarray(evaluation.gamma_gradient),
    )
    h_star, f_star = _energy_values(
        reference, objective, reference.exact_d2, reference.exact_gamma
    )
    energy_column_error = abs(h_star - reference.exact_energy)

    n_spatial = int(reference.n_spatial_orbitals)
    gamma_exact = np.asarray(reference.exact_gamma)
    spatial_gamma = gamma_exact[:n_spatial, :n_spatial] + gamma_exact[n_spatial:, n_spatial:]
    noons = np.linalg.eigvalsh(0.5 * (spatial_gamma + spatial_gamma.T))
    noon_max_deviation = float(np.max(np.minimum(2.0 - noons, noons)))
    rhf_energy = reference_aux.get("rhf_energy_eh")
    correlation_diagnostics = {
        "rhf_energy_eh": rhf_energy,
        "casci_energy_eh": float(reference.exact_energy),
        "rhf_minus_casci_eh": (
            float(rhf_energy - reference.exact_energy)
            if rhf_energy is not None
            else None
        ),
        "noon_max_deviation": noon_max_deviation,
        "noons": [float(value) for value in noons],
        "f_star_minus_h_star_eh": float(f_star - h_star),
    }
    for key in ("casci_regeneration_error_eh", "ci_rdm_mismatch"):
        if key in reference_aux:
            correlation_diagnostics[key] = reference_aux[key]

    prepared = prepare_design(reference, spec)
    rotations = prepared["rotations"]
    pair_vectors = prepared["pair_vectors"]
    design = prepared["design"]
    blocks = prepared["blocks"]
    var_rows = prepared["var_rows"]
    var_cols = prepared["var_cols"]
    dimension = int(len(var_rows))
    oracle = prepared["oracle"]
    exact_covariances = prepared["exact_covariances"]
    basis = prepared["basis"]
    active_frames = basis.active_frames
    print(
        f"[{name}] d={dimension}, n_pairs={len(reference.pairs)}, "
        f"frames={len(rotations)} (active {len(active_frames)}), "
        f"raw condition={basis.raw_condition:.3e}",
        flush=True,
    )

    print(f"[{name}] shadow-free DQG baseline solve", flush=True)
    started = time.perf_counter()
    baseline = solve_dqg_sdp(
        LeakGuardReference(reference),
        positivity_conditions="DQG",
        symmetry_blocked_psd=True,
        **SOLVER_KWARGS,
    )
    baseline_seconds = time.perf_counter() - started
    if baseline.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"[{name}] baseline DQG status {baseline.status}.")

    result_rows: list[dict[str, Any]] = []
    artifacts: dict[str, np.ndarray] = {}
    common = {
        "system": name,
        "basis": spec["basis"],
        "bond_length_angstrom": spec["bond_length"],
        "cas": f"({spec['active_electrons']},{spec['active_orbitals']})",
        "ncore": int(spec["ncore"]),
        "n_spatial_orbitals": int(reference.n_spatial_orbitals),
        "n_pairs": int(len(reference.pairs)),
        "n_variable_rows": dimension,
        "n_frames": int(len(rotations)),
        "n_active_frames": int(len(active_frames)),
        "frame_pool_extensions": int(prepared["pool_extensions"]),
    }

    # ---- probe A: exact-mean structural probe (qr_nuclear only) ----------
    print(f"[{name}] probe A: exact-mean Eq. (11) solve", flush=True)
    started = time.perf_counter()
    exact_shadows = _exact_mean_shadows(basis, reference.exact_d2)
    decomposition_a: dict[str, Any] | None = None
    try:
        probe_a = solve_qr_nuclear(reference, exact_shadows, baseline)
        probe_a_seconds = time.perf_counter() - started
        decomposition_a = decompose(
            reference, objective, probe_a.d2, probe_a.gamma, exact_gradients
        )
        artifacts["probe_a_d2"] = np.asarray(probe_a.d2)
        artifacts["probe_a_gamma"] = np.asarray(probe_a.gamma)
        result_rows.append(
            {
                **common,
                "probe": "A",
                "estimator": "qr_nuclear",
                "solver_status": str(probe_a.status),
                "fit_status": str(probe_a.fit_status),
                "seconds": probe_a_seconds,
                "total_shots": 0,
                "shot_seed": "",
                "shadow_error_trace": float(probe_a.shadow_error_trace or np.nan),
                **decomposition_a,
            }
        )
        print(
            f"[{name}] probe A qr_nuclear: CR_H={decomposition_a['cr_h']:.3f} "
            f"CR_F={decomposition_a['cr_f']:.3f} |dD|={decomposition_a['dd_norm']:.4f} "
            f"({probe_a_seconds:.1f} s, {probe_a.status})",
            flush=True,
        )
    except Exception as error:  # solver failure is data; keep probing
        probe_a_seconds = time.perf_counter() - started
        result_rows.append(
            {
                **common,
                "probe": "A",
                "estimator": "qr_nuclear",
                "solver_status": "solver_error",
                "fit_status": "",
                "seconds": probe_a_seconds,
                "total_shots": 0,
                "shot_seed": "",
                "shadow_error_trace": float("nan"),
                "error": f"{type(error).__name__}: {error}",
                **_failed_decomposition(),
            }
        )
        print(
            f"[{name}] probe A qr_nuclear FAILED after {probe_a_seconds:.1f} s: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

    # ---- probe B: one noisy stream, qr_nuclear + qr_gls ------------------
    counts = np.zeros(len(rotations), dtype=int)
    per_frame = args.shots // len(active_frames)
    remainder = args.shots - per_frame * len(active_frames)
    for position, frame in enumerate(active_frames):
        counts[frame] = per_frame + (1 if position < remainder else 0)
    outcomes = _sample_outcomes(oracle, SHOT_SEED + 200_003, counts)
    noisy_shadows = _finite_shadows(basis, outcomes, counts)

    print(f"[{name}] probe B: {args.shots} shots over {len(active_frames)} "
          f"frames; qr_nuclear solve", flush=True)
    started = time.perf_counter()
    try:
        probe_b_nuclear = solve_qr_nuclear(reference, noisy_shadows, baseline)
        seconds_nuclear = time.perf_counter() - started
        decomposition_b_nuclear = decompose(
            reference, objective, probe_b_nuclear.d2, probe_b_nuclear.gamma,
            exact_gradients,
        )
        artifacts["probe_b_nuclear_d2"] = np.asarray(probe_b_nuclear.d2)
        artifacts["probe_b_nuclear_gamma"] = np.asarray(probe_b_nuclear.gamma)
        result_rows.append(
            {
                **common,
                "probe": "B",
                "estimator": "qr_nuclear",
                "solver_status": str(probe_b_nuclear.status),
                "fit_status": str(probe_b_nuclear.fit_status),
                "seconds": seconds_nuclear,
                "total_shots": int(args.shots),
                "shot_seed": SHOT_SEED,
                "shadow_error_trace": float(
                    probe_b_nuclear.shadow_error_trace or np.nan
                ),
                **decomposition_b_nuclear,
            }
        )
        print(
            f"[{name}] probe B qr_nuclear: CR_H={decomposition_b_nuclear['cr_h']:.3f} "
            f"CR_F={decomposition_b_nuclear['cr_f']:.3f} "
            f"|dD|={decomposition_b_nuclear['dd_norm']:.4f} "
            f"({seconds_nuclear:.1f} s, {probe_b_nuclear.status})",
            flush=True,
        )
    except Exception as error:  # solver failure is data; keep probing
        seconds_nuclear = time.perf_counter() - started
        result_rows.append(
            {
                **common,
                "probe": "B",
                "estimator": "qr_nuclear",
                "solver_status": "solver_error",
                "fit_status": "",
                "seconds": seconds_nuclear,
                "total_shots": int(args.shots),
                "shot_seed": SHOT_SEED,
                "shadow_error_trace": float("nan"),
                "error": f"{type(error).__name__}: {error}",
                **_failed_decomposition(),
            }
        )
        print(
            f"[{name}] probe B qr_nuclear FAILED after {seconds_nuclear:.1f} s: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

    print(f"[{name}] probe B: qr_gls solve", flush=True)
    started = time.perf_counter()
    try:
        probe_b_gls = solve_qr_gls(
            reference, basis, outcomes, counts, blocks, var_rows, var_cols,
            exact_covariances, baseline,
        )
        seconds_gls = time.perf_counter() - started
        decomposition_b_gls = decompose(
            reference, objective, probe_b_gls.d2, probe_b_gls.gamma, exact_gradients
        )
        artifacts["probe_b_gls_d2"] = np.asarray(probe_b_gls.d2)
        artifacts["probe_b_gls_gamma"] = np.asarray(probe_b_gls.gamma)
        result_rows.append(
            {
                **common,
                "probe": "B",
                "estimator": "qr_gls",
                "solver_status": str(probe_b_gls.status),
                "fit_status": str(probe_b_gls.fit_status),
                "seconds": seconds_gls,
                "total_shots": int(args.shots),
                "shot_seed": SHOT_SEED,
                "shadow_error_trace": float("nan"),
                **decomposition_b_gls,
            }
        )
        print(
            f"[{name}] probe B qr_gls: CR_H={decomposition_b_gls['cr_h']:.3f} "
            f"CR_F={decomposition_b_gls['cr_f']:.3f} "
            f"|dD|={decomposition_b_gls['dd_norm']:.4f} "
            f"({seconds_gls:.1f} s, {probe_b_gls.status})",
            flush=True,
        )
    except Exception as error:  # solver failure is data; keep probing
        seconds_gls = time.perf_counter() - started
        result_rows.append(
            {
                **common,
                "probe": "B",
                "estimator": "qr_gls",
                "solver_status": "solver_error",
                "fit_status": "",
                "seconds": seconds_gls,
                "total_shots": int(args.shots),
                "shot_seed": SHOT_SEED,
                "shadow_error_trace": float("nan"),
                "error": f"{type(error).__name__}: {error}",
                **_failed_decomposition(),
            }
        )
        print(
            f"[{name}] probe B qr_gls FAILED after {seconds_gls:.1f} s: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )

    # ---- probe C: random-direction baseline ------------------------------
    scale = decomposition_a["dd_norm"] if decomposition_a is not None else float("nan")
    scale_note = "probe_a_dd_norm"
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
        scale_note = "unit_norm_fallback"
    print(
        f"[{name}] probe C: {args.n_random} random complement directions "
        f"(scale={scale:.4f}, {scale_note})",
        flush=True,
    )
    started = time.perf_counter()
    random_rows = probe_random_directions(
        reference, objective, basis, scale, args.n_random, RANDOM_DIRECTION_SEED,
        exact_gradients=exact_gradients,
    )
    seconds_random = time.perf_counter() - started
    for row in random_rows:
        row.update(
            {
                "system": name,
                "basis": spec["basis"],
                "bond_length_angstrom": spec["bond_length"],
                "cas": common["cas"],
                "scale": scale,
                "scale_note": scale_note,
            }
        )
    cr_h_values = np.asarray([row["cr_h"] for row in random_rows], dtype=float)
    cr_f_values = np.asarray([row["cr_f"] for row in random_rows], dtype=float)
    random_summary = {
        "cr_h_median": float(np.nanmedian(cr_h_values)),
        "cr_h_q25": float(np.nanpercentile(cr_h_values, 25)),
        "cr_h_q75": float(np.nanpercentile(cr_h_values, 75)),
        "cr_f_median": float(np.nanmedian(cr_f_values)),
        "cr_f_q25": float(np.nanpercentile(cr_f_values, 25)),
        "cr_f_q75": float(np.nanpercentile(cr_f_values, 75)),
        "n_directions": int(len(random_rows)),
        "seconds": seconds_random,
    }
    print(
        f"[{name}] probe C: CR_H median={random_summary['cr_h_median']:.3f} "
        f"IQR=[{random_summary['cr_h_q25']:.3f}, {random_summary['cr_h_q75']:.3f}]; "
        f"CR_F median={random_summary['cr_f_median']:.3f} "
        f"({seconds_random:.1f} s)",
        flush=True,
    )

    # ---- validation extras ------------------------------------------------
    gls_exact = None
    if spec.get("gate"):
        print(f"[{name}] validation: qr_gls on exact means must recover D*",
              flush=True)
        gls_exact_result = solve_qr_gls(
            reference, basis,
            _exact_indicator_outcomes(oracle, len(rotations)),
            np.ones(len(rotations), dtype=int),
            blocks, var_rows, var_cols, exact_covariances, baseline,
        )
        gls_exact_dd = float(
            np.linalg.norm(np.asarray(gls_exact_result.d2) - reference.exact_d2)
        )
        print(f"[{name}] qr_gls exact-mean |dD|={gls_exact_dd:.3e}", flush=True)
        gls_exact = {"dd_norm": gls_exact_dd,
                     "status": str(gls_exact_result.status)}

    total_seconds = time.perf_counter() - started_all
    summary = {
        "system": name,
        "spec": {key: spec[key] for key in ("basis", "bond_length", "ncore",
                 "active_electrons", "active_orbitals", "frame_count", "builder")},
        "exact_energy": float(reference.exact_energy),
        "h_star": float(h_star),
        "f_star": float(f_star),
        "energy_identity_error_eh": float(energy_identity_error),
        "energy_column_error_eh": float(energy_column_error),
        "baseline_status": str(baseline.status),
        "baseline_seconds": baseline_seconds,
        "baseline_d2_gap": float(
            np.linalg.norm(np.asarray(baseline.d2) - reference.exact_d2)
        ),
        "raw_condition": float(basis.raw_condition),
        "frame_pool_extensions": int(prepared["pool_extensions"]),
        "active_frames": [int(index) for index in active_frames],
        "shot_counts_probe_b": counts.tolist(),
        "random_summary": random_summary,
        "correlation_diagnostics": correlation_diagnostics,
        "gls_exact_mean_check": gls_exact,
        "peak_rss_gb": _peak_rss_gb(),
        "total_seconds": total_seconds,
    }
    return {
        "rows": result_rows,
        "random_rows": random_rows,
        "summary": summary,
        "artifacts": artifacts,
    }


def _exact_indicator_outcomes(oracle, n_frames: int) -> tuple[np.ndarray, ...]:
    """One-shot 'outcomes' whose empirical mean equals the exact frame mean."""

    outcomes = []
    for index in range(n_frames):
        probabilities = oracle._frame_probabilities(index)
        mean_indicators = probabilities @ oracle.indicators.astype(float)
        # The GLS factor only reads np.mean(outcomes[:count], axis=0), so a
        # single pseudo-outcome row equal to the exact mean is lossless.
        outcomes.append(np.asarray(mean_indicators[None, :], dtype=float))
    return tuple(outcomes)


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)

    def default(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=default, allow_nan=True)
    os.replace(temporary, path)


def _save_npz(path: Path, artifacts: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **artifacts)
    os.replace(temporary, path)


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------

def evaluate_gates(
    summaries: dict[str, dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    energy = {
        "tolerance_eh": GATE_ENERGY_IDENTITY_EH,
        "per_system": {
            name: summary["energy_identity_error_eh"]
            for name, summary in summaries.items()
        },
    }
    energy["max_error_eh"] = max(energy["per_system"].values(), default=0.0)
    energy["pass"] = bool(energy["max_error_eh"] < GATE_ENERGY_IDENTITY_EH)

    identity_rows = [row for row in rows if row.get("probe") in ("A", "B")]
    identity_values = [
        float(row["h_decomposition_identity_meh"])
        for row in identity_rows
        if np.isfinite(float(row.get("h_decomposition_identity_meh", float("nan"))))
    ]
    decomposition = {
        "tolerance_meh": GATE_DECOMPOSITION_MEH,
        "max_error_meh": max(identity_values) if identity_values else None,
        "n_records": len(identity_values),
        "n_solver_error_records": sum(
            1 for row in identity_rows if row.get("solver_status") == "solver_error"
        ),
    }
    decomposition["pass"] = bool(
        identity_values and max(identity_values) < GATE_DECOMPOSITION_MEH
    )

    gate_rows = [
        row
        for row in rows
        if row.get("system") == GATE_SYSTEM and row.get("probe") == "B"
    ]
    n2_gate: dict[str, Any] = {"system": GATE_SYSTEM}
    nuclear = next(
        (row for row in gate_rows if row.get("estimator") == "qr_nuclear"), None
    )
    gls = next((row for row in gate_rows if row.get("estimator") == "qr_gls"), None)
    if nuclear and gls:
        n2_gate.update(
            {
                "cr_h_qr_nuclear": nuclear["cr_h"],
                "cr_h_qr_gls": gls["cr_h"],
                "cr_f_qr_nuclear": nuclear["cr_f"],
                "cr_f_qr_gls": gls["cr_f"],
            }
        )
        n2_gate["pass"] = bool(
            nuclear["cr_h"] < GATE_N2_CR_H_NUCLEAR
            and gls["cr_h"] > nuclear["cr_h"]
        )
    else:
        n2_gate["pass"] = False
        n2_gate["reason"] = "gate-system probe-B records missing"

    gates = {
        "energy_identity": energy,
        "decomposition_identity": decomposition,
        "n2_probe_b_cancellation": n2_gate,
    }
    gates["all_pass"] = bool(
        energy["pass"] and decomposition["pass"] and n2_gate["pass"]
    )
    return gates


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--systems", type=str, default="")
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-abort", action="store_true")
    parser.add_argument(
        "--probe-c-only",
        action="store_true",
        help="Recompute only probe C from checkpoints (no SDP solves).",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="",
        help="Override the SDP solver (fallback ladder for oversized systems).",
    )
    parser.add_argument(
        "--solver-tolerance",
        type=float,
        default=None,
        help="Override the SDP solver tolerance.",
    )
    parser.add_argument(
        "--solve-form",
        type=str,
        default="",
        help="Override MSK_IPAR_INTPNT_SOLVE_FORM for MOSEK (e.g. "
        "MSK_SOLVE_PRIMAL when the dualized form runs out of memory).",
    )
    parser.add_argument(
        "--nuclear-solver",
        type=str,
        default="",
        help="Override the solver for qr_nuclear solves only (e.g. CLARABEL/SCS "
        "when MOSEK exceeds the memory envelope).",
    )
    parser.add_argument(
        "--nuclear-max-iterations",
        type=int,
        default=None,
        help="Iteration cap paired with --nuclear-solver.",
    )
    parser.add_argument(
        "--nuclear-tolerance",
        type=float,
        default=None,
        help="Tolerance paired with --nuclear-solver (SCS eps).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _arguments(argv)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    systems_dir = args.output_dir / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)

    if args.solver:
        SOLVER_KWARGS["solver"] = args.solver
    if args.solver_tolerance is not None:
        SOLVER_KWARGS["tolerance"] = args.solver_tolerance
    if args.nuclear_solver:
        global _NUCLEAR_KWARGS_OVERRIDE
        _NUCLEAR_KWARGS_OVERRIDE = {"solver": args.nuclear_solver}
        if args.nuclear_max_iterations is not None:
            _NUCLEAR_KWARGS_OVERRIDE["max_iterations"] = args.nuclear_max_iterations
        if args.nuclear_tolerance is not None:
            _NUCLEAR_KWARGS_OVERRIDE["tolerance"] = args.nuclear_tolerance
    if args.solve_form:
        import constrained_shadow

        original_options = constrained_shadow._solver_options

        def patched_options(solver, tolerance, max_iterations, verbose,
                            solver_threads=None):
            options = original_options(
                solver, tolerance, max_iterations, verbose, solver_threads
            )
            if str(solver).upper() == "MOSEK":
                options["mosek_params"]["MSK_IPAR_INTPNT_SOLVE_FORM"] = (
                    args.solve_form
                )
            return options

        constrained_shadow._solver_options = patched_options

    selected = [spec for spec in SYSTEMS if args.include_optional or not spec.get("optional")]
    if args.systems:
        wanted = {item.strip() for item in args.systems.split(",") if item.strip()}
        selected = [spec for spec in selected if spec["name"] in wanted]
    if not selected:
        raise SystemExit("No systems selected.")

    all_rows: list[dict[str, Any]] = []
    all_random: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}

    if args.probe_c_only:
        for spec in selected:
            name = spec["name"]
            checkpoint = systems_dir / f"{name}.json"
            if not checkpoint.is_file():
                print(f"[{name}] no checkpoint; skipping probe-C rerun", flush=True)
                continue
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            scale = next(
                (
                    float(row["dd_norm"])
                    for row in payload["rows"]
                    if row.get("probe") == "A"
                ),
                1.0,
            )
            scale_note = "probe_a_dd_norm"
            if not np.isfinite(scale) or scale <= 1e-6:
                scale, scale_note = 1.0, "unit_norm_fallback"
            print(f"[{name}] probe-C-only rerun (scale={scale:.4g})", flush=True)
            reference, _, _aux = build_reference(spec)
            objective = FtPBEEnergyObjective(reference, grid_level=GRID_LEVEL)
            evaluation = objective.evaluate(
                reference.exact_d2, reference.exact_gamma, gradient=True
            )
            exact_gradients = (
                np.asarray(evaluation.d2_gradient),
                np.asarray(evaluation.gamma_gradient),
            )
            prepared = prepare_design(reference, spec)
            random_rows = probe_random_directions(
                reference, objective, prepared["basis"], scale, args.n_random,
                RANDOM_DIRECTION_SEED, exact_gradients=exact_gradients,
            )
            for row in random_rows:
                row.update(
                    {
                        "system": name,
                        "basis": spec["basis"],
                        "bond_length_angstrom": spec["bond_length"],
                        "cas": f"({spec['active_electrons']},{spec['active_orbitals']})",
                        "scale": scale,
                        "scale_note": scale_note,
                    }
                )
            cr_h_values = np.asarray([row["cr_h"] for row in random_rows])
            cr_f_values = np.asarray([row["cr_f"] for row in random_rows])
            cr_f_lin = np.asarray([row["cr_f_linearized"] for row in random_rows])
            payload["random_rows"] = random_rows
            payload["summary"]["random_summary"] = {
                "cr_h_median": float(np.nanmedian(cr_h_values)),
                "cr_h_q25": float(np.nanpercentile(cr_h_values, 25)),
                "cr_h_q75": float(np.nanpercentile(cr_h_values, 75)),
                "cr_f_median": float(np.nanmedian(cr_f_values)),
                "cr_f_q25": float(np.nanpercentile(cr_f_values, 25)),
                "cr_f_q75": float(np.nanpercentile(cr_f_values, 75)),
                "cr_f_linearized_median": float(np.nanmedian(cr_f_lin)),
                "cr_f_linearized_q25": float(np.nanpercentile(cr_f_lin, 25)),
                "cr_f_linearized_q75": float(np.nanpercentile(cr_f_lin, 75)),
                "n_directions": int(len(random_rows)),
            }
            _write_json(checkpoint, payload)

    for spec in selected:
        name = spec["name"]
        checkpoint = systems_dir / f"{name}.json"
        if checkpoint.is_file() and not args.overwrite:
            print(f"[{name}] checkpoint found; skipping compute", flush=True)
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            all_rows.extend(payload["rows"])
            all_random.extend(payload["random_rows"])
            summaries[name] = payload["summary"]
            continue
        try:
            outcome = run_system(spec, args)
        except Exception as error:
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"[{name}] FAILED: {failures[name]}; continuing", flush=True)
            continue
        payload = {
            "rows": outcome["rows"],
            "random_rows": outcome["random_rows"],
            "summary": outcome["summary"],
        }
        _write_json(checkpoint, payload)
        _save_npz(systems_dir / f"{name}.npz", outcome["artifacts"])
        all_rows.extend(outcome["rows"])
        all_random.extend(outcome["random_rows"])
        summaries[name] = outcome["summary"]
        _write_csv(args.output_dir / "results.csv", all_rows)
        _write_csv(args.output_dir / "random_baseline.csv", all_random)
        gates = evaluate_gates(summaries, all_rows)
        _write_json(args.output_dir / "VALIDATION.json", {"gates": gates,
            "failures": failures, "summaries": summaries})
        if not args.no_abort:
            blocking = []
            if not gates["energy_identity"]["pass"]:
                blocking.append("energy_identity")
            if not gates["decomposition_identity"]["pass"]:
                blocking.append("decomposition_identity")
            if name == GATE_SYSTEM:
                if name in failures or not gates["n2_probe_b_cancellation"]["pass"]:
                    blocking.append("n2_probe_b_cancellation")
            if blocking:
                print(
                    f"[{name}] GATE FAILURE: {', '.join(blocking)}; "
                    "stopping before the rest of the grid. See VALIDATION.json.",
                    flush=True,
                )
                raise SystemExit(3)

    gates = evaluate_gates(summaries, all_rows)
    provenance = {"gmsp_tree": str(GMSP)}
    try:
        import subprocess

        head = subprocess.run(
            ["git", "-C", str(GMSP), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if head:
            provenance["gmsp_git_head"] = head
    except Exception as error:  # pragma: no cover
        provenance["gmsp_git_head_error"] = str(error)
    _write_json(
        args.output_dir / "VALIDATION.json",
        {
            "gates": gates,
            "protocol": {
                "frame_pool": "Haar O(n)+U(n) 50/50 random mixture, N2 seeds",
                "complex_fraction": COMPLEX_FRACTION,
                "real_frame_seed": REAL_FRAME_SEED,
                "complex_frame_seed": COMPLEX_FRAME_SEED,
                "placement_seed": PLACEMENT_SEED,
                "row_selection": "QR pivots of covariance-diagonal-whitened "
                "design blocks (exact regularized covariances)",
                "covariance_shrinkage": SHRINKAGE,
                "covariance_relative_floor": RELATIVE_FLOOR,
                "nuclear_weight": NUCLEAR_WEIGHT,
                "grid_level": GRID_LEVEL,
                "probe_b_shots": int(args.shots),
                "probe_b_shot_seed": SHOT_SEED,
                "probe_c_directions": int(args.n_random),
                "probe_c_seed": RANDOM_DIRECTION_SEED,
                "solver": SOLVER_KWARGS,
                "nuclear_solver_override": _NUCLEAR_KWARGS_OVERRIDE or None,
                "solve_form_override": args.solve_form or None,
            },
            "provenance": provenance,
            "failures": failures,
            "summaries": summaries,
        },
    )
    _write_csv(args.output_dir / "results.csv", all_rows)
    _write_csv(args.output_dir / "random_baseline.csv", all_random)
    print(f"Done. all_pass={gates['all_pass']} -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
