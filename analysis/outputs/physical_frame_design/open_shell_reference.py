"""Open-shell CASCI reference on canonical ROHF orbitals.

The closed-shell path (`cancellation_probe.build_canonical_cas_reference`) keeps
``spin=0`` and ``n_alpha=n_beta``.  This module mirrors that builder for
open-shell multiplicities so the measurement pipeline can target systems such
as the O2 triplet, whose exact state has n_alpha != n_beta.

Everything downstream of the resulting ``MolecularReference`` (DQG SDP, frame
design, acquisition, ftPBE evaluation) already only consumes ``n_alpha``,
``n_beta``, the spin blocks of gamma/d2 and the spin-summed density; ftPBE is a
functional of the total density and the on-top pair density, so no spin-resolved
functional change is required.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def build_rohf_cas_reference(spec: dict[str, Any]):
    """CASCI reference on canonical ROHF orbitals for an open-shell state.

    ``spec`` accepts the same keys as the closed-shell builder plus ``spin``
    (PySCF convention: 2S, the number of unpaired electrons) and an optional
    ``symmetry`` (default ``"D2h"``; pass ``None`` for C1).
    """

    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from pyscf import ao2mo, fci, gto, mcscf, scf, symm

    from constrained_shadow import (
        MolecularReference,
        _canonicalize_mo_signs,
        _canonicalize_statevector_phase,
        _hamiltonian_coefficients,
        pair_basis,
        rdm_energy,
        rdms_from_statevector,
    )

    active_electrons = int(spec["active_electrons"])
    active_orbitals = int(spec["active_orbitals"])
    spin = int(spec.get("spin", 0))
    if spin <= 0:
        raise ValueError("build_rohf_cas_reference requires spin > 0.")
    if (active_electrons - spin) % 2:
        raise ValueError("spin and active_electrons must share the same parity.")
    n_alpha = (active_electrons + spin) // 2
    n_beta = (active_electrons - spin) // 2

    symmetry = spec.get("symmetry", "D2h")
    molecule = gto.M(
        atom=spec["atom"],
        basis=spec["basis"],
        charge=int(spec.get("charge", 0)),
        spin=spin,
        unit="Angstrom",
        symmetry=symmetry if symmetry else False,
        verbose=0,
    )
    mean_field = scf.ROHF(molecule)
    mean_field.conv_tol = 1e-10
    mean_field.max_cycle = 200
    mean_field.kernel()
    if not mean_field.converged:
        mean_field = mean_field.newton().run()
    if not mean_field.converged:
        raise RuntimeError(f"{spec['name']}: ROHF did not converge.")
    mo_coeff = _canonicalize_mo_signs(mean_field.mo_coeff)
    mean_field.mo_coeff = mo_coeff

    ncore = spec.get("ncore")
    if ncore is None:
        ncore = (molecule.nelectron - active_electrons) // 2
    ncore = int(ncore)
    if 2 * ncore + active_electrons != molecule.nelectron:
        raise ValueError(
            f"{spec['name']}: ncore={ncore} plus CAS({active_electrons},"
            f"{active_orbitals}) does not cover {molecule.nelectron} electrons."
        )

    casci = mcscf.CASCI(mean_field, active_orbitals, active_electrons)
    casci.ncore = ncore
    casci.kernel()
    if not casci.converged:
        raise RuntimeError(f"{spec['name']}: CASCI did not converge.")
    spin_square = float(
        casci.fcisolver.spin_square(casci.ci, active_orbitals, (n_alpha, n_beta))[0]
    )
    expected = 0.25 * spin * (spin + 2)
    if abs(spin_square - expected) > 1e-6:
        raise RuntimeError(
            f"{spec['name']}: S^2={spin_square}, expected {expected}."
        )

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
            int(symm.irrep_name2id(molecule.groupname, label))
            for label in active_labels
        )
    else:
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
        active_mo_coeff=np.asarray(
            mean_field.mo_coeff[:, ncore : ncore + active_orbitals]
        ),
        n_core_orbitals=ncore,
    )
    aux = {
        "rhf_energy_eh": float(mean_field.e_tot),
        "spin_square": spin_square,
        "spin": spin,
        "n_alpha": n_alpha,
        "n_beta": n_beta,
    }
    return reference, identity_error, aux
