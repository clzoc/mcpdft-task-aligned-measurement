"""Build the vendor-convention reference from the converged CASSCF solution.

Reproduces outputs/casscf_n2_10e8o-style CASSCF(10e,8o)/cc-pVDZ/R=1.10 A
(2 frozen 1s core orbitals) and packs the wave function into the
``MolecularReference`` used by the guarded constrained-shadow toolchain:
spin-orbital pair-basis 1- and 2-RDMs, the low-energy Hamiltonian
coefficients, and the D2h orbital irreps.
"""

from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from pyscf import ao2mo, fci, gto, mcscf, scf, symm  # noqa: E402
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PROTOCOL = ROOT / "mcpdft_measurement_revision" / "guarded_mcpdft_shadow_protocol"
for _path in (PROTOCOL / "code", PROTOCOL / "code" / "vendor", PROTOCOL / "tools"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from constrained_shadow import (  # noqa: E402
    MolecularReference,
    _canonicalize_mo_signs,
    _canonicalize_statevector_phase,
    _hamiltonian_coefficients,
    pair_basis,
    rdm_energy,
    rdms_from_statevector,
)

R_ANG = 1.10
BASIS = "cc-pvdz"
NCAS = 8
NELECAS = 10
OTXC = "ftPBE"
GRID_LEVEL = 1


def build(distance=R_ANG, basis=BASIS, ncas=NCAS, nelecas=NELECAS):
    mol = gto.M(
        atom=f"N 0 0 {-distance / 2.0:.10f}; N 0 0 {distance / 2.0:.10f}",
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        symmetry="D2h",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    mo = _canonicalize_mo_signs(mf.mo_coeff)
    mf.mo_coeff = mo

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.verbose = 0
    mc.kernel()
    mo = _canonicalize_mo_signs(mc.mo_coeff)
    mf.mo_coeff = mo
    mc.mo_coeff = mo
    n_core = int(mc.ncore)

    # The CASSCF kernel leaves ``mc.ci`` in the previous macro-iteration
    # orbital basis; re-converge it in the final orbitals (same snapshot step
    # used by the campaign reference).
    eris = mc.ao2mo(mo)
    e_h, _e_cas, ci = mc.casci(mo, ci0=mc.ci, eris=eris)

    n_alpha = nelecas // 2
    n_beta = nelecas // 2
    n_modes = 2 * ncas
    alpha_strings = fci.cistring.make_strings(range(ncas), n_alpha)
    beta_strings = fci.cistring.make_strings(range(ncas), n_beta)
    ci = np.asarray(ci)
    assert ci.shape == (len(alpha_strings), len(beta_strings)), ci.shape
    statevector = np.zeros(1 << n_modes, dtype=complex)
    for alpha_index, alpha_string in enumerate(alpha_strings):
        for beta_index, beta_string in enumerate(beta_strings):
            basis_state = int(alpha_string) | (int(beta_string) << ncas)
            statevector[basis_state] = ci[alpha_index, beta_index]
    statevector /= np.linalg.norm(statevector)
    statevector = _canonicalize_statevector_phase(statevector)

    pairs = pair_basis(n_modes)
    gamma, d2 = rdms_from_statevector(statevector, n_modes, pairs)

    one_electron, core_energy = mc.get_h1eff(mo)
    two_electron = ao2mo.restore(1, mc.get_h2eff(mo), ncas)
    active_hamiltonian = ElectronicEnergy.from_raw_integrals(
        one_electron, two_electron
    ).second_q_op()
    one_body, two_body = _hamiltonian_coefficients(
        active_hamiltonian, n_modes, pairs
    )

    orbital_labels = symm.label_orb_symm(
        mol, mol.irrep_name, mol.symm_orb, mo, check=True
    )
    active_labels = orbital_labels[n_core:n_core + ncas]
    orbital_irreps = tuple(
        int(symm.irrep_name2id(mol.groupname, label)) for label in active_labels
    )

    reference = MolecularReference(
        statevector=statevector,
        exact_energy=float(e_h),
        nuclear_energy=float(core_energy),
        exact_gamma=gamma,
        exact_d2=d2,
        one_body=one_body,
        two_body=two_body,
        pairs=pairs,
        n_spatial_orbitals=ncas,
        n_spin_orbitals=n_modes,
        n_alpha=n_alpha,
        n_beta=n_beta,
        orbital_irreps=orbital_irreps,
        molecule=mol,
        mean_field=mf,
        active_mo_coeff=np.asarray(mo[:, n_core:n_core + ncas]),
        n_core_orbitals=n_core,
    )
    reconstructed = rdm_energy(d2, gamma, one_body, two_body, core_energy)
    return reference, mc, reconstructed


if __name__ == "__main__":
    ref, mc, reconstructed = build()
    print(f"CASSCF E_H          = {ref.exact_energy:.12f} Eh (kernel {mc.e_tot:.12f}, rdm {reconstructed:.12f})")
    print(f"n_core              = {ref.n_core_orbitals}")
    print(f"active irreps       = {ref.orbital_irreps}")
    print(f"||gamma||_F         = {np.linalg.norm(ref.exact_gamma):.12f}")
    print(f"||d2||_F            = {np.linalg.norm(ref.exact_d2):.12f}")
    print(f"pair-space 1% radius= {0.01*np.linalg.norm(ref.exact_d2):.12f}")
    print(f"E_H 1% band         = +/- {0.01*abs(ref.exact_energy):.12f} Eh")
    print(f"rdm_energy check    = {rdm_energy(ref.exact_d2, ref.exact_gamma, ref.one_body, ref.two_body, ref.nuclear_energy):.12f}")
