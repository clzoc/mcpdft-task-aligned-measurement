"""Guarded MC-PDFT c-optimal design for orbital-rotation shadows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from constrained_shadow import (
    MolecularReference,
    PairVectorDesign,
    ShadowData,
    WeightedFitStatistics,
    _canonicalize_mo_signs,
    _canonicalize_statevector_phase,
    _hamiltonian_coefficients,
    pair_basis,
    rdm_energy,
    rdms_from_statevector,
)


@dataclass(frozen=True)
class SelectionReference:
    """Active-space Hamiltonian and orbitals built without an FCI solve."""

    nuclear_energy: float
    one_body: np.ndarray
    two_body: np.ndarray
    pairs: tuple[tuple[int, int], ...]
    n_spatial_orbitals: int
    n_spin_orbitals: int
    n_alpha: int
    n_beta: int
    orbital_irreps: tuple[int, ...]
    molecule: Any
    mean_field: Any
    active_mo_coeff: np.ndarray
    n_core_orbitals: int

    @property
    def n_electrons(self) -> int:
        return self.n_alpha + self.n_beta


def _build_c2_active_space(
    bond_length: float,
    basis: str,
    frozen_mo_coeff: np.ndarray | None = None,
):
    """Build deterministic RHF/AVAS orbitals shared by selection and truth."""

    from pyscf import fci, gto, mcscf, scf, symm

    molecule = gto.M(
        atom=(
            f"C 0 0 {-0.5 * bond_length:.12f}; "
            f"C 0 0 {0.5 * bond_length:.12f}"
        ),
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        symmetry="D2h",
        verbose=0,
    )
    mean_field = scf.RHF(molecule)
    mean_field.conv_tol = 1e-10
    mean_field.max_cycle = 100
    mean_field.kernel()
    if not mean_field.converged:
        mean_field = mean_field.newton().run()
    if frozen_mo_coeff is None:
        from pyscf.mcscf import avas

        ncas, nelec, orbitals = avas.avas(
            mean_field,
            ["C 2s", "C 2p"],
            threshold=0.1,
            ncore=2,
            canonicalize=True,
            openshell_option=2,
            verbose=0,
        )
        if (int(nelec), int(ncas)) != (8, 8):
            raise RuntimeError(
                "C 2s/2p AVAS did not produce CAS(8e,8o): "
                f"received CAS({int(nelec)}e,{int(ncas)}o)."
            )
        orbitals = _canonicalize_mo_signs(np.asarray(orbitals, dtype=float))
    else:
        supplied = np.asarray(frozen_mo_coeff)
        if np.iscomplexobj(supplied) and np.max(np.abs(supplied.imag)) > 1e-12:
            raise ValueError("Frozen C2 MO coefficients must be real.")
        orbitals = np.asarray(supplied.real, dtype=float)
        expected_shape = np.asarray(mean_field.mo_coeff).shape
        if orbitals.shape != expected_shape:
            raise ValueError(
                "Frozen C2 MO coefficients have shape "
                f"{orbitals.shape}; expected {expected_shape}."
            )
        if not np.all(np.isfinite(orbitals)):
            raise ValueError("Frozen C2 MO coefficients must be finite.")
        overlap = mean_field.get_ovlp()
        metric = orbitals.T @ overlap @ orbitals
        if not np.allclose(metric, np.eye(metric.shape[0]), atol=2e-8):
            maximum_error = float(np.max(np.abs(metric - np.eye(metric.shape[0]))))
            raise ValueError(
                "Frozen C2 MO coefficients are not orthonormal in the AO metric: "
                f"maximum error={maximum_error:.3e}."
            )
    mean_field.mo_coeff = orbitals
    active_problem = mcscf.CASCI(mean_field, 8, (4, 4))
    active_problem.ncore = 2
    active_problem.mo_coeff = orbitals
    active_problem.fcisolver = fci.direct_spin1_symm.FCI(molecule)
    active_problem.fcisolver.wfnsym = "Ag"
    fci.addons.fix_spin_(active_problem.fcisolver, shift=1.0, ss=0.0)
    orbital_labels = symm.label_orb_symm(
        molecule,
        molecule.irrep_name,
        molecule.symm_orb,
        orbitals,
        check=True,
    )
    active_labels = orbital_labels[2:10]
    orbital_irreps = tuple(
        int(symm.irrep_name2id(molecule.groupname, label))
        for label in active_labels
    )
    return molecule, mean_field, active_problem, orbitals, orbital_irreps


def build_c2_selection_reference(
    bond_length: float = 1.25,
    basis: str = "cc-pvtz",
    frozen_mo_coeff: np.ndarray | None = None,
) -> SelectionReference:
    """Build a C2 valence CAS(8,8) Hamiltonian without CASCI/FCI."""

    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from pyscf import ao2mo

    (
        molecule,
        mean_field,
        active_problem,
        orbitals,
        orbital_irreps,
    ) = _build_c2_active_space(bond_length, basis, frozen_mo_coeff)
    one_electron, core_energy = active_problem.get_h1eff(orbitals)
    two_electron = ao2mo.restore(1, active_problem.get_h2eff(orbitals), 8)
    n_modes = 16
    pairs = pair_basis(n_modes)
    active_hamiltonian = ElectronicEnergy.from_raw_integrals(
        one_electron, two_electron
    ).second_q_op()
    one_body, two_body = _hamiltonian_coefficients(
        active_hamiltonian, n_modes, pairs
    )
    return SelectionReference(
        nuclear_energy=float(core_energy),
        one_body=one_body,
        two_body=two_body,
        pairs=pairs,
        n_spatial_orbitals=8,
        n_spin_orbitals=n_modes,
        n_alpha=4,
        n_beta=4,
        orbital_irreps=orbital_irreps,
        molecule=molecule,
        mean_field=mean_field,
        active_mo_coeff=np.asarray(orbitals[:, 2:10]),
        n_core_orbitals=2,
    )


def build_c2_reference(
    bond_length: float = 1.25,
    basis: str = "cc-pvtz",
    frozen_mo_coeff: np.ndarray | None = None,
) -> MolecularReference:
    """Build the post-selection C2 valence CASCI(8,8) benchmark state."""

    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from pyscf import ao2mo, fci

    (
        molecule,
        mean_field,
        active_problem,
        orbitals,
        orbital_irreps,
    ) = _build_c2_active_space(bond_length, basis, frozen_mo_coeff)
    active_problem.kernel(orbitals)
    if not active_problem.converged:
        raise RuntimeError("C2 CASCI did not converge.")
    spin_square = active_problem.fcisolver.spin_square(
        active_problem.ci, 8, (4, 4)
    )[0]
    if abs(float(spin_square)) > 1e-7:
        raise RuntimeError(f"C2 CASCI singlet check failed: S^2={spin_square}.")
    alpha_strings = fci.cistring.make_strings(range(8), 4)
    beta_strings = fci.cistring.make_strings(range(8), 4)
    statevector = np.zeros(1 << 16, dtype=complex)
    for alpha_index, alpha_string in enumerate(alpha_strings):
        for beta_index, beta_string in enumerate(beta_strings):
            state = int(alpha_string) | (int(beta_string) << 8)
            statevector[state] = active_problem.ci[alpha_index, beta_index]
    statevector /= np.linalg.norm(statevector)
    statevector = _canonicalize_statevector_phase(statevector)

    pairs = pair_basis(16)
    gamma, d2 = rdms_from_statevector(statevector, 16, pairs)
    one_electron, core_energy = active_problem.get_h1eff(orbitals)
    two_electron = ao2mo.restore(1, active_problem.get_h2eff(orbitals), 8)
    active_hamiltonian = ElectronicEnergy.from_raw_integrals(
        one_electron, two_electron
    ).second_q_op()
    one_body, two_body = _hamiltonian_coefficients(
        active_hamiltonian, 16, pairs
    )
    exact_energy = float(active_problem.e_tot)
    reconstructed_energy = rdm_energy(
        d2, gamma, one_body, two_body, float(core_energy)
    )
    if not np.isclose(exact_energy, reconstructed_energy, atol=1e-8):
        raise RuntimeError(
            "The C2 CASCI Hamiltonian and pair-basis 2-RDM disagree: "
            f"{exact_energy} versus {reconstructed_energy}."
        )
    return MolecularReference(
        statevector=statevector,
        exact_energy=exact_energy,
        nuclear_energy=float(core_energy),
        exact_gamma=gamma,
        exact_d2=d2,
        one_body=one_body,
        two_body=two_body,
        pairs=pairs,
        n_spatial_orbitals=8,
        n_spin_orbitals=16,
        n_alpha=4,
        n_beta=4,
        orbital_irreps=orbital_irreps,
        molecule=molecule,
        mean_field=mean_field,
        active_mo_coeff=np.asarray(orbitals[:, 2:10]),
        n_core_orbitals=2,
    )


def build_n2_selection_reference(
    bond_length: float = 1.75,
    basis: str = "cc-pvdz",
    active_electrons: int = 10,
    active_orbitals: int = 8,
    active_orbital_signs: Sequence[float] | None = None,
) -> SelectionReference:
    """Build the N2 active Hamiltonian without running CASCI/FCI."""

    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from pyscf import ao2mo, gto, mcscf, scf, symm

    if active_electrons < 2 or active_electrons % 2:
        raise ValueError(
            "The spin-singlet N2 path requires a positive even active-electron count."
        )
    if active_orbitals < 1 or active_electrons > 2 * active_orbitals:
        raise ValueError(
            "The active space must have enough spatial orbitals for its electrons."
        )
    geometry = f"N 0 0 {-bond_length / 2.0}; N 0 0 {bond_length / 2.0}"
    molecule = gto.M(
        atom=geometry,
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        symmetry="D2h",
        verbose=0,
    )
    mean_field = scf.RHF(molecule).run()
    mo_coeff = _canonicalize_mo_signs(mean_field.mo_coeff)
    if active_electrons > molecule.nelectron:
        raise ValueError("Active electrons cannot exceed the molecular electron count.")
    n_core = (molecule.nelectron - active_electrons) // 2
    if n_core + active_orbitals > mean_field.mo_coeff.shape[1]:
        raise ValueError(
            "The requested frozen-core plus active space exceeds the RHF MO space."
        )
    if active_orbital_signs is not None:
        signs = np.asarray(active_orbital_signs, dtype=float)
        if signs.shape != (active_orbitals,) or not np.all(np.abs(signs) == 1.0):
            raise ValueError("Active-orbital signs must contain one +/-1 per orbital.")
        mo_coeff = mo_coeff.copy()
        mo_coeff[:, n_core : n_core + active_orbitals] *= signs[None, :]
    mean_field.mo_coeff = mo_coeff

    active_problem = mcscf.CASCI(mean_field, active_orbitals, active_electrons)
    active_problem.ncore = n_core
    one_electron, core_energy = active_problem.get_h1eff()
    two_electron = ao2mo.restore(
        1, active_problem.get_h2eff(), active_orbitals
    )
    n_modes = 2 * active_orbitals
    pairs = pair_basis(n_modes)
    active_hamiltonian = ElectronicEnergy.from_raw_integrals(
        one_electron, two_electron
    ).second_q_op()
    one_body, two_body = _hamiltonian_coefficients(
        active_hamiltonian, n_modes, pairs
    )

    orbital_labels = symm.label_orb_symm(
        molecule,
        molecule.irrep_name,
        molecule.symm_orb,
        mean_field.mo_coeff,
        check=True,
    )
    active_labels = orbital_labels[n_core : n_core + active_orbitals]
    orbital_irreps = tuple(
        int(symm.irrep_name2id(molecule.groupname, label))
        for label in active_labels
    )
    return SelectionReference(
        nuclear_energy=float(core_energy),
        one_body=one_body,
        two_body=two_body,
        pairs=pairs,
        n_spatial_orbitals=active_orbitals,
        n_spin_orbitals=n_modes,
        n_alpha=active_electrons // 2,
        n_beta=active_electrons // 2,
        orbital_irreps=orbital_irreps,
        molecule=molecule,
        mean_field=mean_field,
        active_mo_coeff=np.asarray(
            mean_field.mo_coeff[:, n_core : n_core + active_orbitals]
        ),
        n_core_orbitals=n_core,
    )


def _build_cr2_active_space(
    bond_length: float,
    basis: str,
) -> tuple[Any, Any, Any, np.ndarray, tuple[int, ...]]:
    """Build Cr2 RHF orbitals and CASCI object without running FCI."""

    from pyscf import gto, mcscf, scf, symm

    molecule = gto.M(
        atom=(
            f"Cr 0 0 {-0.5 * bond_length:.12f}; "
            f"Cr 0 0 {0.5 * bond_length:.12f}"
        ),
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        symmetry=True,
        verbose=0,
        max_memory=12_000,
    )
    mean_field = scf.RHF(molecule)
    mean_field.level_shift = 0.4
    mean_field.conv_tol = 1e-10
    mean_field.max_cycle = 200
    mean_field.kernel()
    if not mean_field.converged:
        mean_field = mean_field.newton()
        mean_field.conv_tol = 1e-10
        mean_field.max_cycle = 100
        mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError("Cr2 RHF did not converge.")

    active_problem = mcscf.CASSCF(mean_field, 12, 12)
    ncore = {"A1g": 5, "A1u": 5}
    ncas = {
        "A1g": 2,
        "A1u": 2,
        "E1ux": 1,
        "E1uy": 1,
        "E1gx": 1,
        "E1gy": 1,
        "E2ux": 1,
        "E2uy": 1,
        "E2gx": 1,
        "E2gy": 1,
    }
    orbitals = mcscf.sort_mo_by_irrep(
        active_problem, mean_field.mo_coeff, ncas, ncore
    )
    orbitals = _canonicalize_mo_signs(np.asarray(orbitals, dtype=float))
    mean_field.mo_coeff = orbitals

    active_problem = mcscf.CASCI(mean_field, 12, (6, 6))
    active_problem.ncore = 18
    active_problem.mo_coeff = orbitals

    d2h_molecule = gto.M(
        atom=(
            f"Cr 0 0 {-0.5 * bond_length:.12f}; "
            f"Cr 0 0 {0.5 * bond_length:.12f}"
        ),
        basis=basis,
        charge=0,
        spin=0,
        unit="Angstrom",
        symmetry="D2h",
        verbose=0,
    )
    orbital_labels = symm.label_orb_symm(
        d2h_molecule,
        d2h_molecule.irrep_name,
        d2h_molecule.symm_orb,
        orbitals,
        check=True,
    )
    active_labels = orbital_labels[18:30]
    orbital_irreps = tuple(
        int(symm.irrep_name2id(d2h_molecule.groupname, label))
        for label in active_labels
    )
    return molecule, mean_field, active_problem, orbitals, orbital_irreps


def build_cr2_selection_reference(
    bond_length: float = 1.68,
    basis: str = "cc-pvtz",
    frozen_mo_coeff: np.ndarray | None = None,
) -> SelectionReference:
    """Build a Cr2 CAS(12,12) Hamiltonian without CASCI/FCI.

    When *frozen_mo_coeff* is supplied those orbitals are used verbatim
    instead of running a fresh RHF + irrep sort.  This guarantees that
    the selection Hamiltonian is built on the exact-reference orbitals.
    """

    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
    from pyscf import ao2mo, gto, mcscf, scf, symm

    if frozen_mo_coeff is None:
        molecule, mean_field, active_problem, orbitals, orbital_irreps = (
            _build_cr2_active_space(bond_length, basis)
        )
    else:
        molecule = gto.M(
            atom=(
                f"Cr 0 0 {-0.5 * bond_length:.12f}; "
                f"Cr 0 0 {0.5 * bond_length:.12f}"
            ),
            basis=basis,
            charge=0,
            spin=0,
            unit="Angstrom",
            symmetry=True,
            verbose=0,
            max_memory=12_000,
        )
        mean_field = scf.RHF(molecule)
        mean_field.kernel()
        if not mean_field.converged:
            raise RuntimeError("Cr2 RHF (frozen MO path) did not converge.")
        orbitals = np.asarray(frozen_mo_coeff, dtype=float)
        expected_shape = np.asarray(mean_field.mo_coeff).shape
        if orbitals.shape != expected_shape:
            raise ValueError(
                "Frozen Cr2 MO coefficients have shape "
                f"{orbitals.shape}; expected {expected_shape}."
            )
        overlap = mean_field.get_ovlp()
        metric = orbitals.T @ overlap @ orbitals
        if not np.allclose(metric, np.eye(metric.shape[0]), atol=2e-8):
            raise RuntimeError(
                "Frozen Cr2 MOs are not AO-metric orthonormal."
            )
        mean_field.mo_coeff = orbitals
        active_problem = mcscf.CASCI(mean_field, 12, (6, 6))
        active_problem.ncore = 18
        active_problem.mo_coeff = orbitals
        d2h_molecule = gto.M(
            atom=(
                f"Cr 0 0 {-0.5 * bond_length:.12f}; "
                f"Cr 0 0 {0.5 * bond_length:.12f}"
            ),
            basis=basis,
            charge=0,
            spin=0,
            unit="Angstrom",
            symmetry="D2h",
            verbose=0,
        )
        orbital_labels = symm.label_orb_symm(
            d2h_molecule,
            d2h_molecule.irrep_name,
            d2h_molecule.symm_orb,
            orbitals,
            check=True,
        )
        active_labels = orbital_labels[18:30]
        orbital_irreps = tuple(
            int(symm.irrep_name2id(d2h_molecule.groupname, label))
            for label in active_labels
        )
    one_electron, core_energy = active_problem.get_h1eff(orbitals)
    two_electron = ao2mo.restore(
        1, active_problem.get_h2eff(orbitals), 12
    )
    n_modes = 24
    pairs = pair_basis(n_modes)
    active_hamiltonian = ElectronicEnergy.from_raw_integrals(
        one_electron, two_electron
    ).second_q_op()
    one_body, two_body = _hamiltonian_coefficients(
        active_hamiltonian, n_modes, pairs
    )
    return SelectionReference(
        nuclear_energy=float(core_energy),
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


def load_blind_shadow_npz(
    path: str | Path, *, load_design: bool = True
) -> ShadowData:
    """Load raw shots while deliberately ignoring archived exact responses.

    ``load_design=False`` keeps the quadratic design implicit in the much
    smaller pair-vector array.  This is the low-memory path for large pools.
    """

    archive_path = Path(path)
    with np.load(archive_path) as arrays:
        required = {
            "rotations",
            "pair_vectors",
            "hits",
            "values",
            "lower_bounds",
            "upper_bounds",
            "shots_per_basis",
        }
        if load_design:
            required.update(
                {
                    "design_data",
                    "design_indices",
                    "design_indptr",
                    "design_shape",
                }
            )
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError(
                f"Shadow NPZ is missing arrays: {', '.join(sorted(missing))}."
            )
        pair_vectors = np.asarray(arrays["pair_vectors"])
        if load_design:
            design = sparse.csr_matrix(
                (
                    np.asarray(arrays["design_data"], dtype=float),
                    np.asarray(arrays["design_indices"], dtype=np.int32),
                    np.asarray(arrays["design_indptr"], dtype=np.int32),
                ),
                shape=tuple(int(value) for value in arrays["design_shape"]),
                copy=False,
            )
        else:
            design = PairVectorDesign(pair_vectors)
        values = np.asarray(arrays["values"], dtype=float)
        return ShadowData(
            rotations=tuple(
                np.asarray(rotation) for rotation in arrays["rotations"]
            ),
            pair_vectors=pair_vectors,
            design=design,
            values=values,
            lower_bounds=np.asarray(arrays["lower_bounds"], dtype=float),
            upper_bounds=np.asarray(arrays["upper_bounds"], dtype=float),
            hits=np.asarray(arrays["hits"], dtype=int),
            shots_per_basis=int(np.asarray(arrays["shots_per_basis"]).item()),
            exact_values=np.full_like(values, np.nan),
            exact_constraints=False,
            occupations=(
                np.asarray(arrays["occupations"], dtype=np.uint64)
                if "occupations" in arrays.files
                else None
            ),
        )


@dataclass(frozen=True)
class CandidateDesignScore:
    index: int
    d_optimal_gain: float
    mcpdft_variance_reduction: float
    mcpdft_fractional_reduction: float
    passes_design_guard: bool


@dataclass(frozen=True)
class GuardedDesignChoice:
    selected_index: int
    design_only_index: int
    design_guard_fraction: float
    maximum_d_optimal_gain: float
    current_mcpdft_standard_error: float
    selected_mcpdft_standard_error: float
    current_d2_trace_standard_error: float
    selected_d2_trace_standard_error: float
    selected: CandidateDesignScore
    design_only: CandidateDesignScore
    scores: tuple[CandidateDesignScore, ...]


@dataclass(frozen=True)
class MultiTargetCandidateScore:
    index: int
    d_optimal_gain: float
    hamiltonian_variance_reduction: float
    hamiltonian_fractional_reduction: float
    mcpdft_variance_reduction: float
    mcpdft_fractional_reduction: float
    passes_design_guard: bool
    passes_hamiltonian_guard: bool


@dataclass(frozen=True)
class SafeMCPDFTChoice:
    selected_index: int
    hamiltonian_only_index: int
    design_only_index: int
    design_guard_fraction: float
    hamiltonian_guard_fraction: float
    maximum_d_optimal_gain: float
    maximum_hamiltonian_variance_reduction: float
    current_hamiltonian_standard_error: float
    selected_hamiltonian_standard_error: float
    current_mcpdft_standard_error: float
    selected_mcpdft_standard_error: float
    selected: MultiTargetCandidateScore
    hamiltonian_only: MultiTargetCandidateScore
    design_only: MultiTargetCandidateScore
    scores: tuple[MultiTargetCandidateScore, ...]


@dataclass(frozen=True)
class LeakGuardShadowData:
    """Selection-safe shadow view that rejects outcome or truth access."""

    rotations: tuple[np.ndarray, ...]
    design: Any
    n_shadows: int
    row_count: int

    @classmethod
    def from_shadow_data(cls, shadows: ShadowData) -> "LeakGuardShadowData":
        return cls(
            rotations=tuple(np.asarray(rotation) for rotation in shadows.rotations),
            design=shadows.design,
            n_shadows=shadows.n_shadows,
            row_count=len(shadows.values),
        )

    def __getattr__(self, name: str) -> Any:
        if name in {
            "values",
            "hits",
            "exact_values",
            "lower_bounds",
            "upper_bounds",
            "occupations",
        }:
            raise RuntimeError(
                f"Selection attempted forbidden shadow-field access: {name}"
            )
        raise AttributeError(name)


class LeakGuardReference:
    """Reference proxy that rejects exact-state access during selection."""

    def __init__(self, reference: Any):
        object.__setattr__(self, "_reference", reference)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("exact_"):
            raise RuntimeError(
                f"Selection attempted forbidden exact-reference access: {name}"
            )
        return getattr(object.__getattribute__(self, "_reference"), name)


def validate_shadow_archive_structure(
    path: str | Path,
    shadows: ShadowData,
    reference: Any,
    required_count: int,
    expected_shots: int,
) -> None:
    """Validate selection-time dimensions without consulting exact-state data."""

    archive_path = Path(path)
    if shadows.n_shadows < required_count:
        raise ValueError(
            f"{archive_path} has {shadows.n_shadows} shadows; "
            f"{required_count} are required."
        )
    if shadows.shots_per_basis != expected_shots:
        raise ValueError(
            f"{archive_path} uses {shadows.shots_per_basis} shots, "
            f"expected {expected_shots}."
        )

    n_spatial = int(reference.n_spatial_orbitals)
    rotation_shape = (n_spatial, n_spatial)
    if any(np.asarray(rotation).shape != rotation_shape for rotation in shadows.rotations):
        raise ValueError(f"{archive_path} does not match the requested active space.")

    n_pairs = len(reference.pairs)
    expected_rows = shadows.n_shadows * n_pairs
    if shadows.pair_vectors.shape != (expected_rows, n_pairs):
        raise ValueError(f"{archive_path} has incompatible pair-vector dimensions.")
    if shadows.design.shape != (expected_rows, n_pairs * n_pairs):
        raise ValueError(f"{archive_path} has incompatible design dimensions.")
    for name in ("values", "lower_bounds", "upper_bounds", "hits"):
        if np.shape(getattr(shadows, name)) != (expected_rows,):
            raise ValueError(f"{archive_path} has incompatible {name} dimensions.")


def acquire_shadow_bases(
    shadow_data: ShadowData, indices: Sequence[int]
) -> ShadowData:
    """Acquire selected bases while stripping all exact-reference responses."""

    selected = tuple(int(index) for index in indices)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Acquired shadow indices must be nonempty and unique.")
    if min(selected) < 0 or max(selected) >= shadow_data.n_shadows:
        raise ValueError("An acquired shadow index is outside the candidate pool.")
    rows_per_shadow = len(shadow_data.values) // shadow_data.n_shadows
    rows = np.concatenate(
        [
            np.arange(index * rows_per_shadow, (index + 1) * rows_per_shadow)
            for index in selected
        ]
    )
    pair_vectors = np.asarray(shadow_data.pair_vectors[rows])
    values = np.asarray(shadow_data.values[rows], dtype=float)
    design = (
        PairVectorDesign(pair_vectors)
        if isinstance(shadow_data.design, PairVectorDesign)
        else shadow_data.design[rows]
    )
    return ShadowData(
        rotations=tuple(shadow_data.rotations[index] for index in selected),
        pair_vectors=pair_vectors,
        design=design,
        values=values,
        lower_bounds=shadow_data.lower_bounds[rows],
        upper_bounds=shadow_data.upper_bounds[rows],
        hits=shadow_data.hits[rows],
        shots_per_basis=shadow_data.shots_per_basis,
        exact_values=np.full_like(values, np.nan),
        exact_constraints=False,
        occupations=(
            None
            if shadow_data.occupations is None
            else np.asarray(shadow_data.occupations, dtype=np.uint64)[list(selected)]
        ),
    )


def symmetric_matrix_gradient_vector(
    matrix: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Map a symmetric matrix gradient to unique symmetric variables."""

    gradient = np.asarray(matrix, dtype=float)
    row_indices = np.asarray(rows, dtype=int)
    column_indices = np.asarray(cols, dtype=int)
    values = gradient[row_indices, column_indices].copy()
    off_diagonal = row_indices != column_indices
    values[off_diagonal] += gradient[
        column_indices[off_diagonal], row_indices[off_diagonal]
    ]
    return values


def shadow_design_blocks(
    shadows: ShadowData | LeakGuardShadowData,
    n_pairs: int,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Return one unique-symmetric-variable design block per shadow basis."""

    row_indices = np.asarray(rows, dtype=int)
    column_indices = np.asarray(cols, dtype=int)
    primary = row_indices * int(n_pairs) + column_indices
    secondary = column_indices * int(n_pairs) + row_indices
    off_diagonal = primary != secondary
    row_count = (
        shadows.row_count
        if isinstance(shadows, LeakGuardShadowData)
        else len(shadows.values)
    )
    rows_per_shadow = row_count // shadows.n_shadows
    blocks = []
    for index in range(shadows.n_shadows):
        start = index * rows_per_shadow
        stop = start + rows_per_shadow
        if isinstance(shadows.design, PairVectorDesign):
            vectors = np.asarray(shadows.design.pair_vectors[start:stop])
            block = np.real(
                np.conjugate(vectors[:, row_indices])
                * vectors[:, column_indices]
            )
            block[:, off_diagonal] *= 2.0
        else:
            block = shadows.design[start:stop, primary].toarray()
            block[:, off_diagonal] += shadows.design[
                start:stop, secondary[off_diagonal]
            ].toarray()
        blocks.append(block)
    return tuple(blocks)


def _shadow_aligned_chunk_ranges(
    observation_count: int,
    shadow_count: int,
    maximum_rows: int,
) -> tuple[tuple[int, int], ...]:
    """Partition ordered observations without splitting a shadow basis."""

    if observation_count < 1 or shadow_count < 1 or maximum_rows < 1:
        raise ValueError("Chunk dimensions must be positive.")
    if observation_count % shadow_count:
        raise ValueError("Observations must contain complete shadow bases.")
    rows_per_shadow = observation_count // shadow_count
    shadows_per_chunk = max(1, maximum_rows // rows_per_shadow)
    rows_per_chunk = shadows_per_chunk * rows_per_shadow
    return tuple(
        (start, min(start + rows_per_chunk, observation_count))
        for start in range(0, observation_count, rows_per_chunk)
    )


def build_weighted_fit_statistics(
    shadows: ShadowData,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    chunk_rows: int = 2048,
    variance_floor: float = 1e-12,
) -> WeightedFitStatistics:
    """Compress the full weighted residual into an exact bounded-memory TSQR.

    The returned factor represents all observations jointly.  Chunking changes
    only how the factor is constructed, not the fitted objective or feasible
    set.
    """

    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive.")
    if variance_floor <= 0.0:
        raise ValueError("variance_floor must be positive.")
    row_indices = np.asarray(rows, dtype=int).reshape(-1)
    column_indices = np.asarray(cols, dtype=int).reshape(-1)
    if row_indices.shape != column_indices.shape or len(row_indices) == 0:
        raise ValueError("rows and cols must be nonempty arrays of equal length.")
    n_pairs = shadows.pair_vectors.shape[1]
    if (
        np.any(row_indices < 0)
        or np.any(column_indices < 0)
        or np.any(row_indices >= n_pairs)
        or np.any(column_indices >= n_pairs)
    ):
        raise ValueError("Weighted-fit variable indices are out of range.")
    observation_count = len(shadows.values)
    if shadows.pair_vectors.shape[0] != observation_count:
        raise ValueError("Pair vectors and shadow observations have different lengths.")
    if shadows.shots_per_basis < 1 or np.any(shadows.hits < 0):
        raise ValueError("Weighted-fit statistics require finite-shot hit counts.")

    off_diagonal = row_indices != column_indices
    factor = np.empty((0, len(row_indices) + 1), dtype=float)
    chunk_ranges = _shadow_aligned_chunk_ranges(
        observation_count, shadows.n_shadows, chunk_rows
    )
    for start, stop in chunk_ranges:
        vectors = np.asarray(shadows.pair_vectors[start:stop])
        block = np.real(
            np.conjugate(vectors[:, row_indices])
            * vectors[:, column_indices]
        )
        block[:, off_diagonal] *= 2.0
        probabilities = (
            np.asarray(shadows.hits[start:stop], dtype=float) + 0.5
        ) / (shadows.shots_per_basis + 1.0)
        variances = np.maximum(
            probabilities * (1.0 - probabilities) / shadows.shots_per_basis,
            variance_floor,
        )
        inverse_errors = 1.0 / np.sqrt(variances)
        augmented = np.empty((stop - start, len(row_indices) + 1), dtype=float)
        augmented[:, :-1] = inverse_errors[:, None] * block
        augmented[:, -1] = -inverse_errors * np.asarray(
            shadows.values[start:stop], dtype=float
        )
        stacked = augmented if factor.size == 0 else np.vstack((factor, augmented))
        factor = np.linalg.qr(stacked, mode="reduced")[1]

    return WeightedFitStatistics(
        rows=row_indices,
        cols=column_indices,
        residual_factor=factor / np.sqrt(observation_count),
        observation_count=observation_count,
    )


def matrix_to_variable_vector(
    matrix: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Extract unique entries from a symmetric matrix."""

    array = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    return array[np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)]


def remove_parallel_component(target: np.ndarray, nuisance: np.ndarray) -> np.ndarray:
    """Remove the Euclidean component of a target parallel to a nuisance."""

    target_vector = np.asarray(target, dtype=float)
    nuisance_vector = np.asarray(nuisance, dtype=float)
    norm_squared = float(nuisance_vector @ nuisance_vector)
    if norm_squared == 0.0:
        return target_vector.copy()
    return target_vector - nuisance_vector * float(
        target_vector @ nuisance_vector / norm_squared
    )


def contraction_gamma_gradient_vector(
    gamma_gradient: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    rows: np.ndarray,
    cols: np.ndarray,
    n_electrons: int,
) -> np.ndarray:
    """Pull a 1-RDM tangent back to unique 2-RDM variables via contraction.

    For pair-basis D2, ``gamma[p,r] = sum_q D[pq,rq] / (N-1)``.  This
    returns the symmetric-variable gradient of ``Tr(G_gamma gamma[D2])``.
    """

    if n_electrons <= 1:
        raise ValueError("n_electrons must exceed one.")
    gradient = 0.5 * (
        np.asarray(gamma_gradient, dtype=float)
        + np.asarray(gamma_gradient, dtype=float).T
    )
    n_modes = gradient.shape[0]
    if gradient.shape != (n_modes, n_modes):
        raise ValueError("gamma_gradient must be square.")
    pair_lookup = {tuple(pair): index for index, pair in enumerate(pairs)}

    def pair_index(first: int, second: int) -> tuple[int | None, int]:
        if first == second:
            return None, 0
        if first < second:
            return pair_lookup[(first, second)], 1
        return pair_lookup[(second, first)], -1

    d2_gradient = np.zeros((len(pairs), len(pairs)), dtype=float)
    scale = 1.0 / (n_electrons - 1)
    for first in range(n_modes):
        for third in range(n_modes):
            coefficient = scale * gradient[first, third]
            if coefficient == 0.0:
                continue
            for second in range(n_modes):
                row, row_sign = pair_index(first, second)
                col, col_sign = pair_index(third, second)
                if row is not None and col is not None:
                    d2_gradient[row, col] += coefficient * row_sign * col_sign
    d2_gradient = 0.5 * (d2_gradient + d2_gradient.T)
    return symmetric_matrix_gradient_vector(d2_gradient, rows, cols)


def predicted_inverse_variances(
    block: np.ndarray,
    d2_variables: np.ndarray,
    shots_per_shadow: int,
    *,
    probability_floor: float = 0.01,
) -> np.ndarray:
    """Predict Bernoulli inverse variances without reading candidate outcomes."""

    if shots_per_shadow < 1:
        raise ValueError("shots_per_shadow must be positive.")
    if not 0.0 < probability_floor < 0.5:
        raise ValueError("probability_floor must lie between zero and one half.")
    probabilities = np.clip(
        np.asarray(block, dtype=float) @ np.asarray(d2_variables, dtype=float),
        probability_floor,
        1.0 - probability_floor,
    )
    variances = probabilities * (1.0 - probabilities) / shots_per_shadow
    return 1.0 / variances


def _information_update(
    block: np.ndarray,
    inverse_variances: np.ndarray,
) -> np.ndarray:
    weighted = np.sqrt(np.asarray(inverse_variances, dtype=float))[:, None] * block
    return weighted.T @ weighted


def design_information(
    blocks: Sequence[np.ndarray],
    selected_indices: Sequence[int],
    d2_variables: np.ndarray,
    shots_per_shadow: int,
    *,
    ridge_fraction: float = 1e-3,
    probability_floor: float = 0.01,
) -> np.ndarray:
    """Build regularized Fisher information from already acquired bases."""

    if not blocks:
        raise ValueError("At least one candidate block is required.")
    if ridge_fraction <= 0.0:
        raise ValueError("ridge_fraction must be positive.")
    n_variables = blocks[0].shape[1]
    reference_update = _information_update(
        blocks[0],
        predicted_inverse_variances(
            blocks[0],
            d2_variables,
            shots_per_shadow,
            probability_floor=probability_floor,
        ),
    )
    scale = max(float(np.trace(reference_update) / n_variables), 1e-12)
    information = ridge_fraction * scale * np.eye(n_variables)
    for index in selected_indices:
        inverse_variances = predicted_inverse_variances(
            blocks[int(index)],
            d2_variables,
            shots_per_shadow,
            probability_floor=probability_floor,
        )
        information += _information_update(blocks[int(index)], inverse_variances)
    return 0.5 * (information + information.T)


def _candidate_score(
    index: int,
    information: np.ndarray,
    block: np.ndarray,
    inverse_variances: np.ndarray,
    target: np.ndarray,
    target_variance: float,
) -> CandidateDesignScore:
    update = _information_update(block, inverse_variances)
    candidate_information = information + update
    sign_base, logdet_base = np.linalg.slogdet(information)
    sign_candidate, logdet_candidate = np.linalg.slogdet(candidate_information)
    if sign_base <= 0 or sign_candidate <= 0:
        raise np.linalg.LinAlgError("Measurement information is not positive definite.")
    remaining_variance = float(
        target @ np.linalg.solve(candidate_information, target)
    )
    reduction = max(target_variance - remaining_variance, 0.0)
    return CandidateDesignScore(
        index=int(index),
        d_optimal_gain=float(logdet_candidate - logdet_base),
        mcpdft_variance_reduction=reduction,
        mcpdft_fractional_reduction=(
            reduction / target_variance if target_variance > 0.0 else 0.0
        ),
        passes_design_guard=False,
    )


def guarded_mcpdft_choice(
    blocks: Sequence[np.ndarray],
    selected_indices: Sequence[int],
    d2_variables: np.ndarray,
    mcpdft_target: np.ndarray,
    shots_per_shadow: int,
    *,
    design_guard_fraction: float = 0.8,
    ridge_fraction: float = 1e-3,
    probability_floor: float = 0.01,
) -> GuardedDesignChoice:
    """Choose the most MC-PDFT-informative candidate inside a D-optimal guard.

    The pure D-optimal candidate is always inside the guard.  Consequently the
    chosen candidate has at least ``design_guard_fraction`` of the maximum
    log-determinant gain, while its predicted MC-PDFT variance reduction is no
    smaller than that of the pure D-optimal candidate.
    """

    if not 0.0 < design_guard_fraction <= 1.0:
        raise ValueError("design_guard_fraction must lie in (0, 1].")
    selected = tuple(int(index) for index in selected_indices)
    if len(set(selected)) != len(selected):
        raise ValueError("selected_indices must be unique.")
    remaining = [index for index in range(len(blocks)) if index not in set(selected)]
    if not remaining:
        raise ValueError("No unselected candidate remains.")
    variables = np.asarray(d2_variables, dtype=float)
    target = np.asarray(mcpdft_target, dtype=float)
    if variables.shape != target.shape or variables.shape != (blocks[0].shape[1],):
        raise ValueError("Variable and target vectors have incompatible shapes.")

    information = design_information(
        blocks,
        selected,
        variables,
        shots_per_shadow,
        ridge_fraction=ridge_fraction,
        probability_floor=probability_floor,
    )
    target_variance = float(target @ np.linalg.solve(information, target))
    raw_scores = []
    for index in remaining:
        inverse_variances = predicted_inverse_variances(
            blocks[index],
            variables,
            shots_per_shadow,
            probability_floor=probability_floor,
        )
        raw_scores.append(
            _candidate_score(
                index,
                information,
                blocks[index],
                inverse_variances,
                target,
                target_variance,
            )
        )
    maximum_gain = max(score.d_optimal_gain for score in raw_scores)
    threshold = design_guard_fraction * maximum_gain
    scores = tuple(
        CandidateDesignScore(
            index=score.index,
            d_optimal_gain=score.d_optimal_gain,
            mcpdft_variance_reduction=score.mcpdft_variance_reduction,
            mcpdft_fractional_reduction=score.mcpdft_fractional_reduction,
            passes_design_guard=score.d_optimal_gain >= threshold - 1e-12,
        )
        for score in raw_scores
    )
    design_only = max(
        scores,
        key=lambda score: (
            score.d_optimal_gain,
            score.mcpdft_variance_reduction,
            -score.index,
        ),
    )
    guarded = [score for score in scores if score.passes_design_guard]
    chosen = max(
        guarded,
        key=lambda score: (
            score.mcpdft_variance_reduction,
            score.d_optimal_gain,
            -score.index,
        ),
    )
    if chosen.mcpdft_variance_reduction + 1e-12 < design_only.mcpdft_variance_reduction:
        raise RuntimeError("The guarded MC-PDFT choice lost to the design baseline.")
    selected_inverse_variances = predicted_inverse_variances(
        blocks[chosen.index],
        variables,
        shots_per_shadow,
        probability_floor=probability_floor,
    )
    selected_information = information + _information_update(
        blocks[chosen.index], selected_inverse_variances
    )
    selected_target_variance = float(
        target @ np.linalg.solve(selected_information, target)
    )
    current_trace_variance = float(np.trace(np.linalg.inv(information)))
    selected_trace_variance = float(np.trace(np.linalg.inv(selected_information)))
    return GuardedDesignChoice(
        selected_index=chosen.index,
        design_only_index=design_only.index,
        design_guard_fraction=float(design_guard_fraction),
        maximum_d_optimal_gain=float(maximum_gain),
        current_mcpdft_standard_error=float(np.sqrt(max(target_variance, 0.0))),
        selected_mcpdft_standard_error=float(
            np.sqrt(max(selected_target_variance, 0.0))
        ),
        current_d2_trace_standard_error=float(
            np.sqrt(max(current_trace_variance, 0.0))
        ),
        selected_d2_trace_standard_error=float(
            np.sqrt(max(selected_trace_variance, 0.0))
        ),
        selected=chosen,
        design_only=design_only,
        scores=scores,
    )


def guarded_target_subspace_choice(
    blocks: Sequence[np.ndarray],
    selected_indices: Sequence[int],
    d2_variables: np.ndarray,
    target_modes: np.ndarray,
    shots_per_shadow: int,
    *,
    design_guard_fraction: float = 0.2,
    ridge_fraction: float = 1e-3,
    probability_floor: float = 0.01,
    allow_repeats: bool = False,
) -> GuardedDesignChoice:
    """Choose an A-optimal candidate for a frozen target-response subspace."""

    if not 0.0 < design_guard_fraction <= 1.0:
        raise ValueError("design_guard_fraction must lie in (0, 1].")
    selected = tuple(int(index) for index in selected_indices)
    if not allow_repeats and len(set(selected)) != len(selected):
        raise ValueError("selected_indices must be unique.")
    remaining = (
        list(range(len(blocks)))
        if allow_repeats
        else [index for index in range(len(blocks)) if index not in set(selected)]
    )
    if not remaining:
        raise ValueError("No unselected candidate remains.")
    variables = np.asarray(d2_variables, dtype=float)
    modes = np.asarray(target_modes, dtype=float)
    expected_variables = blocks[0].shape[1]
    if variables.shape != (expected_variables,):
        raise ValueError("D2 variables have an incompatible shape.")
    if modes.ndim != 2 or modes.shape[1] != expected_variables or len(modes) == 0:
        raise ValueError("target_modes must be a nonempty mode-by-variable matrix.")

    information = design_information(
        blocks,
        selected,
        variables,
        shots_per_shadow,
        ridge_fraction=ridge_fraction,
        probability_floor=probability_floor,
    )

    def target_variance(matrix: np.ndarray) -> float:
        solved = np.linalg.solve(matrix, modes.T)
        return float(np.sum(modes * solved.T))

    current_variance = target_variance(information)
    sign_base, logdet_base = np.linalg.slogdet(information)
    if sign_base <= 0:
        raise np.linalg.LinAlgError("Measurement information is not positive definite.")
    raw_scores = []
    candidate_information: dict[int, np.ndarray] = {}
    for index in remaining:
        inverse_variances = predicted_inverse_variances(
            blocks[index],
            variables,
            shots_per_shadow,
            probability_floor=probability_floor,
        )
        updated = information + _information_update(blocks[index], inverse_variances)
        candidate_information[index] = updated
        sign, logdet = np.linalg.slogdet(updated)
        if sign <= 0:
            raise np.linalg.LinAlgError("Candidate information is not positive definite.")
        remaining_variance = target_variance(updated)
        reduction = max(current_variance - remaining_variance, 0.0)
        raw_scores.append(
            CandidateDesignScore(
                index=index,
                d_optimal_gain=float(logdet - logdet_base),
                mcpdft_variance_reduction=reduction,
                mcpdft_fractional_reduction=(
                    reduction / current_variance if current_variance > 0.0 else 0.0
                ),
                passes_design_guard=False,
            )
        )
    maximum_gain = max(score.d_optimal_gain for score in raw_scores)
    threshold = design_guard_fraction * maximum_gain
    scores = tuple(
        CandidateDesignScore(
            index=score.index,
            d_optimal_gain=score.d_optimal_gain,
            mcpdft_variance_reduction=score.mcpdft_variance_reduction,
            mcpdft_fractional_reduction=score.mcpdft_fractional_reduction,
            passes_design_guard=score.d_optimal_gain >= threshold - 1e-12,
        )
        for score in raw_scores
    )
    design_only = max(
        scores,
        key=lambda score: (
            score.d_optimal_gain,
            score.mcpdft_variance_reduction,
            -score.index,
        ),
    )
    chosen = max(
        (score for score in scores if score.passes_design_guard),
        key=lambda score: (
            score.mcpdft_variance_reduction,
            score.d_optimal_gain,
            -score.index,
        ),
    )
    selected_information = candidate_information[chosen.index]
    selected_variance = target_variance(selected_information)
    current_trace_variance = float(np.trace(np.linalg.inv(information)))
    selected_trace_variance = float(np.trace(np.linalg.inv(selected_information)))
    return GuardedDesignChoice(
        selected_index=chosen.index,
        design_only_index=design_only.index,
        design_guard_fraction=float(design_guard_fraction),
        maximum_d_optimal_gain=float(maximum_gain),
        current_mcpdft_standard_error=float(
            np.sqrt(max(current_variance, 0.0))
        ),
        selected_mcpdft_standard_error=float(
            np.sqrt(max(selected_variance, 0.0))
        ),
        current_d2_trace_standard_error=float(
            np.sqrt(max(current_trace_variance, 0.0))
        ),
        selected_d2_trace_standard_error=float(
            np.sqrt(max(selected_trace_variance, 0.0))
        ),
        selected=chosen,
        design_only=design_only,
        scores=scores,
    )


def guarded_hamiltonian_mcpdft_choice(
    blocks: Sequence[np.ndarray],
    selected_indices: Sequence[int],
    d2_variables: np.ndarray,
    hamiltonian_target: np.ndarray,
    mcpdft_target: np.ndarray,
    shots_per_shadow: int,
    *,
    design_guard_fraction: float = 0.8,
    hamiltonian_guard_fraction: float = 0.8,
    ridge_fraction: float = 1e-3,
    probability_floor: float = 0.01,
) -> SafeMCPDFTChoice:
    """Maximize MC-PDFT information inside design and Hamiltonian guards.

    The Hamiltonian-only candidate maximizes Hamiltonian variance reduction
    among candidates retaining the requested fraction of the best D-optimal
    gain.  It is always admitted to the second guard.  The final candidate
    maximizes MC-PDFT variance reduction within that set, so its predicted
    MC-PDFT gain cannot be smaller than the Hamiltonian-only baseline's gain.
    """

    if not 0.0 < design_guard_fraction <= 1.0:
        raise ValueError("design_guard_fraction must lie in (0, 1].")
    if not 0.0 < hamiltonian_guard_fraction <= 1.0:
        raise ValueError("hamiltonian_guard_fraction must lie in (0, 1].")
    selected_indices = tuple(int(index) for index in selected_indices)
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("selected_indices must be unique.")
    remaining = [
        index for index in range(len(blocks)) if index not in set(selected_indices)
    ]
    if not remaining:
        raise ValueError("No unselected candidate remains.")
    variables = np.asarray(d2_variables, dtype=float)
    hamiltonian = np.asarray(hamiltonian_target, dtype=float)
    mcpdft = np.asarray(mcpdft_target, dtype=float)
    expected_shape = (blocks[0].shape[1],)
    if any(vector.shape != expected_shape for vector in (variables, hamiltonian, mcpdft)):
        raise ValueError("Variable and target vectors have incompatible shapes.")

    information = design_information(
        blocks,
        selected_indices,
        variables,
        shots_per_shadow,
        ridge_fraction=ridge_fraction,
        probability_floor=probability_floor,
    )
    hamiltonian_variance = float(
        hamiltonian @ np.linalg.solve(information, hamiltonian)
    )
    mcpdft_variance = float(mcpdft @ np.linalg.solve(information, mcpdft))
    sign_base, logdet_base = np.linalg.slogdet(information)
    if sign_base <= 0:
        raise np.linalg.LinAlgError("Measurement information is not positive definite.")

    raw = []
    candidate_information: dict[int, np.ndarray] = {}
    for index in remaining:
        inverse_variances = predicted_inverse_variances(
            blocks[index],
            variables,
            shots_per_shadow,
            probability_floor=probability_floor,
        )
        updated = information + _information_update(blocks[index], inverse_variances)
        candidate_information[index] = updated
        sign, logdet = np.linalg.slogdet(updated)
        if sign <= 0:
            raise np.linalg.LinAlgError("Candidate information is not positive definite.")
        h_remaining = float(hamiltonian @ np.linalg.solve(updated, hamiltonian))
        m_remaining = float(mcpdft @ np.linalg.solve(updated, mcpdft))
        h_reduction = max(hamiltonian_variance - h_remaining, 0.0)
        m_reduction = max(mcpdft_variance - m_remaining, 0.0)
        raw.append(
            MultiTargetCandidateScore(
                index=index,
                d_optimal_gain=float(logdet - logdet_base),
                hamiltonian_variance_reduction=h_reduction,
                hamiltonian_fractional_reduction=(
                    h_reduction / hamiltonian_variance
                    if hamiltonian_variance > 0.0
                    else 0.0
                ),
                mcpdft_variance_reduction=m_reduction,
                mcpdft_fractional_reduction=(
                    m_reduction / mcpdft_variance if mcpdft_variance > 0.0 else 0.0
                ),
                passes_design_guard=False,
                passes_hamiltonian_guard=False,
            )
        )

    maximum_d_gain = max(score.d_optimal_gain for score in raw)
    design_threshold = design_guard_fraction * maximum_d_gain
    design_guarded = [
        score for score in raw if score.d_optimal_gain >= design_threshold - 1e-12
    ]
    maximum_h_gain = max(
        score.hamiltonian_variance_reduction for score in design_guarded
    )
    hamiltonian_threshold = hamiltonian_guard_fraction * maximum_h_gain
    scores = tuple(
        MultiTargetCandidateScore(
            index=score.index,
            d_optimal_gain=score.d_optimal_gain,
            hamiltonian_variance_reduction=score.hamiltonian_variance_reduction,
            hamiltonian_fractional_reduction=score.hamiltonian_fractional_reduction,
            mcpdft_variance_reduction=score.mcpdft_variance_reduction,
            mcpdft_fractional_reduction=score.mcpdft_fractional_reduction,
            passes_design_guard=score.d_optimal_gain >= design_threshold - 1e-12,
            passes_hamiltonian_guard=(
                score.d_optimal_gain >= design_threshold - 1e-12
                and score.hamiltonian_variance_reduction
                >= hamiltonian_threshold - 1e-12
            ),
        )
        for score in raw
    )
    design_only = max(
        scores,
        key=lambda score: (
            score.d_optimal_gain,
            score.hamiltonian_variance_reduction,
            score.mcpdft_variance_reduction,
            -score.index,
        ),
    )
    hamiltonian_only = max(
        (score for score in scores if score.passes_design_guard),
        key=lambda score: (
            score.hamiltonian_variance_reduction,
            score.d_optimal_gain,
            score.mcpdft_variance_reduction,
            -score.index,
        ),
    )
    chosen = max(
        (score for score in scores if score.passes_hamiltonian_guard),
        key=lambda score: (
            score.mcpdft_variance_reduction,
            score.hamiltonian_variance_reduction,
            score.d_optimal_gain,
            -score.index,
        ),
    )
    if chosen.mcpdft_variance_reduction + 1e-12 < hamiltonian_only.mcpdft_variance_reduction:
        raise RuntimeError("MC-PDFT choice lost to the Hamiltonian-safe baseline.")
    selected_information = candidate_information[chosen.index]
    selected_h_variance = float(
        hamiltonian @ np.linalg.solve(selected_information, hamiltonian)
    )
    selected_m_variance = float(
        mcpdft @ np.linalg.solve(selected_information, mcpdft)
    )
    return SafeMCPDFTChoice(
        selected_index=chosen.index,
        hamiltonian_only_index=hamiltonian_only.index,
        design_only_index=design_only.index,
        design_guard_fraction=float(design_guard_fraction),
        hamiltonian_guard_fraction=float(hamiltonian_guard_fraction),
        maximum_d_optimal_gain=float(maximum_d_gain),
        maximum_hamiltonian_variance_reduction=float(maximum_h_gain),
        current_hamiltonian_standard_error=float(
            np.sqrt(max(hamiltonian_variance, 0.0))
        ),
        selected_hamiltonian_standard_error=float(
            np.sqrt(max(selected_h_variance, 0.0))
        ),
        current_mcpdft_standard_error=float(np.sqrt(max(mcpdft_variance, 0.0))),
        selected_mcpdft_standard_error=float(
            np.sqrt(max(selected_m_variance, 0.0))
        ),
        selected=chosen,
        hamiltonian_only=hamiltonian_only,
        design_only=design_only,
        scores=scores,
    )
