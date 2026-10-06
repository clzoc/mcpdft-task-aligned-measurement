"""Qiskit validation circuits for particle-number-conserving orbital frames."""

from __future__ import annotations

import numpy as np


def fci_statevector(
    ci: np.ndarray, n_modes: int, n_electrons: tuple[int, int]
) -> np.ndarray:
    """Embed a PySCF alpha/beta CI array in Jordan-Wigner qubit order."""
    from pyscf.fci import cistring

    strings_alpha = cistring.make_strings(range(n_modes), n_electrons[0])
    strings_beta = cistring.make_strings(range(n_modes), n_electrons[1])
    state = np.zeros(2 ** (2 * n_modes), dtype=complex)
    coefficients = np.asarray(ci)
    for alpha_index, alpha_string in enumerate(strings_alpha):
        for beta_index, beta_string in enumerate(strings_beta):
            basis_index = int(alpha_string) | (int(beta_string) << n_modes)
            state[basis_index] = coefficients[alpha_index, beta_index]
    norm = np.linalg.norm(state)
    if norm <= 0.0:
        raise ValueError("CI vector has zero norm")
    return state / norm


def orbital_rotation_circuit(frame: np.ndarray):
    """Return the Qiskit-Nature circuit that measures occupations in ``frame``.

    Spatial directions are rows of ``frame``. The transpose is passed to the
    Bogoliubov circuit because measuring qubit i after evolving the state probes
    column i of its creation-operator transformation. Alpha and beta blocks are
    rotated identically and never mixed.
    """
    from qiskit_nature.second_q.circuit.library import BogoliubovTransform

    rotation = np.asarray(frame, dtype=float)
    if rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
        raise ValueError("frame must be a square matrix")
    if np.max(np.abs(rotation @ rotation.T - np.eye(len(rotation)))) > 1e-9:
        raise ValueError("frame must be orthonormal")
    zeros = np.zeros_like(rotation)
    spin_orbital_rotation = np.block(
        [[rotation.T, zeros], [zeros, rotation.T]]
    ).astype(complex)
    return BogoliubovTransform(spin_orbital_rotation)


def statevector_frame_moments(
    state: np.ndarray, frame: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact pair and spin-summed occupations from the Qiskit circuit."""
    from qiskit.quantum_info import Statevector

    n_modes = len(frame)
    evolved = Statevector(np.asarray(state, dtype=complex)).evolve(
        orbital_rotation_circuit(frame)
    )
    probability = np.abs(evolved.data) ** 2
    indices = np.arange(len(probability), dtype=np.int64)
    alpha = np.stack([(indices >> mode) & 1 for mode in range(n_modes)], axis=1)
    beta = np.stack(
        [(indices >> (n_modes + mode)) & 1 for mode in range(n_modes)], axis=1
    )
    pair = probability @ (alpha * beta)
    total = probability @ (alpha + beta)
    return np.asarray(pair), np.asarray(total)

