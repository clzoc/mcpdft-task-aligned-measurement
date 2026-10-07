#!/usr/bin/env python3
"""MindQuantum circuit-sampling PoC for the guard15 N2 bond-scan frames.

One geometry (N2/cc-pVDZ CAS(10e,8o), R=1.10 A) and one real O(8) frame from
the frozen scan pool (seed 20260716):

  classical : pyscf-rotated CI probabilities used by AcquisitionOracle, the
              sampler the scan campaign actually calls.
  circuit   : |CASCI> statevector --[Givens gates of the frame]-- measure,
              run on the MindQuantum mqvector simulator and sampled.

Both use the same 16-qubit determinant basis:
    basis_state = alpha_mask | (beta_mask << 8),  qubit q = bit q.
MindQuantum conventions verified for 0.12.0: statevector/sample index has
qubit 0 as the least significant bit; for ``UnivMathGate.on([a, b])`` the
local 2-qubit index is ``n_a + 2*n_b`` (first qubit least significant).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
PROTOCOL = ANALYSIS / "mcpdft_measurement_revision" / "guarded_mcpdft_shadow_protocol"
for path in (
    PROTOCOL / "code",
    PROTOCOL / "code" / "vendor",
    PROTOCOL / "tools",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from constrained_shadow import build_n2_reference, random_orthogonal_rotations  # noqa: E402
from run_c2_ftpbe_shot_reallocation import AcquisitionOracle  # noqa: E402

N_QUBITS = 16
N_SPATIAL = 8
FRAME_INDEX = 0
SHOTS = 200_000
SEED = 20260716
BOND = 1.10


def frame_pool() -> np.ndarray:
    base = random_orthogonal_rotations(N_SPATIAL, 30, SEED)
    return np.asarray([np.stack((u, u)) for u in base])


def flat_to_state_index(reference) -> np.ndarray:
    from pyscf.fci import cistring

    alpha = np.asarray(
        cistring.make_strings(range(N_SPATIAL), reference.n_alpha), dtype=np.uint64
    )
    beta = np.asarray(
        cistring.make_strings(range(N_SPATIAL), reference.n_beta), dtype=np.uint64
    )
    return (
        (alpha[:, None] | (beta[None, :] << np.uint64(N_SPATIAL)))
        .reshape(-1)
        .astype(np.int64)
    )


def adjacent_givens(u: np.ndarray):
    """U = R_1^T ... R_m^T D with adjacent real Givens rotations R_k."""
    a = np.array(u, dtype=float)
    n = a.shape[0]
    steps = []
    for column in range(n):
        for row in range(n - 1, column, -1):
            if abs(a[row, column]) <= 1e-15:
                continue
            j, i = row - 1, row
            r = float(np.hypot(a[j, column], a[i, column]))
            c = a[j, column] / r
            s = -a[i, column] / r
            g = np.eye(n)
            g[j, j] = c
            g[j, i] = -s
            g[i, j] = s
            g[i, i] = c
            a = g @ a
            steps.append((j, float(np.arctan2(s, c))))
    diagonal = np.sign(np.diag(a)).astype(float)
    return steps, diagonal, float(np.max(np.abs(a - np.diag(diagonal))))


def givens_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float
    )


def _little_endian_tensor(state: np.ndarray) -> np.ndarray:
    """Reshape with axis k = qubit k (axis 0 = LSB) instead of C-order."""
    return state.reshape([2] * N_QUBITS).transpose(
        tuple(range(N_QUBITS - 1, -1, -1))
    )


def _from_little_endian_tensor(tensor: np.ndarray) -> np.ndarray:
    return tensor.transpose(tuple(range(N_QUBITS - 1, -1, -1))).reshape(-1)


def apply_adjacent_gate(state: np.ndarray, p: int, matrix: np.ndarray) -> np.ndarray:
    """2-qubit gate on qubits (p, p+1); matrix index = 2*n_p + n_{p+1}."""
    tensor = _little_endian_tensor(state)
    tensor = np.moveaxis(tensor, [p, p + 1], [0, 1])
    shape = tensor.shape
    tensor = (matrix @ tensor.reshape(4, -1)).reshape(shape)
    tensor = np.moveaxis(tensor, [0, 1], [p, p + 1])
    return _from_little_endian_tensor(tensor)


def z_gate(state: np.ndarray, q: int) -> np.ndarray:
    tensor = _little_endian_tensor(state)
    tensor = np.moveaxis(tensor, q, 0)
    tensor[1] *= -1.0
    return _from_little_endian_tensor(np.moveaxis(tensor, 0, q))


def apply_frame_numpy(state: np.ndarray, rotation: np.ndarray, spin: int):
    """Apply W(U^T), the pyscf transform_ci_for_orbital_rotation convention."""
    steps, diagonal, residual = adjacent_givens(rotation)
    out = state.copy()
    offset = spin * N_SPATIAL
    for j, theta in steps:
        out = apply_adjacent_gate(out, offset + j, givens_matrix(-theta))
    for q in range(N_SPATIAL):  # reflection parity, det(U) = -1 frames
        if diagonal[q] < 0:
            out = z_gate(out, offset + q)
    return out, residual


def indicator_matrix(pairs) -> np.ndarray:
    states = np.arange(1 << N_QUBITS, dtype=np.uint64)
    return np.column_stack(
        [
            ((states >> np.uint64(first)) & 1)
            * ((states >> np.uint64(second)) & 1)
            for first, second in pairs
        ]
    ).astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=FRAME_INDEX)
    parser.add_argument("--shots", type=int, default=SHOTS)
    args = parser.parse_args()
    frame_index, shots = args.frame, args.shots
    report = {"frame_index": frame_index, "shots": shots, "seed": SEED, "bond": BOND}
    started = time.time()
    reference = build_n2_reference(
        bond_length=BOND, active_electrons=10, active_orbitals=8
    )
    frames = frame_pool()
    oracle = AcquisitionOracle(reference, frames, np.zeros((30, 1)), 1, 0)
    pyscf_probabilities = oracle._frame_probabilities(frame_index)
    state_index = flat_to_state_index(reference)
    pyscf_by_basis = np.zeros(1 << N_QUBITS)
    pyscf_by_basis[state_index] = pyscf_probabilities
    report["logical_build_seconds"] = round(time.time() - started, 3)
    report["pyscf_probability_sum"] = float(np.sum(pyscf_by_basis))

    initial = np.asarray(reference.statevector, dtype=complex)
    state, residual_up = apply_frame_numpy(initial, frames[frame_index][0], 0)
    state, residual_dn = apply_frame_numpy(state, frames[frame_index][1], 1)
    numpy_probabilities = np.abs(state) ** 2
    report["givens_decomposition_residual"] = max(residual_up, residual_dn)
    report["numpy_vs_pyscf_max_abs"] = float(
        np.max(np.abs(numpy_probabilities - pyscf_by_basis))
    )
    report["numpy_vs_pyscf_tv"] = float(
        0.5 * np.sum(np.abs(numpy_probabilities - pyscf_by_basis))
    )

    pairs = reference.pairs[:8]
    indicators = indicator_matrix(pairs)
    report["pair_moments_exact_pyscf"] = (pyscf_by_basis @ indicators).tolist()

    from mindquantum import __version__ as mq_version
    from mindquantum.core.circuit import Circuit
    from mindquantum.core.gates import Measure, UnivMathGate, Z
    from mindquantum.simulator import Simulator

    sim = Simulator("mqvector", N_QUBITS)
    steps, diagonal, _ = adjacent_givens(frames[frame_index][0])
    circuit = Circuit()
    for j, theta in steps:
        # MindQuantum local index of .on([j, j+1]) is n_j + 2*n_{j+1},
        # so the p-MSB Givens matrix must be mirrored: M(theta) -> M(-theta).
        matrix = givens_matrix(theta)
        circuit.append(UnivMathGate(f"G{j}", matrix).on([j, j + 1]))
        circuit.append(
            UnivMathGate(f"G{j}d", matrix).on([N_SPATIAL + j, N_SPATIAL + j + 1])
        )
    for q in range(N_SPATIAL):  # reflection parity, det(U) = -1 frames
        if diagonal[q] < 0:
            circuit.append(Z.on(q))
            circuit.append(Z.on(N_SPATIAL + q))

    sim.set_qs(initial)
    sim.apply_circuit(circuit)
    mq_state = np.asarray(sim.get_qs())
    report["mindquantum"] = {"version": mq_version}
    report["mindquantum"]["statevector_max_abs_vs_numpy"] = float(
        np.max(np.abs(mq_state - state))
    )
    mq_probabilities = np.abs(mq_state) ** 2
    report["mindquantum"]["statevector_tv_vs_pyscf"] = float(
        0.5 * np.sum(np.abs(mq_probabilities - pyscf_by_basis))
    )

    measure = Circuit([Measure().on(q) for q in range(N_QUBITS)])
    sample_started = time.time()
    samples = sim.sampling(measure, shots=shots, seed=SEED % (2**23))
    report["mindquantum"]["sampling_seconds"] = round(time.time() - sample_started, 3)
    counts = np.zeros(1 << N_QUBITS)
    for key, count in samples.bit_string_data.items():
        counts[int(key, 2)] += count
    empirical = counts / shots
    report["mindquantum"]["shots_tv_vs_pyscf"] = float(
        0.5 * np.sum(np.abs(empirical - pyscf_by_basis))
    )
    report["mindquantum"]["shots_tv_vs_numpy"] = float(
        0.5 * np.sum(np.abs(empirical - numpy_probabilities))
    )
    report["mindquantum"]["pair_moments_shots"] = (
        empirical @ indicators
    ).tolist()
    report["mindquantum"]["pair_moments_deviation"] = (
        (empirical @ indicators) - (pyscf_by_basis @ indicators)
    ).tolist()
    report["total_seconds"] = round(time.time() - started, 3)

    (HERE / f"poc_report_f{frame_index}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
