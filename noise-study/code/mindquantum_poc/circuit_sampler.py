#!/usr/bin/env python3
"""MindQuantum frame-circuit sampler with the Wukong-180-2 calibrated noise."""
from __future__ import annotations

import os

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from wukong_noise import load_model  # noqa: E402

N_QUBITS = 16
N_SPATIAL = 8


def _givens_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float
    )


def adjacent_givens(u: np.ndarray):
    """U = R_1^T ... R_m^T D with adjacent Givens rotations (see poc_compare)."""
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
    return steps, diagonal


def gate_error(
    model: dict, logical_qubit: int, edge_index: int, gate_scale: float = 1.0
) -> float:
    """Per-Givens depolarizing rate for one participating logical qubit."""
    p1 = model["physical_1q_error"][logical_qubit]
    p2 = model["physical_cz_error"][edge_index]
    rate = 1.0 - (1.0 - p2) ** 2 * (1.0 - p1) ** 2
    return min(gate_scale * rate, 0.9)


def frame_circuit(
    rotation: np.ndarray, model: dict, noisy: bool, gate_scale: float = 1.0
):
    from mindquantum.core.circuit import Circuit
    from mindquantum.core.gates import (
        BitFlipChannel,
        DepolarizingChannel,
        Measure,
        UnivMathGate,
        Z,
    )

    steps, diagonal = adjacent_givens(rotation)
    circuit = Circuit()
    for j, theta in steps:
        matrix = _givens_matrix(theta)
        for spin in (0, 1):
            offset = spin * N_SPATIAL
            circuit.append(
                UnivMathGate(f"G{spin}_{j}", matrix).on([offset + j, offset + j + 1])
            )
            if noisy:
                edge = j if spin == 0 else N_SPATIAL + j
                for qubit in (offset + j, offset + j + 1):
                    rate = gate_error(model, qubit, edge, gate_scale)
                    if rate > 0.0:
                        circuit.append(
                            DepolarizingChannel(rate).on(qubit)
                        )
    for qubit in range(N_SPATIAL):
        if diagonal[qubit] < 0:
            circuit.append(Z.on(qubit))
            circuit.append(Z.on(N_SPATIAL + qubit))
    if noisy:
        for qubit in range(N_QUBITS):
            rate = model["readout_per_qubit"][qubit]
            if rate > 0.0:
                circuit.append(BitFlipChannel(rate).on(qubit))
    for qubit in range(N_QUBITS):
        circuit.append(Measure().on(qubit))
    return circuit


def sample_histogram(
    initial: np.ndarray,
    rotation: np.ndarray,
    shots: int,
    noisy: bool,
    seed: int,
    model: dict | None = None,
    gate_scale: float = 1.0,
) -> np.ndarray:
    """Return integer counts over the 2^16 measurement outcomes."""
    from mindquantum.simulator import Simulator

    model = load_model() if model is None else model
    circuit = frame_circuit(rotation, model, noisy, gate_scale)
    simulator = Simulator("mqvector", N_QUBITS)
    simulator.set_qs(np.asarray(initial, dtype=complex))
    result = simulator.sampling(
        circuit, shots=int(shots), seed=int(seed) % (2**23)
    )
    counts = np.zeros(1 << N_QUBITS, dtype=np.int64)
    for key, count in result.bit_string_data.items():
        counts[int(key, 2)] += int(count)
    return counts


def indicator_matrix(pairs) -> np.ndarray:
    states = np.arange(1 << N_QUBITS, dtype=np.uint64)
    return np.column_stack(
        [
            ((states >> np.uint64(first)) & 1)
            * ((states >> np.uint64(second)) & 1)
            for first, second in pairs
        ]
    ).astype(np.uint8)


def readout_invert(probability: np.ndarray, model: dict) -> np.ndarray:
    """Tensor-product inverse of the symmetric per-qubit readout confusion."""
    corrected = readout_apply(probability, model)
    corrected = np.maximum(corrected, 0.0)
    return corrected / float(np.sum(corrected))


def readout_apply(probability: np.ndarray, model: dict) -> np.ndarray:
    """Linear (unclipped, unnormalized) tensor-product readout inversion."""
    tensor = np.asarray(probability, dtype=float).reshape((2,) * N_QUBITS)
    for qubit in range(N_QUBITS):
        p = model["readout_per_qubit"][qubit]
        inverse = np.linalg.inv(np.array([[1.0 - p, p], [p, 1.0 - p]]))
        axis = N_QUBITS - 1 - qubit
        tensor = np.tensordot(inverse, tensor, axes=(1, axis))
        tensor = np.moveaxis(tensor, 0, axis)
    return tensor.reshape(-1)


def sector_mask() -> np.ndarray:
    states = np.arange(1 << N_QUBITS, dtype=np.uint64)
    return np.asarray(
        [
            int(state & np.uint64((1 << N_SPATIAL) - 1)).bit_count() == 5
            and (int(state) >> N_SPATIAL).bit_count() == 5
            for state in states
        ],
        dtype=bool,
    )


def postselect(probability: np.ndarray, mask: np.ndarray):
    accepted = float(np.sum(probability[mask]))
    if accepted <= 0.0:
        raise RuntimeError("Post-selection rejected every outcome.")
    selected = np.zeros_like(probability)
    selected[mask] = probability[mask] / accepted
    return selected, accepted


if __name__ == "__main__":
    import time

    model = load_model()
    rng = np.random.default_rng(0)
    rotation = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    initial = np.zeros(1 << N_QUBITS, dtype=complex)
    initial[0] = 1.0
    for noisy in (False, True):
        start = time.time()
        counts = sample_histogram(initial, rotation, 2000, noisy, 12345, model)
        print(
            f"noisy={noisy}: 2000 shots in {time.time() - start:.2f}s, "
            f"{int(counts.sum())} outcomes"
        )
