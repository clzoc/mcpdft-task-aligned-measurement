#!/usr/bin/env python3
"""Export the frame circuits (Givens gate lists) used by the g1.0/g0.2 scan.

For every geometry and both arms, the MindQuantum circuit of each selected
frame is reconstructed from the frozen rotation pool and written as an
explicit gate list (Givens angle per adjacent pair, per spin, diagonal Z
signs), together with the Wukong-180-2 calibrated gate/readout rates at both
gate-noise scales.  The exact initial statevector is not duplicated: it is
rebuilt by the packaged engine context (same code path as sampling).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "code" / "mindquantum_poc"))

import numpy as np  # noqa: E402

import circuit_sampler as cs  # noqa: E402
import run_scan  # noqa: E402
import wukong_noise  # noqa: E402


def gate_rates(model, scale):
    rates = []
    for j in range(cs.N_SPATIAL - 1):
        for spin in (0, 1):
            offset = spin * cs.N_SPATIAL
            edge = j if spin == 0 else cs.N_SPATIAL + j
            rates.append(
                {
                    "j": j,
                    "spin": spin,
                    "logical_qubits": [offset + j, offset + j + 1],
                    "edge_index": edge,
                    "depolarizing_rate": cs.gate_error(model, offset + j, edge, scale),
                }
            )
    return rates


def main() -> None:
    model = wukong_noise.load_model()
    payload = {
        "description": (
            "Frame circuits for the N2 bond scan: each frame is an adjacent "
            "Givens decomposition U = R_m ... R_1 D of the 8x8 same-spin "
            "rotation; every Givens acts on logical qubits (j, j+1) of one "
            "spin block (16 logical qubits total) and is followed by "
            "single-qubit depolarizing channels with the calibrated rates "
            "below.  Readout BitFlip channels use readout_per_qubit."
        ),
        "noise_model": {
            "source_sha256": model["source_sha256"],
            "logical_to_physical": model["logical_to_physical"],
            "physical_1q_error": model["physical_1q_error"],
            "physical_cz_error": model["physical_cz_error"],
            "physical_readout_error": model["physical_readout_error"],
            "convention": model["convention"],
            "gate_scale_note": (
                "gate_scale multiplies only the per-Givens depolarizing rate; "
                "readout BitFlip rates are not scaled"
            ),
        },
        "gate_rates": {
            "1.0": gate_rates(model, 1.0),
            "0.2": gate_rates(model, 0.2),
        },
        "readout_per_qubit": model["readout_per_qubit"],
        "initial_state": {
            "description": "exact N2 statevector (2^16) from the engine context",
            "stored": False,
            "rebuild": "engine.context / circuit_sampler.sample_histogram(initial=c.exact.statevector)",
        },
        "tags": {},
    }
    for tag in run_scan.SYSTEMS:
        _d, c = run_scan.build_context(tag)
        rotations = np.asarray(c.rotations)
        entry = {
            "tag": tag,
            "bond_angstrom": run_scan.SYSTEMS[tag],
            "exact_energy": float(c.exact.exact_energy),
            "arms": {},
        }
        for arm in run_scan.ARMS:
            indices, shots = run_scan.plan_for(c, arm)
            frames = []
            for position, frame in enumerate(indices):
                frame = int(frame)
                steps, diagonal = cs.adjacent_givens(np.real(rotations[frame][0]))
                frames.append(
                    {
                        "frame": frame,
                        "shots": int(shots[position]),
                        "givens": [
                            {"qubits": [j, j + 1], "theta": float(theta)}
                            for j, theta in steps
                        ],
                        "diagonal_signs": [int(x) for x in diagonal],
                    }
                )
            entry["arms"][arm] = {
                "selected_frames": [int(f) for f in indices],
                "total_shots": int(shots.sum()),
                "circuits": frames,
            }
        payload["tags"][tag] = entry
        print(f"{tag}: guard15={len(entry['arms']['guard15_equal']['circuits'])} "
              f"uniform={len(entry['arms']['uniform']['circuits'])}", flush=True)

    out = PKG / "circuits" / "givens_circuits_g1.0_g0.2.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
