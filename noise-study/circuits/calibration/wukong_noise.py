#!/usr/bin/env python3
"""Wukong-180-2 calibrated noise model for the 16-qubit N2 frame circuits.

The calibration workbook is the same one used by the repository hardware
validation (``hardware/本源悟空180-2芯片数据.xlsx``).  Sixteen physical qubits are
chosen as a minimum-cost path so that every logical Givens pair (j, j+1) maps to
a calibrated chip edge; logical qubit q maps to ``path[q]``.

Per frame gate, the effective model folds the native-gate decomposition of a
4x4 Givens gate into the calibrated rates:

    2q depolarizing (both qubits) : p2 = 1 - (1 - p_cz)^2      (two CZ)
    1q depolarizing (each qubit)  : p1 = 1 - (1 - p_1q)^2      (two 1q gates)
    readout (before measurement)  : BitFlipChannel with p_read = 1 - F_read

The workbook stores per-qubit readout fidelity F (column F), 1q fidelity
(column I) and edge ``CZ<i>_<j>:fidelity`` entries (column J).  The path and
the error probabilities are frozen in ``wukong_path16.json`` with the source
sha256.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKBOOK = (
    HERE.parent
    / "mcpdft_measurement_revision"
    / "hardware_validation"
    / "hardware"
    / "本源悟空180-2芯片数据.xlsx"
)
CACHE = HERE / "wukong_path16.json"

PATH = (64, 54, 63, 72, 82, 73, 83, 92, 101, 110, 119, 109, 118, 108, 99, 90)
_ROW = re.compile(r"q(\d+)")
_CZ = re.compile(r"CZ(\d+)_(\d+):([\d.]+)")


def _parse_workbook(path: Path) -> dict:
    import openpyxl

    sheet = openpyxl.load_workbook(path, read_only=True, data_only=True)["qubit"]
    single, readout, cz = {}, {}, {}
    for row in list(sheet.iter_rows(values_only=True))[1:]:
        if not row or row[0] is None:
            continue
        match = _ROW.fullmatch(str(row[0]).strip())
        if match is None:
            continue
        qubit = int(match.group(1))
        readout[qubit] = 1.0 - float(row[5])
        single[qubit] = 1.0 - float(row[8])
        for entry in (str(row[9]) if row[9] else "").split(";"):
            edge = _CZ.fullmatch(entry.strip())
            if edge is None:
                continue
            first, second = int(edge.group(1)), int(edge.group(2))
            cz[tuple(sorted((first, second)))] = 1.0 - float(edge.group(3))
    return {"single": single, "readout": readout, "cz": cz}


def build_model() -> dict:
    raw = _parse_workbook(WORKBOOK)
    for qubit in PATH:
        if qubit not in raw["single"]:
            raise RuntimeError(f"Qubit {qubit} missing from the calibration.")
    for first, second in zip(PATH, PATH[1:]):
        edge = tuple(sorted((first, second)))
        if edge not in raw["cz"]:
            raise RuntimeError(f"Path edge {edge} is not calibrated.")
    p1 = np.array([raw["single"][q] for q in PATH])
    p2 = np.array(
        [raw["cz"][tuple(sorted((a, b)))] for a, b in zip(PATH, PATH[1:])]
    )
    read = np.array([raw["readout"][q] for q in PATH])
    # Fold the native decomposition: one Givens = two CZ + two physical 1q gates
    # per qubit (RZ layers are virtual and are not assigned an error budget).
    # MindQuantum 0.12 only offers single-qubit channels, so the CZ error is
    # approximated by depolarizing each participating qubit with the edge rate.
    return {
        "source": str(WORKBOOK),
        "source_sha256": hashlib.sha256(WORKBOOK.read_bytes()).hexdigest(),
        "logical_to_physical": list(PATH),
        "physical_1q_error": p1.tolist(),
        "physical_readout_error": read.tolist(),
        "physical_cz_error": p2.tolist(),
        "readout_per_qubit": read.tolist(),
        "convention": (
            "p_gate(qubit, edge) = 1-(1-p_cz)^2 (1-p_1q)^2 applied once per "
            "Givens to each of the two participating logical qubits; "
            "readout BitFlipChannel(p_read) before measurement; the two-qubit "
            "CZ error is approximated by per-qubit depolarizing because the "
            "0.12.0 channels are single-qubit"
        ),
    }


def load_model() -> dict:
    if CACHE.exists():
        payload = json.loads(CACHE.read_text())
        if payload.get("source_sha256") == hashlib.sha256(
            WORKBOOK.read_bytes()
        ).hexdigest():
            return payload
    payload = build_model()
    CACHE.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    model = load_model()
    print(json.dumps(model, indent=2))
