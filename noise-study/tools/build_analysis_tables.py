#!/usr/bin/env python3
"""Build the combined g1.0 / g0.2 analysis tables from the packaged results."""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
OUT = PKG / "analysis"
OUT.mkdir(exist_ok=True)

SCALES = {"1.0": PKG / "code/mindquantum_poc/results",
          "0.2": PKG / "code/mindquantum_poc/results_gate020"}

RECORD_COLS = [
    "system", "bond_angstrom", "arm", "variant", "gate_scale",
    "h_error_meh", "f_error_meh", "d2_error", "density_error",
    "contact_error", "gamma_error", "acceptance_post", "prime_seconds",
]

records = []
for scale, directory in SCALES.items():
    for path in sorted(directory.glob("*/variants/*.json")):
        rec = json.loads(path.read_text())
        rec.setdefault("gate_scale", float(scale))
        records.append({col: rec.get(col) for col in RECORD_COLS})

with (OUT / "records_g1.0_g0.2.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=RECORD_COLS)
    writer.writeheader()
    writer.writerows(records)

ARMS = ("guard15_equal", "uniform")
VARIANT_ORDER = [
    "exact", "clean", "raw", "post", "rem",
    "rem_lin", "rem_post", "rem_lin_post",
]
rows = []
for scale in SCALES:
    for arm in ARMS:
        for variant in VARIANT_ORDER:
            sel = [
                r for r in records
                if r["gate_scale"] == float(scale)
                and r["arm"] == arm
                and r["variant"] == variant
            ]
            if not sel:
                continue
            rows.append(
                {
                    "gate_scale": scale,
                    "arm": arm,
                    "variant": variant,
                    "n": len(sel),
                    "mean_h_error_meh": st.mean(r["h_error_meh"] for r in sel),
                    "mean_f_error_meh": st.mean(r["f_error_meh"] for r in sel),
                    "mean_d2_error": st.mean(r["d2_error"] for r in sel),
                    "mean_density_error": st.mean(r["density_error"] for r in sel),
                    "mean_contact_error": st.mean(r["contact_error"] for r in sel),
                    "mean_gamma_error": st.mean(r["gamma_error"] for r in sel),
                }
            )

cols = list(rows[0])
with (OUT / "variant_means_g1.0_vs_g0.2.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=cols)
    writer.writeheader()
    writer.writerows(rows)

print(f"records={len(records)} rows={len(rows)} -> {OUT}")
