#!/usr/bin/env python3
"""Plot an exported frame circuit with MindQuantum's colored SVG style.

Rebuilds one frame circuit from ``circuits/givens_circuits_g1.0_g0.2.json``
— the same gate sequence ``circuit_sampler.frame_circuit`` produces — and
renders it with ``Circuit.svg(style="official")`` (colored mode), converting
the SVG to PDF and PNG with cairosvg.  Two figures are written:

  ideal  : Givens gates (both spin blocks) + parity Z + measurement
  noisy  : ideal + per-Givens DepolarizingChannel (rates from the frozen
           Wukong model at ``--gate-scale``) + per-qubit readout BitFlip
  block  : one Givens block decomposed into elementary gates
           (2 CNOT + CRY(-2θ) = 4 CNOT + 2 RY), numerically verified

All 30 frames share the same gate names, so frame figures differ only in the
parity-Z positions; the default (first guard15 frame of n2_r080) represents
every frame of both arms.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "code" / "mindquantum_poc"))

import numpy as np  # noqa: E402

import circuit_sampler as cs  # noqa: E402
import wukong_noise  # noqa: E402

N_QUBITS = cs.N_QUBITS
N_SPATIAL = cs.N_SPATIAL
CIRCUITS_JSON = PKG / "circuits" / "givens_circuits_g1.0_g0.2.json"


def givens_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float
    )


def block_schematic() -> str:
    """SVG document showing one Givens block's elementary-gate decomposition.

    G(θ) on (q_j, q_j+1)  =  CNOT(j->j+1) · CRY(j+1->j, -2θ) · CNOT(j->j+1)
    = 4 CNOT + 2 RY(±θ), verified numerically below.  In the figure q0 is the
    lower-index wire q_j and q1 is q_j+1 (same in every spin block).
    """
    from mindquantum.core.circuit import Circuit
    from mindquantum.core.gates import RY, UnivMathGate, X
    from mindquantum.io.display._config import _svg_config_official as cfg
    from mindquantum.io.display.circuit_svg_drawer import (
        SVGCircuit,
        SVGContainer,
        Rect,
        Text,
        box,
    )

    theta = 0.6  # placeholder matrix for the labeled block; checked numerically
    block = Circuit([UnivMathGate("G0_j(θ)", givens_matrix(theta)).on([0, 1])])
    symbolic = Circuit([X.on(1, 0), RY({"θ": -2}).on(0, 1), X.on(1, 0)])
    elementary = Circuit(
        [
            X.on(1, 0),
            RY({"θ": -1}).on(0),
            X.on(0, 1),
            RY("θ").on(0),
            X.on(0, 1),
            X.on(1, 0),
        ]
    )
    for circuit in (symbolic, elementary):
        diff = np.max(np.abs(circuit.matrix(pr={"θ": theta}) - givens_matrix(theta)))
        assert diff < 1e-12, f"block decomposition mismatch: {diff}"

    parts = [SVGCircuit(c, dict(cfg), np.inf) for c in (block, symbolic, elementary)]
    root = SVGContainer()
    gap = 60.0
    v_center = (box(parts[0])["top"] + box(parts[0])["bottom"]) / 2
    x, eq_xs = 0.0, []
    for i, part in enumerate(parts):
        b = box(part)
        part.shift(x - b["left"], v_center - (b["top"] + b["bottom"]) / 2)
        root.add(part)
        x += b["width"]
        if i < len(parts) - 1:
            eq_xs.append(x + gap / 2)
            x += gap
    for eq_x in eq_xs:
        eq = Text(eq_x, v_center, "=")
        eq.font_size(34).font_weight("bold").font_family("Arial")
        eq.prop["fill"] = cfg["qubit_name_color"]
        root.add(eq)
    b = box(root)
    pad = 20.0
    width, height = b["width"] + 2 * pad, b["height"] + 2 * pad
    root.shift(pad - b["left"], pad - b["top"])
    background = Rect(0, 0, width, height)
    background.prop["fill"] = cfg["background"]
    doc = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink">'
        f"{background.to_string()}{root.to_string()}</svg>"
    )
    return doc


def build_circuit(entry: dict, model: dict, noisy: bool, gate_scale: float):
    """Same gate order as circuit_sampler.frame_circuit, from the JSON table."""
    from mindquantum.core.circuit import Circuit
    from mindquantum.core.gates import (
        BitFlipChannel,
        DepolarizingChannel,
        Measure,
        UnivMathGate,
        Z,
    )

    circuit = Circuit()
    for step in entry["givens"]:
        j = int(step["qubits"][0])
        matrix = givens_matrix(float(step["theta"]))
        for spin in (0, 1):
            offset = spin * N_SPATIAL
            circuit.append(
                UnivMathGate(f"G{spin}_{j}", matrix).on([offset + j, offset + j + 1])
            )
            if noisy:
                edge = j if spin == 0 else N_SPATIAL + j
                for qubit in (offset + j, offset + j + 1):
                    rate = cs.gate_error(model, qubit, edge, gate_scale)
                    if rate > 0.0:
                        circuit.append(DepolarizingChannel(rate).on(qubit))
    for qubit, sign in enumerate(entry["diagonal_signs"]):
        if sign < 0:
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


def render(circuit, stem: str, outdir: Path, png_scale: float):
    svg_path = outdir / f"{stem}.svg"
    circuit.svg(style="official").to_file(str(svg_path))
    return render_doc(svg_path.read_text(encoding="utf-8"), stem, outdir, png_scale)


def render_doc(doc: str, stem: str, outdir: Path, png_scale: float):
    import cairosvg

    svg_path = outdir / f"{stem}.svg"
    svg_path.write_text(doc, encoding="utf-8")
    width = float(re.search(r'<svg[^>]*\bwidth="([\d.]+)', doc).group(1))
    scale = min(png_scale, 16000.0 / width)
    pdf_path = svg_path.with_suffix(".pdf")
    png_path = svg_path.with_suffix(".png")
    cairosvg.svg2pdf(bytestring=doc.encode("utf-8"), write_to=str(pdf_path))
    cairosvg.svg2png(bytestring=doc.encode("utf-8"), write_to=str(png_path), scale=scale)
    print(f"  {pdf_path.name}, {png_path.name} ({int(width * scale)} px wide)")
    return svg_path, pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="n2_r080")
    parser.add_argument("--arm", default="guard15_equal")
    parser.add_argument("--frame", type=int, default=None,
                        help="default: first selected frame of the arm")
    parser.add_argument("--gate-scale", type=float, default=1.0)
    parser.add_argument("--png-scale", type=float, default=2.0)
    parser.add_argument("--outdir", type=Path, default=PKG / "circuits" / "figures")
    parser.add_argument("--block-only", action="store_true",
                        help="only draw the Givens-block decomposition schematic")
    parser.add_argument("--no-block", action="store_true",
                        help="skip the Givens-block decomposition schematic")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    if not args.block_only:
        payload = json.loads(CIRCUITS_JSON.read_text())
        tag_entry = payload["tags"][args.tag]
        arm = tag_entry["arms"][args.arm]
        frame = args.frame if args.frame is not None else arm["selected_frames"][0]
        entry = next(c for c in arm["circuits"] if c["frame"] == frame)
        model = wukong_noise.load_model()

        stem = f"{args.tag}_frame{frame:02d}"
        print(f"{args.tag} {args.arm} frame {frame} ({len(entry['givens'])} Givens steps)")
        ideal = build_circuit(entry, model, noisy=False, gate_scale=args.gate_scale)
        print(f"  {len(ideal)} gates ->", end="")
        render(ideal, f"{stem}_ideal", args.outdir, args.png_scale)
        noisy = build_circuit(entry, model, noisy=True, gate_scale=args.gate_scale)
        print(f"  {len(noisy)} gates ->", end="")
        render(noisy, f"{stem}_noisy_g{args.gate_scale}", args.outdir, args.png_scale)
    if not args.no_block:
        print("Givens block decomposition (verified against givens_matrix):")
        render_doc(block_schematic(), "block_givens_decomposition",
                   args.outdir, args.png_scale)


if __name__ == "__main__":
    main()
