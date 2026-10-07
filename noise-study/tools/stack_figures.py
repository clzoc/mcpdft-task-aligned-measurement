#!/usr/bin/env python3
"""Stack figure PDFs vertically into a single-page PDF (+ PNG raster).

The first input becomes the top panel; vector content is merged, not
rasterized.  By default every panel is scaled to the widest panel's width
(--no-match-width disables); panels are centered horizontally.  Default
figure: the frame-03 ideal circuit on top of the Givens-block
decomposition schematic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
FIGURES = PKG / "circuits" / "figures"


def stack_pdf(inputs: list[Path], output: Path, gap: float, margin: float,
              match_width: bool = False):
    from pypdf import PdfReader, PdfWriter, Transformation

    pages = []
    for path in inputs:
        page = PdfReader(str(path)).pages[0]
        pages.append((page, float(page.mediabox.width), float(page.mediabox.height)))
    widest = max(w for _, w, _ in pages)
    scales = [widest / w if match_width else 1.0 for _, w, _ in pages]
    width = widest + 2 * margin
    height = sum(h * s for (_, _, h), s in zip(pages, scales))
    height += gap * (len(pages) - 1) + 2 * margin
    writer = PdfWriter()
    canvas = writer.add_blank_page(width=width, height=height)
    y = height - margin
    for (page, w, h), scale in zip(pages, scales):
        y -= h * scale
        ctm = Transformation().scale(scale).translate((width - w * scale) / 2, y)
        canvas.merge_transformed_page(page, ctm)
        y -= gap
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        writer.write(handle)
    return width, height, scales


def rasterize(pdf: Path, png: Path, scale: float):
    import pymupdf

    doc = pymupdf.open(str(pdf))
    pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    pix.save(str(png))
    return pix.width, pix.height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=[
        FIGURES / "n2_r080_frame03_ideal.pdf",
        FIGURES / "block_givens_decomposition.pdf",
    ])
    parser.add_argument("--output", type=Path,
                        default=FIGURES / "n2_r080_frame03_with_block.pdf")
    parser.add_argument("--gap", type=float, default=36.0, help="points")
    parser.add_argument("--margin", type=float, default=24.0, help="points")
    parser.add_argument("--no-match-width", dest="match_width", action="store_false",
                        help="keep every panel at its natural width")
    parser.set_defaults(match_width=True)
    parser.add_argument("--png-scale", type=float, default=2.0)
    args = parser.parse_args()

    width, height, scales = stack_pdf(
        args.inputs, args.output, args.gap, args.margin, args.match_width
    )
    print(f"{args.output.name}: {width:.0f} x {height:.0f} pt, "
          f"{len(args.inputs)} panels, scales {[round(s, 3) for s in scales]}")
    png = args.output.with_suffix(".png")
    w, h = rasterize(args.output, png, args.png_scale)
    print(f"{png.name}: {w} x {h} px")


if __name__ == "__main__":
    main()
