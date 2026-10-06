#!/usr/bin/env python3
"""Color variant of the N2 DQG ftPBE constraint-range figure.

Identical layout to make_figure.py; only the grayscale encodings are
replaced by a colorblind-safe, muted scientific palette:

  blue  #4C72B0  shared classical part E_C / lower extreme
  orange #DD8452 nonclassical part E_xc, on-top part E_ot / upper extreme

Output: figures/n2-ftpbe-constraint-range-color.pdf / .png
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ftpbe-range-figure-mpl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
FIGURE_STEM = FIGURE_DIR / "n2-ftpbe-constraint-range-color"

MAHA = 1e3  # Eh -> mHa

# C_COMMON = "#4C72B0"  # E_C, lower extreme
C_COMMON = "#6D8BC0"  # E_C, lower extreme
# C_XC = "#DD8452"      # E_xc / E_ot, upper extreme
C_XC = "#7DA291"      # E_xc / E_ot, upper extreme
C_BAND = "#dbe4f0"    # E_C band shading


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans",
                                "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 12.0,
            "axes.labelsize": 11.0,
            "axes.titlesize": 10.5,
            "axes.linewidth": 0.85,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 8,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    set_style()
    tight = load(HERE / "range_tight_results.json")

    ref = tight["reference"]
    ref_f = ref["f_total"]
    ref_c = ref["c_common"]
    ref_ot = ref["f_on_top"]
    ref_uh = ref["e_h"] - ref_c
    ref_h = ref["e_h"]

    def common(record: dict) -> float:
        return record.get("c_common", record.get("f_non_ontop"))

    def point(record: dict) -> dict:
        return dict(
            f=(record["f_total"] - ref_f) * MAHA,
            c=(common(record) - ref_c) * MAHA,
            ot=(record["f_on_top"] - ref_ot) * MAHA,
            eh=(record["e_h"] - ref_h) * MAHA,
            uh=(record["e_h"] - common(record) - ref_uh) * MAHA,
            d2_pct=100.0 * record["d2_error_relative"],
        )

    tight_min = point(tight["minimum"])
    tight_max = point(tight["maximum"])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.9))
    axa, axb, axc, axd = axes.ravel()

    # ---------------------------------------------------------------- (a)
    # E_H budget: dE_H = dE_C + dE_xc (common classical part + the
    # nonclassical two-electron remainder that E_ot replaces in E_F).
    # Component bars are drawn from zero; the dashed marker lands on
    # the sum.
    width = 0.34
    for record, offset in ((tight_min, 0.0), (tight_max, 1.15)):
        c_bar = record["c"]
        r_bar = record["uh"]  # dE_xc = dE_H - dE_C
        edge = "0.1"
        axa.bar(offset, c_bar, width=width, color=C_COMMON, edgecolor=edge,
                linewidth=0.7, hatch="//")
        axa.bar(offset + width, r_bar, width=width,
                color=C_XC, edgecolor=edge, linewidth=0.9, hatch="\\\\")
        end = record["eh"]
        axa.hlines(end, offset - 0.28, offset + 1.5 * width,
                   color="0.0", linewidth=0.9, linestyle="--")
        axa.plot([offset + 1.5 * width], [end], marker="D", color="0.0",
                 markersize=4.4, zorder=5)
        axa.annotate(f"{end:+.2f}", (offset + 1.5 * width, end),
                     xytext=(7, 0), textcoords="offset points",
                     ha="left", va="center", fontsize=8.5,
                     fontweight="bold",
                     bbox=dict(boxstyle="square,pad=0.1", fc="white",
                               ec="none", alpha=0.85))
        value_box = dict(boxstyle="square,pad=0.12", fc="white", ec="none",
                         alpha=0.85)
        axa.annotate(f"{c_bar:+.2f}", (offset, c_bar / 2.0),
                     ha="center", va="center", fontsize=7.2, rotation=90,
                     color="0.1", bbox=value_box)
        axa.annotate(f"{r_bar:+.2f}", (offset + width, r_bar / 2.0),
                     ha="center", va="center", fontsize=7.2, rotation=90,
                     color="0.1", bbox=value_box)
    axa.axhline(0.0, color="0.0", linewidth=0.8, linestyle=":")
    axa.set_xticks([width / 2.0, 1.15 + width / 2.0])
    axa.set_xticklabels(["lower extreme", "upper extreme"], fontsize=8)
    axa.set_ylabel("energy shift from reference  (mHa)")
    axa.set_ylim(-6.5, 6.5)
    axa.set_xlim(-0.5, 2.35)
    axa.set_title("$E_H = E_C + E_{\\mathrm{xc}}$ budget of the extremes",
                  fontsize=10, loc="left")
    axa.legend(
        handles=[
            Patch(facecolor=C_COMMON, edgecolor="0.1", linewidth=0.7,
                  hatch="//", label="$\\Delta E_C$ (common)"),
            Patch(facecolor=C_XC, edgecolor="0.1", linewidth=0.9,
                  hatch="\\\\",
                  label="$\\Delta E_{\\mathrm{xc}}$ (nonclassical)"),
        ],
        loc="upper left", bbox_to_anchor=(0.02, 0.99), frameon=False,
        fontsize=7.5, handlelength=1.5,
    )

    # ---------------------------------------------------------------- (b)
    # Waterfall: segment heights ARE the contributions, so the stack top
    # lands on dE_F = dE_C + dE_ot (checked against the data: exact).
    width = 0.34
    for tag, record, offset in (
        ("lower", tight_min, 0.0),
        ("upper", tight_max, 1.15),
    ):
        c_bar = record["c"]
        ot_bar = record["ot"]
        edge = "0.1"
        axb.bar(offset, c_bar, width=width, color=C_COMMON, edgecolor=edge,
                linewidth=0.7, hatch="//")
        axb.bar(offset + width, ot_bar, width=width, bottom=c_bar,
                color=C_XC, edgecolor=edge, linewidth=0.9, hatch="\\\\")
        end = record["f"]
        axb.hlines(end, offset - 0.28, offset + 2 * width + 0.18,
                   color="0.0", linewidth=0.9, linestyle="--")
        axb.plot([offset + 1.5 * width], [end], marker="D", color="0.0",
                 markersize=4.4, zorder=5)
        axb.annotate(f"{end:+.2f}", (offset + 1.5 * width, end),
                     xytext=(0, -14 if tag == "lower" else 8),
                     textcoords="offset points", ha="center", fontsize=8.5,
                     fontweight="bold")
        value_box = dict(boxstyle="square,pad=0.12", fc="white", ec="none",
                         alpha=0.85)
        axb.annotate(f"{c_bar:+.2f}", (offset, c_bar / 2.0),
                     ha="center", va="center", fontsize=7.2, rotation=90,
                     color="0.1", bbox=value_box)
        axb.annotate(f"{ot_bar:+.2f}", (offset + width, c_bar + ot_bar / 2.0),
                     ha="center", va="center", fontsize=7.2, rotation=90,
                     color="0.1", bbox=value_box)
    axb.axhline(0.0, color="0.0", linewidth=0.8, linestyle=":")
    axb.set_xticks([width / 2.0, 1.15 + width / 2.0])
    axb.set_xticklabels(["lower extreme", "upper extreme"], fontsize=8)
    axb.set_ylabel("energy shift from reference  (mHa)")
    axb.set_ylim(-17.5, 17.5)
    axb.set_xlim(-0.5, 2.05)
    axb.set_title("$E_F = E_C + E_{\\mathrm{ot}}$ budget of the extremes",
                  fontsize=10, loc="left")
    axb.legend(
        handles=[
            Patch(facecolor=C_COMMON, edgecolor="0.1", linewidth=0.7,
                  hatch="//", label="$\\Delta E_C$ (common)"),
            Patch(facecolor=C_XC, edgecolor="0.1", linewidth=0.9,
                  hatch="\\\\", label="$\\Delta E_{\\mathrm{ot}}$ (on-top)"),
        ],
        loc="upper left", bbox_to_anchor=(0.02, 0.99), frameon=False,
        fontsize=7.5, handlelength=1.5,
    )

    # ---------------------------------------------------------------- (c)
    # Utilizations are 99.9924-100.0000%, so the axis is zoomed next to
    # the cap; the exact value is printed above every bar.
    labels = ["$|\\Delta E_H| \\leq 1$ mHa", "$|\\Delta E_C| \\leq 5$ mHa",
              "$\\|\\Delta D_2\\|_F \\leq 1.01\\%$"]
    util_min = [100.0 * abs(tight_min["eh"]) / 1.0,
                100.0 * abs(tight_min["c"]) / 5.0,
                100.0 * tight_min["d2_pct"] / 1.01]
    util_max = [100.0 * abs(tight_max["eh"]) / 1.0,
                100.0 * abs(tight_max["c"]) / 5.0,
                100.0 * tight_max["d2_pct"] / 1.01]
    x = np.arange(3)
    axc.bar(x - 0.19, util_min, 0.36, color=C_COMMON, edgecolor="0.1",
            linewidth=0.7, hatch="//", label="lower extreme")
    axc.bar(x + 0.19, util_max, 0.36, color=C_XC, edgecolor="0.1",
            linewidth=0.9, hatch="\\\\", label="upper extreme")
    axc.axhline(100.0, color="0.0", linewidth=0.9, linestyle="--")
    for xi, v in zip(x, util_min):
        axc.annotate(f"{v:.5f}", (xi - 0.19, v), xytext=(0, 2),
                     textcoords="offset points", ha="center", fontsize=6.2,
                     bbox=dict(boxstyle="square,pad=0.1", fc="white",
                               ec="none", alpha=0.85))
    for xi, v in zip(x, util_max):
        axc.annotate(f"{v:.5f}", (xi + 0.19, v), xytext=(0, 11),
                     textcoords="offset points", ha="center", fontsize=6.2,
                     bbox=dict(boxstyle="square,pad=0.1", fc="white",
                               ec="none", alpha=0.85))
    axc.set_xticks(x)
    axc.set_xticklabels(labels, fontsize=8)
    axc.set_ylim(99.99, 100.005)
    axc.set_yticks([99.99, 99.995, 100.0])
    axc.set_yticklabels(["99.990", "99.995", "100 (cap)"])
    axc.set_ylabel("constraint utilization  (%)")
    axc.set_title("all three caps saturate at both extremes",
                  fontsize=10, loc="left")
    axc.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), frameon=False,
               fontsize=7, ncols=2, handlelength=1.4, columnspacing=1.0)

    # ---------------------------------------------------------------- (d)
    axd.axvspan(-5, 5, color=C_BAND, zorder=0)
    xs = np.linspace(-13, 16, 200)
    axd.plot(xs, -xs, color="0.55", linewidth=0.8, zorder=1)
    axd.plot(xs, tight_min["f"] - xs, color=C_COMMON, linewidth=1.0,
             zorder=1)
    axd.plot(xs, tight_max["f"] - xs, color=C_XC, linewidth=1.0, zorder=1)
    for wall in (-5, 5):
        axd.axvline(wall, color="0.45", linewidth=0.8, linestyle="--",
                    zorder=2)
    text_box = dict(boxstyle="square,pad=0.12", fc="white", ec="none",
                    alpha=0.8)
    axd.text(-12.6, -10.9,
             "$\\Delta E_F = \\Delta E_C + \\Delta E_{\\mathrm{ot}}$",
             fontsize=7.5, color="0.3", ha="left", va="center",
             bbox=text_box)
    axd.text(-11.2, -2.25, "$-13.64$", fontsize=6.8, color=C_COMMON,
             rotation=-47, bbox=text_box)
    axd.text(10.8, 3.0, "$+13.58$", fontsize=6.8, color=C_XC,
             rotation=-47, bbox=text_box)
    axd.text(13.4, -13.15, "0", fontsize=6, color="0.45", ha="center",
             va="bottom")
    points = [
        ("ref", 0.0, 0.0, "D", "0.0", "reference"),
        ("tight min", tight_min["c"], tight_min["ot"], "s", C_COMMON,
         "tight min"),
        ("tight max", tight_max["c"], tight_max["ot"], "s", C_XC,
         "tight max"),
    ]
    for tag, xv, yv, marker, color, label in points:
        axd.plot([xv], [yv], linestyle="none", marker=marker,
                 markersize=5.0, markerfacecolor="white" if tag == "ref" else color,
                 markeredgecolor=color, markeredgewidth=1.1,
                 zorder=5, label=label)
    for tag, xv, yv in (("tight min", tight_min["c"], tight_min["ot"]),
                        ("tight max", tight_max["c"], tight_max["ot"])):
        dx, dy = ((7, 3) if tag == "tight max" else (-7, 4))
        axd.annotate(tag, (xv, yv), xytext=(dx, dy), textcoords="offset points",
                     fontsize=7.2, ha="left" if tag == "tight max" else "right",
                     color="0.0")
    axd.set_xlim(-13, 16)
    axd.set_ylim(-13.4, 11.0)
    axd.set_xlabel("$\\Delta E_C$  (mHa)")
    axd.set_ylabel("$\\Delta E_{\\mathrm{ot}}$  (mHa)")
    axd.set_title("mechanism plane: $E_C$ wall vs on-top shift",
                  fontsize=10, loc="left")
    axd.legend(loc="upper left", frameon=False, fontsize=7.2,
               handlelength=1.2)

    fig.tight_layout()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIGURE_STEM) + ".pdf")
    fig.savefig(str(FIGURE_STEM) + ".png", dpi=300)
    print(f"Figure: {FIGURE_STEM}.pdf / .png")


if __name__ == "__main__":
    main()
