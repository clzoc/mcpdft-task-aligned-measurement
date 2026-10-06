#!/usr/bin/env python3
"""Panel-d assets for guard15_method_schematic, re-rendered at the exact column
size from the same precomputed tables behind the Material figures, without
panel letters; legends match the source Material figures.
Style follows guard15_equal_real30/plot_equilibrium_errorbars_vertical.py and
guard15_equal_real30_n2_scan120k/plot_scan_energy_bars.py. Run this before
make_schematic.py.

  assets/panel_d1.png  N2, R=1.10 A: Hamiltonian and MC-PDFT error vs shot
                       budget (data: guard15_equal_real30/figures/
                       equilibrium_errorbars/rmse_ci95.csv)
  assets/panel_d2.png  N2 bond scan at 120k shots: signed MC-PDFT error bars
                       (data: guard15_equal_real30_n2_scan120k/figures/
                       signed_errors/scan_signed_errors_mean_ci95.csv)
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "method_schematic" / "assets"
EQ_CSV = ROOT / "guard15_equal_real30" / "figures" / "equilibrium_errorbars" / "rmse_ci95.csv"
SCAN_CSV = (ROOT / "guard15_equal_real30_n2_scan120k" / "figures" / "signed_errors"
            / "scan_signed_errors_mean_ci95.csv")

DPI = 600
W_IN = 0.226 * 7.205  # panel-d column width in the schematic figure

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
})

ARMS = ("uniform", "guard15_equal_mu0", "guard15_equal")
LINESTY = {
    "uniform": dict(color="#3C5488", marker="o"),
    "guard15_equal_mu0": dict(color="#00A087", marker="^"),
    "guard15_equal": dict(color="#E64B35", marker="s"),
}
ARMLAB = {"uniform": "Uniform30",
          "guard15_equal_mu0": "Select15 $\\mu=0$",
          "guard15_equal": "Select15 $\\mu=2$"}
LAB_FS, TICK_FS = 6.2, 5.6


def make_d1():
    table = {}
    with EQ_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["system"] == "n2":
                table[int(row["budget"]), row["arm"], row["metric"]] = (
                    float(row["rmse"]), float(row["ci95_low"]), float(row["ci95_high"]))
    budgets = (30000, 60000, 120000, 240000)
    panels = [
        dict(metric="h_error_meh", ylabel="Hamiltonian error (mHa)",
             ylim=(0, 42), yticks=[1.6, 10, 20, 30, 40]),
        dict(metric="f_error_meh", ylabel="MC-PDFT error (mHa)",
             ylim=(0, 32), yticks=[1.6, 10, 20, 30]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(W_IN, W_IN * 1319 / 977), sharex=True)
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.125, top=0.995, hspace=0.12)
    x = np.array(budgets) / 1000
    for ax, panel in zip(axes, panels):
        for arm in ARMS:
            st = LINESTY[arm]
            y, lo, hi = zip(*(table[b, arm, panel["metric"]] for b in budgets))
            y = np.array(y)
            ax.errorbar(x, y, yerr=np.vstack((y - np.array(lo), np.array(hi) - y)),
                        color=st["color"], ls="-", marker=st["marker"], markersize=3.2,
                        mfc="white", mec=st["color"], mew=0.8, linewidth=0.9,
                        elinewidth=0.7, capsize=1.6, capthick=0.7,
                        label=ARMLAB[arm] if ax is axes[0] else None,
                        zorder=2 if arm == "guard15_equal_mu0" else 3)
        ax.axhline(1.6, color="0.35", ls=(0, (4, 3)), linewidth=0.7, zorder=1)
        ax.set_ylim(panel["ylim"])
        ax.set_yticks(panel["yticks"], labels=[f"{t:g}" for t in panel["yticks"]])
        for tick, lbl in zip(ax.get_yticks(), ax.get_yticklabels()):
            if np.isclose(tick, 1.6):
                lbl.set_color("0.35")
        ax.set_ylabel(panel["ylabel"], fontsize=LAB_FS, labelpad=1.5)
        ax.tick_params(labelsize=TICK_FS, pad=1.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(loc="upper right", fontsize=5.6, frameon=False, handlelength=1.4,
                   handletextpad=0.5, labelspacing=0.35, borderaxespad=0.1)
    ax.set_xscale("log", base=2)
    ax.set_xlim(26, 278)
    ax.set_xticks(x, labels=["30k", "60k", "120k", "240k"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("total measurement shots", fontsize=LAB_FS, labelpad=1.8)
    fig.savefig(ASSETS / "panel_d1.png", dpi=DPI)
    plt.close(fig)
    print("wrote", ASSETS / "panel_d1.png")


def make_d2():
    stats = {}
    with SCAN_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["metric"] == "F":
                bond = float(row["panel"].split("=")[1])
                stats[row["group"], bond] = float(row["mean_error_mHa"])
    bonds = (.80, .90, 1.00, 1.10, 1.25, 1.45, 1.60, 1.80, 2.00, 2.20, 2.50)
    barsty = {"uniform": dict(color="#3C5488", hatch="//"),
              "guard15_equal": dict(color="#E64B35", hatch="\\\\")}
    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 800 / 977))
    fig.subplots_adjust(left=0.215, right=0.985, bottom=0.21, top=0.985)
    width = .055
    for i, b in enumerate(bonds):
        order = sorted(barsty, key=lambda a: -abs(stats[a, b]))  # longest first
        for arm in order:
            ax.bar(b, stats[arm, b], width=width, color=barsty[arm]["color"],
                   edgecolor="0.1", linewidth=0.6, hatch=barsty[arm]["hatch"], zorder=3,
                   label=ARMLAB[arm] if i == 0 else None)
    ax.legend(loc="upper right", fontsize=5.6, frameon=False, handlelength=1.2,
              handletextpad=0.5, labelspacing=0.35, borderaxespad=0.1)
    ax.axhline(0., color="0.0", linewidth=0.7, linestyle=":", zorder=2)
    for v in (-1.6, 1.6):
        ax.axhline(v, color="0.35", linewidth=0.7, linestyle="--", zorder=2)
    ax.set_ylim(-4.5, 16.5)
    ax.set_yticks([-1.6, 0, 1.6, 5, 10, 15],
                  labels=["-1.6", "0", "1.6", "5", "10", "15"])
    for lbl in ax.get_yticklabels():
        if lbl.get_text() in ("-1.6", "1.6"):
            lbl.set_color("0.35")
    ax.set_ylabel("MC-PDFT error (mHa)", fontsize=LAB_FS, labelpad=1.5)
    ax.set_xlabel("bond length (\u00c5)", fontsize=LAB_FS, labelpad=1.8)
    ax.set_xlim(.72, 2.58)
    ax.set_xticks(bonds, labels=["0.8", "", "", "1.1", "", "1.45",
                               "", "1.8", "2.0", "2.2", "2.5"])
    ax.tick_params(labelsize=TICK_FS, pad=1.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(ASSETS / "panel_d2.png", dpi=DPI)
    plt.close(fig)
    print("wrote", ASSETS / "panel_d2.png")


if __name__ == "__main__":
    make_d1()
    make_d2()
