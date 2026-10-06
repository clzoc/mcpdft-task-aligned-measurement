#!/usr/bin/env python3
"""Nature-style method schematic for the guard15 + mu=2 ftPBE-regularized
band-SDP shadow-tomography protocol shared by guard15_equal_real30 and
guard15_equal_real30_n2_scan120k.

Layout (left to right):
  a  H-F sensitivity mismatch (crops of Material/n2-ftpbe-constraint-range-color.png)
  b  guard15 pilot + backward-greedy frame selection + equal allocation
  c  mu=2 on-top (ftPBE) regularization inside the DQG band SDP
  d  measured effect: budget scaling (N2) and N2 dissociation scan
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "method_schematic"
ASSETS = HERE / "assets"

# ---------------------------------------------------------------- fonts/style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.default": "regular",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

HERO = "#0072B2"    # guard15 mu=2 (ours)
MID = "#56B4E9"     # guard15 mu=0
BASE = "#666666"    # uniform baseline
KEPT = HERO
BANDF = "#DCE9F7"
FACC = "#CC79A7"    # F/on-top accent
INK = "#222222"

FIGW, FIGH = 7.205, 4.60  # inches (183 mm double column)
fig = plt.figure(figsize=(FIGW, FIGH))
RATIO = FIGW / FIGH


def frac_ax(x0, y0, w, h):
    return fig.add_axes([x0, y0, w, h])


def panel_letter(x, s):
    fig.text(x, 0.985, s, fontsize=10, fontweight="bold", ha="left", va="top", color=INK)


def panel_title(x0, w, s, fs=7.8):
    fig.text(x0 + w / 2, 0.952, s, fontsize=fs, fontweight="bold", ha="center", va="top", color=INK)




# =================================================================== panel a
AX0, AW = 0.018, 0.222
panel_letter(AX0 - 0.006, "a")
panel_title(AX0, AW, "$E_H$–$E_F$ sensitivity mismatch")

def load_img(name):
    return np.asarray(Image.open(ASSETS / name).convert("RGB"))


IMG_PANELS = []  # (name, x0, y0, w, h, axim) for PIL compositing of the PNG


def place_image(name, x0, y0, w, h):
    ax = frac_ax(x0, y0, w, h)
    axim = ax.imshow(load_img(name))
    ax.axis("off")
    IMG_PANELS.append((name, x0, y0, w, h, axim))


img_eh = "crop_eh.png"     # 1115 x 810
img_ef = "crop_ef.png"     # 1040 x 810
img_pl = "crop_plane.png"  # 918 x 875

h1 = AW * (810 / 1115) * RATIO
h2 = AW * (810 / 1040) * RATIO

place_image(img_eh, AX0, 0.922 - h1, AW, h1)
place_image(img_ef, AX0, 0.922 - h1 - 0.006 - h2, AW, h2)

ycall = 0.922 - h1 - 0.006 - h2 - 0.006
fig.text(AX0 + AW / 2, ycall,
         "same small 2-RDM moves ($\\|\\Delta D_2\\|_F \\leq 1.01\\%$):\n"
         "$|\\Delta E_H| \\leq 1$ mHa   vs   $|\\Delta E_F| \\approx 13.6$ mHa",
         fontsize=5.9, ha="center", va="top", color=INK,
         bbox=dict(boxstyle="round,pad=0.32", facecolor="#FDF3E7", edgecolor="#E0A458", linewidth=0.7))

w3 = 0.200
place_image(img_pl, AX0 + (AW - w3) / 2, 0.018, w3, w3 * (889 / 918) * RATIO)

# =================================================================== panel b
BX0, BW = 0.272, 0.200
panel_letter(BX0 - 0.004, "b")
panel_title(BX0, BW, "frame selection")

rng = np.random.default_rng(7)
ANGLES = rng.uniform(0.15, np.pi - 0.15, 30)
SELECTED = {3, 5, 8, 10, 13, 14, 17, 18, 19, 21, 22, 23, 24, 27, 28}  # actual r0_b120000


def frame_grid(ax, mode):
    ax.set_xlim(-0.55, 9.55); ax.set_ylim(-0.62, 2.62)
    ax.set_aspect("equal"); ax.axis("off")
    for i in range(30):
        r, c = divmod(i, 10)
        x, y = c, 2 - r
        if mode == "pre":
            fc, ec, lc = "#A6CEE3", "#3F6E8C", "#3F6E8C"
        else:
            if i in SELECTED:
                fc, ec, lc = KEPT, "#004C77", "white"
            else:
                fc, ec, lc = "#E3E3E3", "#B0B0B0", "#B0B0B0"
        ax.add_patch(plt.Circle((x, y), 0.36, facecolor=fc, edgecolor=ec, linewidth=0.7, zorder=2))
        a = ANGLES[i]
        dx, dy = 0.24 * np.cos(a), 0.24 * np.sin(a)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=lc, linewidth=0.8, zorder=3,
                solid_capstyle="round")


def stage_label(y, s):
    fig.text(BX0, y, s, fontsize=6.8, fontweight="bold", ha="left", va="top", color="#333333")


stage_label(0.918, "pilot: every frame $\\times$ 500 shots")
g1 = frac_ax(BX0, 0.745, BW, 0.165); frame_grid(g1, "pre")
fig.text(BX0 + BW / 2, 0.733, "pool: 30 Haar $O(8)$ frames",
         fontsize=6.2, ha="center", va="top", color=INK)

stage_label(0.662, "backward-greedy screen (pilot-only)")
ins = frac_ax(BX0 + 0.028, 0.430, BW - 0.045, 0.215)
sizes = [30, 28, 26, 24, 22, 20, 18, 16, 15]
vals = [1.000, 1.006, 1.013, 1.021, 1.031, 1.043, 1.058, 1.072, 1.0837]
sizes2 = [15, 12, 10, 8, 6, 4, 2]
vals2 = [1.0837, 1.14, 1.21, 1.33, 1.50, 1.74, 2.05]
ins.axhline(1.0, color="#999999", linewidth=0.6, linestyle=":")
ins.plot(sizes, vals, "-", color=HERO, linewidth=1.2, zorder=3)
ins.plot(sizes2, vals2, "--", color="#AAAAAA", linewidth=1.0, zorder=2)
ins.plot([15], [1.0837], "o", color=HERO, markersize=3.6, zorder=4)
ins.annotate("stop at $K = 15$", xy=(15, 1.0837), xytext=(23.5, 1.45), fontsize=6.0,
             arrowprops=dict(arrowstyle="-", color=INK, linewidth=0.6,
                             connectionstyle="arc3,rad=0.25"),
             ha="center", color=INK)
ins.set_xlim(30.8, 1.2)
ins.set_ylim(0.94, 2.2)
ins.set_xticks([30, 20, 15, 2])
ins.tick_params(labelsize=5.6, pad=1.2)
ins.set_xlabel("subset size $|S|$", fontsize=6.0, labelpad=1.5)
ins.set_ylabel("worst variance / uniform", fontsize=6.0, labelpad=1.5)
for sp in ("top", "right"):
    ins.spines[sp].set_visible(False)

stage_label(0.358, "keep $K = 15$, equal shots")
g2 = frac_ax(BX0, 0.205, BW, 0.150); frame_grid(g2, "post")
fig.text(BX0 + BW / 2, 0.193,
         "$(B{-}15{,}000)/15$ new shots per kept frame\n"
         "kept pilots reused, dropped ones discarded",
         fontsize=5.9, ha="center", va="top", color=INK, linespacing=1.4)

# =================================================================== panel c
CX0, CW = 0.516, 0.200
panel_letter(CX0 - 0.004, "c")
panel_title(CX0, CW, "$\\mu = 2$ ftPBE regularization")

box = FancyBboxPatch((CX0, 0.745), CW, 0.173, transform=fig.transFigure,
                     boxstyle="round,pad=0.008", facecolor="#F1F6FB",
                     edgecolor="#9DB6CC", linewidth=0.8, zorder=2)
fig.patches.append(box)
fig.text(CX0 + CW / 2, 0.906, "band SDP on the 15 selected frames",
         fontsize=6.6, fontweight="bold", ha="center", va="top", color=INK, zorder=3)
fig.text(CX0 + CW / 2, 0.872,
         "$\\min_{D,\\gamma,E^{\\pm}}\\; H \\;+\\; \\mu\\,F_{\\mathrm{lin}} \\;+\\; \\lambda\\,\\mathrm{Tr}(E^{+}{+}E^{-})$",
         fontsize=7.4, ha="center", va="top", color=INK, zorder=3)
fig.text(CX0 + CW / 2, 0.800,
         "s.t. DQG 2-positivity, spin, contraction,\n"
         "row band $\\bar{y} \\in A\\,\\mathrm{vec}(D \\pm \\Delta)$",
         fontsize=5.9, ha="center", va="top", color=INK, zorder=3, linespacing=1.5)

ep = frac_ax(CX0 + 0.022, 0.070, CW - 0.010, 0.615)
ep.set_xlim(-3.1, 3.1); ep.set_ylim(-15.5, 15.5)
ep.add_patch(Ellipse((0.0, 0.0), 3.0, 27.0, facecolor=BANDF, edgecolor="#8FB6D9",
                     linewidth=0.8, zorder=1))
ep.axhline(0, color="#BBBBBB", linewidth=0.5, zorder=2)
ep.axvline(0, color="#BBBBBB", linewidth=0.5, zorder=2)
ep.text(0, 14.9, "measurement-compatible set", fontsize=5.6, color="#4A7BA6",
        ha="center", va="top", zorder=3)
ep.plot([0], [0], marker="*", color="black", markersize=7, zorder=6, linestyle="none")
ep.annotate("exact CAS", xy=(0, 0), xytext=(1.9, -3.4), fontsize=6.0,
            arrowprops=dict(arrowstyle="-", color="black", linewidth=0.5), ha="center")
ep.plot([-0.6], [-5.5], marker="D", color="#888888", markersize=4.5, zorder=5, linestyle="none",
        markerfacecolor="white", markeredgewidth=1.0)
ep.annotate("DQG anchor", xy=(-0.6, -5.5), xytext=(-2.05, -9.3), fontsize=6.0, color="#666666",
            arrowprops=dict(arrowstyle="-", color="#888888", linewidth=0.5), ha="center")
ep.plot([0.82], [10.6], marker="o", markersize=5, markerfacecolor="white",
        markeredgecolor=BASE, markeredgewidth=1.2, zorder=5, linestyle="none")
ep.text(1.18, 10.6, "$\\mu = 0$:\n$F$ drifts", fontsize=6.0, color=BASE, ha="left", va="center")
ep.plot([0.35], [2.3], marker="o", markersize=5.5, color=HERO, zorder=6, linestyle="none")
ep.text(1.18, 2.3, "$\\mu = 2$:\n$F$ pulled in", fontsize=6.0, color=HERO, ha="left", va="center",
        fontweight="bold")
ep.annotate("", xy=(0.42, 2.9), xytext=(0.78, 10.0),
            arrowprops=dict(arrowstyle="-|>", color=FACC, linewidth=1.4,
                            connectionstyle="arc3,rad=-0.35"), zorder=4)
ep.text(1.38, 6.6, "$\\mu\\,F_{\\mathrm{lin}}$", rotation=78,
        fontsize=5.8, color=FACC, ha="center", va="center")
ep.set_xlabel("$E_H$ error (mHa)", fontsize=6.8, labelpad=1.5)
ep.set_ylabel("$E_F$ error (mHa)", fontsize=6.8, labelpad=1.5)
ep.set_xticks([-2, 0, 2]); ep.set_yticks([-10, 0, 10])
ep.tick_params(labelsize=6.0, pad=1.2)
for sp in ("top", "right"):
    ep.spines[sp].set_visible(False)

# =================================================================== panel d
DX0, DWID = 0.758, 0.230
panel_letter(DX0 - 0.004, "d")
panel_title(DX0, DWID, "effect on $H$ and $F$ errors")

# Panel-d images are re-rendered at this column size from the same precomputed
# tables behind the Material figures; see make_panel_d.py (run it first).
wd = 0.226
xd = DX0 + (DWID - wd) / 2
hd1 = wd * (1319 / 977) * RATIO
hd2 = wd * (800 / 977) * RATIO
fig.text(DX0 + DWID / 2, 0.926, "N$_2$, $R = 1.10$ $\\AA$",
         fontsize=6.2, ha="center", va="top", color=INK)
place_image("panel_d1.png", xd, 0.896 - hd1, wd, hd1)
fig.text(DX0 + DWID / 2, 0.896 - hd1 - 0.012, "N$_2$ dissociation, 120k shots",
         fontsize=6.2, ha="center", va="top", color=INK)
place_image("panel_d2.png", xd, 0.896 - hd1 - 0.064 - hd2, wd, hd2)

# ------------------------------------------------------------------ save
out_pdf = HERE / "guard15_method_schematic.pdf"
out_svg = HERE / "guard15_method_schematic.svg"
out_png = HERE / "guard15_method_schematic.png"
DPI = 600
fig.savefig(out_pdf, dpi=DPI)
fig.savefig(out_svg)
print("wrote", out_pdf)
print("wrote", out_svg)

# PNG: Agg's imshow resampler corrupts some rows at high dpi; render with the
# images hidden and composite the crops with PIL at their exact pixel rects.
for _, _, _, _, _, axim in IMG_PANELS:
    axim.set_visible(False)
fig.savefig(out_png, dpi=DPI)
canvas = Image.open(out_png)
W, H = canvas.size
for name, x0, y0, w, h, _ in IMG_PANELS:
    px = int(round(x0 * W))
    pw = int(round(w * W))
    ph = int(round(h * H))
    py = H - int(round((y0 + h) * H))
    tile = Image.open(ASSETS / name).convert("RGB").resize((pw, ph), Image.LANCZOS)
    canvas.paste(tile, (px, py))
canvas.save(out_png, dpi=(DPI, DPI))
print("wrote", out_png)
