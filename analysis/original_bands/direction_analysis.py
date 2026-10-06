#!/usr/bin/env python3
"""Per-frame truth/noise and solver-movement direction analysis for original_bands.

Sub-commands
------------
frames  per-frame exact signal, finite-shot noise, whitened ratio, and the
        H/F response of each frame's noise (linearized at the DQG anchor).
solve   rerun one band solve (spin- or full-error-basis) and cache d2/gamma
        plus the unconstrained raw shadow estimate in ``direction_cache/``.
report  merge existing result JSONs with the cached solves: anchor vs solved
        H/F/D2 errors, movement direction cosines, H-down/F-up cases.

    python direction_analysis.py frames
    python direction_analysis.py solve --arm full --budget 30000 --stream 0
    python direction_analysis.py report
"""

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / "original_global_fit"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import context_108 as m  # noqa: E402
import research as r  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("original_bands_run",
                                               HERE / "run.py")
band = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = band
_spec.loader.exec_module(band)

CACHE = HERE / "direction_cache"
PILOT_INDEX_BASE = 200
SSEED0, SSEED_STEP = 202610018100, 104729
MAX_SAMPLE_BUDGET = 300_000
ARMS = {
    "spin": HERE / "results",
    "full": HERE / "results_full",
    "noproj": HERE.parent / "original_global_fit" / "results_noproj",
    "wrapper": HERE.parent / "original_global_fit" / "results",
}


def setup():
    resource.setrlimit(resource.RLIMIT_AS, (24 * 1024 ** 3, 24 * 1024 ** 3))
    c = m.context()
    rotations, _ = m.pool_uniform_family(c)
    m.install_uniform_family(c, rotations)
    return c


def production_seed(stream):
    return SSEED0 + SSEED_STEP * (PILOT_INDEX_BASE + stream)


def sampled(c, stream, budget):
    active = tuple(range(len(c.rotations)))
    counts = r.j._equal_counts(len(c.rotations), active, budget,
                               c.args.allocation_chunk)
    max_counts = r.j._equal_counts(len(c.rotations), active,
                                   max(MAX_SAMPLE_BUDGET, budget),
                                   c.args.allocation_chunk)
    data = r.j._sample_outcomes(c.oracle, production_seed(stream), max_counts)
    return counts, data


def design_matrix(c):
    return np.vstack([c.blocks[f] for f in range(len(c.blocks))]) @ c.lift


def anchor_values(c):
    return np.concatenate([c.blocks[f] @ c.base.d2[c.geo["rows"], c.geo["cols"]]
                           for f in range(len(c.blocks))])


def unit(vector):
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def gradient_h(c):
    return r.j._hamiltonian_gradient(c.sel, c.geo["rows"], c.geo["cols"])


def gradient_f(c, d2, gamma):
    return r.j._raw_ftpbe_gradient(c.objective, c.sel, d2, gamma,
                                   c.geo["rows"], c.geo["cols"])


def energies(c, d2, gamma):
    return r.j._energy_values(c.sel, c.objective, d2, gamma)


def frame_statistics(c, budget, stream, data=None):
    counts, sampled_data = sampled(c, stream, budget)
    if data is None:
        data = sampled_data
    base_y = anchor_values(c)
    rows = []
    for frame in range(len(c.rotations)):
        number = int(counts[frame])
        exact = np.asarray(c.exact_y[frame], dtype=float)
        finite = data[frame][:number].mean(0)
        delta = finite - exact
        signal = exact - base_y[frame * len(exact):(frame + 1) * len(exact)]
        probs = c.oracle._frame_probabilities(frame)
        indicators = c.oracle.indicators.astype(float)
        mean = probs @ indicators
        cov = (indicators.T * probs) @ indicators - np.outer(mean, mean)
        cov = cov / number
        whitened = np.linalg.pinv(0.5 * (cov + cov.T), rcond=1e-12)
        mapper = c.blocks[frame] @ c.lift
        pinv = np.linalg.pinv(mapper, rcond=1e-10)
        theta_noise = pinv @ delta
        theta_signal = pinv @ signal
        rows.append(dict(
            frame=frame, n_shots=number,
            exact_norm=float(np.linalg.norm(exact)),
            exact_rms=float(np.sqrt(np.mean(exact ** 2))),
            exact_ratio=float(np.linalg.norm(exact)
                              / max(np.linalg.norm(delta), 1e-30)),
            signal_norm=float(np.linalg.norm(signal)),
            noise_norm=float(np.linalg.norm(delta)),
            signal_rms=float(np.sqrt(np.mean(signal ** 2))),
            noise_rms=float(np.sqrt(np.mean(delta ** 2))),
            raw_ratio=float(np.linalg.norm(signal)
                            / max(np.linalg.norm(delta), 1e-30)),
            whitened_signal=float(np.sqrt(signal @ whitened @ signal)),
            whitened_noise=float(np.sqrt(delta @ whitened @ delta)),
            theta_noise_norm=float(np.linalg.norm(theta_noise)),
            theta_signal_norm=float(np.linalg.norm(theta_signal)),
            h_noise_meh=1000.0 * float(c.h @ theta_noise),
            f_noise_meh=1000.0 * float(c.f @ theta_noise),
            h_signal_meh=1000.0 * float(c.h @ theta_signal),
            f_signal_meh=1000.0 * float(c.f @ theta_signal),
            cos_noise_h=cosine(theta_noise, c.h),
            cos_noise_f=cosine(theta_noise, c.f),
            cos_noise_signal=cosine(theta_noise, theta_signal),
        ))
    return rows


def command_frames(c, budgets, streams):
    records = []
    for stream in streams:
        _, data = sampled(c, stream, max(budgets))
        for budget in budgets:
            for row in frame_statistics(c, budget, stream, data):
                row.update(budget=budget, stream=stream)
                records.append(row)
            print(f"frames budget={budget} stream={stream} done", flush=True)
    fields = list(records[0])
    target = HERE / "frame_noise.csv"
    lines = [",".join(fields)]
    for record in records:
        lines.append(",".join(str(record[field]) for field in fields))
    target.write_text("\n".join(lines) + "\n")
    print("wrote", target)
    return records


def command_solve(c, arm, budget, stream):
    CACHE.mkdir(exist_ok=True)
    target = CACHE / f"{arm}_b{budget}_r{stream}.npz"
    if target.exists():
        print("EXISTS", target)
        return
    band.SOLVE_BAND = band.band_solver(arm == "spin")
    counts, data = sampled(c, stream, budget)
    raw = band.all_rows_shadow(c, data, counts)
    started = time.time()
    result = band.solve_original_raw(c, raw)
    assert result.status == "optimal", result.status
    scores = r.score(c, result)
    seconds = time.time() - started
    design = design_matrix(c)
    base_values = anchor_values(c)
    theta_raw = np.linalg.lstsq(design, raw.values - base_values, rcond=None)[0]
    np.savez_compressed(
        target, d2=result.d2, gamma=result.gamma,
        theta_move=old_coordinates(c, result.d2, c.base.d2),
        theta_raw=theta_raw, values=raw.values, counts=counts,
        scores=json.dumps(scores), arm=arm, budget=budget, stream=stream,
        seconds=seconds, maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(f"SOLVE {arm} b={budget} r={stream} D={scores['d2_error']:.5f} "
          f"H={scores['h_error_meh']:+.3f} F={scores['f_error_meh']:+.3f} "
          f"sec={seconds:.1f}", flush=True)


def old_coordinates(c, d2, reference):
    return c.geo["Z"].T @ (c.geo["scale"] * (d2[c.geo["rows"], c.geo["cols"]]
                                             - reference[c.geo["rows"],
                                                         c.geo["cols"]]))


def command_report(c):
    anchor = r.score(c, c.base)
    href, fref = c.href, c.fref
    rows = []
    for arm, folder in ARMS.items():
        for path in sorted(folder.glob("b*_r*.json")):
            record = json.loads(path.read_text())
            h, f, d2 = (record["h_error_meh"], record["f_error_meh"],
                        record["d2_error"])
            cache = CACHE / f"{arm if arm in ('spin', 'full') else 'full'}" \
                f"_b{int(record['budget'])}_r{int(record['stream'])}.npz"
            row = dict(arm=arm, budget=int(record["budget"]),
                       stream=int(record["stream"]), h_post=h, f_post=f,
                       d2_post=d2,
                       h_anchor=anchor["h_error_meh"],
                       f_anchor=anchor["f_error_meh"],
                       d2_anchor=anchor["d2_error"],
                       h_size_change=abs(h) - abs(anchor["h_error_meh"]),
                       f_size_change=abs(f) - abs(anchor["f_error_meh"]),
                       h_components=record.get("h_d2_meh"),
                       h_gamma=record.get("h_gamma_meh"),
                       f_on_top=record.get("f_on_top_meh"),
                       f_non_ontop=record.get("f_non_ontop_meh"),
                       cr_h=record.get("cr_h"),
                       cr_f_parts=record.get("cr_f_parts"))
            if cache.exists() and arm in ("spin", "full"):
                stored = np.load(cache)
                d2s, gs = stored["d2"], stored["gamma"]
                theta_move = stored["theta_move"]
                theta_err = theta_move - c.exact_z
                g_h = c.h
                g_f_anchor = gradient_f(c, c.base.d2, c.base.gamma) @ c.lift
                g_f_sol = gradient_f(c, d2s, gs) @ c.lift
                h_sol, f_sol = energies(c, d2s, gs)
                gx_h = gradient_h(c)
                gx_f_anchor = gradient_f(c, c.base.d2, c.base.gamma)
                gx_f_sol = gradient_f(c, d2s, gs)
                dx = (d2s - c.base.d2)[c.geo["rows"], c.geo["cols"]]
                dx_truth = (c.exact.exact_d2
                            - c.base.d2)[c.geo["rows"], c.geo["cols"]]
                ex = (d2s - c.exact.exact_d2)[c.geo["rows"], c.geo["cols"]]
                ex_anchor = (c.base.d2
                             - c.exact.exact_d2)[c.geo["rows"], c.geo["cols"]]
                tang = c.lift @ (c.geo["Z"].T
                                 @ (c.geo["scale"] * dx))
                row.update(
                    moved_norm_x=float(np.linalg.norm(dx)),
                    moved_tangent_fraction=float(np.linalg.norm(tang)
                                                 / np.linalg.norm(dx)),
                    truth_norm_x=float(np.linalg.norm(dx_truth)),
                    error_norm_x=float(np.linalg.norm(ex)),
                    error_anchor_norm_x=float(np.linalg.norm(ex_anchor)),
                    h_move_lin_meh=1000.0 * float(gx_h @ dx),
                    f_move_anchor_lin_meh=1000.0 * float(gx_f_anchor @ dx),
                    cos_dx_h=cosine(dx, gx_h),
                    cos_dx_f_anchor=cosine(dx, gx_f_anchor),
                    cos_dx_f_sol=cosine(dx, gx_f_sol),
                    cos_dx_truth=cosine(dx, dx_truth),
                    cos_dx_raw=cosine(dx, c.lift @ stored["theta_raw"]),
                    cos_err_anchor=cosine(ex, ex_anchor),
                    cos_errx_h=cosine(ex, gx_h),
                    cos_errx_f_sol=cosine(ex, gx_f_sol),
                    cos_anchor_err_h=cosine(ex_anchor, gx_h),
                    cos_anchor_err_f=cosine(ex_anchor, gx_f_anchor),
                    cos_g_h_f_anchor=cosine(gx_h, gx_f_anchor),
                    cos_g_h_f_sol=cosine(gx_h, gx_f_sol),
                    theta_move_norm=float(np.linalg.norm(theta_move)),
                    theta_err_norm=float(np.linalg.norm(theta_err)),
                    theta_truth_norm=float(np.linalg.norm(c.exact_z)),
                    theta_raw_norm=float(np.linalg.norm(stored["theta_raw"])),
                    h_move_meh=1000.0 * (h_sol - href)
                    - anchor["h_error_meh"],
                    f_move_meh=1000.0 * (f_sol - fref)
                    - anchor["f_error_meh"],
                    h_lin_meh=1000.0 * float(g_h @ theta_move),
                    f_lin_anchor_meh=1000.0 * float(g_f_anchor @ theta_move),
                    f_lin_sol_meh=1000.0 * float(g_f_sol @ theta_move),
                    cos_move_h=cosine(theta_move, g_h),
                    cos_move_f_anchor=cosine(theta_move, g_f_anchor),
                    cos_move_f_sol=cosine(theta_move, g_f_sol),
                    cos_move_truth=cosine(theta_move, c.exact_z),
                    cos_err_h=cosine(theta_err, g_h),
                    cos_err_f_sol=cosine(theta_err, g_f_sol),
                    cos_raw_move=cosine(stored["theta_raw"], theta_move),
                    cos_raw_truth=cosine(stored["theta_raw"], c.exact_z),
                    cos_h_f_anchor=cosine(g_h, g_f_anchor),
                    mosek_seconds=float(stored["seconds"]),
                    maxrss_kb=float(stored["maxrss"]))
            rows.append(row)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    lines = [",".join(fields)]
    for row in rows:
        lines.append(",".join("" if row.get(field) is None else str(row[field])
                              for field in fields))
    target = HERE / "solver_direction.csv"
    target.write_text("\n".join(lines) + "\n")
    print("wrote", target)
    print(f"anchor H={anchor['h_error_meh']:+.3f} F={anchor['f_error_meh']:+.3f} "
          f"D2={anchor['d2_error']:.5f} | cos(gH,gF)={cosine(c.h, gradient_f(c, c.base.d2, c.base.gamma) @ c.lift):+.3f}")
    pattern = [row for row in rows if row["h_size_change"] < 0
               and row["f_size_change"] > 0]
    print(f"H-size-down & F-size-up cases: {len(pattern)}/{len(rows)}")
    for row in sorted(pattern, key=lambda item: item["f_size_change"],
                      reverse=True):
        print(f"  {row['arm']:8s} b={row['budget']:6d} r={row['stream']} "
              f"|H| {abs(row['h_anchor']):.2f}->{abs(row['h_post']):.2f} "
              f"|F| {abs(row['f_anchor']):.2f}->{abs(row['f_post']):.2f}")
    frame_summary()
    return rows


def frame_summary():
    import csv
    source = HERE / "frame_noise.csv"
    if not source.exists():
        return
    records = list(csv.DictReader(source.open()))
    frames = sorted({int(record["frame"]) for record in records})
    budgets = sorted({int(record["budget"]) for record in records})
    lines = ["budget,frame,exact_ratio,correction_ratio,whitened_ratio,"
             "h_noise_mean_meh,h_noise_std_meh,f_noise_mean_meh,"
             "f_noise_std_meh,hf_opposite_fraction,cos_noise_h,"
             "cos_noise_f,h_signal_meh,f_signal_meh"]
    for budget in budgets:
        for frame in frames:
            group = [record for record in records
                     if int(record["budget"]) == budget
                     and int(record["frame"]) == frame]
            h = np.array([float(record["h_noise_meh"]) for record in group])
            f = np.array([float(record["f_noise_meh"]) for record in group])
            opposite = float(np.mean(h * f < 0))
            lines.append(",".join(str(value) for value in (
                budget, frame,
                np.mean([float(record["exact_ratio"]) for record in group]),
                np.mean([float(record["raw_ratio"]) for record in group]),
                np.mean([float(record["whitened_signal"])
                         / max(float(record["whitened_noise"]), 1e-30)
                         for record in group]),
                np.mean(h), np.std(h), np.mean(f), np.std(f), opposite,
                np.mean([float(record["cos_noise_h"]) for record in group]),
                np.mean([float(record["cos_noise_f"]) for record in group]),
                np.mean([float(record["h_signal_meh"]) for record in group]),
                np.mean([float(record["f_signal_meh"]) for record in group]))))
    target = HERE / "frame_summary.csv"
    target.write_text("\n".join(lines) + "\n")
    print("wrote", target)


def command_figure(c):
    import csv

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/original_bands_mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [row for row in csv.DictReader(
        (HERE / "solver_direction.csv").open()) if row["cos_dx_h"]]
    anchor = r.score(c, c.base)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    axis = axes[0]
    axis.axhline(0, color="#999999", linewidth=0.8)
    axis.axvline(0, color="#999999", linewidth=0.8)
    colors = {15000: "#D55E00", 30000: "#0072B2", 90000: "#009E73",
              120000: "#CC79A7", 60000: "#E69F00"}
    for row in rows:
        x0, y0 = anchor["f_error_meh"], anchor["h_error_meh"]
        x1, y1 = float(row["f_post"]), float(row["h_post"])
        color = colors.get(int(row["budget"]), "#555555")
        axis.annotate("", xy=(x1, y1), xytext=(x0, y0),
                      arrowprops=dict(arrowstyle="-|>", color=color,
                                      linewidth=1.6))
        axis.scatter([x1], [y1], color=color, s=18, zorder=3)
        axis.annotate(f"b{int(row['budget'])//1000}k r{row['stream']}",
                      (x1, y1), fontsize=6.5, xytext=(3, 3),
                      textcoords="offset points")
    axis.scatter([0], [0], marker="*", s=110, color="black", zorder=4,
                 label="exact")
    axis.scatter([anchor["f_error_meh"]], [anchor["h_error_meh"]],
                 marker="o", s=55, color="#555555", label="anchor (start)")
    axis.annotate("", xy=(0, 0),
                  xytext=(anchor["f_error_meh"], anchor["h_error_meh"]),
                  arrowprops=dict(arrowstyle="-|>", color="#555555",
                                  linestyle="--", linewidth=1.2))
    axis.set_xlabel("F (MC-PDFT) error / mHa")
    axis.set_ylabel("H error / mHa")
    axis.set_title("solver movement in the energy plane", fontsize=10)
    axis.legend(frameon=False, fontsize=8)
    axis.grid(alpha=0.25)

    axis = axes[1]
    labels = [f"b{int(row['budget'])//1000}k r{row['stream']}" for row in rows]
    metrics = (("cos_dx_truth", "move vs truth", "#009E73"),
               ("cos_dx_h", "move vs gH", "#0072B2"),
               ("cos_dx_f_anchor", "move vs gF", "#D55E00"),
               ("cos_err_anchor", "error vs anchor err", "#CC79A7"))
    positions = np.arange(len(rows))
    width = 0.2
    for index, (key, label, color) in enumerate(metrics):
        axis.bar(positions + index * width,
                 [float(row[key]) for row in rows], width, label=label,
                 color=color)
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xticks(positions + 1.5 * width)
    axis.set_xticklabels(labels, fontsize=7)
    axis.set_ylabel("direction cosine (2-RDM)")
    axis.set_title("2-RDM movement / error directions", fontsize=10)
    axis.legend(frameon=False, fontsize=7)
    axis.grid(alpha=0.25, axis="y")

    axis = axes[2]
    frame_rows = list(csv.DictReader((HERE / "frame_noise.csv").open()))
    for row in frame_rows:
        if int(row["budget"]) != 30000:
            continue
        axis.scatter(float(row["f_noise_meh"]), float(row["h_noise_meh"]),
                     s=10, alpha=0.5, color="#0072B2")
    axis.axhline(0, color="#999999", linewidth=0.8)
    axis.axvline(0, color="#999999", linewidth=0.8)
    axis.set_xlabel("per-frame noise -> F / mHa")
    axis.set_ylabel("per-frame noise -> H / mHa")
    axis.set_title("frame noise projections, B=30k (5 streams)",
                   fontsize=10)
    axis.grid(alpha=0.25)

    figure.tight_layout()
    path = HERE / "direction_summary.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("frames", "solve", "report",
                                            "figure"))
    parser.add_argument("--budgets", default="15000,30000,120000")
    parser.add_argument("--streams", default="0,1,2,3,4")
    parser.add_argument("--arm", choices=("spin", "full"), default="full")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--stream", type=int)
    arguments = parser.parse_args()
    c = setup()
    if arguments.command == "frames":
        command_frames(c, [int(v) for v in arguments.budgets.split(",")],
                       [int(v) for v in arguments.streams.split(",")])
    elif arguments.command == "solve":
        command_solve(c, arguments.arm, arguments.budget, arguments.stream)
    elif arguments.command == "figure":
        command_figure(c)
    else:
        command_report(c)


if __name__ == "__main__":
    main()
