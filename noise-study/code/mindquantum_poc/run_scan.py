#!/usr/bin/env python3
"""Noisy-circuit N2 bond scan (stream 0, guard15 vs uniform).

Phase ``sample`` : per geometry, sample noiseless and Wukong-calibrated noisy
                   outcome histograms for the frozen scan plans.
Phase ``solve``  : turn histograms into pair moments for the variants

    exact     oracle probabilities (no shots, no circuit)
    clean     noiseless circuit sampling
    raw       noisy circuit sampling
    post      noisy + particle-number post-selection
    rem       noisy + readout matrix inversion
    rem_post  noisy + readout inversion + post-selection

and run the frozen DQG estimator plus scoring for each variant.
Phase ``summarize`` aggregates the per-geometry records and plots the scan.
"""
from __future__ import annotations

import os

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
SCAN = ANALYSIS / "guard15_equal_real30_n2_scan120k"
for _path in (HERE, SCAN, ANALYSIS / "guard15_lammu"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import circuit_sampler as cs  # noqa: E402

BONDS = (0.80, 0.90, 1.00, 1.10, 1.25, 1.45, 1.60, 1.80, 2.00, 2.20, 2.50)
SYSTEMS = {f"n2_r{int(round(b * 100)):03d}": b for b in BONDS}
STREAM = 0
BUDGET = 120000
ARMS = ("guard15_equal", "uniform")
MU = {"guard15_equal": 2.0, "uniform": 0.0}
OUT = HERE / "results"
VARIANTS = (
    "exact",
    "clean",
    "raw",
    "post",
    "rem",
    "rem_lin",
    "rem_post",
    "rem_lin_post",
)
N_THREADS = 3


def results_dir(gate_scale: float) -> Path:
    if abs(gate_scale - 1.0) < 1e-12:
        return OUT
    return HERE / f"results_gate{int(round(gate_scale * 100)):03d}"


def seed_for(tag: str, arm: str, frame: int, noisy: bool) -> int:
    key = f"{tag}|{arm}|{frame}|{int(noisy)}|202610018100".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16) % (2**23)


def build_context(tag: str):
    """The scan geometry context, with the 1.10 A equilibrium computed too."""
    import engine as e

    g = e.g
    g.HERE = SCAN / "contexts"
    g.N2_SCAN.update(SYSTEMS)
    with np.load(SCAN / "pool_real30.npz") as archive:
        rotations = np.array(archive["rotations"])
    original = g.c108.pool_uniform_family
    g.c108.pool_uniform_family = lambda c: (rotations, np.zeros(30, dtype=bool))
    try:
        c = g.n2_geometry_context(tag)
    finally:
        g.c108.pool_uniform_family = original
    g.da.band.SOLVE_BAND = g.da.band.band_solver(True)
    return e.g.da, c


def plan_for(c, arm: str):
    if arm == "guard15_equal":
        path = SCAN / "plans" / c.system.replace("n2_108_", "n2_") / f"r{STREAM}_b{BUDGET}.json"
        record = json.loads(path.read_text())
        indices = np.asarray(record["indices"], dtype=int)
        counts = np.asarray(record["counts"], dtype=int)
        return indices, counts[indices]
    indices = np.arange(30, dtype=int)
    return indices, np.full(30, BUDGET // 30, dtype=int)


def sample_phase(
    tag: str,
    gate_scale: float = 1.0,
    limit_frames: int | None = None,
    shots_scale: float = 1.0,
) -> None:
    started = time.time()
    _d, c = build_context(tag)
    model = cs.load_model()
    initial = np.asarray(c.exact.statevector, dtype=complex)
    rotations = np.asarray(c.rotations)
    folder = results_dir(gate_scale) / tag
    folder.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        indices, shots = plan_for(c, arm)
        if limit_frames is not None:
            indices = indices[:limit_frames]
            shots = shots[:limit_frames]
        shots = np.maximum(np.rint(shots * shots_scale).astype(int), 100)
        noiseless = np.zeros((len(indices), 1 << cs.N_QUBITS), dtype=np.int32)
        noisy = np.zeros_like(noiseless)
        for position, frame in enumerate(indices):
            start = time.time()
            for target, is_noisy in ((noiseless, False), (noisy, True)):
                counts = cs.sample_histogram(
                    initial, np.real(rotations[frame][0]),
                    int(shots[position]),
                    is_noisy, seed_for(tag, arm, int(frame), is_noisy), model,
                    gate_scale,
                )
                target[position] = counts.astype(np.int32)
            print(
                f"[{tag} {arm} frame {int(frame)}] "
                f"{time.time() - start:.1f}s",
                flush=True,
            )
        np.savez_compressed(
            folder / f"{arm}_histograms.npz",
            noiseless=noiseless,
            noisy=noisy,
            indices=indices,
            shots=shots,
        )
        print(
            f"[{tag} {arm}] sampled {len(indices)} frames, "
            f"noisy shots {int(shots.sum())}, {time.time() - started:.0f}s total",
            flush=True,
        )
    (folder / "context_meta.json").write_text(
        json.dumps(
            {
                "tag": tag,
                "system": c.system,
                "bond_angstrom": SYSTEMS[tag],
                "stream": STREAM,
                "noise_model": model["source_sha256"],
                "path": model["logical_to_physical"],
                "exact_energy": float(c.exact.exact_energy),
            },
            indent=2,
        )
    )


def assemble(d, c, moments_by_frame, indices, shots_per_frame):
    offsets = np.cumsum([0] + [len(block) for block in c.blocks])
    rows = np.concatenate(
        [np.arange(offsets[f], offsets[f + 1]) for f in indices]
    )
    vectors = np.asarray(c.vectors)[rows]
    values = np.concatenate([moments_by_frame[int(f)] for f in indices])
    design = d.r.j.quadratic_design(vectors)[np.arange(len(values))].tocsr()
    shots_per_row = np.concatenate(
        [
            np.full(len(c.blocks[int(f)]), float(shots_per_frame[position]))
            for position, f in enumerate(indices)
        ]
    )
    hits = np.rint(values * shots_per_row).astype(int)
    hits = np.maximum(hits, 0)
    return d.band.vendor.ShadowData(
        rotations=tuple(np.asarray(c.rotations[int(f)]) for f in indices),
        pair_vectors=vectors,
        design=design,
        values=values,
        lower_bounds=values.copy(),
        upper_bounds=values.copy(),
        hits=hits,
        shots_per_basis=0,
        exact_values=np.full(len(values), np.nan),
        exact_constraints=False,
        occupations=None,
    )


def solve_variants(d, c, arm: str, origin_raw, values_by_variant: dict, threads: int):
    """One solver build per arm; the observation basis spans all variants."""
    from merit_solver import MeritSolver
    from reusable_band_solver import ReusableBandSolver

    origin = np.asarray(origin_raw.values, dtype=float).copy()
    basis = np.column_stack(
        [values_by_variant[name] - origin for name in values_by_variant]
    )
    if arm == "uniform":
        solver = ReusableBandSolver(
            c, origin_raw, "spin", band=d.band,
            observation_basis=basis, solver_threads=threads,
        )
        solve = lambda values: (solver.solve(values), solver.last_stats)
    else:
        solver = MeritSolver(d, c, origin_raw, basis, threads=threads)
        solve = lambda values: (solver.solve(values, 1.0, MU[arm]), solver.stats)
    return solve


def solve_phase(tag: str, gate_scale: float = 1.0) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (20 * 1024**3, 20 * 1024**3))
    started = time.time()
    d, c = build_context(tag)
    model = cs.load_model()
    pairs = c.exact.pairs
    indicators = cs.indicator_matrix(pairs).astype(float)
    mask = cs.sector_mask()
    folder = results_dir(gate_scale) / tag
    (folder / "variants").mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        archive = np.load(folder / f"{arm}_histograms.npz")
        indices = np.asarray(archive["indices"], dtype=int)
        shots = np.asarray(archive["shots"], dtype=float)
        noiseless = np.asarray(archive["noiseless"], dtype=float)
        noisy = np.asarray(archive["noisy"], dtype=float)
        moments = {name: {} for name in VARIANTS}
        acceptance = {"post": None, "rem_lin_post": None}
        for position, frame in enumerate(indices):
            frame = int(frame)
            clean_probability = noiseless[position] / noiseless[position].sum()
            noisy_probability = noisy[position] / noisy[position].sum()
            raw_moments = noisy_probability @ indicators
            post_probability, accepted = cs.postselect(noisy_probability, mask)
            rem_probability = cs.readout_invert(noisy_probability, model)
            rem_post_probability, rem_accepted = cs.postselect(rem_probability, mask)
            linear = cs.readout_apply(noisy_probability, model)
            rem_lin_moments = linear @ indicators
            linear_post = np.where(mask, linear, 0.0)
            rem_lin_post_moments = (
                linear_post / np.sum(linear_post)
            ) @ indicators
            moments["clean"][frame] = clean_probability @ indicators
            moments["raw"][frame] = raw_moments
            moments["post"][frame] = post_probability @ indicators
            moments["rem"][frame] = rem_probability @ indicators
            moments["rem_post"][frame] = rem_post_probability @ indicators
            moments["rem_lin"][frame] = rem_lin_moments
            moments["rem_lin_post"][frame] = rem_lin_post_moments
            acceptance["post"] = (
                accepted if acceptance["post"] is None
                else 0.5 * (acceptance["post"] + accepted)
            )
            acceptance["rem_lin_post"] = (
                rem_accepted
                if acceptance["rem_lin_post"] is None
                else 0.5 * (acceptance["rem_lin_post"] + rem_accepted)
            )
        moments["exact"] = {
            int(f): np.asarray(c.oracle._frame_probabilities(int(f)))
            @ np.asarray(c.oracle.indicators, dtype=float)
            for f in indices
        }

        pending = [
            name
            for name in VARIANTS
            if not (
                folder / "variants" / f"{arm}_{name}.json"
            ).exists()
        ]
        if not pending:
            print(f"[{tag} {arm}] all variants cached", flush=True)
            continue
        origin_raw = assemble(d, c, moments["raw"], indices, shots)
        vectors = {
            name: assemble(d, c, moments[name], indices, shots).values
            for name in pending
        }
        solve = solve_variants(d, c, arm, origin_raw, vectors, N_THREADS)
        for name, values in vectors.items():
            result, stats = solve(values)
            scores = d.r.score(c, result)
            record = dict(
                system=tag,
                bond_angstrom=SYSTEMS[tag],
                arm=arm,
                variant=name,
                gate_scale=gate_scale,
                stream=STREAM,
                budget=BUDGET,
                selected_frames=len(indices),
                fit_shots=int(shots.sum()),
                mu_ftpbe=MU[arm],
                acceptance_post=(
                    acceptance["post"]
                    if name in ("post", "rem_post")
                    else acceptance["rem_lin_post"]
                    if name in ("rem_lin_post",)
                    else None
                ),
                solver_stats=stats,
                prime_seconds=round(time.time() - started, 1),
                **scores,
            )
            path = folder / "variants" / f"{arm}_{name}.json"
            np.savez_compressed(
                path.with_suffix(".npz"),
                values=values,
                d2=result.d2,
                gamma=result.gamma,
            )
            path.write_text(json.dumps(record, indent=2))
            print(
                f"[{tag} {arm} {name}] H={scores['h_error_meh']:.4f} "
                f"F={scores['f_error_meh']:.4f} D2={scores['d2_error']:.4f}",
                flush=True,
            )
        print(
            f"[{tag} {arm}] acceptance post={acceptance['post']:.4f} "
            f"rem_lin_post={acceptance['rem_lin_post']:.4f}",
            flush=True,
        )


def _load_rows(directory: Path, gate_scale: float) -> list[dict]:
    rows = []
    for tag in SYSTEMS:
        for arm in ARMS:
            for variant in VARIANTS:
                path = directory / tag / "variants" / f"{arm}_{variant}.json"
                if not path.exists():
                    continue
                record = json.loads(path.read_text())
                rows.append(
                    dict(
                        system=tag,
                        bond_angstrom=SYSTEMS[tag],
                        arm=arm,
                        variant=variant,
                        gate_scale=gate_scale,
                        h_error_meh=record["h_error_meh"],
                        f_error_meh=record["f_error_meh"],
                        d2_error=record["d2_error"],
                        acceptance_post=record.get("acceptance_post"),
                    )
                )
    return rows


def _aggregate(rows: list[dict], gate_scale: float) -> list[dict]:
    aggregate = []
    for arm in ARMS:
        for variant in VARIANTS:
            subset = [r for r in rows if r["arm"] == arm and r["variant"] == variant]
            if not subset:
                continue
            h = np.array([r["h_error_meh"] for r in subset])
            f = np.array([r["f_error_meh"] for r in subset])
            d2 = np.array([r["d2_error"] for r in subset])
            accepted = [
                r["acceptance_post"]
                for r in subset
                if r["acceptance_post"] is not None
            ]
            aggregate.append(
                dict(
                    gate_scale=gate_scale,
                    arm=arm,
                    variant=variant,
                    geometries=len(subset),
                    h_rmse_meh=float(np.sqrt(np.mean(h**2))),
                    h_bias_meh=float(np.mean(h)),
                    f_rmse_meh=float(np.sqrt(np.mean(f**2))),
                    f_bias_meh=float(np.mean(f)),
                    d2_mean=float(np.mean(d2)),
                    d2_rms=float(np.sqrt(np.mean(d2**2))),
                    acceptance_post=float(np.mean(accepted)) if accepted else None,
                )
            )
    return aggregate


def summarize(scales=(1.0,)) -> None:
    import csv

    all_aggregate = []
    for scale in scales:
        directory = results_dir(scale)
        rows = _load_rows(directory, scale)
        aggregate = _aggregate(rows, scale)
        for name, table in (("summary", rows), ("summary_aggregate", aggregate)):
            (directory / f"{name}.json").write_text(json.dumps(table, indent=2))
            if table:
                with (directory / f"{name}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                    writer.writeheader()
                    writer.writerows(table)
        print(f"[scale {scale}] summarized {len(rows)} records", flush=True)
        for row in aggregate:
            print(
                f"  {row['arm']:14} {row['variant']:11} "
                f"H_RMSE={row['h_rmse_meh']:9.3f} F_RMSE={row['f_rmse_meh']:9.3f} "
                f"D2={row['d2_mean']:.4f}",
                flush=True,
            )
        if rows:
            _plot(directory, rows, aggregate)
        all_aggregate += aggregate
    (OUT / "summary_by_scale.json").write_text(
        json.dumps(all_aggregate, indent=2)
    )
    if all_aggregate:
        with (OUT / "summary_by_scale.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_aggregate[0]))
            writer.writeheader()
            writer.writerows(all_aggregate)
    _plot_sweep(all_aggregate, scales)


def _plot(directory: Path, rows, aggregate) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = VARIANTS
    colors = {
        "exact": "black",
        "clean": "tab:green",
        "raw": "tab:red",
        "post": "tab:orange",
        "rem": "tab:blue",
        "rem_lin": "tab:cyan",
        "rem_post": "tab:purple",
        "rem_lin_post": "tab:brown",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row_index, arm in enumerate(ARMS):
        for column, (metric, label) in enumerate(
            (
                ("h_error_meh", "H error (mHa)"),
                ("f_error_meh", "F error (mHa)"),
                ("d2_error", "2-RDM Frobenius error"),
            )
        ):
            axis = axes[row_index, column]
            for variant in variants:
                subset = sorted(
                    (r for r in rows if r["arm"] == arm and r["variant"] == variant),
                    key=lambda r: r["bond_angstrom"],
                )
                if len(subset) != len(SYSTEMS):
                    continue
                axis.plot(
                    [r["bond_angstrom"] for r in subset],
                    [r[metric] for r in subset],
                    marker="o" if variant != "exact" else None,
                    ls="--" if variant == "exact" else "-",
                    color=colors[variant],
                    label=variant,
                    alpha=0.85,
                )
            axis.set(
                xlabel="N2 bond length (Angstrom)",
                ylabel=label,
                title=f"{arm}",
            )
            axis.grid(alpha=0.25)
            if row_index == 0 and column == 0:
                axis.legend(fontsize=8, ncol=2)
    fig.savefig(directory / "noise_scan.png", dpi=180)
    fig.savefig(directory / "noise_scan.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for axis, arm in zip(axes, ARMS):
        subset = [a for a in aggregate if a["arm"] == arm]
        positions = np.arange(len(subset))
        axis.bar(
            positions - 0.2, [a["h_rmse_meh"] for a in subset], 0.4,
            label="H RMSE", color="tab:blue",
        )
        axis.bar(
            positions + 0.2, [a["f_rmse_meh"] for a in subset], 0.4,
            label="F RMSE", color="tab:orange",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels([a["variant"] for a in subset], rotation=30, ha="right")
        axis.set(ylabel="RMSE (mHa)", title=f"{arm}: aggregate over 11 geometries")
        axis.grid(alpha=0.25, axis="y")
        axis.legend()
    fig.savefig(directory / "noise_scan_aggregate.png", dpi=180)
    fig.savefig(directory / "noise_scan_aggregate.pdf")
    plt.close(fig)


def _plot_sweep(aggregate, scales) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = ("raw", "post", "rem", "rem_lin", "rem_lin_post")
    colors = {
        "raw": "tab:red",
        "post": "tab:orange",
        "rem": "tab:blue",
        "rem_lin": "tab:cyan",
        "rem_lin_post": "tab:brown",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for row_index, arm in enumerate(ARMS):
        for column, (metric, label) in enumerate(
            (
                ("h_rmse_meh", "H RMSE (mHa)"),
                ("f_rmse_meh", "F RMSE (mHa)"),
            )
        ):
            axis = axes[row_index, column]
            for variant in variants:
                points = sorted(
                    (
                        a
                        for a in aggregate
                        if a["arm"] == arm and a["variant"] == variant
                    ),
                    key=lambda a: a["gate_scale"],
                )
                if not points:
                    continue
                axis.plot(
                    [p["gate_scale"] for p in points],
                    [p[metric] for p in points],
                    "o-",
                    color=colors[variant],
                    label=variant,
                )
            clean = [
                a
                for a in aggregate
                if a["arm"] == arm and a["variant"] == "clean"
            ]
            if clean:
                axis.axhline(
                    clean[0][metric], color="tab:green", ls="--", label="clean"
                )
            axis.set(
                xlabel="gate-noise scale (1.0 = Wukong calibration)",
                ylabel=label,
                title=arm,
            )
            axis.grid(alpha=0.25)
            if row_index == 0 and column == 0:
                axis.legend(fontsize=8)
    fig.savefig(OUT / "noise_scan_sweep.png", dpi=180)
    fig.savefig(OUT / "noise_scan_sweep.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sample", "solve", "summarize"))
    parser.add_argument("--tag", choices=tuple(SYSTEMS))
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--shots-scale", type=float, default=1.0)
    parser.add_argument("--gate-scale", type=float, default=1.0)
    parser.add_argument("--scales", type=str, default="1.0")
    args = parser.parse_args()
    if args.command == "sample":
        if args.tag is None:
            parser.error("sample requires --tag")
        sample_phase(
            args.tag, args.gate_scale, args.limit_frames, args.shots_scale
        )
    elif args.command == "solve":
        if args.tag is None:
            parser.error("solve requires --tag")
        solve_phase(args.tag, args.gate_scale)
    else:
        summarize(tuple(float(s) for s in args.scales.split(",")))


if __name__ == "__main__":
    main()
