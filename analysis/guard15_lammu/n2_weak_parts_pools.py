#!/usr/bin/env python3
"""N2 (10e,8o) R=1.10: four-target weak-parts rule on pure same-spin pools.

Pools (30 frames each, no asymmetric frame):
  real30    30 Haar O(8) rotations, applied identically to alpha and beta
  complex30 30 Haar U(8) rotations, applied identically to alpha and beta

For every pool the four weak targets of ``weak_parts_rule`` are selected and
solved at 120k (spin error basis, merit H + Tr(E), i.e. lambda,mu = (1,0));
the no-pilot uniform allocation of the same pool is the baseline.  Streams
sample with ``direction_analysis.production_seed`` so arm pilots are prefixes
of the production draws.

    python n2_weak_parts_pools.py check
    python n2_weak_parts_pools.py run --pools real30 --streams 0,1,2,3,4,5,6,7
    python n2_weak_parts_pools.py summarize
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gl15 as g  # noqa: E402
import weak_parts_rule as rule  # noqa: E402
from merit_solver import MeritSolver  # noqa: E402
from subset_measurements import selected_raw  # noqa: E402

OUT = HERE / "weak_parts"
POOLS_DIR = OUT / "pools"
SYSTEM = "n2"
BUDGET = 120000
REAL_SEED = 20260716
COMPLEX_SEED = 271828
POOLS = ("real30", "complex30")
TARGETS = rule.TARGETS
STREAMS = tuple(range(8))
LAMBDA = 1.0
MU = 0.0


def pool_frames(name):
    path = POOLS_DIR / f"{name}.npz"
    if path.exists():
        with np.load(path) as archive:
            return np.asarray(archive["rotations"])
    n_spatial = 8
    if name == "real30":
        from constrained_shadow import random_orthogonal_rotations
        base = random_orthogonal_rotations(n_spatial, rule.FRAMES, REAL_SEED)
        kind, seed = "real_same_spin", REAL_SEED
    elif name == "complex30":
        from run_n2_paper_nuclear_hybrid_trajectories import _random_unitaries
        base = _random_unitaries(n_spatial, rule.FRAMES, COMPLEX_SEED)
        kind, seed = "complex_same_spin", COMPLEX_SEED
    else:
        raise ValueError(f"Unknown pool {name}")
    frames = np.asarray([np.stack((u, u)) for u in base], dtype=complex)
    POOLS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rotations=frames, kind=kind, seed=seed,
                        digest=g.allocation.digest(frames))
    return frames


def install_pool(c, name):
    frames = pool_frames(name)
    g.c108.install_uniform_family(c, frames)
    return frames


def arm_path(pool, target, stream, budget=BUDGET):
    return OUT / pool / target / f"b{budget}_r{stream}.json"


def uniform_path(pool, stream, budget=BUDGET):
    return OUT / pool / "uniform" / f"b{budget}_r{stream}.json"


def save_record(path, record, arrays):
    g.allocation.save(path, record)
    g.atomic_npz(path.with_suffix(".npz"), **arrays)


def solve_arm(d, c, data, selection, threads):
    counts = np.asarray(selection["counts"], dtype=int)
    indices = np.asarray(selection["indices"], dtype=int)
    raw = selected_raw(d, c, data, counts, indices)
    solver = MeritSolver(d, c, raw, np.ones((len(raw.values), 1)), threads=threads)
    result = solver.solve(raw.values, LAMBDA, MU)
    if result.status != "optimal":
        raise RuntimeError(result.status)
    scores = d.r.score(c, result)
    payload = dict(d2=result.d2, gamma=result.gamma, values=raw.values,
                   counts=counts, indices=indices)
    stats = dict(solver.stats)
    del solver, raw
    gc.collect()
    return scores, payload, stats


def run_stream(d, c, pool, stream, targets, budget, threads):
    pending = [t for t in targets if not arm_path(pool, t, stream, budget).exists()]
    uniform_pending = not uniform_path(pool, stream, budget).exists()
    if not pending and not uniform_pending:
        print("EXISTS", pool, stream, flush=True)
        return
    started = time.time()
    model = rule.build_model(c, d, stream)
    selections = {t: rule.select(model, c, t, budget) for t in pending}
    if pending:
        maximum = np.max(np.asarray([selections[t]["counts"] for t in pending]), axis=0)
        data = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream), maximum)
        for target in pending:
            selection = selections[target]
            scores, payload, stats = solve_arm(d, c, data, selection, threads)
            record = dict(pool=pool, stream=stream, budget=budget,
                          system=SYSTEM, rule="weak_parts_v1",
                          error_basis="spin", lambda_radius=LAMBDA, mu_ftpbe=MU,
                          pilot_shots=rule.FRAMES * rule.PILOT,
                          seed=int(d.production_seed(stream)),
                          status="optimal", solver_stats=stats,
                          seconds=time.time() - started)
            record.update(selection)
            record.update(scores)
            save_record(arm_path(pool, target, stream, budget), record, payload)
            print("ARM %s %-7s r%d K %d H %+.3f F %+.3f D2 %.4f guard %s"
                  % (pool, target, stream, selection["k"], scores["h_error_meh"],
                     scores["f_error_meh"], scores["d2_error"],
                     selection["lineality_guard_ratio"]), flush=True)
        del data
        gc.collect()
    if uniform_pending:
        counts = np.full(rule.FRAMES, budget // rule.FRAMES, dtype=int)
        data = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream), counts)
        raw = d.band.all_rows_shadow(c, data, counts)
        result = d.band.solve_original_raw(c, raw)
        if result.status != "optimal":
            raise RuntimeError(result.status)
        scores = d.r.score(c, result)
        record = dict(pool=pool, target="uniform", stream=stream, budget=budget,
                      system=SYSTEM, rule="uniform_no_pilot", error_basis="spin",
                      lambda_radius=LAMBDA, mu_ftpbe=MU, pilot_shots=0,
                      seed=int(d.production_seed(stream)), status="optimal",
                      counts=counts.tolist(), seconds=time.time() - started, **scores)
        save_record(uniform_path(pool, stream, budget), record,
                    dict(d2=result.d2, gamma=result.gamma))
        print("UNI %s r%d H %+.3f F %+.3f D2 %.4f"
              % (pool, stream, scores["h_error_meh"], scores["f_error_meh"],
                 scores["d2_error"]), flush=True)
        del raw, result, data
        gc.collect()


def command_check(pools):
    d, c = g.context(SYSTEM)
    d.band.SOLVE_BAND = d.band.band_solver(True)
    for pool in pools:
        frames = install_pool(c, pool)
        model = rule.build_model(c, d, 0)
        print("POOL", pool, "frames", frames.shape, "kind",
              "real" if np.max(np.abs(frames.imag)) == 0 else "complex",
              "rank", model["pool_rank"], flush=True)
        for key, value in model["channel_diagnostics"].items():
            print("  channel", key, value, flush=True)
        print("  lineality", model["lineality_diagnostics"]["constraint_rank"],
              model["lineality_diagnostics"]["lineality_dimension"],
              model["lineality_diagnostics"]["spectra"], flush=True)
        for target in TARGETS:
            try:
                selection = rule.select(model, c, target, BUDGET)
            except (RuntimeError, ValueError) as error:
                print("  target %-7s unavailable: %s" % (target, error), flush=True)
                continue
            print("  target %-7s k %2d k_floor %2d cap %.3f lin %s guard %s"
                  % (target, selection["k"], selection["k_floor"],
                     selection["captured_fraction"], selection["lineality_active"],
                     selection["lineality_guard_ratio"]), flush=True)


def command_run(pools, streams, targets, threads):
    d, c = g.context(SYSTEM)
    d.band.SOLVE_BAND = d.band.band_solver(True)
    for pool in pools:
        install_pool(c, pool)
        print("RUNPOOL", pool, "rank",
              rule.design_rank(np.asarray([b @ c.lift for b in c.blocks])), flush=True)
        for stream in streams:
            run_stream(d, c, pool, stream, targets, BUDGET, threads)


def _rmse(values):
    return float(np.sqrt(np.mean(np.square(values))))


def command_summarize(pools, streams):
    rows = []
    for pool in pools:
        present = [s for s in streams if uniform_path(pool, s).exists()]
        if not present:
            continue
        uniform = [json.loads(uniform_path(pool, s).read_text()) for s in present]
        base_h = np.asarray([r["h_error_meh"] for r in uniform])
        base_f = np.asarray([r["f_error_meh"] for r in uniform])
        base_d2 = np.asarray([r["d2_error"] for r in uniform])
        for target in TARGETS:
            files = [arm_path(pool, target, s) for s in present]
            if not all(p.exists() for p in files):
                continue
            arm = [json.loads(p.read_text()) for p in files]
            ah = np.asarray([r["h_error_meh"] for r in arm])
            af = np.asarray([r["f_error_meh"] for r in arm])
            ad2 = np.asarray([r["d2_error"] for r in arm])
            loss_arm = ah ** 2 + af ** 2
            loss_base = base_h ** 2 + base_f ** 2
            rows.append(dict(
                pool=pool, target=target, streams=len(present),
                k=float(np.mean([r["k"] for r in arm])),
                h_rmse=_rmse(ah), f_rmse=_rmse(af), d2=float(ad2.mean()),
                h_bias=float(ah.mean()), f_bias=float(af.mean()),
                joint_mse=float(loss_arm.mean()),
                uniform_h_rmse=_rmse(base_h), uniform_f_rmse=_rmse(base_f),
                uniform_d2=float(base_d2.mean()),
                uniform_joint_mse=float(loss_base.mean()),
                delta_loss=float((loss_arm - loss_base).mean()),
                wins=int(np.count_nonzero(loss_arm < loss_base))))
    fields = ["pool", "target", "streams", "k", "h_rmse", "f_rmse", "d2",
              "h_bias", "f_bias", "joint_mse", "uniform_h_rmse",
              "uniform_f_rmse", "uniform_d2", "uniform_joint_mse",
              "delta_loss", "wins"]
    target = OUT / "summary.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print("%s %-7s n=%d K=%.1f | arm H %.2f F %.2f D2 %.4f | unif H %.2f "
              "F %.2f D2 %.4f | dL %.2f wins %d/%d"
              % (row["pool"], row["target"], row["streams"], row["k"],
                 row["h_rmse"], row["f_rmse"], row["d2"],
                 row["uniform_h_rmse"], row["uniform_f_rmse"], row["uniform_d2"],
                 row["delta_loss"], row["wins"], row["streams"]), flush=True)
    print("wrote", target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "summarize"))
    parser.add_argument("--pools", default=",".join(POOLS))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--streams", default=",".join(map(str, STREAMS)))
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args()
    pools = tuple(arguments.pools.split(","))
    streams = tuple(int(v) for v in arguments.streams.split(","))
    targets = tuple(arguments.targets.split(","))
    if arguments.command == "check":
        command_check(pools)
    elif arguments.command == "run":
        command_run(pools, streams, targets, arguments.threads)
    else:
        command_summarize(pools, streams)


if __name__ == "__main__":
    main()
