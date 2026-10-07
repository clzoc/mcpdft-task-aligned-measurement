#!/usr/bin/env python3
"""N2 (10e,8o) R=1.10: guard15 selection on pure same-spin pools.

Same pools, pilot seeds, budget accounting and (lambda,mu)=(1,0) solver as
``n2_weak_parts_pools``; the only change is the frame rule: K=15 point on the
pilot-only backward-greedy guarded path (``gl15.guard_selection``) instead of
the four weak-parts targets.  The no-pilot uniform baselines of both pools are
reused from ``weak_parts/<pool>/uniform``.

    python n2_guard15_pools.py check
    python n2_guard15_pools.py run --pools real30,complex30 --streams 0,1,2,3,4,5,6,7
    python n2_guard15_pools.py summarize
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
import n2_weak_parts_pools as wp  # noqa: E402
import weak_parts_rule as rule  # noqa: E402
from merit_solver import MeritSolver  # noqa: E402
from subset_measurements import selected_raw  # noqa: E402

OUT = HERE / "guard15_pools"
WEAK = HERE / "weak_parts"
SYSTEM = "n2"
BUDGET = 120000
LAMBDA = 1.0
MU = 0.0
TARGET = "guard15"
POOLS = ("real30", "complex30")
STREAMS = tuple(range(8))


def install_pool(c, name):
    frames = wp.pool_frames(name)
    g.c108.install_uniform_family(c, frames)
    return frames


def arm_path(pool, stream, budget=BUDGET):
    return OUT / pool / TARGET / f"b{budget}_r{stream}.json"


def uniform_path(pool, stream, budget=BUDGET):
    return WEAK / pool / "uniform" / f"b{budget}_r{stream}.json"


def identifiable_basis(designs):
    """Row-space basis so a rank-deficient pool is screened in its affine space."""
    stacked = np.vstack(designs)
    _, singular, vt = np.linalg.svd(stacked, full_matrices=False)
    cut = singular[0] * 1e-10 if singular.size else 0.0
    keep = singular > cut
    return vt.T[:, keep], int(keep.sum())


def build_model(c, d, stream):
    counts = np.full(g.FRAME_COUNT, g.PILOT, dtype=int)
    pilot = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream), counts)
    covariances = np.asarray([d.r.robust_covariance(c, sample) for sample in pilot])
    designs = np.asarray([block @ c.lift for block in c.blocks])
    basis, rank = identifiable_basis(designs)
    reduced = np.asarray([block @ basis for block in designs])
    return dict(designs=reduced, covariances=covariances,
                h=basis.T @ c.h, f=basis.T @ c.f, pool_rank=rank,
                full_designs=designs, h_full=c.h, f_full=c.f)


def subset_risks(model, indices):
    designs = model["full_designs"]
    covariances = model["covariances"]
    if model["pool_rank"] == g.AFFINE_RANK:
        return g.allocation.build_risks(designs[indices], covariances[indices],
                                        model["h_full"], model["f_full"])
    return rule.risk_model(designs[indices], covariances[indices],
                           model["h_full"], model["f_full"], model["pool_rank"])


def guard_plan(model, budget=BUDGET):
    chosen = g.guard_selection(model, budget)
    indices = np.asarray(chosen["indices"], dtype=int)
    risks = subset_risks(model, indices)
    if risks["rank"] != model["pool_rank"]:
        raise ValueError(f"K={g.K} subset loses rank {model['pool_rank']}")
    unused = (g.FRAME_COUNT - g.K) * g.PILOT
    measured = budget - unused
    fit_counts, diagnostic = g.allocation.allocate(
        risks, measured, np.full(g.K, g.PILOT, dtype=int), "joint")
    counts = np.full(g.FRAME_COUNT, g.PILOT, dtype=int)
    counts[indices] = fit_counts
    if int(counts.sum()) != budget:
        raise ValueError(f"counts sum {int(counts.sum())} != budget {budget}")
    return dict(target=TARGET, k=int(g.K), indices=indices.tolist(),
                counts=counts.tolist(), fit_counts=np.asarray(fit_counts).tolist(),
                measured_fit_shots=int(measured), unused_pilot_shots=int(unused),
                pilot_shots=int(g.FRAME_COUNT * g.PILOT),
                pool_rank=int(model["pool_rank"]),
                guard_proxy={key: chosen[key] for key in
                             ("worst", "plane", "d2", "h", "f")},
                allocation_diagnostic=diagnostic)


def run_stream(d, c, pool, stream, budget, threads):
    path = arm_path(pool, stream, budget)
    if path.exists():
        print("EXISTS", pool, stream, flush=True)
        return
    started = time.time()
    model = build_model(c, d, stream)
    plan = guard_plan(model, budget)
    counts = np.asarray(plan["counts"], dtype=int)
    indices = np.asarray(plan["indices"], dtype=int)
    data = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream), counts)
    raw = selected_raw(d, c, data, counts, indices)
    solver = MeritSolver(d, c, raw, np.ones((len(raw.values), 1)), threads=threads)
    result = solver.solve(raw.values, LAMBDA, MU)
    if result.status != "optimal":
        raise RuntimeError(result.status)
    scores = d.r.score(c, result)
    record = dict(pool=pool, target=TARGET, stream=stream, budget=budget,
                  system=SYSTEM, rule="guard15", error_basis="spin",
                  lambda_radius=LAMBDA, mu_ftpbe=MU,
                  seed=int(d.production_seed(stream)), status="optimal",
                  solver_stats=dict(solver.stats), seconds=time.time() - started)
    record.update(plan)
    record.update(scores)
    g.allocation.save(path, record)
    g.atomic_npz(path.with_suffix(".npz"), d2=result.d2, gamma=result.gamma,
                 values=raw.values, counts=counts, indices=indices)
    print("ARM %s %-7s r%d K %d H %+.3f F %+.3f D2 %.4f guard %s"
          % (pool, TARGET, stream, plan["k"], scores["h_error_meh"],
             scores["f_error_meh"], scores["d2_error"],
             plan["guard_proxy"]["worst"]), flush=True)
    del solver, raw, data, result, model
    gc.collect()


def command_check(pools):
    d, c = g.context(SYSTEM)
    d.band.SOLVE_BAND = d.band.band_solver(True)
    for pool in pools:
        frames = install_pool(c, pool)
        model = build_model(c, d, 0)
        plan = guard_plan(model)
        proxy = plan["guard_proxy"]
        print("POOL", pool, "frames", frames.shape, "rank", model["pool_rank"],
              "indices", plan["indices"], flush=True)
        print("  proxy worst %.4f plane %.4f d2 %.4f h %.4f f %.4f"
              % (proxy["worst"], proxy["plane"], proxy["d2"], proxy["h"], proxy["f"]),
              flush=True)
        print("  fit_counts", plan["fit_counts"], "sum", sum(plan["fit_counts"]),
              flush=True)


def command_run(pools, streams, threads):
    d, c = g.context(SYSTEM)
    d.band.SOLVE_BAND = d.band.band_solver(True)
    for pool in pools:
        install_pool(c, pool)
        print("RUNPOOL", pool, flush=True)
        for stream in streams:
            run_stream(d, c, pool, stream, BUDGET, threads)


def _rmse(values):
    return float(np.sqrt(np.mean(np.square(values))))


def command_summarize(pools, streams):
    rows = []
    for pool in pools:
        present = [s for s in streams
                   if arm_path(pool, s).exists() and uniform_path(pool, s).exists()]
        if not present:
            continue
        uniform = [json.loads(uniform_path(pool, s).read_text()) for s in present]
        base_h = np.asarray([r["h_error_meh"] for r in uniform])
        base_f = np.asarray([r["f_error_meh"] for r in uniform])
        base_d2 = np.asarray([r["d2_error"] for r in uniform])
        arm = [json.loads(arm_path(pool, s).read_text()) for s in present]
        ah = np.asarray([r["h_error_meh"] for r in arm])
        af = np.asarray([r["f_error_meh"] for r in arm])
        ad2 = np.asarray([r["d2_error"] for r in arm])
        loss_arm = ah ** 2 + af ** 2
        loss_base = base_h ** 2 + base_f ** 2
        rows.append(dict(
            pool=pool, target=TARGET, streams=len(present),
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
    parser.add_argument("--streams", default=",".join(map(str, STREAMS)))
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    pools = tuple(arguments.pools.split(","))
    streams = tuple(int(v) for v in arguments.streams.split(","))
    if arguments.command == "check":
        command_check(pools)
    elif arguments.command == "run":
        command_run(pools, streams, arguments.threads)
    else:
        command_summarize(pools, streams)


if __name__ == "__main__":
    main()
