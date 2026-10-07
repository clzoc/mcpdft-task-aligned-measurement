#!/usr/bin/env python3
"""CO (10e,8o) R=1.80 cc-pVDZ: weak-parts five-target rule on a 30-frame real
same-spin pool, merit ``(lambda, mu) = (1, 2)``, budget 120k, vs no-pilot
uniform.

Frame pool: 30 Haar O(8) real rotations applied identically to alpha and beta,
same seed convention as the N2 ``real30`` pool.  Selection and allocation use
``weak_parts_rule`` exactly as for N2; only the merit weights differ.

    python co_weak_parts.py check
    python co_weak_parts.py run --streams 0,1,2,3,4,5,6,7
    python co_weak_parts.py summarize
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import tempfile

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/co_weak_parts_mpl")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
for _path in (HERE,
              ANALYSIS / "outputs" / "physical_frame_design",
              ANALYSIS / "outputs" / "mcpdft_cancellation_universality",
              ANALYSIS / "mcpdft_measurement_revision"
              / "guarded_mcpdft_shadow_protocol" / "code"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import gl15 as g  # noqa: E402
import weak_parts_rule as rule  # noqa: E402
from merit_solver import MeritSolver  # noqa: E402
from subset_measurements import selected_raw  # noqa: E402

OUT = HERE / "co_weak_parts"
POOL = "real30"
SYSTEM = "co_r180"
BOND = 1.80
BUDGET = 120000
REAL_SEED = 20260716
LAMBDA = 1.0
MU = 2.0
TARGETS = rule.TARGETS
STREAMS = tuple(range(8))


def real_frames():
    from constrained_shadow import random_orthogonal_rotations
    n_spatial = 8
    base = random_orthogonal_rotations(n_spatial, rule.FRAMES, REAL_SEED)
    return np.asarray([np.stack((u, u)) for u in base], dtype=complex)


def co_context(tag=SYSTEM, bond=BOND):
    import experiment as ex
    import cancellation_probe as cp
    import mcpdft_derandomization as md
    spec = dict(name=tag,
                atom=f"C 0 0 {-bond / 2:.6f}; O 0 0 {bond / 2:.6f}",
                basis="cc-pvdz", active_electrons=10, active_orbitals=8,
                ncore=2, bond_length=bond, frame_count=30,
                builder="canonical", symmetry="C2v")
    reference, identity_error, aux = cp.build_canonical_cas_reference(spec)
    wrapped = md.LeakGuardReference(reference)
    original = ex.build_system
    ex.build_system = lambda system: (wrapped, reference)
    try:
        c = ex.context(tag)
    finally:
        ex.build_system = original
    c.system = tag
    anchor = HERE / "anchors" / f"{tag}.npz"
    if anchor.exists():
        with np.load(anchor) as archive:
            base = SimpleNamespace(d2=np.array(archive["d2"]),
                                   gamma=np.array(archive["gamma"]),
                                   status="optimal")
    else:
        base = g.da.r.j.solve_dqg_sdp(
            c.sel, solver=c.args.solver, tolerance=c.args.solver_tolerance,
            max_iterations=c.args.max_iterations, solver_threads=1,
            positivity_conditions="DQG", symmetry_blocked_psd=True)
        if base.status != "optimal":
            raise RuntimeError(f"Anchor solve failed at {tag}: {base.status}")
        anchor.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=anchor.parent, suffix=".npz",
                                         delete=False) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, d2=base.d2, gamma=base.gamma)
        temporary.replace(anchor)
    c.base = base
    c.theta0 = base.d2[c.geo["rows"], c.geo["cols"]]
    c.exact_z = g.da.r.coordinates(c, c.exact.exact_d2, c.theta0)
    c.star_eval = c.objective.evaluate(c.exact.exact_d2, c.exact.exact_gamma,
                                       gradient=False)
    c.href, c.fref = g.da.r.j._energy_values(c.sel, c.objective,
                                             c.exact.exact_d2,
                                             c.exact.exact_gamma)
    c.h = g.da.r.j._hamiltonian_gradient(c.sel, c.geo["rows"],
                                         c.geo["cols"]) @ c.lift
    c.f = g.da.r.j._raw_ftpbe_gradient(c.objective, c.sel, c.base.d2,
                                       c.base.gamma, c.geo["rows"],
                                       c.geo["cols"]) @ c.lift
    g.c108.install_uniform_family(c, real_frames())
    return c


def arm_path(target, stream, budget=BUDGET):
    return OUT / POOL / target / f"b{budget}_r{stream}.json"


def uniform_path(stream, budget=BUDGET):
    return OUT / POOL / "uniform" / f"b{budget}_r{stream}.json"


def save_record(path, record, arrays):
    g.allocation.save(path, record)
    g.atomic_npz(path.with_suffix(".npz"), **arrays)


def solve_arm(d, c, data, selection, threads):
    counts = np.asarray(selection["counts"], dtype=int)
    indices = np.asarray(selection["indices"], dtype=int)
    raw = selected_raw(d, c, data, counts, indices)
    solver = MeritSolver(d, c, raw, np.ones((len(raw.values), 1)),
                         threads=threads)
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


def run_stream(d, c, stream, targets, budget, threads):
    pending = [t for t in targets if not arm_path(t, stream, budget).exists()]
    uniform_pending = not uniform_path(stream, budget).exists()
    if not pending and not uniform_pending:
        print("EXISTS", stream, flush=True)
        return
    started = time.time()
    model = rule.build_model(c, d, stream)
    selections = {t: rule.select(model, c, t, budget) for t in pending}
    if pending:
        maximum = np.max(np.asarray([selections[t]["counts"] for t in pending]),
                         axis=0)
        data = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream),
                                      maximum)
        for target in pending:
            selection = selections[target]
            scores, payload, stats = solve_arm(d, c, data, selection, threads)
            record = dict(pool=POOL, stream=stream, budget=budget,
                          system=SYSTEM, rule="weak_parts_v1",
                          error_basis="spin", lambda_radius=LAMBDA,
                          mu_ftpbe=MU, pilot_shots=rule.FRAMES * rule.PILOT,
                          seed=int(d.production_seed(stream)),
                          status="optimal", solver_stats=stats,
                          seconds=time.time() - started)
            record.update(selection)
            record.update(scores)
            save_record(arm_path(target, stream, budget), record, payload)
            print("ARM %-7s r%d K %d H %+.3f F %+.3f D2 %.4f guard %s"
                  % (target, stream, selection["k"], scores["h_error_meh"],
                     scores["f_error_meh"], scores["d2_error"],
                     selection["lineality_guard_ratio"]), flush=True)
        del data
        gc.collect()
    if uniform_pending:
        counts = np.full(rule.FRAMES, budget // rule.FRAMES, dtype=int)
        data = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream),
                                      counts)
        raw = d.band.all_rows_shadow(c, data, counts)
        result = d.band.solve_original_raw(c, raw)
        if result.status != "optimal":
            raise RuntimeError(result.status)
        scores = d.r.score(c, result)
        record = dict(pool=POOL, target="uniform", stream=stream,
                      budget=budget, system=SYSTEM, rule="uniform_no_pilot",
                      error_basis="spin", lambda_radius=1.0, mu_ftpbe=0.0,
                      pilot_shots=0, seed=int(d.production_seed(stream)),
                      status="optimal", counts=counts.tolist(),
                      seconds=time.time() - started, **scores)
        save_record(uniform_path(stream, budget), record,
                    dict(d2=result.d2, gamma=result.gamma))
        print("UNI r%d H %+.3f F %+.3f D2 %.4f"
              % (stream, scores["h_error_meh"], scores["f_error_meh"],
                 scores["d2_error"]), flush=True)
        del raw, result, data
        gc.collect()


def command_check(streams):
    d, c = g.context("n2")
    d.band.SOLVE_BAND = d.band.band_solver(True)
    c = co_context()
    frames = real_frames()
    rank = rule.design_rank(np.asarray([b @ c.lift for b in c.blocks]))
    anchor = d.r.score(c, c.base)
    print("POOL", POOL, "frames", frames.shape,
          "kind", "real" if np.max(np.abs(frames.imag)) == 0 else "complex",
          "rank", rank, flush=True)
    print("ANCHOR H %+.3f F %+.3f D2 %.4f"
          % (anchor["h_error_meh"], anchor["f_error_meh"], anchor["d2_error"]),
          flush=True)
    print("HREF %.6f FREF %.6f" % (c.href, c.fref), flush=True)
    model = rule.build_model(c, d, streams[0])
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


def command_run(streams, targets, threads):
    d, c = g.context("n2")
    d.band.SOLVE_BAND = d.band.band_solver(True)
    c = co_context()
    print("RUNPOOL", POOL, "rank",
          rule.design_rank(np.asarray([b @ c.lift for b in c.blocks])),
          flush=True)
    for stream in streams:
        run_stream(d, c, stream, targets, BUDGET, threads)


def _rmse(values):
    return float(np.sqrt(np.mean(np.square(values))))


def command_summarize(streams):
    present = [s for s in streams if uniform_path(s).exists()]
    if not present:
        return
    uniform = [json.loads(uniform_path(s).read_text()) for s in present]
    base_h = np.asarray([r["h_error_meh"] for r in uniform])
    base_f = np.asarray([r["f_error_meh"] for r in uniform])
    base_d2 = np.asarray([r["d2_error"] for r in uniform])
    rows = []
    for target in TARGETS:
        files = [arm_path(target, s) for s in present]
        if not all(p.exists() for p in files):
            continue
        arm = [json.loads(p.read_text()) for p in files]
        ah = np.asarray([r["h_error_meh"] for r in arm])
        af = np.asarray([r["f_error_meh"] for r in arm])
        ad2 = np.asarray([r["d2_error"] for r in arm])
        loss_arm = ah ** 2 + af ** 2
        loss_base = base_h ** 2 + base_f ** 2
        rows.append(dict(
            system=SYSTEM, pool=POOL, target=target, streams=len(present),
            k=float(np.mean([r["k"] for r in arm])),
            h_rmse=_rmse(ah), f_rmse=_rmse(af), d2=float(ad2.mean()),
            h_bias=float(ah.mean()), f_bias=float(af.mean()),
            joint_mse=float(loss_arm.mean()),
            uniform_h_rmse=_rmse(base_h), uniform_f_rmse=_rmse(base_f),
            uniform_d2=float(base_d2.mean()),
            uniform_joint_mse=float(loss_base.mean()),
            delta_loss=float((loss_arm - loss_base).mean()),
            wins=int(np.count_nonzero(loss_arm < loss_base))))
    fields = ["system", "pool", "target", "streams", "k", "h_rmse", "f_rmse",
              "d2", "h_bias", "f_bias", "joint_mse", "uniform_h_rmse",
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
        print("%-7s n=%d K=%.1f | arm H %.2f F %.2f D2 %.4f | unif H %.2f "
              "F %.2f D2 %.4f | dL %.1f wins %d/%d"
              % (row["target"], row["streams"], row["k"],
                 row["h_rmse"], row["f_rmse"], row["d2"],
                 row["uniform_h_rmse"], row["uniform_f_rmse"],
                 row["uniform_d2"], row["delta_loss"], row["wins"],
                 row["streams"]), flush=True)
    print("wrote", target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "summarize"))
    parser.add_argument("--targets", default="h_only")
    parser.add_argument("--streams", default=",".join(map(str, STREAMS)))
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args()
    streams = tuple(int(v) for v in arguments.streams.split(","))
    targets = tuple(arguments.targets.split(","))
    if arguments.command == "check":
        command_check(streams)
    elif arguments.command == "run":
        command_run(streams, targets, arguments.threads)
    else:
        command_summarize(streams)


if __name__ == "__main__":
    main()
