#!/usr/bin/env python3
"""Pilot-only shot allocation with the existing Maple-band estimator frozen.

The allocation model uses a fixed geometric inverse ONLY to score measurement
noise. No inverse, covariance weighting, or target penalty enters the SDP.
The joint H/F covariance constraint protects every energy combination, including
their contrast; the Frobenius constraint protects the rest of the 2-RDM.

Examples:
  python allocate_measurements.py design --streams 0,1,2
  python allocate_measurements.py solve --basis spin --streams 0 --budgets 30000
  python allocate_measurements.py report
"""

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/band_allocation_mpl")

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy.linalg import eigh

HERE = Path(__file__).resolve().parent
OUT = HERE / "measurement_design"
PILOT = 500
DEFAULT_BUDGETS = (15000, 30000, 60000, 90000, 120000)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=convert) + "\n")
    temporary.replace(path)


def digest(*arrays):
    result = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        result.update(str((array.shape, array.dtype.str)).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def build_risks(designs, covariances, h, f):
    """Inputs are known geometry, pilot covariances, and anchor gradients only."""
    design = np.vstack(designs)
    u, singular, vt = np.linalg.svd(design, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-10))
    if rank != design.shape[1]:
        raise ValueError("The full pool must identify the entire affine space.")
    inverse = (vt.T / singular) @ u.T
    energy = np.stack((h, f))
    joint, frobenius = [], []
    start = 0
    for block, covariance in zip(designs, covariances):
        back = inverse[:, start:start + len(block)]
        propagated = back @ covariance @ back.T
        local = energy @ propagated @ energy.T
        joint.append((local + local.T) / 2)
        frobenius.append(np.trace(propagated))
        start += len(block)
    joint = np.asarray(joint)
    frobenius = np.asarray(frobenius)
    # At uniform n=B/F: covariance = F/B * sum(K_f).
    eigenvalues, vectors = eigh(joint.sum(0))
    if eigenvalues[0] <= eigenvalues[-1] * 1e-12:
        raise ValueError("H/F covariance is numerically rank deficient.")
    whitening = (vectors / np.sqrt(eigenvalues)).T
    whitened = np.array([whitening @ k @ whitening.T for k in joint])
    np.testing.assert_allclose(whitened.sum(0), np.eye(2), atol=1e-8)
    np.testing.assert_allclose(inverse @ design, np.eye(rank), atol=1e-8)
    return dict(joint=joint, frobenius=frobenius, whitened=whitened,
                whitening=whitening, rank=rank,
                condition=float(singular[0] / singular[-1]),
                gradient_cosine=float(h @ f / np.linalg.norm(h) / np.linalg.norm(f)))


def risk_ratios(risks, counts):
    counts = np.asarray(counts)
    frame_count, budget = len(counts), int(sum(counts))
    factors = budget / (frame_count * counts)
    joint = np.einsum("f,fij->ij", factors, risks["joint"])
    uniform_joint = risks["joint"].sum(0)
    plane = np.einsum("f,fij->ij", factors, risks["whitened"])
    return dict(h_variance=float(joint[0, 0] / uniform_joint[0, 0]),
                f_variance=float(joint[1, 1] / uniform_joint[1, 1]),
                d2_variance=float(factors @ risks["frobenius"] / sum(risks["frobenius"])),
                worst_energy_variance=float(eigh(plane, eigvals_only=True)[-1]))


def allocate(risks, budget, lower, mode="joint"):
    """Small convex minimax design; integer totals and previous counts preserved."""
    lower = np.asarray(lower, dtype=int)
    number = len(lower)
    if budget < lower.sum():
        raise ValueError("Budget is below already acquired shots.")
    if budget == lower.sum():
        return lower.copy(), dict(status="no_new_shots", **risk_ratios(risks, lower))
    fractions = cp.Variable(number)
    reciprocal = cp.Variable(number, nonneg=True)
    worst = cp.Variable(nonneg=True)
    # x_f=n_f/B. At uniform allocation reciprocal/F = 1.
    constraints = [cp.sum(fractions) == 1, fractions >= lower / budget,
                   cp.inv_pos(fractions) <= reciprocal]
    d2weights = risks["frobenius"] / sum(risks["frobenius"])
    constraints.append(d2weights @ reciprocal / number <= worst)
    if mode == "joint":
        plane = sum(reciprocal[k] * risks["whitened"][k] / number
                    for k in range(number))
        constraints.append(worst * np.eye(2) - plane >> 0)
    elif mode == "marginal":
        for axis in range(2):
            weights = risks["joint"][:, axis, axis]
            constraints.append(weights @ reciprocal / (number * sum(weights)) <= worst)
    else:
        raise ValueError(mode)
    problem = cp.Problem(cp.Minimize(worst), constraints)
    problem.solve(solver="MOSEK", eps=1e-9,
                  mosek_params={"MSK_IPAR_NUM_THREADS": 1})
    if problem.status != "optimal":
        raise RuntimeError(f"Allocation failed: {problem.status}")
    continuous = np.maximum(budget * fractions.value, lower)
    counts = np.maximum(np.floor(continuous).astype(int), lower)
    if counts.sum() > budget:
        raise RuntimeError("Allocation rounding exceeded budget.")
    remainder = budget - int(sum(counts))
    if remainder:
        order = np.argsort(-(continuous - counts), kind="stable")
        counts[order[:remainder]] += 1
    assert sum(counts) == budget and np.all(counts >= lower)
    return counts, dict(status=problem.status, allocation_solver="MOSEK",
                        continuous_minimax=float(worst.value),
                        **risk_ratios(risks, counts))


def context():
    import direction_analysis as d
    return d, d.setup()


def design_stage(d, c, streams, budgets, mode):
    designs = np.array([a @ c.lift for a in c.blocks])
    # The affine coordinates are orthonormal for the full D2 Frobenius norm.
    np.testing.assert_allclose((c.lift.T * c.geo["scale"]**2) @ c.lift,
                               np.eye(c.lift.shape[1]), atol=1e-9)
    for stream in streams:
        folder = OUT / mode / f"r{stream}"
        folder.mkdir(parents=True, exist_ok=True)
        pilot_counts = np.full(len(designs), PILOT, dtype=int)
        pilot = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream), pilot_counts)
        covariances = np.array([d.r.robust_covariance(c, x) for x in pilot])
        risks = build_risks(designs, covariances, c.h, c.f)
        np.savez_compressed(folder / "pilot_model.npz", **risks,
                            covariances=covariances, designs=designs, h=c.h, f=c.f,
                            design_digest=digest(designs),
                            pilot_digest=digest(*pilot))
        previous = pilot_counts
        for budget in budgets:
            counts, diagnostic = allocate(risks, budget, previous, mode)
            record = dict(method=mode, stream=stream, budget=budget,
                          pilot_shots=int(sum(pilot_counts)), pilot_per_frame=PILOT,
                          production_shots=int(budget - sum(pilot_counts)),
                          counts=counts, extra_counts=counts-pilot_counts,
                          production_active_frames=int(np.sum(counts > PILOT)),
                          seed=d.production_seed(stream), oracle_inputs=False,
                          proxy="fixed_unweighted_geometric_inverse_sandwich",
                          design_digest=digest(designs), pilot_digest=digest(*pilot),
                          covariance="pilot_particle_sector_shrinkage_mass_50",
                          rank=risks["rank"], condition=risks["condition"],
                          gradient_cosine=risks["gradient_cosine"],
                          schedule=budgets, **diagnostic)
            path = folder / f"allocation_b{budget}.json"
            if path.exists():
                old = json.loads(path.read_text())
                if old != json.loads(json.dumps(record, default=lambda x: x.tolist()
                                                if isinstance(x, np.ndarray) else x.item())):
                    raise RuntimeError(f"Refusing to overwrite changed design: {path}")
            else:
                save(path, record)
            previous = counts
            print("DESIGN", mode, stream, budget, diagnostic, flush=True)


def solve_stage(d, c, streams, budgets, mode, basis, allocation_basis=None,
                backend="original", solver_threads=1):
    if c.args.solver.upper() != "MOSEK":
        raise RuntimeError("N2 validation requires MOSEK.")
    d.band.SOLVE_BAND = d.band.band_solver(basis == "spin")
    source_hashes = {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (HERE / "run.py", Path(d.band.vendor.__file__))}
    for stream in streams:
        reusable = None
        schedules = {}
        for budget in budgets:
            if mode == "uniform":
                counts = d.r.j._equal_counts(len(c.rotations), tuple(range(len(c.rotations))),
                                              budget, c.args.allocation_chunk)
            else:
                design_folder = OUT / mode
                if mode == "response":
                    design_folder = design_folder / (allocation_basis or basis)
                path = design_folder / f"r{stream}" / f"allocation_b{budget}.json"
                record = json.loads(path.read_text())
                assert record["seed"] == d.production_seed(stream)
                counts = np.array(record["counts"], dtype=int)
            schedules[budget] = counts
        maximum_counts = np.max(list(schedules.values()), axis=0)
        data = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream), maximum_counts)
        observation_bank = [np.concatenate([sample[:int(n)].mean(0)
                                             for sample,n in zip(data, counts)])
                            for counts in schedules.values()]
        for budget, counts in schedules.items():
            folder = OUT / "results" / basis / mode
            path = folder / f"b{budget}_r{stream}.json"
            if path.exists():
                old = json.loads(path.read_text())
                assert old["counts"] == counts.tolist()
                print("EXISTS", path, flush=True)
                continue
            raw = d.band.all_rows_shadow(c, data, counts)
            start = time.time()
            print("SOLVE_START", basis, mode, budget, stream, flush=True)
            if backend == "reusable":
                from reusable_band_solver import ReusableBandSolver
                if reusable is None:
                    observation_basis = np.column_stack([values-raw.values for values in observation_bank])
                    reusable = ReusableBandSolver(c, raw, basis, band=d.band,
                                                  observation_basis=observation_basis,
                                                  solver_threads=solver_threads)
                result = reusable.solve(raw.values)
                solver_stats = reusable.last_stats
            else:
                result = d.band.solve_original_raw(c, raw)
                solver_stats = dict(backend="original")
            if result.status != "optimal":
                raise RuntimeError(result.status)
            scores = d.r.score(c, result)
            seconds = time.time() - start
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path.with_suffix(".npz"), d2=result.d2,
                                gamma=result.gamma, values=raw.values, counts=counts)
            anchor = d.r.score(c, c.base)
            save(path, dict(basis=basis, method=mode, stream=stream, budget=budget,
                            allocation_basis=allocation_basis or basis,
                            backend=backend, solver_stats=solver_stats,
                            counts=counts, pilot_shots=PILOT*len(counts),
                            production_shots=budget-PILOT*len(counts),
                            seed=d.production_seed(stream), status=result.status,
                            seconds=seconds, estimator_source_hashes=source_hashes,
                            observation_digest=digest(raw.values),
                            shadow_error_trace=getattr(result, "shadow_error_trace", None),
                            h_better_f_worse=bool(abs(scores["h_error_meh"]) < abs(anchor["h_error_meh"])
                                                 and abs(scores["f_error_meh"]) > abs(anchor["f_error_meh"])),
                            **scores))
            print("SOLVE_DONE", basis, mode, budget, stream,
                  {k:scores[k] for k in ("h_error_meh", "f_error_meh", "d2_error")},
                  "seconds", round(seconds, 1), flush=True)


def report():
    records = {}
    for basis, folder in (("spin", HERE / "results"), ("full", HERE / "results_full")):
        for path in folder.glob("b*_r*.json"):
            rec = json.loads(path.read_text())
            rec.update(basis=basis, method="uniform")
            records[(basis, "uniform", rec["budget"], rec["stream"])] = rec
    for path in (OUT / "results").glob("*/*/b*_r*.json"):
        rec = json.loads(path.read_text())
        records[(rec["basis"], rec["method"], rec["budget"], rec["stream"])] = rec
    rows = []
    keys = sorted(set((r["basis"], r["method"], r["budget"]) for r in records.values()))
    for basis, method, budget in keys:
        if method == "uniform":
            continue
        samples = [rec for rec in records.values()
                   if (rec["basis"], rec["method"], rec["budget"]) == (basis, method, budget)
                   and (basis, "uniform", budget, rec["stream"]) in records]
        if not samples:
            continue
        uniform = [records[(basis, "uniform", budget, rec["stream"])] for rec in samples]
        row = dict(basis=basis, method=method, budget=budget, n=len(samples),
                   streams=",".join(str(r["stream"]) for r in samples))
        for label, field in (("h", "h_error_meh"), ("f", "f_error_meh"), ("d2", "d2_error")):
            a = np.array([r[field] for r in samples])
            b = np.array([r[field] for r in uniform])
            row[label+"_rmse"] = float(np.sqrt(np.mean(a*a)))
            row[label+"_uniform_rmse"] = float(np.sqrt(np.mean(b*b)))
            row[label+"_rmse_ratio"] = row[label+"_rmse"] / row[label+"_uniform_rmse"]
            row[label+"_bias"] = float(np.mean(a))
        row["three_way_wins"] = sum(all(abs(a[field]) < abs(b[field]) for field in
                                        ("h_error_meh", "f_error_meh", "d2_error"))
                                     for a, b in zip(samples, uniform))
        rows.append(row)
    if not rows:
        print("No paired results yet.")
        return
    with (OUT / "paired_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("design", "solve", "report"))
    parser.add_argument("--streams", default="0,1,2")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--method", choices=("joint", "marginal", "uniform", "response"), default="joint")
    parser.add_argument("--basis", choices=("spin", "full"), default="spin")
    parser.add_argument("--allocation-basis", choices=("spin", "full"),
                        help="Replay the same calibrated counts in either fixed estimator.")
    parser.add_argument("--backend", choices=("original", "reusable"), default="original")
    parser.add_argument("--solver-threads", type=int, default=1)
    args = parser.parse_args()
    if args.solver_threads < 1:
        parser.error("--solver-threads must be positive.")
    if args.backend == "original" and args.solver_threads != 1:
        parser.error("The original backend uses its fixed one-thread setting.")
    if args.stage == "report":
        report()
        return
    streams = list(map(int, args.streams.split(",")))
    budgets = sorted(set(map(int, args.budgets.split(","))))
    if min(budgets) < 30*PILOT:
        parser.error("Every pool frame requires a 500-shot pilot: total budget >= 15000.")
    d, c = context()
    if args.stage == "design":
        if args.method in ("uniform", "response"):
            parser.error("Use calibrate_band_response.py for response designs; uniform needs no design stage.")
        design_stage(d, c, streams, budgets, args.method)
    else:
        solve_stage(d, c, streams, budgets, args.method, args.basis,
                    args.allocation_basis, args.backend, args.solver_threads)


if __name__ == "__main__":
    main()
