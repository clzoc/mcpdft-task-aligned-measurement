#!/usr/bin/env python3
"""Calibrate shot allocation against the unchanged band SDP using pilot data.

Central paired multiplier perturbations isolate per-frame response. They use
the empirical *within-shot* covariance, not independent pair-observable noise.
No exact RDM, exact energy, or production observation is used for allocation.
This is an experimental design surrogate, not a certificate of SDP risk.

Calibration is classical and expensive: 1 + 2 * draws * frames SDP solves per
pilot and error basis. Results are cached so individual frame jobs can resume.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from dataclasses import replace
import hashlib
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

import allocate_measurements as allocation


def multiplier_direction(sample, seed):
    """E[v v.T | sample] equals its unbiased single-shot sample covariance."""
    values = np.asarray(sample, dtype=float)
    centered = values - values.mean(0)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1., 1.], size=len(values))
    return centered.T @ signs / np.sqrt(len(values)-1)


def response_moments(base, plus, minus, epsilon):
    """All arguments are (full D2 matrix, [H,F] energies), with energies in Ha.

The odd response is a finite-difference covariance proxy. The uncentered paired
response adds sensitivity to local nonlinear drift. Both energy matrices are
PSD; the latter is a finite-scale second moment, not an asymptotic covariance.
"""
    dxp, dxm = plus[0]-base[0], minus[0]-base[0]
    dep, dem = plus[1]-base[1], minus[1]-base[1]
    odd_x = (dxp-dxm)/(2*epsilon)
    odd_e = (dep-dem)/(2*epsilon)
    even_x = (dxp+dxm)/(2*epsilon)
    even_e = (dep+dem)/(2*epsilon)
    joint_odd = np.outer(odd_e, odd_e)
    joint_even = np.outer(even_e, even_e)
    return dict(joint=joint_odd+joint_even, joint_odd=joint_odd,
                joint_even=joint_even,
                frobenius=float(np.sum(odd_x**2)+np.sum(even_x**2)),
                frobenius_odd=float(np.sum(odd_x**2)),
                frobenius_even=float(np.sum(even_x**2)),
                energy_even=even_e)


def atomic_npz(path, **arrays):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def calibrate(basis, stream, frames, draws, amplitude, base_only=False,
              draw_indices=None, backend="original", solver_threads=1):
    import direction_analysis as d
    c = d.setup()
    d.band.SOLVE_BAND = d.band.band_solver(basis == "spin")
    counts = np.full(len(c.rotations), allocation.PILOT, dtype=int)
    pilot = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream), counts)
    raw = d.band.all_rows_shadow(c, pilot, counts)
    if c.args.solver.upper() != "MOSEK":
        raise RuntimeError("N2 validation requires MOSEK.")
    offset = np.cumsum([0] + [x.shape[1] for x in pilot])
    active_draws = list(range(draws) if draw_indices is None else draw_indices)
    columns = []
    for frame in frames:
        for draw in active_draws:
            vector = np.zeros_like(raw.values)
            vector[offset[frame]:offset[frame+1]] = multiplier_direction(
                pilot[frame], [20261002, stream, frame, draw])
            columns.append(vector)
    observation_basis = np.column_stack(columns)
    folder = allocation.OUT / "response" / basis / f"r{stream}"
    folder.mkdir(parents=True, exist_ok=True)
    epsilon = amplitude / np.sqrt(allocation.PILOT)
    config = dict(basis=basis, stream=stream, pilot_per_frame=allocation.PILOT,
                  pilot_shots=int(sum(counts)), pilot_digest=allocation.digest(*pilot),
                  epsilon=epsilon, draws=draws, amplitude_in_pilot_se=amplitude,
                  estimator_hash=hashlib.sha256((d.HERE / "run.py").read_bytes()).hexdigest(),
                  oracle_inputs=False, perturbation="centered_shot_multiplier",
                  proxy="paired_uncentered_response_second_moment")
    path = folder / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise RuntimeError("Calibration configuration changed; use a fresh output directory.")
    if not path.exists():
        allocation.save(path, config)

    reusable = None

    def solve(values, stem):
        nonlocal reusable
        path = folder / (stem + ".npz")
        if path.exists():
            archive = np.load(path)
            assert str(archive["values_digest"]) == allocation.digest(values)
            return archive["d2"], archive["energies"]
        perturbed = replace(raw, values=values, lower_bounds=values.copy(),
                            upper_bounds=values.copy())
        start = time.time()
        print("CALIBRATE_START", basis, stream, stem, flush=True)
        if backend == "reusable":
            from reusable_band_solver import ReusableBandSolver
            if reusable is None:
                reusable = ReusableBandSolver(c, raw, basis, band=d.band,
                                              observation_basis=observation_basis,
                                              solver_threads=solver_threads)
            result = reusable.solve(values)
            solver_stats = reusable.last_stats
        else:
            result = d.band.solve_original_raw(c, perturbed)
            solver_stats = dict(backend="original")
        if result.status != "optimal":
            raise RuntimeError(result.status)
        # Known operators and the returned state only. Do not call score().
        energies = np.array(d.energies(c, result.d2, result.gamma))
        atomic_npz(path, d2=result.d2, gamma=result.gamma,
                   energies=energies, values_digest=allocation.digest(values),
                   seconds=time.time()-start,
                   peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   solver_stats=json.dumps(solver_stats), backend=backend)
        print("CALIBRATE_DONE", basis, stream, stem,
              round(time.time()-start, 1), flush=True)
        return result.d2, energies

    base = solve(raw.values, "pilot_fit")
    if base_only:
        return
    for frame in frames:
        for draw in active_draws:
            file = folder / f"response_f{frame}_d{draw}.npz"
            if file.exists():
                continue
            seed = [20261002, stream, frame, draw]
            direction = multiplier_direction(pilot[frame], seed)
            displacement = np.zeros_like(raw.values)
            displacement[offset[frame]:offset[frame+1]] = epsilon*direction
            plus = solve(raw.values+displacement, f"plus_f{frame}_d{draw}")
            minus = solve(raw.values-displacement, f"minus_f{frame}_d{draw}")
            atomic_npz(file, **response_moments(base, plus, minus, epsilon),
                       direction=direction, seed=seed)


def design(basis, stream, budgets):
    folder = allocation.OUT / "response" / basis / f"r{stream}"
    config = json.loads((folder / "config.json").read_text())
    all_moments = []
    for frame in range(30):
        moments = []
        for draw in range(config["draws"]):
            path = folder / f"response_f{frame}_d{draw}.npz"
            if not path.exists():
                raise RuntimeError(f"Incomplete calibration: missing {path.name}")
            moments.append(dict(np.load(path)))
        all_moments.append(moments)
    joint = np.array([np.mean([v["joint"] for v in group], axis=0) for group in all_moments])
    frobenius = np.array([np.mean([v["frobenius"] for v in group]) for group in all_moments])
    eigenvalues, vectors = eigh(joint.sum(0))
    if eigenvalues[0] <= eigenvalues[-1]*1e-12:
        raise RuntimeError("The calibrated H/F response is rank deficient.")
    whitening = (vectors / np.sqrt(eigenvalues)).T
    risks = dict(joint=joint, frobenius=frobenius,
                 whitened=np.array([whitening @ k @ whitening.T for k in joint]))
    previous = np.full(30, allocation.PILOT, dtype=int)
    for budget in budgets:
        counts, diagnostics = allocation.allocate(risks, budget, previous)
        allocation.save(folder / f"allocation_b{budget}.json",
                        dict(method="response", basis=basis, stream=stream,
                             budget=budget, counts=counts,
                             pilot_shots=15000, production_shots=budget-15000,
                             seed=config_seed(stream), schedule=budgets,
                             calibration_complete=True, oracle_inputs=False,
                             prediction_is_sdp_error_guarantee=False,
                             **diagnostics))
        previous = counts
        print("RESPONSE_DESIGN", basis, stream, budget, diagnostics, flush=True)


def config_seed(stream):
    import direction_analysis as d
    return d.production_seed(stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("calibrate", "design"))
    parser.add_argument("--basis", choices=("spin", "full"), default="spin")
    parser.add_argument("--stream", type=int, default=0)
    parser.add_argument("--frames", default=",".join(map(str, range(30))))
    parser.add_argument("--draws", type=int, default=2,
                        help="Two draws are a screen; repeat more directions before claims.")
    parser.add_argument("--amplitude", type=float, default=0.25,
                        help="Perturbation size in pilot standard-error units.")
    parser.add_argument("--budgets", default="15000,30000,60000,90000,120000")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--draw-indices", help="Subset of 0..draws-1, for resumed jobs.")
    parser.add_argument("--backend", choices=("original", "reusable"), default="original")
    parser.add_argument("--solver-threads", type=int, default=1)
    args = parser.parse_args()
    if args.solver_threads < 1:
        parser.error("--solver-threads must be positive.")
    if args.backend == "original" and args.solver_threads != 1:
        parser.error("The original backend uses its fixed one-thread setting.")
    if args.draws < 1 or not 0 < args.amplitude <= 1:
        parser.error("Require draws >= 1 and 0 < amplitude <= 1.")
    frames = list(map(int, args.frames.split(",")))
    if not frames or min(frames) < 0 or max(frames) >= 30:
        parser.error("Frame indices must be in 0..29.")
    if args.stage == "calibrate":
        draw_indices = None if args.draw_indices is None else list(map(int, args.draw_indices.split(",")))
        if draw_indices is not None and (not draw_indices or min(draw_indices)<0 or max(draw_indices)>=args.draws):
            parser.error("Draw indices must lie in 0..draws-1.")
        calibrate(args.basis, args.stream, frames, args.draws, args.amplitude,
                  args.base_only, draw_indices, args.backend, args.solver_threads)
    else:
        design(args.basis, args.stream, sorted(set(map(int, args.budgets.split(",")))))
