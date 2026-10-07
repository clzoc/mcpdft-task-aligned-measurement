#!/usr/bin/env python3
"""Shared machinery for the guard15 + (lambda,mu)=(0.5,1) arm.

The frame rule and the merit change are frozen:
  * K=15 point on the pilot-only backward-greedy guarded path (screen_frame_subsets);
  * merit = H + mu * F_lin + lambda * Tr(E), with (lambda, mu) = (0.5, 1);
  * everything else (raw-row bands, DQG, anchor, MOSEK 1e-8) unchanged.

N2 is the original_bands N2 (10e,8o) context; C2 is research.context("c2")
(8e,8o). Both pools are Uniform30 with 120 rows/frame and affine rank 181.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
ORIGINAL_BANDS = ANALYSIS/"original_bands"
if str(ORIGINAL_BANDS) not in sys.path:
    sys.path.insert(0, str(ORIGINAL_BANDS))
if str(ANALYSIS/"outputs"/"physical_shot_threshold"/"response_design") not in sys.path:
    sys.path.insert(0, str(ANALYSIS/"outputs"/"physical_shot_threshold"/"response_design"))

import allocate_measurements as allocation  # noqa: E402
import direction_analysis as da  # noqa: E402
import context_108 as c108  # noqa: E402
from calibrate_band_response import atomic_npz  # noqa: E402
from fselection_solver import make as make_fit  # noqa: E402
from screen_frame_subsets import screen  # noqa: E402
from subset_measurements import selected_raw, sha256  # noqa: E402

SYSTEMS = ("n2", "c2")
N2_SCAN = {"n2_r080": 0.80, "n2_r090": 0.90, "n2_r100": 1.00, "n2_r125": 1.25,
           "n2_r145": 1.45, "n2_r160": 1.60, "n2_r180": 1.80}
N2_SCAN["n2_r110"] = 1.10
N2_SCAN.update({"n2_r200": 2.00, "n2_r220": 2.20, "n2_r250": 2.50})
FRAME_COUNT = 30
PILOT = allocation.PILOT
AFFINE_RANK = 181
K = 15
LAMBDA = 0.5
MU = 1.0
BUDGETS = (30000, 60000, 90000, 120000)
STREAMS = tuple(range(8))
METHOD = "guard15_lammu"
N2_FROZEN_PLANS = allocation.OUT/"guard_subsets"/"plans"/"guard15"
N2_FROZEN_CACHE = allocation.OUT/"guard_subsets"/"models"


C2_LIT = "c2_pvtz_r125"


def context(system):
    if system == "n2":
        return da, da.setup()
    if system == "c2":
        c = da.r.context("c2")
        da.r.install_frames(c, "uniform30")
        return da, c
    if system == C2_LIT:
        return da, c2_literature_context(system)
    if system in N2_SCAN:
        return da, n2_geometry_context(system)
    raise ValueError(f"Unknown system {system}")


def c2_literature_context(tag=C2_LIT):
    """C2 (8e,8o) at R=1.25 with the MC-PDFT literature basis cc-pVTZ (AVAS).

    Uses the guarded protocol's build_c2_reference; the campaign's own
    research.context('c2') is cc-pVDZ/R=1.40/canonical and is not reused.
    """
    import experiment as ex
    code = ANALYSIS/"mcpdft_measurement_revision"/"guarded_mcpdft_shadow_protocol"/"code"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    import mcpdft_derandomization as md
    reference = md.build_c2_reference(1.25, basis="cc-pvtz")
    original = ex.build_system
    ex.build_system = lambda system: (md.LeakGuardReference(reference), reference)
    try:
        c = ex.context(tag)
    finally:
        ex.build_system = original
    c.system = tag
    anchor = HERE/"anchors"/f"{tag}.npz"
    if anchor.exists():
        with np.load(anchor) as archive:
            base = SimpleNamespace(d2=np.array(archive["d2"]),
                                   gamma=np.array(archive["gamma"]), status="optimal")
    else:
        base = da.r.j.solve_dqg_sdp(
            c.sel, solver=c.args.solver, tolerance=c.args.solver_tolerance,
            max_iterations=c.args.max_iterations, solver_threads=1,
            positivity_conditions="DQG", symmetry_blocked_psd=True)
        if base.status != "optimal":
            raise RuntimeError(f"Anchor solve failed at {tag}: {base.status}")
        anchor.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(anchor, d2=base.d2, gamma=base.gamma)
    c.base = base
    c.theta0 = base.d2[c.geo["rows"], c.geo["cols"]]
    c.exact_z = da.r.coordinates(c, c.exact.exact_d2, c.theta0)
    c.star_eval = c.objective.evaluate(c.exact.exact_d2, c.exact.exact_gamma,
                                       gradient=False)
    c.href, c.fref = da.r.j._energy_values(c.sel, c.objective,
                                           c.exact.exact_d2, c.exact.exact_gamma)
    c.h = da.r.j._hamiltonian_gradient(c.sel, c.geo["rows"], c.geo["cols"]) @ c.lift
    c.f = da.r.j._raw_ftpbe_gradient(c.objective, c.sel, c.base.d2, c.base.gamma,
                                     c.geo["rows"], c.geo["cols"]) @ c.lift
    from run_c2_random_pilot_joint_design import _c2_random_frames
    rotations, _, _ = _c2_random_frames(c.sel.n_spatial_orbitals, 30, .5,
                                        *da.r.ex.POOLS[0])
    c108.install_uniform_family(c, np.asarray(rotations))
    return c


def n2_geometry_context(tag):
    """N2 (10e,8o) cc-pVDZ context at an arbitrary bond length.

    Mirrors context_108.context() (exact rebuilt by the pipeline, DQG anchor
    cached under this folder) and installs the same Uniform30 family from the
    cached rotations; frame probabilities are cached per geometry by research.
    """
    bond = N2_SCAN[tag]
    name = f"n2_108_r{int(round(bond*100)):03d}"
    da.r.ex.SYSTEMS[name] = dict(kind="n2", bond_length=bond, basis="cc-pvdz",
                                 active_electrons=10, active_orbitals=8)
    c = da.r.ex.context(name)
    c.system = name
    anchor = HERE/"anchors"/f"{tag}.npz"
    if anchor.exists():
        with np.load(anchor) as archive:
            base = SimpleNamespace(d2=np.array(archive["d2"]),
                                   gamma=np.array(archive["gamma"]), status="optimal")
    else:
        base = da.r.j.solve_dqg_sdp(
            da.r.j.LeakGuardReference(c.sel), solver=c.args.solver,
            tolerance=c.args.solver_tolerance,
            max_iterations=c.args.max_iterations, solver_threads=1,
            positivity_conditions="DQG", symmetry_blocked_psd=True)
        if base.status != "optimal":
            raise RuntimeError(f"Anchor solve failed at {tag}: {base.status}")
        anchor.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(anchor, d2=base.d2, gamma=base.gamma)
    c.base = base
    c.theta0 = base.d2[c.geo["rows"], c.geo["cols"]]
    c.exact_z = da.r.coordinates(c, c.exact.exact_d2, c.theta0)
    c.star_eval = c.objective.evaluate(c.exact.exact_d2, c.exact.exact_gamma,
                                       gradient=False)
    c.href, c.fref = da.r.j._energy_values(c.sel, c.objective,
                                           c.exact.exact_d2, c.exact.exact_gamma)
    c.h = da.r.j._hamiltonian_gradient(c.sel, c.geo["rows"], c.geo["cols"]) @ c.lift
    c.f = da.r.j._raw_ftpbe_gradient(c.objective, c.sel, c.base.d2, c.base.gamma,
                                     c.geo["rows"], c.geo["cols"]) @ c.lift
    rotations, _ = c108.pool_uniform_family(c)
    c108.install_uniform_family(c, rotations)
    return c


def baseline_dir(tag):
    return HERE/"baseline"/tag


def run_baseline(tag, d, c, streams, budget=120000):
    """No-pilot uniform global-band spin baseline via original_bands/run.py."""
    folder = baseline_dir(tag)
    folder.mkdir(parents=True, exist_ok=True)
    da.band.SOLVE_BAND = da.band.band_solver(True)
    da.band.FOLDER = folder
    for stream in streams:
        da.band.run_stream(c, tag, "orig_band_spin", stream, (budget,))


def pilot_cache(system, stream):
    return HERE/"pilots"/system/f"pilot_observations_r{stream}.npz"


def load_pilots(system, d, c, stream):
    cache = pilot_cache(system, stream)
    if cache.exists():
        with np.load(cache) as archive:
            return np.array(archive["samples"]), np.array(archive["covariances"])
    if system == "n2":
        path = allocation.OUT/"holdout"/f"pilot_observations_r{stream}.npz"
        if path.exists():
            with np.load(path) as archive:
                samples, covariances = np.array(archive["samples"]), np.array(archive["covariances"])
        else:
            counts = np.full(FRAME_COUNT, PILOT, dtype=int)
            samples = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream), counts)
            covariances = np.array([d.r.robust_covariance(c, x) for x in samples])
    else:
        counts = np.full(FRAME_COUNT, PILOT, dtype=int)
        samples = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream), counts)
        covariances = np.array([d.r.robust_covariance(c, x) for x in samples])
    cache.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(cache, samples=samples, covariances=covariances)
    return samples, covariances


def model(system, d, c, stream):
    samples, covariances = load_pilots(system, d, c, stream)
    designs = np.array([block @ c.lift for block in c.blocks])
    risks = allocation.build_risks(designs, covariances, c.h, c.f)
    if risks["rank"] != AFFINE_RANK:
        raise ValueError(f"{system} pool rank {risks['rank']} != {AFFINE_RANK}")
    return dict(designs=designs, covariances=covariances, h=c.h, f=c.f,
                rank=risks["rank"], condition=risks["condition"]), samples


def guard_selection(model, budget):
    result = screen(model["designs"], model["covariances"], model["h"], model["f"], (budget,))
    points = [entry for entry in result[budget]["path"] if len(entry["indices"]) == K]
    if len(points) != 1:
        raise ValueError(f"No unique K={K} path point at budget {budget}")
    return points[0]


def selection_path(system, stream, budget):
    return HERE/"selections"/system/"guard15"/f"r{stream}_b{budget}.json"


def plan_path(system, stream, budget):
    return HERE/"plans"/system/"guard15"/f"r{stream}"/f"allocation_b{budget}.json"


def make_plan(system, d, c, stream, budget, write=True):
    path = plan_path(system, stream, budget)
    if path.exists():
        return json.loads(path.read_text()), "new-folder"
    frozen = N2_FROZEN_PLANS/f"r{stream}"/f"allocation_b{budget}.json"
    if system == "n2" and frozen.exists():
        record = json.loads(frozen.read_text())
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
            allocation.save(path, record)
        return record, "frozen-n2"
    built, samples = model(system, d, c, stream)
    chosen = guard_selection(built, budget)
    indices = np.asarray(chosen["indices"], dtype=int)
    risks = allocation.build_risks(built["designs"][indices], built["covariances"][indices],
                                   built["h"], built["f"])
    if risks["rank"] != AFFINE_RANK:
        raise ValueError(f"K={K} subset loses rank {AFFINE_RANK}")
    unused = (FRAME_COUNT-K)*PILOT
    measured = budget-unused
    fit_counts, diagnostic = allocation.allocate(risks, measured, np.full(K, PILOT, dtype=int), "joint")
    counts = np.full(FRAME_COUNT, PILOT, dtype=int)
    counts[indices] = fit_counts
    record = dict(method="guard15", system=system, stream=stream, budget=budget,
                  totalbudget=budget, total_budget=budget, acquired_shots=budget,
                  measured_budget=measured, fit_shots=measured, indices=indices,
                  counts=counts, fit_counts=fit_counts, n_pool_frames=FRAME_COUNT,
                  n_fit_frames=K, pilot_per_frame=PILOT, pilot_shots=FRAME_COUNT*PILOT,
                  unused_pilot_shots=unused, production_shots=budget-FRAME_COUNT*PILOT,
                  unselected_pilot_in_fit=False, seed=d.production_seed(stream),
                  oracle_inputs=False, selection_uses_truth=False,
                  count_policy="joint_HF_D2_geometric_allocation",
                  selection_rule="K=15 point on pilot-only backward guarded path minimizing "
                                 "max(worst H/F-plane variance ratio, D2 variance ratio) vs full30 uniform",
                  guard_proxy=dict(worst=chosen["worst"], plane=chosen["plane"], d2=chosen["d2"],
                                   h=chosen["h"], f=chosen["f"]),
                  allocation_diagnostic=diagnostic,
                  merit=dict(lambda_radius=LAMBDA, mu_ftpbe=MU,
                             form="H + mu*F_lin(anchor) + lambda*Tr(E)"))
    selection = dict(system=system, stream=stream, budget=budget, indices=indices,
                     fixed_subset_count=K, selection_uses_truth=False,
                     guard_proxy=record["guard_proxy"], seed=record["seed"])
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        allocation.save(path, record)
        allocation.save(selection_path(system, stream, budget), selection)
    return record, "designed"


def f_gradients(c):
    evaluation = c.objective.evaluate(c.base.d2, c.base.gamma, gradient=True)
    return (np.asarray(evaluation.d2_gradient, dtype=float),
            np.asarray(evaluation.gamma_gradient, dtype=float))


def result_path(system, stream, budget):
    return HERE/"results"/system/METHOD/f"b{budget}_r{stream}.json"


def solve_cell(system, d, c, stream, budget, lam=LAMBDA, mu=MU, threads=4):
    path = result_path(system, stream, budget)
    if path.exists():
        return json.loads(path.read_text()), "exists"
    record = json.loads(plan_path(system, stream, budget).read_text())
    counts = np.asarray(record["counts"], dtype=int)
    indices = np.asarray(record["indices"], dtype=int)
    data = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream), counts)
    template = selected_raw(d, c, data, counts, indices)
    values = template.values
    grad_d2, grad_gamma = f_gradients(c)
    fit = make_fit(c, template, "spin", band=d.band,
                   observation_basis=np.ones((len(values), 1)),
                   solver_threads=threads,
                   additional_d2_objective=mu*grad_d2,
                   additional_gamma_objective=mu*grad_gamma,
                   nucleus_weight=lam)
    result = fit.solve(values)
    if result.status != "optimal":
        raise RuntimeError(result.status)
    scores = d.r.score(c, result)
    hashes = {str(p.relative_to(ANALYSIS)): sha256(p)
              for p in (ORIGINAL_BANDS/"run.py", Path(d.band.vendor.__file__))}
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(path.with_suffix(".npz"), d2=result.d2, gamma=result.gamma, values=values,
               counts=counts, indices=indices)
    output = dict(record, system=system, stream=stream, budget=budget, method=METHOD,
                  basis="spin", status=result.status, lambda_radius=lam, mu_ftpbe=mu,
                  solver="MOSEK", solver_threads=threads, solver_tolerance=1e-8,
                  backend="reusable", solver_stats=dict(fit.last_stats),
                  shadow_error_trace=getattr(result, "shadow_error_trace", None),
                  observation_digest=allocation.digest(values),
                  estimator_source_hashes=hashes, **scores)
    allocation.save(path, output)
    return output, "solved"
