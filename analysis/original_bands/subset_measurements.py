#!/usr/bin/env python3
"""Fixed 12-frame diagnostic, selected using pilot information only.

The 60k-budget greedy path defines one full-rank 12-frame subset per pilot.
That subset is held fixed at 30k, 60k, 90k, and 120k. All 30 x 500 pilot
shots count toward every budget; unselected pilots are acquired but omitted
from the fit. The existing band DQG estimator receives only selected rows.

Examples:
  python original_bands/subset_measurements.py design --streams 0,1,2
  python original_bands/subset_measurements.py solve --basis spin --streams 0 --budgets 30000
  python original_bands/subset_measurements.py solve --basis spin --streams 0 --backend reusable --solver-threads 1
"""

import argparse
import gc
import hashlib
import json
import resource
import time
from pathlib import Path

import allocate_measurements as allocation
import numpy as np
from scipy.linalg import eigh

from calibrate_band_response import atomic_npz
from screen_frame_subsets import screen


FRAME_COUNT = 30
SUBSET_COUNT = 12
AFFINE_RANK = 181
SELECTION_BUDGET = 60000
DEFAULT_BUDGETS = (30000, 60000, 90000, 120000)
METHOD = "subset12"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_unchanged(path, record):
    """Refuse to silently replace a predeclared selection or shot schedule."""
    serializable = json.loads(json.dumps(record, default=json_value, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != serializable:
            raise RuntimeError(f"Refusing to overwrite a changed subset design: {path}")
    else:
        allocation.save(path, serializable)


def load_pilot_model(stream):
    folder = allocation.OUT / "joint" / f"r{stream}"
    path = folder / "pilot_model.npz"
    with np.load(path) as archive:
        model = {name: np.array(archive[name]) for name in archive.files}
    geometry_path = path
    if "designs" not in model:
        geometry_path = allocation.OUT / "joint" / "r0" / "pilot_model.npz"
        with np.load(geometry_path) as reference:
            if str(reference["design_digest"]) != str(model["design_digest"]):
                raise ValueError("Cannot reuse r0 designs: design digests differ.")
            model["designs"] = np.array(reference["designs"])
    if allocation.digest(model["designs"]) != str(model["design_digest"]):
        raise ValueError("Stored measurement geometry does not match its digest.")
    if model["designs"].shape[0] != FRAME_COUNT or model["designs"].shape[2] != AFFINE_RANK:
        raise ValueError("This diagnostic requires the N2 Uniform30, 181-dimensional geometry.")
    # The prior allocations record the pilot's seed without loading an oracle.
    provenance_path = folder / f"allocation_b{SELECTION_BUDGET}.json"
    provenance = json.loads(provenance_path.read_text())
    for key in ("pilot_digest", "design_digest"):
        if provenance[key] != str(model[key]):
            raise ValueError(f"Pilot provenance mismatch: {key}")
    provenance = dict(seed=int(provenance["seed"]),
                      pilot_model_file=str(path), pilot_model_sha256=sha256(path),
                      geometry_file=str(geometry_path), geometry_sha256=sha256(geometry_path),
                      pilot_digest=str(model["pilot_digest"]),
                      design_digest=str(model["design_digest"]))
    return model, provenance


def relative_to_full_pool(subset_risks, full_model, fit_counts, total_budget):
    """Compare with uniform 30 at the same total cost, including unused pilot."""
    joint = np.einsum("f,fij->ij", 1 / np.asarray(fit_counts, dtype=float),
                      subset_risks["joint"])
    full_joint = FRAME_COUNT / total_budget * full_model["joint"].sum(0)
    eigenvalues, vectors = eigh(full_joint)
    whitening = (vectors / np.sqrt(eigenvalues)).T
    d2 = float(np.sum(subset_risks["frobenius"] / fit_counts))
    uniform_d2 = float(FRAME_COUNT / total_budget * full_model["frobenius"].sum())
    plane = float(eigh(whitening @ joint @ whitening.T, eigvals_only=True)[-1])
    return dict(h_variance=float(joint[0, 0] / full_joint[0, 0]),
                f_variance=float(joint[1, 1] / full_joint[1, 1]),
                d2_variance=d2 / uniform_d2, worst_energy_variance=plane,
                worst_joint_d2=max(plane, d2 / uniform_d2),
                h_second_moment_ha2=float(joint[0, 0]),
                f_second_moment_ha2=float(joint[1, 1]), d2_second_moment=d2)


def validate_record(record):
    indices = np.asarray(record["indices"], dtype=int)
    counts = np.asarray(record["counts"], dtype=int)
    fitted = np.asarray(record["fit_counts"], dtype=int)
    budget = record["budget"]
    assert indices.shape == (SUBSET_COUNT,) and len(set(indices)) == SUBSET_COUNT
    assert np.all((indices >= 0) & (indices < FRAME_COUNT))
    assert counts.shape == (FRAME_COUNT,) and np.all(counts >= allocation.PILOT)
    assert counts.sum() == budget == record["totalbudget"] == record["total_budget"]
    np.testing.assert_array_equal(counts[indices], fitted)
    unselected = np.setdiff1d(np.arange(FRAME_COUNT), indices)
    assert np.all(counts[unselected] == allocation.PILOT)
    assert fitted.sum() == record["measured_budget"] == record["fit_shots"]
    assert fitted.sum() + record["unused_pilot_shots"] == budget
    assert record["pilot_shots"] == FRAME_COUNT * allocation.PILOT
    assert record["production_shots"] == budget - record["pilot_shots"]
    assert record["rank"] == AFFINE_RANK
    assert not record["oracle_inputs"]


def design(streams, budgets):
    for stream in streams:
        model, provenance = load_pilot_model(stream)
        folder = allocation.OUT / METHOD / f"r{stream}"
        folder.mkdir(parents=True, exist_ok=True)
        selection_path = folder / "selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            for key, value in provenance.items():
                if selection[key] != value:
                    raise ValueError(f"The fixed subset's provenance changed: {key}")
            indices = np.asarray(selection["indices"], dtype=int)
        else:
            paths = screen(model["designs"], model["covariances"], model["h"], model["f"],
                           (SELECTION_BUDGET,))
            path = paths[SELECTION_BUDGET]["path"]
            candidates = [entry for entry in path if len(entry["indices"]) == SUBSET_COUNT]
            if len(candidates) != 1:
                raise ValueError("The full-rank greedy path did not reach exactly 12 frames.")
            chosen = candidates[0]
            indices = np.asarray(chosen["indices"], dtype=int)
            selection = dict(method=METHOD, stream=stream, indices=indices,
                             fixed_subset_count=SUBSET_COUNT,
                             selection_budget=SELECTION_BUDGET,
                             selection_rule="12-frame point on pilot-only 60k full-rank greedy path",
                             subset_count_chosen_before_new_sdp_results=True,
                             purpose="predeclared diagnostic of redundant noisy band constraints",
                             selection_proxy=chosen, greedy_path=path,
                             oracle_inputs=False, **provenance)
            save_unchanged(selection_path, selection)
        risks = allocation.build_risks(model["designs"][indices], model["covariances"][indices],
                                       model["h"], model["f"])
        if risks["rank"] != AFFINE_RANK:
            raise ValueError("The selected subset does not retain all 181 directions.")
        atomic_npz(folder / "subset_model.npz", **risks, indices=indices,
                   design_digest=provenance["design_digest"],
                   pilot_digest=provenance["pilot_digest"])
        previous = np.full(SUBSET_COUNT, allocation.PILOT, dtype=int)
        unused_pilot = (FRAME_COUNT - SUBSET_COUNT) * allocation.PILOT
        for budget in budgets:
            measured_budget = budget - unused_pilot
            fit_counts, diagnostic = allocation.allocate(risks, measured_budget, previous)
            counts = np.full(FRAME_COUNT, allocation.PILOT, dtype=int)
            counts[indices] = fit_counts
            record = dict(method=METHOD, stream=stream, budget=budget,
                          totalbudget=budget, total_budget=budget,
                          measured_budget=measured_budget, fit_shots=measured_budget,
                          acquired_shots=budget, indices=indices, counts=counts,
                          fit_counts=fit_counts, extra_counts=counts-allocation.PILOT,
                          n_pool_frames=FRAME_COUNT, n_fit_frames=SUBSET_COUNT,
                          pilot_per_frame=allocation.PILOT,
                          pilot_shots=FRAME_COUNT*allocation.PILOT,
                          production_shots=budget-FRAME_COUNT*allocation.PILOT,
                          unused_pilot_shots=unused_pilot,
                          unselected_pilot_in_fit=False,
                          selection_budget=SELECTION_BUDGET, schedule=budgets,
                          rank=risks["rank"], condition=risks["condition"],
                          oracle_inputs=False, proxy="fixed_unweighted_geometric_inverse_sandwich",
                          allocation_solver="MOSEK",
                          proxy_vs_subset_uniform=diagnostic,
                          proxy_vs_full_uniform=relative_to_full_pool(risks, model, fit_counts, budget),
                          prediction_is_sdp_error_guarantee=False, **provenance)
            validate_record(record)
            save_unchanged(folder / f"allocation_b{budget}.json", record)
            previous = fit_counts
            print("SUBSET_DESIGN", stream, budget, "indices", indices.tolist(),
                  "fitted_shots", measured_budget, "rank", risks["rank"],
                  "full_uniform_proxy", record["proxy_vs_full_uniform"]["worst_joint_d2"], flush=True)


def selected_raw(d, c, data, counts, indices):
    """Assemble only selected rows; unequal sample counts never weight the SDP."""
    offsets = np.cumsum([0] + [len(block) for block in c.blocks])
    rows = np.concatenate([np.arange(offsets[f], offsets[f+1]) for f in indices])
    vectors = np.asarray(c.vectors)[rows]
    samples = [data[f][:int(counts[f])] for f in indices]
    values = np.concatenate([x.mean(0) for x in samples])
    hits = np.concatenate([x.sum(0).astype(int) for x in samples])
    assert len(values) == len(vectors)
    design = d.r.j.quadratic_design(vectors)[np.arange(len(values))].tocsr()
    return d.band.vendor.ShadowData(
        rotations=tuple(c.rotations[f] for f in indices), pair_vectors=vectors,
        design=design, values=values,
        lower_bounds=values.copy(), upper_bounds=values.copy(), hits=hits,
        # This scalar cannot express nonuniform counts. The fixed nuclear-band
        # branch ignores it; per-frame counts are stored in the result records.
        shots_per_basis=0, exact_values=np.full(len(values), np.nan),
        exact_constraints=False, occupations=None)


def solve(streams, budgets, basis, backend="original", solver_threads=1):
    if backend not in ("original", "reusable"):
        raise ValueError(f"Unknown backend: {backend}")
    if not isinstance(solver_threads, int) or solver_threads < 1:
        raise ValueError("solver_threads must be a positive integer")
    if backend == "original" and solver_threads != 1:
        raise ValueError("The frozen original backend requires solver_threads=1.")
    d, c = allocation.context()
    if str(c.args.solver).upper() != "MOSEK":
        raise ValueError("This run requires MOSEK for the fixed DQG estimator.")
    d.band.SOLVE_BAND = d.band.band_solver(basis == "spin")
    hashes = {str(p.relative_to(allocation.HERE.parent)): sha256(p)
              for p in (allocation.HERE / "run.py", Path(d.band.vendor.__file__))}
    for stream in streams:
        reusable = None
        records = {budget: json.loads((allocation.OUT / METHOD / f"r{stream}" /
                                      f"allocation_b{budget}.json").read_text()) for budget in budgets}
        for record in records.values():
            validate_record(record)
            if record["seed"] != d.production_seed(stream):
                raise ValueError("The allocation uses a different pilot seed.")
        indices = np.asarray(records[budgets[0]]["indices"], dtype=int)
        assert all(record["indices"] == indices.tolist() for record in records.values())
        geometry = np.array([block @ c.lift for block in c.blocks])
        assert allocation.digest(geometry) == records[budgets[0]]["design_digest"]
        maximum_counts = np.max([r["counts"] for r in records.values()], axis=0)
        data = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream), maximum_counts)
        assert allocation.digest(*(x[:allocation.PILOT] for x in data)) == records[budgets[0]]["pilot_digest"]
        if backend == "reusable":
            # The subset and row order stay fixed across this stream's budgets.
            # Only this small observation span enters CVXPY's DPP parameter map.
            template = selected_raw(d, c, data, records[budgets[0]]["counts"], indices)
            observation_bank = {
                budget: np.concatenate([data[frame][:int(record["counts"][frame])].mean(0)
                                         for frame in indices])
                for budget, record in records.items()}
            observation_basis = np.column_stack([values - template.values
                                                  for values in observation_bank.values()])
        for budget in budgets:
            design = records[budget]
            counts = np.asarray(design["counts"], dtype=int)
            if backend == "reusable":
                values = observation_bank[budget]
            else:
                raw = selected_raw(d, c, data, counts, indices)
                values = raw.values
            folder = allocation.OUT / "results" / basis / METHOD
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"b{budget}_r{stream}.json"
            if path.exists():
                old = json.loads(path.read_text())
                assert old["counts"] == counts.tolist() and old["indices"] == indices.tolist()
                assert old["observation_digest"] == allocation.digest(values)
                assert old["estimator_source_hashes"] == hashes
                print("EXISTS", path, flush=True)
                continue
            start = time.time()
            print("SUBSET_SOLVE_START", basis, stream, budget,
                  "backend", backend, "solver_threads", solver_threads, flush=True)
            if backend == "reusable":
                from reusable_band_solver import ReusableBandSolver
                if reusable is None:
                    reusable = ReusableBandSolver(c, template, basis, band=d.band,
                                                  observation_basis=observation_basis,
                                                  solver_threads=solver_threads)
                result = reusable.solve(values)
                solver_stats = dict(reusable.last_stats)
            else:
                result = d.band.solve_original_raw(c, raw)
                solver_stats = dict(backend="original", solver_threads=solver_threads)
            if result.status != "optimal":
                raise RuntimeError(result.status)
            scores = d.r.score(c, result)
            seconds = time.time() - start
            anchor = d.r.score(c, c.base)
            atomic_npz(path.with_suffix(".npz"), d2=result.d2, gamma=result.gamma,
                       values=values, counts=counts, indices=indices,
                       fit_counts=np.asarray(design["fit_counts"]))
            allocation.save(path, dict(
                **design, basis=basis, allocation_basis="pilot_geometry", system="n2_108",
                solver="MOSEK", backend=backend, solver_threads=solver_threads,
                solver_stats=solver_stats,
                status=result.status, seconds=seconds,
                estimator_source_hashes=hashes, observation_digest=allocation.digest(values),
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                shadow_error_trace=getattr(result, "shadow_error_trace", None),
                h_better_f_worse=bool(abs(scores["h_error_meh"]) < abs(anchor["h_error_meh"])
                                     and abs(scores["f_error_meh"]) > abs(anchor["f_error_meh"])),
                **scores))
            print("SUBSET_SOLVE_DONE", basis, stream, budget,
                  {key: scores[key] for key in ("h_error_meh", "f_error_meh", "d2_error")},
                  "seconds", round(seconds, 1), flush=True)
        if backend == "reusable":
            # CVXPY/adapter closures can form cycles. Drop the previous stream's
            # graph before constructing another large graph in this process.
            reusable = None
            gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("design", "solve"))
    parser.add_argument("--streams", default="0,1,2")
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--basis", choices=("spin", "full"), default="spin")
    parser.add_argument("--backend", choices=("original", "reusable"), default="original")
    parser.add_argument("--solver-threads", type=int, default=1,
                        help="MOSEK threads for reusable; original requires 1 (default)")
    args = parser.parse_args()
    streams = sorted(set(map(int, args.streams.split(","))))
    budgets = sorted(set(map(int, args.budgets.split(","))))
    if not budgets or budgets[0] < FRAME_COUNT * allocation.PILOT:
        parser.error("Every budget must include the complete 30 x 500 pilot.")
    if args.solver_threads < 1:
        parser.error("--solver-threads must be positive")
    if args.backend == "original" and args.solver_threads != 1:
        parser.error("The frozen original backend requires --solver-threads 1.")
    if args.stage == "design":
        design(streams, budgets)
    else:
        solve(streams, budgets, args.basis, args.backend, args.solver_threads)


if __name__ == "__main__":
    main()
