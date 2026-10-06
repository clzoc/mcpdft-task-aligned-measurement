#!/usr/bin/env python3
"""Faithful Maple band formulation of the original global 2-RDM fit.

No ridge-LS projection and no equality correction: the raw finite-shot rows
enter the SDP directly as PSD error-matrix bands (``sv2RDM_example.mpl``
``consqc2``), and the objective is ``energy + Tr(E1+E2)`` solved with DQG.
By default the error matrices are restricted to the same spin-adapted
structure as the 2-RDM (``E_aa = E_bb``, ``E_triplet = E_aa``, no
antisymmetric-symmetric cross block, zero mixed-symmetry entries), which is
the representation used by the Maple sample.  ``--error-basis full`` instead
uses the vendor's unrestricted pair-basis E (kept for comparison only).

    python run.py --streams 0 --budgets 30000
    python run.py --streams 0 --budgets 30000 --error-basis full
    python run.py --system c2 --streams 0 --budgets 15000
"""

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CONTEXT = HERE.parent / "original_global_fit"
sys.path.insert(0, str(CONTEXT))
import context_108 as m  # noqa: E402
import research as r  # noqa: E402
import constrained_shadow as vendor  # noqa: E402

BUDGETS = (15_000, 30_000, 60_000, 90_000, 120_000, 150_000, 240_000,
           300_000)
PILOT_INDEX_BASE = 200
SSEED0, SSEED_STEP = 202610018100, 104729
NUCLEAR_WEIGHT = 1.0
FOLDER = None


def output_folder(system, error_basis):
    if system == "n2_108":
        return HERE / ("results" if error_basis == "spin" else "results_full")
    suffix = "" if error_basis == "spin" else "_full"
    return HERE / f"results_{system}{suffix}"

BAND_MARKER = (
    "            corrected_d2 = d2 + shadow_error_positive - shadow_error_negative\n"
    "            prediction = shadow_data.design @ cp.reshape(\n"
    "                corrected_d2, (n_pairs * n_pairs,), order=\"C\"\n"
    "            )\n"
    "            constraints.append(prediction == shadow_data.values)\n"
)
BAND_REPLACEMENT = (
    "            band_lower = shadow_data.design @ cp.reshape(\n"
    "                d2 + shadow_error_positive - shadow_error_negative,\n"
    "                (n_pairs * n_pairs,), order=\"C\",\n"
    "            )\n"
    "            band_upper = shadow_data.design @ cp.reshape(\n"
    "                d2 - shadow_error_positive + shadow_error_negative,\n"
    "                (n_pairs * n_pairs,), order=\"C\",\n"
    "            )\n"
    "            constraints.append(band_lower <= shadow_data.values)\n"
    "            constraints.append(shadow_data.values <= band_upper)\n"
    "            if _BAND_SPIN_CONSTRAINED:\n"
    "                for _row in range(n_pairs):\n"
    "                    for _col in range(_row + 1, n_pairs):\n"
    "                        if (pair_alpha_count[_row]\n"
    "                                != pair_alpha_count[_col]\n"
    "                                or pair_irreps[_row] != pair_irreps[_col]):\n"
    "                            constraints.append(\n"
    "                                shadow_error_positive[_row, _col] == 0)\n"
    "                            constraints.append(\n"
    "                                shadow_error_negative[_row, _col] == 0)\n"
    "                if reference.n_alpha == reference.n_beta:\n"
    "                    _spatial_pairs = pair_basis(n_spatial)\n"
    "                    _aa = np.array([pair_to_index[_p]\n"
    "                                    for _p in _spatial_pairs], dtype=int)\n"
    "                    _bb = np.array([\n"
    "                        pair_to_index[(_f + n_spatial, _s + n_spatial)]\n"
    "                        for _f, _s in _spatial_pairs], dtype=int)\n"
    "                    _ab = np.array([\n"
    "                        pair_to_index[(_f, _s + n_spatial)]\n"
    "                        for _f in range(n_spatial)\n"
    "                        for _s in range(n_spatial)], dtype=int)\n"
    "                    _anti = np.zeros(\n"
    "                        (n_spatial * n_spatial, len(_spatial_pairs)))\n"
    "                    _sym_pairs = (tuple((_p, _p)\n"
    "                                        for _p in range(n_spatial))\n"
    "                                  + _spatial_pairs)\n"
    "                    _sym = np.zeros(\n"
    "                        (n_spatial * n_spatial, len(_sym_pairs)))\n"
    "                    for _c, (_f, _s) in enumerate(_spatial_pairs):\n"
    "                        _anti[_f * n_spatial + _s, _c] = 1.0 / math.sqrt(2.0)\n"
    "                        _anti[_s * n_spatial + _f, _c] = -1.0 / math.sqrt(2.0)\n"
    "                    for _c, (_f, _s) in enumerate(_sym_pairs):\n"
    "                        if _f == _s:\n"
    "                            _sym[_f * n_spatial + _s, _c] = 1.0\n"
    "                        else:\n"
    "                            _sym[_f * n_spatial + _s, _c] = 1.0 / math.sqrt(2.0)\n"
    "                            _sym[_s * n_spatial + _f, _c] = 1.0 / math.sqrt(2.0)\n"
    "                    for _e in (shadow_error_positive,\n"
    "                               shadow_error_negative):\n"
    "                        _e_aa = _e[np.ix_(_aa, _aa)]\n"
    "                        _e_bb = _e[np.ix_(_bb, _bb)]\n"
    "                        _e_ab = _e[np.ix_(_ab, _ab)]\n"
    "                        constraints.extend([\n"
    "                            _e_aa == _e_bb,\n"
    "                            _anti.T @ _e_ab @ _anti == _e_aa,\n"
    "                            _anti.T @ _e_ab @ _sym == 0,\n"
    "                        ])\n"
)

SOLVE_BAND = None


def band_solver(spin_constrained):
    """Vendor SDP with the Maple Eq.-(11) inequality bands restored."""

    source_path = Path(vendor.__file__)
    source = source_path.read_text(encoding="utf-8")
    if source.count(BAND_MARKER) != 1:
        raise RuntimeError("The band injection marker is not unique.")
    module = types.ModuleType("constrained_shadow_band")
    module.__file__ = str(source_path)
    sys.modules[module.__name__] = module
    exec(compile(source.replace(BAND_MARKER, BAND_REPLACEMENT),
                 str(source_path), "exec"), module.__dict__)
    module.__dict__["_BAND_SPIN_CONSTRAINED"] = bool(spin_constrained)
    return module.solve_dqg_sdp


def all_rows_shadow(c, data, counts):
    values, hits = [], []
    for frame, number in enumerate(counts):
        sample = data[frame][: int(number)]
        values.append(sample.mean(0))
        hits.append(sample.sum(0).astype(int))
    values = np.concatenate(values)
    hits = np.concatenate(hits)
    design = r.j.quadratic_design(np.asarray(c.vectors))
    d2_map = design[np.arange(len(values))].tocsr()
    return vendor.ShadowData(
        rotations=tuple(c.rotations),
        pair_vectors=np.asarray(c.vectors),
        design=d2_map,
        values=values,
        lower_bounds=values.copy(),
        upper_bounds=values.copy(),
        hits=hits,
        shots_per_basis=int(counts[0]),
        exact_values=np.full(len(values), np.nan),
        exact_constraints=False,
        occupations=None,
    )


def solve_original_raw(c, raw):
    return SOLVE_BAND(
        r.j.LeakGuardReference(c.sel), shadow_data=raw,
        shadow_error_weight=NUCLEAR_WEIGHT, selection_objective="energy",
        solver=c.args.solver, tolerance=c.args.solver_tolerance,
        max_iterations=c.args.max_iterations, solver_threads=1,
        positivity_conditions="DQG", symmetry_blocked_psd=True,
        initial_d2=c.base.d2, initial_gamma=c.base.gamma,
        initial_corrected_d2=c.base.d2)


def run_stream(c, system, arm, stream, budgets):
    prod_seed = SSEED0 + SSEED_STEP * (PILOT_INDEX_BASE + stream)
    pending = [budget for budget in budgets
               if not (FOLDER / f"b{budget}_r{stream}.json").exists()]
    if not pending:
        print(f"SKIP r={stream}", flush=True)
        return
    active = tuple(range(len(c.rotations)))
    max_counts = r.j._equal_counts(len(c.rotations), active, max(budgets),
                                   c.args.allocation_chunk)
    data = r.j._sample_outcomes(c.oracle, prod_seed, max_counts)
    for budget in budgets:
        file = FOLDER / f"b{budget}_r{stream}.json"
        if file.exists():
            continue
        counts = r.j._equal_counts(len(c.rotations), active, budget,
                                   c.args.allocation_chunk)
        raw = all_rows_shadow(c, data, counts)
        started = time.time()
        try:
            result = solve_original_raw(c, raw)
            assert result.status == "optimal", result.status
            scores = r.score(c, result)
        except (AssertionError, RuntimeError) as error:
            print(f"FAILED b={budget} r={stream} {error!r}", flush=True)
            continue
        seconds = time.time() - started
        record = dict(
            system=system, arm=arm, frame="uniform30_raw_error_matrix_bands",
            formulation="maple_consqc2_bands_no_projection",
            error_basis=("spin_adapted" if "spin" in arm else "full"),
            n_frames=30, budget=int(budget), pilot_shots=0,
            production_shots=int(counts.sum()), stream=int(stream),
            production_seed=int(prod_seed), active_frames=len(active),
            nuclear_weight=NUCLEAR_WEIGHT, status=str(result.status),
            seconds=seconds, convergence="full_budget_prefix_coupled",
            shadow_error_trace=getattr(result, "shadow_error_trace", None),
            **scores,
        )
        r.save(file, record)
        print(f"{arm} b={budget} r={stream} D={record['d2_error']:.5f} "
              f"H={record['h_error_meh']:+.3f} F={record['f_error_meh']:+.3f} "
              f"sec={seconds:.1f}", flush=True)


def main(system, streams, budgets, error_basis):
    global SOLVE_BAND, FOLDER
    resource.setrlimit(resource.RLIMIT_AS, (24 * 1024 ** 3, 24 * 1024 ** 3))
    FOLDER = output_folder(system, error_basis)
    FOLDER.mkdir(exist_ok=True)
    SOLVE_BAND = band_solver(error_basis == "spin")
    arm = "orig_band_spin" if error_basis == "spin" else "orig_band_full"
    if system == "n2_108":
        c = m.context()
        rotations, _ = m.pool_uniform_family(c)
        m.install_uniform_family(c, rotations)
    else:
        c = r.context(system)
        r.install_frames(c, "uniform30")
    for stream in streams:
        run_stream(c, system, arm, stream, budgets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=("n2_108", "c2"),
                        default="n2_108")
    parser.add_argument("--streams", default="0,1,2,3,4")
    parser.add_argument("--budgets", default=",".join(str(v) for v in BUDGETS))
    parser.add_argument("--error-basis", choices=("spin", "full"),
                        default="spin")
    arguments = parser.parse_args()
    main(arguments.system,
         [int(v) for v in arguments.streams.split(",")],
         [int(v) for v in arguments.budgets.split(",")],
         arguments.error_basis)
