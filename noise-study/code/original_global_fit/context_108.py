#!/usr/bin/env python3
"""N2 (10e,8o) R=1.10 A context for the original global-fit baseline.

Self-contained copy of the n2_108 context: adds the full-valence N2 to the
in-memory registry, loads the cached DQG anchor (``anchor/anchor.npz``) and the
Uniform30 frame family (``pool_uniform30.npz``).  Only the pieces needed by
``run.py`` are kept.
"""

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/response_design_mpl")

import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent

# Locate the campaign pipeline (``response_design/research.py``) whether this
# folder sits inside ``outputs/physical_shot_threshold`` or directly under
# ``analysis``.
PIPELINE = None
for base in (HERE, *HERE.parents):
    for relative in ("outputs/physical_shot_threshold/response_design",
                     "physical_shot_threshold/response_design",
                     "response_design"):
        candidate = base / relative
        if (candidate / "research.py").exists():
            PIPELINE = candidate
            break
    if PIPELINE is not None:
        break
if PIPELINE is None:
    raise RuntimeError("Could not locate response_design/research.py")
sys.path.insert(0, str(PIPELINE))
import research as r  # noqa: E402

SYSTEM = "n2_108"
r.ex.SYSTEMS[SYSTEM] = dict(kind="n2", bond_length=1.10, basis="cc-pvdz",
                            active_electrons=10, active_orbitals=8)


def context():
    folder = HERE / "anchor"
    folder.mkdir(parents=True, exist_ok=True)
    c = r.ex.context(SYSTEM)
    c.system = SYSTEM
    file = folder / "anchor.npz"
    if file.exists():
        a = np.load(file)
        base = SimpleNamespace(d2=a["d2"], gamma=a["gamma"], status="optimal")
    else:
        base = r.j.solve_dqg_sdp(
            r.j.LeakGuardReference(c.sel), solver=c.args.solver,
            tolerance=c.args.solver_tolerance,
            max_iterations=c.args.max_iterations, solver_threads=1,
            positivity_conditions="DQG", symmetry_blocked_psd=True)
        assert base.status == "optimal"
        np.savez_compressed(file, d2=base.d2, gamma=base.gamma)
    c.base = base
    c.theta0 = base.d2[c.geo["rows"], c.geo["cols"]]
    c.exact_z = r.coordinates(c, c.exact.exact_d2, c.theta0)
    c.star_eval = c.objective.evaluate(c.exact.exact_d2, c.exact.exact_gamma,
                                       gradient=False)
    c.href, c.fref = r.j._energy_values(c.sel, c.objective,
                                        c.exact.exact_d2, c.exact.exact_gamma)
    c.h = r.j._hamiltonian_gradient(c.sel, c.geo["rows"], c.geo["cols"]) @ c.lift
    c.f = r.j._raw_ftpbe_gradient(c.objective, c.sel, c.base.d2, c.base.gamma,
                                  c.geo["rows"], c.geo["cols"]) @ c.lift
    return c


def pool_uniform_family(c):
    """Uniform30 frame family: 29 same-spin frames + 1 alpha/beta frame."""

    path = HERE / "pool_uniform30.npz"
    if path.exists():
        data = np.load(path)
        return data["rotations"], data["asymmetric"].astype(bool)
    from run_c2_random_pilot_joint_design import _c2_random_frames

    rotations, _, asymmetric = _c2_random_frames(
        c.sel.n_spatial_orbitals, 30, .5, *r.ex.POOLS[0])
    rotations = np.asarray(rotations)
    np.savez_compressed(path, rotations=rotations, asymmetric=asymmetric,
                        digest=r.digest(rotations))
    print("POOL", SYSTEM, "digest", r.digest(rotations), flush=True)
    return rotations, np.asarray(asymmetric, dtype=bool)


def install_uniform_family(c, rotations):
    c.rotations = np.asarray(rotations)
    c.vectors, _, c.blocks, _, _ = r.ex.cp.build_design(c.sel, c.rotations)
    c.oracle = r.j.AcquisitionOracle(c.exact, c.rotations, c.vectors, 1, 0)
    folder = r.HERE / "cache" / c.system
    folder.mkdir(parents=True, exist_ok=True)
    key = r.digest(c.rotations, c.oracle.ci)
    cache = folder / (key + ".npz")
    if cache.exists():
        c.oracle._probabilities = dict(enumerate(np.load(cache)["probabilities"]))
    c.exact_y = np.array([a @ c.exact.exact_d2[c.geo["rows"], c.geo["cols"]]
                          for a in c.blocks])
    probs = np.array([c.oracle._frame_probabilities(k)
                      for k in range(len(c.rotations))])
    np.testing.assert_allclose(probs @ c.oracle.indicators, c.exact_y,
                               atol=2e-10, rtol=0)
    if not cache.exists():
        with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz",
                                         delete=False) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, probabilities=probs)
        temporary.replace(cache)
