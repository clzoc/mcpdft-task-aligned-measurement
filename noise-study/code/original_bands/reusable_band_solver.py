#!/usr/bin/env python3
"""Reuse the unchanged band SDP graph while replacing observation values.

Only the right-hand sides of the two measurement bands are parameterized.
Constraints, objective, solver options, initialization, and result extraction
are taken from the existing vendor solver and ``run.py`` band substitution.
Each solve clears numerical warm-start state, while CVXPY's DPP compilation
cache remains available. This changes computational reuse, not the estimator.
For MOSEK, a structural check proves that only the dual linear objective
depends on parameters, allowing the constant PSD conversion to be reused.
Temporary solve allocations are collected and released before and after
each solve; current and peak resident memory are reported separately.

Usage from a calibration worker::

    fit = ReusableBandSolver(context, raw_shadow, basis="spin",
                             observation_basis=perturbation_directions)
    result = fit.solve(raw_shadow.values)
    print(fit.last_stats)

The standalone N2 validation is explicit and expensive::

    python reusable_band_solver.py --validate-n2 --basis spin --repeats 2

Without that flag the self-check solves only a tiny four-spin-orbital problem.

For memory control, a large problem must supply a fixed N-by-K observation
basis V. Observations are represented as y = y0 + V p, with only K parameter
coefficients. For frame calibration the columns are the planned multiplier
directions; exact and production observations are unnecessary. Representing
every observation by a separate DPP parameter can exhaust memory in CVXPY.
"""

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import ast
import copy
import ctypes
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import resource
import textwrap
import time
from types import MethodType, SimpleNamespace

import numpy as np
from scipy import sparse


def _release_temporary_memory():
    """Collect dead solve data and return free libc arenas to the OS."""
    gc.collect()
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return False
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return bool(trim(0))


def _current_rss_gib():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


class _MosekRhsCache:
    """Cache a proven constant MOSEK SDP conversion on one CVXPY Problem.

    The formatted CVXPY tensor stacks vec([A, b]) in column order. We verify
    that every parameter coefficient is in b's linear-constraint rows, while
    the complete objective tensor is constant. Consequently dual A, dual B,
    cone dimensions, PSD A_bar_data and PSD c_bar_data are all invariant.
    Only the dual linear objective c=-b changes. A private solver-adapter
    copy avoids changing CVXPY's global MOSEK adapter or other Problems.
    """

    def __init__(self, problem):
        self.ready = False
        self.hits = 0
        self.validate_against_original = False
        original_get_data = problem.get_problem_data

        def capture(*args, **kwargs):
            data, chain, inverse = original_get_data(*args, **kwargs)
            self._install(data, chain, inverse[-1])
            problem.get_problem_data = original_get_data
            return data, chain, inverse

        problem.get_problem_data = capture

    def _install(self, data, chain, inverse):
        import cvxpy.settings as settings
        from cvxpy.lin_ops.lin_op import CONSTANT_ID
        from cvxpy.reductions.cone2cone import affine2direct as cones
        from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK

        if not isinstance(chain.solver, MOSEK) or not data.get("dualized"):
            raise RuntimeError("Static SDP cache requires the dualized MOSEK adapter.")
        self.formatted = data[settings.PARAM_PROB]
        if not self.formatted.formatted:
            raise RuntimeError("Expected the cached, formatted parameter cone program.")
        cone = data["K_dir"]
        if cone[cones.DUAL_EXP] or cone[cones.DUAL_POW3D]:
            raise RuntimeError("Static SDP cache only supports linear, SOC and PSD cones.")
        tensor = self.formatted.A.tocsc()
        rows, remainder = divmod(tensor.shape[0], self.formatted.x.size + 1)
        if remainder:
            raise RuntimeError("Unexpected CVXPY formatted parameter tensor layout.")
        constant = self.formatted.param_id_to_col[CONSTANT_ID]
        dynamic_columns = [k for k in range(tensor.shape[1]) if k != constant]
        dynamic = tensor[:, dynamic_columns]
        nonzero_rows = dynamic.indices[dynamic.data != 0]
        offset = rows * self.formatted.x.size
        linear_end = cone[cones.FREE] + cone[cones.NONNEG]
        if (np.any(nonzero_rows < offset)
                or np.any(nonzero_rows >= offset + linear_end)):
            raise RuntimeError(
                "Cannot cache MOSEK SDP data: parameters change the constraint "
                "matrix or a nonlinear-cone right-hand side.")
        dynamic_objective = self.formatted.q.tocsc()[:, dynamic_columns]
        if np.any(dynamic_objective.data != 0):
            raise RuntimeError("Cannot cache MOSEK SDP data: objective has parameters.")
        # MOSEK removes the PSD columns from dual A and c. With no exponential
        # or power cones, its remaining linear c is exactly this prefix of -b.
        prefix = linear_end + sum(cone[cones.SOC])
        self.linear_rhs_tensor = tensor[offset:offset + prefix, :].tocsr()
        self.static_data = dict(data)
        self.inverse = inverse
        self.original_adapter = chain.solver
        initial_c = self._linear_c()
        if not np.array_equal(initial_c, data[settings.C]):
            raise RuntimeError("Cached RHS tensor does not reproduce MOSEK's initial c.")
        local_adapter = copy.copy(chain.solver)
        local_adapter.apply = MethodType(
            lambda _adapter, parameter_program: self.apply(parameter_program),
            local_adapter)
        chain.reductions[-1] = local_adapter
        chain.solver = local_adapter
        self.ready = True

    def _linear_c(self):
        from cvxpy.cvxcore.python import canonInterface

        program = self.formatted
        parameter_vector = canonInterface.get_parameter_vector(
            program.total_param_size, program.param_id_to_col,
            program.param_id_to_size,
            lambda parameter_id: np.asarray(program.id_to_param[parameter_id].value))
        return -np.asarray(self.linear_rhs_tensor @ parameter_vector).ravel()

    @staticmethod
    def _assert_same_data(actual, expected):
        """Expensive independent checks used only by the tiny self-check."""
        import cvxpy.settings as settings

        assert actual["K_dir"] == expected["K_dir"]
        difference = actual[settings.A] - expected[settings.A]
        assert not np.any(difference.data != 0)
        for key in (settings.B, settings.C):
            np.testing.assert_array_equal(actual[key], expected[key])
        for key in ("A_bar_data", "c_bar_data"):
            assert len(actual[key]) == len(expected[key])
            for cached, fresh in zip(actual[key], expected[key]):
                assert cached[:-1] == fresh[:-1]
                for left, right in zip(cached[-1], fresh[-1]):
                    np.testing.assert_array_equal(left, right)

    def apply(self, parameter_program):
        import cvxpy.settings as settings

        if parameter_program is not self.formatted:
            raise RuntimeError("MOSEK parameter program changed after static caching.")
        data = dict(self.static_data)
        data[settings.C] = self._linear_c()
        if self.validate_against_original:
            fresh, inverse = self.original_adapter.apply(parameter_program)
            self._assert_same_data(data, fresh)
            assert inverse[settings.OBJ_OFFSET] == self.inverse[settings.OBJ_OFFSET]
        self.hits += 1
        return data, self.inverse


def _band_module():
    # This import locates the same vendor used by the production wrapper.
    import direction_analysis
    return direction_analysis.band


def _replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Vendor solver changed: expected one source marker: "
                           + old[:100])
    return source.replace(old, new, 1)


def _make_builder(band, spin_constrained, observation_basis):
    """Generate a closure from the production source; fail on source drift."""
    original = band.band_solver(spin_constrained)
    vendor_source = Path(band.vendor.__file__).read_text(encoding="utf-8")
    source = _replace_once(vendor_source, band.BAND_MARKER,
                           band.BAND_REPLACEMENT)
    tree = ast.parse(source)
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef)
                   and node.name == "solve_dqg_sdp"]
    if len(definitions) != 1:
        raise RuntimeError("Expected exactly one vendor solve_dqg_sdp.")
    source = ast.get_source_segment(source, definitions[0]) + "\n"
    source = _replace_once(source, "def solve_dqg_sdp(",
                           "def _build_parameter_problem(")
    source = _replace_once(
        source,
        "            constraints.append(band_lower <= shadow_data.values)\n"
        "            constraints.append(shadow_data.values <= band_upper)\n",
        "            _coefficients = cp.Parameter(\n"
        "                shape=(_REUSABLE_OBSERVATION_BASIS.shape[1],),\n"
        "                name='band_observation_coefficients',\n"
        "                value=np.zeros(_REUSABLE_OBSERVATION_BASIS.shape[1]))\n"
        "            _observations = (\n"
        "                np.asarray(shadow_data.values, dtype=float).copy()\n"
        "                + _REUSABLE_OBSERVATION_BASIS @ _coefficients)\n"
        "            constraints.append(band_lower <= _observations)\n"
        "            constraints.append(_observations <= band_upper)\n",
    )
    # The public wrapper below uses the single-stage band branch exclusively.
    # Keep the extraction verbatim so all returned diagnostics stay identical.
    solve_start = source.index("    if linear_fit_only:\n")
    extract_marker = (
        "    if status not in OPTIMAL_STATUSES or d2.value is None "
        "or gamma.value is None:\n"
    )
    if source.count(extract_marker) != 1:
        raise RuntimeError("Vendor result-extraction marker changed.")
    extraction = source[source.index(extract_marker, solve_start):]
    replacement = '''\
    energy_problem = cp.Problem(cp.Minimize(merit_function), constraints)
    if not energy_problem.is_dcp(dpp=True):
        raise RuntimeError("Parameterized band problem is not DPP compliant.")
    _initial_values = [
        (_variable, None if _variable.value is None else _variable.value.copy())
        for _variable in energy_problem.variables()
    ]

    def _solve_values(_values):
        nonlocal weighted_fit_final_rmse
        _coefficients.value = np.asarray(_values, dtype=float)
        for _variable, _initial in _initial_values:
            _variable.value = None if _initial is None else _initial.copy()
        # A fresh production Problem has an empty numerical solver cache.
        # Leave _cache (the DPP reduction graph) intact.
        energy_problem._solver_cache.clear()
        energy_problem.solve(**options)
        status = energy_problem.status
'''
    replacement += textwrap.indent(extraction, "    ")
    replacement += (
        "\n    return _solve_values, energy_problem, _coefficients, options\n"
    )
    source = source[:solve_start] + replacement
    namespace = dict(original.__globals__)
    namespace["_REUSABLE_OBSERVATION_BASIS"] = sparse.csr_matrix(observation_basis)
    exec(compile(source, str(Path(__file__).resolve()) + ":generated", "exec"),
         namespace)
    return namespace["_build_parameter_problem"], hashlib.sha256(
        (vendor_source + band.BAND_REPLACEMENT).encode()).hexdigest()


class ReusableBandSolver:
    """One fixed frame design/error basis, with repeatable observed values.

    Construct a new instance if frame selection, design, operators, solver
    options, or anchor changes. Unequal shot counts need no graph changes:
    this fixed estimator uses each raw row mean, without shot-count weights.
    Instances are process-local and must not be used concurrently.

    ``observation_basis`` is an N-by-K matrix spanning all planned changes
    from ``raw_shadow.values``. It only limits which inputs this compiled
    instance accepts; each accepted input uses the original estimator. Input
    reconstruction must match at absolute tolerance 1e-12. More than 256 rows
    require an explicit basis to prevent a large DPP parameter tensor.
    """

    def __init__(self, context, raw_shadow, basis="spin", *, band=None,
                 observation_basis=None, solver_threads=1):
        if basis not in ("spin", "full"):
            raise ValueError("basis must be spin or full")
        if not isinstance(solver_threads, int) or solver_threads < 1:
            raise ValueError("solver_threads must be a positive integer")
        band = _band_module() if band is None else band
        started = time.perf_counter()
        self._origin = np.asarray(raw_shadow.values, dtype=float).copy()
        self._shape = self._origin.shape
        if self._origin.ndim != 1 or not np.isfinite(self._origin).all():
            raise ValueError("Initial observations must be a finite vector.")
        if observation_basis is None:
            if len(self._origin) > 256:
                raise ValueError(
                    "Large DPP problems require observation_basis (N-by-K) "
                    "to avoid excessive compilation memory. Supply the "
                    "planned observation displacement directions.")
            observation_basis = np.eye(len(self._origin))
        elif sparse.issparse(observation_basis):
            observation_basis = observation_basis.toarray()
        self.observation_basis = np.asarray(observation_basis, dtype=float).copy()
        if (self.observation_basis.ndim != 2
                or self.observation_basis.shape[0] != len(self._origin)
                or self.observation_basis.shape[1] < 1
                or not np.isfinite(self.observation_basis).all()):
            raise ValueError("observation_basis must be a finite N-by-K matrix, K >= 1.")
        self._coefficient_map = np.linalg.pinv(self.observation_basis, rcond=1e-12)
        builder, source_hash = _make_builder(band, basis == "spin",
                                             self.observation_basis)
        self.basis = basis
        self.solver_threads = solver_threads
        self._solve_values, self.problem, self.parameter, self.options = builder(
            band.r.j.LeakGuardReference(context.sel), shadow_data=raw_shadow,
            shadow_error_weight=band.NUCLEAR_WEIGHT,
            selection_objective="energy", solver=context.args.solver,
            tolerance=context.args.solver_tolerance,
            max_iterations=context.args.max_iterations,
            solver_threads=self.solver_threads,
            positivity_conditions="DQG", symmetry_blocked_psd=True,
            initial_d2=context.base.d2, initial_gamma=context.base.gamma,
            initial_corrected_d2=context.base.d2)
        self.build_seconds = time.perf_counter() - started
        self.source_hash = source_hash
        self.solve_count = 0
        self.last_stats = None
        self._mosek_cache = (_MosekRhsCache(self.problem)
                             if str(context.args.solver).upper() == "MOSEK"
                             else None)

    def solve(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != self._shape or not np.isfinite(values).all():
            raise ValueError("Observations must have the original shape and be finite.")
        _release_temporary_memory()
        rss_before = _current_rss_gib()
        started = time.perf_counter()
        coefficients = self._coefficient_map @ (values - self._origin)
        represented = self._origin + self.observation_basis @ coefficients
        representation_error = float(np.max(np.abs(represented-values), initial=0.))
        if representation_error > 1e-12:
            raise ValueError(
                "Observations lie outside the supplied affine span: maximum "
                f"reconstruction error {representation_error:.3g} exceeds 1e-12.")
        result = self._solve_values(coefficients)
        wall = time.perf_counter() - started
        _release_temporary_memory()
        self.solve_count += 1
        stats = self.problem.solver_stats
        self.last_stats = dict(
            basis=self.basis, solve_index=self.solve_count,
            solver_threads=self.solver_threads,
            build_seconds=self.build_seconds,
            wall_seconds=wall,
            compilation_seconds=self.problem.compilation_time,
            solver_seconds=stats.solve_time,
            solver_setup_seconds=stats.setup_time,
            solver_iterations=stats.num_iters,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            rss_before_solve_gib=rss_before,
            rss_after_cleanup_gib=_current_rss_gib(),
            status=str(result.status), estimator_source_hash=self.source_hash,
            numerical_warm_start_reused=False,
            dpp_cache_ready=self.problem._cache.param_prog is not None,
            observation_rows=len(self._origin),
            parameter_count=self.observation_basis.shape[1],
            observation_reconstruction_error=representation_error,
            mosek_static_cache_ready=(self._mosek_cache.ready
                                      if self._mosek_cache else False),
            mosek_static_cache_hits=(self._mosek_cache.hits
                                     if self._mosek_cache else 0),
        )
        return result


def _self_check(solver="SCS"):
    """Compare unchanged and parameterized solves on a tiny physical fixture."""
    band = _band_module()
    vendor = band.vendor
    pairs = vendor.pair_basis(4)
    d2 = np.zeros((len(pairs), len(pairs)))
    d2[pairs.index((0, 2)), pairs.index((0, 2))] = 1.
    gamma = np.diag([1., 0., 1., 0.])
    reference = vendor.MolecularReference(
        statevector=np.zeros(16), exact_energy=-2., nuclear_energy=0.,
        exact_gamma=gamma, exact_d2=d2,
        one_body=np.diag([-1., -.2, -1., -.2]),
        two_body=np.eye(len(pairs)) * .15, pairs=pairs,
        n_spatial_orbitals=2, n_spin_orbitals=4,
        n_alpha=1, n_beta=1, orbital_irreps=(0, 0))
    vectors = np.eye(len(pairs))
    design = vendor.quadratic_design(vectors)
    values = design @ d2.ravel()
    values = .96 * values + .04 / len(values)
    raw = vendor.ShadowData(
        rotations=(np.eye(2),), pair_vectors=vectors,
        design=sparse.csr_matrix(design), values=values,
        lower_bounds=values.copy(), upper_bounds=values.copy(),
        hits=np.rint(500 * values).astype(int), shots_per_basis=500,
        exact_values=np.full(len(values), np.nan), exact_constraints=False,
        occupations=None)
    context = SimpleNamespace(
        sel=reference, base=SimpleNamespace(d2=d2, gamma=gamma),
        args=SimpleNamespace(solver=solver, solver_tolerance=1e-7,
                             max_iterations=20000))
    rows = []
    for basis in ("spin", "full"):
        band.SOLVE_BAND = band.band_solver(basis == "spin")
        directions = np.column_stack((np.ones(len(values)),
                                       np.linspace(-1., 1., len(values))))
        repeat = ReusableBandSolver(context, raw, basis, band=band,
                                    observation_basis=directions)
        if repeat._mosek_cache is not None:
            repeat._mosek_cache.validate_against_original = True
        samples = (values, values + directions @ [.003, .001],
                   values + directions @ [-.002, .0003],
                   values + directions @ [.0005, -.0002], values)
        for index, sample in enumerate(samples):
            baseline = band.solve_original_raw(context, replace(raw, values=sample))
            actual = repeat.solve(sample)
            np.testing.assert_allclose(actual.d2, baseline.d2, atol=1e-10, rtol=0)
            np.testing.assert_allclose(actual.gamma, baseline.gamma, atol=1e-10, rtol=0)
            np.testing.assert_allclose(actual.shadow_error_trace,
                                       baseline.shadow_error_trace,
                                       atol=1e-10, rtol=0)
            rows.append(dict(sample=index, max_abs_d2_difference=float(
                np.max(np.abs(actual.d2-baseline.d2))), **repeat.last_stats))
        if repeat._mosek_cache is not None:
            repeat._mosek_cache.validate_against_original = False
            actual = repeat.solve(values)
            np.testing.assert_array_equal(actual.d2, baseline.d2)
            rows.append(dict(sample="production_cache_path",
                             max_abs_d2_difference=0., **repeat.last_stats))
        invalid = values.copy()
        invalid[0] += .001
        try:
            repeat.solve(invalid)
        except ValueError as error:
            assert "affine span" in str(error)
        else:
            raise AssertionError("Out-of-span observations were accepted.")
    print(json.dumps(dict(self_check="passed", solves=rows), indent=2), flush=True)


def _validate_n2(args):
    import direction_analysis as d
    context = d.setup()
    counts, data = d.sampled(context, args.stream, args.budget)
    raw = d.band.all_rows_shadow(context, data, counts)
    reference_path = d.CACHE / f"{args.basis}_b{args.budget}_r{args.stream}.npz"
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    with np.load(reference_path) as archive:
        expected_d2 = archive["d2"].copy()
        expected_gamma = archive["gamma"].copy()
    changed = raw.values.copy()
    changed[:len(changed)//len(counts)] += 1e-4
    fit = ReusableBandSolver(context, raw, args.basis, band=d.band,
                             observation_basis=(changed-raw.values)[:, None],
                             solver_threads=args.solver_threads)
    records = []
    for index in range(args.repeats):
        result = fit.solve(raw.values)
        delta = float(np.max(np.abs(result.d2 - expected_d2)))
        np.testing.assert_allclose(result.d2, expected_d2, atol=1e-8, rtol=0)
        np.testing.assert_allclose(result.gamma, expected_gamma, atol=1e-8, rtol=0)
        record = dict(repeat=index, max_abs_d2_difference=delta,
                      **fit.last_stats)
        records.append(record)
        print(json.dumps(record), flush=True)
    if args.compare_perturbed:
        d.band.SOLVE_BAND = d.band.band_solver(args.basis == "spin")
        started = time.perf_counter()
        original = d.band.solve_original_raw(context, replace(raw, values=changed))
        original_seconds = time.perf_counter() - started
        actual = fit.solve(changed)
        np.testing.assert_allclose(actual.d2, original.d2, atol=1e-8, rtol=0)
        np.testing.assert_allclose(actual.gamma, original.gamma, atol=1e-8, rtol=0)
        records.append(dict(perturbed=True,
                            original_wall_seconds=original_seconds,
                            max_abs_d2_difference=float(np.max(np.abs(
                                actual.d2-original.d2))), **fit.last_stats))
        print(json.dumps(records[-1]), flush=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-n2", action="store_true")
    parser.add_argument("--basis", choices=("spin", "full"), default="spin")
    parser.add_argument("--budget", type=int, default=30000)
    parser.add_argument("--stream", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--solver-threads", type=int, default=1)
    parser.add_argument("--compare-perturbed", action="store_true")
    parser.add_argument("--self-check-solver", default="SCS",
                        choices=("SCS", "MOSEK", "CLARABEL"))
    parser.add_argument("--output")
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    if arguments.solver_threads < 1:
        parser.error("--solver-threads must be positive")
    if arguments.validate_n2:
        _validate_n2(arguments)
    else:
        _self_check(arguments.self_check_solver)
