"""Local guard15 solver: reuse graph across H/F/radius weights and observations.

Frozen upstream solver files are never edited. The MOSEK cache is admitted only
after checking that A and PSD data are constant; both dual B and c are updated.
"""
import copy
from types import MethodType

import numpy as np
import cvxpy as cp

import gl15
import reusable_band_solver as rbs


class MeritCache(rbs._MosekRhsCache):
    def _vector(self):
        from cvxpy.cvxcore.python import canonInterface
        p = self.formatted
        return canonInterface.get_parameter_vector(
            p.total_param_size, p.param_id_to_col, p.param_id_to_size,
            lambda i: np.asarray(p.id_to_param[i].value))

    def _install(self, data, chain, inverse):
        import cvxpy.settings as s
        from cvxpy.lin_ops.lin_op import CONSTANT_ID
        from cvxpy.reductions.cone2cone import affine2direct as cones
        from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK
        if not isinstance(chain.solver, MOSEK) or not data.get('dualized'):
            raise RuntimeError('Expected dualized MOSEK')
        self.formatted = data[s.PARAM_PROB]
        p = self.formatted
        cone = data['K_dir']
        if cone[cones.DUAL_EXP] or cone[cones.DUAL_POW3D]:
            raise RuntimeError('Unsupported cones')
        tensor = p.A.tocsc()
        rows, remainder = divmod(tensor.shape[0], p.x.size + 1)
        if remainder:
            raise RuntimeError('Unexpected parameter tensor')
        dynamic = [k for k in range(tensor.shape[1])
                   if k != p.param_id_to_col[CONSTANT_ID]]
        delta = tensor[:, dynamic]
        nonzero = delta.indices[delta.data != 0]
        offset = rows * p.x.size
        linear_end = cone[cones.FREE] + cone[cones.NONNEG]
        if np.any(nonzero < offset) or np.any(nonzero >= offset + linear_end):
            raise RuntimeError('Parameters change A or nonlinear cone RHS')
        prefix = linear_end + sum(cone[cones.SOC])
        self.linear_rhs_tensor = tensor[offset:offset + prefix, :].tocsr()
        self.objective_tensor = p.q.tocsr()
        self.static_data = dict(data)
        self.inverse = inverse
        self.original_adapter = chain.solver
        vector = self._vector()
        q = np.asarray(self.objective_tensor @ vector).ravel()
        np.testing.assert_array_equal(q[:-1], data[s.B])
        np.testing.assert_array_equal(-np.asarray(self.linear_rhs_tensor @ vector).ravel(), data[s.C])
        np.testing.assert_allclose(q[-1], inverse[s.OBJ_OFFSET], atol=0, rtol=0)
        adapter = copy.copy(chain.solver)
        adapter.apply = MethodType(lambda _, program: self.apply(program), adapter)
        chain.reductions[-1] = adapter
        chain.solver = adapter
        self.ready = True

    def apply(self, program):
        import cvxpy.settings as s
        if program is not self.formatted:
            raise RuntimeError('Parameter program changed')
        vector = self._vector()
        q = np.asarray(self.objective_tensor @ vector).ravel()
        data = dict(self.static_data)
        data[s.B] = q[:-1]
        data[s.C] = -np.asarray(self.linear_rhs_tensor @ vector).ravel()
        inverse = dict(self.inverse)
        inverse[s.OBJ_OFFSET] = q[-1]
        if self.validate_against_original:
            expected, expected_inverse = self.original_adapter.apply(program)
            self._assert_same_data(data, expected)
            np.testing.assert_array_equal(inverse[s.OBJ_OFFSET], expected_inverse[s.OBJ_OFFSET])
        self.hits += 1
        return data, inverse


class MeritSolver:
    def __init__(self, d, c, raw, directions, threads=2, gradients=None):
        # Disable only the old RHS-only cache while constructing this instance.
        original_cache = rbs._MosekRhsCache
        rbs._MosekRhsCache = lambda problem: None
        try:
            self.fit = gl15.make_fit(c, raw, 'spin', band=d.band,
                                   observation_basis=directions,
                                   solver_threads=threads, nucleus_weight=1.)
        finally:
            rbs._MosekRhsCache = original_cache
        variables = {v.name(): v for v in self.fit.problem.variables()}
        d2, gamma = variables['D'], variables['gamma']
        radius = cp.trace(variables['shadow_error_positive']) + cp.trace(variables['shadow_error_negative'])
        gd, gg = gl15.f_gradients(c) if gradients is None else gradients
        linear_f = cp.sum(cp.multiply((gd + gd.T) / 2, d2)) + cp.sum(cp.multiply((gg + gg.T) / 2, gamma))
        # Original problem is H + T; subtract T algebraically to retain the
        # exact Hamiltonian expression (including upstream conventions).
        h = self.fit.problem.objective.expr - radius
        self.weights = cp.Parameter(3, nonneg=True, value=[1., 0., 1.])
        self.fit.problem._objective = cp.Minimize(
            self.weights[0]*h + self.weights[1]*linear_f + self.weights[2]*radius)
        if not self.fit.problem.is_dcp(dpp=True):
            raise RuntimeError('Weight parameterization is not DPP')
        self.fit._mosek_cache = MeritCache(self.fit.problem)

    def solve(self, values, lam, mu, normalized=False):
        scale = 1. / (1. + mu) if normalized else 1.
        self.weights.value = scale * np.array([1., mu, lam])
        result = self.fit.solve(values)
        if result.status != 'optimal':
            raise RuntimeError(result.status)
        return result

    @property
    def stats(self):
        return dict(self.fit.last_stats, objective_parameter_count=3)
