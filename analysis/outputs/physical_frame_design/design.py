"""State-independent design for the physical targets protected in reconstruction.

This module accepts arrays and particle counts only. It cannot access a CI state,
an exact RDM, an energy evaluator, or simulated measurement outcomes.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular, svd
from scipy.optimize import minimize


def inverse(a):
    return cho_solve(cho_factor((a + a.T) / 2), np.eye(len(a)))


def identifiable_covariance(information, targets):
    """A pseudoinverse is valid only after checking target estimability."""
    eig, vec = np.linalg.eigh((information + information.T) / 2)
    keep = eig > eig[-1] * 1e-10
    missing = float(np.linalg.norm(targets @ vec[:, ~keep]))
    if missing > 1e-7:
        raise ValueError(f'Physical targets are not identifiable: null-space overlap {missing}')
    covariance = (vec[:, keep] / eig[keep]) @ vec[:, keep].T
    return covariance, int(keep.sum()), missing


def blue_factors(blocks,covariances,counts,lift,targets):
    """Solve GLS without squaring its condition number or estimating rank from I.

    Structural rank is decided from the unweighted observation map. The
    whitened solve is then performed entirely in that identified space.
    """
    xblocks=[a@lift for a in blocks]
    stacked=np.vstack(xblocks)
    _,singular,v=svd(stacked,full_matrices=stacked.shape[0]<stacked.shape[1])
    keep=np.zeros(len(v),dtype=bool)
    keep[:len(singular)]=singular>singular[0]*1e-10
    observed=v[keep].T
    null=v[~keep].T
    missing=float(np.linalg.norm(targets@null))
    if missing>1e-7:
        raise ValueError(f'Physical targets are not identifiable from measurement geometry: {missing}')
    whitened=[]
    whites=[]
    for x,cov,count in zip(xblocks,covariances,counts):
        white=np.sqrt(count)*solve_triangular(cholesky((cov+cov.T)/2,lower=True),
                                             np.eye(len(cov)),lower=True)
        whites.append(white)
        whitened.append(white@x@observed)
    u,s,v=svd(np.vstack(whitened),full_matrices=False)
    if s[-1]<=s[0]*np.finfo(float).eps*max(np.vstack(whitened).shape):
        raise ValueError('Identified weighted design exceeds numerical precision')
    right=observed@(v.T/s)
    covariance=right@right.T
    influences=[]
    start=0
    for white in whites:
        stop=start+len(white)
        influences.append(right@u[start:stop].T@white)
        start=stop
    return covariance,influences,int(keep.sum()),missing,float(s[0]/s[-1])


def sector_covariance(n, na, nb, pairs):
    occupations = []
    for alpha in combinations(range(n), na):
        for beta in combinations(range(n), nb):
            x = np.zeros(2 * n)
            x[list(alpha)] = 1
            x[np.asarray(beta) + n] = 1
            occupations.append(x)
    occupations = np.asarray(occupations)
    pairs = np.asarray(pairs)
    values = occupations[:, pairs[:, 0]] * occupations[:, pairs[:, 1]]
    centered = values - values.mean(0)
    covariance = centered.T @ centered / len(values)
    eig, vec = np.linalg.eigh(covariance)
    keep = eig > 1e-10
    # Exact null directions are fixed-particle identities, not noisy observations.
    whitening = (vec[:, keep] / np.sqrt(eig[keep])).T
    return covariance, whitening


@dataclass
class Geometry:
    n: int
    pairs: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    lift: np.ndarray
    whitening: np.ndarray
    density: np.ndarray
    contact: np.ndarray

    def vectors(self, u):
        p, q = self.pairs.T
        spin = np.kron(np.eye(2), u)
        wedge = spin[p[:, None], p] * spin[q[:, None], q] - spin[p[:, None], q] * spin[q[:, None], p]
        return wedge.T

    def block(self, u):
        v = self.vectors(u)
        return (v[:, self.rows].conj() * v[:, self.cols]).real * np.where(self.rows == self.cols, 1, 2)

    def information(self, u):
        x = self.whitening @ self.block(u) @ self.lift
        return x.T @ x

    @property
    def weight(self):
        return self.density.T @ self.density / len(self.density) + self.contact.T @ self.contact / len(self.contact)


def diagnostics(geo, rotations):
    info = sum(geo.information(u) for u in rotations) / len(rotations)
    eig = np.linalg.eigvalsh(info)
    rank = int(np.sum(eig > eig[-1] * 1e-10))
    c, rank, missing = identifiable_covariance(info, np.vstack((geo.density, geo.contact)))
    return dict(rank=rank, dimension=len(info), risk=float(np.trace(geo.weight @ c)),
                density_risk=float(np.trace(geo.density @ c @ geo.density.T) / len(geo.density)),
                contact_risk=float(np.trace(geo.contact @ c @ geo.contact.T) / len(geo.contact)),
                all_risk=float(np.trace(c) / len(c)) if rank == len(info) else None,
                target_null_overlap=missing, condition=float(eig[-1] / eig[-rank]),
                minimum_eigenvalue=float(eig[0]),
                max_unitarity_error=float(max(np.linalg.norm(u.conj().T @ u - np.eye(geo.n)) for u in rotations)))


def greedy(geo, pool, count, criterion='physical', refine=False, maxiter=50, weight=None, information=None, reference_ridge=None):
    infos = [geo.information(u) for u in pool] if information is None else information
    dim = len(infos[0])
    ridge = 1e-3 * np.trace(np.mean(infos, axis=0)) / dim if reference_ridge is None else reference_ridge
    precision = ridge * np.eye(dim)
    q = geo.weight if weight is None else weight
    available = list(range(len(pool)))
    chosen, indices, log = [], [], []
    for step in range(count):
        def score(a):
            if criterion == 'dopt':
                return -np.linalg.slogdet(a)[1]
            return np.trace(q @ inverse(a))
        scores = [score(precision + infos[i]) for i in available]
        index = available[int(np.argmin(scores))]
        u = pool[index].copy()
        before = float(min(scores))
        diag = dict(step=step + 1, seed_pool_index=index, before=before)
        if refine:
            u, opt = refine_frame(geo, precision, u, q, maxiter=maxiter)
            diag.update(opt)
        precision += geo.information(u) if refine else infos[index]
        diag['after'] = float(score(precision))
        assert diag['after'] <= before + 1e-7 * max(abs(before), 1)
        log.append(diag)
        chosen.append(u)
        indices.append(index)
        available.remove(index)
    return np.asarray(chosen), dict(indices=indices, ridge=ridge, steps=log, **diagnostics(geo, chosen))


def refine_frame(geo, precision, start, weight, maxiter=50):
    """Optimize U exp(K), K anti-Hermitian, with derivatives of design risk only."""
    import torch
    torch.set_num_threads(1)
    dtype = torch.float64
    cdtype = torch.complex128
    n = geo.n
    basis = []
    # Diagonal phases are redundant at the starting frame but harmless after
    # a finite update. Include them to cover the full unitary Lie algebra.
    for i in range(n):
        a = np.zeros((n, n), complex)
        a[i, i] = 1j
        basis.append(a)
        for j in range(i + 1, n):
            a = np.zeros((n, n), complex)
            a[i, j], a[j, i] = 1, -1
            basis.append(a)
            a = np.zeros((n, n), complex)
            a[i, j] = a[j, i] = 1j
            basis.append(a)
    generators = torch.tensor(np.asarray(basis), dtype=cdtype)
    u0 = torch.tensor(start, dtype=cdtype)
    p, r = torch.tensor(geo.pairs.T, dtype=torch.long)
    rows, cols = torch.tensor(geo.rows), torch.tensor(geo.cols)
    scale = torch.tensor(np.where(geo.rows == geo.cols, 1., 2.))
    whiten = torch.tensor(geo.whitening, dtype=dtype)
    lift = torch.tensor(geo.lift, dtype=dtype)
    current = torch.tensor(precision, dtype=dtype)
    q = torch.tensor(weight, dtype=dtype)
    def function(x, return_rotation=False):
        x = torch.tensor(x, dtype=dtype, requires_grad=True)
        k = torch.einsum('a,aij->ij', x.to(cdtype), generators)
        u = u0 @ torch.matrix_exp(k)
        if return_rotation:
            return u.detach().numpy()
        spin = torch.block_diag(u, u)
        v = (spin[p[:, None], p] * spin[r[:, None], r] - spin[p[:, None], r] * spin[r[:, None], p]).T
        block = (v[:, rows].conj() * v[:, cols]).real * scale
        a = whiten @ block @ lift
        value = torch.trace(torch.linalg.solve(current + a.T @ a, q))
        value.backward()
        return float(value.detach()), x.grad.detach().numpy()
    x0 = np.zeros(len(basis))
    initial, gradient = function(x0)
    direction = np.random.default_rng(9011).normal(size=len(x0))
    direction /= np.linalg.norm(direction)
    epsilon = 1e-5
    finite_difference = (function(epsilon*direction)[0]-function(-epsilon*direction)[0])/(2*epsilon)
    analytic = gradient @ direction
    gradient_error = abs(finite_difference-analytic)/max(1.,abs(finite_difference),abs(analytic))
    assert gradient_error < 2e-4, (gradient_error,finite_difference,analytic)
    result = minimize(function, x0, jac=True, method='L-BFGS-B',
                      options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-7, maxls=25))
    u = function(result.x, return_rotation=True) if result.fun < initial else start.copy()
    return u, dict(optimizer_success=bool(result.success), iterations=int(result.nit),
                   evaluations=int(result.nfev), gradient_norm=float(np.linalg.norm(gradient)),
                   initial_risk=float(initial), optimized_risk=float(min(result.fun, initial)),
                   gradient_check_relative_error=float(gradient_error))
