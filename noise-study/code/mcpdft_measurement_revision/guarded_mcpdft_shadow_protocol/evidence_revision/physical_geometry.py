"""Extract the actual solver's affine equalities without solving its SDP.

This avoids maintaining a second, potentially divergent implementation of spin,
trace, contraction and spatial equalities. No production solver is patched.
The temporary Problem interception is local to this single-threaded diagnostic.
"""
from contextlib import contextmanager
import numpy as np
from scipy import sparse
from scipy.linalg import null_space
import cvxpy as cp
from run_evidence import j
from constrained_shadow import contract_one_rdm


def solver_equalities(reference, rows, cols):
    n=len(reference.pairs);m=reference.n_spin_orbitals;d=len(rows)
    captured={}
    class Captured(Exception):pass
    original=cp.Problem
    def intercept(objective, constraints):
        captured['constraints']=constraints
        raise Captured()
    try:
        cp.Problem=intercept
        j.solve_dqg_sdp(j.LeakGuardReference(reference),solver='MOSEK',
             positivity_conditions='DQG',symmetry_blocked_psd=True,
             initial_d2=np.zeros((n,n)),initial_gamma=np.zeros((m,m)))
    except Captured:
        pass
    finally:
        cp.Problem=original
    if not captured:raise RuntimeError('Failed to capture solver constraints')
    # CVXPY expression gradients use column-major variable vectorization.
    dr=[];dc=[];dv=[];gammas=[]
    for k,(a,b) in enumerate(zip(rows,cols)):
        dr.append(a+b*n);dc.append(k);dv.append(1.)
        if a!=b:dr.append(b+a*n);dc.append(k);dv.append(1.)
        D=np.zeros((n,n));D[a,b]=D[b,a]=1.
        gammas.append(contract_one_rdm(D,m,reference.n_electrons,reference.pairs).ravel(order='F'))
    Dmap=sparse.coo_matrix((dv,(dr,dc)),shape=(n*n,d)).tocsr()
    Gmap=sparse.csr_matrix(np.asarray(gammas).T)
    matrices=[];offsets=[]
    for con in captured['constraints']:
        if not isinstance(con,cp.constraints.zero.Equality):continue
        expr=con.expr
        jac=sparse.csr_matrix((expr.size,d))
        for variable,gradient in expr.grad.items():
            if gradient is None:raise RuntimeError('Missing equality Jacobian')
            if variable.name()=='D':jac+=gradient.T@Dmap
            elif variable.name()=='gamma':jac+=gradient.T@Gmap
            else:raise RuntimeError(variable.name())
        matrices.append(jac.toarray())
        offsets.extend(np.asarray(expr.value).ravel(order='F'))
    E=np.vstack(matrices);offset=np.array(offsets)
    return E,offset


def physical_lineality(reference, D, gamma, rows, cols, tolerance=1e-8):
    from types import SimpleNamespace
    local=SimpleNamespace(**vars(reference),n_electrons=reference.n_electrons,
                          exact_d2=D,exact_gamma=gamma)
    L,diag=j._dqg_lineality_basis(local,rows,cols,active_tolerance=tolerance)
    E,offset=solver_equalities(reference,rows,cols)
    # Absolute singular threshold: near-zero E@L must not be interpreted as
    # full rank merely because all its singular values are tiny.
    projected=E@L
    _,s,Vh=np.linalg.svd(projected,full_matrices=True)
    cutoff=max(1e-9,1e-9*np.linalg.norm(E,2))
    rank=int(np.count_nonzero(s>cutoff))
    corrected=L@Vh[rank:].T
    diag.update({'affine_equality_rank':int(np.linalg.matrix_rank(E,tol=cutoff)),
        'legacy_lineality_dimension':L.shape[1],
        'physical_lineality_dimension':corrected.shape[1],
        'legacy_equality_violation':float(np.linalg.norm(E@L)),
        'physical_equality_violation':float(np.linalg.norm(E@corrected)),
        'anchor_equality_residual':float(np.linalg.norm(E@D[rows,cols]+offset)),
        'equality_singular_cutoff':cutoff})
    return corrected,diag,E
