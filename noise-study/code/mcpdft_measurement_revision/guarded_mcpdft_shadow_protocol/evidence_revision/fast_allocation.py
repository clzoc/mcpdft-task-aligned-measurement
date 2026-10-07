"""Algebraically equivalent Woodbury scoring without forming every trial C."""
import numpy as np
from scipy.linalg import cholesky, solve_triangular, cho_solve, eigh
from run_n2_random_adaptive_allocation import (
    _information, _inverse_information, _risks, _risk_score, _equal_counts)


def woodbury_allocation_paths(basis, production_budgets, minimum, chunk, targets):
    active=basis.active_frames
    counts=np.zeros(len(basis.rotations),dtype=int)
    counts[list(active)]=minimum
    C=_inverse_information(_information(basis,counts))
    paths={'pilot_adaptive':{},'uniform':{}};diagnostics=[]
    modes=np.vstack((targets.hamiltonian,targets.ftpbe_modes))
    for budget in production_budgets:
        if budget<int(counts.sum()) or budget%chunk:raise ValueError('Invalid budget')
        while int(counts.sum())<budget:
            base_d=float(targets.d2_weights@np.diag(C))
            base_modes=np.einsum('ki,ij,kj->k',modes,C,modes,optimize=True)
            candidates=[]
            for f in active:
                A=basis.blocks[f]
                P=A@C
                S=basis.covariances[f]/chunk+P@A.T
                R=solve_triangular(cholesky((S+S.T)/2,lower=True),P,lower=True)
                reduction_d=float(np.einsum('ki,i,ki->',R,targets.d2_weights,R))
                remaining=base_modes-np.sum((R@modes.T)**2,axis=0)
                risks=np.array([base_d-reduction_d,remaining[0],np.max(remaining[1:])])
                candidates.append((_risk_score(risks,targets),int(counts[f]),f,R))
            _,_,f,R=min(candidates,key=lambda x:(x[0][0],x[0][1],x[1],x[2]))
            C-=R.T@R;C=(C+C.T)/2
            counts[f]+=chunk
        paths['pilot_adaptive'][int(budget)]=counts.copy()
        uniform=_equal_counts(len(counts),active,int(budget),chunk)
        paths['uniform'][int(budget)]=uniform
        for method,n in (('uniform',uniform),('pilot_adaptive',counts)):
            risks=_risks(_inverse_information(_information(basis,n)),targets)
            score=_risk_score(risks,targets)
            diagnostics.append({'method':method,'production_shots':int(budget),
                'predicted_max_capacity_ratio':score[0],
                'predicted_mean_capacity_ratio':score[1],
                'predicted_d2_rms':float(np.sqrt(risks[0])),
                'predicted_hamiltonian_rms_meh':float(1000*np.sqrt(risks[1])),
                'predicted_ftpbe_rms_meh':float(1000*np.sqrt(risks[2])),
                'shot_counts':','.join(map(str,n))})
    return paths,diagnostics


def allocation_paths(basis,production_budgets,minimum,chunk,targets):
    """For square literal-row A, C=A^-1 blockdiag(Sigma_f/n_f) A^-T.

    Thus every linear/quadratic task risk is a sum a_f/n_f. Fall back to
    Woodbury for a non-square design or an initially clipped inverse.
    """
    active=tuple(basis.active_frames)
    A=np.vstack([basis.blocks[f] for f in active])
    d=basis.rank
    if A.shape!=(d,d):
        return woodbury_allocation_paths(basis,production_budgets,minimum,chunk,targets)
    counts=np.zeros(len(basis.rotations),dtype=int);counts[list(active)]=minimum
    initial=_information(basis,counts)
    eigenvalues=np.linalg.eigvalsh(initial)
    if eigenvalues[0]<=max(eigenvalues[-1]*1e-11,1e-13):
        return woodbury_allocation_paths(basis,production_budgets,minimum,chunk,targets)
    inverse=np.linalg.solve(A,np.eye(d))
    modes=np.vstack((targets.hamiltonian,targets.ftpbe_modes))
    coefficients=np.zeros((2+len(targets.ftpbe_modes),len(counts)))
    offset=0
    for f in active:
        r=len(basis.blocks[f]);T=inverse[:,offset:offset+r]@cholesky(basis.covariances[f],lower=True)
        coefficients[0,f]=np.einsum('ik,i,ik->',T,targets.d2_weights,T)
        coefficients[1:,f]=np.sum((modes@T)**2,axis=1)
        offset+=r
    paths={'pilot_adaptive':{},'uniform':{}};diagnostics=[]
    for budget in production_budgets:
        if budget<int(counts.sum()) or budget%chunk:raise ValueError('Invalid budget')
        while int(counts.sum())<budget:
            base=np.sum(coefficients[:,active]/counts[list(active)],axis=1)
            candidates=[]
            for f in active:
                trial=base-coefficients[:,f]*(1/counts[f]-1/(counts[f]+chunk))
                risks=np.array([trial[0],trial[1],np.max(trial[2:])])
                candidates.append((_risk_score(risks,targets),int(counts[f]),f))
            _,_,f=min(candidates,key=lambda x:(x[0][0],x[0][1],x[1],x[2]))
            counts[f]+=chunk
        paths['pilot_adaptive'][int(budget)]=counts.copy()
        uniform=_equal_counts(len(counts),active,int(budget),chunk)
        paths['uniform'][int(budget)]=uniform
        for method,n in (('uniform',uniform),('pilot_adaptive',counts)):
            risks=_risks(_inverse_information(_information(basis,n)),targets)
            score=_risk_score(risks,targets)
            diagnostics.append({'method':method,'production_shots':int(budget),
                'predicted_max_capacity_ratio':score[0],
                'predicted_mean_capacity_ratio':score[1],
                'predicted_d2_rms':float(np.sqrt(risks[0])),
                'predicted_hamiltonian_rms_meh':float(1000*np.sqrt(risks[1])),
                'predicted_ftpbe_rms_meh':float(1000*np.sqrt(risks[2])),
                'shot_counts':','.join(map(str,n))})
    return paths,diagnostics


def greedy_frame_order(args,blocks,covariances,targets,lineality):
    updates=tuple(A.T@np.linalg.solve(S,A) for A,S in zip(blocks,covariances))
    d=blocks[0].shape[1]
    full=sum(updates,np.zeros((d,d)))
    ridge=max(args.design_ridge_fraction*np.trace(full)/d,1e-12)
    regularized=ridge*np.eye(d)
    restricted_updates=tuple(lineality.T@J@lineality for J in updates)
    restricted=np.zeros((lineality.shape[1],lineality.shape[1]))
    covariance_logdet=[np.linalg.slogdet(S)[1] for S in covariances]
    modes=np.vstack((targets.hamiltonian,targets.ftpbe_modes))
    remaining=set(range(len(blocks)));selected=[];diagnostics=[]
    while remaining:
        chol=cholesky((regularized+regularized.T)/2,lower=True)
        C=cho_solve((chol,True),np.eye(d))
        base_d=float(targets.d2_weights@np.diag(C))
        base_modes=np.einsum('ki,ij,kj->k',modes,C,modes,optimize=True)
        candidates=[]
        for f in sorted(remaining):
            A=blocks[f];P=A@C
            S=covariances[f]+P@A.T
            R=solve_triangular(cholesky((S+S.T)/2,lower=True),P,lower=True)
            mode_risks=base_modes-np.sum((R@modes.T)**2,axis=0)
            risks=np.array([base_d-np.einsum('ki,i,ki->',R,targets.d2_weights,R),
                            mode_risks[0],np.max(mode_risks[1:])])
            score=_risk_score(risks,targets)
            gain=float(np.linalg.slogdet(S)[1]-covariance_logdet[f])
            conic=None if not lineality.shape[1] else float(eigh(
                restricted+restricted_updates[f],subset_by_index=[0,0],
                check_finite=False,eigvals_only=True)[0]/(len(selected)+1))
            candidates.append({'index':f,'target_max':score[0],'target_mean':score[1],
                               'd_optimal_gain':gain,'conic_gain':conic})
        max_d=max(x['d_optimal_gain'] for x in candidates)
        max_c=max(float(x['conic_gain'] or 0.) for x in candidates)
        eligible=[x for x in candidates if x['d_optimal_gain']>=args.d_optimal_guard_fraction*max_d-1e-12
           and (max_c<=0 or float(x['conic_gain'] or 0.)>=args.conic_guard_fraction*max_c-1e-12)]
        choice=min(eligible,key=lambda x:(x['target_max'],x['target_mean'],
                                         -float(x['conic_gain'] or 0.),-x['d_optimal_gain'],x['index']))
        f=choice['index'];selected.append(f);remaining.remove(f)
        regularized+=updates[f];restricted+=restricted_updates[f]
        diagnostics.append({'order':len(selected),'pool_index':f,
            'target_max':choice['target_max'],'target_mean':choice['target_mean'],
            'd_optimal_gain':choice['d_optimal_gain'],
            'restricted_gain_per_frame_shot':choice['conic_gain']})
    return tuple(selected),diagnostics
