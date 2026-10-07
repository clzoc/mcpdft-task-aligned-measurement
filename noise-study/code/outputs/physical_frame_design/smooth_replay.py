"""Same-data replay with a sector-supported multinomial covariance prior.

One half pseudo-count per allowed occupation configuration (Jeffreys prior
for a saturated multinomial model). QR rows, production means, frames, total
shots, kappa, and the original nuclear-norm objective stay exactly matched.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time,resource
import numpy as np
from scipy.special import comb
from experiment import (HERE,context,j,literal_basis,protected_solve,save,
                        identifiable_covariance)
from constrained_shadow import ShadowData,LinearRDMConstraints
from scipy.linalg import eigh
from experiment import pc
from design import sector_covariance,blue_factors


def smooth_covariance(outcomes,n,na,nb,pairs):
    """Posterior predictive pair covariance; the prior obeys particle identities."""
    values=np.asarray(outcomes,dtype=float)
    count=len(values)
    pairs=np.asarray(pairs)
    prior_cov,_=sector_covariance(n,na,nb,pairs)
    mean0=[]
    for i,k in pairs:
        if i<n and k<n:
            p=na*(na-1)/(n*(n-1))
        elif i>=n and k>=n:
            p=nb*(nb-1)/(n*(n-1))
        else:
            p=na*nb/(n*n)
        mean0.append(p)
    mean0=np.asarray(mean0)
    prior_second=prior_cov+np.outer(mean0,mean0)
    alpha=.5*int(comb(n,na,exact=True))*int(comb(n,nb,exact=True))
    mean=(values.sum(0)+alpha*mean0)/(count+alpha)
    second=(values.T@values+alpha*prior_second)/(count+alpha)
    covariance=(second-np.outer(mean,mean))
    covariance=(covariance+covariance.T)/2
    # Only stabilize exact identity nullspaces. The prior supplies the variance
    # floor on stochastic directions; this numerical epsilon does not do that.
    covariance+=1e-8*np.max(np.diag(covariance))*np.eye(len(pairs))
    return covariance


def lock_solve(c,baseline,shadows,theta0,covariance,zhat):
    """Fix the DQG-feasible physical inputs before H+nuclear completion."""
    g,r=c.blind.density,c.blind.contact
    reg=np.linalg.solve(g@covariance@g.T,g@covariance@r.T).T
    factors=[]
    for w in (g,r-reg@g):
        eig,u=eigh(w@covariance@w.T)
        assert eig[0]>0
        factors.append((u/np.sqrt(eig)).T@w*np.sqrt(.01/len(w)))
    factor=np.vstack(factors)
    l=factor@c.geo['Z'].T*c.geo['scale'][None,:]
    target=l@theta0+factor@zhat
    affine=pc.affine_map(l,c.geo,len(c.sel.pairs),-target)
    first=j.solve_dqg_sdp(j.LeakGuardReference(c.sel),affine_objective=affine,
        selection_objective='affine_least_squares',solver=c.args.solver,
        tolerance=c.args.solver_tolerance,max_iterations=c.args.max_iterations,solver_threads=1,
        positivity_conditions='DQG',symmetry_blocked_psd=True,
        initial_d2=baseline.d2,initial_gamma=baseline.gamma)
    assert first.status=='optimal',first.status
    w=np.vstack((g,r))
    physical=w@c.geo['Z'].T*c.geo['scale'][None,:]
    m=physical@first.d2[c.geo['rows'],c.geo['cols']]
    mapping=pc.affine_map(physical,c.geo,len(c.sel.pairs),np.zeros(len(w))).d2_map
    band=1e-8
    fixed=LinearRDMConstraints(d2_map=mapping,lower_bounds=m-band,upper_bounds=m+band,
                               name='fixed_mc_pdft_inputs')
    result=j.solve_dqg_sdp(j.LeakGuardReference(c.sel),shadow_data=shadows,
        linear_constraints=fixed,selection_objective='energy',shadow_error_weight=c.args.nuclear_weight,
        solver=c.args.solver,tolerance=c.args.solver_tolerance,max_iterations=c.args.max_iterations,
        solver_threads=1,positivity_conditions='DQG',symmetry_blocked_psd=True,
        initial_d2=first.d2,initial_gamma=first.gamma,initial_corrected_d2=first.d2)
    change=physical@(result.d2-first.d2)[c.geo['rows'],c.geo['cols']]
    energy_change=1000*(c.objective.evaluate(result.d2,result.gamma,gradient=False).total_energy-
                       c.objective.evaluate(first.d2,first.gamma,gradient=False).total_energy)
    assert np.max(abs(change))<2e-7,np.max(abs(change))
    assert abs(energy_change)<.005,energy_change
    diagnostics=dict(primary_status=first.status,physical_lock_band=band,
                     maximum_physical_change=float(np.max(abs(change))),
                     completion_ftpbe_change_meh=float(energy_change))
    return result,diagnostics,dict(stage1_d2=first.d2,stage1_gamma=first.gamma,stage1_target=m)


def main(system,pools,methods,budgets,lock=False):
    c=context(system)
    baseline=j.solve_dqg_sdp(j.LeakGuardReference(c.sel),solver=c.args.solver,
        tolerance=c.args.solver_tolerance,max_iterations=c.args.max_iterations,solver_threads=1,
        positivity_conditions='DQG',symmetry_blocked_psd=True)
    rows,cols=c.geo['rows'],c.geo['cols']
    theta0=baseline.d2[rows,cols]
    href,fref=j._energy_values(c.sel,c.objective,c.exact.exact_d2,c.exact.exact_gamma)
    files=sorted((HERE/'results'/system).rglob('*_protected.json'))
    for parent in files:
        old=json.loads(parent.read_text())
        if old['pool'] not in pools or old['method'] not in methods or old['budget'] not in budgets:
            continue
        arm='locked' if lock else 'protected_smooth'
        file=parent.with_name(parent.name.replace('_protected.json',f'_{arm}.json'))
        if file.exists():
            continue
        started=time.time()
        arrays=np.load(parent.with_suffix('.npz'))
        design_id=old.get('design_id',old['method'])
        archive=np.load(HERE/'designs'/system/f"pool{old['pool']}"/f'{design_id}.npz')
        rotations=archive['rotations']
        vectors,_,blocks,_,_=j._build_random_design(c.sel,rotations)
        if design_id.startswith('pilot8_'):
            # Regenerate the original 30-frame pilot and take its selected rows.
            full=np.load(HERE/'designs'/system/f"pool{old['pool']}"/'random30.npz')['rotations']
            fullvectors,_,_,_,_=j._build_random_design(c.sel,full)
            oracle=j.AcquisitionOracle(c.exact,full,fullvectors,1,0)
            fullpilot=j._sample_outcomes(oracle,old['pilot_seed'],np.full(30,500))
            indices=json.loads((HERE/'designs'/system/f"pool{old['pool']}"/f'{design_id}.json').read_text())['indices']
            pilot=tuple(fullpilot[i] for i in indices)
        else:
            oracle=j.AcquisitionOracle(c.exact,rotations,vectors,1,0)
            pilot=j._sample_outcomes(oracle,old['pilot_seed'],arrays['pilot_counts'])
        oldcovs=tuple(j._regularized_single_covariance(x.astype(float)) for x in pilot)
        basis=literal_basis(old['method'],rotations,vectors,blocks,oldcovs)
        values=arrays['qr_values']
        assert len(values)==len(basis.global_rows)
        shadows=ShadowData(rotations=tuple(basis.rotations[i] for i in basis.active_frames),
            pair_vectors=vectors[basis.global_rows],design=basis.design[basis.global_rows],
            values=values,lower_bounds=values.copy(),upper_bounds=values.copy(),
            hits=np.full(len(values),-1),shots_per_basis=0,exact_values=np.full(len(values),np.nan),
            exact_constraints=False,occupations=None)
        covs=tuple(smooth_covariance(x,c.blind.n,c.sel.n_alpha,c.sel.n_beta,c.sel.pairs) for x in pilot)
        info=np.zeros((c.lift.shape[1],)*2)
        rhs=np.zeros(len(info))
        for a,v,count,mean in zip(blocks,covs,arrays['counts'],arrays['full_values']):
            x=a@c.lift
            back=count*np.linalg.solve(v,x).T
            info+=back@x
            rhs+=back@(mean-a@theta0)
        covariance,influences,rank,missing,condition=blue_factors(
            blocks,covs,arrays['counts'],c.lift,np.vstack((c.blind.density,c.blind.contact)))
        zhat=sum(back@(mean-a@theta0) for back,mean,a in zip(influences,arrays['full_values'],blocks))
        extra,extra_arrays={},{}
        if lock:
            result,extra,extra_arrays=lock_solve(c,baseline,shadows,theta0,covariance,zhat)
        else:
            result=protected_solve(c,baseline,shadows,theta0,covariance,zhat)
        assert result.status=='optimal',result.status
        metrics=j._comparison_metrics(result.d2,result.gamma,c.exact.exact_d2,href,fref,c.sel,c.objective)
        delta=c.geo['Z'].T@(c.geo['scale']*(result.d2[rows,cols]-c.exact.exact_d2[rows,cols]))
        record=dict(old)
        record.update(arm=arm,parent_record=str(parent.relative_to(HERE)),
            covariance_method='sector_multinomial_half_pseudocount',status=result.status,seconds=time.time()-started,
            equality_residual=float(np.linalg.norm(c.geo['E']@result.d2[rows,cols]+c.geo['offset'])),
            information_rank=rank,target_null_overlap=missing,
            blue_solver='whitened_svd',blue_condition=condition,
            density_error=float(np.linalg.norm(c.blind.density@delta)),
            contact_error=float(np.linalg.norm(c.blind.contact@delta)),
            gamma_error=float(np.linalg.norm(result.gamma-c.exact.exact_gamma)),
            modeled_density_variance=float(np.trace(c.blind.density@covariance@c.blind.density.T)),
            modeled_contact_variance=float(np.trace(c.blind.contact@covariance@c.blind.contact.T)),**metrics,**extra)
        np.savez_compressed(file.with_suffix('.npz'),d2=result.d2,gamma=result.gamma,C=covariance,zhat=zhat,lift=c.lift,
            counts=arrays['counts'],pilot_counts=arrays['pilot_counts'],full_values=arrays['full_values'],qr_values=values,
            **extra_arrays)
        save(file,record)
        print('LOCKED' if lock else 'SMOOTH',system,old['pool'],old['seed'],old['budget'],old['method'],
              'F',metrics['signed_ftpbe_error_meh'],'D',metrics['d2_error'],'seconds',record['seconds'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('system',choices=['n2','f2'])
    parser.add_argument('--pools',default='0,1,2')
    parser.add_argument('--methods',default='random30,dopt8,physical8,generated8,augmented8')
    parser.add_argument('--budgets',default='150000,300000')
    parser.add_argument('--lock',action='store_true')
    cli=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    main(cli.system,list(map(int,cli.pools.split(','))),cli.methods.split(','),list(map(int,cli.budgets.split(','))),cli.lock)
