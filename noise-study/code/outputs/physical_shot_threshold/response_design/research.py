"""All-row DQG controls and H/F response-directed measurement design.

Truth enters acquisition and score() only. design_stage uses the DQG anchor,
known operators, particle-sector covariance and existing state-blind frames.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/response_design_mpl'
import sys, json, time, argparse, resource, hashlib, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.linalg import eigh, svd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'cancellation_repair'))
import study as old
from study import ex, j, pc, save, digest, counts, coordinates, score
from design import blue_factors, greedy, refine_frame, inverse, sector_covariance
from smooth_replay import smooth_covariance
from protection_variants import solve_variant

SOURCE_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_snapshots=HERE/'source_snapshots'
_snapshots.mkdir(exist_ok=True)
_snapshot=_snapshots/(SOURCE_HASH+'.py')
if not _snapshot.exists():_snapshot.write_bytes(Path(__file__).read_bytes())


def context(system):
    c = ex.context(system)
    c.system = system
    a = np.load(ROOT/'cancellation_repair/results'/system/'anchor.npz')
    c.base = SimpleNamespace(d2=a['d2'], gamma=a['gamma'], status='optimal')
    c.theta0 = c.base.d2[c.geo['rows'], c.geo['cols']]
    c.exact_z = coordinates(c, c.exact.exact_d2, c.theta0)
    c.star_eval = c.objective.evaluate(c.exact.exact_d2, c.exact.exact_gamma, gradient=False)
    c.href, c.fref = j._energy_values(c.sel,c.objective,c.exact.exact_d2,c.exact.exact_gamma)
    c.h = j._hamiltonian_gradient(c.sel,c.geo['rows'],c.geo['cols'])@c.lift
    c.f = j._raw_ftpbe_gradient(c.objective,c.sel,c.base.d2,c.base.gamma,c.geo['rows'],c.geo['cols'])@c.lift
    print('CONTEXT',system,flush=True)
    return c


def install_frames(c, name):
    if name=='uniform30':
        sys.path.insert(0,str(ROOT))
        from uniform import uniform_frames
        rotations,c.vectors,c.blocks,_,_=uniform_frames(c,c.system,0)
        c.rotations=np.asarray(rotations)
    elif name == 'generated8':
        path = old.OLD/'designs'/c.system/'pool0/generated8.npz'
    else:
        path = HERE/'designs'/c.system/(name+'.npz')
    if name!='uniform30':
        c.rotations = np.load(path)['rotations']
        c.vectors, _, c.blocks, _, _ = j._build_random_design(c.sel, c.rotations)
    c.oracle = j.AcquisitionOracle(c.exact, c.rotations, c.vectors, 1, 0)
    folder = HERE/'cache'/c.system
    folder.mkdir(parents=True,exist_ok=True)
    key = digest(c.rotations,c.oracle.ci)
    cache = folder/(key+'.npz')
    if cache.exists():
        c.oracle._probabilities = dict(enumerate(np.load(cache)['probabilities']))
    c.exact_y = np.array([a@c.exact.exact_d2[c.geo['rows'],c.geo['cols']] for a in c.blocks])
    probs = np.array([c.oracle._frame_probabilities(k) for k in range(len(c.rotations))])
    np.testing.assert_allclose(probs@c.oracle.indicators,c.exact_y,atol=2e-10,rtol=0)
    if not cache.exists():
        with tempfile.NamedTemporaryFile(dir=folder,suffix='.npz',delete=False) as handle:
            temporary=Path(handle.name)
        np.savez_compressed(temporary,probabilities=probs)
        temporary.replace(cache)


def geometry_fit(c, covs, ns, y):
    targets = np.vstack((c.blind.density,c.blind.contact,c.h,c.f))
    C, backs, rank, missing, condition = blue_factors(c.blocks,covs,ns,c.lift,targets)
    zhat = sum(b@(v-a@c.theta0) for b,v,a in zip(backs,y,c.blocks))
    eig, u = eigh((C+C.T)/2)
    assert eig[-rank] > 0
    W = (u[:,-rank:]/np.sqrt(eig[-rank:])).T
    # This is the sufficient GLS statistic on the identifiable affine space.
    np.testing.assert_allclose(W@C@W.T,np.eye(rank),atol=2e-7,rtol=0)
    theta=c.theta0+c.lift@zhat
    # Constant discarded by quadratic GLS, required by the square-root loss.
    sector=c.blind.whitening
    # Particle identities have exactly zero stochastic variance. Numerical
    # epsilon in their covariance nullspace must not become measured noise.
    irreducible=sum(float(n*(sector@(v-a@theta))@np.linalg.solve(
        sector@cov@sector.T,sector@(v-a@theta))) for v,a,cov,n in zip(y,c.blocks,covs,ns))
    full_irreducible=sum(float(n*(v-a@theta)@np.linalg.solve(cov,v-a@theta))
        for v,a,cov,n in zip(y,c.blocks,covs,ns))
    return SimpleNamespace(C=C,backs=backs,rank=rank,missing=missing,condition=condition,
                           W=W,zhat=zhat,covs=covs,counts=ns,irreducible=irreducible,
                           full_irreducible=full_irreducible)


def robust_covariance(c,outcomes,alpha=50.):
    """Particle-sector mixture with fixed total mass, independent of CAS size."""
    values=np.asarray(outcomes,dtype=float)
    n=c.sel.n_spatial_orbitals;na=c.sel.n_alpha;nb=c.sel.n_beta
    prior_cov,_=sector_covariance(n,na,nb,c.sel.pairs)
    mean0=np.array([na*(na-1)/(n*(n-1)) if i<n and k<n else
        nb*(nb-1)/(n*(n-1)) if i>=n and k>=n else na*nb/(n*n) for i,k in c.sel.pairs])
    denominator=len(values)+alpha
    mean=(values.sum(0)+alpha*mean0)/denominator
    second=(values.T@values+alpha*(prior_cov+np.outer(mean0,mean0)))/denominator
    cov=second-np.outer(mean,mean)
    cov=(cov+cov.T)/2
    return cov+1e-8*np.max(np.diag(cov))*np.eye(len(cov))


def gls(c,p,tau=0.):
    L=p.W@(c.geo['Z'].T*c.geo['scale'][None,:])
    # Scale the objective uniformly for the solver; tau is in SE units.
    scale=1/np.sqrt(p.rank)
    affine=pc.affine_map(scale*L,c.geo,len(c.sel.pairs),-scale*(L@c.theta0+p.W@p.zhat))
    kw={}
    if tau:
        coefficient=tau/(p.rank*np.sqrt(c.h@p.C@c.h))
        kw=dict(additional_d2_objective=coefficient*c.sel.two_body,
                additional_gamma_objective=coefficient*c.sel.one_body)
    return j.solve_dqg_sdp(j.LeakGuardReference(c.sel),affine_objective=affine,
        selection_objective='affine_least_squares',solver=c.args.solver,
        tolerance=c.args.solver_tolerance,max_iterations=c.args.max_iterations,solver_threads=1,
        positivity_conditions='DQG',symmetry_blocked_psd=True,
        initial_d2=c.base.d2,initial_gamma=c.base.gamma,**kw)


def response_solve(c,p,shadows):
    B=np.vstack((c.h,c.f))
    eig,u=eigh(B@p.C@B.T)
    assert eig[0]>0
    joint=(u/np.sqrt(eig)).T@B*np.sqrt(.01/len(B))
    # Retain a small physical-input penalty to avoid leaving every other mode
    # to the nuclear completion. All targets share the SAME measured zhat.
    factor=np.vstack([joint]+[np.sqrt(.1)*w for w in old.factors(c,p.C)])
    L=factor@(c.geo['Z'].T*c.geo['scale'][None,:])
    affine=pc.affine_map(L,c.geo,len(c.sel.pairs),-L@c.theta0-factor@p.zhat)
    return j.solve_dqg_sdp(j.LeakGuardReference(c.sel),shadow_data=ex._blind_shadows(shadows),
        affine_objective=affine,selection_objective='affine_least_squares',
        additional_d2_objective=c.sel.two_body,additional_gamma_objective=c.sel.one_body,
        shadow_error_weight=c.args.nuclear_weight,solver=c.args.solver,
        tolerance=c.args.solver_tolerance,max_iterations=c.args.max_iterations,solver_threads=1,
        positivity_conditions='DQG',symmetry_blocked_psd=True,
        initial_d2=c.base.d2,initial_gamma=c.base.gamma,initial_corrected_d2=c.base.d2)


def consistent_gls(c,p,tau):
    """Retain quadratic GLS when certified; cap the square-root H slope at .9."""
    result=gls(c,p,tau)
    z=coordinates(c,result.d2,c.theta0)
    residual=np.sqrt(p.irreducible+np.linalg.norm(p.W@(z-p.zhat))**2)
    eta=tau/(2*max(residual,1e-14))
    details=dict(uncapped_eta=float(eta),eta_cap=.9,consistency_refit=bool(eta>.9),
                 irreducible_chisquare=p.irreducible,initial_residual_norm=float(residual))
    if eta<=.9:
        # The two convex objectives have identical gradients at this solution;
        # the same primal/dual KKT conditions therefore certify the root loss.
        return result,details
    from square_root import solve_dqg_sdp
    L=p.W@(c.geo['Z'].T*c.geo['scale'][None,:])
    offset=np.r_[-L@c.theta0-p.W@p.zhat,np.sqrt(p.irreducible)]/np.sqrt(p.rank)
    L=np.vstack((L,np.zeros(L.shape[1])))/np.sqrt(p.rank)
    affine=pc.affine_map(L,c.geo,len(c.sel.pairs),offset)
    coefficient=.9/(np.sqrt(c.h@p.C@c.h)*np.sqrt(p.rank))
    result=solve_dqg_sdp(j.LeakGuardReference(c.sel),affine_objective=affine,
        selection_objective='affine_least_squares',additional_d2_objective=coefficient*c.sel.two_body,
        additional_gamma_objective=coefficient*c.sel.one_body,solver=c.args.solver,
        tolerance=c.args.solver_tolerance,max_iterations=c.args.max_iterations,solver_threads=1,
        positivity_conditions='DQG',symmetry_blocked_psd=True,
        initial_d2=result.d2,initial_gamma=result.gamma)
    return result,details


def record(c, p, folder, stem, method, result, seconds, metadata):
    assert result.status=='optimal',result.status
    z=coordinates(c,result.d2,c.theta0)
    rec=dict(system=c.system,method=method,seconds=seconds,status=result.status,
        information_rank=p.rank,blue_condition=p.condition,
        fit_chisquare=float(np.linalg.norm(p.W@(z-p.zhat))**2),
        source_hash=SOURCE_HASH,
        linear_h_error_meh=float(1000*c.h@(p.zhat-c.exact_z)),
        linear_f_error_meh=float(1000*c.f@(p.zhat-c.exact_z)),
        working_h_se_meh=float(1000*np.sqrt(c.h@p.C@c.h)),
        working_f_se_meh=float(1000*np.sqrt(c.f@p.C@c.f)),**score(c,result),**metadata)
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/(stem+'.npz'),d2=result.d2,gamma=result.gamma,
        zhat=p.zhat,C=p.C,W=p.W,counts=p.counts)
    save(folder/(stem+'.json'),rec)
    print('RESULT',stem,'H',round(rec['h_error_meh'],5),'F',round(rec['f_error_meh'],5),
          'D',round(rec['d2_error'],5),'sec',round(seconds,1),flush=True)


def replay(c, pilots=(0,1), streams=(0,1)):
    install_frames(c,'generated8')
    source=ROOT/'cancellation_repair/results'/c.system
    folder=HERE/'development'/c.system
    for pilot in pilots:
        pd=j._sample_outcomes(c.oracle,202609301100+7919*pilot,counts(15000,len(c.rotations)))
        covariance_sets=dict(emp=tuple(j._regularized_single_covariance(x.astype(float)) for x in pd),
            smooth=tuple(smooth_covariance(x,c.sel.n_spatial_orbitals,c.sel.n_alpha,c.sel.n_beta,c.sel.pairs) for x in pd))
        for stream in streams:
            src=np.load(source/f'p{pilot}_r{stream}_b75000_NN.npz')
            for covname,covs in covariance_sets.items():
                p=geometry_fit(c,covs,src['counts'],src['full_values'])
                for tau in (0.,1.):
                    method=f'gls_{covname}_t{tau:g}'
                    stem=f'p{pilot}_r{stream}_{method}'
                    if (folder/(stem+'.json')).exists(): continue
                    started=time.time()
                    result=gls(c,p,tau)
                    record(c,p,folder,stem,method,result,time.time()-started,
                        dict(pilot=pilot,stream=stream,budget=75000,frame='generated8',
                             covariance=covname,tau=tau,cohort='development',
                             observation_digest=digest(src['full_values'])))


def design_stage(c):
    folder=HERE/'designs'/c.system
    folder.mkdir(parents=True,exist_ok=True)
    pool=np.load(old.OLD/'designs'/c.system/'pool0/random30.npz')['rotations']
    # Reuse physical frames as starting proposals only, not as restrictions.
    generated=np.load(old.OLD/'designs'/c.system/'pool0/generated8.npz')['rotations']
    pool=np.concatenate((pool,generated))
    infos=[c.blind.information(u) for u in pool]
    ridge=.001*np.trace(np.mean(infos,axis=0))/len(c.h)
    C0=inverse(np.mean(infos,axis=0)+ridge*np.eye(len(c.h)))
    qh=np.outer(c.h,c.h)/(c.h@C0@c.h)
    qf=np.outer(c.f,c.f)/(c.f@C0@c.f)
    qp=c.blind.weight/np.trace(c.blind.weight@C0)
    weight=.495*(qh+qf)+.01*qp
    if (folder/'response12.npz').exists():
        print('DESIGN EXISTS',flush=True)
        return
    rotations,diag=greedy(c.blind,pool,12,weight=weight,refine=True,maxiter=35,information=infos)
    # One coordinate-exchange sweep corrects greedy ordering effects.
    exchange=[]
    for k in range(len(rotations)):
        precision=ridge*np.eye(len(c.h))+sum(c.blind.information(u) for i,u in enumerate(rotations) if i!=k)
        rotations[k],record_=refine_frame(c.blind,precision,rotations[k],weight,maxiter=35)
        exchange.append(record_)
        print('EXCHANGE',c.system,k,flush=True)
    def risks(us):
        info=sum(c.blind.information(u) for u in us)/len(us)
        eig,v=eigh(info)
        keep=eig>eig[-1]*1e-10
        C=(v[:,keep]/eig[keep])@v[:,keep].T
        target=np.vstack((c.h,c.f,c.blind.density,c.blind.contact))
        missing=float(np.linalg.norm(target@v[:,~keep]))
        assert missing<1e-7,missing
        return dict(task_risk=float(np.trace(weight@C)),h_risk=float(c.h@C@c.h),
            f_risk=float(c.f@C@c.f),physical_risk=float(np.trace(c.blind.weight@C)),
            rank=int(keep.sum()),target_null_overlap=missing)
    for name,us in [('response12',rotations),('generated8',generated)]:
        if name=='response12':
            np.savez_compressed(folder/(name+'.npz'),rotations=us,weight=weight,h=c.h,f=c.f)
        diag[name]=risks(us)
    diag.update(exchange=exchange,weight_fractions=[.495,.495,.01],gradient_reference='DQG anchor',
        covariance='fixed-particle maximally mixed',frame_count=12,oracle_inputs=False,
        frame_digest=digest(rotations),source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(folder/'design.json',diag)
    print('DESIGN',c.system,diag['generated8'],diag['response12'],flush=True)


def trials(c,frame,pilots,streams,methods,cohort):
    if cohort=='validation':
        frozen=json.loads((HERE/'FREEZE.json').read_text())
        assert set(methods)<=set(frozen['allowed_methods'][frame])
        assert set(pilots)<=set(frozen['pilots']) and tuple(streams)==(0,)
    install_frames(c,frame)
    if cohort=='validation' and frame=='response12':
        assert digest(c.rotations)==frozen['frame_digests'][c.system]
    folder=HERE/cohort/c.system/frame
    folder.mkdir(parents=True,exist_ok=True)
    n=len(c.rotations)
    pilot_counts=counts(15000,n);ns=counts(60000,n)
    for pilot in pilots:
        if cohort=='development': pseed=202609301100+7919*pilot
        else: pseed=202610017100+7919*pilot
        pd=j._sample_outcomes(c.oracle,pseed,pilot_counts)
        empirical=tuple(j._regularized_single_covariance(x.astype(float)) for x in pd)
        smooth=tuple(smooth_covariance(x,c.sel.n_spatial_orbitals,c.sel.n_alpha,c.sel.n_beta,c.sel.pairs) for x in pd)
        robust=tuple(robust_covariance(c,x) for x in pd)
        basis=ex.literal_basis(frame,c.rotations,c.vectors,c.blocks,empirical)
        for stream in streams:
            if cohort=='development': seed=202609302100+104729*pilot+1543*stream
            else: seed=202610018100+104729*pilot+1543*stream
            data=j._sample_outcomes(c.oracle,seed,ns)
            y=np.asarray([v.mean(0) for v in data])
            shadows=j._finite_shadows(basis,data,ns)
            fitting={name:geometry_fit(c,covs,ns,y) for name,covs in [('emp',empirical),('smooth',smooth),('robust',robust)]}
            obsfile=folder/f'p{pilot}_r{stream}_observations.npz'
            if obsfile.exists():
                prev=np.load(obsfile)
                np.testing.assert_array_equal(prev['full_values'],y)
                np.testing.assert_array_equal(prev['cov_emp'],empirical)
            np.savez_compressed(obsfile,
                full_values=y,pilot_means=np.array([v.mean(0) for v in pd]),counts=ns,
                pilot_counts=pilot_counts,cov_emp=empirical,cov_smooth=smooth,cov_robust=robust,global_rows=basis.global_rows,
                rotations=c.rotations,h=c.h,f=c.f,theta0=c.theta0)
            for method in methods:
                stem=f'p{pilot}_r{stream}_{method}'
                if (folder/(stem+'.json')).exists():continue
                covariance='smooth' if 'smooth' in method else ('robust' if 'robust' in method else 'emp')
                p=fitting[covariance]
                started=time.time()
                print('START',c.system,cohort,frame,pilot,stream,method,flush=True)
                details={}
                if method.startswith('consistent_'):
                    result,details=consistent_gls(c,p,float(method.split('_t')[1]))
                elif method.startswith('gls_'):
                    result=gls(c,p,float(method.split('_t')[1]))
                elif method.startswith('total_response'):
                    result=response_solve(c,p,shadows)
                else:
                    variant=method.removesuffix('_smooth').removesuffix('_robust')
                    result=solve_variant(c,c.base,shadows,c.theta0,p.C,p.zhat,variant)
                record(c,p,folder,stem,method,result,time.time()-started,
                    dict(pilot=pilot,stream=stream,budget=75000,frame=frame,
                        pilot_shots=15000,production_shots=60000,pilot_seed=pseed,production_seed=seed,
                        covariance=covariance,cohort=cohort,observation_digest=digest(y),
                        frame_digest=digest(c.rotations),**details))


def uniform_trials(c,pilots,cohort):
    install_frames(c,'uniform30')
    out=HERE/cohort/c.system/'uniform30';out.mkdir(parents=True,exist_ok=True)
    design=j._constraint_basis_from_covariances('uniform30',tuple(range(30)),c.rotations,c.vectors,c.blocks,
        tuple(np.eye(len(c.sel.pairs)) for _ in range(30)),len(c.sel.pairs),len(c.geo['rows']))
    ns=j._equal_counts(30,design.basis.active_frames,75000,c.args.allocation_chunk)
    assert sum(ns)==75000
    for pilot in pilots:
        stem=f'p{pilot}_r0_original'
        if (out/(stem+'.json')).exists():continue
        seed=202610018100+104729*pilot
        data=j._sample_outcomes(c.oracle,seed,ns)
        shadows=j._finite_shadows(design.basis,data,ns)
        print('START',c.system,cohort,'uniform30',pilot,flush=True)
        start=time.time()
        result=old.original(c,c.base,shadows)
        assert result.status=='optimal'
        rec=dict(system=c.system,cohort=cohort,method='original',frame='uniform30',pilot=pilot,
            stream=0,budget=75000,pilot_shots=0,production_shots=75000,production_seed=seed,
            seconds=time.time()-start,status=result.status,source_hash=SOURCE_HASH,
            covariance='identity_for_structural_QR_no_pilot',frame_digest=digest(c.rotations),**score(c,result))
        full=np.array([x[:n].mean(0) if n else np.zeros(len(c.sel.pairs)) for x,n in zip(data,ns)])
        rec['observation_digest']=digest(full)
        np.savez_compressed(out/(stem+'.npz'),d2=result.d2,gamma=result.gamma,counts=ns,
            full_values=full,global_rows=design.basis.global_rows,rotations=c.rotations)
        save(out/(stem+'.json'),rec)
        print('RESULT',c.system,'uniform30',pilot,'H',rec['h_error_meh'],'F',rec['f_error_meh'],flush=True)


def exact_checks(c):
    install_frames(c,'response12')
    obs=np.load(HERE/'development'/c.system/'response12/p0_r0_observations.npz')
    pd=j._sample_outcomes(c.oracle,202609301100,obs['pilot_counts'])
    covs=tuple(robust_covariance(c,x) for x in pd)
    p=geometry_fit(c,covs,obs['counts'],c.exact_y)
    folder=HERE/'exact_checks'/c.system
    for tau,ratio in [(0.,1.),(8.,1.),(8.,10000.)]:
        stem=f'exact_t{tau:g}_information{ratio:g}'
        if (folder/(stem+'.json')).exists():continue
        q=SimpleNamespace(**vars(p));q.C=p.C/ratio;q.W=p.W*np.sqrt(ratio)
        start=time.time();result=gls(c,q,tau)
        record(c,q,folder,stem,'oracle_exact',result,time.time()-start,
            dict(oracle_arm=True,covariance='robust',tau=tau,information_multiplier=ratio))


if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('system',choices=['c2','n2']);a.add_argument('stage',choices=['replay','design','trials','uniform','exact'])
    a.add_argument('--pilots',default='0,1');a.add_argument('--streams',default='0,1')
    a.add_argument('--frame',default='response12');a.add_argument('--cohort',default='development')
    a.add_argument('--methods',default='shrink,total_response,gls_emp_t4,gls_emp_t8')
    args=a.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    c=context(args.system)
    if args.stage=='design':design_stage(c)
    elif args.stage=='uniform':uniform_trials(c,tuple(map(int,args.pilots.split(','))),args.cohort)
    elif args.stage=='exact':exact_checks(c)
    elif args.stage=='trials':trials(c,args.frame,tuple(map(int,args.pilots.split(','))),
        tuple(map(int,args.streams.split(','))),args.methods.split(','),args.cohort)
    else:replay(c,tuple(map(int,args.pilots.split(','))),tuple(map(int,args.streams.split(','))))
