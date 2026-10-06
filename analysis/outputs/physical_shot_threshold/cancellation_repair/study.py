"""Paired factorial intervention. Exact inputs are labeled oracle arms only."""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/cancellation_repair_mpl'
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
OLD = HERE.parent.parent / 'physical_frame_design'
sys.path.insert(0, str(OLD))
import numpy as np
from scipy.linalg import eigh
from scipy.stats import norm
import experiment as ex
from experiment import j, pc
from design import blue_factors
from constrained_shadow import LinearRDMConstraints, contract_one_rdm, dqg_matrices


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=convert) + '\n')
    temporary.replace(path)


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def counts(total, n):
    v = np.full(n, total // n, dtype=int)
    v[:total % n] += 1
    return v


def coordinates(c, d2, theta0):
    return c.geo['Z'].T @ (c.geo['scale'] * (d2[c.geo['rows'], c.geo['cols']] - theta0))


def original(c, baseline, shadows):
    return j._solve_eq11(c.args, c.sel, shadows, baseline.d2, baseline.gamma, baseline.d2)


def setup(system):
    print('SETUP',system,flush=True)
    c = ex.context(system)
    c.out = HERE / 'results' / system
    c.out.mkdir(parents=True, exist_ok=True)
    file = c.out / 'anchor.npz'
    if file.exists():
        a = np.load(file)
        baseline = SimpleNamespace(d2=a['d2'], gamma=a['gamma'], status='optimal')
    else:
        baseline = j.solve_dqg_sdp(j.LeakGuardReference(c.sel), solver=c.args.solver,
            tolerance=c.args.solver_tolerance, max_iterations=c.args.max_iterations,
            solver_threads=1, positivity_conditions='DQG', symmetry_blocked_psd=True)
        assert baseline.status == 'optimal'
        np.savez_compressed(file, d2=baseline.d2, gamma=baseline.gamma)
    c.base = baseline
    c.theta0 = baseline.d2[c.geo['rows'], c.geo['cols']]
    c.exact_z = coordinates(c, c.exact.exact_d2, c.theta0)
    np.savez_compressed(c.out/'exact_reference.npz',d2=c.exact.exact_d2,gamma=c.exact.exact_gamma)
    c.star_eval = c.objective.evaluate(c.exact.exact_d2, c.exact.exact_gamma, gradient=False)
    c.href, c.fref = j._energy_values(c.sel, c.objective, c.exact.exact_d2, c.exact.exact_gamma)
    c.rotations = np.load(OLD / 'designs' / system / 'pool0' / 'generated8.npz')['rotations']
    c.vectors, _, c.blocks, _, _ = j._build_random_design(c.sel, c.rotations)
    c.oracle = j.AcquisitionOracle(c.exact, c.rotations, c.vectors, 1, 0)
    cache=c.out/'simulator_probabilities.npz'
    cache_key=digest(c.rotations,c.oracle.ci)
    if cache.exists():
        cached=np.load(cache)
        assert str(cached['key'])==cache_key
        c.oracle._probabilities={k:v for k,v in enumerate(cached['probabilities'])}
    c.exact_y = np.array([a @ c.exact.exact_d2[c.geo['rows'], c.geo['cols']] for a in c.blocks])
    # Independent simulator-to-linear-map check of every full frame mean.
    born_y = np.array([c.oracle._frame_probabilities(k) @ c.oracle.indicators for k in range(len(c.rotations))])
    if not cache.exists():
        np.savez_compressed(cache,key=cache_key,probabilities=np.array([c.oracle._frame_probabilities(k) for k in range(len(c.rotations))]))
    np.testing.assert_allclose(c.exact_y, born_y, atol=2e-10, rtol=0)
    c.h = j._hamiltonian_gradient(c.sel, c.geo['rows'], c.geo['cols']) @ c.lift
    np.savez_compressed(c.out/'geometry.npz',**c.geo,exact_z=c.exact_z)
    save(c.out / 'reference_checks.json', dict(system=system, configuration=ex.SYSTEMS[system],
        frame_digest=digest(c.rotations), born_map_error=float(np.max(abs(c.exact_y-born_y))),
        tangent_dimension=len(c.exact_z), anchor=score(c, baseline),
        solver_tolerance=c.args.solver_tolerance, nuclear_weight=c.args.nuclear_weight))
    print('SETUP_DONE',system,'tangent',len(c.exact_z),flush=True)
    return c


def pilot(c, p, budget):
    n = len(c.rotations)
    pseed = 202609301100 + 7919*p
    pcounts = counts(15000, n)
    data = j._sample_outcomes(c.oracle, pseed, pcounts)
    covs = tuple(j._regularized_single_covariance(x.astype(float)) for x in data)
    basis = ex.literal_basis('generated8', c.rotations, c.vectors, c.blocks, covs)
    ns = counts(budget-15000, n)
    targets = np.vstack((c.blind.density, c.blind.contact, c.h[None, :]))
    covariance, backs, rank, missing, condition = blue_factors(c.blocks, covs, ns, c.lift, targets)
    blue_exact = sum(back @ (y-a@c.theta0) for back, y, a in zip(backs,c.exact_y,c.blocks))
    np.testing.assert_allclose(targets@blue_exact, targets@c.exact_z, atol=2e-8, rtol=0)
    np.savez_compressed(c.out/f'p{p}_b{budget}_measurement.npz',global_rows=basis.global_rows,
        covariances=np.asarray(covs),C=covariance,backs=np.asarray(backs),blocks=np.asarray(c.blocks),
        counts=ns,pilot_counts=pcounts,pilot_means=np.array([x.mean(0) for x in data]))
    save(c.out/f'p{p}_b{budget}_measurement.json',dict(pilot_seed=pseed,information_rank=rank,
        tangent_dimension=len(c.exact_z),qr_rows=len(basis.global_rows),
        qr_row_digest=digest(basis.global_rows),blue_condition=condition,target_null_overlap=missing))
    return SimpleNamespace(p=p, pseed=pseed, counts=ns, pcounts=pcounts, data=data, covs=covs,
        basis=basis, C=covariance, backs=backs, rank=rank, missing=missing, condition=condition,
        blue_exact=blue_exact)


def production(c, p, r):
    seed = 202609302100 + 104729*p.p + 1543*r
    data = j._sample_outcomes(c.oracle, seed, p.counts)
    y = np.asarray([x.mean(0) for x in data])
    noisy = j._finite_shadows(p.basis, data, p.counts)
    exact_values = c.exact_y.reshape(-1)[p.basis.global_rows]
    exact = replace(noisy, values=exact_values, lower_bounds=exact_values.copy(),
                    upper_bounds=exact_values.copy())
    np.testing.assert_allclose(noisy.values, y.reshape(-1)[p.basis.global_rows], atol=0, rtol=0)
    zhat = sum(b @ (v-a@c.theta0) for b, v, a in zip(p.backs,y,c.blocks))
    return SimpleNamespace(seed=seed, y=y, noisy=noisy, exact=exact, zhat=zhat)


def factors(c, covariance):
    g, r = c.blind.density, c.blind.contact
    reg = np.linalg.solve(g@covariance@g.T, g@covariance@r.T).T
    groups = [g, r-reg@g]
    result = []
    for w in groups:
        eig, u = eigh(w@covariance@w.T)
        assert eig[0] > 0
        result.append((u/np.sqrt(eig)).T @ w * np.sqrt(.01/len(w)))
    return result


def solve(c, p, shadows, target_z, arm, guard=None, relax=False):
    if arm.startswith('original'):
        return original(c, c.base, shadows)
    fs = factors(c, p.C)
    factor = np.vstack(fs)
    deviation = factor @ target_z
    if arm == 'shrink':
        deviation = np.concatenate([np.sign(f@target_z)*np.maximum(abs(f@target_z)-np.sqrt(.01/len(f)),0)
                                    for f in fs])
    to_z = c.geo['Z'].T*c.geo['scale'][None,:]
    l = factor@to_z
    affine = pc.affine_map(l, c.geo, len(c.sel.pairs), -l@c.theta0-deviation)
    bands = None
    if guard is not None:
        white = guard['white']
        lm = white@to_z
        target = lm@c.theta0+white@target_z
        mapping = pc.affine_map(lm, c.geo, len(c.sel.pairs), np.zeros(len(lm)))
        bands = LinearRDMConstraints(d2_map=mapping.d2_map,
            lower_bounds=target-guard['threshold'], upper_bounds=target+guard['threshold'],
            allow_minimax_relaxation=relax, name='joint_total_response_bands')
    return j.solve_dqg_sdp(j.LeakGuardReference(c.sel), shadow_data=ex._blind_shadows(shadows),
        affine_objective=affine, selection_objective='affine_least_squares',
        additional_d2_objective=c.sel.two_body, additional_gamma_objective=c.sel.one_body,
        linear_constraints=bands, shadow_error_weight=c.args.nuclear_weight,
        solver=c.args.solver, tolerance=c.args.solver_tolerance, max_iterations=c.args.max_iterations,
        solver_threads=1, positivity_conditions='DQG', symmetry_blocked_psd=True,
        initial_d2=c.base.d2, initial_gamma=c.base.gamma, initial_corrected_d2=c.base.d2)


def score(c, result):
    dd = result.d2-c.exact.exact_d2
    dg = result.gamma-c.exact.exact_gamma
    z = coordinates(c,result.d2,c.theta0)
    err_z = z-c.exact_z
    h2 = 1000*float(np.sum(c.sel.two_body*dd))
    hg = 1000*float(np.sum(c.sel.one_body*dg))
    ev = c.objective.evaluate(result.d2,result.gamma,gradient=False)
    fn = 1000*(ev.non_ontop_energy-c.star_eval.non_ontop_energy)
    fo = 1000*(ev.on_top_energy-c.star_eval.on_top_energy)
    eh, ef = j._energy_values(c.sel,c.objective,result.d2,result.gamma)
    identity = max(abs(h2+hg-1000*(eh-c.href)), abs(fn+fo-1000*(ef-c.fref)))
    minimum = min(float(np.linalg.eigvalsh(m)[0]) for m in dqg_matrices(result.d2,result.gamma,c.sel.pairs))
    contraction = float(np.linalg.norm(result.gamma-contract_one_rdm(result.d2,c.sel.n_spin_orbitals,c.sel.n_electrons,c.sel.pairs)))
    eq = float(np.linalg.norm(c.geo['E']@result.d2[c.geo['rows'],c.geo['cols']]+c.geo['offset']))
    assert identity<1e-7 and minimum>-2e-6 and contraction<1e-6 and eq<1e-6, (identity,minimum,contraction,eq)
    return dict(h_error_meh=1000*(eh-c.href), f_error_meh=1000*(ef-c.fref),
        d2_error=float(np.linalg.norm(dd)), gamma_error=float(np.linalg.norm(dg)),
        density_error=float(np.linalg.norm(c.blind.density@err_z)),
        contact_error=float(np.linalg.norm(c.blind.contact@err_z)),
        h_d2_meh=h2,h_gamma_meh=hg,f_non_ontop_meh=fn,f_on_top_meh=fo,
        h_cross_meh2=2*h2*hg,f_cross_meh2=2*fn*fo,
        cr_h=abs(h2+hg)/(abs(h2)+abs(hg)) if abs(h2)+abs(hg)>1e-5 else None,
        cr_f_parts=abs(fn+fo)/(abs(fn)+abs(fo)) if abs(fn)+abs(fo)>1e-5 else None,
        identity_error_meh=identity,minimum_dqg_eigenvalue=minimum,contraction_error=contraction,
        equality_residual=eq)


def record(c,p,s,r,budget,arm,shadows,target,guard=None):
    path=c.out/f'p{p.p}_r{r}_b{budget}_{arm}.json'
    if path.exists():
        old=json.loads(path.read_text())
        assert old['observation_digest']==digest(shadows.values,target,p.C), path
        return old
    start=time.time()
    print('START',c.sel.molecule.atom,p.p,r,arm,flush=True)
    relaxed=False
    try:
        result=solve(c,p,shadows,target,arm,guard)
    except RuntimeError as e:
        if guard is None or 'infeasible' not in str(e).lower(): raise
        relaxed=True
        result=solve(c,p,shadows,target,arm,guard,relax=True)
    assert result.status=='optimal',result.status
    metrics=score(c,result)
    exact_gate=(metrics['d2_error']<1e-5 and abs(metrics['h_error_meh'])<.01 and abs(metrics['f_error_meh'])<.01)
    rec=dict(system=c.out.name,pilot=p.p,stream=r,budget=budget,arm=arm,
        pilot_seed=p.pseed,production_seed=s.seed,pilot_shots=int(sum(p.pcounts)),
        production_shots=int(sum(p.counts)),status=result.status,seconds=time.time()-start,
        information_rank=p.rank,target_null_overlap=p.missing,blue_condition=p.condition,
        observation_digest=digest(shadows.values,target,p.C),full_observation_digest=digest(s.y),
        oracle_arm=arm in ('EE','NE','EN','original_E','guard_EE'),relaxed=relaxed,
        exact_gate_pass=exact_gate if arm in ('EE','original_E','guard_EE') else None,
        source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**metrics)
    corrected=getattr(result,'corrected_d2',None)
    if corrected is not None:
        rec.update(shadow_correction_nuclear_norm=float(np.linalg.norm(corrected-result.d2,ord='nuc')),
            corrected_measurement_residual=float(np.linalg.norm(shadows.predict(corrected)-shadows.values)))
    if guard is not None:
        z=coordinates(c,result.d2,c.theta0)
        residual=guard['white']@(z-target)
        rec.update(guard_residual=residual,guard_threshold=guard['threshold'],
            guard_max_violation=max(0.,float(max(abs(residual))-guard['threshold'])),
            guard_original_mahalanobis=None,
            guard_truth_residual=guard['white']@(c.exact_z-target),
            linear_f_error_meh=1000*float(guard['g']@(z-c.exact_z)),
            linearization_remainder_meh=metrics['f_error_meh']-1000*float(guard['g']@(z-c.exact_z)))
        if not relaxed: assert rec['guard_max_violation']<1e-4
    np.savez_compressed(path.with_suffix('.npz'),d2=result.d2,gamma=result.gamma,
        C=p.C,zhat=target,qr_values=shadows.values,full_values=s.y,
        counts=p.counts,pilot_counts=p.pcounts,
        corrected_d2=corrected if corrected is not None else np.empty((0,0)))
    save(path,rec)
    print('DONE',c.out.name,p.p,r,arm,'H',round(rec['h_error_meh'],5),'F',round(rec['f_error_meh'],5),
          'D',round(rec['d2_error'],6),'seconds',round(rec['seconds'],1),flush=True)
    if arm in ('EE','guard_EE') and not exact_gate:
        print('EXACT_GATE_FAILED_RETAINED',path,flush=True)
    return rec


def make_guard(c,p,budget):
    file=c.out/f'p{p.p}_pilot_reference.npz'
    if file.exists():
        d=np.load(file); ref=SimpleNamespace(d2=d['d2'],gamma=d['gamma'])
    else:
        shadows=j._finite_shadows(p.basis,p.data,p.pcounts)
        start=time.time(); ref=original(c,c.base,shadows)
        assert ref.status=='optimal'
        np.savez_compressed(file,d2=ref.d2,gamma=ref.gamma)
        save(file.with_suffix('.json'),dict(seconds=time.time()-start,pilot_seed=p.pseed,
            shots=int(sum(p.pcounts)),exact_state_used=False,status=ref.status))
    g=j._raw_ftpbe_gradient(c.objective,c.sel,ref.d2,ref.gamma,c.geo['rows'],c.geo['cols'])@c.lift
    B=np.vstack((c.h,g))
    # Recheck target identifiability independently of physical-feature targets.
    blue_factors(c.blocks,p.covs,p.counts,c.lift,B/np.linalg.norm(B,axis=1)[:,None])
    S=B@p.C@B.T
    ev,u=eigh(S)
    keep=ev>ev[-1]*1e-10
    white=(u[:,keep]/np.sqrt(ev[keep])).T@B
    np.testing.assert_allclose(white@p.C@white.T,np.eye(sum(keep)),atol=1e-8)
    threshold=float(norm.ppf(1-.05/(2*sum(keep))))
    # Finite difference along three deterministic directions, including the anchor path.
    zp=coordinates(c,ref.d2,c.theta0)
    rng=np.random.default_rng(94821)
    checks=[]
    for v in [coordinates(c,c.base.d2,c.theta0)-zp,rng.normal(size=len(g)),rng.normal(size=len(g))]:
        v=v/np.linalg.norm(v)
        mat=pc.matrix(c.lift@v,c.geo,len(c.sel.pairs))
        gam=contract_one_rdm(mat,c.sel.n_spin_orbitals,c.sel.n_electrons,c.sel.pairs)
        eps=1e-5
        plus=c.objective.evaluate(ref.d2+eps*mat,ref.gamma+eps*gam,gradient=False).total_energy
        minus=c.objective.evaluate(ref.d2-eps*mat,ref.gamma-eps*gam,gradient=False).total_energy
        fd=(plus-minus)/(2*eps)
        checks.append(dict(analytic=float(g@v),finite_difference=fd,error=abs(fd-g@v)))
    assert max(x['error'] for x in checks)<2e-5,checks
    diag=dict(B=B,covariance=S,white=white,rank=int(sum(keep)),threshold=threshold,
        f_gradient_checks=checks,pilot_exact_state_used=False)
    save(c.out/f'p{p.p}_b{budget}_guard_definition.json',diag)
    return dict(white=white,g=g,threshold=threshold,zp=zp)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('system',choices=['c2','n2'])
    parser.add_argument('--stage',choices=['factorial','candidates','audit'],default='factorial')
    parser.add_argument('--pilots',default='0,1')
    parser.add_argument('--streams',default='0,1')
    parser.add_argument('--budget',type=int,default=75000)
    args=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    c=setup(args.system)
    if args.stage=='audit':
        checked=[]
        for file in sorted(c.out.glob('p*_r*_b*.json')):
            rec=json.loads(file.read_text())
            if 'arm' not in rec:continue
            arr=np.load(file.with_suffix('.npz'))
            result=SimpleNamespace(d2=arr['d2'],gamma=arr['gamma'])
            fresh=score(c,result)
            for key in ['h_error_meh','f_error_meh','h_d2_meh','h_gamma_meh','f_non_ontop_meh','f_on_top_meh',
                        'd2_error','gamma_error','density_error','contact_error']:
                np.testing.assert_allclose(fresh[key],rec[key],atol=1e-7,rtol=0,err_msg=str(file)+' '+key)
            checked.append(file.name)
        save(c.out/'independent_energy_audit.json',dict(all_passed=True,records=len(checked),files=checked))
        print('ENERGY_AUDIT_COMPLETE',args.system,len(checked),flush=True)
        return
    for pp in map(int,args.pilots.split(',')):
        p=pilot(c,pp,args.budget)
        guard=make_guard(c,p,args.budget) if args.stage=='candidates' else None
        for rr in map(int,args.streams.split(',')):
            s=production(c,p,rr)
            if args.stage=='factorial':
                record(c,p,s,-1,args.budget,'original_E',s.exact,c.exact_z)
                record(c,p,s,-1,args.budget,'EE',s.exact,c.exact_z)
                for arm,shadow,target in [('original_N',s.noisy,s.zhat),('NE',s.noisy,c.exact_z),
                                          ('EN',s.exact,s.zhat),('NN',s.noisy,s.zhat)]:
                    record(c,p,s,rr,args.budget,arm,shadow,target)
            else:
                # Stage two requires a completed and validated factorial for this stream.
                for arm in ['EE','NE','EN','NN','original_N']:
                    r=-1 if arm=='EE' else rr
                    assert (c.out/f'p{pp}_r{r}_b{args.budget}_{arm}.json').exists()
                record(c,p,s,-1,args.budget,'guard_EE',s.exact,c.exact_z,guard)
                for arm in ['shrink','guard_hf']:
                    record(c,p,s,rr,args.budget,arm,s.noisy,s.zhat,guard if arm=='guard_hf' else None)
    print('STAGE_COMPLETE',args.system,args.stage,flush=True)


if __name__=='__main__': main()
