#!/usr/bin/env python3
"""Guard15 with equal post-pilot shots; fixed real Haar same-spin pool."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/guard15_equal_real30_mpl'
import argparse
import csv
import fcntl
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'guard15_lammu'))
import numpy as np
import gl15 as g
import co_weak_parts as co
import n2_guard15_pools as gp
from merit_solver import MeritSolver
from reusable_band_solver import ReusableBandSolver
from subset_measurements import selected_raw

BUDGETS = (30000, 60000, 120000, 240000)
SYSTEMS = ('n2', 'co_eq')
ARMS = ('guard15_equal', 'guard15_equal_mu0', 'exclusive_guard15_mu2',
        'exclusive_guard15_mu0', 'uniform')
MU = dict(guard15_equal=2., guard15_equal_mu0=0., exclusive_guard15_mu2=2.,
          exclusive_guard15_mu0=0., uniform=0.)
STREAMS = tuple(range(8))

def save(path, value):
    g.allocation.save(path, value)

def result_path(system, arm, stream, budget):
    return HERE/'results'/system/arm/f'b{budget}_r{stream}.json'

def context(system):
    # Copy anchors locally; retain the existing chemistry and frame-cache code.
    g.c108.HERE = HERE/'contexts'
    co.HERE = HERE/'contexts'
    if system == 'n2':
        c = g.c108.context()
        g.c108.install_uniform_family(c, co.real_frames())
    else:
        c = co.co_context(tag="co_eq", bond=1.128)
    g.da.band.SOLVE_BAND = g.da.band.band_solver(True)
    return g.da, c

def initialize():
    frames = co.real_frames()
    np.testing.assert_array_equal(frames[:, 0], frames[:, 1])
    assert np.max(np.abs(frames.imag)) == 0
    for u in frames[:, 0]:
        np.testing.assert_allclose(u.T@u, np.eye(8), atol=2e-15)
    g.atomic_npz(HERE/'pool_real30.npz', rotations=frames, seed=co.REAL_SEED)
    sources = [Path(__file__), Path(g.__file__), Path(co.__file__), Path(gp.__file__),
               ROOT/'original_bands/screen_frame_subsets.py',
               ROOT/'original_bands/reusable_band_solver.py',
               ROOT/'original_bands/run.py', ROOT/'guard15_lammu/merit_solver.py',
               ROOT/'original_bands/subset_measurements.py',
               Path(g.da.band.vendor.__file__), Path(g.da.r.j.__file__)]
    protocol = dict(
        name=HERE.name, budgets=BUDGETS, streams=STREAMS, arms=ARMS,
        systems=dict(n2=dict(atom='N 0 0 0; N 0 0 1.10', bond_angstrom=1.10,
                            basis='cc-pvdz', cas=[10,8], ncore=2,
                            context='original_global_fit/context_108.context'),
                     co_eq=dict(atom='C 0 0 -0.564; O 0 0 0.564', bond_angstrom=1.128,
                                  geometry_source='NIST ground-state experimental re=1.128323 A, rounded to 1.128 A',
                                  geometry_source_url='https://webbook.nist.gov/cgi/cbook.cgi?ID=C630080&Mask=1000',
                                  basis='cc-pvdz', cas=[10,8], ncore=2, symmetry='C2v',
                                  builder='canonical')),
        pool=dict(kind='Haar O(8), real, identical alpha/beta rotations', frames=30,
                  seed=co.REAL_SEED, digest=g.allocation.digest(frames),
                  fixed_across_systems_streams_budgets=True),
        pilot_per_frame=500, pilot_total=15000, selected_frames=15,
        selection='Existing guard15 backward-greedy path, in identifiable pool row space; K=15',
        production_allocation='(B-15000)/15 new shots on each selected frame',
        selected_pilots_reused=True, unselected_pilots_in_fit=False,
        fit_shots='B-7500', selection_uses_truth=False,
        guard_merit=dict(lambda_radius=1., mu_ftpbe=2., form='H + 2 F_lin(anchor) + Tr(E)'),
        arm_definitions={a:dict(lambda_radius=1.,mu_ftpbe=MU[a],
                               frames='all30' if a=='uniform' else
                               'complement of guard15' if a.startswith('exclusive_') else 'guard15',
                               pilot_shots=0 if a=='uniform' else 15000)
                         for a in ARMS},
        exclusive_rule='Set complement of the same pilot-selected guard15; no new selection; equal remaining shots on complement15; reuse complement pilots',
        baseline=dict(rule='uniform30, no pilot', shots_per_frame='B/30',
                      estimator='original global_band spin', lambda_radius=1., mu_ftpbe=0.),
        error_basis='spin', positivity='DQG', full_E=False,
        seed_formula='202610018100 + 104729*(200+stream)',
        coupling='Same per-frame RNG prefixes across arms and budgets; independent shot streams, fixed pool',
        evaluation='H and actual nonlinear FTPBE F versus exact CAS reference; truth used only for simulated acquisition and scoring',
        solver='MOSEK, tolerance 1e-8; no numerical warm-start reuse',
        compilation='One budget and one arm per subprocess; exact raw observations; original first completed results retained, see runtime_amendment.json',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    save(HERE/'protocol.json', protocol)
    print('INITIALIZED', flush=True)

def selections(d, c, system, stream):
    folder=HERE/'plans'/system
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/f'r{stream}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        paths={b:folder/f'r{stream}_b{b}.json' for b in BUDGETS}
        pilot_path=HERE/'pilots'/system/f'r{stream}.npz'
        if all(p.exists() for p in paths.values()) and pilot_path.exists():
            plans={b:json.loads(p.read_text()) for b,p in paths.items()}
            with np.load(pilot_path) as z:
                samples=np.array(z['samples'])
            assert all(p['pilot_digest']==g.allocation.digest(samples) for p in plans.values())
            return plans,samples
        return build_selections(d,c,system,stream)

def build_selections(d, c, system, stream):
    model = gp.build_model(c, d, stream)
    samples = d.r.j._sample_outcomes(c.oracle, d.production_seed(stream), np.full(30,500))
    pilot_digest = g.allocation.digest(np.asarray(samples))
    (HERE/'pilots'/system).mkdir(parents=True, exist_ok=True)
    g.atomic_npz(HERE/'pilots'/system/f'r{stream}.npz', samples=np.asarray(samples),
                 covariances=model['covariances'])
    screened = g.screen(model['designs'], model['covariances'], model['h'], model['f'], BUDGETS)
    plans = {}
    for budget in BUDGETS:
        chosen, = [p for p in screened[budget]['path'] if len(p['indices']) == 15]
        indices = np.asarray(chosen['indices'], dtype=int)
        assert np.linalg.matrix_rank(np.vstack(model['designs'][indices]), tol=1e-9) == model['pool_rank']
        counts = np.full(30,500,dtype=int)
        counts[indices] += (budget-15000)//15
        assert counts.sum() == budget
        plans[budget] = dict(indices=indices.tolist(), counts=counts.tolist(),
                             fit_counts=counts[indices].tolist(), pool_rank=model['pool_rank'],
                             pilot_shots=15000, unused_pilot_shots=7500,
                             production_shots=budget-15000, fit_shots=budget-7500,
                             pilot_digest=pilot_digest,
                             guard_proxy={k:chosen[k] for k in ('worst','plane','d2','h','f')})
        save(HERE/'plans'/system/f'r{stream}_b{budget}.json', plans[budget])
    assert all(plans[b]['indices'] == plans[BUDGETS[0]]['indices'] for b in BUDGETS)
    print('SELECTION', system, stream, 'rank', model['pool_rank'], 'indices', indices.tolist(), flush=True)
    return plans, samples

def plans_for_arm(guard_plans, arm):
    assert arm in ARMS and arm!='uniform'
    plans={}
    for budget,source in guard_plans.items():
        plan=dict(source)
        guard_indices=list(source['indices'])
        plan['guard_indices']=guard_indices
        plan['selection_role']='guard15'
        if arm.startswith('exclusive_'):
            indices=np.setdiff1d(np.arange(30),guard_indices)
            assert len(indices)==15 and not set(indices).intersection(guard_indices)
            counts=np.full(30,500,dtype=int)
            counts[indices]+=(budget-15000)//15
            plan.update(indices=indices.tolist(),counts=counts.tolist(),
                        fit_counts=counts[indices].tolist(),selection_role='exclusive_guard15')
            plan['source_guard_proxy']=plan.pop('guard_proxy')
        assert sum(plan['counts'])==budget and sum(plan['fit_counts'])==budget-7500
        plans[budget]=plan
    return plans

def worker(system, arm, stream, threads, budget_only=None):
    started = time.time()
    resource.setrlimit(resource.RLIMIT_AS, (20*1024**3,20*1024**3))
    budgets = (budget_only,) if budget_only is not None else BUDGETS
    if all(result_path(system,arm,stream,b).exists() and result_path(system,arm,stream,b).with_suffix('.npz').exists() for b in budgets):
        return
    d,c = context(system)
    print('CONTEXT', system, 'seconds', time.time()-started, flush=True)
    if arm != 'uniform':
        guard_plans,pilots = selections(d,c,system,stream)
        plans=plans_for_arm(guard_plans,arm)
        indices=np.asarray(plans[budgets[0]]['indices'])
        fit_rank=int(np.linalg.matrix_rank(np.vstack([c.blocks[i]@c.lift for i in indices]),tol=1e-9))
        for plan in plans.values():plan['fit_design_rank']=fit_rank
        print('DESIGN',arm,'indices',indices.tolist(),'rank',fit_rank,flush=True)
    else:
        plans = {b:dict(indices=list(range(30)), counts=[b//30]*30, fit_counts=[b//30]*30,
                        pilot_shots=0, unused_pilot_shots=0, production_shots=b, fit_shots=b)
                 for b in BUDGETS}
        pilots = None
    maximum = np.max([plans[b]['counts'] for b in budgets], axis=0)
    data = d.r.j._sample_outcomes(c.oracle,d.production_seed(stream),maximum)
    if pilots is not None:
        for frame in range(30):
            np.testing.assert_array_equal(data[frame][:500],pilots[frame])
    prefix_digest = g.allocation.digest(np.asarray([x[:500] for x in data]))
    raws = {b:selected_raw(d,c,data,np.asarray(plans[b]['counts']),np.asarray(plans[b]['indices']))
            for b in budgets}
    del data
    mate=dict(guard15_equal='guard15_equal_mu0',guard15_equal_mu0='guard15_equal',
              exclusive_guard15_mu2='exclusive_guard15_mu0',exclusive_guard15_mu0='exclusive_guard15_mu2').get(arm)
    for b in budgets:
        other=result_path(system,mate,stream,b).with_suffix('.npz') if mate else None
        if other is not None and other.exists():
            with np.load(other) as z:
                np.testing.assert_array_equal(raws[b].values,z['values'])
                np.testing.assert_array_equal(plans[b]['counts'],z['counts'])
                np.testing.assert_array_equal(plans[b]['indices'],z['indices'])
    origin = raws[budgets[0]].values
    directions = (np.column_stack([raws[b].values-origin for b in budgets[1:]])
                  if len(budgets)>1 else np.ones((len(origin),1)))
    if arm != 'uniform':
        solver = MeritSolver(d,c,raws[budgets[0]],directions,threads=threads)
    else:
        solver = ReusableBandSolver(c,raws[budgets[0]],'spin',band=d.band,
                                    observation_basis=directions,solver_threads=threads)
    for budget in budgets:
        path = result_path(system,arm,stream,budget)
        if path.exists() and path.with_suffix('.npz').exists():
            continue
        raw = raws[budget]
        result = solver.solve(raw.values,1.,MU[arm]) if arm != 'uniform' else solver.solve(raw.values)
        assert result.status == 'optimal'
        stats = solver.stats if arm != 'uniform' else solver.last_stats
        scores = d.r.score(c,result)
        record = dict(system=system, arm=arm, stream=stream, budget=budget,
                      pool='real30', pool_digest=g.allocation.digest(co.real_frames()),
                      seed=int(d.production_seed(stream)), prefix500_digest=prefix_digest,
                      error_basis='spin', lambda_radius=1., mu_ftpbe=MU[arm],
                      status=result.status, solver_stats=stats, seconds=time.time()-started,
                      **plans[budget], **scores)
        path.parent.mkdir(parents=True, exist_ok=True)
        g.atomic_npz(path.with_suffix('.npz'),d2=result.d2,gamma=result.gamma,
                     values=raw.values,counts=np.asarray(plans[budget]['counts']),
                     indices=np.asarray(plans[budget]['indices']))
        save(path,record)
        print('RESULT',system,arm,stream,budget,'H',scores['h_error_meh'],'F',scores['f_error_meh'],
              'peak_GiB',stats['peak_rss_gib'],flush=True)

def summarize():
    rows=[]
    details=[]
    contrasts=[]
    for system in SYSTEMS:
        for budget in BUDGETS:
            records={a:[] for a in ARMS}
            for s in STREAMS:
                paths={a:result_path(system,a,s,budget) for a in ARMS}
                if not all(p.exists() for p in paths.values()):
                    continue
                pair={a:json.loads(p.read_text()) for a,p in paths.items()}
                assert len({r['prefix500_digest'] for r in pair.values()})==1
                for a,b in [('guard15_equal','guard15_equal_mu0'),('exclusive_guard15_mu2','exclusive_guard15_mu0')]:
                    for key in ('counts','indices','pilot_digest'):
                        assert pair[a][key]==pair[b][key]
                    with np.load(paths[a].with_suffix('.npz')) as za,np.load(paths[b].with_suffix('.npz')) as zb:
                        np.testing.assert_array_equal(za['values'],zb['values'])
                guard=set(pair['guard15_equal']['indices']);exclusive=set(pair['exclusive_guard15_mu2']['indices'])
                assert not guard.intersection(exclusive) and guard.union(exclusive)==set(range(30))
                for a,r in pair.items():
                    assert r['status']=='optimal' and r['error_basis']=='spin'
                    counts=np.array(r['counts']); indices=np.array(r['indices'])
                    assert counts.sum()==budget and len(indices)==(30 if a=='uniform' else 15)
                    assert np.all(counts[indices]==(budget//30 if a=='uniform' else 500+(budget-15000)//15))
                    assert r['mu_ftpbe']==MU[a] and r['lambda_radius']==1.
                    if a!='uniform':
                        assert np.all(counts[np.setdiff1d(np.arange(30),indices)]==500)
                        assert r['fit_shots']==budget-7500 and r['pilot_shots']==15000
                    with np.load(paths[a].with_suffix('.npz')) as z:
                        np.testing.assert_array_equal(z['counts'],counts)
                        np.testing.assert_array_equal(z['indices'],indices)
                    records[a].append(r)
                    details.append({k:r[k] for k in ('system','arm','stream','budget','lambda_radius','mu_ftpbe','h_error_meh','f_error_meh','d2_error','minimum_dqg_eigenvalue','equality_residual')})
            if not records[ARMS[0]]:
                continue
            row=dict(system=system,budget=budget,streams=len(records[ARMS[0]]))
            losses={}
            for arm in ARMS:
                h=np.array([r['h_error_meh'] for r in records[arm]])
                f=np.array([r['f_error_meh'] for r in records[arm]])
                losses[arm]=h*h+f*f
                row.update({arm+'_h_rmse':float(np.sqrt(np.mean(h*h))),
                            arm+'_f_rmse':float(np.sqrt(np.mean(f*f))),
                            arm+'_joint_rmse':float(np.sqrt(np.mean(losses[arm]))),
                            arm+'_h_bias':float(h.mean()),arm+'_f_bias':float(f.mean()),
                            arm+'_d2_mean':float(np.mean([r['d2_error'] for r in records[arm]]))})
            delta=losses[ARMS[0]]-losses['uniform']
            row.update(delta_joint_mse=float(delta.mean()),wins=int(np.sum(delta<0)))
            rows.append(row)
            comparisons=[(a,'uniform') for a in ARMS if a!='uniform']+[
                ('guard15_equal','guard15_equal_mu0'),
                ('exclusive_guard15_mu2','exclusive_guard15_mu0'),
                ('guard15_equal','exclusive_guard15_mu2'),
                ('guard15_equal_mu0','exclusive_guard15_mu0')]
            for left,right in comparisons:
                delta=losses[left]-losses[right]
                contrasts.append(dict(system=system,budget=budget,streams=len(delta),left=left,right=right,
                                      delta_joint_mse=float(delta.mean()),
                                      delta_standard_error=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None,
                                      left_wins=int(np.sum(delta<0))))
    save(HERE/'summary.json',rows)
    save(HERE/'ablation_contrasts.json',contrasts)
    for name,table in [('summary',rows),('per_stream',details),('ablation_contrasts',contrasts)]:
        if table:
            with (HERE/f'{name}.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print(json.dumps(rows,indent=2),flush=True)
    if len(details)==len(SYSTEMS)*len(STREAMS)*len(BUDGETS)*len(ARMS):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
        for i,system in enumerate(SYSTEMS):
            rr=[r for r in rows if r['system']==system]
            for j,(metric,label) in enumerate([('h_rmse','H RMSE (mHa)'),('f_rmse','F RMSE (mHa)'),('joint_rmse','Joint RMSE (mHa)')]):
                for arm in ARMS:
                    axes[i,j].plot([r['budget']/1000 for r in rr],[r[arm+'_'+metric] for r in rr],'-o',label=arm)
                axes[i,j].set(xlabel='Total shots (K)',ylabel=label,title=system)
                axes[i,j].grid(alpha=.25)
                axes[i,j].legend()
        fig.savefig(HERE/'rmse.png',dpi=180)
        fig.savefig(HERE/'rmse.pdf')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','worker','summarize'])
    p.add_argument('--system',choices=SYSTEMS);p.add_argument('--arm',choices=ARMS)
    p.add_argument('--stream',type=int);p.add_argument('--threads',type=int,default=3)
    p.add_argument('--budget',type=int,choices=BUDGETS)
    a=p.parse_args()
    if a.command=='init': initialize()
    elif a.command=='summarize': summarize()
    else: worker(a.system,a.arm,a.stream,a.threads,a.budget)
