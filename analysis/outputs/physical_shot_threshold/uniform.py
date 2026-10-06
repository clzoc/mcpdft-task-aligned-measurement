"""The manuscript Uniform30 protocol: structural QR, no pilot, full budget."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,time,argparse,resource
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'physical_frame_design'))
import numpy as np
from experiment import context,j,save,cp,POOLS
from constrained_shadow import dqg_matrices,contract_one_rdm

PAPER=HERE.parents[1]/'mcpdft_measurement_revision/guarded_mcpdft_shadow_protocol/sweeps/n2_random_pilot_joint_design'


def uniform_frames(c,system,pool):
    """Article frame family for the baseline.

    N2/F2 use the frozen random-mixed pool.  C2/O2 use the published C2 family
    (``_c2_random_frames``): 29 same-spin mixed frames plus one frame with
    independent alpha/beta unitaries, because the symmetry-reduced variable
    space of these irrep patterns has a two-dimensional structural null space
    that same-spin frames cannot cover.
    """

    if system in ('c2','o2'):
        from run_c2_random_pilot_joint_design import _c2_random_frames
        rotations,_,_=_c2_random_frames(c.sel.n_spatial_orbitals,30,.5,*POOLS[pool])
        rotations=tuple(np.asarray(rotation) for rotation in rotations)
        vectors,_,blocks,rows,cols=cp.build_design(c.sel,rotations)
        return rotations,vectors,blocks,rows,cols
    archive=np.load(HERE.parent/'physical_frame_design/designs'/system/f'pool{pool}/random30.npz')
    rotations=archive['rotations']
    vectors,_,blocks,rows,cols=j._build_random_design(c.sel,rotations)
    return rotations,vectors,blocks,rows,cols


def main(system,pools,seeds,budgets,cohort,replay=False):
    c=context(system)
    base=j.solve_dqg_sdp(j.LeakGuardReference(c.sel),solver=c.args.solver,
        tolerance=c.args.solver_tolerance,max_iterations=c.args.max_iterations,solver_threads=1,
        positivity_conditions='DQG',symmetry_blocked_psd=True)
    assert base.status=='optimal'
    href,fref=j._energy_values(c.sel,c.objective,c.exact.exact_d2,c.exact.exact_gamma)
    checks=[]
    for pool in pools:
        rotations,vectors,blocks,rows,cols=uniform_frames(c,system,pool)
        design=j._constraint_basis_from_covariances('uniform30',tuple(range(len(rotations))),rotations,vectors,blocks,
            tuple(np.eye(len(c.sel.pairs)) for _ in rotations),len(c.sel.pairs),len(rows))
        basis=design.basis
        paths={b:j._equal_counts(30,basis.active_frames,b,c.args.allocation_chunk) for b in budgets}
        out=HERE/('paper_replay' if replay else 'results')/system/f'pool{pool}'
        out.mkdir(parents=True,exist_ok=True)
        design_file=out/'uniform30_design.npz'
        if design_file.exists():
            saved=np.load(design_file)
            assert np.array_equal(saved['global_rows'],basis.global_rows)
            assert np.array_equal(saved['active_frames'],basis.active_frames)
            assert np.array_equal(saved['rotations'],rotations)
        else:
            np.savez_compressed(design_file,rotations=rotations,global_rows=basis.global_rows,
                rows=rows,cols=cols,active_frames=basis.active_frames)
        print('DESIGN',system,pool,'active',len(basis.active_frames),'rows',len(basis.global_rows),flush=True)
        oracle=j.AcquisitionOracle(c.exact,rotations,vectors,1,0)
        for seed in seeds:
            prod_seed=seed+200003 if replay else 202609290000+104729*pool+1543*seed+100000000
            outcomes=j._sample_outcomes(oracle,prod_seed,paths[max(budgets)])
            warm_d,warm_g,warm_corrected=base.d2,base.gamma,base.d2
            for budget in sorted(budgets):
                file=out/f'uniform30_seed{seed}_b{budget}_original.json'
                if file.exists():
                    a=np.load(file.with_suffix('.npz'))
                    warm_d,warm_g,warm_corrected=a['d2'],a['gamma'],a['corrected_d2']
                    continue
                counts=paths[budget]
                shadows=j._finite_shadows(basis,outcomes,counts)
                start=time.time()
                result=j._solve_eq11(c.args,c.sel,shadows,warm_d,warm_g,warm_corrected)
                assert result.status=='optimal'
                warm_d,warm_g,warm_corrected=result.d2,result.gamma,result.corrected_d2
                metrics=j._comparison_metrics(result.d2,result.gamma,c.exact.exact_d2,href,fref,c.sel,c.objective)
                minimum=min(float(np.linalg.eigvalsh(m)[0]) for m in dqg_matrices(result.d2,result.gamma,c.sel.pairs))
                contraction=float(np.linalg.norm(result.gamma-contract_one_rdm(result.d2,c.sel.n_spin_orbitals,c.sel.n_electrons,c.sel.pairs)))
                eq=float(np.linalg.norm(c.geo['E']@result.d2[rows,cols]+c.geo['offset']))
                assert minimum>-2e-6 and contraction<1e-6 and eq<1e-6
                record=dict(system=system,pool=pool,seed=seed,method='uniform30',arm='original',
                    cohort=cohort,budget=budget,pilot_shots=0,production_shots=int(counts.sum()),frames=30,
                    active_production_frames=len(basis.active_frames),production_seed=prod_seed,
                    covariance_method='identity_for_structural_QR_no_pilot',kappa=0,
                    status=result.status,reused=False,seconds=time.time()-start,
                    minimum_dqg_eigenvalue=minimum,contraction_error=contraction,equality_residual=eq,
                    gamma_error=float(np.linalg.norm(result.gamma-c.exact.exact_gamma)),**metrics)
                if replay:
                    oldfile=PAPER/f'shot_seed_{seed}/checkpoints/uniform30_{budget}.json'
                    old=json.loads(oldfile.read_text())
                    assert np.array_equal(counts,np.fromstring(old['shot_counts'],sep=',',dtype=int))
                    diffs=dict(budget=budget,active_frames_match=True,
                        d2_error_difference=metrics['d2_error']-old['d2_error_to_exact'],
                        h_error_difference_meh=metrics['hamiltonian_error_meh']-old['hamiltonian_error_to_exact_meh'],
                        f_error_difference_meh=metrics['ftpbe_error_meh']-old['ftpbe_error_to_exact_meh'])
                    checks.append(diffs)
                    record['paper_replay_comparison']=diffs
                full=np.asarray([x[:n].mean(0) if n else np.zeros(len(c.sel.pairs)) for x,n in zip(outcomes,counts)])
                np.savez_compressed(file.with_suffix('.npz'),d2=result.d2,gamma=result.gamma,
                    corrected_d2=result.corrected_d2,counts=counts,pilot_counts=np.zeros(30,dtype=int),
                    full_values=full,qr_values=shadows.values,lift=c.lift)
                save(file,record)
                print('UNIFORM',system,pool,seed,budget,'D',round(metrics['d2_error'],6),
                    'H',round(metrics['signed_hamiltonian_error_meh'],4),
                    'F',round(metrics['signed_ftpbe_error_meh'],4),'seconds',round(record['seconds'],2),flush=True)
    if replay:
        save(HERE/'paper_replay_checks.json',checks)

if __name__=='__main__':
    from experiment import SYSTEMS
    p=argparse.ArgumentParser();p.add_argument('system',choices=sorted(SYSTEMS))
    p.add_argument('--pools',default='0,1,2');p.add_argument('--seeds',default='0,1')
    p.add_argument('--budgets',default='75000,150000,300000');p.add_argument('--cohort',default='exploration')
    p.add_argument('--replay',action='store_true');a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    main(a.system,list(map(int,a.pools.split(','))),list(map(int,a.seeds.split(','))),
         list(map(int,a.budgets.split(','))),a.cohort,a.replay)
