"""Matched-cost finite-shot validation; exact data are simulator/scoring only.

Registered systems (see ``SYSTEMS``): N2 (6,6), F2 (8,6), C2 (8,8) and the
open-shell O2 triplet (12,8)."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLCONFIGDIR'] = '/tmp/physical_frame_design_mpl'
import sys, json, time, argparse, hashlib, resource
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.linalg import svd, eigh, qr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GUARD = ROOT / 'mcpdft_measurement_revision/guarded_mcpdft_shadow_protocol'
for path in (GUARD/'tools', GUARD/'code', GUARD/'code/vendor', GUARD/'evidence_revision',
             ROOT/'outputs/mcpdft_cancellation_universality', ROOT/'outputs/mcpdft_physical_scheme'):
    sys.path.insert(0, str(path))
import common as pc
from physical_geometry import solver_equalities
from run_evidence import base_design
from run_n2_random_adaptive_allocation import _blind_shadows
import run_n2_random_pilot_joint_design as j
import cancellation_probe as cp
sys.path.insert(0, str(HERE))
from design import Geometry, sector_covariance, greedy, diagnostics, inverse, identifiable_covariance, refine_frame, blue_factors
from open_shell_reference import build_rohf_cas_reference

POOLS = ((20260716, 271828, 314159), (20260929, 161803, 141421), (20251001, 173205, 223606))
METHODS = ('random30', 'random8', 'dopt8', 'physical8', 'generated8', 'hamiltonian8', 'augmented8')

# Per-system geometry and active space. `kind='n2'` reuses the frozen N2
# selection/reference builders; `kind='canonical'` is the closed-shell CASCI
# builder from the cancellation probe; `kind='open_shell'` uses the ROHF
# builder (n_alpha != n_beta).  `bond_length` is in Angstrom, `spin` follows
# the PySCF convention 2S.
SYSTEMS = {
    'n2': dict(kind='n2', bond_length=1.10, basis='cc-pvdz',
               active_electrons=6, active_orbitals=6),
    'f2': dict(kind='canonical', name='f2_r1.41', basis='cc-pvdz',
               atom='F 0 0 -0.706; F 0 0 0.706', bond_length=1.412,
               active_electrons=8, active_orbitals=6, ncore=5),
    'c2': dict(kind='canonical', name='c2_r1.40', basis='cc-pvdz',
               atom='C 0 0 -0.70; C 0 0 0.70', bond_length=1.40,
               active_electrons=8, active_orbitals=8, ncore=2),
    'o2': dict(kind='open_shell', name='o2_r1.209', basis='cc-pvdz',
               atom='O 0 0 -0.6045; O 0 0 0.6045', bond_length=1.209,
               active_electrons=12, active_orbitals=8, ncore=2, spin=2),
}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def build_system(system):
    """Return (selection, exact) references for a registered system."""
    if system not in SYSTEMS:
        raise KeyError(f'Unknown system {system!r}; registered: {sorted(SYSTEMS)}')
    cfg = SYSTEMS[system]
    if cfg['kind'] == 'n2':
        selection = j.build_n2_selection_reference(
            cfg['bond_length'], basis=cfg['basis'],
            active_electrons=cfg['active_electrons'],
            active_orbitals=cfg['active_orbitals'])
        exact = j.build_n2_reference(
            cfg['bond_length'], basis=cfg['basis'],
            active_electrons=cfg['active_electrons'],
            active_orbitals=cfg['active_orbitals'])
        return selection, exact
    spec = dict(name=cfg['name'], atom=cfg['atom'], basis=cfg['basis'],
                active_electrons=cfg['active_electrons'],
                active_orbitals=cfg['active_orbitals'], ncore=cfg['ncore'],
                bond_length=cfg['bond_length'], frame_count=30,
                builder=cfg['kind'])
    if cfg['kind'] == 'open_shell':
        spec['spin'] = cfg['spin']
        exact, _, _ = build_rohf_cas_reference(spec)
    else:
        exact, _, _ = cp.build_canonical_cas_reference(spec)
    return j.LeakGuardReference(exact), exact


def context(system):
    args = j._arguments([])
    args.solver_threads = 1
    selection, exact = build_system(system)
    objective = j.FtPBEEnergyObjective(selection, grid_level=1)
    n = int(selection.n_spatial_orbitals)
    dummy = (np.eye(n),)
    _, _, _, rows, cols = j._build_random_design(selection, dummy)
    e, offset = solver_equalities(selection, rows, cols)
    scale = np.sqrt(np.where(rows == cols, 1., 2.))
    _, singular, v = svd(e / scale, full_matrices=True)
    rank = np.sum(singular > max(singular[0] * 1e-9, 1e-10))
    z = v[rank:].T
    geo = dict(E=e, offset=offset, Z=z, scale=scale, rows=rows, cols=cols)
    maps, meta, gamma = pc.feature_maps(selection, objective, geo)
    g = maps['density']
    r, _ = pc.rowspace(maps['density_contact'] - maps['density_contact'] @ g.T @ g)
    lift = z / scale[:, None]
    _, white = sector_covariance(n, selection.n_alpha, selection.n_beta, selection.pairs)
    blind = Geometry(n, np.asarray(selection.pairs), rows, cols, lift, white, g, r)
    return SimpleNamespace(args=args, sel=selection, exact=exact, objective=objective,
                           geo=geo, maps=maps, gamma=gamma, lift=lift, blind=blind)


def design_stage(system, pool, iterations=50):
    start = time.time()
    c = context(system)
    rotations, _ = j._random_mixed_frames(c.sel.n_spatial_orbitals, 30, .5, *POOLS[pool])
    rotations = np.asarray(rotations)
    dst = HERE/'designs'/system/f'pool{pool}'
    dst.mkdir(parents=True, exist_ok=True)
    # Validate the independent fast block implementation against the old backend.
    vectors, _, blocks, rows, cols = j._build_random_design(c.sel, rotations)
    for u, block in zip(rotations, blocks):
        np.testing.assert_allclose(c.blind.block(u), block, atol=1e-13)
    for name in METHODS:
        if (dst/f'{name}.npz').exists():
            continue
        began = time.time()
        if name.startswith('random'):
            us = rotations[:int(name.replace('random', ''))]
            diag = diagnostics(c.blind, us)
        elif name in ('hamiltonian8', 'augmented8'):
            h = j._hamiltonian_gradient(c.sel, rows, cols) @ c.lift
            h /= np.linalg.norm(h)
            if name == 'hamiltonian8':
                us, diag = greedy(c.blind, rotations, 8, weight=np.outer(h,h))
            else:
                # H and F are only proposal directions; acceptance/selection is
                # controlled by the same physical-subspace risk as other arms.
                baseline = j.solve_dqg_sdp(j.LeakGuardReference(c.sel), solver=c.args.solver,
                    tolerance=c.args.solver_tolerance, solver_threads=1,
                    positivity_conditions='DQG', symmetry_blocked_psd=True)
                f = j._raw_ftpbe_gradient(c.objective, c.sel, baseline.d2, baseline.gamma, rows, cols) @ c.lift
                f /= np.linalg.norm(f)
                infos = [c.blind.information(u) for u in rotations]
                ridge = .001 * np.trace(np.mean(infos, axis=0)) / c.lift.shape[1]
                precision = np.eye(c.lift.shape[1]) * ridge
                extras, proposal_details = [], []
                for label, target in [('H',h),('F',f)]:
                    q = np.outer(target,target)
                    scores = [float(target @ inverse(precision+i) @ target) for i in infos]
                    for k in np.argsort(scores)[:2]:
                        u, record = refine_frame(c.blind, precision, rotations[k], q, iterations)
                        extras.append(u)
                        proposal_details.append(dict(target=label, index=int(k), **record))
                augmented = np.concatenate((rotations,extras))
                us, diag = greedy(c.blind, augmented, 8, refine=True, maxiter=iterations,reference_ridge=ridge)
                diag.update(proposals=proposal_details, augmented_selected=int(sum(i>=30 for i in diag['indices'])))
        else:
            us, diag = greedy(c.blind, rotations, 8, criterion='dopt' if name == 'dopt8' else 'physical',
                              refine=(name == 'generated8'), maxiter=iterations)
        assert diag.get('target_null_overlap', 0.) < 1e-7, diag
        diag.update(system=system, pool=pool, method=name, seconds=time.time()-began,
                    frame_digest=hashlib.sha256(us.tobytes()).hexdigest(),
                    exact_state_used=False, energy_gradient_used=(name in ('hamiltonian8','augmented8')), candidate_shots=0)
        np.savez_compressed(dst/f'{name}.npz', rotations=us, rows=rows, cols=cols, lift=c.lift,
                            density=c.blind.density, contact=c.blind.contact)
        save(dst/f'{name}.json', diag)
        print(system, pool, name, diag, flush=True)
    print('DESIGN_DONE', system, pool, time.time()-start, flush=True)


def protected_solve(c, baseline, shadows, theta0, covariance, zhat):
    g, r = c.blind.density, c.blind.contact
    reg = np.linalg.solve(g @ covariance @ g.T, g @ covariance @ r.T).T
    conditional = r - reg @ g
    factors = []
    for w in (g, conditional):
        eig, u = eigh(w @ covariance @ w.T)
        assert eig[0] > 0
        factors.append((u / np.sqrt(eig)).T @ w * np.sqrt(.01 / len(w)))
    factor = np.vstack(factors)
    l = factor @ c.geo['Z'].T * c.geo['scale'][None, :]
    target = l @ theta0 + factor @ zhat
    affine = pc.affine_map(l, c.geo, len(c.sel.pairs), -target)
    return j.solve_dqg_sdp(j.LeakGuardReference(c.sel), shadow_data=_blind_shadows(shadows),
        affine_objective=affine, selection_objective='affine_least_squares',
        additional_d2_objective=c.sel.two_body, additional_gamma_objective=c.sel.one_body,
        shadow_error_weight=c.args.nuclear_weight, solver=c.args.solver,
        tolerance=c.args.solver_tolerance, max_iterations=c.args.max_iterations, solver_threads=1,
        positivity_conditions='DQG', symmetry_blocked_psd=True,
        initial_d2=baseline.d2, initial_gamma=baseline.gamma, initial_corrected_d2=baseline.d2)


def literal_basis(name, rotations, vectors, blocks, covs):
    """Old literal QR measurement constraints, allowing incomplete nuisance rank."""
    stacked = np.vstack(blocks)
    scale = np.sqrt(np.concatenate([np.diag(v) for v in covs]))
    _, triangular, pivots = qr((stacked / scale[:, None]).T, mode='economic', pivoting=True)
    tol = abs(triangular[0, 0]) * max(stacked.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(abs(np.diag(triangular)) > tol))
    chosen = np.sort(pivots[:rank])
    nrow = len(blocks[0])
    local = tuple(chosen[(chosen//nrow)==k] % nrow for k in range(len(rotations)))
    return j.ConstraintBasis(name=name, rotations=tuple(rotations), pair_vectors=vectors,
        design=j.quadratic_design(vectors), global_rows=chosen, rows_by_frame=local,
        blocks=tuple(a[r] for a,r in zip(blocks,local)),
        covariances=tuple(v[np.ix_(r,r)] for v,r in zip(covs,local)),
        raw_condition=float(np.linalg.cond(stacked[chosen])), fisher_condition=np.nan,
        rank=stacked.shape[1])


def pilot_design_stage(system, pool, seeds):
    c = context(system)
    rotations, _ = j._random_mixed_frames(c.sel.n_spatial_orbitals, 30, .5, *POOLS[pool])
    vectors, _, blocks, rows, cols = j._build_random_design(c.sel, rotations)
    oracle = j.AcquisitionOracle(c.exact, rotations, vectors, 1, 0)
    dst = HERE/'designs'/system/f'pool{pool}'
    dst.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        if (dst/f'pilot8_seed{seed}.npz').exists():
            continue
        pilot_seed = 202609290000 + 104729*pool + 1543*seed
        outcomes = j._sample_outcomes(oracle, pilot_seed, np.full(30,500))
        covs = tuple(j._regularized_single_covariance(x.astype(float)) for x in outcomes)
        infos = [(a@c.lift).T@np.linalg.solve(v,a@c.lift) for a,v in zip(blocks,covs)]
        us, diag = greedy(c.blind, rotations, 8, information=infos)
        indices = np.asarray(diag['indices'])
        diag.update(system=system,pool=pool,seed=seed,pilot_seed=pilot_seed,candidate_shots=15000,
                    energy_gradient_used=False, exact_state_used=False,
                    pilot_covariance_risk=float(8*np.trace(c.blind.weight@np.linalg.pinv(sum(infos[i] for i in indices)))))
        name = f'pilot8_seed{seed}'
        np.savez_compressed(dst/f'{name}.npz',rotations=us,rows=rows,cols=cols,lift=c.lift,
                            density=c.blind.density,contact=c.blind.contact,
                            pilot_covariances=np.asarray(covs)[indices],pilot_counts=np.full(30,500))
        save(dst/f'{name}.json',diag)
        print('PILOT_DESIGN',system,pool,seed,indices,flush=True)


def run_stage(system, pool, seeds, budgets, methods, arms):
    c = context(system)
    baseline = j.solve_dqg_sdp(j.LeakGuardReference(c.sel), solver=c.args.solver,
        tolerance=c.args.solver_tolerance, max_iterations=c.args.max_iterations, solver_threads=1,
        positivity_conditions='DQG', symmetry_blocked_psd=True)
    assert baseline.status == 'optimal'
    rows, cols = c.geo['rows'], c.geo['cols']
    theta0 = baseline.d2[rows, cols]
    href, fref = j._energy_values(c.sel, c.objective, c.exact.exact_d2, c.exact.exact_gamma)
    out = HERE/'results'/system/f'pool{pool}'
    out.mkdir(parents=True, exist_ok=True)
    for method in methods:
        archive = np.load(HERE/'designs'/system/f'pool{pool}'/f'{method}.npz')
        np.testing.assert_allclose(archive['lift'], c.lift, atol=1e-11)
        rotations = archive['rotations']
        vectors, _, blocks, _, _ = j._build_random_design(c.sel, rotations)
        oracle = j.AcquisitionOracle(c.exact, rotations, vectors, 1, 2026092900)
        nframes = len(rotations)
        for seed in seeds:
            # Separate independent pilots for EVERY trial. No shared-pilot pseudo-replication.
            pilot_seed = 202609290000 + 104729 * pool + 1543 * seed
            production_seed = pilot_seed + 100000000
            if 'pilot_covariances' in archive:
                assert method == f'pilot8_seed{seed}'
                pilot_counts = archive['pilot_counts']
                covs = tuple(archive['pilot_covariances'])
            else:
                pilot_counts = np.full(nframes, 15000 // nframes)
                pilot_counts[:15000 % nframes] += 1
                pilot = j._sample_outcomes(oracle, pilot_seed, pilot_counts)
                covs = tuple(j._regularized_single_covariance(x.astype(float)) for x in pilot)
            basis = literal_basis(method, rotations, vectors, blocks, covs)
            maximum = np.full(nframes, (max(budgets)-15000)//nframes)
            maximum[:(max(budgets)-15000) % nframes] += 1
            outcomes = j._sample_outcomes(oracle, production_seed, maximum)
            for budget in budgets:
                counts = np.full(nframes, (budget-15000)//nframes)
                counts[:(budget-15000) % nframes] += 1
                shadows = j._finite_shadows(basis, outcomes, counts)
                # All bitstring pair statistics contribute to BLUE. The baseline
                # nuclear-norm constraints retain their established literal QR basis.
                info = np.zeros((c.lift.shape[1],)*2)
                rhs = np.zeros(len(info))
                y = []
                for k, (a, v) in enumerate(zip(blocks, covs)):
                    x = a @ c.lift
                    back = counts[k] * np.linalg.solve(v, x).T
                    info += back @ x
                    mean = outcomes[k][:counts[k]].mean(0)
                    rhs += back @ (mean - a @ theta0)
                    y.append(mean)
                covariance,influences,information_rank,null_overlap,blue_condition=blue_factors(
                    blocks,covs,counts,c.lift,np.vstack((c.blind.density,c.blind.contact)))
                zhat=sum(back@(mean-a@theta0) for back,mean,a in zip(influences,y,blocks))
                for arm in arms:
                    file = out/f'{method}_seed{seed}_b{budget}_{arm}.json'
                    if file.exists():
                        continue
                    began = time.time()
                    if arm == 'original':
                        result = j._solve_eq11(c.args, c.sel, shadows, baseline.d2, baseline.gamma, baseline.d2)
                    else:
                        result = protected_solve(c, baseline, shadows, theta0, covariance, zhat)
                    assert result.status == 'optimal', result.status
                    metrics = j._comparison_metrics(result.d2, result.gamma, c.exact.exact_d2, href, fref, c.sel, c.objective)
                    delta = c.geo['Z'].T @ (c.geo['scale'] * (result.d2[rows, cols] - c.exact.exact_d2[rows, cols]))
                    eq = float(np.linalg.norm(c.geo['E'] @ result.d2[rows, cols] + c.geo['offset']))
                    rec = dict(system=system, pool=pool, seed=seed, method=method.split('_seed')[0],
                        design_id=method, arm=arm, budget=budget,
                        pilot_shots=int(pilot_counts.sum()), production_shots=int(counts.sum()),
                        frames=nframes, pilot_seed=pilot_seed, production_seed=production_seed,
                        seconds=time.time()-began, status=result.status, equality_residual=eq,
                        information_rank=information_rank, target_null_overlap=null_overlap,
                        blue_solver='whitened_svd',blue_condition=blue_condition,
                        density_error=float(np.linalg.norm(c.blind.density @ delta)),
                        contact_error=float(np.linalg.norm(c.blind.contact @ delta)),
                        gamma_error=float(np.linalg.norm(result.gamma-c.exact.exact_gamma)),
                        modeled_density_variance=float(np.trace(c.blind.density @ covariance @ c.blind.density.T)),
                        modeled_contact_variance=float(np.trace(c.blind.contact @ covariance @ c.blind.contact.T)), **metrics)
                    np.savez_compressed(file.with_suffix('.npz'), d2=result.d2, gamma=result.gamma,
                                        counts=counts, pilot_counts=pilot_counts, C=covariance, zhat=zhat,lift=c.lift,
                                        full_values=np.asarray(y), qr_values=shadows.values)
                    save(file, rec)
                    print('FIT', system, pool, seed, budget, method, arm,
                          'F',round(metrics['signed_ftpbe_error_meh'],4), 'D',round(metrics['d2_error'],5),
                          'seconds',round(rec['seconds'],2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['design', 'pilot', 'run'])
    parser.add_argument('system', choices=sorted(SYSTEMS))
    parser.add_argument('--pool', type=int, default=0)
    parser.add_argument('--seeds', default='0,1,2,3')
    parser.add_argument('--budgets', default='150000,300000')
    parser.add_argument('--methods', default=','.join(METHODS))
    parser.add_argument('--arms', default='original,protected')
    parser.add_argument('--iterations', type=int, default=50)
    cli = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    if cli.stage == 'design':
        design_stage(cli.system, cli.pool, cli.iterations)
    elif cli.stage == 'pilot':
        pilot_design_stage(cli.system,cli.pool,list(map(int,cli.seeds.split(','))))
    else:
        run_stage(cli.system, cli.pool, list(map(int,cli.seeds.split(','))),
                  list(map(int,cli.budgets.split(','))), cli.methods.split(','), cli.arms.split(','))
