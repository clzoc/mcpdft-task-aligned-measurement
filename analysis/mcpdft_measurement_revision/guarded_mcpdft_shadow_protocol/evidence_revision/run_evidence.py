#!/usr/bin/env python3
"""Paired, resumable evidence experiments. See EXPERIMENT_PLAN.md."""
from __future__ import annotations

import argparse
import fcntl
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/evidence-revision-mpl')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))
import numpy as np
from scipy import sparse
from scipy.linalg import cholesky, solve_triangular, qr
import run_n2_random_pilot_joint_design as j
import run_c2_random_pilot_joint_design as c
from run_n2_ftpbe_shot_cost_oracle import _exact_single_covariance
from constrained_shadow import AffineRDMObjective, dqg_matrices, quadratic_design
from fast_allocation import allocation_paths, greedy_frame_order
j._allocation_paths = allocation_paths


class LazyQuadraticRows:
    """Materialize only the literal rows requested by the final reconstruction.

    Prefix screening needs blocks/covariances, not the full quadratic CSR map.
    Keeping all prefix CSR maps alive used several GB for 100-frame pools.
    The requested rows still use the unchanged vendor quadratic_design routine.
    """
    def __init__(self, pair_vectors):
        self.pair_vectors = np.asarray(pair_vectors)

    @property
    def shape(self):
        return (len(self.pair_vectors), self.pair_vectors.shape[1] ** 2)

    def __getitem__(self, rows):
        return quadratic_design(self.pair_vectors[rows])


j.quadratic_design = LazyQuadraticRows


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def save(path, obj):
    j._atomic_json(path, obj)


def equal_design(design, budget, pilot):
    return replace(design, paths={budget: j._equal_counts(len(design.pool_indices),
                   design.basis.active_frames, budget, 500)}, pilot_shots=pilot)


def nested_frames(selection, system, pool, width):
    """The first 30 are fixed; extensions never change earlier frame IDs."""
    offset = pool * 1009
    factory = c._c2_random_frames if system == 'c2' else j._random_mixed_frames
    base = factory(selection.n_spatial_orbitals, 30, .5,
                   20260716 + offset, 271828 + offset, 314159 + offset)
    rotations, flags = list(base[0]), list(base[1])
    if width > 30:
        # Always draw the full extension so 50 is exactly a prefix of 100.
        extra, extra_flags = j._random_mixed_frames(selection.n_spatial_orbitals,
                      70, .5, 920001 + offset, 920003 + offset, 920009 + offset)
        # Interleave real and complex to retain balanced nested pools.
        real = [x for x, f in zip(extra, extra_flags) if not f]
        complex_ = [x for x, f in zip(extra, extra_flags) if f]
        for a, b in zip(real, complex_):
            for rotation, flag in ((a, False), (b, True)):
                rotations.append(np.stack((rotation, rotation)) if system == 'c2' else rotation)
                flags.append(flag)
    return tuple(rotations[:width]), np.asarray(flags[:width])


def gls_factor(design, outcomes, counts, full_blocks, full_rows, full_cols,
               covariances, n_pairs, all_rows):
    """Lossless QR of whitened [A,-y]; retain ALL coefficient columns."""
    pieces = []
    for local, pool in enumerate(design.pool_indices):
        count = int(counts[local])
        if count == 0:
            continue
        rows = np.arange(full_blocks[pool].shape[0]) if all_rows else design.basis.rows_by_frame[local]
        if not len(rows):
            continue
        mean = np.mean(outcomes[pool][:count], axis=0)[rows]
        cov = covariances[pool][np.ix_(rows, rows)]
        chol = cholesky(cov, lower=True)
        augmented = np.column_stack((full_blocks[pool][rows], -mean))
        pieces.append(np.sqrt(count) * solve_triangular(chol, augmented, lower=True))
    stacked = np.vstack(pieces)
    # Scaling the whole objective improves numerical conditioning without
    # altering its minimizer. QR compression discards only a zero residual block.
    normalization = np.sqrt(sum(int(x) for x in counts))
    factor = qr(stacked / normalization, mode='r')[0][:min(stacked.shape)]
    nr, nc = factor[:, :-1].shape
    indices = full_rows * n_pairs + full_cols
    mapping = sparse.coo_matrix((factor[:, :-1].ravel(),
          (np.repeat(np.arange(nr), nc), np.tile(indices, nr))),
          shape=(nr, n_pairs*n_pairs)).tocsr()
    return AffineRDMObjective(d2_map=mapping, offset=factor[:, -1],
                              name='full_covariance_gls'), stacked, factor


def fit_gls(args, selection, design, outcomes, counts, full_blocks, full_rows,
            full_cols, covariances, pilot, all_rows):
    objective, _, _ = gls_factor(design, outcomes, counts, full_blocks,
             full_rows, full_cols, covariances, len(selection.pairs), all_rows)
    return j.solve_dqg_sdp(j.LeakGuardReference(selection),
             affine_objective=objective, selection_objective='affine_least_squares',
             solver=args.solver, tolerance=args.solver_tolerance,
             max_iterations=args.max_iterations, solver_threads=args.solver_threads,
             positivity_conditions='DQG', symmetry_blocked_psd=True,
             initial_d2=pilot.d2, initial_gamma=pilot.gamma)


def base_design(name, indices, rotations, vectors, blocks, covs, dimension, pairs):
    return j._constraint_basis_from_covariances(name, indices, rotations, vectors,
                                                blocks, covs, pairs, dimension)


def restore_policies(path, total, rotations, vectors, blocks, covs, oracle):
    """Resume the already frozen experiment, without re-selecting its policies."""
    data=json.loads(path.read_text())
    designs={}
    exact_covs=None
    for name,record in data['policies'].items():
        covariance=covs
        if name=='uniform_no_pilot':
            covariance=tuple(np.eye(blocks[0].shape[0]) for _ in rotations)
        elif name in ('oracle_covariance','oracle_both'):
            if exact_covs is None:
                exact_covs=tuple(_exact_single_covariance(oracle._frame_probabilities(i),
                    oracle.indicators.astype(float))[1] for i in range(len(rotations)))
            covariance=exact_covs
        design=base_design(name,record['indices'],rotations,vectors,blocks,covariance,
                           blocks[0].shape[1],blocks[0].shape[0])
        counts=np.asarray(record['counts'],dtype=int)
        pilot=int(record['pilot_shots'])
        if int(counts.sum())+pilot!=total:raise ValueError('Frozen budget differs')
        np.testing.assert_array_equal(j._pool_counts(design,counts,len(rotations)),record['pool_counts'])
        designs[name]=replace(design,paths={total-pilot:counts},pilot_shots=pilot)
    return designs,data['details']


def choose(args, rotations, vectors, blocks, covs, targets, lineality, pilot_total):
    order, diagnostics = greedy_frame_order(args, blocks, covs, targets, lineality)
    design, sizes, _ = j._screen_joint_prefixes(args, order, rotations, vectors,
        blocks, covs, blocks[0].shape[0], blocks[0].shape[1],
        (args.budgets[-1]-pilot_total,), targets, lineality, pilot_total)
    return design, {'ordering': list(order), 'ordering_diagnostics': diagnostics,
                    'prefix_screen': sizes}


def geometry_stability(selection, pilot, rows, cols, out):
    ref = SimpleNamespace(**vars(selection), n_electrons=selection.n_electrons,
                          exact_d2=pilot.d2, exact_gamma=pilot.gamma)
    records, bases = [], {}
    for tol in (1e-10, 1e-8, 1e-6):
        L, diag = j._dqg_lineality_basis(ref, rows, cols, active_tolerance=tol)
        bases[tol] = L
        trace = (rows == cols).astype(float)
        records.append({'threshold': tol, **diag,
                        'trace_constraint_residual': float(np.linalg.norm(trace @ L))})
    P = bases[1e-8] @ bases[1e-8].T
    for row in records:
        L = bases[row['threshold']]
        row['projector_distance_to_1e8'] = float(np.linalg.norm(L@L.T-P))
    save(out / 'geometry_thresholds.json', records)
    j._atomic_npz(out / 'geometry.npz', lineality=bases[1e-8], pilot_d2=pilot.d2,
                   pilot_gamma=pilot.gamma)
    return bases[1e-8]


def run_seed(args, cli, seed, selection, exact, objective, rotations, vectors,
             blocks, rows, cols, full_blocks, full_rows, full_cols, baseline):
    out = cli.output / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        print(f'{out.name}: complete', flush=True)
        return
    total, pairs = cli.budget, len(selection.pairs)
    pilot_total = len(rotations)*500
    production = total-pilot_total
    oracle = j.AcquisitionOracle(exact, rotations, vectors, 1, seed)
    pilot_outcomes = j._sample_outcomes(oracle, seed+100003,
                                      np.full(len(rotations), 500))
    covs = tuple(j._regularized_single_covariance(x.astype(float)) for x in pilot_outcomes)
    full = base_design('pilot_uniform', range(len(rotations)), rotations, vectors,
                      blocks, covs, len(rows), pairs)
    pp = out / 'pilot.npz'
    cached_pilot = HERE/'results'/f'{cli.system}_main_pool0_m30'/f'seed_{seed}'/'pilot.npz'
    if cli.suite=='physical' and not pp.exists() and cached_pilot.exists():
        with np.load(cached_pilot) as a:
            j._atomic_npz(pp,d2=a['d2'],gamma=a['gamma'])
    if pp.exists():
        with np.load(pp) as a:
            pilot = SimpleNamespace(d2=a['d2'], gamma=a['gamma'])
    else:
        print(f'{out.name}: pilot solve', flush=True)
        pilot = j._solve_eq11(args, selection,
                  j._finite_shadows(full.basis, pilot_outcomes, np.full(len(rotations), 500)),
                  baseline.d2, baseline.gamma, baseline.d2)
        j._atomic_npz(pp, d2=pilot.d2, gamma=pilot.gamma)
    gH = j._hamiltonian_gradient(selection, rows, cols)
    gF = np.vstack([j._raw_ftpbe_gradient(objective, selection, x.d2, x.gamma,
                                         rows, cols) for x in (pilot, baseline)])
    targets = j._target_model(args, rows, cols, gH, gF)
    L = geometry_stability(selection, pilot, rows, cols, out)
    print(f'{out.name}: design; d={len(rows)}, lineality={L.shape[1]}', flush=True)
    frozen_path=out/'frozen_policies.json'
    if frozen_path.exists():
        designs,details=restore_policies(frozen_path,total,rotations,vectors,blocks,covs,oracle)
        joint=designs['joint']
        print(f'{out.name}: restored frozen policies',flush=True)
        check_paths,_=allocation_paths(joint.basis,(production,),args.production_minimum,
                                       args.allocation_chunk,targets)
        same=np.array_equal(check_paths['pilot_adaptive'][production],joint.paths[production])
        save(out/'separable_allocation_check.json',{'same_frozen_joint_counts':bool(same),
             'policy_remains_frozen':True,'square_row_count':sum(len(x) for x in joint.basis.blocks),
             'dimension':joint.basis.rank})
        if not same:
            from fast_allocation import woodbury_allocation_paths
            j._allocation_paths=woodbury_allocation_paths
            print('Separable path differed numerically: retaining frozen policy and Woodbury for new designs.',flush=True)
    else:
        joint, detail = choose(args, rotations, vectors, blocks, covs, targets, L, pilot_total)
        designs = {'pilot_uniform': equal_design(full, production, pilot_total),
                   'joint': joint}
        details = {'joint': detail}
        if cli.suite=='physical':
            from physical_geometry import physical_lineality
            physical, geometry_diag, E = physical_lineality(selection,pilot.d2,pilot.gamma,rows,cols)
            save(out/'physical_geometry.json',geometry_diag)
            j._atomic_npz(out/'physical_geometry.npz',lineality=physical,equalities=E)
            print(f'{out.name}: physical dimension={physical.shape[1]}',flush=True)
            designs['physical_joint'],details['physical_joint']=choose(
                 args,rotations,vectors,blocks,covs,targets,physical,pilot_total)
        if cli.suite == 'main':
            allocate, _ = j._attach_adaptive_paths(args, full, (production,), targets, pilot_total)
            designs['allocate_only'] = allocate
            designs['select_only'] = equal_design(joint, production, pilot_total)
            structural = base_design('uniform_no_pilot', range(len(rotations)), rotations,
                           vectors, blocks, tuple(np.eye(pairs) for _ in rotations), len(rows), pairs)
            designs['uniform_no_pilot'] = equal_design(structural, total, 0)
            variants = [('no_lineality_veto', replace_namespace(args, conic_guard_fraction=0.), L, targets),
                        ('no_lineality', replace_namespace(args, conic_guard_fraction=0.), np.empty((len(rows),0)), targets),
                        ('global_min_guard', args, np.eye(len(rows)), targets),
                        ('no_ftpbe', args, L, replace(targets, ftpbe_modes=np.zeros_like(gF)))]
            for name, setting, geometry, target in variants:
                print(f'{out.name}: design {name}', flush=True)
                designs[name], details[name] = choose(setting, rotations, vectors, blocks,
                                                       covs, target, geometry, pilot_total)
            if cli.system == 'n2':
                exact_covs = tuple(_exact_single_covariance(oracle._frame_probabilities(i),
                                    oracle.indicators.astype(float))[1] for i in range(len(rotations)))
                exact_g = j._raw_ftpbe_gradient(objective, selection, exact.exact_d2,
                                              exact.exact_gamma, rows, cols)
                oracle_target = replace(targets, ftpbe_modes=np.vstack((exact_g, exact_g)))
                for name, covariance, target in (
                      ('oracle_covariance', exact_covs, targets),
                      ('oracle_gradient', covs, oracle_target),
                      ('oracle_both', exact_covs, oracle_target)):
                    print(f'{out.name}: design {name}', flush=True)
                    designs[name], details[name] = choose(args, rotations, vectors, blocks,
                                                 covariance, target, L, pilot_total)
    # Freeze every path before generating any production outcomes.
    frozen, maxima = {}, np.zeros(len(rotations), dtype=int)
    for name, design in designs.items():
        budget = total-design.pilot_shots
        counts = design.paths[budget]
        pool_counts = j._pool_counts(design, counts, len(rotations))
        assert int(pool_counts.sum())+design.pilot_shots == total
        maxima = np.maximum(maxima, pool_counts)
        frozen[name] = {'indices': list(design.pool_indices), 'counts': counts.tolist(),
                        'pool_counts': pool_counts.tolist(), 'pilot_shots': design.pilot_shots,
                        'oracle': name.startswith('oracle'),
                        'same_counts_as_joint': np.array_equal(pool_counts,
                            j._pool_counts(joint, joint.paths[production], len(rotations)))}
    save(out / 'frozen_policies.json', {'policies': frozen, 'details': details,
         'production_not_yet_generated': True, 'pilot_reused_as_production': False})
    print(f'{out.name}: acquire paired production', flush=True)
    outcomes = j._sample_outcomes(oracle, seed+200003, maxima)
    exact_h, exact_f = j._energy_values(selection, objective, exact.exact_d2, exact.exact_gamma)
    results = []
    cache = {}
    for name, design in designs.items():
        counts = design.paths[total-design.pilot_shots]
        estimators = ['qr_nuclear']
        if cli.suite in ('main','physical') and name in ('pilot_uniform','joint','physical_joint'):
            estimators += ['qr_gls','all_gls']
        for estimator in estimators:
            path = out / f'{name}__{estimator}.json'
            cache_key = (estimator, tuple(frozen[name]['pool_counts']),
                         tuple(design.basis.global_rows), tuple(design.pool_indices))
            if path.exists() and path.with_suffix('.npz').exists():
                saved=json.loads(path.read_text())
                with np.load(path.with_suffix('.npz')) as a:
                    np.testing.assert_array_equal(a['counts'],counts)
                    np.testing.assert_array_equal(a['indices'],design.pool_indices)
                    cache[cache_key]=SimpleNamespace(d2=a['d2'],gamma=a['gamma'],status=saved['status'])
                results.append(saved)
                continue
            # Identical interventions reuse a solve, but retain each labeled record.
            start = time.perf_counter()
            if cache_key in cache:
                result = cache[cache_key]
            elif estimator == 'qr_nuclear':
                print(f'{out.name}: solve {name}/{estimator}', flush=True)
                shadows = j._finite_shadows(design.basis, j._local_outcomes(design, outcomes), counts)
                result = j._solve_eq11(args, selection, shadows, baseline.d2,
                                      baseline.gamma, baseline.d2)
            else:
                print(f'{out.name}: solve {name}/{estimator}', flush=True)
                result = fit_gls(args, selection, design, outcomes, counts, full_blocks,
                         full_rows, full_cols, covs, pilot, estimator == 'all_gls')
            cache[cache_key] = result
            metrics = j._comparison_metrics(result.d2, result.gamma, exact.exact_d2,
                                 exact_h, exact_f, selection, objective)
            info = j._information(design.basis, counts)
            eig = np.linalg.eigvalsh(info)
            matrix_min = {label: float(np.linalg.eigvalsh(M)[0]) for label,M in
                 zip(('D','Q','G'), dqg_matrices(result.d2, result.gamma, selection.pairs))}
            record = {'system': cli.system, 'pool_seed': cli.pool_seed,
                      'width': cli.width, 'shot_seed': seed, 'method': name,
                      'estimator': estimator, 'total_shots': total,
                      'pilot_shots': design.pilot_shots, 'production_shots': int(counts.sum()),
                      'selected_frames': len(design.pool_indices), 'status': result.status,
                      'seconds': time.perf_counter()-start, **metrics,
                      'energy_hit': metrics['hamiltonian_error_meh']<=1.6 and metrics['ftpbe_error_meh']<=1.6,
                      'all_hit': metrics['d2_error']<=.03 and metrics['hamiltonian_error_meh']<=1.6 and metrics['ftpbe_error_meh']<=1.6,
                      'kappa_loc': j._restricted_gain_per_total_shot(info, L, int(counts.sum())),
                      'information_condition': float(eig[-1]/eig[0]),
                      'dqg_min_eigenvalues': matrix_min,
                      'oracle': name.startswith('oracle'),
                      'production_stream': seed+200003}
            save(path, record)
            j._atomic_npz(out/f'{name}__{estimator}.npz', d2=result.d2, gamma=result.gamma,
                          counts=counts, indices=np.asarray(design.pool_indices))
            results.append(record)
            print(f"  errors {metrics['d2_error']:.5f}, {metrics['hamiltonian_error_meh']:.3f}, {metrics['ftpbe_error_meh']:.3f}", flush=True)
    j._write_csv(out/'results.csv', results)
    save(out/'COMPLETE.json', {'records':len(results)})


def replace_namespace(args, **kwargs):
    return SimpleNamespace(**{**vars(args), **kwargs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=['n2','c2'], required=True)
    parser.add_argument('--suite', choices=['main','nested','physical'], default='main')
    parser.add_argument('--pool-seed', type=int, default=0)
    parser.add_argument('--width', type=int, default=30)
    parser.add_argument('--seeds', default='20260811,20260812,20260813,20260814,20260815')
    parser.add_argument('--budget', type=int, default=300000)
    parser.add_argument('--output', type=Path)
    cli = parser.parse_args()
    if cli.width not in (30,50,100) or cli.budget<=cli.width*1000:
        parser.error('Use width 30/50/100 and enough shots for pilot and production.')
    cli.output = cli.output or HERE/'results'/f'{cli.system}_{cli.suite}_pool{cli.pool_seed}_m{cli.width}'
    cli.output.mkdir(parents=True, exist_ok=True)
    lock=(cli.output/'.run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    requested=[int(s) for s in cli.seeds.split(',')]
    config = {**vars(cli), 'output':str(cli.output), 'code_sha256':digest(Path(__file__).read_bytes())}
    config['seeds'] = [int(s) for s in cli.seeds.split(',')]
    previous = cli.output/'configuration.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        for key in ('system','suite','pool_seed','width','budget'):
            if old[key]!=config[key]:
                raise ValueError(f'Resume configuration differs: {key}')
    if all((cli.output/f'seed_{s}'/'COMPLETE.json').exists() for s in requested):
        print(f'{cli.output.name}: all requested seeds complete',flush=True)
        return
    with (cli.output/'configuration_history.jsonl').open('a') as history:
        if previous.exists():history.write(json.dumps({'event':'previous_configuration',**old})+'\n')
        history.write(json.dumps({'event':'resume_configuration',**config})+'\n')
    source_manifest={}
    for source in (Path(__file__),HERE/'fast_allocation.py',HERE/'physical_geometry.py',
                   ROOT/'tools'/'run_n2_random_pilot_joint_design.py',
                   ROOT/'tools'/'run_n2_random_adaptive_allocation.py',
                   ROOT/'code'/'vendor'/'constrained_shadow.py'):
        source_manifest[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    save(cli.output/'source_manifest.json',source_manifest)
    save(previous, config)
    args = (c._arguments([]) if cli.system=='c2' else j._arguments([]))
    args.frame_count=cli.width
    args.budgets=(cli.budget,)
    args.candidate_sizes=tuple(sorted(set(args.candidate_sizes+(cli.width,))))
    args.solver_threads=2
    print(f'Build {cli.system} reference', flush=True)
    if cli.system=='c2':
        selection = c.build_c2_selection_reference(args.bond_length, basis=args.basis)
        exact = c.build_c2_reference(args.bond_length, basis=args.basis,
                                    frozen_mo_coeff=np.asarray(selection.mean_field.mo_coeff))
    else:
        selection = j.build_n2_selection_reference(args.bond_length, basis=args.basis,
                      active_electrons=args.active_electrons, active_orbitals=args.active_orbitals)
        exact = j.build_n2_reference(args.bond_length, basis=args.basis,
                      active_electrons=args.active_electrons, active_orbitals=args.active_orbitals)
    np.testing.assert_allclose(selection.one_body, exact.one_body, atol=2e-9)
    np.testing.assert_allclose(selection.two_body, exact.two_body, atol=2e-9)
    objective = j.FtPBEEnergyObjective(selection, grid_level=args.grid_level)
    rotations, flags = nested_frames(selection, cli.system, cli.pool_seed, cli.width)
    builder = c._c2_random_design if cli.system=='c2' else j._build_random_design
    vectors, _, full_blocks, full_rows, full_cols = builder(selection, rotations)
    if cli.system=='c2':
        # Freeze chart using the 30-frame base, regardless of pool width.
        _, rows, cols, columns, rank = c._observable_coordinate_chart(full_blocks[:30], full_rows, full_cols)
        blocks = tuple(b[:,columns] for b in full_blocks)
    else:
        blocks, rows, cols = full_blocks, full_rows, full_cols
    j._atomic_npz(cli.output/'pool.npz', rotations=np.asarray(rotations), flags=flags,
                   rows=rows, cols=cols)
    save(cli.output/'pool_fingerprints.json', [digest(r) for r in rotations])
    bp=cli.output/'baseline.npz'
    cached_baseline=HERE/'results'/f'{cli.system}_main_pool0_m30'/'baseline.npz'
    if not bp.exists() and cached_baseline.exists():
        with np.load(cached_baseline) as a:
            j._atomic_npz(bp,d2=a['d2'],gamma=a['gamma'])
    if bp.exists():
        with np.load(bp) as a:
            baseline=SimpleNamespace(d2=a['d2'], gamma=a['gamma'])
    else:
        print('Solve shadow-free DQG', flush=True)
        baseline=j.solve_dqg_sdp(j.LeakGuardReference(selection), solver=args.solver,
          tolerance=args.solver_tolerance, max_iterations=args.max_iterations,
          solver_threads=args.solver_threads, positivity_conditions='DQG', symmetry_blocked_psd=True)
        j._atomic_npz(bp,d2=baseline.d2,gamma=baseline.gamma)
    for seed in config['seeds']:
        run_seed(args,cli,seed,selection,exact,objective,rotations,vectors,blocks,rows,cols,
                 full_blocks,full_rows,full_cols,baseline)
    save(cli.output/'COMPLETE.json', {'seeds':config['seeds']})


if __name__ == '__main__':
    main()
