#!/usr/bin/env python3
"""Oracle design-signal audit: which signals actually track each energy error?

Extends the main-suite geometry audit to every suite (main, physical, nested)
and asks, with paired statistics, which computable design signals genuinely
track the realized D2/Hamiltonian/ftPBE errors. Signals are properties of the
frozen design only; realized errors come from the saved reconstruction records.

Outputs under results/:
  oracle_signal_records.csv       per (run, shot_seed, method) signal table
  oracle_signal_correlations.csv  within-seed and across-seed Spearman panels
  oracle_selection_headroom.csv   signal-based design selection vs oracle pick
  oracle_width_deltas.csv         nested-pool widening: Delta signal vs Delta error
  oracle_signals.png              specificity heatmap + captured headroom
"""
from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_evidence import j, c, HERE, base_design
from physical_geometry import physical_lineality

RESULTS = HERE / 'results'
METRICS = ('d2_error', 'hamiltonian_error_meh', 'ftpbe_error_meh')
TARGET_OF_METRIC = dict(zip(METRICS, 'DHF'))
SIGNALS = (
    'kappa_loc', 'information_condition', 'selected_frames',
    'physical_kappa', 'physical_lineality_dimension',
    'working_q_D', 'working_q_H', 'working_q_F',
    'subspace_only_q_D', 'subspace_only_q_H', 'subspace_only_q_F',
    'alloc_entropy', 'alloc_max_share', 'alloc_effective_frames',
    'alloc_nonzero_frames',
)


def allocation_features(counts):
    counts = np.asarray(counts, dtype=float)
    shares = counts[counts > 0] / counts.sum()
    return {
        'alloc_nonzero_frames': int(len(shares)),
        'alloc_entropy': float(-(shares * np.log(shares)).sum() / np.log(len(shares))
                         if len(shares) > 1 else 0.0),
        'alloc_max_share': float(shares.max()),
        'alloc_effective_frames': float(1.0 / np.square(shares).sum()),
    }


def build_reference(system, args):
    if system == 'c2':
        selection = c.build_c2_selection_reference(args.bond_length, basis=args.basis)
        exact = c.build_c2_reference(args.bond_length, basis=args.basis,
                    frozen_mo_coeff=np.asarray(selection.mean_field.mo_coeff))
    else:
        selection = j.build_n2_selection_reference(args.bond_length, basis=args.basis,
                    active_electrons=args.active_electrons,
                    active_orbitals=args.active_orbitals)
        exact = j.build_n2_reference(args.bond_length, basis=args.basis,
                active_electrons=args.active_electrons,
                active_orbitals=args.active_orbitals)
    return selection, exact


def run_design(selection, system, rotations):
    builder = c._c2_random_design if system == 'c2' else j._build_random_design
    vectors, _, blocks, rows, cols = builder(selection, rotations)
    if system == 'c2':
        blocks, rows, cols, _, _ = c._observable_coordinate_chart(blocks, rows, cols)
    return vectors, blocks, rows, cols


def seed_signals(system, run, seed_dir, selection, exact, args, vectors, blocks,
                 rows, cols, rotations):
    pilot = np.load(seed_dir / 'pilot.npz')
    seed = int(seed_dir.name.split('_')[1])
    physical, _, _ = physical_lineality(selection, pilot['d2'], pilot['gamma'],
                                        rows, cols)
    oracle = j.AcquisitionOracle(exact, rotations, vectors, 1, seed)
    outcomes = j._sample_outcomes(oracle, seed + 100003, np.full(len(rotations), 500))
    cov = tuple(j._regularized_single_covariance(x.astype(float)) for x in outcomes)
    objective = j.FtPBEEnergyObjective(selection, grid_level=1)
    baseline = np.load(run / 'baseline.npz')
    modes = np.vstack([j._raw_ftpbe_gradient(objective, selection, x['d2'],
                       x['gamma'], rows, cols) for x in (pilot, baseline)])
    targets = j._target_model(args, rows, cols,
                              j._hamiltonian_gradient(selection, rows, cols), modes)
    identity = tuple(np.eye(blocks[0].shape[0]) for _ in rotations)
    policies = json.loads((seed_dir / 'frozen_policies.json').read_text())['policies']
    records = []
    for method, policy in sorted(policies.items()):
        if policy['oracle']:
            continue
        covs = identity if method == 'uniform_no_pilot' else cov
        design = base_design(method, policy['indices'], rotations, vectors, blocks,
                             covs, len(rows), len(selection.pairs))
        counts = np.asarray(policy['counts'], dtype=int)
        information = j._information(design.basis, counts)
        kappa = j._restricted_gain_per_total_shot(information, physical,
                                                  int(counts.sum()))
        risks = j._risks(j._inverse_information(information),
                         targets) / targets.capacities_squared
        projected = physical @ j._inverse_information(
            physical.T @ information @ physical) @ physical.T
        subspace_risks = j._risks(projected, targets) / targets.capacities_squared
        records.append({
            'system': system, 'run': run.name, 'shot_seed': seed, 'method': method,
            'physical_lineality_dimension': int(physical.shape[1]),
            'physical_kappa': None if kappa is None else float(kappa),
            'limiting_working_risk': 'DHF'[int(np.argmax(risks))],
            'selected_frames': int(len(policy['indices'])),
            **{f'working_q_{k}': float(v) for k, v in zip('DHF', risks)},
            **{f'subspace_only_q_{k}': float(v) for k, v in zip('DHF', subspace_risks)},
            **allocation_features(counts),
        })
    return records


def signal_table():
    tables = []
    references = {}
    runs = sorted(p for p in RESULTS.glob('*_pool*_m*')
                  if (p / 'pool.npz').exists())
    for run in runs:
        system = run.name.split('_')[0]
        if system not in references:
            args = c._arguments([]) if system == 'c2' else j._arguments([])
            print('Build', system, 'reference', flush=True)
            references[system] = (args, *build_reference(system, args))
        args, selection, exact = references[system]
        rotations = tuple(np.load(run / 'pool.npz')['rotations'])
        vectors, blocks, rows, cols = run_design(selection, system, rotations)
        for seed_dir in sorted(run.glob('seed_*')):
            if not (seed_dir / 'frozen_policies.json').exists():
                continue
            tables.extend(seed_signals(system, run, seed_dir, selection, exact,
                                       args, vectors, blocks, rows, cols,
                                       rotations))
            print(system, run.name, seed_dir.name, 'audited', flush=True)
    signals = pd.DataFrame(tables)

    observations = pd.read_csv(RESULTS / 'all_observations.csv')
    design_levels = ['kappa_loc', 'information_condition']
    levels = (observations.groupby(['run', 'shot_seed', 'method'])[design_levels]
              .first().reset_index())
    merged = signals.merge(levels, on=['run', 'shot_seed', 'method'], how='left')
    errors = observations[['run', 'shot_seed', 'method', 'estimator', *METRICS,
                           'signed_hamiltonian_error_meh',
                           'signed_ftpbe_error_meh', 'all_hit', 'energy_hit']]
    merged = merged.merge(errors, on=['run', 'shot_seed', 'method'], how='left')
    # Recovered seeds live in <suite>_seed<ID> run directories; pair designs
    # only within their own suite, never across pools or geometries.
    merged['suite'] = merged.run.str.replace(r'_seed\d+$', '', regex=True)
    merged.to_csv(RESULTS / 'oracle_signal_records.csv', index=False)
    return merged


def within_seed_correlations(df, min_methods=6, signals=SIGNALS):
    """Per suite x seed, Spearman across methods; then median/sign over seeds."""
    rows = []
    data = df[df.estimator == 'qr_nuclear']
    for (suite, seed), group in data.groupby(['suite', 'shot_seed']):
        if group.method.nunique() < min_methods:
            continue
        system = group.system.iloc[0]
        for metric in METRICS:
            for signal in signals:
                pair = group[[metric, signal]].dropna()
                if pair[signal].nunique() < 3:
                    continue
                rho = stats.spearmanr(pair[signal], pair[metric]).statistic
                rows.append({'system': system, 'suite': suite,
                             'shot_seed': seed, 'metric': metric,
                             'signal': signal, 'spearman': rho,
                             'n_methods': len(pair)})
    per_seed = pd.DataFrame(rows)
    summary = (per_seed.groupby(['system', 'metric', 'signal'])
               .agg(median_spearman=('spearman', 'median'),
                    min_spearman=('spearman', 'min'),
                    max_spearman=('spearman', 'max'),
                    sign_consistency=('spearman',
                                      lambda s: float(np.mean(np.sign(s) ==
                                                    np.sign(np.median(s))))),
                    n_seeds=('spearman', 'size'))
               .reset_index())
    summary['panel'] = 'within_seed_across_methods'
    return per_seed, summary


def across_seed_correlations(df, method='joint', signals=SIGNALS):
    rows = []
    data = df[(df.estimator == 'qr_nuclear') & (df.method == method)]
    for (system, suite), group in data.groupby(['system', 'suite']):
        if len(group) < 3:
            continue
        for metric in METRICS:
            for signal in signals:
                pair = group[[metric, signal]].dropna()
                if pair[signal].nunique() < 3:
                    continue
                rows.append({'system': system, 'suite': suite, 'metric': metric,
                             'signal': signal, 'n_seeds': len(pair),
                             'spearman': stats.spearmanr(
                                 pair[signal], pair[metric]).statistic,
                             'panel': f'across_seed_{method}'})
    return pd.DataFrame(rows)


def selection_headroom(df, min_methods=6, signals=SIGNALS):
    """Pick the design with the smallest signal; compare with the oracle pick.

    captured = (error_joint - error_signal_pick) / (error_joint - error_oracle)
    Only seeds where joint is not already the oracle pick are informative.
    """
    rows = []
    data = df[df.estimator == 'qr_nuclear']
    for (suite, seed), group in data.groupby(['suite', 'shot_seed']):
        if group.method.nunique() < min_methods:
            continue
        joint = group.loc[group.method == 'joint']
        if joint.empty:
            continue
        system = group.system.iloc[0]
        for metric in METRICS:
            oracle_error = float(group[metric].min())
            joint_error = float(joint[metric].iloc[0])
            gap = joint_error - oracle_error
            row = {'system': system, 'suite': suite, 'shot_seed': seed,
                   'metric': metric,
                   'joint_error': joint_error, 'oracle_error': oracle_error,
                   'oracle_method': group.loc[group[metric].idxmin(), 'method']}
            for signal in signals:
                pick = group.loc[group[signal].idxmin()]
                error = float(pick[metric])
                row[f'pick__{signal}'] = pick['method']
                row[f'captured__{signal}'] = ((joint_error - error) / gap
                                              if gap > 0 else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def width_deltas(df):
    nested = df[(df.estimator == 'qr_nuclear')
                & df.run.str.contains('nested')].copy()
    parts = nested.run.str.extract(r'pool(?P<pool>\d+)_m(?P<width>\d+)')
    nested['pool'] = parts.pool.astype(int)
    nested['width'] = parts.width.astype(int)
    base = nested[nested.width == 30].set_index(['pool', 'shot_seed', 'method'])
    rows = []
    for width in (50, 100):
        wide = nested[nested.width == width].set_index(['pool', 'shot_seed',
                                                        'method'])
        joined = wide.join(base, lsuffix='_wide', rsuffix='_base')
        for metric in METRICS:
            for signal in SIGNALS:
                paired = joined[[f'{metric}_wide', f'{metric}_base',
                                 f'{signal}_wide', f'{signal}_base']].dropna()
                if len(paired) < 5:
                    continue
                d_error = paired[f'{metric}_wide'] - paired[f'{metric}_base']
                d_signal = paired[f'{signal}_wide'] - paired[f'{signal}_base']
                if d_signal.nunique() < 3:
                    continue
                rows.append({'width': width, 'metric': metric, 'signal': signal,
                             'n_pairs': len(paired),
                             'spearman_delta': stats.spearmanr(
                                 d_signal, d_error).statistic,
                             'median_delta_error_when_signal_up':
                                 float(d_error[d_signal > 0].median())
                                 if (d_signal > 0).any() else np.nan,
                             'median_delta_error_when_signal_down':
                                 float(d_error[d_signal < 0].median())
                                 if (d_signal < 0).any() else np.nan})
    return pd.DataFrame(rows)


def plot_panels(per_seed_summary, headroom):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    focus = [s for s in SIGNALS if not s.startswith('subspace')]
    for system, ax in zip(('n2', 'c2'), axes):
        table = (per_seed_summary[per_seed_summary.system == system]
                 .pivot(index='signal', columns='metric', values='median_spearman')
                 .reindex(focus)[list(METRICS)])
        image = ax.imshow(table.values, cmap='RdBu_r', vmin=-1, vmax=1,
                          aspect='auto')
        ax.set_xticks(range(3), [r'$\varepsilon_D$', r'$\varepsilon_H$',
                                 r'$\varepsilon_F$'])
        ax.set_yticks(range(len(table)), table.index, fontsize=8)
        ax.set_title(f'{system}: median within-seed Spearman')
        for i in range(len(table)):
            for k in range(3):
                value = table.values[i, k]
                if np.isfinite(value):
                    ax.text(k, i, f'{value:.2f}', ha='center', va='center',
                            fontsize=7)
        fig.colorbar(image, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(RESULTS / 'oracle_signals.png', dpi=160)
    fig.savefig(RESULTS / 'oracle_signals.svg')
    plt.close(fig)


def main():
    if '--from-saved' in sys.argv:
        merged = pd.read_csv(RESULTS / 'oracle_signal_records.csv')
        if 'suite' not in merged:
            merged['suite'] = merged.run.str.replace(r'_seed\d+$', '',
                                                     regex=True)
            merged.to_csv(RESULTS / 'oracle_signal_records.csv', index=False)
    else:
        merged = signal_table()
    per_seed, summary = within_seed_correlations(merged)
    across = across_seed_correlations(merged)
    headroom = selection_headroom(merged)
    deltas = width_deltas(merged)
    pd.concat([summary, across]).to_csv(
        RESULTS / 'oracle_signal_correlations.csv', index=False)
    per_seed.to_csv(RESULTS / 'oracle_signal_correlations_per_seed.csv',
                    index=False)
    headroom.to_csv(RESULTS / 'oracle_selection_headroom.csv', index=False)
    deltas.to_csv(RESULTS / 'oracle_width_deltas.csv', index=False)
    plot_panels(summary, headroom)

    captured = [k for k in headroom.columns if k.startswith('captured__')]
    medians = (headroom.groupby(['system', 'metric'])[captured].median()
               .rename(columns=lambda k: k.replace('captured__', '')))
    print('\nMedian captured headroom (fraction of joint->oracle gap):')
    print(median_table := medians.round(2).to_string())
    (RESULTS / 'oracle_headroom_summary.txt').write_text(median_table)
    print('\nTop within-seed signals per system/metric (median Spearman):')
    for (system, metric), group in summary.groupby(['system', 'metric']):
        top = group.reindex(group.median_spearman.abs().sort_values(
            ascending=False).index).head(3)
        print(system, metric, ' | '.join(
            f'{r.signal}:{r.median_spearman:.2f}' for r in top.itertuples()))


if __name__ == '__main__':
    main()
