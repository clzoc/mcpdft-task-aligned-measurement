#!/usr/bin/env python3
"""120K N2 bond scan using the frozen equal-allocation guard15 engine."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PRIMARY=ROOT/'guard15_equal_real30'
import engine as e
import numpy as np

BONDS=(.80,.90,1.00,1.10,1.25,1.45,1.60,1.80,2.00,2.20,2.50)
SYSTEMS={f'n2_r{int(round(b*100)):03d}':b for b in BONDS}
ARMS=('guard15_equal','uniform')
BUDGET=120000
STREAMS=tuple(range(8))
e.HERE=HERE
e.SYSTEMS=tuple(SYSTEMS)
e.BUDGETS=(BUDGET,)
e.ARMS=ARMS
e.STREAMS=STREAMS

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def context(tag):
    if tag=='n2_r110':
        raise RuntimeError('1.10 Angstrom must be imported from completed primary results, never recomputed.')
    g=e.g
    g.HERE=HERE/'contexts'
    g.N2_SCAN.update(SYSTEMS)
    # The existing chemistry/anchor path is unchanged; replace its pool loader
    # so the scan installs the requested all-real same-spin pool directly.
    with np.load(HERE/'pool_real30.npz') as z:
        rotations=np.array(z['rotations'])
    original=g.c108.pool_uniform_family
    g.c108.pool_uniform_family=lambda c:(rotations,np.zeros(30,dtype=bool))
    try:
        c=g.n2_geometry_context(tag)
    finally:
        g.c108.pool_uniform_family=original
    np.testing.assert_array_equal(c.rotations,rotations)
    g.da.band.SOLVE_BAND=g.da.band.band_solver(True)
    return g.da,c

e.context=context

def initialize():
    primary=json.loads((PRIMARY/'protocol.json').read_text())
    with np.load(HERE/'pool_real30.npz') as z:
        rotations=np.array(z['rotations'])
    assert e.g.allocation.digest(rotations)==primary['pool']['digest']
    assert rotations.shape==(30,2,8,8)
    assert np.max(np.abs(rotations.imag))==0
    np.testing.assert_array_equal(rotations[:,0],rotations[:,1])
    files=[HERE/'scan.py',HERE/'engine.py',HERE/'run_campaign.py',HERE/'after_primary.py']
    protocol=dict(name=HERE.name,bonds_angstrom=BONDS,systems=SYSTEMS,budget=BUDGET,
                  streams=STREAMS,arms=ARMS,total_results=176,new_solves=160,reused_results=16,
                  chemistry=dict(molecule='N2',basis='cc-pvdz',cas_electrons=10,cas_orbitals=8,ncore=2,
                                 builder='same canonical RHF/CASCI N2 context as primary, existing n2_geometry_context'),
                  pool=primary['pool'],pilot_per_frame=500,pilot_shots=15000,
                  selection='guard15 K=15 backward-greedy path in identifiable pool subspace, independently per geometry/stream',
                  guard=dict(lambda_radius=1.,mu_ftpbe=2.,new_shots_per_selected_frame=7000,
                             fit_shots_per_selected_frame=7500,fit_shots=112500,unused_pilot_shots=7500,
                             objective='H + 2*F_lin(geometry DQG anchor) + Tr(E)'),
                  uniform=dict(lambda_radius=1.,mu_ftpbe=0.,pilot_shots=0,shots_per_frame=4000,
                               fit_shots=120000,objective='original global_band spin H + Tr(E)'),
                  error_basis='spin',full_E=False,positivity='DQG',solver='MOSEK',tolerance=1e-8,
                  seed_formula=primary['seed_formula'],same_pool_and_seed_per_stream_across_geometries=True,
                  reuse_equilibrium=dict(source=str(PRIMARY),source_system='n2',target_system='n2_r110',
                                         rule='copy original JSON metrics and NPZ arrays; record source hashes; no new solve'),
                  dependency=dict(source_campaign=str(PRIMARY/'campaign_state.json'),required_status='complete',
                                  required_results=320,scan_starts_only_after_primary=True),
                  reporting='H/F RMSE and bias in mHa separately; mean and RMS unnormalized Frobenius 2-RDM error; all eight per-stream values',
                  source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
                  anchor_sha256={tag:sha(HERE/'contexts/anchors'/f'{tag}.npz') for tag in SYSTEMS})
    e.save(HERE/'protocol.json',protocol)
    print('INITIALIZED 11 geometries, 176 results, 160 new solves, 16 reused',flush=True)

def atomic_copy(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert sha(source)==sha(target),f'Existing copy mismatch: {target}'
        return
    with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as h:
        temp=Path(h.name)
    shutil.copyfile(source,temp)
    temp.replace(target)

def import_equilibrium():
    state=json.loads((PRIMARY/'campaign_state.json').read_text())
    assert state['status']=='complete','Primary must complete before importing and launching scan'
    protocol=json.loads((HERE/'protocol.json').read_text())
    imported=[]
    for stream in STREAMS:
        records={}
        for arm in ARMS:
            source=PRIMARY/'results/n2'/arm/f'b{BUDGET}_r{stream}.json'
            arrays=source.with_suffix('.npz')
            d=json.loads(source.read_text())
            assert d['system']=='n2' and d['arm']==arm and d['stream']==stream and d['budget']==BUDGET
            assert d['status']=='optimal' and d['error_basis']=='spin'
            assert d['lambda_radius']==1. and d['mu_ftpbe']==(2. if arm=='guard15_equal' else 0.)
            assert d['pool_digest']==protocol['pool']['digest']
            assert sum(d['counts'])==BUDGET
            indices=np.array(d['indices']);counts=np.array(d['counts'])
            assert len(indices)==(15 if arm=='guard15_equal' else 30)
            assert np.all(counts[indices]==(7500 if arm=='guard15_equal' else 4000))
            with np.load(arrays) as z:
                np.testing.assert_array_equal(z['counts'],counts)
                np.testing.assert_array_equal(z['indices'],indices)
            target=e.result_path('n2_r110',arm,stream,BUDGET)
            atomic_copy(arrays,target.with_suffix('.npz'))
            imported_record=dict(d,system='n2_r110',bond_angstrom=1.10,source_system='n2',
                                 reused_from=str(source),source_result_sha256=sha(source),
                                 source_array_sha256=sha(arrays),no_new_solve=True)
            if target.exists():assert json.loads(target.read_text())==imported_record
            else:e.save(target,imported_record)
            imported.append(dict(target=str(target),source=str(source),source_sha256=sha(source),
                                 array_sha256=sha(arrays)))
            records[arm]=d
        assert len({d['prefix500_digest'] for d in records.values()})==1
        atomic_copy(PRIMARY/'plans/n2'/f'r{stream}_b120000.json',HERE/'plans/n2_r110'/f'r{stream}_b120000.json')
        atomic_copy(PRIMARY/'pilots/n2'/f'r{stream}.npz',HERE/'pilots/n2_r110'/f'r{stream}.npz')
    e.save(HERE/'equilibrium_reuse.json',dict(count=len(imported),records=imported))
    print('IMPORTED 16 unchanged numerical results at 1.10 Angstrom',flush=True)

def summarize():
    details=[];rows=[]
    for tag,bond in SYSTEMS.items():
        paired={arm:[] for arm in ARMS}
        for stream in STREAMS:
            paths={a:e.result_path(tag,a,stream,BUDGET) for a in ARMS}
            if not all(p.exists() for p in paths.values()):continue
            pair={a:json.loads(p.read_text()) for a,p in paths.items()}
            assert len({d['prefix500_digest'] for d in pair.values()})==1
            for arm,d in pair.items():
                assert d['status']=='optimal' and d['error_basis']=='spin'
                assert d['budget']==BUDGET and d['lambda_radius']==1 and d['mu_ftpbe']==(2 if arm==ARMS[0] else 0)
                counts=np.array(d['counts']);indices=np.array(d['indices'])
                assert counts.sum()==BUDGET
                assert len(indices)==(15 if arm==ARMS[0] else 30)
                assert np.all(counts[indices]==(7500 if arm==ARMS[0] else 4000))
                if arm==ARMS[0]:
                    assert np.all(counts[np.setdiff1d(np.arange(30),indices)]==500)
                    assert d['pilot_shots']==15000 and d['fit_shots']==112500
                else:assert d['pilot_shots']==0 and d['fit_shots']==120000
                with np.load(paths[arm].with_suffix('.npz')) as z:
                    np.testing.assert_array_equal(z['counts'],counts)
                    np.testing.assert_array_equal(z['indices'],indices)
                paired[arm].append(d)
                details.append(dict(system=tag,bond_angstrom=bond,stream=stream,budget=BUDGET,arm=arm,
                                    lambda_radius=d['lambda_radius'],mu_ftpbe=d['mu_ftpbe'],
                                    h_error_mHa=d['h_error_meh'],f_error_mHa=d['f_error_meh'],
                                    d2_frobenius_error=d['d2_error'],reused=d.get('no_new_solve',False)))
        if not paired[ARMS[0]]:continue
        row=dict(system=tag,bond_angstrom=bond,budget=BUDGET,streams=len(paired[ARMS[0]]))
        for arm,ds in paired.items():
            h=np.array([d['h_error_meh'] for d in ds]);f=np.array([d['f_error_meh'] for d in ds]);dd=np.array([d['d2_error'] for d in ds])
            row.update({arm+'_h_rmse_mHa':float(np.sqrt(np.mean(h*h))),arm+'_f_rmse_mHa':float(np.sqrt(np.mean(f*f))),
                        arm+'_h_bias_mHa':float(h.mean()),arm+'_f_bias_mHa':float(f.mean()),
                        arm+'_d2_mean':float(dd.mean()),arm+'_d2_rms':float(np.sqrt(np.mean(dd*dd)))})
        rows.append(row)
    for name,table in [('summary',rows),('per_stream',details)]:
        e.save(HERE/f'{name}.json',table)
        if table:
            with (HERE/f'{name}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    complete=len(details)==176 and len(rows)==11 and all(row['streams']==8 for row in rows)
    e.save(HERE/'validation.json',dict(complete=complete,results=len(details),expected_results=176,
                                     geometries=len(rows),all_present_records_optimal=True))
    print('SUMMARY',len(details),'of 176 results;',len(rows),'geometries; complete',complete,flush=True)
    if not complete:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,(metric,label) in zip(axes,[('h_rmse_mHa','H RMSE (mHa)'),('f_rmse_mHa','F RMSE (mHa)'),('d2_mean','Mean 2-RDM Frobenius error')]):
        for arm in ARMS:
            ax.plot([r['bond_angstrom'] for r in rows],[r[arm+'_'+metric] for r in rows],'-o',label=arm)
        ax.set(xlabel='N2 bond length (Angstrom)',ylabel=label)
        ax.grid(alpha=.25);ax.legend()
    fig.savefig(HERE/'scan_errors.png',dpi=180)
    fig.savefig(HERE/'scan_errors.pdf')
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','import-equilibrium','worker','summarize'])
    p.add_argument('--system',choices=tuple(SYSTEMS));p.add_argument('--arm',choices=ARMS)
    p.add_argument('--stream',type=int,choices=STREAMS);p.add_argument('--budget',type=int,choices=[BUDGET],default=BUDGET)
    p.add_argument('--threads',type=int,default=3);a=p.parse_args()
    if a.command=='init':initialize()
    elif a.command=='import-equilibrium':import_equilibrium()
    elif a.command=='summarize':summarize()
    else:
        if a.system=='n2_r110':raise RuntimeError('Equilibrium must be reused, not recomputed')
        e.worker(a.system,a.arm,a.stream,a.threads,a.budget)
