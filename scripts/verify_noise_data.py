"""Verify the supplied circuit-noise records and build SI calibration/result tables."""
from pathlib import Path
import csv, hashlib, json, math, re, shutil
import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT/'generated/noise-tables'
MS.mkdir(parents=True, exist_ok=True)
PKG = ROOT/'noise-study'
OUT = ROOT/'generated/noise-validation'
OUT.mkdir(parents=True, exist_ok=True)
model_file = PKG / 'circuits/calibration/wukong_path16.json'
model = json.loads(model_file.read_text())
workbook = PKG / 'circuits/calibration/本源悟空180-2芯片数据.xlsx'
assert hashlib.sha256(workbook.read_bytes()).hexdigest() == model['source_sha256']
sheet = openpyxl.load_workbook(workbook, read_only=True, data_only=True)['qubit']
qubits, edges = {}, {}
for row in list(sheet.iter_rows(values_only=True))[1:]:
    if not row or not re.fullmatch(r'q\d+', str(row[0])):
        continue
    q = int(row[0][1:])
    qubits[q] = (1-float(row[8]), 1-float(row[5]))
    for a,b,f in re.findall(r'CZ(\d+)_(\d+):([\d.]+)', str(row[9])):
        edges[tuple(sorted((int(a),int(b))))] = 1-float(f)
path = model['logical_to_physical']
for q,physical in enumerate(path):
    assert np.allclose(qubits[physical], [model['physical_1q_error'][q], model['readout_per_qubit'][q]], atol=1e-14)
for j in range(15):
    assert abs(edges[tuple(sorted(path[j:j+2]))]-model['physical_cz_error'][j]) < 1e-14

records = []
histogram_count = 0
for scale,folder in [(1.,'results'),(.2,'results_gate020')]:
    directory = PKG/'code/mindquantum_poc'/folder
    for p in sorted(directory.glob('*/variants/*.json')):
        r=json.loads(p.read_text()); r.setdefault('gate_scale',1.0); assert r['gate_scale']==scale and r['stream']==0 and r['budget']==120000
        assert r['fit_shots']==(112500 if r['arm']=='guard15_equal' else 120000)
        assert r['mu_ftpbe']==(2 if r['arm']=='guard15_equal' else 0)
        assert r['solver_stats']['status'] in ('optimal','optimal_inaccurate')
        records.append(r)
    for p in sorted(directory.glob('*/*_histograms.npz')):
        with np.load(p) as a:
            n=15 if p.name.startswith('guard15') else 30
            assert a['noisy'].shape==a['noiseless'].shape==(n,65536)
            assert np.all(a['shots']==(7500 if n==15 else 4000))
            assert np.array_equal(a['noisy'].sum(axis=1),a['shots'])
            assert np.array_equal(a['noiseless'].sum(axis=1),a['shots'])
            if n==15:
                plan=json.loads((PKG/'code/guard15_equal_real30_n2_scan120k/plans'/p.parent.name/'r0_b120000.json').read_text())
                assert list(a['indices'])==plan['indices']
        histogram_count+=1
assert len(records)==352 and histogram_count==44
summary = list(csv.DictReader((PKG/'analysis/summary_by_scale.csv').open()))
variants=['exact','clean','raw','post','rem','rem_lin','rem_post','rem_lin_post']
stats={}
for scale in [.2,1.]:
    for arm in ['uniform','guard15_equal']:
        for variant in variants:
            rows=[r for r in records if r['gate_scale']==scale and r['arm']==arm and r['variant']==variant]
            assert len(rows)==11 and len({r['bond_angstrom'] for r in rows})==11
            values={x:math.sqrt(sum(r[f'{x}_error_meh']**2 for r in rows)/11) for x in ['h','f']}
            stored=next(r for r in summary if float(r['gate_scale'])==scale and r['arm']==arm and r['variant']==variant)
            for x in ['h','f']: assert abs(values[x]-float(stored[f'{x}_rmse_meh'])) < 1e-9
            stats[f'{scale}/{arm}/{variant}']=values

def table(caption,label,cols,header,rows):
    return '\n'.join([r'\begin{table}[!htbp]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\small',
        r'\begin{tabular}{'+cols+'}',r'\toprule',header+r' \\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}',r'\end{table}',''])

rows=[]
for q,p in enumerate(path):
    rows.append(f'{q} & {p} & {100*model["physical_1q_error"][q]:.2f} & {100*model["readout_per_qubit"][q]:.2f}'+r' \\')
(MS/'noise-calibration-qubits.tex').write_text(table(
    r'\textbf{Qubit calibration inputs for the circuit-noise model.} Logical-to-physical mapping on Origin Wukong-180-2. Single-qubit gate infidelity $e^{(1)}_q=1-F^{(1)}_q$ and symmetric readout-flip probability $r_q=1-F^{\mathrm{read}}_q$ are reported as percentages. Values are taken from the platform calibration workbook exported on 16 July 2026. The same readout probabilities are used at $\mathrm{scale}=0.2$ and $\mathrm{scale}=1.0$.',
    'tab:noise-qubits','rrrr',r'Logical qubit & Physical qubit & $100e^{(1)}_q$ & $100r_q$',rows))
rows=[]
for j in range(15):
    p2=model['physical_cz_error'][j]
    rates=[100*(1-(1-p2)**2*(1-model['physical_1q_error'][q])**2) for q in [j,j+1]]
    rate_text='-- & --' if j==7 else f'{rates[0]:.4f} & {rates[1]:.4f}'
    rows.append(f'{j}--{j+1} & {path[j]}--{path[j+1]} & {100*p2:.2f} & '+rate_text+r' \\')
(MS/'noise-calibration-edges.tex').write_text(table(
    r'\textbf{Edge calibration inputs and effective Givens-block noise parameters.} All entries are percentages. $e^{(\mathrm{CZ})}_{ij}=1-F^{(\mathrm{CZ})}_{ij}$ is the platform CZ infidelity. The last two columns give the effective single-qubit depolarizing parameters applied after each Givens block on edge $(i,j)$ at $\mathrm{scale}=1.0$; they are multiplied by 0.2 for $\mathrm{scale}=0.2$. The logical 7--8 edge connects the two spin sectors and is not used by the measurement circuit.',
    'tab:noise-edges','llrrr',r'Logical edge & Physical edge & $100e^{(\mathrm{CZ})}_{ij}$ & $100p_{i,ij}(1)$ & $100p_{j,ij}(1)$',rows))
labels={'exact':'Exact means','clean':'Ideal gates/readout','raw':'Raw','post':'Post', 'rem':'Clipped REM','rem_lin':'REM-lin','rem_post':'Clipped REM + post','rem_lin_post':'REM-lin + post'}
rows=[]
for g in [.2,1.]:
    for v in variants:
        nums=[stats[f'{g}/{arm}/{v}'][x] for arm in ['uniform','guard15_equal'] for x in ['h','f']]
        rows.append(f'{g:g} & {labels[v]} & '+' & '.join(f'{a:.3f}' for a in nums)+r' \\')
    if g==.2:rows.append(r'\midrule')
(MS/'noise-summary-table.tex').write_text(table(
    r'\textbf{Scan-wide energy RMSEs for the circuit-noise processing controls.} N$_2$/cc-pVDZ/CAS(10e,8o), 11 geometries from 0.80 to 2.50~\AA, one fixed stream-0 plan and one circuit-sampling realization per geometry and condition. RMSEs are $[\sum_{k=1}^{11}(\Delta E_{X,k})^2/11]^{1/2}$ in m$E_{\mathrm h}$, relative to exact CASCI Hamiltonian ($H$) and total ftPBE ($F$) references. Uniform30 and Select15 use $(\lambda,\mu)=(1,0)$ and $(1,2)$, respectively, under the 120,000-shot protocol accounting described in Section~S9. Exact means use no sampled data; ideal gates/readout retain finite sampling. The remaining rows use the same noisy histograms at each $\mathrm{scale}$ and differ only in processing. REM-lin retains signed quasi-probabilities; clipped REM sets negative entries to zero before normalization. Post denotes restriction to $N_\alpha=N_\beta=5$.',
    'tab:noise-summary','rlrrrr',r'Scale & Processing & Uniform $H$ & Uniform $F$ & Select $H$ & Select $F$',rows))

for name in ['summary_by_scale.csv','records_g1.0_g0.2.csv']:
    shutil.copy2(PKG/'analysis'/name,OUT/name)
shutil.copy2(workbook,OUT/'wukong1802-calibration.xlsx')
shutil.copy2(model_file,OUT/'wukong_path16.json')
validation=dict(records_verified=len(records),histogram_files_verified=histogram_count,aggregate_groups_verified=len(stats),
    calibration_sha256=model['source_sha256'],calibration_source='Origin Quantum Cloud platform, confirmed by author',
    workbook_export_timestamp_utc='2026-07-16T11:22:01Z',device_calibration_timestamp='not recorded in the supplied workbook',
    stream=0,selection='fixed archived ideal-measurement pilot plan; no noisy-pilot reselection',
    noisy_fit_shots={'Uniform30':120000,'Select15':112500},nominal_protocol_budget=120000,
    mitigation='unclipped tensor-product readout inversion followed by signed particle-sector normalization',
    stats=stats)
(OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print('Verified 352 records, 44 histogram files and 32 RMSE groups; built three SI tables.')
