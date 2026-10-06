"""Validate the distributed records and statistics without chemistry or SDP solves."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EQ=ROOT/'analysis/guard15_equal_real30'
SCAN=ROOT/'analysis/guard15_equal_real30_n2_scan120k'
ARMS=('uniform','guard15_equal_mu0','guard15_equal','exclusive_guard15_mu0','exclusive_guard15_mu2')
BUDGETS=(30000,60000,120000,240000)

def close(a,b,tol=1e-9):
    np.testing.assert_allclose(a,b,atol=tol,rtol=0)

def digest(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256()
    h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-manifest',action='store_true')
    args=parser.parse_args()
    if not args.skip_manifest:
        manifest=ROOT/'provenance/SHA256SUMS'
        for line in manifest.read_text().splitlines():
            expected,rel=line.split('  ',1)
            actual=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
            assert actual==expected,rel
    records={}; scan={};total=0
    for campaign,store,expected in [(EQ,records,320),(SCAN,scan,176)]:
        files=sorted((campaign/'results').glob('*/*/*.json'))
        assert len(files)==expected
        for path in files:
            d=json.loads(path.read_text());total+=1
            key=(d['system'],int(d['budget']),d['arm'],int(d['stream']))
            assert key not in store;store[key]=d
            assert d['status']=='optimal'
            assert sum(d['counts'])==d['budget']
            assert d['lambda_radius']==1
            assert d['mu_ftpbe']==(2 if d['arm'] in ('guard15_equal','exclusive_guard15_mu2') else 0)
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                assert z['d2'].shape==(120,120) and z['gamma'].shape==(16,16)
                close(np.trace(z['d2']),45,1e-6);close(np.trace(z['gamma']),10,1e-6)
                np.testing.assert_array_equal(z['counts'],d['counts'])
                np.testing.assert_array_equal(z['indices'],d['indices'])
                assert np.all(np.isfinite(z['values']))
            close(d['h_error_meh'],d['h_gamma_meh']+d['h_d2_meh'],1e-7)
            close(d['f_error_meh'],d['f_non_ontop_meh']+d['f_on_top_meh'],1e-7)
            assert d['minimum_dqg_eigenvalue']>-2e-6
            assert d['equality_residual']<1e-6 and d['contraction_error']<1e-6
            if d['arm']!='uniform':
                assert len(d['indices'])==15 and sum(d['fit_counts'])==d['budget']-7500
                pilot=campaign/'pilots'/d['system']/f'r{d["stream"]}.npz'
                with np.load(pilot,allow_pickle=False) as z:assert digest(z['samples'])==d['pilot_digest']
    matched=0
    for system in ('n2','co_eq'):
        for b in BUDGETS:
            for r in range(8):
                for a,z in [('guard15_equal','guard15_equal_mu0'),('exclusive_guard15_mu2','exclusive_guard15_mu0')]:
                    with np.load(EQ/'results'/system/a/f'b{b}_r{r}.npz') as x,np.load(EQ/'results'/system/z/f'b{b}_r{r}.npz') as y:
                        for k in ('values','indices','counts'):np.testing.assert_array_equal(x[k],y[k])
                    matched+=1
    ci=list(csv.DictReader((ROOT/'source-data/equilibrium_rmse_ci95.csv').open()))
    rngs={s:np.random.default_rng(20261006+i).integers(0,8,size=(100000,8)) for i,s in enumerate(('n2','co_eq'))}
    for row in ci:
        s,b,a=row['system'],int(row['budget']),row['arm']
        v=np.array([records[s,b,a,r][row['metric']] for r in range(8)])
        close(np.sqrt(np.mean(v*v)),float(row['rmse']))
        bounds=np.percentile(np.sqrt(np.mean(v[rngs[s]]**2,axis=1)),[2.5,97.5])
        close(bounds,[float(row['ci95_low']),float(row['ci95_high'])])
    sci=list(csv.DictReader((ROOT/'source-data/scan_signed_mean_ci95.csv').open()))
    for row in sci:
        bond=float(row['panel'].split('=')[1]);system=f'n2_r{round(bond*100):03d}'
        key='h_error_meh' if row['metric']=='H' else 'f_error_meh'
        v=np.array([scan[system,120000,row['group'],r][key] for r in range(8)])
        close(v.mean(),float(row['mean_error_mHa']))
        close(np.percentile(v[rngs['n2']].mean(axis=1),[2.5,97.5]),
              [float(row['ci95_low_mHa']),float(row['ci95_high_mHa'])])
    for arm in ('uniform','guard15_equal'):
        for r in range(8):
            with np.load(EQ/'results/n2'/arm/f'b120000_r{r}.npz') as x,np.load(SCAN/'results/n2_r110'/arm/f'b120000_r{r}.npz') as y:
                for k in x.files:np.testing.assert_array_equal(x[k],y[k])
    report=dict(result_records=total,equilibrium_records=320,scan_records=176,
                reused_scan_records=16,paired_objective_comparisons=matched,
                independently_reproduced_rmse_intervals=len(ci),
                independently_reproduced_scan_mean_intervals=len(sci),
                manifest_verified=not args.skip_manifest)
    out=ROOT/'generated';out.mkdir(exist_ok=True)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

