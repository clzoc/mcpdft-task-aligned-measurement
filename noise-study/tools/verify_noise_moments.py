"""Independently verify archived pair moments from one frame in every histogram file."""
from pathlib import Path
from itertools import combinations
import json, os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np

MS=Path(__file__).resolve().parents[1]
PKG=MS
model=json.loads((PKG/'circuits/calibration/wukong_path16.json').read_text())
z=np.arange(65536,dtype=np.uint32)
bits=((z[:,None]>>np.arange(16))&1).astype(float)
ind=np.column_stack([bits[:,i]*bits[:,j] for i,j in combinations(range(16),2)])
mask=(bits[:,:8].sum(1)==5)&(bits[:,8:].sum(1)==5)
max_difference=0.; checked=0; signed_sector_masses=[]
for folder in ['results','results_gate020']:
    for path in sorted((PKG/'code/mindquantum_poc'/folder).glob('*/*_histograms.npz')):
        arm=path.stem.removesuffix('_histograms')
        with np.load(path) as a:
            raw=a['noisy'][0].astype(float)/a['noisy'][0].sum()
            clean=a['noiseless'][0].astype(float)/a['noiseless'][0].sum()
        corrected=raw.copy()
        # Invert each two-outcome binary channel directly on paired entries.
        for q,r in enumerate(model['readout_per_qubit']):
            t=corrected.reshape(-1,2,2**q)
            total=t[:,0,:]+t[:,1,:]
            t[:,0,:]=(t[:,0,:]-r*total)/(1-2*r)
            t[:,1,:]=(t[:,1,:]-r*total)/(1-2*r)
        mass=corrected[mask].sum();assert mass>0
        signed_sector_masses.append(float(mass))
        conditioned=np.where(mask,corrected/mass,0.)
        for variant,p in [('raw',raw),('clean',clean),('rem_lin',corrected),('rem_lin_post',conditioned)]:
            expected=p@ind
            with np.load(path.parent/'variants'/f'{arm}_{variant}.npz') as a:
                saved=a['values'][:120]
            diff=float(np.max(np.abs(expected-saved)))
            assert diff<2e-12,(path,variant,diff)
            max_difference=max(max_difference,diff);checked+=1
out=MS.parent/'generated/noise-validation/moment-validation.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(frame_variant_blocks_checked=checked,max_absolute_moment_difference=max_difference,
    sector_mass_range=[min(signed_sector_masses),max(signed_sector_masses)],
    implementation='Independent pairwise channel inverse; all scales, geometries and arms, first sampled frame.'),indent=2)+'\n')
print(out.read_text())
