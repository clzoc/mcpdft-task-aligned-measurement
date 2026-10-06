"""Postprocess saved RDMs: molecular one-electron error for the MC-PDFT energy.
Rebuild only the two classical references using their recorded builders' settings.
Validate the orbital/RDM convention against all stored gamma and H1 errors.
No measurements, frame selections or SDP reconstructions are rerun.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import csv,json
import numpy as np
from pyscf import gto,scf,mcscf
ROOT=Path(__file__).resolve().parents[1]
EQ=ROOT/'analysis/guard15_equal_real30'
OUT=ROOT/'source-data'
rows=[];checks={}
for system,atom,sym in [('n2','N 0 0 -0.55; N 0 0 0.55','D2h'),('co_eq','C 0 0 -0.564; O 0 0 0.564','C2v')]:
 mol=gto.M(atom=atom,basis='cc-pvdz',charge=0,spin=0,unit='Angstrom',symmetry=sym,verbose=0)
 mf=scf.RHF(mol)
 if system=='co_eq':mf.conv_tol=1e-10;mf.max_cycle=100
 mf.kernel();assert mf.converged
 mo=mf.mo_coeff.copy()
 for j in range(mo.shape[1]):
  piv=np.flatnonzero(np.abs(mo[:,j])>1e-10)
  if len(piv) and mo[piv[0],j]<0:mo[:,j]*=-1
 mf.mo_coeff=mo
 mc=mcscf.CASCI(mf,8,10);mc.ncore=2;mc.kernel();assert mc.converged
 ga,gb=mc.fcisolver.make_rdm1s(mc.ci,8,(5,5))
 gamma=np.zeros((16,16));gamma[:8,:8]=ga;gamma[8:,8:]=gb
 hbare=mo[:,2:10].T@mf.get_hcore()@mo[:,2:10]
 heff,_=mc.get_h1eff(mo)
 h_error_max=0;g_error_max=0
 for path in sorted((EQ/'results'/system).glob('*/*.json')):
  d=json.loads(path.read_text())
  with np.load(path.with_suffix('.npz')) as z:dg=z['gamma']-gamma
  spin_sum=dg[:8,:8]+dg[8:,8:]
  effective=1000*float(np.sum(heff*spin_sum))
  bare=1000*float(np.sum(hbare*spin_sum))
  h_error_max=max(h_error_max,abs(effective-d['h_gamma_meh']))
  g_error_max=max(g_error_max,abs(np.linalg.norm(dg)-d['gamma_error']))
  rows.append(dict(system=system,arm=d['arm'],budget=d['budget'],stream=d['stream'],molecular_one_electron_error_mHa=bare))
 assert h_error_max<1e-4,(system,h_error_max)
 assert g_error_max<1e-7,(system,g_error_max)
 checks[system]=dict(max_effective_one_electron_validation_difference_mHa=h_error_max,max_gamma_norm_validation_difference=g_error_max,reference_casci_energy=float(mc.e_tot))
 print(system,checks[system],flush=True)
with (OUT/'molecular_one_electron_errors.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'one_electron_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
