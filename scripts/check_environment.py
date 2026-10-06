"""Check imports and versions without running chemistry or a licensed solve."""
import importlib,importlib.metadata as md,runpy,sys,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mcpdft-release-mpl')
ROOT=Path(__file__).resolve().parents[1]
for name in ['numpy','scipy','pyscf','cvxpy','mosek','matplotlib','qiskit','qiskit_aer','qiskit_nature']:
    importlib.import_module(name)
from pyscf import mcpdft
print('Python',sys.version.split()[0])
for name in ['numpy','scipy','pyscf','cvxpy','Mosek','matplotlib','qiskit','qiskit-aer','qiskit-nature']:
    print(name,md.version(name))
runpy.run_path(str(ROOT/'analysis/guard15_equal_real30/experiment.py'),run_name='import_check')
local=[]
for m in list(sys.modules.values()):
    p=getattr(m,'__file__',None)
    if p:
        p=Path(p).resolve()
        if '/manuscript/analysis/' in str(p):raise RuntimeError(f'External workspace dependency: {p}')
        if p.is_relative_to(ROOT):local.append(str(p.relative_to(ROOT)))
print('Imported',len(set(local)),'repository modules; PySCF MC-PDFT available.')
print('MOSEK import checked; an SDP calculation additionally requires a valid local license.')
