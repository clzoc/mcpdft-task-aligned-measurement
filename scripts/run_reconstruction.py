"""Run one fresh reconstruction using original kernels and a separate output tree."""
from pathlib import Path
import argparse,importlib.util,shutil,sys

ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m
    spec.loader.exec_module(m)
    return m

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('campaign',choices=['equilibrium','scan'])
    p.add_argument('--system',required=True)
    p.add_argument('--arm',required=True,choices=['uniform','guard15_equal','guard15_equal_mu0','exclusive_guard15_mu0','exclusive_guard15_mu2'])
    p.add_argument('--budget',type=int,default=120000,choices=[30000,60000,120000,240000])
    p.add_argument('--stream',type=int,default=0,choices=range(8))
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--output',type=Path,default=ROOT/'runs/reproduction')
    a=p.parse_args();base=a.output.resolve()
    if base==ROOT or base.is_relative_to(ROOT/'analysis'):
        p.error('Choose an output outside the archived analysis tree.')
    name='guard15_equal_real30' if a.campaign=='equilibrium' else 'guard15_equal_real30_n2_scan120k'
    archive=ROOT/'analysis'/name;dest=base/name
    dest.mkdir(parents=True,exist_ok=True)
    shutil.copytree(archive/'contexts',dest/'contexts',dirs_exist_ok=True)
    shutil.copy2(archive/'pool_real30.npz',dest/'pool_real30.npz')
    if a.campaign=='equilibrium':
        if a.system not in ('n2','co_eq'):p.error('Equilibrium system must be n2 or co_eq.')
        engine=load('release_equilibrium',archive/'experiment.py')
        engine.HERE=dest
    else:
        scan=load('release_scan',archive/'scan.py')
        if a.system not in scan.SYSTEMS:p.error('Unknown scan geometry tag.')
        if a.budget!=120000 or a.arm not in ('uniform','guard15_equal'):
            p.error('Scan uses 120000 shots with uniform or guard15_equal.')
        if a.system=='n2_r110':
            source=base/'guard15_equal_real30/results/n2'/a.arm/f'b120000_r{a.stream}.json'
            if not source.exists():p.error('Recalculate the corresponding equilibrium 120k record first; 1.10 A is reused, not solved again.')
            import json,hashlib
            target=dest/'results/n2_r110'/a.arm/source.name;target.parent.mkdir(parents=True,exist_ok=True)
            d=json.loads(source.read_text());d.update(system='n2_r110',bond_angstrom=1.10,source_system='n2',no_new_solve=True,reused_from=str(source),source_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_array_sha256=hashlib.sha256(source.with_suffix('.npz').read_bytes()).hexdigest())
            target.write_text(json.dumps(d,indent=2)+'\n');shutil.copy2(source.with_suffix('.npz'),target.with_suffix('.npz'))
            print('Reused equilibrium record:',target);return
        scan.HERE=dest;scan.e.HERE=dest;engine=scan.e
    engine.worker(a.system,a.arm,a.stream,a.threads,a.budget)
    print('Result:',dest/'results'/a.system/a.arm/f'b{a.budget}_r{a.stream}.json')

if __name__=='__main__':main()

