"""Run the original noise driver in a separate tree without overwriting archived data."""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'noise-study/code'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['sample','solve','summarize'])
    parser.add_argument('--tag', choices=[f'n2_r{x:03d}' for x in [80,90,100,110,125,145,160,180,200,220,250]])
    parser.add_argument('--gate-scale', type=float, choices=[0.2,1.0], default=0.2)
    parser.add_argument('--scales', default='0.2,1.0')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/noise-reproduction')
    args = parser.parse_args()
    if args.command != 'summarize' and args.tag is None:
        parser.error('--tag is required for sampling and fitting')
    output = args.output.resolve()
    if output == ROOT or output.is_relative_to(SOURCE) or output == ROOT/'noise-study':
        parser.error('choose a separate output directory')
    code = output/'code'
    marker = output/'noise-reproduction.json'
    if not marker.exists():
        if code.exists():
            parser.error('output contains an unrecognized code tree; choose a new directory')
        def ignore(directory, names):
            skipped = {'__pycache__'}
            if Path(directory).name == 'mindquantum_poc':
                skipped.update(n for n in names if n.startswith('results'))
            return skipped.intersection(names)
        output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SOURCE, code, ignore=ignore)
        marker.write_text(json.dumps({'source':str(SOURCE),'fixed_plans':True,'archived_noise_results_copied':False},indent=2)+'\n')
    command = [sys.executable,str(code/'mindquantum_poc/run_scan.py'),args.command]
    if args.tag:
        command += ['--tag',args.tag]
    command += ['--gate-scale',str(args.gate_scale),'--scales',args.scales]
    env = dict(os.environ)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[key] = '1'
    subprocess.run(command,cwd=output,env=env,check=True)

if __name__ == '__main__':
    main()
