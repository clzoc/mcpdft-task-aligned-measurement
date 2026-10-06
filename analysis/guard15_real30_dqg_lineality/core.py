import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/real30_lineality_mpl')
from pathlib import Path
import importlib.util
import json
import sys
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'guard15_equal_real30'
ORACLE=HERE.parent/'guard15_real30_oracle_selection'


def save(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def engine():
    spec=importlib.util.spec_from_file_location('lineality_original_engine',SOURCE/'experiment.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module
