"""Specialize the existing DQG builder without changing the original module.

The sole algebraic change is ||affine residual||_2 in place of its square.
All constraints, solver options, outputs and validation remain the same.
The exact generated function and vendor hash are archived for review.
"""
import inspect,hashlib,json
from pathlib import Path
import constrained_shadow as vendor

source=inspect.getsource(vendor.solve_dqg_sdp)
needle='''merit_function = cp.sum_squares(
            cp.multiply(1.0 / objective_scales, affine_residual)
        )'''
replacement='''merit_function = cp.norm(
            cp.multiply(1.0 / objective_scales, affine_residual), 2
        )'''
assert source.count(needle)==1
specialized=source.replace(needle,replacement)
namespace=dict(vars(vendor))
exec(compile(specialized,str(Path(__file__).with_name('square_root_generated.py')),'exec'),namespace)
solve_dqg_sdp=namespace['solve_dqg_sdp']
HERE=Path(__file__).resolve().parent
(HERE/'square_root_generated.txt').write_text(specialized)
(HERE/'square_root_source.json').write_text(json.dumps(dict(
    vendor_path=vendor.__file__,vendor_sha256=hashlib.sha256(Path(vendor.__file__).read_bytes()).hexdigest(),
    function_sha256=hashlib.sha256(specialized.encode()).hexdigest(),replacements=1),indent=2)+'\n')
