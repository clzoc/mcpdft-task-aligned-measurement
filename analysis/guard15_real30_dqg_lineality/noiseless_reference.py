#!/usr/bin/env python3
"""Full-pool noiseless optimum of the unchanged lambda=1, mu=2 estimator."""
from core import HERE,save,engine
import argparse
from dataclasses import replace
import resource
import time
import numpy as np


def main(system):
    path=HERE/'references'/f'{system}_noiseless.json'
    if path.exists() and path.with_suffix('.npz').exists():return
    resource.setrlimit(resource.RLIMIT_AS,(20*1024**3,20*1024**3))
    start=time.time();e=engine();d,c=e.context(system)
    values=np.concatenate(c.exact_y)
    raw=d.band.all_rows_shadow(c,tuple(y[None,:] for y in c.exact_y),np.ones(30,dtype=int))
    raw=replace(raw,values=values,lower_bounds=values.copy(),upper_bounds=values.copy(),
                hits=np.full(len(values),-1),shots_per_basis=0)
    solver=e.MeritSolver(d,c,raw,np.ones((len(values),1)),threads=1)
    result=solver.solve(values,1.,2.)
    assert result.status=='optimal'
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path.with_suffix('.npz'),d2=result.d2,gamma=result.gamma,values=values)
    save(path,dict(system=system,oracle_reference=True,all_frames=30,noiseless=True,
                   lambda_radius=1.,mu_ftpbe=2.,status=result.status,
                   seconds=time.time()-start,**d.r.score(c,result)))
    print('NOISELESS_COMPLETE',system,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--system',choices=['n2','co_eq'],required=True)
    main(p.parse_args().system)
