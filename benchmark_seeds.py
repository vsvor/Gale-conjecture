#!/usr/bin/env python3
"""Small fixed-budget comparison against the spherical initializer.

Run from the repository containing borsuk_search.py and tune_solution.py, or
supply --engine-dir. Does NOT assert larger cores or better final optima.
The original engine's incomplete-bracket behavior is retained, and recorded.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
os.environ['BORSUK_BLAS_THREADS']='1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-dir',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dimensions',default='5,6,7,8')
    p.add_argument('--seeds',default='124,125')
    p.add_argument('--backend',default='native',choices=['native','python','external','pysat','auto'])
    p.add_argument('--solver',default='kissat')
    p.add_argument('--sat-timeout',type=float,default=.8)
    a=p.parse_args()
    if a.out.exists():p.error('Use a new output directory.')
    dims=list(map(int,a.dimensions.split(',')));seeds=list(map(int,a.seeds.split(',')))
    if any(d not in (5,6,7,8) for d in dims):p.error('Dimensions must lie in 5..8.')
    engine=a.engine_dir.expanduser().resolve()
    if not (engine/'borsuk_search.py').exists():p.error('Engine module not found.')
    sys.path.insert(0,str(engine))
    import numpy as np
    import borsuk_search as bs
    import tune_solution as ts
    from simplex_product_seed import generate
    a.out.mkdir(parents=True)
    counts={5:70,6:90,7:110,8:130}; rows=[]
    for dim in dims:
        m=counts[dim]
        for seed in seeds:
            for method in ('sphere','product'):
                tic=time.monotonic();rng=np.random.default_rng(seed)
                x=bs.make_seed(dim,m,'sphere',rng) if method=='sphere' else generate(dim,m,rng)[0]
                x=bs.normalize(x)
                with bs.GraphOracle(m,dim+1,a.backend,a.solver,a.sat_timeout) as oracle:
                    best=bs.evaluate(x,oracle); initial=[best.lower,best.upper]; bestx=x.copy()
                    for _ in range(2):
                        ids=bs.prune_witness(oracle,best.witness,bs.distances2(bestx,oracle.pairs),rng,checks=60,noise=.04)
                        xx,_=bs.optimize_witness(bestx,oracle.pairs,ids,steps=20,polish=0)
                        new=bs.evaluate(xx,oracle)
                        if new.lower>best.lower:bestx,best=xx,new
                    opt=SimpleNamespace(backend=a.backend,solver=a.solver,sat_timeout=a.sat_timeout,pysat_solver='g4')
                    pool=ts.OraclePool(dim+1,opt)
                    try:
                        exact=ts.assess(ts.Geometry.from_float(bestx),pool,best.witness,best.colors)
                        folder=a.out/f'd{dim}_{method}_s{seed}'
                        np.save(folder.parent/f'd{dim}_{method}_s{seed}_initial.npy',x)
                        ts.save_state(folder,exact,dim+1,dict(pilot=True,warmup_rounds=2,steps_per_stage=20,
                                                           prune_checks_per_round=60,seed=seed,method=method))
                        row=dict(dim=dim,m=m,seed=seed,method=method,initial=initial,
                                 lower=exact.lower,upper=exact.upper,exact=exact.exact,
                                 elapsed=round(time.monotonic()-tic,3),oracle_calls=oracle.calls,
                                 unknowns=oracle.unknowns+pool.stats()['unknowns'])
                        rows.append(row)
                        (a.out/'pilot.json').write_text(json.dumps(rows,indent=2)+'\n')
                        print(json.dumps(row),flush=True)
                    finally:pool.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
