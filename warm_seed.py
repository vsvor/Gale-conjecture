#!/usr/bin/env python3
"""Graph-screen and briefly warm up a generated seed, using the existing engine.

Probes several explicit ratios BEFORE binary search, to avoid optimizing a
nearly useless complete-graph witness when the first bisection query times out.
No UNKNOWN result is used as UNSAT. Output uses the tuner's exact-decimal path.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
os.environ['BORSUK_BLAS_THREADS']='1'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('start',type=Path)
    p.add_argument('--engine-dir',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--k',type=int)
    p.add_argument('--probes',default='.94,.92,.90,.88,.875,.87,.86,.84')
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--prune-checks',type=int,default=60)
    p.add_argument('--seed',type=int,default=125)
    p.add_argument('--backend',default='native',choices=['auto','native','python','external','pysat'])
    p.add_argument('--solver',default='kissat')
    p.add_argument('--sat-timeout',type=float,default=2.)
    p.add_argument('--pysat-solver',default='g4')
    a=p.parse_args(argv)
    if a.out.exists():p.error('Use a new output directory.')
    if min(a.rounds,a.steps,a.prune_checks)<0 or a.sat_timeout<0:p.error('Budgets must be nonnegative.')
    engine=a.engine_dir.expanduser().resolve()
    if not (engine/'tune_solution.py').is_file():p.error('Supply the directory containing the engine and tuner.')
    sys.path.insert(0,str(engine))
    import numpy as np
    import borsuk_search as bs
    import tune_solution as ts
    try:
        g,w,c,savedk=ts.load_source(a.start)
        k=a.k if a.k is not None else (savedk or g.dim+1)
        if not 1<=k<g.m:raise ValueError('Require 1 <= k < point count.')
        targets=[Fraction(s) for s in a.probes.split(',')]
        if any(not 0<t<1 for t in targets):raise ValueError('Probe ratios must lie strictly between 0 and 1.')
        # Full graph is a mathematically known base, NOT an assumed timed-out graph.
        w=np.arange(len(g.pairs),dtype=np.int32)
        c=np.zeros(g.m,dtype=np.int32)
        state=ts.make_state(g,w,c)
        pool=ts.OraclePool(k,a); log=[]
        a.out.mkdir(parents=True)
        rng=np.random.default_rng(a.seed)
        try:
            oracle=pool.get(g.m)
            for t in targets:
                if not state.lower2 < t*t <= state.upper2:
                    continue
                ids=g.threshold(t*t,inclusive=True)
                ans=oracle.solve(ids)
                log.append({'stage':'probe','target':str(t),'status':ans.status.value})
                if ans.status is bs.Status.UNSAT:
                    if min(g.q[int(i)] for i in ids)>state.lower2*g.diameter2:
                        w=ids
                elif ans.status is bs.Status.SAT and ans.colors is not None:
                    if g.partition_max(ans.colors)<g.partition_max(c):c=ans.colors
                state=ts.make_state(g,w,c,unresolved=ans.status is bs.Status.UNKNOWN)
            state=ts.assess(g,pool,witness=w,colors=c)
            log.append({'stage':'initial_assessment','lower':state.lower,'upper':state.upper,'resolved':state.exact})
            ts.save_state(a.out/'initial',state,k,{'probe_log':log.copy()})
            for r in range(a.rounds):
                ids=bs.prune_witness(pool.get(state.g.m),state.witness,
                                     bs.distances2(state.g.x,state.g.pairs),rng,
                                     checks=a.prune_checks,noise=.04)
                xx,_=bs.optimize_witness(state.g.x,state.g.pairs,ids,steps=a.steps,polish=0)
                candidate=ts.assess(ts.Geometry.from_float(xx),pool,witness=ids,colors=state.colors)
                accepted=candidate.lower2>state.upper2
                if accepted:state=candidate
                log.append({'stage':'warmup','round':r+1,'candidate_lower':candidate.lower,
                            'candidate_upper':candidate.upper,'accepted':accepted,
                            'retained_lower':state.lower,'retained_upper':state.upper})
                (a.out/'log.json').write_text(json.dumps(log,indent=2)+'\n')
            (a.out/'log.json').write_text(json.dumps(log,indent=2)+'\n')
            ts.save_state(a.out/'best',state,k,{'warm_seed':True,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'stages':log})
        finally:pool.close()
        print(json.dumps({'lower':state.lower,'upper':state.upper,'resolved':state.exact,
                          'directory':str(a.out/'best')},indent=2))
    except (ValueError,RuntimeError,OSError) as exc:p.error(str(exc))
    return 0

if __name__=='__main__':raise SystemExit(main())
