#!/usr/bin/env python3
"""Search two successive essential splits, including first splits with zero LP gain.
Exploratory auxiliary seeds may have coincident copies; exported states may not.
Requires the accompanying borsuk_search.py, tune_solution.py, split_tune.py.
"""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
from pathlib import Path
import argparse,itertools,json,time
import numpy as np
import borsuk_search as bs
import tune_solution as ts
import split_tune as st

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('start',type=Path);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--seed',type=int,default=739);p.add_argument('--max-omit',type=int,default=3)
 p.add_argument('--prune',type=int,default=0);p.add_argument('--limit',type=int,default=0)
 p.add_argument('--first-limit',type=int,default=0);p.add_argument('--zero-trials',type=int,default=24)
 p.add_argument('--jitter',type=float,default=0.01);p.add_argument('--sat-timeout',type=float,default=2.)
 p.add_argument('--backend',default='native');p.add_argument('--solver',default='kissat')
 a=p.parse_args()
 _,_,_,saved_k=ts.load_source(a.start)
 if saved_k is not None and saved_k != 4:
  p.error('paired_splits.py currently supports k=4 only; this saved input declares a different k.')
 a.out.mkdir(exist_ok=False,parents=True)
 args=ts.build_parser().parse_args([str(a.start),'--backend',a.backend,'--solver',a.solver,'--sat-timeout',str(a.sat_timeout)])
 args.full_constraints=False;args.slsqp_iters=350
 pool=ts.OraclePool(4,args);rng=np.random.default_rng(a.seed);start=time.monotonic();stats={}
 def log(event,**data):
  data={'event':event,'seconds':round(time.monotonic()-start,3),**data}
  with (a.out/'progress.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
  print(json.dumps(data),flush=True)
 def choices(x,ids,min_vertex=0,max_vertex=None):
  m=len(x);pairs=bs.pairs_for(m);oracle=pool.get(m);neworacle=pool.get(m+1)
  q=bs.distances2(x,pairs);active=ids[q[ids]<=q[ids].min()+1e-8];critical=set()
  for e in active:
   ans=oracle.solve(ids[ids!=e]);stats['critical_'+ans.status.value]=stats.get('critical_'+ans.status.value,0)+1
   if ans.status is bs.Status.SAT:critical.add(int(e))
  for v in range(min_vertex,m if max_vertex is None else max_vertex):
   neighbors=[];essential=set()
   for e in active:
    u,z=map(int,pairs[e])
    if v in (u,z):
     other=z if v==u else u;neighbors.append(other)
     if int(e) in critical:essential.add(other)
   for left,right in st.omission_pairs(neighbors,essential,a.max_omit):
    wid=st.split_edge_ids(m,pairs[ids],v,left,right)
    ans=neworacle.solve(wid);stats['split_'+ans.status.value]=stats.get('split_'+ans.status.value,0)+1
    yield (ans.status,v,left,right,wid)
 try:
  g,w,c,k=ts.load_source(a.start);s=ts.assess(g,pool,w,c);best=s
  ts.save_state(a.out/'input',s,4,{})
  x=bs.normalize(g.x);ids=g.threshold(s.lower2)
  if a.prune:ids=bs.prune_witness(pool.get(g.m),ids,bs.distances2(x,g.pairs),rng,checks=a.prune,noise=.04)
  first=[(v,l,r,w) for status,v,l,r,w in choices(x,ids) if status is bs.Status.UNSAT]
  rng.shuffle(first)
  if a.first_limit:first=first[:a.first_limit]
  log('first_splits',number=len(first),stats=stats)
  zero=[];tested=0;positive=0;optimized=0
  def optimize(y,w,meta):
   nonlocal best,optimized
   z,report=ts.slsqp_once(y,w,args,args.ftol);optimized+=1
   ratio=bs.witness_ratio(z,bs.pairs_for(len(z)),w)
   data={**meta,'ratio':ratio,'slsqp':report}
   if ratio>best.lower+1e-10:
    try:trial=ts.assess(ts.Geometry.from_float(bs.normalize(z)),pool,w,witness_known=True)
    except ValueError:return
    if trial.lower2>best.upper2:
     best=trial;folder=a.out/f'improvement_{optimized:03d}'
     ts.save_state(folder,trial,4,data);ts.save_state(a.out/'best',trial,4,data)
     np.savetxt(folder/'optimization_edges.csv',bs.pairs_for(len(y))[w],delimiter=',',fmt='%d');np.save(folder/'optimization_seed.npy',y)
     log('improvement',ratio=best.lower,points=len(y),trial=optimized)
   log('optimized',**data)
  for index,(v,left,right,w1) in enumerate(first):
   y=np.vstack((x,x[v]))
   for status,v2,left2,right2,w2 in choices(y,w1,min_vertex=v+1,max_vertex=g.m):
    tested+=1
    if status is bs.Status.UNSAT:
     yy=np.vstack((y,y[v2]));audit,_=ts.linearized_audit(yy,w2)
     meta={'first_vertex':v,'first_left':left,'first_right':right,'second_vertex':v2,'second_left':left2,'second_right':right2,'audit':audit}
     if audit['success'] and audit['linearized_gain']>1e-7:
      positive+=1;optimize(yy,w2,meta)
     elif audit['success']:
      if len(zero)<a.zero_trials:zero.append((yy,w2,meta))
      else:
       j=int(rng.integers(max(1,stats.get('split_UNSAT',1))))
       if j<a.zero_trials:zero[j]=(yy,w2,meta)
    if a.limit and tested>=a.limit:break
   log('first_done',index=index+1,first_vertex=v,pair_tests=tested,positive_LP=positive,best=best.lower,stats=stats)
   if a.limit and tested>=a.limit:break
  for yy,w2,meta in zero:
   yy=bs.normalize(yy+rng.normal(size=yy.shape)*a.jitter)
   optimize(yy,w2,{**meta,'zero_LP_jitter':a.jitter})
  summary={'initial_lower':s.lower,'best_lower':best.lower,'best_upper':best.upper,'best_points':best.g.m,'pair_tests':tested,'positive_LP':positive,'optimized':optimized,'stats':stats,'oracle':pool.stats(),'seconds':time.monotonic()-start}
  (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
 finally:pool.close()
if __name__=='__main__':main()
