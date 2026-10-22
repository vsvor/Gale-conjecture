#!/usr/bin/env python3
"""Fit -> diameter-safe resample -> free-coordinate optimize -> refit.

Exceptions are preserved as vertices. Models guide sampling only. The incumbent
is separate from the beam of exploratory descendants. All best replacements
are compared using exact squared bounds on canonical decimal coordinates.

Examples:
  python distill_search.py search --out run --backend native
  python distill_search.py search run --out continuation --cycles 5
  python distill_search.py verify run/best --backend python

Python >=3.10, NumPy/SciPy, borsuk_search.py and tune_solution.py are required.
"""
from __future__ import annotations

# The imported engine also sets these, before importing numpy.
import os
for _v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_v]=os.environ.get('BORSUK_BLAS_THREADS','1')
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import borsuk_search as bs
import tune_solution as ts
import curve_model as cm

ROOT=Path(__file__).resolve().parent
VERSION='1.0'


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def publish(source: Path, destination: Path) -> None:
    """Two-rename checkpoint publication. Immutable source remains recoverable."""
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_name('.'+destination.name+'.new');old=destination.with_name(destination.name+'.previous')
    if tmp.exists():shutil.rmtree(tmp)
    shutil.copytree(source,tmp)
    if old.exists():shutil.rmtree(old)
    if destination.exists():destination.rename(old)
    tmp.rename(destination)


def bounds_dict(s: ts.State) -> dict:
    return {'minimum_separation_over_diameter':math.sqrt(float(Fraction(min(s.g.q),s.g.diameter2))),
            'points':s.g.m,'lower':s.lower,'upper':s.upper,'exact':s.exact,
            'lower_squared':[str(s.lower2.numerator),str(s.lower2.denominator)],
            'upper_squared':[str(s.upper2.numerator),str(s.upper2.denominator)],
            'coordinates_sha256':cm.digest(s.g)}


def strictly_improves(new: ts.State, old: ts.State | None, min_gain: float) -> bool:
    return old is None or (new.lower2 > old.upper2 and new.lower-old.upper >= min_gain)


def local_args(config: dict) -> argparse.Namespace:
    a=ts.build_parser().parse_args(['unused'])
    for key in ('backend','solver','sat_timeout','pysat_solver','steps','slsqp_iters',
                'polish_passes','full_constraints','ftol'):
        if key in config:setattr(a,key,config[key])
    a.constraint_passes=8;a.feasibility_tol=1e-9;a.ratio_tol=1e-10
    return a


def read_state(path: Path, pool: ts.OraclePool) -> ts.State:
    g,w,c,k=ts.load_source(path)
    if k is not None and k!=pool.k:raise ValueError('Saved k differs from requested k.')
    return ts.assess(g,pool,w,c,witness_known=False)


def read_lineage(path: Path) -> tuple[dict,list[int]]:
    model=json.loads((path/'model.json').read_text())
    meta=json.loads((path/'lineage.json').read_text())
    protected=list(map(int,meta['protected_exceptions']))
    if len(model.get('labels',[]))!=len(model['reference']):
        raise ValueError('Model labels/reference row counts disagree.')
    if len(protected)!=len(set(protected)) or any(i<0 or i>=len(model['reference']) for i in protected):
        raise ValueError('Invalid exception lineage.')
    geometry,_,_,_=ts.load_source(path)
    if meta['coordinates_sha256']!=cm.digest(geometry) or not np.array_equal(np.asarray(model['reference']),geometry.x):
        raise ValueError('Checkpoint coordinates/lineage mismatch. Do not reorder or edit checkpoint rows; start a fresh grouped run instead.')
    return model,protected


def save_node(path: Path,s: ts.State,model: dict,protected: list[int],metadata: dict,
              fit: dict | None=None) -> None:
    if path.exists():raise ValueError(f'Refusing to overwrite a node: {path}')
    tmp=path.with_name('.'+path.name+'.partial')
    if tmp.exists():shutil.rmtree(tmp)
    ts.save_state(tmp,s,int(metadata['k']),metadata)
    write_json(tmp/'model.json',model)
    write_json(tmp/'geometry_diagnostics.json',{
        'minimum_separation_over_diameter':math.sqrt(float(Fraction(min(s.g.q),s.g.diameter2))),
        'pairs_closer_than_1e_minus8_D':[list(map(int,s.g.pairs[i])) for i,q in enumerate(s.g.q) if q*10**16<s.g.diameter2],
        'qualification':'Insertion spacing does not constrain subsequent free optimization; no automatic merging is performed.'})
    write_json(tmp/'lineage.json',{**metadata,'version':VERSION,'protected_exceptions':protected,
               'coordinates_sha256':cm.digest(s.g),'rows_never_reordered_or_deleted':True,
               'original_point_count':metadata.get('original_point_count',len(model['reference']))})
    if fit is not None:write_json(tmp/'fit.json',fit)
    path.parent.mkdir(parents=True,exist_ok=True);tmp.rename(path)


def circle_from_group(x: np.ndarray,name: str) -> dict:
    if len(x)<3:raise ValueError(f'Group {name} needs at least three points.')
    center=x.mean(axis=0);_,ss,vt=np.linalg.svd(x-center,full_matrices=False)
    if len(ss)<2 or ss[1]<1e-8*max(ss[0],1e-12):raise ValueError(f'Group {name} is essentially collinear.')
    u,v=vt[:2];xy=(x-center)@vt[:2].T
    a=np.column_stack((2*xy,np.ones(len(x))));z=np.linalg.lstsq(a,np.sum(xy**2,axis=1),rcond=None)[0]
    center=center+z[0]*u+z[1]*v;r2=float(z[2]+z[0]**2+z[1]**2)
    if r2<=0:raise ValueError(f'No nondegenerate circle initializer for {name}.')
    angles=np.sort(np.mod(np.arctan2((x-center)@v,(x-center)@u),2*np.pi))
    gaps=np.diff(np.r_[angles,angles[0]+2*np.pi]);j=int(gaps.argmax())
    lo=float(angles[(j+1)%len(x)]);hi=lo+float(2*np.pi-gaps[j])
    return {'name':name,'kind':'circle','center':center.tolist(),'u':u.tolist(),'v':v.tolist(),
            'radius':math.sqrt(r2),'interval':[lo,hi],'support':len(x),'mirror':None}


def initialize_model(g: ts.Geometry,args) -> tuple[dict,list[int]]:
    if args.groups:
        data=json.loads(args.groups.read_text());curves=[];used=set()
        for name,indices in data['groups'].items():
            indices=list(map(int,indices))
            if len(set(indices))!=len(indices) or any(i<0 or i>=g.m for i in indices):
                raise ValueError(f'Invalid indices in group {name}.')
            curves.append(circle_from_group(g.x[indices],name));used.update(indices)
        if not curves:raise ValueError('No curve groups supplied.')
        protected=sorted(set(map(int,data.get('exceptions',[]))) | (set(range(g.m))-used))
        if any(i<0 or i>=g.m for i in protected):raise ValueError('Invalid exception index.')
        return {'curves':curves,'reference':g.x.tolist(),'protected':protected,'symmetry':None},protected
    source=args.template or ROOT/'templates'/f'{args.branch_kind}.json'
    model=json.loads(source.read_text());ref=np.asarray(model['reference'])
    if g.m<len(ref):raise ValueError('This template needs its original rows; use --groups for other configurations.')
    aligned=cm.transform(model,g.x)
    D=math.sqrt(float(bs.distances2(g.x,g.pairs).max()))
    # Template indices are only used under an explicit ordered-geometry check.
    if aligned['registration_rms']> .02*D:
        raise ValueError('Input does not match the ordered 81-point template (RMS > .02 D). '
                         'For another ordering/configuration, provide --groups groups.json; '
                         'resume descendants from their checkpoint directory.')
    return aligned,list(map(int,model['protected']))


def run_local(job: dict) -> dict:
    """Separate process entry. Only this job's node directories are written."""
    config=job['config'];args=local_args(config);pool=ts.OraclePool(job['k'],args)
    folder=Path(job['destination']);folder.parent.mkdir(parents=True,exist_ok=True)
    seed_path=Path(job['source']);model,protected=read_lineage(seed_path)
    log_path=folder.with_suffix('.progress.jsonl');records=[];rng=np.random.default_rng(job['seed'])
    def log(event,**data):
        payload={'event':event,**data}
        with log_path.open('a') as f:f.write(json.dumps(payload,default=str)+'\n')
    try:
        best=read_state(seed_path,pool)
        high_seen=best.upper;completed=0;stopped=False
        start_lower=best.lower
        for round_no in range(config['local_rounds']):
            current=best
            if not len(current.witness):
                log('local_stop',reason='No positive non-colorable witness for this point count.')
                break
            slack=Fraction(config['slacks'][round_no%len(config['slacks'])])
            threshold=current.lower2*(1-slack)**2
            ids=current.g.threshold(threshold)
            # This graph contains a checked UNSAT witness. Only proven UNSAT
            # deletions in prune_witness can remove constraints from it.
            q=bs.distances2(current.g.x,current.g.pairs)
            ids=bs.prune_witness(pool.get(current.g.m),ids,q,rng,
                                 checks=config['prune_checks'],noise=.04)
            x=current.g.x.copy()
            if config['opt_jitter']>0 and (round_no>0 or job.get('control',False)):
                D=math.sqrt(float(q.max()));x+=rng.normal(0,config['opt_jitter']*D,x.shape)
            if config['steps']:
                x,_=bs.optimize_witness(x,current.g.pairs,ids,steps=config['steps'],polish=0,
                                         powers=(32.,128.,512.,2048.))
            polishing=[]
            for pass_no in range(config['polish_passes']):
                x,r=ts.slsqp_once(x,ids,args,args.ftol);polishing.append(r)
            try:
                ng=ts.Geometry.from_float(bs.normalize(x))
            except ValueError as exc:
                log('trial_rejected',round=round_no+1,reason=str(exc));continue
            new=ts.assess(ng,pool,ids,current.colors,witness_known=True)
            high_seen=max(high_seen,new.upper)
            # Local baseline is never lost to a bad kick or coordinate rounding.
            retained=bool(new.lower2>=best.upper2)
            if retained:best=new
            record={'round':round_no+1,'slack':str(slack),'pruned_edges':len(ids),
                    'trial':bounds_dict(new),'retained':retained,'polishing':polishing}
            records.append(record);completed+=1
            log('local_round',round=round_no+1,lower=best.lower,upper=best.upper,retained=retained)
            if completed>=config['early_after'] and high_seen<job['record_lower']-config['early_gap']:
                stopped=True;break
        best.polish={'status':'budgeted_local_search','rounds':records,
                     'qualification':'Repeated local calls, not a stationarity or global optimality certificate.'}
        # Geometric refitting occurs AFTER optimization, against current points.
        model,fit=cm.refit(model,best.g.x,protected,config['fit_tol'],config['fit_iters'],
                            config['symmetry'],config['symmetry_tol'])
        old_meta=json.loads((seed_path/'lineage.json').read_text())
        meta={**old_meta,'k':job['k'],'stage':'optimized','parent':str(seed_path),
              'seed':job['seed'],'generation':job['generation']}
        save_node(folder,best,model,protected,meta,fit)
        report={'destination':str(folder),'completed_rounds':completed,'early_stopped':stopped,
                'start_lower':start_lower,'result':bounds_dict(best),'fit_inliers':fit['inliers'],
                'exceptions':fit['exceptions'],'solver':pool.stats()}
        write_json(folder/'worker.json',report)
        return report
    finally:pool.close()


def choose_beam(paths: list[Path],states: dict[str,ts.State],width: int,gap: float,
                rng: np.random.Generator) -> list[Path]:
    """One exploit slot, then near-best diverse/young exploratory descendants.

    Neutral additions are allowed into the beam; best/ remains a separate object.
    """
    unique={}
    for p in paths:
        s=states[str(p)];key=cm.digest(s.g)
        # Last path wins for identical arrays, so newer model fits are retained.
        previous=unique.get(key)
        if previous is None:
            unique[key]=p
        else:
            def freshness(q):
                return (int(json.loads((q/'lineage.json').read_text()).get('generation',0)),str(q))
            if freshness(p)>freshness(previous):unique[key]=p
    ps=list(unique.values())
    top=max(states[str(p)].lower for p in ps)
    eligible=[p for p in ps if states[str(p)].upper>=top-gap]
    def order(p):
        s=states[str(p)];meta=json.loads((p/'lineage.json').read_text())
        # Resolve tiny numerical differences only for EXPLORATORY selection.
        return (round(s.lower,9),int(meta.get('generation',0)),s.g.m,str(p))
    eligible.sort(key=order,reverse=True)
    chosen=[eligible.pop(0)]
    while eligible and len(chosen)<width:
        # Among a small top band, randomize to keep near-best branches alive.
        window=eligible[:max(4,2*width)]
        ages=np.array([1+max(0,int(json.loads((p/'lineage.json').read_text()).get('generation',0))) for p in window],float)
        idx=int(rng.choice(len(window),p=ages/ages.sum()))
        chosen.append(eligible.pop(idx))
    return chosen


def verify(path: Path,args) -> dict:
    if path.is_dir() and not (path/'points.csv').exists() and (path/'best').exists():path=path/'best'
    out=bs.verify_candidate(path,args.backend,args.solver,args.sat_timeout,args.pysat_solver)
    if (path/'lineage.json').exists():
        g,_,_,_=ts.load_source(path);model,protected=read_lineage(path)
        meta=json.loads((path/'lineage.json').read_text())
        if meta['coordinates_sha256']!=cm.digest(g):raise ValueError('Lineage coordinate hash mismatch.')
        if len(model['reference'])!=g.m:raise ValueError('Model row count differs from geometry.')
        if not np.array_equal(np.asarray(model['reference']),g.x):
            raise ValueError('Model reference does not match saved geometry.')
        if any(model['labels'][i]>=0 for i in protected):raise ValueError('A protected exception was fitted.')
        out['protected_exceptions']=protected;out['lineage_checked']=True
    return out


def run_search(args) -> dict:
    if args.out.exists():raise ValueError('Use a new --out directory; resume by passing the OLD run as start.')
    args.out.mkdir(parents=True)
    t0=time.monotonic();config=vars(args).copy();config={k:v for k,v in config.items() if k not in ('out','start','template','groups','command')}
    config['slacks']=[str(Fraction(v)) for v in args.slacks.split(',')]
    local=local_args(config);pool=ts.OraclePool(args.k,local)
    states={};beam=[];best_path=None;best=None;generation0=0
    def log(event,**data):
        entry={'event':event,'elapsed_seconds':round(time.monotonic()-t0,3),**data}
        with (args.out/'progress.jsonl').open('a') as f:f.write(json.dumps(entry,default=str)+'\n')
        print(json.dumps(entry,default=str),flush=True)
    def consider(p):
        nonlocal best,best_path
        s=states[str(p)]
        same=best is not None and cm.digest(s.g)==cm.digest(best.g)
        improve=strictly_improves(s,best,args.min_gain)
        tighten=same and (s.lower2>best.lower2 or s.upper2<best.upper2)
        if improve or tighten:
            old=best;best=s;best_path=p;publish(p,args.out/'best')
            log('best_saved',**bounds_dict(s),previous_upper=old.upper if old else None)
    def manifest(generation,pending=False):
        write_json(args.out/'manifest.json',{'version':VERSION,'generation':generation,'generation_pending':pending,
                    'seed':args.seed,'k':args.k,'best':str(best_path.relative_to(args.out)),
                    'beam':[str(p.relative_to(args.out)) for p in beam],
                    'max_points':args.max_points,'settings':config})
    try:
        if (args.start/'manifest.json').is_file():
            old=json.loads((args.start/'manifest.json').read_text())
            if int(old['k'])!=args.k:raise ValueError('Resume k mismatch.')
            generation0=int(old['generation']);args.seed=old['seed'] if args.seed is None else args.seed
            sources=[args.start/p for p in old['beam']]
            certified=args.start/old['best']
            sources=list(dict.fromkeys([certified]+sources))
            for i,p in enumerate(sources):
                s=read_state(p,pool);m,protected=read_lineage(p)
                m,fit=cm.refit(m,s.g.x,protected,args.fit_tol,args.fit_iters,args.symmetry,args.symmetry_tol)
                target=args.out/'nodes'/f'resume_{i:03d}'
                meta=json.loads((p/'lineage.json').read_text());meta.update(k=args.k,resumed_from=str(p))
                save_node(target,s,m,protected,meta,fit);states[str(target)]=s;consider(target)
                if p==certified or p in [args.start/q for q in old['beam']]:beam.append(target)
            beam=choose_beam(beam,states,args.beam,args.beam_gap,np.random.default_rng(args.seed+generation0))
            if old.get('generation_pending'):log('resume_warning',message='Old generation was incomplete; continuing from its last committed beam and saved best.')
        else:
            args.seed=20260910 if args.seed is None else args.seed
            s=read_state(args.start,pool)
            if s.g.dim!=3:raise ValueError('This geometric distillation program currently supports R^3 only.')
            if (args.start/'model.json').exists():m,protected=read_lineage(args.start)
            else:m,protected=initialize_model(s.g,args)
            m,fit=cm.refit(m,s.g.x,protected,args.fit_tol,args.fit_iters,args.symmetry,args.symmetry_tol)
            target=args.out/'nodes'/'initial'
            save_node(target,s,m,protected,{'k':args.k,'generation':0,'original_point_count':s.g.m,'stage':'initial'},fit)
            states[str(target)]=s;beam=[target];consider(target)
            log('initial_fit',inliers=fit['inliers'],protected=protected,exceptions=fit['exceptions'])
        largest=max(states[str(p)].g.m for p in beam)
        if args.max_points is None:args.max_points=max(201,largest+args.cycles*args.add)
        if args.max_points<largest:raise ValueError('--max-points is smaller than the retained beam.')
        config.update(seed=args.seed,max_points=args.max_points)
        write_json(args.out/'settings.json',{**vars(args),'source_scope':'existing point lineage; no point deletions or fitting projection'})
        rng=np.random.default_rng(args.seed+generation0)
        manifest(generation0)
        for generation in range(generation0+1,generation0+args.cycles+1):
            log('generation_start',generation=generation,parents=len(beam),record=best.lower)
            jobs=[];candidates=list(beam)+[best_path]
            for parent_no,parent in enumerate(beam):
                source=states[str(parent)];model,protected=read_lineage(parent)
                for child in range(args.children):
                    control=args.children>1 and child==args.children-1
                    add=0 if control else min(args.add,args.max_points-source.g.m)
                    seed=int(np.random.SeedSequence([args.seed,generation,parent_no,child]).generate_state(1)[0])
                    ng,nm,ins=cm.resample(source,model,add,np.random.default_rng(seed),
                          args.jitter*(1+child),args.exception_fraction,args.min_separation,
                          respect_symmetry=not(args.break_symmetry_every>0 and generation%args.break_symmetry_every==0 and child==0))
                    colors=cm.greedy_partition(ng,source.colors,args.k)
                    w=ts.edge_ids(ng.m,source.g.pairs[source.witness])
                    # No solver necessary for the inherited witness: original
                    # coordinates and all old pair distances are unchanged.
                    ns=ts.make_state(ng,w,colors,unresolved=source.unresolved)
                    d=args.out/'generations'/f'{generation:04d}'/f'p{parent_no:02d}_c{child:02d}'
                    seedpath=d/'seed'
                    meta0=json.loads((parent/'lineage.json').read_text())
                    meta={**meta0,'k':args.k,'generation':generation,'parent':str(parent),
                          'seed':seed,'stage':'resampled','control':control}
                    # Seed model reference is current; relabel safely without a fit.
                    D=math.sqrt(float(bs.distances2(ng.x,ng.pairs).max()))
                    labs,params,err=cm.labels_for(nm,ng.x,protected,args.fit_tol,D)
                    nm.update(labels=labs.tolist(),parameters=params.tolist(),reference=ng.x.tolist())
                    save_node(seedpath,ns,nm,protected,meta);write_json(seedpath/'insertion.json',ins)
                    states[str(seedpath)]=ns;consider(seedpath)
                    log('seed_created',generation=generation,parent=parent_no,child=child,points=ng.m,
                        added=ins['added'],diameter_exactly_unchanged=ins['diameter_exactly_unchanged'],control=control)
                    jobs.append({'source':str(seedpath),'destination':str(d/'optimized'),
                                 'config':config,'seed':seed,'k':args.k,'generation':generation,
                                 'record_lower':best.lower,'control':control})
            # Save useful seed/record updates even if interrupted inside a worker.
            manifest(generation-1,pending=True)
            if args.workers==1:
                iterator=(run_local(job) for job in jobs)
                executor=None
            else:
                executor=ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'))
                futures=[executor.submit(run_local,job) for job in jobs]
                iterator=(f.result() for f in as_completed(futures))
            try:
                for result in iterator:
                    p=Path(result['destination']);s=read_state(p,pool);states[str(p)]=s
                    candidates.append(p);consider(p)
                    log('child_complete',**result)
                    manifest(generation-1,pending=True)
            finally:
                if executor is not None:executor.shutdown(wait=True,cancel_futures=True)
            beam=choose_beam(candidates,states,args.beam,args.beam_gap,np.random.default_rng(args.seed+generation))
            publish(beam[0],args.out/'latest')
            manifest(generation)
            log('generation_complete',generation=generation,beam=[str(p.relative_to(args.out)) for p in beam],
                record_lower=best.lower,record_upper=best.upper)
        if args.final_polish_passes and len(best.witness):
            a=local_args(config);a.steps=0;a.polish_passes=args.final_polish_passes
            a.full_constraints=True;a.slsqp_iters=args.final_slsqp_iters;a.refreshes=2
            a.polish_kicks=1;a.stall_passes=2
            p=args.out/'final_polish';model,protected=read_lineage(best_path)
            log('final_polish_start',points=best.g.m)
            polished=ts.local_tune(best.g,best.witness,pool,np.random.default_rng(args.seed+999),a,
                                   lambda event,**d:log('final_'+event,**d),smooth=False,colors=best.colors)
            model,fit=cm.refit(model,polished.g.x,protected,args.fit_tol,args.fit_iters,args.symmetry,args.symmetry_tol)
            meta=json.loads((best_path/'lineage.json').read_text());meta.update(k=args.k,stage='final_polish',generation=generation0+args.cycles)
            save_node(p,polished,model,protected,meta,fit);states[str(p)]=polished;consider(p)
            manifest(generation0+args.cycles)
        summary={'best':bounds_dict(best),'best_directory':str((args.out/'best').resolve()),
                 'cycles_completed':args.cycles,'beam':[str(p.relative_to(args.out)) for p in beam],
                 'elapsed_seconds':time.monotonic()-t0,'solver':pool.stats(),
                 'qualification':'Heuristic search; graph UNSAT results are not independently proof-logged. Fitting never deletes/projects observations.'}
        write_json(args.out/'summary.json',summary);log('finished',**summary);return summary
    finally:pool.close()


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    subs=p.add_subparsers(dest='command',required=True)
    s=subs.add_parser('search',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    s.add_argument('start',nargs='?',type=Path,default=ROOT/'best_known/3d_81p_088871')
    s.add_argument('--out',type=Path,required=True)
    s.add_argument('--k',type=int,default=4)
    s.add_argument('--cycles',type=int,default=8,help='Additional generations, including on resume.')
    s.add_argument('--children',type=int,default=3,help='Children per parent; last is a no-insertion control when >1.')
    s.add_argument('--beam',type=int,default=2,help='Number of near-best exploratory parents.')
    s.add_argument('--beam-gap',type=float,default=.002,help='Absolute ratio gap for exploratory beam, not best replacement.')
    s.add_argument('--workers',type=int,default=1)
    s.add_argument('--add',type=int,default=12,help='Maximum added vertices per growing child.')
    s.add_argument('--max-points',type=int)
    s.add_argument('--seed',type=int)
    s.add_argument('--branch-kind',choices=['circle','parabola'],default='circle',help='Initial short-branch template; long arcs remain circular.')
    s.add_argument('--template',type=Path,help='Custom model JSON with ordered reference points and protected exceptions.')
    s.add_argument('--groups',type=Path,help='Alternative fresh-input index groups; unassigned points are exceptions.')
    s.add_argument('--fit-tol',type=float,default=.012,help='Residual cutoff in diameter units; rejected points are retained.')
    s.add_argument('--fit-iters',type=int,default=60)
    s.add_argument('--symmetry',choices=['auto','off'],default='auto',help='Fit/sample reflection only when current correspondence is close enough.')
    s.add_argument('--symmetry-tol',type=float,default=2e-4)
    s.add_argument('--break-symmetry-every',type=int,default=4,help='Disable paired sampling for one growing child every N generations; 0 disables.')
    s.add_argument('--jitter',type=float,default=.001,help='New-point sampling jitter in diameter units.')
    s.add_argument('--exception-fraction',type=float,default=.2,help='Target fraction of new proposals near unmodelled vertices.')
    s.add_argument('--min-separation',type=float,default=.001,help='New-point spacing only; existing exceptions are never merged.')
    s.add_argument('--local-rounds',type=int,default=3)
    s.add_argument('--steps',type=int,default=60)
    s.add_argument('--prune-checks',type=int,default=400)
    s.add_argument('--slacks',default='0,0.004,0.012')
    s.add_argument('--opt-jitter',type=float,default=.001,help='Free-coordinate optimization kicks; not projection onto the model.')
    s.add_argument('--slsqp-iters',type=int,default=180)
    s.add_argument('--polish-passes',type=int,default=2)
    s.add_argument('--ftol',type=float,default=1e-12)
    s.add_argument('--full-constraints',action='store_true')
    s.add_argument('--early-after',type=int,default=2)
    s.add_argument('--early-gap',type=float,default=.015,help='Skip unpromising local rounds; any earlier competitive upper bound protects the task.')
    s.add_argument('--min-gain',type=float,default=1e-10,help='Minimum ratio increase for a new best, in addition to exact lower > old upper.')
    s.add_argument('--final-polish-passes',type=int,default=4)
    s.add_argument('--final-slsqp-iters',type=int,default=400)
    bs.add_backend_options(s)
    v=subs.add_parser('verify');v.add_argument('path',type=Path);bs.add_backend_options(v)
    return p


def main(argv=None) -> int:
    p=parser();a=p.parse_args(argv)
    try:
        if not math.isfinite(a.sat_timeout) or a.sat_timeout<0:raise ValueError('--sat-timeout must be finite and nonnegative; 0 disables the time limit.')
        if a.command=='verify':print(json.dumps(verify(a.path,a),indent=2));return 0
        for key in ('cycles','add','prune_checks','steps','local_rounds','polish_passes','final_polish_passes','break_symmetry_every'):
            if getattr(a,key)<0:raise ValueError(f'--{key.replace("_","-")} cannot be negative.')
        for key in ('k','children','beam','workers','fit_iters','slsqp_iters','early_after','final_slsqp_iters'):
            if getattr(a,key)<1:raise ValueError(f'--{key.replace("_","-")} must be positive.')
        for key in ('fit_tol','symmetry_tol','min_separation','ftol'):
            if not math.isfinite(getattr(a,key)) or getattr(a,key)<=0:raise ValueError(f'Invalid {key}.')
        for key in ('jitter','opt_jitter','beam_gap','early_gap','min_gain'):
            if not math.isfinite(getattr(a,key)) or getattr(a,key)<0:raise ValueError(f'Invalid {key}.')
        if not 0<=a.exception_fraction<=1:raise ValueError('Exception fraction must be in [0,1].')
        if a.seed is not None and a.seed<0:raise ValueError('Seed must be nonnegative.')
        sl=[Fraction(v) for v in a.slacks.split(',')]
        if not sl or any(v<0 or v>=1 for v in sl):raise ValueError('Slacks must be comma-separated values in [0,1).')
        a.start=a.start.expanduser().resolve();a.out=a.out.expanduser().resolve()
        run_search(a);return 0
    except KeyboardInterrupt:
        print('Interrupted. Completed nodes, best/, and the last committed beam remain available.',file=sys.stderr);return 130
    except (ValueError,RuntimeError,OSError,ImportError,KeyError) as exc:
        print(f'Error: {exc}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
