#!/usr/bin/env python3
"""Refine the true hull edges, and bound the COMPLETE continuous 1-skeleton.

Requires the accompanying borsuk_search.py and surface_search.py, NumPy/SciPy.
Search supports full-dimensional 3D inputs consisting of their hull vertices.
All old coordinates are fixed. New coordinates are exact dyadic edge points.
Examples:
  python edge_search.py search best_known/input49 --mode uniform --levels 1,2,3,4 --out run
  python edge_search.py search best_known/input49 --mode crossing --rounds 30 --out run2
  python edge_search.py verify run/latest --backend python
  python verify_skeleton.py run/cover --check-lower
"""
from __future__ import annotations
import borsuk_search as bs
import surface_search as ss
import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import json
import math
import shutil
import sys
import time
import numpy as np
from scipy.optimize import minimize
from verify_skeleton import verify as verify_cover, sqrt_interval

VERSION='1.0'
require=ss.require

def true_edges(base, scale, max_triples=500000):
    require(len(base[0])==3, 'This implementation supports full-dimensional 3D source hulls.')
    facets, report=ss.find_facets(base,scale,'exhaustive',max_triples)
    incident=defaultdict(set)
    for f in facets:
        plane=ss.supporting_plane(base,f)
        for e in combinations(f,2):
            incident[tuple(sorted(e))].add(plane)
    # A triangulation diagonal belongs to only one exact supporting plane.
    edges=sorted(e for e,planes in incident.items() if len(planes)>=2)
    require({v for e in edges for v in e}==set(range(len(base))),
            'Start from the original hull vertices, not a surface-search cloud: '
            'some source points are not extreme vertices. No source points were deleted.')
    return facets,edges,report

def edge_spec(cloud, edge, q):
    a,b=edge
    require(0<=q<=cloud.Q, 'Edge parameter outside [0,1].')
    # Full supporting triangle representation keeps surface_search verification compatible.
    face=next(f for f in cloud.facets if a in f and b in f)
    weights=tuple(cloud.Q-q if i==a else q if i==b else 0 for i in face)
    return ss.SurfacePoint(tuple(face),weights)

def initial_chains(cloud, edges):
    return [[(0,a),(cloud.Q,b)] for a,b in edges]

def add_sites(cloud, edges, chains, sites):
    """sites=(edge_index, dyadic_numerator); keep each chain ordered and unique."""
    mapping={r:i for i,r in enumerate(cloud.rows)}; added=0
    for ei,q in sites:
        if any(oldq==q for oldq,_ in chains[ei]):
            continue
        spec=edge_spec(cloud,edges[ei],q); row=cloud.point_row(spec)
        if row in mapping:
            index=mapping[row]
        else:
            index=len(cloud.rows)
            require(cloud.append(spec), 'Internal duplicate insertion error.')
            mapping[row]=index;added+=1
        chains[ei].append((q,index))
    for chain in chains:
        chain.sort()
    return added

def cross_sites(cloud, edges, chains, colors, limit=0):
    proposed=[]
    for ei,chain in enumerate(chains):
        a,b=edges[ei]
        length2=ss.sqdist(cloud.base[a],cloud.base[b])
        for (q0,u),(q1,v) in zip(chain,chain[1:]):
            if colors[u]!=colors[v] and q1-q0>=2:
                proposed.append(((q1-q0)**2*length2,ei,(q0+q1)//2))
    # Longest unresolved segments first; exact lengths, no clustering score.
    proposed.sort(key=lambda z:(-z[0],z[1],z[2]))
    if limit:
        proposed=proposed[:limit]
    return [(ei,q) for _,ei,q in proposed]

def uniform_sites(cloud, edges, level):
    count=2**level
    require(cloud.Q % count==0, 'Uniform level exceeds --grid-bits.')
    return [(ei,cloud.Q*j//count) for ei in range(len(edges)) for j in range(1,count)]

def folded_solve(edges,n,k,session,passes=3):
    """Safe graph folds/degree peeling. Not necessary for the geometric method."""
    adj=[0]*n
    for u,v in edges:
        adj[u]|=1<<v;adj[v]|=1<<u
    active=(1<<n)-1;removed=[]
    for _ in range(passes):
        order=sorted((v for v in range(n) if active>>v&1),
                     key=lambda v:((adj[v]&active).bit_count(),v))
        changed=False
        for v in order:
            if not active>>v&1:
                continue
            neighbors=adj[v]&active
            parent=None; discard=neighbors.bit_count()<k
            if not discard:
                for u in reversed(order):
                    if u!=v and active>>u&1 and neighbors & ~(adj[u]&active)==0:
                        parent=u;discard=True;break
            if discard:
                active &= ~(1<<v); removed.append((v,parent));changed=True
        if not changed:
            break
    kept_edges=[(u,v) for u,v in edges if active>>u&1 and active>>v&1]
    status,colors,witness=session.solve_edges(kept_edges,n)
    if status is bs.Status.SAT and colors is not None:
        for v,_ in removed:
            colors[v]=-1
        for v,parent in reversed(removed):
            if parent is not None:
                require(colors[parent]>=0,'Fold restoration order error.')
                colors[v]=colors[parent]
            else:
                forbidden={colors[u] for u in range(n) if adj[v]>>u&1 and colors[u]>=0}
                colors[v]=next(c for c in range(k) if c not in forbidden)
        require(all(colors[u]!=colors[v] for u,v in edges),'Invalid restored coloring.')
    return status,colors,witness,{'folded_vertices':len(removed),'remaining_edges':len(kept_edges)}

def cover_problem(cloud, colors, chains, k):
    """Convex minimax squared-distance problem with one cut per color transition.

    Internal points of a monochromatic run are omitted from the constraints:
    they are convex combinations of its retained endpoints. This omission is
    exact, and substantially reduces the dense optimization problem.
    """
    transitions=[(u,v) for chain in chains for (_,u),(_,v) in zip(chain,chain[1:])
                 if colors[u]!=colors[v]]
    keep=set(range(cloud.nbase))
    for u,v in transitions:
        keep.update((u,v))
    indices=sorted(keep);mapping={v:i for i,v in enumerate(indices)}
    x=cloud.array()[indices];n=len(x);m=len(transitions)
    groups=[set(mapping[v] for v in indices if colors[v]==c) for c in range(k)]
    for h,(u,v) in enumerate(transitions):
        groups[colors[u]].add(n+h);groups[colors[v]].add(n+h)
    pairs=np.asarray(sorted({p for g in groups for p in combinations(sorted(g),2)}),dtype=int).reshape(-1,2)
    if m:
        segments=np.asarray([(mapping[u],mapping[v]) for u,v in transitions],int)
        aa,bb=segments.T;directions=x[bb]-x[aa]
    else:
        aa=np.empty(0,dtype=int);directions=np.empty((0,3))
    ii,jj=pairs.T
    rows_i=np.flatnonzero(ii>=n);rows_j=np.flatnonzero(jj>=n)
    cache={}
    def values(z):
        if 'z' in cache and np.array_equal(z,cache['z']):
            return cache['q'],cache['jac']
        y=np.vstack((x,x[aa]+z[:m,None]*directions))
        dif=y[ii]-y[jj];q=np.einsum('ij,ij->i',dif,dif)
        jac=np.zeros((len(pairs),m+1));jac[:,-1]=1.
        if len(rows_i):
            ids=ii[rows_i]-n
            jac[rows_i,ids]-=2*np.einsum('ij,ij->i',dif[rows_i],directions[ids])
        if len(rows_j):
            ids=jj[rows_j]-n
            jac[rows_j,ids]+=2*np.einsum('ij,ij->i',dif[rows_j],directions[ids])
        cache.update(z=z.copy(),q=q,jac=jac)
        return q,jac
    return transitions,pairs,values

def optimize_cuts(cloud, colors, chains, k, iterations=300, seconds=30.):
    transitions,pairs,values=cover_problem(cloud,colors,chains,k)
    m=len(transitions)
    z=np.r_[np.full(m,.5),1.]
    z[-1]=float(np.max(values(z)[0],initial=0.))
    best=z.copy();best_q=z[-1]
    def retain(z):
        nonlocal best,best_q
        if not np.isfinite(z).all():
            return
        z=np.asarray(z).copy();z[:m]=np.clip(z[:m],0.,1.)
        q=float(np.max(values(z)[0],initial=0.))
        if q<best_q:
            best=z.copy();best_q=q
    report={'variables':m,'constraints':len(pairs),'solver':'midpoint cuts only','success':True,'iterations':0}
    if m and iterations:
        class CoverTimeout(Exception):
            pass
        started=time.monotonic()
        callback_count=0
        def checked_values(z):
            if seconds>0 and time.monotonic()-started>=seconds:
                raise CoverTimeout
            return values(z)
        def callback(z):
            nonlocal callback_count
            callback_count+=1
            retain(z)
            if seconds>0 and time.monotonic()-started>=seconds:
                raise CoverTimeout
        try:
            result=minimize(lambda z:z[-1],z,method='SLSQP',
                            jac=lambda z:np.r_[np.zeros(m),1.],
                            bounds=[(0.,1.)]*(m+1),
                            constraints=[{'type':'ineq','fun':lambda z:z[-1]-checked_values(z)[0],
                                          'jac':lambda z:checked_values(z)[1]}],
                            callback=callback, options={'ftol':2e-13,'maxiter':iterations})
            retain(result.x)
            report.update(solver='SLSQP',success=bool(result.success),iterations=int(result.nit),message=str(result.message))
        except CoverTimeout:
            report.update(solver='SLSQP',success=False,iterations=callback_count,
                          message='Wall-time budget reached; best recomputed cover retained.')
    report['floating_cover_upper']=math.sqrt(best_q)
    # The accepted upper bound is recomputed after rational quantization below.
    return {p:float(best[i]) for i,p in enumerate(transitions)},report

def make_cover(cloud, edges, chains, colors, cuts, k, source_witness, cut_bits=52):
    R=2**cut_bits;T=cloud.Q*R; groups=[set() for _ in range(k)];pieces=[]
    for (a,b),chain in zip(edges,chains):
        parts=[]
        def add(l,r,c):
            if parts and parts[-1][2]==c and parts[-1][1]==l:
                parts[-1][1]=r
            else:
                parts.append([l,r,int(c)])
        for (q0,u),(q1,v) in zip(chain,chain[1:]):
            if colors[u]==colors[v]:
                add(q0*R,q1*R,colors[u])
            else:
                w=max(0,min(R,round(cuts[(u,v)]*R)))
                q=q0*R+(q1-q0)*w
                add(q0*R,q,colors[u]);add(q,q1*R,colors[v])
        for l,r,c in parts:
            for q in (l,r):
                groups[c].add(tuple((T-q)*cloud.base[a][j]+q*cloud.base[b][j] for j in range(3)))
        pieces.append({'vertices':[int(a),int(b)],'parts':parts})
    qmax=max((ss.sqdist(a,b) for g in groups for a,b in combinations(g,2)),default=0)
    upper=Fraction(qmax,cloud.Dbase*T*T)
    lower=Fraction(min((ss.sqdist(cloud.base[u],cloud.base[v]) for u,v in source_witness),default=0),cloud.Dbase)
    data={'version':VERSION,'k':k,'denominator':str(T),'edges':pieces,
          'lower_squared':[str(lower.numerator),str(lower.denominator)],
          'upper_squared':[str(upper.numerator),str(upper.denominator)],
          'source_witness':source_witness,
          'qualification':'Complete 1-skeleton upper cover, not a cover of the faces or the solid hull.'}
    return data,upper

def source_bounds(cloud, path, session):
    colors=ss.read_partition(path,len(cloud.rows),session.k)
    file=ss.coordinate_file(path).parent/'witness_edges.csv'
    previous=None
    if file.is_file() and file.read_text().strip():
        w=np.loadtxt(file,delimiter=',',dtype=np.int32,ndmin=2)
        require(w.shape[1]==2 and np.all(w>=0) and np.all(w<cloud.nbase), 'Invalid source witness file.')
        w=sorted({tuple(sorted(map(int,e))) for e in w})
        require(all(u<v for u,v in w),'Witness contains a loop.')
        status,_,ww=session.solve_edges(w,cloud.nbase)
        require(status is bs.Status.UNSAT, f'Source witness not verified: {status.value}. Increase --sat-timeout.')
        previous=ss.Bounds(ss.lower_from_witness(cloud,ww),ss.upper_from_colors(cloud,colors),ww,colors)
    return ss.evaluate_exact(cloud,session,previous,colors,calls=40)

def save_state(folder,cloud,bounds,k,baseline,metadata,edges,chains):
    ss.save_snapshot(folder,cloud,bounds,k,baseline,metadata)
    (folder/'skeleton.json').write_text(json.dumps({'version':VERSION,'edges':edges,'chains':chains})+'\n')

def run_search(a):
    require(not a.out.exists(),'Output directory exists; choose a new --out directory.')
    require(a.k>=1 and a.rounds>=0 and a.batch>=0 and a.cover_iters>=0,'Invalid nonnegative integer option.')
    require(1<=a.grid_bits<=52 and 1<=a.cut_bits<=60,'Invalid grid/cut bit count.')
    require(a.gain>=0 and a.tolerance>=0,'--gain and --tolerance must be nonnegative.')
    require(math.isfinite(a.cover_seconds) and a.cover_seconds>=0 and math.isfinite(a.sat_timeout) and a.sat_timeout>=0, 'Invalid timeout.')
    levels=[int(x) for x in a.levels.split(',')]
    require(levels and all(1<=l<=a.grid_bits for l in levels) and levels==sorted(set(levels)),
            '--levels must be distinct increasing positive integers, within --grid-bits.')
    cloud,resumed,report=ss.load_start(a.input,a.grid_bits,'exhaustive',a.max_triples)
    facets,edges,hull_report=true_edges(cloud.base,cloud.base_scale,a.max_triples)
    cloud.facets=facets
    sf=ss.coordinate_file(a.input).parent/'skeleton.json'
    if resumed:
        require(sf.is_file(),'This is a surface-search checkpoint, not an edge-search checkpoint. '
                'Start with its original hull-vertex configuration instead.')
        state=json.loads(sf.read_text());require(state['edges']==[list(e) for e in edges],'Saved hull edges differ.')
        chains=[[tuple(map(int,p)) for p in chain] for chain in state['chains']]
        require(len(chains)==len(edges),'Wrong chain count.')
        for edge,chain in zip(edges,chains):
            require(chain==sorted(set(chain)) and chain[0]==(0,edge[0]) and chain[-1]==(cloud.Q,edge[1]),'Invalid edge chain.')
            for q,i in chain:
                require(0<=i<len(cloud.rows) and cloud.rows[i]==cloud.point_row(edge_spec(cloud,edge,q)),
                        'Saved chain coordinates disagree.')
    else:
        chains=initial_chains(cloud,edges)
    if a.max_points is None:
        a.max_points=max(len(cloud.rows), cloud.nbase+(2**levels[-1]-1)*len(edges)) if a.mode=='uniform' else max(2000,len(cloud.rows))
    require(a.max_points>=len(cloud.rows),'--max-points is smaller than the input.')
    a.out.mkdir(parents=True)
    start=time.monotonic()
    def log(event,**data):
        data={'event':event,'seconds':round(time.monotonic()-start,3),**data}
        with (a.out/'progress.jsonl').open('a') as f:f.write(json.dumps(data,default=str)+'\n')
        print(json.dumps(data,default=str),flush=True)
    (a.out/'settings.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')
    (a.out/'hull.json').write_text(json.dumps(hull_report|{'edges':edges},indent=2)+'\n')
    session=ss.Session(a.max_points,a.k,a.backend,a.solver,a.sat_timeout,a.pysat_solver)
    try:
        # On continuation the sample witness may use added points, so obtain a separately
        # verified source-vertex witness for the self-contained cover certificate.
        if resumed:
            original=ss.Cloud(cloud.base,cloud.base_scale,cloud.Q,cloud.facets)
            source=ss.evaluate_exact(original,session,calls=40)
            source_witness=source.witness
            colors=ss.read_partition(a.input,len(cloud.rows),a.k)
            bounds=ss.evaluate_exact(cloud,session,colors=colors,calls=40)
        else:
            bounds=source_bounds(cloud,a.input,session);source_witness=bounds.witness
        baseline=Fraction(bounds.upper,cloud.D)
        best_lower=Fraction(bounds.lower,cloud.D);cover_upper=Fraction(1)
        finite_best=best_lower;stop='round_budget_exhausted';completed=0
        save_state(a.out/'best',cloud,bounds,a.k,baseline,{'stage':'initial'},edges,chains)
        log('initial',**bounds.payload(cloud),hull_edges=len(edges))
        schedule=levels if a.mode=='uniform' else list(range(1,a.rounds+1))
        for step in [None]+schedule:
            if step is not None:
                if a.mode=='uniform':
                    sites=uniform_sites(cloud,edges,step)
                    known={(ei,q) for ei,chain in enumerate(chains) for q,_ in chain}
                    sites=[s for s in sites if s not in known]
                else:
                    require(bounds.colors is not None,'No partition available for crossing-edge refinement.')
                    sites=cross_sites(cloud,edges,chains,bounds.colors,a.batch)
                if not sites:
                    if a.mode=='uniform':
                        continue
                    stop='no_unresolved_dyadic_segment';break
                room=a.max_points-len(cloud.rows)
                if room<=0:
                    stop='point_budget_exhausted';break
                if len(sites)>room:
                    if a.mode=='uniform':
                        stop='insufficient_budget_for_complete_uniform_level';break
                    sites=sites[:room]
                oldcolors=bounds.colors
                added=add_sites(cloud,edges,chains,sites)
                if not added:
                    stop='no_new_sites';break
                # The target is a proposal threshold; all reported bounds use exact distances.
                rup=Fraction(ss.sqrt_bounds(Fraction(bounds.upper,cloud.D),18)[1])
                target=(rup+a.gain)**2
                p,q=target.numerator,target.denominator
                ge=[e for e,d in zip(cloud.pairs,cloud.q) if d*q>cloud.D*p]
                if a.fold:
                    status,colors,w,foldinfo=folded_solve(ge,len(cloud.rows),a.k,session)
                else:
                    status,colors,w=session.solve_edges(ge,len(cloud.rows));foldinfo={}
                if status is bs.Status.SAT and colors is not None:
                    bounds=ss.Bounds(bounds.lower,ss.upper_from_colors(cloud,colors),bounds.witness,colors)
                else:
                    fallback=ss.extend_partition(cloud,oldcolors,a.k)
                    if status is bs.Status.UNSAT:
                        bounds=ss.Bounds(ss.lower_from_witness(cloud,w),ss.upper_from_colors(cloud,fallback),w,fallback)
                        bounds=ss.evaluate_exact(cloud,session,bounds,calls=a.eval_checks)
                    else:
                        bounds=ss.Bounds(bounds.lower,ss.upper_from_colors(cloud,fallback),bounds.witness,fallback,True)
                completed+=1
                best_lower=max(best_lower,Fraction(bounds.lower,cloud.D))
                log('sample',step=step,added=added,target_status=status.value,threshold_edges=len(ge),
                    **bounds.payload(cloud),**foldinfo)
            colors=bounds.colors or [0]*len(cloud.rows)
            cuts,optimization=optimize_cuts(cloud,colors,chains,a.k,a.cover_iters,a.cover_seconds)
            cover,upper=make_cover(cloud,edges,chains,colors,cuts,a.k,source_witness,a.cut_bits)
            require(upper>=Fraction(bounds.lower,cloud.D),'Cover conflicts with lower bound.')
            if upper<cover_upper or not (a.out/'cover'/'cover.json').exists():
                cover_upper=upper;folder=a.out/'cover';folder.mkdir(exist_ok=True)
                ss.write_rows(folder/'source_points.csv',cloud.base,cloud.base_scale)
                cover['optimization']=optimization
                (folder/'cover.json').write_text(json.dumps(cover,indent=2)+'\n')
                # Different implementation recomputes all supporting planes and their edge intersections.
                verified=verify_cover(folder,max_triples=a.max_triples)
                (folder/'verification.json').write_text(json.dumps(verified,indent=2)+'\n')
                log('cover_improved',continuous_upper=ss.sqrt_bounds(upper)[1],**optimization)
            meta={'mode':a.mode,'step':step,'continuous_skeleton_upper':ss.sqrt_bounds(cover_upper)[1]}
            save_state(a.out/'latest',cloud,bounds,a.k,baseline,meta,edges,chains)
            if Fraction(bounds.lower,cloud.D)>finite_best:
                finite_best=Fraction(bounds.lower,cloud.D)
                save_state(a.out/'best',cloud,bounds,a.k,baseline,meta,edges,chains)
            # Outward rounded endpoints make early termination conservative.
            lo=Fraction(ss.sqrt_bounds(best_lower,18)[0]);up=Fraction(ss.sqrt_bounds(cover_upper,18)[1])
            if up-lo<=a.tolerance:
                stop='continuous_skeleton_gap_below_tolerance';break
        result={'points':len(cloud.rows),'completed_steps':completed,'stop':stop,
                'finite_lower':ss.sqrt_bounds(best_lower)[0],
                'finite_sample_upper':ss.sqrt_bounds(Fraction(bounds.upper,cloud.D))[1],
                'continuous_skeleton_upper':ss.sqrt_bounds(cover_upper)[1],
                'initial_sample_upper':ss.sqrt_bounds(baseline)[1],
                'certified_strict_improvement':best_lower>baseline,
                'meaningful_improvement_above_gain':Fraction(ss.sqrt_bounds(best_lower,18)[0])>
                                                    Fraction(ss.sqrt_bounds(baseline,18)[1])+a.gain,
                'backend':session.oracle.backend,
                'qualification':'Bounds concern the 1-skeleton, not the whole convex hull. '
                                'Solver UNSAT is not independently proof-logged.'}
        (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        log('finished',**result)
        return result
    finally:
        session.close()

def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('search');s.add_argument('input',type=Path);s.add_argument('--out',type=Path,required=True)
    s.add_argument('--k',type=int,default=4)
    s.add_argument('--mode',choices=('crossing','uniform'),default='crossing')
    s.add_argument('--levels',default='1,2,3,4',help='Uniform subdivisions are 2**level; ignored in crossing mode.')
    s.add_argument('--rounds',type=int,default=30)
    s.add_argument('--batch',type=int,default=0,help='Crossing mode: midpoints per round; 0 means every bichromatic segment.')
    s.add_argument('--max-points',type=int)
    s.add_argument('--gain',type=Fraction,default=Fraction('1e-10'),help='Ratio step for the SAT target, not a reporting tolerance.')
    s.add_argument('--tolerance',type=Fraction,default=Fraction('1e-10'),help='Stop when continuous upper minus lower is this small.')
    s.add_argument('--cover-iters',type=int,default=300,help='Cut-position SLSQP iterations; 0 uses midpoint covers only.')
    s.add_argument('--cover-seconds',type=float,default=30.,help='Per-cover wall budget, checked between optimizer calls; 0 is unlimited.')
    s.add_argument('--grid-bits',type=int,default=32);s.add_argument('--cut-bits',type=int,default=52)
    s.add_argument('--max-triples',type=int,default=500000)
    s.add_argument('--eval-checks',type=int,default=35)
    s.add_argument('--fold',action='store_true',help='Safe degree peeling / dominated-neighborhood reductions before solving.')
    for parser in (s,):
        parser.add_argument('--backend',choices=('auto','native','python','external','pysat'),default='auto')
        parser.add_argument('--solver',default='kissat');parser.add_argument('--sat-timeout',type=float,default=30.)
        parser.add_argument('--pysat-solver',default='g4')
    v=sub.add_parser('verify');v.add_argument('folder',type=Path)
    v.add_argument('--backend',choices=('auto','native','python','external','pysat'),default='python')
    v.add_argument('--solver',default='kissat');v.add_argument('--sat-timeout',type=float,default=120.)
    v.add_argument('--pysat-solver',default='g4')
    return p

def main(argv=None):
    a=build_parser().parse_args(argv)
    try:
        if a.command=='search':
            run_search(a)
        else:
            result=ss.verify_surface(a.folder,a)
            state=json.loads((a.folder/'skeleton.json').read_text())
            cloud,_,_=ss.load_start(a.folder,32,'verified',0)
            _,edges,_=true_edges(cloud.base,cloud.base_scale)
            require(state['edges']==[list(e) for e in edges], 'Incorrect saved 1-skeleton.')
            for spec in cloud.specs[cloud.nbase:]:
                support={v for v,w in zip(spec.vertices,spec.weights) if w}
                require(len(support)<=2 and any(support<=set(e) for e in edges),'A point is not on a true hull edge.')
            result['all_added_points_on_true_hull_edges']=True
            print(json.dumps(result,indent=2))
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Completed latest/best/cover checkpoints remain available.',file=sys.stderr);return 130
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as e:
        print(f'Error: {e}',file=sys.stderr);return 2

if __name__=='__main__':
    raise SystemExit(main())
