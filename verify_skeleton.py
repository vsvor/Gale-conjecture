#!/usr/bin/env python3
"""Verify a closed-segment cover of the COMPLETE 1-skeleton of a 3D hull.

Upper-bound verification uses only the Python standard library and integer
arithmetic. Add --check-lower to verify the source witness with borsuk_search.py.
A solver UNSAT result is not an independently checked SAT proof.
"""
from __future__ import annotations
import argparse
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import json
import math
import sys

def require(condition, text):
    if not condition:
        raise ValueError(text)

def squared(a, b):
    return sum((u-v)**2 for u, v in zip(a,b))

def read_rows(path):
    data = [[Fraction(t.strip()) for t in line.strip().split(',')]
            for line in Path(path).read_text(encoding='ascii').splitlines()
            if line.strip() and not line.lstrip().startswith('#')]
    require(data and all(len(p)==3 for p in data), 'Expected nonempty 3D CSV coordinates.')
    scale = math.lcm(*(v.denominator for p in data for v in p))
    return [tuple(int(v*scale) for v in p) for p in data], scale

def exact_skeleton(rows, max_triples=500000):
    """Independent exhaustive plane enumeration; intersect faces to get true edges."""
    n = len(rows)
    require(n >= 4 and len(set(rows)) == n, 'Need at least four distinct source points.')
    require(math.comb(n,3) <= max_triples, 'Too many source triples for exact enumeration.')
    planes = {}
    full_dim = False
    for a,b,c in combinations(range(n),3):
        u=tuple(rows[b][i]-rows[a][i] for i in range(3))
        v=tuple(rows[c][i]-rows[a][i] for i in range(3))
        normal=(u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        if normal==(0,0,0):
            continue
        offset=sum(x*y for x,y in zip(normal,rows[a]))
        signs=[sum(x*y for x,y in zip(normal,p))-offset for p in rows]
        lo,hi=min(signs),max(signs)
        full_dim |= lo!=0 or hi!=0
        if lo < 0 < hi or lo==hi==0:
            continue
        if lo>=0:
            normal=tuple(-v for v in normal);offset=-offset
        g=math.gcd(*normal)
        key=tuple(v//g for v in normal)+(offset//g,)
        planes[key]=frozenset(i for i,s in enumerate(signs) if s==0)
    require(full_dim and len(planes)>=4, 'The source hull is not full dimensional.')
    edges=set()
    for face1,face2 in combinations(planes.values(),2):
        common=sorted(face1 & face2)
        if len(common)>=2:
            edges.add(max(combinations(common,2),key=lambda e:squared(rows[e[0]],rows[e[1]])))
    require(bool(edges), 'No hull edges found.')
    return sorted(edges), len(planes)

def sqrt_interval(q, digits=15):
    f=10**digits
    l=math.isqrt(q.numerator*f*f//q.denominator)
    u=l if l*l*q.denominator==q.numerator*f*f else l+1
    def text(v):return f'{v//f}.{v%f:0{digits}d}'
    return text(l),text(u)

def verify(folder, check_lower=False, backend='python', timeout=120., max_triples=500000):
    folder=Path(folder)
    data=json.loads((folder/'cover.json').read_text())
    rows,_=read_rows(folder/'source_points.csv')
    edges,faces=exact_skeleton(rows,max_triples)
    # This condition is needed to transfer the source-set lower bound to the skeleton.
    require(set(range(len(rows))) == {v for e in edges for v in e},
            'The certificate requires every source point to be a hull vertex.')
    k=int(data['k']);Q=int(data['denominator'])
    require(k>=1 and Q>=1, 'Invalid k or parameter denominator.')
    groups=[set() for _ in range(k)]
    supplied=set();piece_count=0
    for edge in data['edges']:
        a,b=map(int,edge['vertices'])
        require(0<=a<b<len(rows) and (a,b) in edges, 'Not a true source hull edge.')
        require((a,b) not in supplied, 'Repeated hull edge.')
        supplied.add((a,b)); last=0
        for l,r,c in edge['parts']:
            require(all(isinstance(v,int) and not isinstance(v,bool) for v in (l,r,c)),
                    'Interval endpoints and colors must be integers.')
            require(l==last and l<=r<=Q and 0<=c<k, 'Gap, overlap, or invalid color in edge cover.')
            for t in (l,r):
                groups[c].add(tuple((Q-t)*rows[a][j]+t*rows[b][j] for j in range(3)))
            last=r;piece_count+=1
        require(last==Q, 'An edge is not completely covered.')
    require(supplied==set(edges), 'Some edges of the complete 1-skeleton are missing.')
    D=max(squared(a,b) for a,b in combinations(rows,2))
    part_diameters=[max((squared(a,b) for a,b in combinations(g,2)),default=0) for g in groups]
    upper=Fraction(max(part_diameters),D*Q*Q)
    saved=Fraction(*map(int,data['upper_squared']))
    require(saved==upper, 'Stored upper bound differs from the recomputed segment-cover diameter.')
    witness=[tuple(map(int,e)) for e in data['source_witness']]
    require(all(0<=u<v<len(rows) for u,v in witness), 'Invalid source witness edge.')
    lower=Fraction(min((squared(rows[u],rows[v]) for u,v in witness),default=0),D)
    require(lower==Fraction(*map(int,data['lower_squared'])), 'Stored source lower bound differs.')
    require(lower<=upper, 'Inconsistent upper and lower bounds.')
    result={'complete_skeleton_cover_verified':True,'source_points':len(rows),'hull_edges':len(edges),
            'hull_facets':faces,'colored_segments':piece_count,'part_endpoint_counts':list(map(len,groups)),
            'lower_decimal':sqrt_interval(lower)[0], 'upper_decimal':sqrt_interval(upper)[1],
            'lower_squared':[str(lower.numerator),str(lower.denominator)],
            'upper_squared':[str(upper.numerator),str(upper.denominator)],
            'source_witness_checked':False,
            'qualification':'Upper bound covers the entire 1-skeleton, NOT the faces or solid hull. '
                            'Lower bound additionally requires non-k-colorability of the source witness.'}
    if check_lower:
        import numpy as np
        import borsuk_search as bs
        if not witness and len(rows)<=k:
            status=bs.Status.UNSAT  # zero lower bound needs no witness
        else:
            require(bool(witness), 'A nonzero source lower bound requires a witness.')
            with bs.GraphOracle(len(rows),k,backend,timeout=timeout) as oracle:
                ids=[u*(2*len(rows)-u-1)//2+v-u-1 for u,v in witness]
                status=oracle.solve(ids).status
            require(status is bs.Status.UNSAT, f'Source witness not verified: {status.value}')
        result['source_witness_checked']=True
        result['witness_backend']=backend
        result['qualification']+=' Noncolorability is solver-checked, not independently proof-logged.'
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('--check-lower',action='store_true')
    p.add_argument('--backend',choices=('python','native'),default='python')
    p.add_argument('--timeout',type=float,default=120.)
    a=p.parse_args()
    try:
        print(json.dumps(verify(a.folder,a.check_lower,a.backend,a.timeout),indent=2))
        return 0
    except (ValueError,OSError,KeyError,TypeError) as e:
        print(f'Error: {e}',file=sys.stderr);return 2

if __name__=='__main__':
    raise SystemExit(main())
