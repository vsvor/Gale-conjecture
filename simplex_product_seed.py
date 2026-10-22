#!/usr/bin/env python3
"""Fresh simplex-product/cap initializers for Borsuk search in dimensions 5--8.

This is a proposal generator, NOT a certificate, optimizer, or established
high-dimensional construction. It retains no old record. Only NumPy is needed.

Example:
    python simplex_product_seed.py --dim 6 --points 100 --seed 10 --out seed6
Import:
    x = make_seed(6, 100, np.random.default_rng(10))
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

VERSION = '0.1.0'
# Approximate fitted R5 shape, not five-dimensional record coordinates.
FIT5 = np.array([0.5664861410792998, 0.1174892176590891,
                 0.12954876088150585, 0.5960185155827592,
                 0.36983417019229614, 0.45839379107888545])
BASE_SCALES = np.array([math.sqrt(1/3), math.sqrt(3/8)] * 3)
# a,c,e belong to E; b,d,f belong to F.
RATIOS = FIT5 / BASE_SCALES


def regular_simplex(d: int) -> np.ndarray:
    """Return d+1 unit vectors with off-diagonal inner products -1/d."""
    if d < 1:
        raise ValueError('Simplex dimension must be positive.')
    h = np.zeros((d+1, d))
    for j in range(d):
        h[:j+1, j] = 1/math.sqrt((j+1)*(j+2))
        h[j+1, j] = -(j+1)/math.sqrt((j+1)*(j+2))
    return h*math.sqrt((d+1)/d)


def diameter(x: np.ndarray) -> float:
    """Blockwise numerical diameter, without an m-by-m-by-d temporary."""
    largest = 0.
    for start in range(0, len(x), 128):
        delta = x[start:start+128, None, :] - x[None, :, :]
        largest = max(largest, float(np.einsum('ijk,ijk->ij', delta, delta).max()))
    return math.sqrt(largest)


@dataclass
class Shape:
    p: int
    q: int
    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    hE: float
    rE: float
    hF: float
    rF: float
    alpha0: float
    alpha1: float
    beta0: float
    beta1: float

    @classmethod
    def from_scalars(cls, p: int, q: int, scalars) -> 'Shape':
        a, b, c, d, e, f = map(float, scalars)
        if not all(math.isfinite(z) and z > 0 for z in (a,b,c,d,e,f)):
            raise ValueError('Shape scalars must be positive finite numbers.')
        if a-e <= 1e-5 or d-f <= 1e-5:
            raise ValueError('Require a>e and d>f, separated from degeneracy.')
        hE = (a*a+b*b-e*e-f*f)/(2*(a-e))
        hF = (c*c+d*d-e*e-f*f)/(2*(d-f))
        rE, rF = math.hypot(a-hE,b), math.hypot(d-hF,c)
        if min(a-hE, e-hE, d-hF, f-hF) <= 0 or max(rE,rF) > 2.:
            raise ValueError('Shape is outside the small outward-cap family.')
        return cls(p,q,a,b,c,d,e,f,hE,rE,hF,rF,
                   math.atan2(-b,a-hE), math.atan2(f,e-hE),
                   math.atan2(-c,d-hF), math.atan2(e,f-hF))


def sample_shape(p: int, q: int, rng: np.random.Generator,
                 shape_noise: float, swap_roles: bool | None) -> tuple[Shape, bool]:
    """Scale fitted coefficients by unit-edge simplex circumradii; add log noise.

    Role-swapping avoids permanently assigning the R5 triangle's shape to the
    first factor, especially when p=q. This is extrapolation for dimensions >5.
    """
    swapped = bool(rng.integers(2)) if swap_roles is None else swap_roles
    ratios = RATIOS.copy()
    if swapped:
        ratios = ratios[[3,2,1,0,5,4]]
    scales = np.array([math.sqrt(p/(2*(p+1))), math.sqrt(q/(2*(q+1)))]*3)
    for _ in range(100):
        values = ratios*scales*np.exp(rng.normal(0,shape_noise,size=6))
        try:
            return Shape.from_scalars(p,q,values), swapped
        except ValueError:
            continue
    raise ValueError('Unable to sample a nondegenerate shape; reduce --shape-noise.')


def meridian(shape: Shape, u: np.ndarray, v: np.ndarray,
             i: int, j: int, angles: np.ndarray, side: str) -> np.ndarray:
    t = np.atleast_1d(angles)
    if side == 'E':
        return np.c_[(shape.hE+shape.rE*np.cos(t))[:,None]*u[i],
                     (shape.rE*np.sin(t))[:,None]*v[j]]
    if side == 'F':
        return np.c_[(shape.rF*np.sin(t))[:,None]*u[i],
                     (shape.hF+shape.rF*np.cos(t))[:,None]*v[j]]
    raise ValueError('side must be E or F')


def sites(shape: Shape, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray,list[dict]]:
    out, tags = [], []
    for family, left, right in [('A',shape.a,-shape.b),
                                ('B',-shape.c,shape.d),('C',shape.e,shape.f)]:
        for i in range(shape.p+1):
            for j in range(shape.q+1):
                out.append(np.r_[left*u[i],right*v[j]])
                tags.append({'family':family,'i':i,'j':j,'site':len(tags)})
    for i in range(shape.p+1):
        out.append(np.r_[(shape.hE+shape.rE)*u[i],np.zeros(shape.q)])
        tags.append({'family':'P','i':i,'j':None,'site':len(tags)})
    for j in range(shape.q+1):
        out.append(np.r_[np.zeros(shape.p),(shape.hF+shape.rF)*v[j]])
        tags.append({'family':'Q','i':None,'j':j,'site':len(tags)})
    return np.array(out), tags


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < 1e-14:
        raise ValueError('Numerically zero direction.')
    return x/n


def candidate_pool(shape: Shape, u: np.ndarray, v: np.ndarray, count: int,
                   rng: np.random.Generator, width: float) -> tuple[np.ndarray,list[dict]]:
    """Mixture: 30% site neighborhoods, 40% meridians, 25% local caps, 5% wild.

    This is deliberately NOT uniform sampling on a full sphere or cap. Broad
    transverse cap proposals explore directions not represented by meridians.
    """
    base, base_tags = sites(shape,u,v)
    out, tags = [], []
    for _ in range(count):
        i,j = int(rng.integers(shape.p+1)),int(rng.integers(shape.q+1))
        which = rng.random()
        side = 'E' if rng.random() < .5 else 'F'
        lo, hi = (shape.alpha0,shape.alpha1) if side=='E' else (shape.beta0,shape.beta1)
        if which < .30:
            b = int(rng.integers(len(base)))
            point = base[b]+rng.normal(size=shape.p+shape.q)*.012/math.sqrt(shape.p+shape.q)
            tag = dict(base_tags[b], mode='site_neighborhood')
        elif which < .70:
            # Uniform parameter is uniform arclength on these circular meridians.
            t = float(rng.uniform(lo,hi))
            point = meridian(shape,u,v,i,j,np.array([t]),side)[0]
            tag = {'mode':'meridian','family':side,'i':i,'j':j,'angle':t}
        elif which < .95:
            t = float(rng.uniform(lo,hi))
            base_dir = v[j] if side=='E' else u[i]
            noise = rng.normal(size=len(base_dir))
            noise -= noise.dot(base_dir)*base_dir
            if rng.random() < .8:
                direction = _unit(base_dir+width*_unit(noise))
            else:
                # Sparse combinations can explore between two observed meridians.
                verts = v if side=='E' else u
                other = (j if side=='E' else i)
                other = (other+int(rng.integers(1,len(verts)))) % len(verts)
                weight = rng.uniform(.2,.5)
                direction = _unit((1-weight)*base_dir+weight*verts[other])
            if side=='E':
                point = np.r_[(shape.hE+shape.rE*math.cos(t))*u[i],shape.rE*math.sin(t)*direction]
            else:
                point = np.r_[shape.rF*math.sin(t)*direction,(shape.hF+shape.rF*math.cos(t))*v[j]]
            tag = {'mode':'cap','family':side,'i':i,'j':j,'angle':t}
        else:
            point = _unit(rng.normal(size=shape.p+shape.q))*rng.uniform(.52,.63)
            tag = {'mode':'wild','family':'wild','i':None,'j':None}
        out.append(point); tags.append(tag)
    return np.array(out),tags


def _features(pool: np.ndarray, landmarks: np.ndarray, D: float) -> np.ndarray:
    q = np.sum((pool[:,None,:]-landmarks[None,:,:])**2,axis=2)/D**2
    thresholds = np.array([.84,.88,.92,.96])**2
    bits = (q[:,:,None] >= thresholds).reshape(len(pool),-1).astype(float)
    # Each feature distance combines normalized Hamming distance with geometry.
    return np.c_[bits/math.sqrt(bits.shape[1]),.3*pool/D]


def generate(dim: int, points: int, rng: np.random.Generator, *,
             split: tuple[int,int] | None = None, shape_noise: float = .012,
             direction_noise: float = .006, jitter: float = .001,
             cap_width: float = .15, tilt: float = .006,
             pool_factor: int = 24, min_separation: float = .003,
             diameter_slack: float = .02, selection: str = 'spatial',
             swap_roles: bool | None = None, site_fraction: float = 1.) -> tuple[np.ndarray,dict[str,Any]]:
    """Generate a fresh finite seed and diagnostics; do not preserve a record.

    Default split is floor(dim/2)+ceil(dim/2). All input random choices use rng.
    Positional arguments match the engine's typical initializer convention.
    'min_separation' is checked in final diameter units. Candidate failure raises
    explicitly instead of introducing duplicates or reducing separation silently.
    """
    if dim not in range(5,9):
        raise ValueError('This experimental initializer supports dimensions 5--8.')
    if points < dim+2:
        raise ValueError('Require at least dim+2 points (more than k=dim+1).')
    p,q = split if split is not None else (dim//2,dim-dim//2)
    if p+q != dim or min(p,q)<2:
        raise ValueError('Require p+q=dim and p,q>=2.')
    for label,value in [('shape_noise',shape_noise),('direction_noise',direction_noise),
                        ('jitter',jitter),('cap_width',cap_width),('tilt',tilt),
                        ('min_separation',min_separation),('diameter_slack',diameter_slack)]:
        if not math.isfinite(value) or value<0:
            raise ValueError(f'{label} must be finite and nonnegative.')
    if not math.isfinite(site_fraction) or not 0 <= site_fraction <= 1:
        raise ValueError('Require 0 <= site_fraction <= 1.')
    if pool_factor<2 or selection not in ('spatial','signature','random'):
        raise ValueError('Require pool_factor>=2 and a valid selection method.')
    s,swapped = sample_shape(p,q,rng,shape_noise,swap_roles)
    u,v=regular_simplex(p),regular_simplex(q)
    for dirs in (u,v):
        if direction_noise:
            dirs += rng.normal(size=dirs.shape)*direction_noise/math.sqrt(dirs.shape[1])
            dirs /= np.linalg.norm(dirs,axis=1)[:,None]
    base,tags = sites(s,u,v)
    if jitter:
        base += rng.normal(size=base.shape)*jitter/math.sqrt(dim)
    shear=np.eye(dim)
    if tilt:
        cross=rng.normal(size=(p,q))*tilt/math.sqrt(dim)
        shear[:p,p:]=cross; shear[p:,:p]=cross.T
    base=base@shear
    D0=diameter(base)
    # No growth-preservation theorem is claimed. Keep a controlled diameter
    # relative to the fresh noisy site frame so remote candidates cannot dominate.
    Dlimit=(1+diameter_slack)*D0
    limit2=Dlimit*Dlimit
    sep2=(min_separation*Dlimit)**2
    site_budget=min(points,max(dim+2,int(math.ceil(site_fraction*len(base)))))
    if site_budget>=len(base):
        selected=[r.copy() for r in base]
        selected_tags=[dict(t,mode='site') for t in tags]
    else:
        # Keep poles, then cover A/B/C and transverse index pairs as evenly as possible.
        first=[i for i,t in enumerate(tags) if t['family'] in ('P','Q')]
        stacks=[list(rng.permutation([i for i,t in enumerate(tags) if t['family']==f]))
                for f in ('A','B','C')]
        while len(first)<site_budget:
            for stack in stacks:
                if stack and len(first)<site_budget:first.append(stack.pop())
        selected=[base[i].copy() for i in first]
        selected_tags=[dict(tags[i],mode='site') for i in first]
    a=np.array(selected)
    if len(a)>1:
        qq=np.sum((a[:,None,:]-a[None,:,:])**2,axis=2)
        np.fill_diagonal(qq,np.inf)
        if qq.min()<sep2:
            raise ValueError('Site points violate minimum separation; change seed or shape parameters.')
    total_pool=max(512,pool_factor*points)
    candidates,candidate_tags=candidate_pool(s,u,v,total_pool,rng,cap_width)
    if jitter:
        candidates += rng.normal(size=candidates.shape)*jitter/math.sqrt(dim)
    candidates=candidates@shear
    # Chunk-free arrays here have candidate_count * selected_count entries; no
    # candidate-by-candidate distance matrix is formed.
    d2=np.sum((candidates[:,None,:]-a[None,:,:])**2,axis=2)
    nearest=d2.min(axis=1); farthest=d2.max(axis=1)
    eligible=(nearest>=sep2)&(farthest<=limit2)
    if selection=='signature':
        features=_features(candidates,base,D0)
        ref=_features(a,base,D0)
        score=np.full(len(candidates),np.inf)
        for r in ref:
            score=np.minimum(score,np.sum((features-r)**2,axis=1))
    else:
        features=None; score=nearest.copy()
    modes=['meridian','cap','site_neighborhood','wild']
    # Quota schedule avoids pure farthest-point selection starving the meridians.
    schedule=['meridian','meridian','cap','site_neighborhood','meridian','cap',
              'site_neighborhood','meridian','cap','wild']
    rng.shuffle(schedule)
    deficit=points-len(selected)
    rejected_diameter=int(np.sum(farthest>limit2))
    for j in range(deficit):
        target=schedule[j%len(schedule)]
        group=np.array([t['mode']==target for t in candidate_tags])
        valid=eligible&group
        if not valid.any():valid=eligible.copy()
        if not valid.any():
            raise ValueError(f'Candidate pool exhausted after {len(selected)} of {points} points. '
                             'Increase --pool-factor/--diameter-slack or reduce --min-separation.')
        ids=np.flatnonzero(valid)
        idx=int(rng.choice(ids)) if selection=='random' else int(ids[np.argmax(score[ids])])
        point=candidates[idx].copy()
        selected.append(point); selected_tags.append(candidate_tags[idx])
        now=np.sum((candidates-point)**2,axis=1)
        nearest=np.minimum(nearest,now); farthest=np.maximum(farthest,now)
        eligible &= (nearest>=sep2)&(farthest<=limit2)
        eligible[idx]=False
        if selection=='signature':
            score=np.minimum(score,np.sum((features-features[idx])**2,axis=1))
        else:score=nearest.copy()
    x=np.array(selected)
    D=diameter(x); center=x.mean(axis=0)
    x=(x-center)/D
    if len(np.unique(x,axis=0))!=points or np.linalg.matrix_rank(x)!=dim:
        raise ValueError('Generated points are duplicated or fail to span the ambient space.')
    qmin=np.inf
    for i in range(points-1):
        qmin=min(qmin,float(np.sum((x[i+1:]-x[i])**2,axis=1).min()))
    if math.sqrt(qmin)<min_separation*(1-1e-9):
        raise RuntimeError('Final minimum-separation check failed.')
    summary={mode:sum(t['mode']==mode for t in selected_tags) for mode in ['site']+modes}
    return x,{'version':VERSION,'dimension':dim,'points':points,'k_default':dim+1,
              'split':[p,q],'shape':asdict(s),'roles_swapped':swapped,
              'simplex_E':u.tolist(),'simplex_F':v.tolist(),'linear_deformation':shear.tolist(),
              'centering_before_scale':center.tolist(),'normalization_diameter':D,
              'fresh_site_count':len(base),'fresh_site_diameter':D0,
              'diameter_ratio_to_site_cloud':D/D0,'final_diameter_float':diameter(x),
              'minimum_separation_float':math.sqrt(qmin),'candidate_pool':total_pool,
              'initial_candidates_rejected_by_diameter':rejected_diameter,
              'selection':selection,'sample_counts':summary,'point_tags':selected_tags,
              'options':dict(shape_noise=shape_noise,direction_noise=direction_noise,jitter=jitter,
                             cap_width=cap_width,tilt=tilt,pool_factor=pool_factor,
                             min_separation=min_separation,diameter_slack=diameter_slack,site_fraction=site_fraction),
              'qualification':'Fresh numerical initializer, NOT a certified ratio or improvement. '
                              'R5 fitted family is extrapolated to dimensions 6--8. '
                              'Perturbations/deformation mean final points are not exact cap members.'}


def make_seed(dim:int, points:int, rng:np.random.Generator, **kwargs) -> np.ndarray:
    """Convenience adapter: return only the coordinate array."""
    return generate(dim,points,rng,**kwargs)[0]


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dim',type=int,required=True,choices=range(5,9))
    p.add_argument('--points',type=int,required=True)
    p.add_argument('--split',help='e.g. 2,3; default floor(dim/2),ceil(dim/2).')
    p.add_argument('--seed',type=int,default=20260912)
    p.add_argument('--shape-noise',type=float,default=.012)
    p.add_argument('--direction-noise',type=float,default=.006)
    p.add_argument('--jitter',type=float,default=.001)
    p.add_argument('--cap-width',type=float,default=.15)
    p.add_argument('--tilt',type=float,default=.006)
    p.add_argument('--pool-factor',type=int,default=24)
    p.add_argument('--min-separation',type=float,default=.003)
    p.add_argument('--diameter-slack',type=float,default=.02)
    p.add_argument('--site-fraction',type=float,default=1.,help='Fraction of product sites retained, plus all poles. 0.75 explores incomplete scaffolds.')
    p.add_argument('--selection',choices=['spatial','signature','random'],default='spatial')
    p.add_argument('--swap-roles',choices=['random','no','yes'],default='random')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.out.exists():raise ValueError('Output exists; use a new directory.')
        split=None if a.split is None else tuple(map(int,a.split.split(',')))
        if split is not None and len(split)!=2:raise ValueError('--split must contain two dimensions.')
        kws={k:getattr(a,k) for k in ['shape_noise','direction_noise','jitter','cap_width','tilt',
                                     'pool_factor','min_separation','diameter_slack','selection','site_fraction']}
        swap={'random':None,'no':False,'yes':True}[a.swap_roles]
        x,meta=generate(a.dim,a.points,np.random.default_rng(a.seed),split=split,swap_roles=swap,**kws)
        meta['seed']=a.seed
        a.out.mkdir(parents=True)
        np.save(a.out/'points.npy',x)
        np.savetxt(a.out/'points.csv',x,delimiter=',',fmt='%.17g')
        (a.out/'generator.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(json.dumps({k:meta[k] for k in ['dimension','points','split','minimum_separation_float',
                                           'sample_counts','diameter_ratio_to_site_cloud','qualification']},indent=2))
        return 0
    except (ValueError,RuntimeError,OSError) as exc:
        p.error(str(exc))
    return 2

if __name__=='__main__':
    raise SystemExit(main())
