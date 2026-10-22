#!/usr/bin/env python3
"""Generate structured starting points from the fitted, reflection-symmetric 81-point record.

Every seed retains the original 81 binary floating-point rows, unchanged. New
points are sampled on nine fitted curves or near the geometric residual points.
Proposed new points are contracted when necessary to preserve the OLD diameter,
checking new-new distances as well as new-old distances. The default final check
uses exact integer arithmetic on the NPY binary values.

This is an initializer, not a new partition-ratio optimizer or an automatic
optimize-fit loop. Requires numpy; --kind parabola additionally uses no extra
packages. The fitting program compare_curves.py requires scipy.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
REF=np.array([1.,1.,-1.])


def arclength(t,k,kind):
    t=np.asarray(t,dtype=float)
    if kind=='parabola':
        return (t*np.sqrt(1+(k*t)**2)+np.arcsinh(k*t)/k)/2
    return t


def branch_value(b,t):
    t=np.atleast_1d(t).astype(float)
    d=np.array(b['tangent']);n=np.array(b['normal']);J=np.array(b['junction'])
    k=b['curvature']
    if b['kind']=='circle':
        # Stable version of (1-cos(k*t))/k near the junction.
        return J+np.sin(k*t)[:,None]/k*d+2*np.sin(k*t/2)[:,None]**2/k*n
    if b['kind']=='parabola':
        return J+t[:,None]*d+0.5*k*t[:,None]**2*n
    return J+t[:,None]*d


def components(template,comparison,kind):
    result=[]
    for label in ['arc_plus','arc_zero']:
        b=template[label];lo,hi=b['angle_interval_radians']
        result.append({'name':label,'paired':label=='arc_plus','length':b['radius']*(hi-lo),
                       'curve':'outer_circle','data':b})
    for sign in ['plus','minus']:
        branches=comparison['fits'][sign+'_near_apex']['models'][kind]['branches']
        for b in branches:
            lo,hi=b['parameter_interval']
            result.append({'name':'tripod_'+sign+'_'+b['branch'],'paired':b['branch']=='side',
                           'length':float(arclength(hi,b['curvature'],kind)-arclength(lo,b['curvature'],kind)),
                           'curve':'branch','data':b})
    return result


def sample_curve(component,fractions):
    b=component['data'];f=np.asarray(fractions)
    if component['curve']=='outer_circle':
        lo,hi=b['angle_interval_radians'];t=lo+(hi-lo)*f
        return np.array(b['center'])+b['radius']*(np.cos(t)[:,None]*np.array(b['basis_e'])+
                                               np.sin(t)[:,None]*np.array(b['basis_f']))
    lo,hi=b['parameter_interval']
    if b['kind']=='parabola':
        k=b['curvature'];target=arclength(lo,k,b['kind'])+f*(arclength(hi,k,b['kind'])-arclength(lo,k,b['kind']))
        a=np.full(len(f),lo);z=np.full(len(f),hi)
        for _ in range(50):
            mid=(a+z)/2;mask=arclength(mid,k,b['kind'])<target
            a=np.where(mask,mid,a);z=np.where(mask,z,mid)
        t=(a+z)/2
    else:t=lo+(hi-lo)*f
    return branch_value(b,t)


def exact_geometry(seed,source):
    """Proof for the stored binary coordinates, not a tolerance-based check."""
    if seed.shape[1:]!=source.shape[1:] or not np.array_equal(seed[:len(source)],source):
        raise ValueError('Original rows were changed.')
    ratios=[[float(v).as_integer_ratio() for v in row] for row in seed]
    Q=max(q for row in ratios for _,q in row)
    rows=[tuple(p*(Q//q) for p,q in row) for row in ratios]
    if len(set(rows))!=len(rows):raise ValueError('The seed contains duplicate rows.')
    D=0;minimum=None;new_max=0
    for j in range(len(rows)):
        b=rows[j]
        for i in range(j):
            q=sum((u-v)**2 for u,v in zip(rows[i],b))
            minimum=q if minimum is None else min(minimum,q)
            if j<len(source):D=max(D,q)
            else:new_max=max(new_max,q)
    if new_max>D:raise ValueError('Exact binary check: an added pair increased the diameter.')
    return {'original_rows_unchanged':True,'distinct_points':len(rows),
            'diameter_exactly_unchanged':True,'arithmetic':'integers on exact NPY binary values',
            'common_coordinate_denominator':str(Q),'diameter_squared_integer':str(D),
            'maximum_new_pair_squared_integer':str(new_max),
            'minimum_separation_over_diameter':math.sqrt(minimum/D),
            'qualification':'This verifies geometry only. It inherits whatever lower bound is verified '
                             'for the source; it does not certify an improved ratio or a partition upper bound.'}


def contract_pair(y,accepted_uv,limit2):
    """Common radial factor for all members, preserving their reflection relation.

    The center 0 is the midpoint of two original points. Its distances to every
    previous accepted point are < limit; the whole ray-feasible set is an interval.
    Within-orbit pair distances are included; completion-body membership alone is
    not sufficient for several additions.
    """
    y=np.atleast_2d(y)
    def feasible(alpha):
        z=alpha*y
        if np.max(np.sum((z[:,None]-accepted_uv[None])**2,axis=2))>limit2:return False
        return len(z)==1 or np.sum((z[0]-z[1])**2)<=limit2
    if feasible(1.):return y,1.
    if not feasible(0.):raise ValueError('Contraction center is not feasible.')
    lo,hi=0.,1.
    for _ in range(55):
        mid=(lo+hi)/2
        if feasible(mid):lo=mid
        else:hi=mid
    alpha=lo*(1-2e-12)
    return alpha*y,alpha


def make_seed(template,comparison,source,count,kind='circle',jitter=.001,residual_fraction=.15,seed=20260910):
    if count<len(source):raise ValueError(f'Anchored mode requires at least {len(source)} points.')
    rng=np.random.default_rng(seed)
    fr=template['frame'];B=np.array(fr['basis_columns_a_b_n']);m=np.array(fr['origin']);D=fr['diameter']
    uv=(source-m)@B/D
    curves=components(template,comparison,kind)
    residuals=comparison['fits']['minus_near_apex']['excluded']
    mirror=np.array(fr['mirror_permutation'])
    orbit=[i for i in residuals if i<mirror[i]]
    nres=2*int(round((count-len(source))*residual_fraction/2))
    nres=min(nres,2*((count-len(source))//2))
    remaining=count-len(source)-nres
    counts=np.zeros(len(curves),int)
    weights=np.array([c['length']*(2 if c['paired'] else 1) for c in curves])
    # Reserve a sample on each independent curve family when the budget allows.
    for i,c in enumerate(curves):
        cost=2 if c['paired'] else 1
        if remaining>=cost:counts[i]+=1;remaining-=cost
    while remaining:
        valid=np.array([not c['paired'] or remaining>=2 for c in curves])
        p=weights*valid;p/=p.sum()
        i=int(rng.choice(len(curves),p=p));counts[i]+=1;remaining-=2 if curves[i]['paired'] else 1
    proposals=[]
    for c,n in zip(curves,counts):
        if not n:continue
        frac=(np.arange(n)+rng.uniform(.12,.88,n))/n
        coords=sample_curve(c,frac)
        for j,y in enumerate(coords):
            perturb=rng.normal(0,jitter,3)
            if not c['paired']:perturb[2]=0.;y[2]=0.
            y=y+perturb
            pair=np.array([y,y*REF]) if c['paired'] else y[None]
            proposals.append((pair,{'family':c['name'],'arclength_fraction':float(frac[j])}))
    # Residual points remain structural anchors; add local convex combinations,
    # not samples of the poor global negative-tripod fit.
    for j in range(nres//2):
        i=orbit[j%len(orbit)];r=uv[i]
        if r[2]<0:i=int(mirror[i]);r=uv[i]
        neighbors=np.argsort(np.sum((uv-r)**2,axis=1))[1:6]
        q=rng.dirichlet(np.ones(len(neighbors)))@uv[neighbors]
        mix=rng.uniform(.08,.35);y=(1-mix)*r+mix*q
        proposals.append((np.array([y,y*REF]),{'family':'residual_reservoir','source_index':int(i),'mixture':float(mix)}))
    rng.shuffle(proposals)
    accepted=list(uv.copy());new_xyz=[];records=[]
    limit2=(1-1e-11)**2
    for yy,meta in proposals:
        z,alpha=contract_pair(yy,np.array(accepted),limit2)
        # Check original coordinates too: the frame is floating-point orthogonal.
        xyz=m+D*(z@B.T)
        existing=np.vstack((source,np.array(new_xyz).reshape(-1,3)))
        q=np.sum((xyz[:,None]-existing[None])**2,axis=2)
        if float(q.min())<=1e-24*D*D:
            raise ValueError('A proposed point nearly coincides with an existing point; change --seed.')
        if len(xyz)==2 and np.linalg.norm(xyz[0]-xyz[1])<=1e-12*D:
            raise ValueError('Near-coincident reflected pair; change --seed.')
        if float(q.max())>D*D or (len(xyz)==2 and np.sum((xyz[0]-xyz[1])**2)>D*D):
            raise ValueError('Floating-point contraction failed.')
        idx=len(source)+len(new_xyz)
        new_xyz.extend(xyz);accepted.extend(z)
        records.append({**meta,'indices':list(range(idx,idx+len(z))),'radial_contraction':float(alpha)})
    final=np.vstack((source,np.array(new_xyz).reshape(-1,3)))
    assert len(final)==count
    return final,records


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--points',type=int,default=201)
    p.add_argument('--kind',choices=['circle','parabola'],default='circle',help='Tripod branch type. Outer arcs stay circular.')
    p.add_argument('--jitter',type=float,default=.001,help='Symmetry-respecting Gaussian jitter, in old diameter units.')
    p.add_argument('--residual-fraction',type=float,default=.15,help='Fraction of additions near the ten unmodelled points.')
    p.add_argument('--seed',type=int,default=20260910)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--skip-exact-check',action='store_true',help='Skip the final quadratic-time integer geometry check.')
    a=p.parse_args(argv)
    if a.jitter<0 or not 0<=a.residual_fraction<=1:p.error('Invalid jitter or residual fraction.')
    if a.out.exists():p.error('Use a new output directory.')
    try:
        template=json.loads((ROOT/'templates/curves3.json').read_text())
        comparison=json.loads((ROOT/'templates/curves3_comparison.json').read_text())
        source=np.load(ROOT/'templates/source3.npy',allow_pickle=False)
        sourcehash=hashlib.sha256(np.asarray(source,dtype='<f8').tobytes()).hexdigest()
        if sourcehash!=template['input_array_sha256']:raise ValueError('Source coordinates do not match the template.')
        x,records=make_seed(template,comparison,source,a.points,a.kind,a.jitter,a.residual_fraction,a.seed)
        exact=None if a.skip_exact_check else exact_geometry(x,source)
        report={'source_array_sha256':sourcehash,'parameters':{**vars(a),'out':str(a.out)},
                'source_points':len(source),'total_points':len(x),'records':records,'exact_geometry':exact,
                'contracted_orbits':sum(r['radial_contraction']<1 for r in records),
                'min_contraction':min((r['radial_contraction'] for r in records),default=1.),
                'on_fitted_curves':a.jitter==0 and all(r['radial_contraction']==1 and r['family']!='residual_reservoir' for r in records),
                'qualification':'An anchored, geometry-checked INITIAL SEED. Not an improvement claim. '
                                 'New points need not lie on the fitted curves after jitter/contraction. '
                                 'The original rows are unchanged; additions respect reflection numerically, '
                                 'while the original set itself is only symmetric to floating precision.'}
        a.out.mkdir(parents=True)
        np.save(a.out/'points.npy',x)
        np.savetxt(a.out/'points.csv',x,delimiter=',',fmt='%.17g')
        # CSV is for inspection; exact geometry refers to the NPY binary values.
        report['canonical_coordinates']='points.npy; points.csv is a decimal convenience copy'
        (a.out/'sampling.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'points':len(x),'exact_geometry':exact,'contracted_orbits':report['contracted_orbits'],
                          'minimum_contraction':report['min_contraction'],'output':str(a.out)},indent=2))
    except (OSError,ValueError,KeyError) as e:p.error(str(e))

if __name__=='__main__':main()
