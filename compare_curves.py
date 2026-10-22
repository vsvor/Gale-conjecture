#!/usr/bin/env python3
"""Equal-flexibility circular/parabolic branch fits through the observed tripod apices.
The comparison fixes the same reflection, point subsets, and branch memberships.
Negative near-apex subset = points within a radius (default 0.25 original diameters).
These geometric fits do not optimize or certify the partition ratio.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os
from pathlib import Path
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[k] = '1'
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parent
REF=np.array([1.,1.,-1.])


def basis(p, sign, side):
    """Unit tangent d and unit curvature direction n perpendicular to d."""
    alpha=float(p[0])
    if side:
        beta,psi=map(float,p[1:3])
        d=np.array([np.sin(alpha),-np.cos(alpha)*np.cos(beta),np.cos(alpha)*np.sin(beta)])
        n=np.array([-float(sign),0.,0.]); n-= (n@d)*d; n/=np.linalg.norm(n)
        n=n*np.cos(psi)+np.cross(d,n)*np.sin(psi)
    else:
        d=np.array([np.sin(alpha),np.cos(alpha),0.])
        n=-sign*np.array([np.cos(alpha),-np.sin(alpha),0.])
    return d,n


def parabola_parameters(q,d,n,k):
    # d/dt ||q - t*d - k*t^2*n/2||^2 = 0.
    u=q@d; v=q@n
    A=0.5*k*k
    pp=(1-k*v)/A; qq=-u/A
    disc=(qq/2)**2+(pp/3)**3
    z=np.sqrt(np.maximum(disc,0.))
    t=np.cbrt(-qq/2+z)+np.cbrt(-qq/2-z)
    roots=np.tile(t[:,None],(1,3))
    ids=disc<0
    if np.any(ids):
        p=pp[ids]; q0=qq[ids]
        angle=np.arccos(np.clip((-q0/2)/np.sqrt(-(p/3)**3),-1,1))
        roots[ids]=2*np.sqrt(-p/3)[:,None]*np.cos((angle[:,None]+2*np.pi*np.arange(3))/3)
    roots=np.column_stack((np.maximum(roots,0.),np.zeros(len(q))))
    curves=roots[:,:,None]*d+0.5*k*roots[:,:,None]**2*n
    err=np.sum((curves-q[:,None,:])**2,axis=2)
    return roots[np.arange(len(q)),err.argmin(axis=1)]


def project(q,p,sign,side,kind):
    d,n=basis(p,sign,side)
    if kind=='line':
        t=np.maximum(q@d,0.)
        return t[:,None]*d,t
    k=np.exp(p[-1])
    if kind=='parabola':
        t=parabola_parameters(q,d,n,k)
        y=t[:,None]*d+0.5*k*t[:,None]**2*n
    else:
        R=1/k; u=q@d; v=q@n
        angles=np.arctan2(u,R-v)
        candidates=np.column_stack((np.clip(angles,0,np.pi),np.zeros(len(q)),np.full(len(q),np.pi)))
        curves=np.sin(candidates)[:,:,None]/k*d+(1-np.cos(candidates))[:,:,None]/k*n
        j=np.sum((curves-q[:,None,:])**2,axis=2).argmin(axis=1)
        t=candidates[np.arange(len(q)),j]/k
        y=curves[np.arange(len(q)),j]
    return y,t


def fit_branch(q,sign,side,kind,restarts,seed):
    rng=np.random.default_rng(seed)
    if side:
        guess=np.array([0.,1.,0.])
        lo=[-1.4,.02,-np.pi];hi=[1.4,np.pi-.02,np.pi]
    else:
        guess=np.array([0.]);lo=[-1.4];hi=[1.4]
    if kind!='line':
        guess=np.r_[guess,np.log(2.)];lo+=[np.log(.05)];hi+=[np.log(30.)]
    lo=np.array(lo);hi=np.array(hi)
    best=None; all_runs=[]
    for j in range(restarts):
        p0=guess.copy()
        if j:
            p0+=rng.normal(0,.3,len(p0))
            if side and j%4==0:p0[1]=rng.uniform(.3,2.7)
        p0=np.clip(p0,lo+1e-5,hi-1e-5)
        ans=least_squares(lambda p:(project(q,p,sign,side,kind)[0]-q).ravel(),p0,
                          bounds=(lo,hi),ftol=1e-12,gtol=1e-12,xtol=1e-12,max_nfev=600)
        all_runs.append({'cost':float(2*ans.cost),'success':bool(ans.success),'nfev':ans.nfev})
        if best is None or ans.cost<best.cost:best=ans
    p=best.x;feet,t=project(q,p,sign,side,kind);d,n=basis(p,sign,side)
    errors=np.linalg.norm(feet-q,axis=1)
    return {'kind':kind,'shape_parameters':p.tolist(),'tangent':d.tolist(),'normal':n.tolist(),
            'curvature':None if kind=='line' else float(np.exp(p[-1])),
            'parameter_interval':[0.,float(max(t,default=0.))],
            'rms':float(np.sqrt(np.mean(errors**2))),'max':float(errors.max()),
            'distance_residuals':errors.tolist(),'point_parameters':t.tolist(),'runs':all_runs,
            'shape_dof':len(p)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--radius',type=float,default=.25)
    parser.add_argument('--restarts',type=int,default=16)
    parser.add_argument('--seed',type=int,default=20260910)
    parser.add_argument('--out',type=Path,default=ROOT/'experiments/curve_comparison.json')
    args=parser.parse_args(argv)
    if args.out.exists(): parser.error('Use a new output file; an existing fit must not be overwritten.')
    if args.radius<=0 or args.restarts<1:parser.error('Radius and restarts must be positive.')
    source=np.load(ROOT/'templates/source3.npy',allow_pickle=False)
    model=json.loads((ROOT/'templates/curves3.json').read_text())
    if hashlib.sha256(np.asarray(source,dtype='<f8').tobytes()).hexdigest()!=model['input_array_sha256']:
        raise ValueError('Source fingerprint differs from the fitted 81-point configuration.')
    fr=model['frame'];y=(source-np.array(fr['origin']))@np.array(fr['basis_columns_a_b_n'])/fr['diameter']
    result={'radius':args.radius,'source_array_sha256':model['input_array_sha256'],'frame':fr,'fits':{},
            'method':'One-sided planar curves with the same fixed junctions, same branch assignments, '
                     'same point weights and same 6 shape degrees of freedom per curved tripod. '
                     'Fitted local minima, not globally certified.'}
    for label,sgn in [('plus',1),('minus',-1)]:
        group=model['geometric_groups']['tripod_'+label]
        J=np.array([sgn*.5,0.,0.])
        inliers=[i for i in group if np.linalg.norm(y[i]-J)<=args.radius]
        for subset,idx in [('all',group),('near_apex',inliers)]:
            if label=='plus' and subset=='all': continue
            data={};sides={}
            middle=[i for i in idx if abs(y[i,2])<1e-9 and np.linalg.norm(y[i]-J)>1e-9]
            positive=[i for i in idx if y[i,2]>1e-9]
            for kind in ['line','circle','parabola']:
                branches=[]
                for nm,ids,side in [('central',middle,False),('side',positive,True)]:
                    if len(ids)<2:raise ValueError('Insufficient samples on branch.')
                    # Average reflected observations, with equal weight per orbit member.
                    q=y[ids]-J
                    if side:
                        partner=np.array(fr['mirror_permutation'])[ids]
                        q=(y[ids]+y[partner]*REF)/2-J
                    b=fit_branch(q,sgn,side,kind,args.restarts,args.seed+sgn+len(ids))
                    b.update({'branch':nm,'indices':ids,'junction':J.tolist(),'sign':sgn})
                    branches.append(b)
                errs=branches[0]['distance_residuals']+branches[1]['distance_residuals']*2+[0.]
                data[kind]={'rms':float(np.sqrt(np.mean(np.array(errs)**2))),
                            'max':max(errs),'n':len(idx),'branches':branches}
                print(label,subset,kind,'n',len(idx),'rms',data[kind]['rms'],'max',data[kind]['max'],flush=True)
            result['fits'][label+'_'+subset]={'indices':idx,'excluded':[i for i in group if i not in idx],
                                            'models':data}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
