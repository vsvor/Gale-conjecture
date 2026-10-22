"""Robust, exception-preserving curve fits and diameter-safe resampling.

Models are proposal distributions, NOT restrictions on the geometric optimizer.
Every public fitting function returns a new model; it never projects its input.
All coordinate-preservation assertions in resample() use the canonical integers.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

import borsuk_search as bs
import tune_solution as ts


def digest(g: ts.Geometry) -> str:
    return hashlib.sha256(json.dumps([g.digits, g.rows], separators=(',', ':')).encode()).hexdigest()


def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-14 or not np.isfinite(norm):
        raise ValueError('Degenerate direction in a curve model.')
    return np.asarray(v, float) / norm


def value(curve: dict, t) -> np.ndarray:
    t = np.atleast_1d(t).astype(float)
    if curve['kind'] == 'circle':
        return np.asarray(curve['center']) + curve['radius'] * (
            np.cos(t)[:, None]*np.asarray(curve['u']) + np.sin(t)[:, None]*np.asarray(curve['v']))
    a, b, c = np.asarray(curve['coefficients'])
    return a + t[:, None]*b + t[:, None]**2*c


def closest(curve: dict, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest points on a finite circular or quadratic arc, including endpoints."""
    x = np.atleast_2d(x)
    lo, hi = curve['interval']
    if curve['kind'] == 'circle':
        z = x - curve['center']
        angle = np.arctan2(z @ np.asarray(curve['v']), z @ np.asarray(curve['u']))
        angle += 2*np.pi*np.round(((lo+hi)/2-angle)/(2*np.pi))
        choices = np.stack((np.clip(angle-2*np.pi, lo, hi), np.clip(angle, lo, hi),
                            np.clip(angle+2*np.pi, lo, hi), np.full(len(x), lo), np.full(len(x), hi)), axis=1)
        yy = value(curve, choices.ravel()).reshape(len(x), 5, 3)
        idx = np.sum((yy-x[:, None])**2, axis=2).argmin(axis=1)
        t = choices[np.arange(len(x)), idx]
        proj = yy[np.arange(len(x)), idx]
    else:
        a, b, c = np.asarray(curve['coefficients'])
        ts_ = []
        for row in x:
            r = a-row
            # (a+b*t+c*t^2-x).(b+2*c*t)=0
            poly = np.trim_zeros([2*np.dot(c,c), 3*np.dot(b,c),
                                  np.dot(b,b)+2*np.dot(r,c), np.dot(r,b)], 'f')
            roots = np.roots(poly) if len(poly)>1 else []
            candidates = [lo, hi] + [float(z.real) for z in roots
                                     if abs(z.imag) < 1e-8 and lo <= z.real <= hi]
            q = np.sum((value(curve, candidates)-row)**2, axis=1)
            ts_.append(candidates[int(np.argmin(q))])
        t = np.asarray(ts_)
        proj = value(curve, t)
    return t, proj, np.linalg.norm(proj-x, axis=1)


def arclength_grid(curve: dict, count: int = 513) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(*curve['interval'], count)
    if curve['kind'] == 'circle':
        s = (t-t[0])*curve['radius']
    else:
        points = value(curve, t)
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return t, s


def sample(curve: dict, fractions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, s = arclength_grid(curve)
    if s[-1] < 1e-12:
        raise ValueError('Zero-length sampling arc.')
    params = np.interp(np.asarray(fractions)*s[-1], s, t)
    return params, value(curve, params)


def transform(model: dict, new_x: np.ndarray) -> dict:
    """Register the old model by known ROW CORRESPONDENCES, not nearest neighbors.

    Existing rows are never reordered/deleted within a lineage. A similarity
    registration removes the harmless normalization/rigid motion of optimizers.
    """
    out = copy.deepcopy(model)
    old = np.asarray(model['reference'])
    if len(new_x) < len(old):
        raise ValueError('A lineage lost or reordered points; start a fresh grouped run.')
    new = new_x[:len(old)]
    co, cn = old.mean(axis=0), new.mean(axis=0)
    u, _, vt = np.linalg.svd((old-co).T @ (new-cn))
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1; rot = u @ vt
    denom = float(np.sum((old-co)**2))
    scale = float(np.sum(((old-co)@rot)*(new-cn))/denom)
    if not 0.1 < scale < 10:
        raise ValueError('Model registration scale is implausible; wrong input/lineage.')
    def pos(p): return ((np.asarray(p)-co)*scale@rot+cn).tolist()
    def vec(p): return (np.asarray(p)@rot).tolist()
    for c in out['curves']:
        if c['kind']=='circle':
            c['center'] = pos(c['center']); c['u'] = vec(c['u']); c['v'] = vec(c['v'])
            c['radius'] *= scale
        else:
            a,b,q = c['coefficients']
            c['coefficients'] = [pos(a), (scale*np.asarray(b)@rot).tolist(), (scale*np.asarray(q)@rot).tolist()]
    sym = out.get('symmetry')
    if sym:
        sym['origin'] = pos(sym['origin']); sym['normal'] = vec(sym['normal'])
    out['reference'] = new_x.tolist()
    out['registration_rms'] = float(np.sqrt(np.mean(np.sum((((old-co)*scale@rot+cn)-new)**2, axis=1))))
    return out


def reflect(x: np.ndarray, sym: dict) -> np.ndarray:
    n = unit(np.asarray(sym['normal'])); o = np.asarray(sym['origin'])
    x = np.asarray(x)
    return x-2*((x-o)@n)[..., None]*n


def reflected_curve(curve: dict, sym: dict, name: str) -> dict:
    out = copy.deepcopy(curve); out['name'] = name
    n = unit(np.asarray(sym['normal']))
    def v(a):
        a = np.asarray(a); return (a-2*np.dot(a,n)*n).tolist()
    if curve['kind']=='circle':
        out['center'] = reflect(np.array(curve['center']), sym).tolist()
        out['u'], out['v'] = v(curve['u']), v(curve['v'])
    else:
        a,b,c = curve['coefficients']
        out['coefficients'] = [reflect(np.array(a),sym).tolist(),v(b),v(c)]
    return out


def symmetry_error(model: dict, x: np.ndarray, diameter: float) -> float | None:
    sym = model.get('symmetry')
    if not sym: return None
    perm = sym.get('point_permutation', [])
    valid = [(i,j) for i,j in enumerate(perm) if 0<=j<len(x)]
    if not valid: return None
    i,j = np.asarray(valid).T
    known=float(np.max(np.linalg.norm(reflect(x[i],sym)-x[j],axis=1))/diameter)
    all_points=float(cKDTree(x).query(reflect(x,sym))[0].max()/diameter)
    return max(known,all_points)


def labels_for(model: dict, x: np.ndarray, protected: list[int], tolerance: float,
               diameter: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projections = [closest(c,x) for c in model['curves']]
    residuals = np.stack([p[2] for p in projections],axis=1)
    labels = residuals.argmin(axis=1)
    err = residuals[np.arange(len(x)),labels]
    params = np.stack([p[0] for p in projections],axis=1)[np.arange(len(x)),labels]
    labels[err > tolerance*diameter] = -1
    labels[np.asarray(protected, int)] = -1
    return labels, params, err/diameter


def fit_curve(curve: dict, x: np.ndarray, diameter: float, tau: float,
              iterations: int, sym: dict | None = None) -> tuple[dict, dict]:
    """Locally regularized geometric fit. Crowded arclength bins are downweighted."""
    old = copy.deepcopy(curve)
    pars, _, errs = closest(old,x)
    lo,hi = old['interval']
    # Normalize optimization parameters by D and trust-region rotation scale.
    tgrid,sgrid = arclength_grid(old)
    pos = np.interp(pars,tgrid,sgrid) / max(sgrid[-1],1e-12)
    bins = np.clip((pos*12).astype(int),0,11)
    weights = 1/np.bincount(bins,minlength=12)[bins]
    weights /= weights.sum()
    central = sym is not None and old.get('mirror')==old['name']
    if old['kind']=='circle':
        c0 = np.asarray(old['center']); u0 = np.asarray(old['u']); v0 = np.asarray(old['v'])
        if central:
            # Fixed symmetry plane, with in-plane center shifts.
            def build(p):
                c = copy.deepcopy(old); c['center'] = (c0+diameter*(p[0]*u0+p[1]*v0)).tolist()
                c['radius'] = float(old['radius']*np.exp(p[2])); return c
            p0 = np.zeros(3); lower=[-.12,-.12,-.5];upper=[.12,.12,.5]
        else:
            def build(p):
                c = copy.deepcopy(old); r = Rotation.from_rotvec(p[3:6]).as_matrix()
                c['center'] = (c0+diameter*p[:3]).tolist()
                c['u'] = (r@u0).tolist(); c['v'] = (r@v0).tolist()
                c['radius'] = float(old['radius']*np.exp(p[6]));return c
            p0 = np.zeros(7); lower=[-.12]*3+[-.5]*4;upper=[.12]*3+[.5]*4
    else:
        coeff = np.asarray(old['coefficients'])
        if central:
            n = unit(np.asarray(sym['normal'])); d = unit(coeff[1]-np.dot(coeff[1],n)*n)
            f = np.cross(n,d); basis=np.array([d,f])
            def build(p):
                c=copy.deepcopy(old);c['coefficients']=(coeff+diameter*p.reshape(3,2)@basis).tolist();return c
            p0=np.zeros(6)
        else:
            def build(p):
                c=copy.deepcopy(old);c['coefficients']=(coeff+diameter*p.reshape(3,3)).tolist();return c
            p0=np.zeros(9)
        lower=np.full(len(p0),-.12);upper=np.full(len(p0),.12)
    def objective(p):
        # soft_l1 is applied to weighted Cartesian geometric residuals.
        _, yy, _ = closest(build(p), x)
        return np.r_[((yy-x)/diameter*np.sqrt(weights[:,None])).ravel(), .01*p]
    initial = objective(p0)
    res=least_squares(objective,p0,bounds=(lower,upper),loss='soft_l1',
                      f_scale=tau/(3*math.sqrt(len(x))),max_nfev=iterations,
                      ftol=1e-8,xtol=1e-8,gtol=1e-8)
    candidate=build(res.x)
    # Do not accept a larger robust objective even on budget exhaustion.
    def cost(v):
        f=tau/(3*math.sqrt(len(x)));return float(np.sum(2*f*f*(np.sqrt(1+(v/f)**2)-1)))
    accepted=cost(objective(res.x)) <= cost(initial)
    fitted = candidate if accepted else old
    t, _, err = closest(fitted,x)
    # Restrict to supported arc ranges, not arbitrary full circles or extrapolation.
    if len(x)>=4 and np.ptp(t)>1e-7:
        span=float(np.ptp(t)); fitted['interval']=[float(max(lo,t.min()-.02*span)),float(min(hi,t.max()+.02*span))]
    return fitted, {'points':len(x),'success':bool(res.success),'accepted':accepted,
                    'nfev':int(res.nfev),'message':str(res.message),
                    'rms_before':float(np.sqrt(np.mean(errs**2))/diameter),
                    'rms_after':float(np.sqrt(np.mean(err**2))/diameter),
                    'bound_reached':bool(np.any(np.abs(res.active_mask)>0))}


def refit(model: dict, x: np.ndarray, protected: list[int], tau: float=.012,
          iterations: int=60, symmetry: str='auto', symmetry_tol: float=2e-4) -> tuple[dict,dict]:
    x=np.asarray(x); before=x.copy()
    out=transform(model,x)
    diameter=math.sqrt(float(bs.distances2(x,bs.pairs_for(len(x))).max()))
    error=symmetry_error(out,x,diameter)
    symmetric=bool(symmetry=='auto' and error is not None and error<=symmetry_tol)
    reports=[]
    # Two fit/reclassify sweeps. Large-residual points never drag a curve across a gap.
    for sweep in range(2):
        labels,_,_=labels_for(out,x,protected,3*tau,diameter)
        visited=set()
        byname={c['name']:j for j,c in enumerate(out['curves'])}
        for i,curve in enumerate(list(out['curves'])):
            if i in visited: continue
            visited.add(i); ids=np.flatnonzero(labels==i)
            xx=x[ids]
            mate=byname.get(curve.get('mirror')) if symmetric else None
            if mate is not None and mate!=i:
                visited.add(mate)
                xx=np.vstack((xx,reflect(x[labels==mate],out['symmetry'])))
            if len(xx)<4:
                reports.append({'sweep':sweep,'curve':curve['name'],'points':len(xx),'status':'too_few_inliers_keep_previous'})
                continue
            c,r=fit_curve(curve,xx,diameter,tau,iterations,out['symmetry'] if symmetric else None)
            out['curves'][i]=c
            if mate is not None and mate!=i:
                cm=reflected_curve(c,out['symmetry'],out['curves'][mate]['name']);cm['mirror']=c['name']
                out['curves'][mate]=cm
            reports.append({'sweep':sweep,'curve':curve['name'],**r})
    labels,t,err=labels_for(out,x,protected,tau,diameter)
    out['reference']=x.tolist();out['labels']=labels.tolist();out['parameters']=t.tolist()
    out['protected']=list(map(int,protected));out['symmetry_active']=symmetric
    for j,c in enumerate(out['curves']):
        keep=labels==j
        c['support']=int(keep.sum())
        c['trusted_rms']=float(np.sqrt(np.mean(err[keep]**2))) if keep.any() else None
    report={'tolerance_in_diameter_units':tau,'exceptions':np.flatnonzero(labels<0).tolist(),
            'protected_exceptions':protected,'inliers':int(np.sum(labels>=0)),
            'symmetry_applied_to_fit':symmetric,'symmetry_error':error,
            'trusted_rms':float(np.sqrt(np.mean(err[labels>=0]**2))) if (labels>=0).any() else None,
            'point_residuals':err.tolist(),'labels':labels.tolist(),'curve_fits':reports,
            'input_coordinates_unchanged':bool(np.array_equal(x,before)),
            'qualification':'Numerical proposal-model fit. No input point is projected or deleted.'}
    if not np.array_equal(x,before):raise AssertionError('Fitting changed its observations.')
    return out,report


def greedy_partition(g: ts.Geometry, old: np.ndarray, k: int) -> np.ndarray:
    """Extend an existing partition heuristically; all outputs are valid partitions."""
    c=list(map(int,old))
    for i in range(len(c),g.m):
        scores=[]
        for color in range(k):
            scores.append(max((sum((a-b)**2 for a,b in zip(g.rows[i],g.rows[j]))
                               for j in range(i) if c[j]==color),default=0))
        c.append(int(np.argmin(scores)))
    return np.asarray(c,np.int32)


def add_exact(g: ts.Geometry, yy: np.ndarray, accepted: list[list[int]],
              center: np.ndarray, min_sep: float) -> tuple[list[list[int]],float] | None:
    """Contract only new points, including mutual conflicts, then check integer distances."""
    yy=np.atleast_2d(yy); D=math.sqrt(float(bs.distances2(g.x,g.pairs).max()))
    current=np.asarray(accepted,dtype=float)/10**g.digits
    directions=yy-center
    def feasible(a):
        z=center+a*directions
        return (np.sum((z[:,None]-current[None])**2,axis=2).max() <= (D*(1-1e-10))**2
                and np.sum((z[:,None]-z[None])**2,axis=2).max() <= (D*(1-1e-10))**2)
    if feasible(1.):alpha=1.
    else:
        if not feasible(0.):raise ValueError('Diameter-midpoint contraction center is infeasible.')
        lo,hi=0.,1.
        for _ in range(48):
            mid=(lo+hi)/2
            if feasible(mid):lo=mid
            else:hi=mid
        alpha=lo*(1-1e-9)
    for _ in range(8):
        z=center+alpha*directions
        rows=[[int(format(float(a),f'.{g.digits}f').replace('.','')) for a in row] for row in z]
        sofar=list(accepted);good=True
        for r in rows:
            qs=[sum((a-b)**2 for a,b in zip(r,s)) for s in sofar]
            if max(qs)>g.diameter2:good=False;break
            if min(qs)<=g.diameter2*min_sep**2 or r in sofar:return None
            sofar.append(r)
        if good:return rows,alpha
        alpha*=.999
    return None


def resample(state: ts.State, model: dict, count: int, rng: np.random.Generator,
             jitter: float=.001, exception_fraction: float=.2, min_sep: float=.001,
             respect_symmetry: bool=True) -> tuple[ts.Geometry,dict,dict]:
    """Append exact-diameter-safe points; never replace any observation by a fit.

    Coverage is chosen by arclength gaps after insertions, rather than repeatedly
    taking the first few high-scoring sites. Near-active witness endpoints get
    extra weight. Insufficient spacing returns a smaller batch, explicitly logged.
    """
    if count<0:raise ValueError('Negative addition count.')
    out=copy.deepcopy(model);g=state.g
    if count==0:return g,out,{'added':0,'requested':0,'original_rows_unchanged':True,'diameter_exactly_unchanged':True,'records':[]}
    # Increase the decimal grid for integer/low-precision CSV inputs, without
    # changing any original rational coordinate. Otherwise new points could
    # only occupy the old coarse lattice.
    input_digits=g.digits
    if g.digits<17:
        multiplier=10**(17-g.digits)
        g=ts.Geometry([[v*multiplier for v in r] for r in g.rows],17)
    source_rows=[r.copy() for r in g.rows]
    D=math.sqrt(float(bs.distances2(g.x,g.pairs).max())); ij=g.pairs[int(np.argmax(g.q))]
    center=(g.x[ij[0]]+g.x[ij[1]])/2
    sym=model.get('symmetry');symmetric=respect_symmetry and model.get('symmetry_active',False)
    if symmetric:
        # Project ONLY the contraction center, not data. Validate exact feasibility later.
        center=(center+reflect(center,sym))/2
        if np.max(np.sum((g.x-center)**2,axis=1)) >= (D*(1-1e-8))**2:
            symmetric=False;center=(g.x[ij[0]]+g.x[ij[1]])/2
    labels=np.asarray(model['labels']);byname={c['name']:i for i,c in enumerate(model['curves'])}
    allowed=[];visited=set()
    for i,c in enumerate(model['curves']):
        if i in visited or c.get('support',0)<3:continue
        mate=byname.get(c.get('mirror')) if symmetric else None
        if mate is not None:visited.add(mate)
        visited.add(i); allowed.append((i,mate))
    exceptions=np.flatnonzero(labels<0).tolist()
    endpoints=np.zeros(g.m)
    if len(state.witness):
        short=[i for i in state.witness if g.q[int(i)] <= min(g.q[int(j)] for j in state.witness)+g.diameter2*1e-7]
        if short: np.add.at(endpoints,g.pairs[short].ravel(),1)
    weights=[]
    for i,mate in allowed:
        _,ss=arclength_grid(model['curves'][i]);mask=labels==i
        weights.append(max(ss[-1]/D,.05)*(1+np.mean(endpoints[mask]) if mask.any() else 1))
    rows=[r.copy() for r in g.rows];new_records=[];new_labels=[];attempts=0;used={i:0 for i,_ in allowed}
    # A density-independent reservoir. Protections affect fitting, never geometry.
    num_res=int(round(count*exception_fraction)) if exceptions else 0
    res_used=0
    while len(rows)<g.m+count and attempts<max(100,50*count):
        attempts+=1;remaining=g.m+count-len(rows)
        reservoir=(not allowed or (res_used<num_res and (num_res-res_used>=remaining or rng.random()<max(exception_fraction,.25))))
        if reservoir:
            if not exceptions:break
            i=int(rng.choice(exceptions))
            near=np.argsort(np.sum((g.x-g.x[i])**2,axis=1))[1:min(g.m,6)]
            neighbor=rng.dirichlet(np.ones(len(near)))@g.x[near]
            mix=rng.uniform(.05,.25);y=(1-mix)*g.x[i]+mix*neighbor+rng.normal(0,jitter*D,3)
            yy=y[None];labs=[-1];tag={'family':'exception_neighborhood','source_index':i}
            if symmetric and remaining>=2 and np.linalg.norm(y-reflect(y,sym))>min_sep*D:
                yy=np.array([y,reflect(y,sym)]);labs=[-1,-1]
            cost=len(yy)
        else:
            valid=[j for j,(i,mate) in enumerate(allowed) if not (mate is not None and mate!=i and remaining==1)]
            if not valid:
                # Odd cardinalities can always use an exception/singleton proposal.
                symmetric=False;continue
            # Initially cover every family, subsequently balance arclength/active weight.
            weights_now=np.asarray([weights[j]/(1+used[allowed[j][0]]) for j in valid]);weights_now/=weights_now.sum()
            j=valid[int(rng.choice(len(valid),p=weights_now))];i,mate=allowed[j];curve=model['curves'][i]
            t,_,err=closest(curve,np.asarray(rows,dtype=float)/10**g.digits)
            tt,ss=arclength_grid(curve)
            covered=np.interp(t[err<=.025*D],tt,ss)/ss[-1]
            cuts=np.unique(np.r_[0.,covered,1.]);gaps=np.diff(cuts)
            if not len(gaps):continue
            probs=gaps**2; probs/=probs.sum()
            slot=int(rng.choice(len(gaps),p=probs));frac=cuts[slot]+rng.uniform(.2,.8)*gaps[slot]
            par,coords=sample(curve,np.array([frac]));y=coords[0]+rng.normal(0,jitter*D,3)
            yy=y[None];labs=[i]
            if symmetric and mate==i:
                y=(y+reflect(y,sym))/2;yy=y[None]
            elif symmetric and mate is not None:
                yy=np.array([y,reflect(y,sym)]);labs=[i,mate]
            cost=len(yy);tag={'family':curve['name'],'arclength_fraction':float(frac),'curve_parameter':float(par[0])}
            used[i]+=1
        candidate=add_exact(g,yy,rows,center,min_sep)
        if candidate is None:continue
        extra,alpha=candidate;start=len(rows);rows.extend(extra);new_labels.extend(labs)
        new_records.append({**tag,'indices':list(range(start,len(rows))),'contraction':alpha})
        if reservoir:res_used+=cost
        if sym:
            perm=out['symmetry']['point_permutation']
            while len(perm)<start:perm.append(-1)
            if cost==2 and symmetric:perm.extend([start+1,start])
            elif symmetric and not reservoir:perm.append(start)
            else:perm.extend([-1]*cost)
    newg=ts.Geometry(rows,g.digits)
    if newg.rows[:g.m]!=source_rows or newg.diameter2!=g.diameter2:
        raise AssertionError('A distillation seed changed old coordinates or the exact diameter.')
    out['labels']=model['labels']+new_labels
    # Keep prior reference for registering after unconstrained optimization.
    # Extend it now with the actual additions, not their pre-contraction curve sites.
    out['reference']=newg.x.tolist()
    report={'requested':count,'added':newg.m-g.m,'attempts':attempts,'records':new_records,
            'original_rows_unchanged':True,'diameter_exactly_unchanged':True,
            'canonical_decimal_digits':g.digits,'source_input_decimal_digits':input_digits,'source_digest':digest(state.g),'seed_digest':digest(newg),
            'source_diameter_squared_integer':str(g.diameter2),'seed_diameter_squared_integer':str(newg.diameter2),
            'symmetry_used_for_sampling':symmetric,'exception_additions':res_used,
            'min_separation_requested':min_sep,
            'qualification':'Certificate of insertion geometry, not a strict ratio gain. Old points were not projected.'}
    return newg,out,report
