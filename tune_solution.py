#!/usr/bin/env python3
"""Delete/insert/polish a saved Borsuk min-max-ratio configuration.

Requires the accompanying borsuk_search.py, NumPy and SciPy. Run --help.
All accepted bests use exact integer squared-distance comparisons. SLSQP and
its first-order audit are numerical, fixed-graph diagnostics, not proofs of
local or global optimality. Timeout never means non-colourability.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = os.environ.get('BORSUK_BLAS_THREADS', '1')

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import logsumexp
try:
    import borsuk_search as bs
except ImportError as exc:
    raise SystemExit('Put tune_solution.py beside borsuk_search.py (both are in the ZIP).') from exc

VERSION = '1.0'


def edge_ids(m: int, edges: np.ndarray) -> np.ndarray:
    """Indices in numpy.triu_indices(m, 1), without constructing a dictionary."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not len(e):
        return np.empty(0, dtype=np.int32)
    u, v = np.minimum(e[:, 0], e[:, 1]), np.maximum(e[:, 0], e[:, 1])
    if np.any(u < 0) or np.any(v >= m) or np.any(u == v):
        raise ValueError('Invalid edge endpoints.')
    return np.asarray(u * (2*m-u-1)//2 + v-u-1, dtype=np.int32)


@dataclass
class Geometry:
    rows: list[list[int]]
    digits: int

    def __post_init__(self) -> None:
        self.m = len(self.rows)
        self.dim = len(self.rows[0]) if self.rows else 0
        if self.m < 2 or self.dim < 1 or any(len(r) != self.dim for r in self.rows):
            raise ValueError('Expected at least two points of consistent dimension.')
        if len(set(map(tuple, self.rows))) != self.m:
            raise ValueError('Coincident decimal coordinates; candidate rejected.')
        self.x = np.asarray([[float(Decimal(a).scaleb(-self.digits)) for a in r]
                             for r in self.rows])
        bs.validate_points(self.x)
        self.pairs = bs.pairs_for(self.m)
        self.q = [sum((a-b)**2 for a, b in zip(self.rows[int(u)], self.rows[int(v)]))
                  for u, v in self.pairs]
        self.diameter2 = max(self.q)
        if self.diameter2 == 0:
            raise ValueError('Zero diameter.')

    @classmethod
    def from_float(cls, x: np.ndarray, digits: int = 17) -> Geometry:
        x = bs.validate_points(x)
        rows = [[int(format(float(a), f'.{digits}f').replace('.', '')) for a in r]
                for r in x]
        return cls(rows, digits)

    def subset(self, keep: np.ndarray) -> Geometry:
        return Geometry([self.rows[int(i)].copy() for i in keep], self.digits)

    def partition_max(self, colors: np.ndarray) -> int:
        return max((q for (u, v), q in zip(self.pairs, self.q)
                    if colors[u] == colors[v]), default=0)

    def threshold(self, ratio2: Fraction, inclusive: bool = True) -> np.ndarray:
        a, b = ratio2.numerator*self.diameter2, ratio2.denominator
        return np.array([i for i, q in enumerate(self.q)
                         if (q*b >= a if inclusive else q*b > a)], dtype=np.int32)


@dataclass
class State:
    g: Geometry
    witness: np.ndarray
    colors: np.ndarray
    lower2: Fraction
    upper2: Fraction
    unresolved: bool = False
    polish: dict[str, Any] | None = None

    @property
    def lower(self) -> float:
        return math.sqrt(float(self.lower2))

    @property
    def upper(self) -> float:
        return math.sqrt(float(self.upper2))

    @property
    def exact(self) -> bool:
        return self.lower2 == self.upper2


def make_state(g: Geometry, witness: np.ndarray, colors: np.ndarray,
               unresolved: bool = False) -> State:
    witness = np.unique(np.asarray(witness, dtype=np.int32))
    colors = np.asarray(colors, dtype=np.int32)
    low = min((g.q[int(i)] for i in witness), default=0)
    high = g.partition_max(colors)
    if low > high:
        raise RuntimeError('Contradictory bounds: witness/model or edge mapping is invalid.')
    return State(g, witness, colors, Fraction(low, g.diameter2),
                 Fraction(high, g.diameter2), unresolved)


class OraclePool:
    """Bound the number of live incremental solvers when point counts change."""
    def __init__(self, k: int, args: argparse.Namespace):
        self.k, self.args = k, args
        self.items: OrderedDict[int, bs.GraphOracle] = OrderedDict()
        self.retired = {'calls': 0, 'unknowns': 0, 'cache_hits': 0}

    def get(self, m: int) -> bs.GraphOracle:
        if m not in self.items:
            a = self.args
            self.items[m] = bs.GraphOracle(m, self.k, a.backend, a.solver,
                                          a.sat_timeout, a.pysat_solver)
        self.items.move_to_end(m)
        while len(self.items) > 6:
            _, old = self.items.popitem(last=False)
            for key in self.retired:
                self.retired[key] += getattr(old, key)
            old.close()
        return self.items[m]

    def stats(self) -> dict[str, int]:
        return {key: val + sum(getattr(o, key) for o in self.items.values())
                for key, val in self.retired.items()}

    def close(self) -> None:
        for oracle in self.items.values():
            oracle.close()
        self.items.clear()


def assess(g: Geometry, pool: OraclePool, witness: np.ndarray | None = None,
           colors: np.ndarray | None = None, witness_known: bool = False) -> State:
    """Binary-search integer distances, seeded by an UNSAT witness and a partition.

    If a query times out, retain a rigorous lower/upper bracket. With a timeout,
    best replacement still needs new.lower >= old.upper, not merely a better
    unverified estimate. The graph K_m is the initial, mathematically UNSAT base.
    """
    m, k = g.m, pool.k
    oracle = pool.get(m)
    if m <= k:
        return make_state(g, np.empty(0, np.int32), np.arange(m))
    if colors is None:
        colors = np.arange(m, dtype=np.int32) % k
    else:
        colors = np.array(colors, dtype=np.int32, copy=True)
        if colors.shape != (m,) or np.any(colors < 0) or np.any(colors >= k):
            raise ValueError('Invalid partition.')
    best_w = np.arange(len(g.q), dtype=np.int32)
    low = min(g.q)
    high = g.partition_max(colors)
    unresolved = False
    if witness is not None and len(witness):
        w = np.unique(np.asarray(witness, dtype=np.int32))
        if np.any(w < 0) or np.any(w >= len(g.q)):
            raise ValueError('Invalid witness IDs.')
        ans = bs.Answer(bs.Status.UNSAT) if witness_known else oracle.solve(w)
        if ans.status is bs.Status.UNSAT:
            best_w = w
            low = min(g.q[int(i)] for i in w)
        elif ans.status is bs.Status.SAT:
            raise ValueError('The supplied witness is k-colourable.')
        else:
            unresolved = True
    levels = sorted(set(g.q))
    import bisect
    while low < high:
        left = bisect.bisect_right(levels, low)
        right = bisect.bisect_right(levels, high)
        if left >= right:
            raise RuntimeError('Inconsistent integer threshold bracket.')
        value = levels[(left+right-1)//2]
        # Include the WHOLE tied-distance level. Testing a partial tie without
        # retaining prefix indices could otherwise fail to advance the bracket.
        ids = np.array([i for i, q in enumerate(g.q) if q >= value], dtype=np.int32)
        ans = oracle.solve(ids)
        if ans.status is bs.Status.SAT:
            if ans.colors is None:
                unresolved = True
                break
            colors = ans.colors.copy()
            high = g.partition_max(colors)
        elif ans.status is bs.Status.UNSAT:
            best_w = (ans.core.copy() if ans.core is not None and len(ans.core)
                      else ids.copy())
            low = min(g.q[int(i)] for i in best_w)
        else:
            unresolved = True
            break
    return make_state(g, best_w, colors, unresolved)


def load_source(path: Path) -> tuple[Geometry, np.ndarray | None, np.ndarray | None, int | None]:
    if path.is_dir() and not (path/'points.csv').exists() and (path/'best').is_dir():
        path = path/'best'
    directory = path if path.is_dir() else path.parent
    source = path/'points.csv' if path.is_dir() else path
    if source.suffix.lower() == '.csv':
        rows_d = [[Decimal(v.strip()) for v in line.split(',')]
                  for line in source.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith('#')]
        if not rows_d or any(not v.is_finite() for row in rows_d for v in row):
            raise ValueError('CSV must contain finite numeric coordinates without a header.')
        digits = max(0, max(-v.as_tuple().exponent for row in rows_d for v in row))
        if digits > 1000:
            raise ValueError('Coordinate precision above 1000 decimals is not supported.')
        # Fraction avoids rounding under the decimal module default context.
        rows = [[int(Fraction(v)*10**digits) for v in row] for row in rows_d]
        g = Geometry(rows, digits)
    else:
        g = Geometry.from_float(bs.normalize(bs.load_points(source)))
    w, colors, saved_k = None, None, None
    # Sidecars must correspond to the selected canonical filename, not any
    # arbitrary coordinate file that happens to sit in the same directory.
    if source.name in ('points.csv', 'points.npy') and (directory/'certificate.json').exists():
        meta = json.loads((directory/'certificate.json').read_text())
        saved_k = int(meta['k'])
        ep = directory/'witness_edges.csv'
        if ep.exists() and ep.read_text().strip():
            w = edge_ids(g.m, np.loadtxt(ep, delimiter=',', dtype=np.int64, ndmin=2))
        pp = directory/'partition.json'
        if pp.exists():
            data = json.loads(pp.read_text())
            if data is not None:
                colors = np.asarray(data, dtype=np.int32)
    return g, w, colors, saved_k


def reduce_vertices(state: State, pool: OraclePool, rng: np.random.Generator,
                    checks: int, min_points: int,
                    log: Callable[..., None] = lambda *a, **k: None) -> tuple[State, dict[str, Any]]:
    """Delete v only after proving R(X-v) >= the current partition upper bound.

    Use exact integer cross multiplication and recompute the subset diameter.
    Removed vertices are isolated in the original oracle, avoiding a solver
    rebuild at every attempted deletion. No numerical loss tolerance is used.
    """
    g, k = state.g, pool.k
    oracle = pool.get(g.m)
    keep = np.ones(g.m, dtype=bool)
    target = state.upper2
    current_w = state.witness.copy()
    q = g.q
    used, deleted, unknown = 0, [], 0
    degree = np.bincount(g.pairs[state.witness].ravel(), minlength=g.m)
    priority = np.argsort(degree + rng.random(g.m)*max(1, degree.max(initial=0)/2))
    for vertex in priority:
        if used >= checks or keep.sum() <= max(k+1, min_points):
            break
        v = int(vertex)
        trial_keep = keep.copy(); trial_keep[v] = False
        induced = np.flatnonzero(trial_keep[g.pairs].all(axis=1))
        D = max(q[int(i)] for i in induced)
        a, b = target.numerator*D, target.denominator
        ids = np.array([i for i in induced if q[int(i)]*b >= a], dtype=np.int32)
        used += 1
        ans = oracle.solve(ids)
        if ans.status is bs.Status.UNSAT:
            keep = trial_keep
            current_w = (ans.core.copy() if ans.core is not None and len(ans.core)
                         else ids)
            deleted.append(v)
            U = max((q[int(i)] for i in induced
                     if state.colors[g.pairs[i, 0]] == state.colors[g.pairs[i, 1]]), default=0)
            # Tighten the next deletion requirement, so a gain from a shrinking
            # diameter cannot silently be lost by the following deletion.
            target = Fraction(U, D)
            log('delete', vertex=v, points=int(keep.sum()),
                preserved_at_least=math.sqrt(float(target)))
        elif ans.status is bs.Status.UNKNOWN:
            unknown += 1
    report = {'checks': used, 'deleted_original_indices': deleted, 'unknowns': unknown,
              'criterion': 'new lower bound >= preceding partition upper bound, exact integers'}
    if not deleted:
        return state, report
    chosen = np.flatnonzero(keep)
    inverse = np.full(g.m, -1, dtype=np.int32); inverse[chosen] = np.arange(len(chosen))
    new_g = g.subset(chosen)
    edges = inverse[g.pairs[current_w]]
    if np.any(edges < 0):
        raise RuntimeError('Deleted vertex remains in witness.')
    out = assess(new_g, pool, edge_ids(len(chosen), edges), state.colors[chosen], True)
    if out.lower2 < state.upper2:
        raise RuntimeError('Internal error: vertex reduction lost the certified ratio.')
    return out, report


class ConstrainedProblem:
    """Maximise t, all squared distances <=1, witness distances >=t.

    Fix the first point at zero to remove translation. All remaining coordinate
    bounds [-1.1,1.1] are inactive for genuine diameter-one configurations.
    Diameter constraints can be generated lazily; witness constraints are all
    present from the start. Missing diameter constraints are checked after
    every solve and added before convergence can be reported.
    """
    def __init__(self, x: np.ndarray, ids: np.ndarray, full: bool = False):
        x = bs.normalize(x); x -= x[0]
        self.m, self.d = x.shape
        self.pairs = bs.pairs_for(self.m)
        self.ids = np.unique(np.asarray(ids, np.int32))
        self.nv = (self.m-1)*self.d+1
        self.start = x
        q = bs.distances2(x, self.pairs)
        self.active_diameter = (np.arange(len(q), dtype=np.int32) if full else
                               np.flatnonzero((q >= 0.97) | (self.pairs[:, 0] == 0)))
        self.grad = np.zeros(self.nv); self.grad[-1] = -1.

    def pack(self, x: np.ndarray) -> np.ndarray:
        xx = bs.normalize(x); xx -= xx[0]
        q = bs.distances2(xx, self.pairs)
        return np.r_[xx[1:].ravel(), min(q[self.ids])]

    def unpack(self, z: np.ndarray) -> np.ndarray:
        return np.vstack((np.zeros((1, self.d)), z[:-1].reshape(self.m-1, self.d)))

    def distance_jac(self, x: np.ndarray, ids: np.ndarray) -> np.ndarray:
        e = self.pairs[ids]
        dif = 2*(x[e[:, 0]]-x[e[:, 1]])
        out = np.zeros((len(e), self.nv))
        for endpoint, sign in ((0, 1.), (1, -1.)):
            valid = e[:, endpoint] != 0
            rows = np.flatnonzero(valid)
            cols = (e[valid, endpoint, None]-1)*self.d+np.arange(self.d)
            out[rows[:, None], cols] = sign*dif[valid]
        return out

    def cons(self, z: np.ndarray) -> np.ndarray:
        q = bs.distances2(self.unpack(z), self.pairs)
        return np.r_[1-q[self.active_diameter], q[self.ids]-z[-1]]

    def jac(self, z: np.ndarray) -> np.ndarray:
        x = self.unpack(z)
        A = -self.distance_jac(x, self.active_diameter)
        B = self.distance_jac(x, self.ids); B[:, -1] = -1.
        return np.vstack((A, B))


def linearized_audit(x: np.ndarray, ids: np.ndarray, active_tol: float = 1e-8,
                     time_limit: float = 30.) -> tuple[dict[str, Any], np.ndarray | None]:
    """Numerical fixed-witness, near-active first-order diagnostic.

    Maximise eta subject to grad(diameter-edge^2).v<=0,
    eta<=grad(short-witness-edge^2).v, |v|_infinity<=1, 0<=eta<=1.
    Near-active constraints are intentionally conservative. A zero optimum is
    NOT an exact stationarity certificate and says nothing about other graphs.
    """
    p = ConstrainedProblem(x, ids)
    xx = p.start
    q = bs.distances2(xx, p.pairs)
    t = float(q[p.ids].min())
    dia = np.flatnonzero(q >= q.max()-active_tol)
    short = p.ids[q[p.ids] <= t+active_tol]
    A = p.distance_jac(xx, dia)
    B = -p.distance_jac(xx, short); B[:, -1] = 1.
    A = np.vstack((A, B))
    result = linprog(p.grad, A_ub=A, b_ub=np.zeros(len(A)),
                     bounds=[(-1., 1.)]*(p.nv-1)+[(0., 1.)], method='highs',
                     options={'time_limit': time_limit,
                              'primal_feasibility_tolerance': 1e-9,
                              'dual_feasibility_tolerance': 1e-9})
    report = {'success': bool(result.success), 'message': str(result.message),
              'active_tolerance': active_tol, 'active_diameter_edges': len(dia),
              'active_witness_edges': len(short),
              'linearized_gain': float(max(0., result.x[-1])) if result.success else None,
              'scope': 'numerical near-active first-order test for this fixed witness only'}
    if not result.success:
        return report, None
    return report, np.vstack((np.zeros((1, p.d)), result.x[:-1].reshape(p.m-1, p.d)))


def slsqp_once(x: np.ndarray, ids: np.ndarray, args: argparse.Namespace,
               ftol: float) -> tuple[np.ndarray, dict[str, Any]]:
    p = ConstrainedProblem(x, ids, args.full_constraints)
    best = p.start.copy()
    best_ratio = bs.witness_ratio(best, p.pairs, p.ids)
    z = p.pack(best)
    reports: list[dict[str, Any]] = []
    all_checked = False

    def record(zz: np.ndarray) -> None:
        nonlocal best, best_ratio
        if not np.isfinite(zz).all():
            return
        xx = p.unpack(zz)
        rr = bs.witness_ratio(xx, p.pairs, p.ids)
        if math.isfinite(rr) and rr > best_ratio:
            best, best_ratio = xx.copy(), rr

    for cut_pass in range(args.constraint_passes):
        result = minimize(lambda zz: -zz[-1], z, method='SLSQP', jac=lambda zz: p.grad,
                          constraints=[{'type': 'ineq', 'fun': p.cons, 'jac': p.jac}],
                          bounds=[(-1.1, 1.1)]*(p.nv-1)+[(0., 1.)], callback=record,
                          options={'maxiter': args.slsqp_iters, 'ftol': ftol, 'disp': False})
        record(result.x)
        finite = bool(np.isfinite(result.x).all())
        xx = p.unpack(result.x) if finite else p.unpack(z)
        q = bs.distances2(xx, p.pairs)
        dia_bad = np.flatnonzero(q > 1+args.feasibility_tol)
        missing = np.setdiff1d(dia_bad, p.active_diameter)
        violation = max(0., float(q.max()-1),
                        float(result.x[-1]-q[p.ids].min())) if finite else None
        feasible = bool(finite and violation <= args.feasibility_tol)
        all_checked = not len(missing)
        reports.append({'success': bool(result.success), 'status': int(result.status),
                        'message': str(result.message), 'iterations': int(result.nit),
                        'full_constraint_violation': violation, 'feasible': feasible,
                        'diameter_constraints': len(p.active_diameter),
                        'missing_violated_constraints': len(missing), 'ftol': ftol})
        if not len(missing):
            break
        p.active_diameter = np.union1d(p.active_diameter, missing)
        z = p.pack(best)
    # record() can keep an earlier iterate better than result.x. Such a point is
    # useful, but it must not inherit the terminal solver's convergence status.
    result_ratio = bs.witness_ratio(xx, p.pairs, p.ids)
    terminal_matches = abs(best_ratio-result_ratio) <= args.ratio_tol
    report = {'runs': reports, 'all_diameter_constraints_checked': all_checked,
              'terminal_success': bool(reports[-1]['success'] and reports[-1]['feasible']
                                       and all_checked and terminal_matches),
              'terminal_matches_retained_ratio': terminal_matches,
              'retained_ratio': best_ratio}
    return bs.normalize(best), report


def polish_fixed(x: np.ndarray, ids: np.ndarray, rng: np.random.Generator,
                 args: argparse.Namespace,
                 log: Callable[..., None] = lambda *a, **k: None) -> tuple[np.ndarray, dict[str, Any]]:
    """Restart SLSQP until two successful stagnant passes plus an LP audit.

    Also try small coordinate kicks. Exhausting a budget is explicitly distinct
    from the 'numerically_stationary' status. The best actual ratio is retained
    even if a solver exits unsuccessfully or produces infeasible coordinates.
    """
    best = bs.normalize(x)
    pairs = bs.pairs_for(len(best))
    ratio = bs.witness_ratio(best, pairs, ids)
    logs: list[dict[str, Any]] = []
    stable, kicks_used = 0, 0
    status = 'polish_budget_exhausted'
    last_audit: dict[str, Any] = {}
    for iteration in range(args.polish_passes):
        before = ratio
        xx, report = slsqp_once(best, ids, args, args.ftol)
        rr = bs.witness_ratio(xx, pairs, ids)
        if rr > ratio:
            best, ratio = xx, rr
        gain = ratio-before
        stable = stable+1 if gain <= args.ratio_tol and report['terminal_success'] else 0
        last_audit, direction = linearized_audit(best, ids, args.active_tol, args.lp_timeout)
        entry = {'pass': iteration+1, 'gain': gain, 'ratio': ratio,
                 'successful_stagnant_passes': stable, **report, 'audit': last_audit}
        logs.append(entry)
        log('polish', pass_number=iteration+1, gain=gain, ratio=ratio,
            solver_success=report['terminal_success'],
            linearized_gain=last_audit['linearized_gain'])
        lp_small = (last_audit['success'] and
                    last_audit['linearized_gain'] <= args.stationarity_tol)
        if stable >= args.stall_passes and (lp_small or kicks_used < args.polish_kicks):
            if kicks_used >= args.polish_kicks:
                status = 'numerically_stationary'
                break
            kick = bs.normalize(best + rng.normal(size=best.shape)*args.polish_jitter)
            kicked, kick_report = slsqp_once(kick, ids, args, args.ftol)
            kr = bs.witness_ratio(kicked, pairs, ids)
            kicks_used += 1
            entry['kick'] = {'number': kicks_used, 'ratio': kr, **kick_report}
            if kr > ratio:
                best, ratio = kicked, kr
            stable = 0  # Require fresh unperturbed passes at the retained point.
        elif direction is not None and not lp_small:
            # LP is only a suggestion: evaluate the real nonlinear ratio at
            # every step. A false/impractical direction never degrades best.
            base = best-best[0]
            for step in (1e-2, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
                trial = base+step*direction
                tr = bs.witness_ratio(trial, pairs, ids)
                if tr > ratio+args.ratio_tol:
                    best, ratio, stable = bs.normalize(trial), tr, 0
                    entry['accepted_lp_step'] = step
                    break
    final_audit, _ = linearized_audit(best, ids, args.active_tol, args.lp_timeout)
    if status == 'numerically_stationary' and (not final_audit['success'] or
            final_audit['linearized_gain'] > args.stationarity_tol):
        status = 'audit_unresolved'
    return best, {'status': status, 'passes': logs, 'kicks': kicks_used,
                  'final_audit': final_audit, 'ratio_tolerance': args.ratio_tol,
                  'feasibility_tolerance': args.feasibility_tol,
                  'qualification': 'Not a proof of local/global optimality or exhaustion of other witnesses.'}


def local_tune(g: Geometry, ids: np.ndarray, pool: OraclePool,
               rng: np.random.Generator, args: argparse.Namespace,
               log: Callable[..., None], smooth: bool = True,
               colors: np.ndarray | None = None) -> State:
    """Optimise a known UNSAT witness, evaluate the whole set, refresh and polish."""
    x = g.x.copy()
    ids = np.asarray(ids, dtype=np.int32)
    pairs = g.pairs
    if smooth and args.steps:
        log('smooth_start', points=g.m, edges=len(ids))
        x, _ = bs.optimize_witness(x, pairs, ids, steps=args.steps, polish=0,
                                   powers=(32., 128., 512., 2048.))
    best = assess(g, pool, ids, colors, witness_known=True)
    reports = []
    current_ids = ids
    for refresh in range(args.refreshes):
        log('slsqp_start', points=g.m, refresh=refresh+1, edges=len(current_ids))
        x, report = polish_fixed(x, current_ids, rng, args, log)
        new_g = Geometry.from_float(bs.normalize(x))
        result = assess(new_g, pool, current_ids, best.colors, witness_known=True)
        record = {'refresh': refresh+1, **report,
                  'full_ratio_lower': result.lower, 'full_ratio_upper': result.upper}
        reports.append(record)
        if result.lower2 >= best.upper2:
            best = result
        # The numerical polish concerns current_ids; evaluation may discover a
        # different obstruction with a larger ratio. In that case polish again.
        witness_value = bs.witness_ratio(new_g.x, pairs, current_ids)
        changed = result.lower > witness_value+args.ratio_tol
        same_as_best = best is result
        retained_equivalent = (abs(best.lower-result.lower) <= args.ratio_tol and
                               report['status'] == 'numerically_stationary')
        if not changed and (same_as_best or retained_equivalent):
            audit = linearized_audit(best.g.x, best.witness, args.active_tol, args.lp_timeout)[0]
            status = report['status']
            if status == 'numerically_stationary' and (not audit['success'] or
                    audit['linearized_gain'] > args.stationarity_tol):
                status = 'saved_coordinate_audit_unresolved'
            best.polish = {'status': status, 'refreshes': reports,
                           'saved_coordinate_audit': audit}
            return best
        x, current_ids = best.g.x.copy(), best.witness.copy()
    best.polish = {'status': 'refresh_budget_exhausted', 'refreshes': reports,
                   'saved_coordinate_audit': linearized_audit(
                       best.g.x, best.witness, args.active_tol, args.lp_timeout)[0]}
    return best


def canonical_partition(colors: np.ndarray) -> tuple[int, ...]:
    labels: dict[int, int] = {}
    return tuple(labels.setdefault(int(c), len(labels)) for c in colors)


def partition_bank(state: State, pool: OraclePool, rng: np.random.Generator,
                   count: int) -> list[np.ndarray]:
    """Obtain different valid upper-bound partitions by permuting SAT labels."""
    bank = [state.colors.copy()]
    seen = {canonical_partition(state.colors)}
    old_ids = state.g.threshold(state.upper2, inclusive=False)
    edges = state.g.pairs[old_ids]
    oracle = pool.get(state.g.m)
    for _ in range(max(0, 3*(count-1))):
        if len(bank) >= count:
            break
        perm = rng.permutation(state.g.m)
        inverse = np.argsort(perm)
        ans = oracle.solve(edge_ids(state.g.m, inverse[edges]))
        if ans.status is not bs.Status.SAT or ans.colors is None:
            continue
        color = np.empty(state.g.m, np.int32); color[perm] = ans.colors
        key = canonical_partition(color)
        if key not in seen:
            seen.add(key); bank.append(color)
    return bank


def completion_candidates(x: np.ndarray, count: int, rng: np.random.Generator,
                          outer_slack: float = 0.) -> np.ndarray:
    """Sample the diameter completion intersection_i B(x_i, D), not just conv(X).

    For a ray c+s*u from the centroid, the exact positive boundary parameter is
    min_i [u.(x_i-c)+sqrt((u.(x_i-c))^2+D^2-|x_i-c|^2)].
    Floating feasibility is checked; exact acceptance is performed downstream.
    """
    m, d = x.shape
    center = x.mean(axis=0)
    z = x-center
    D2 = float(bs.distances2(x, bs.pairs_for(m)).max())
    directions = rng.normal(size=(count, d))
    take = min(m, count//4)
    directions[:take] = z[rng.choice(m, take, replace=False)]
    for j in range(take, min(2*take, count)):
        a, b = rng.choice(m, 2, replace=False)
        directions[j] = z[a]+z[b] if j % 2 else z[a]-z[b]
    norms = np.linalg.norm(directions, axis=1)
    directions[norms < 1e-14] = rng.normal(size=(int((norms < 1e-14).sum()), d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    dot = directions@z.T
    rad = dot*dot+D2-np.einsum('ij,ij->i', z, z)[None, :]
    s = (dot+np.sqrt(np.maximum(rad, 0))).min(axis=1)
    fractions = np.ones(count)*(1-2e-10)
    inner = rng.random(count) < 0.22
    fractions[inner] = rng.uniform(0.65, 0.98, int(inner.sum()))
    if outer_slack:
        shell = (~inner) & (rng.random(count) < 0.3)
        fractions[shell] = 1+rng.uniform(0.15, 1., int(shell.sum()))*outer_slack
    return center+(s*fractions)[:, None]*directions


def coverage_score(point: np.ndarray, x: np.ndarray, colors: np.ndarray, k: int) -> float:
    q = np.sum((x-point)**2, axis=1)
    return min((float(q[colors == c].max()) if np.any(colors == c) else -math.inf)
               for c in range(k))


def refine_insert_point(point: np.ndarray, x: np.ndarray, colors: np.ndarray,
                        k: int, iters: int) -> np.ndarray:
    """Smooth max-min colour coverage, inside the diameter completion body."""
    groups = [np.flatnonzero(colors == c) for c in range(k)]
    if any(not len(group) for group in groups):
        return point
    D2 = float(bs.distances2(x, bs.pairs_for(len(x))).max())
    best, score = point.copy(), coverage_score(point, x, colors, k)
    start = point.copy()
    for beta in (80., 400.):
        def objective(p: np.ndarray) -> tuple[float, np.ndarray]:
            diff = p-x; q = np.einsum('ij,ij->i', diff, diff)
            vals, grads = [], []
            for group in groups:
                a = beta*q[group]/D2
                la = logsumexp(a)
                vals.append((la-math.log(len(group)))/beta)
                grads.append(np.exp(a-la)@(2*diff[group]/D2))
            vals = np.array(vals); grads = np.array(grads)
            b = -beta*vals; lb = logsumexp(b)
            # Minimise the negative smooth minimum of per-colour smooth maxima.
            return float(lb/beta), -np.exp(b-lb)@grads
        result = minimize(objective, start, jac=True, method='SLSQP',
                          constraints=[{'type': 'ineq',
                                        'fun': lambda p: D2-np.sum((x-p)**2, axis=1),
                                        'jac': lambda p: 2*(x-p)}],
                          options={'maxiter': iters, 'ftol': 1e-11})
        if np.isfinite(result.x).all():
            point2 = result.x
            # Retract tiny feasibility errors toward the feasible centroid.
            center = x.mean(axis=0)
            if np.sum((x-point2)**2, axis=1).max() > D2:
                lo, hi = 0., 1.
                for _ in range(45):
                    mid = (lo+hi)/2
                    if np.sum((x-(center+mid*(point2-center)))**2, axis=1).max() <= D2:
                        lo = mid
                    else:
                        hi = mid
                point2 = center+lo*(1-1e-12)*(point2-center)
            new_score = coverage_score(point2, x, colors, k)
            if new_score > score:
                best, score = point2.copy(), new_score
            start = point2
    return best


def adjacency(m: int, edges: np.ndarray) -> np.ndarray:
    out = np.zeros((m, m), dtype=bool)
    if len(edges):
        out[edges[:, 0], edges[:, 1]] = True
        out |= out.T
    return out


def redundant_new_vertex(full_adj: np.ndarray, graph_adj: np.ndarray,
                         old_m: int, k: int) -> str | None:
    """Filter geometric twins and valid nonadjacent graph folds.

    True twins can increase chromatic number. Rejecting true twins is an
    explicit exploration preference, not a theorem that they never help.
    Only false twins and nonadjacent domination are colour-extension redundant.
    """
    m = len(graph_adj)
    for v in range(old_m, m):
        if int(graph_adj[v].sum()) < k:
            return 'new_degree_below_k'
        for u in range(m):
            if u == v:
                continue
            mask = np.ones(m, bool); mask[[u, v]] = False
            if np.array_equal(full_adj[v, mask], full_adj[u, mask]):
                return 'geometric_true_twin' if full_adj[u, v] else 'geometric_false_twin'
            if not graph_adj[u, v] and np.all(~graph_adj[v] | graph_adj[u]):
                return 'nonadjacent_dominated_neighbourhood'
    return None


def coloring_extends(colors: np.ndarray, graph_adj: np.ndarray, k: int) -> bool:
    """Small exact list-colouring test for extending one old coloring to a block."""
    m0, total = len(colors), len(graph_adj)
    allowed = []
    for v in range(m0, total):
        forbidden = set(map(int, colors[graph_adj[v, :m0]]))
        allowed.append([c for c in range(k) if c not in forbidden])
    order = sorted(range(total-m0), key=lambda i: len(allowed[i]))
    chosen = [-1]*(total-m0)
    def visit(j: int) -> bool:
        if j == len(order):
            return True
        v = order[j]
        for c in allowed[v]:
            if all(chosen[u] != c or not graph_adj[m0+u, m0+v] for u in range(len(chosen))):
                chosen[v] = c
                if visit(j+1):
                    return True
                chosen[v] = -1
        return False
    return visit(0)


def essential_extension(old_colors: np.ndarray, g: Geometry, graph_ids: np.ndarray,
                        pool: OraclePool) -> tuple[np.ndarray | None, str]:
    """The old skeleton is SAT by its supplied coloring; the extension must be UNSAT.

    Require every inserted point to be essential in this comparison graph:
    deleting it must give SAT. This is a structural relevance test, not a claim
    that the full old geometric threshold graph becomes colorable on deletion.
    """
    old_m = len(old_colors)
    edges = g.pairs[graph_ids]
    old = edges[(edges < old_m).all(axis=1)]
    if len(old) and np.any(old_colors[old[:, 0]] == old_colors[old[:, 1]]):
        raise ValueError('The insertion skeleton is not coloured by old_colors.')
    oracle = pool.get(g.m)
    answer = oracle.solve(graph_ids)
    if answer.status is not bs.Status.UNSAT:
        return None, answer.status.value
    for v in range(old_m, g.m):
        trial = graph_ids[np.all(edges != v, axis=1)]
        sub = oracle.solve(trial)
        if sub.status is bs.Status.UNSAT:
            return None, 'redundant_new_point'
        if sub.status is bs.Status.UNKNOWN:
            return None, 'essentiality_UNKNOWN'
    return (answer.core.copy() if answer.core is not None and len(answer.core)
            else graph_ids.copy()), 'essential_UNSAT_extension'


def witness_compaction(g: Geometry, ids: np.ndarray, min_points: int = 0) -> tuple[Geometry, np.ndarray]:
    """Exploratory core extraction; preserves its witness bound, not necessarily R(X)."""
    edges = g.pairs[ids]
    used = np.unique(edges.ravel())
    if len(used) < min(min_points, g.m):
        unused = np.setdiff1d(np.arange(g.m), used)
        used = np.sort(np.r_[used, unused[:min_points-len(used)]])
    if len(used) == g.m:
        return g, ids
    inverse = np.full(g.m, -1, dtype=np.int32); inverse[used] = np.arange(len(used))
    return g.subset(used), edge_ids(len(used), inverse[edges])


@dataclass
class Proposal:
    g: Geometry
    ids: np.ndarray
    meta: dict[str, Any]


def insertion_proposals(state: State, pool: OraclePool, rng: np.random.Generator,
                        args: argparse.Namespace, log: Callable[..., None]) -> list[Proposal]:
    if args.trials == 0 or state.g.m >= args.max_points:
        return []
    x = bs.normalize(state.g.x)
    old_m, d = x.shape
    bank = partition_bank(state, pool, rng, args.colorings)
    points = completion_candidates(x, args.candidates, rng, args.outer_slack)
    initial_scores = np.array([coverage_score(p, x, bank[0], pool.k) for p in points])
    refined = []
    for j, idx in enumerate(np.argsort(-initial_scores)[:args.refine_candidates]):
        refined.append(refine_insert_point(points[idx], x, bank[j % len(bank)],
                                           pool.k, args.insertion_iters))
    if refined:
        points = np.vstack((points, refined))
    # Reject near-coincident geometric duplicates before any graph construction.
    ds = np.sum((points[:, None, :]-x[None, :, :])**2, axis=2)
    points = points[ds.min(axis=1) > args.min_separation**2]
    if not len(points):
        return []
    variants = [(i, slack) for slack in args.slacks for i in range(len(bank))]
    proposals: list[Proposal] = []
    seen: set[tuple[Any, ...]] = set()
    rejections: dict[str, int] = {}
    screened = 0
    max_new = min(args.add_max, args.max_points-old_m)
    if max_new < args.add_min:
        return []
    # Precompute one representative of each old-neighbourhood signature for
    # every threshold. Avoid wasting the SAT budget on many identical clones.
    diverse: dict[float, np.ndarray] = {}
    ds = np.sum((points[:, None, :]-x[None, :, :])**2, axis=2)
    for slack in args.slacks:
        cut = state.upper*(1-slack)
        keys: set[bytes] = set(); selected = []
        for j in rng.permutation(len(points)):
            key = np.packbits(ds[j] > cut*cut*(1+2e-11 if slack == 0 else 1.)).tobytes()
            if key not in keys:
                keys.add(key); selected.append(j)
        diverse[slack] = np.asarray(selected, dtype=np.int32)
    for attempt in range(args.candidate_attempts):
        if screened >= args.screen_checks or len(proposals) >= 3*args.trials:
            break
        bi, slack = variants[attempt % len(variants)]
        colors = bank[bi]
        count = int(rng.integers(args.add_min, max_new+1))
        available = diverse[slack]
        if len(available) < count:
            continue
        picked = rng.choice(available, count, replace=False)
        xx = np.vstack((x, points[picked]))
        if count > 1 and bs.distances2(points[picked], bs.pairs_for(count)).min() < args.min_separation**2:
            continue
        cut = state.upper*(1-slack)
        m = len(xx); pairs = bs.pairs_for(m)
        q = bs.distances2(xx, pairs)
        if q.max() > (1+args.outer_slack)**2+1e-12:
            rejections['block_diameter_expansion'] = rejections.get('block_diameter_expansion', 0)+1
            continue
        # Use a strict threshold with a small relative gap at slack zero, so
        # tied bottleneck edges do not turn the full geometric graph into an
        # artificial clique for the novelty filter. Witness bounds are later
        # recomputed from the actual integer distances, not this cutoff.
        cutoff2 = cut*cut*(1+2e-11 if slack == 0 else 1.)
        threshold_ids = np.flatnonzero(q > cutoff2)
        selected_pairs = pairs[threshold_ids]
        old_edges = (selected_pairs < old_m).all(axis=1)
        allow = np.ones(len(selected_pairs), bool)
        allow[old_edges] = (colors[selected_pairs[old_edges, 0]] !=
                            colors[selected_pairs[old_edges, 1]])
        graph_ids = threshold_ids[allow]
        A = adjacency(m, pairs[graph_ids])
        key = (bi, slack, count, np.packbits(A).tobytes())
        if key in seen:
            continue
        seen.add(key)
        reason = redundant_new_vertex(adjacency(m, pairs[threshold_ids]), A, old_m, pool.k)
        if reason is None and coloring_extends(colors, A, pool.k):
            reason = 'known_coloring_extends'
        if reason is not None:
            rejections[reason] = rejections.get(reason, 0)+1
            continue
        screened += 1
        try:
            g = Geometry.from_float(xx)
        except ValueError:
            continue
        w, reason = essential_extension(colors, g, graph_ids, pool)
        if w is None:
            rejections[reason] = rejections.get(reason, 0)+1
            continue
        oracle = pool.get(m)
        w = bs.prune_witness(oracle, w, bs.distances2(g.x, pairs), rng,
                             checks=args.edge_prune, noise=0.02)
        meta = {'kind': 'insertion', 'old_points': old_m, 'new_points': count,
                'slack': slack, 'comparison_threshold': cut,
                'initial_diameter_expansion': math.sqrt(float(q.max())),
                'old_coloring': colors.tolist(), 'old_skeleton_edges':
                    pairs[graph_ids][(pairs[graph_ids] < old_m).all(axis=1)].tolist(),
                'initial_extension_edges': pairs[graph_ids].tolist(),
                'new_coordinates': points[picked].tolist(),
                'screen': 'old skeleton SAT, full extension UNSAT, every new-point deletion SAT',
                'witness_edges': len(w)}
        if args.compact_trials:
            g, w = witness_compaction(g, w, args.min_points)
            meta['exploratory_core_points'] = g.m
        proposals.append(Proposal(g, w, meta))
        log('insertion_found', new_points=count, trial_points=g.m, slack=slack,
            witness_ratio=bs.witness_ratio(g.x, g.pairs, w), edges=len(w))
    proposals.sort(key=lambda p: (-bs.witness_ratio(p.g.x, p.g.pairs, p.ids), p.g.m))
    generated = len(proposals)
    proposals = proposals[:args.trials]
    log('insertion_screen_finished', proposals=len(proposals), generated=generated, sat_screen_calls=screened,
        partitions=len(bank), rejection_counts=rejections)
    return proposals


def save_state(directory: Path, state: State, k: int, metadata: dict[str, Any]) -> dict[str, Any]:
    """Same coordinate/certificate format as the original search and viewer."""
    directory.mkdir(parents=True, exist_ok=True)
    g = state.g
    edges = g.pairs[state.witness]
    geometry = bs.rational_geometry(g.rows, edges, state.colors)
    data = {'version': 'tuner-'+VERSION, 'dimension': g.dim, 'm': g.m, 'k': k,
            'floating_lower': state.lower, 'floating_upper': state.upper,
            'threshold_resolved': state.exact, 'coordinate_decimal_digits': g.digits,
            'duplicate_rows_separated_by_1e_minus_17': 0, 'witness_edges': len(edges),
            'geometry': geometry, 'run': metadata,
            'qualification': 'Geometry and comparisons use integer arithmetic. Graph UNSAT '
                             'results are not independently proof-logged. SLSQP/LP reports '
                             'are numerical fixed-graph diagnostics, not optimality proofs.'}
    (directory/'points.csv').write_text('\n'.join(
        ','.join(str(a) if g.digits == 0 else bs.format_integer_point(a, g.digits) for a in row) for row in g.rows)+'\n')
    np.save(directory/'points.npy', g.x)
    np.savetxt(directory/'witness_edges.csv', edges, delimiter=',', fmt='%d')
    bs.write_cnf(directory/'witness.cnf', g.m*k, bs.graph_cnf(g.m, k, edges))
    (directory/'partition.json').write_text(json.dumps(state.colors.tolist())+'\n')
    (directory/'polish.json').write_text(json.dumps(state.polish or {'status': 'not_polished'}, indent=2)+'\n')
    temp = directory/'certificate.json.tmp'
    temp.write_text(json.dumps(data, indent=2)+'\n'); temp.replace(directory/'certificate.json')
    return data


class Checkpoints:
    def __init__(self, out: Path, k: int, log: Callable[..., None]):
        self.out, self.k, self.log = out, k, log
        self.best: State | None = None
        self.number = 0

    def consider(self, state: State, reason: str, force: bool = False) -> bool:
        old = self.best
        improves = old is None or state.lower2 > old.upper2
        simplifies = (old is not None and state.lower2 >= old.upper2 and state.g.m < old.g.m)
        same_geometry = old is not None and state.g.rows == old.g.rows and state.g.digits == old.g.digits
        report_update = (same_geometry and state.polish is not None and state.polish != old.polish and
                         state.lower2 >= old.lower2 and state.upper2 <= old.upper2)
        stationary_plateau = (old is not None and state.lower2 >= old.upper2 and state.g.m <= old.g.m
                              and (state.polish or {}).get('status') == 'numerically_stationary'
                              and (old.polish or {}).get('status') != 'numerically_stationary')
        if not (force or improves or simplifies or report_update or stationary_plateau):
            return False
        if old is not None and not same_geometry and state.lower2 < old.upper2:
            raise RuntimeError('Refusing an uncertified best replacement.')
        self.number += 1
        history = self.out/'history'/f'{self.number:06d}'
        meta = {'reason': reason, 'sequence': self.number,
                'previous_upper': old.upper if old else None,
                'acceptance': ('same coordinates: bound/report refinement' if same_geometry else
                               'exact new lower >= old upper; gain, fewer vertices, or stronger numerical status')}
        save_state(history, state, self.k, meta)
        staged = self.out/'.best.new'; previous = self.out/'best.previous'
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(history, staged)
        if previous.exists():
            shutil.rmtree(previous)
        if (self.out/'best').exists():
            (self.out/'best').rename(previous)
        staged.rename(self.out/'best')
        self.best = state
        self.log('best_saved', reason=reason, points=state.g.m, lower=state.lower,
                 upper=state.upper, exact=state.exact,
                 polish=(state.polish or {}).get('status', 'not_polished'))
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('start', type=Path, help='Saved best folder, search root, or coordinates file.')
    parser.add_argument('--out', type=Path, default=Path('tuned_solution'), help='New output directory.')
    parser.add_argument('--k', type=int, help='Number of parts; saved k, otherwise dimension+1.')
    parser.add_argument('--seed', type=int, default=20260908)
    parser.add_argument('--cycles', type=int, default=20, help='Stochastic topology-tuning cycles.')
    parser.add_argument('--delete-checks', type=int, default=150, help='Single-vertex deletion checks per cycle; 0 disables.')
    parser.add_argument('--min-points', type=int, default=0, help='Minimum points retained during deletions/core extraction, at least k+1.')
    parser.add_argument('--max-points', type=int, help='Upper trial point count; default starting count + 24.')
    parser.add_argument('--add', default='1:3', help='Inserted block size A:B (or a fixed A).')
    parser.add_argument('--trials', type=int, default=6, help='Maximum SAT-screened insertion trials per cycle.')
    parser.add_argument('--candidates', type=int, default=384, help='Geometric insertion candidates.')
    parser.add_argument('--refine-candidates', type=int, default=16, help='Promising new points to refine inside the completion body.')
    parser.add_argument('--insertion-iters', type=int, default=70, help='SLSQP iterations for refining a single proposed point.')
    parser.add_argument('--candidate-attempts', type=int, default=3000, help='Cheap block/neighbourhood attempts per cycle.')
    parser.add_argument('--screen-checks', type=int, default=150, help='Full-extension SAT tests per cycle, before essentiality checks.')
    parser.add_argument('--colorings', type=int, default=3, help='Different old colourings used for insertion skeletons.')
    parser.add_argument('--slacks', default='0,0.004,0.012,0.03,0.08,0.15', help='Relative threshold relaxations for alternative witnesses.')
    parser.add_argument('--outer-slack', type=float, default=0.04,
                        help='Allowed exploratory diameter expansion during insertion; 0 keeps every block inside the old diameter.')
    parser.add_argument('--min-separation', type=float, default=1e-5, help='Reject nearly coincident insertions, in diameter units.')
    parser.add_argument('--edge-prune', type=int, default=80, help='UNSAT-preserving edge deletion tests per witness.')
    parser.add_argument('--compact-trials', action=argparse.BooleanOptionalAction, default=True,
                        help='Extract the used-vertex core of exploratory witnesses (not a ratio-preserving deletion claim).')
    parser.add_argument('--witness-trials', type=int, default=1, help='Old-point alternative-witness trials per cycle.')
    parser.add_argument('--explore-loss', type=float, default=0.002,
                        help='Relative lower-bound deficit allowed in an exploratory branch; best is NEVER degraded.')
    parser.add_argument('--steps', type=int, default=100, help='L-BFGS iterations at each of four smoothing levels; 0 skips.')
    parser.add_argument('--refreshes', type=int, default=3, help='Reevaluate/refresh/polish rounds per local trial.')
    parser.add_argument('--slsqp-iters', type=int, default=300)
    parser.add_argument('--polish-passes', type=int, default=8, help='Maximum repeated SLSQP calls per witness.')
    parser.add_argument('--stall-passes', type=int, default=2, help='Successful no-gain passes required before numerical stationarity.')
    parser.add_argument('--polish-kicks', type=int, default=1, help='Additional tiny perturbed SLSQP starts before stopping.')
    parser.add_argument('--polish-jitter', type=float, default=1e-4, help='Polishing perturbation in diameter units.')
    parser.add_argument('--ftol', type=float, default=1e-12)
    parser.add_argument('--ratio-tol', type=float, default=1e-10, help='Numerical polishing stagnation tolerance, NOT an acceptance/deletion tolerance.')
    parser.add_argument('--feasibility-tol', type=float, default=1e-9)
    parser.add_argument('--active-tol', type=float, default=1e-8, help='Squared-distance near-active tolerance for first-order LP audit.')
    parser.add_argument('--stationarity-tol', type=float, default=1e-7, help='Maximum LP gain for a numerical stationary report.')
    parser.add_argument('--lp-timeout', type=float, default=30.)
    parser.add_argument('--constraint-passes', type=int, default=8)
    parser.add_argument('--full-constraints', action='store_true', help='Put every diameter constraint in SLSQP immediately (slower).')
    parser.add_argument('--polish-only', action='store_true', help='No deletions, insertions or smoothing: repeated SLSQP/audit only.')
    parser.add_argument('--quiet', action='store_true')
    bs.add_backend_options(parser)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    ab = list(map(int, args.add.split(':')))
    if len(ab) == 1:
        ab *= 2
    if len(ab) != 2 or ab[0] < 1 or ab[1] < ab[0] or ab[1] > 6:
        raise ValueError('--add must be A:B with 1 <= A <= B <= 6.')
    args.add_min, args.add_max = ab
    args.slacks = [float(x.strip()) for x in args.slacks.split(',')]
    if not args.slacks or any(not 0 <= x < 0.5 for x in args.slacks):
        raise ValueError('--slacks must be a comma-separated list in [0, 0.5).')
    nonnegative = ('cycles', 'delete_checks', 'min_points', 'trials', 'refine_candidates',
                   'screen_checks', 'edge_prune', 'witness_trials', 'steps', 'polish_kicks')
    positive = ('candidates', 'insertion_iters', 'candidate_attempts', 'colorings', 'refreshes',
                'slsqp_iters', 'polish_passes', 'stall_passes', 'constraint_passes')
    if any(getattr(args, p) < 0 for p in nonnegative) or any(getattr(args, p) < 1 for p in positive):
        raise ValueError('Invalid nonnegative/positive iteration count.')
    for name in ('ftol', 'ratio_tol', 'feasibility_tol', 'active_tol', 'stationarity_tol', 'lp_timeout'):
        v = getattr(args, name)
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be finite and positive.')
    if args.sat_timeout < 0 or not math.isfinite(args.sat_timeout):
        raise ValueError('--sat-timeout must be finite and nonnegative; 0 disables the timeout.')
    if not 0 <= args.outer_slack < 0.5:
        raise ValueError('--outer-slack must be in [0,0.5).')
    if not 0 <= args.explore_loss < 0.1:
        raise ValueError('--explore-loss must be in [0,0.1).')
    if min(args.polish_jitter, args.min_separation) < 0:
        raise ValueError('Jitter/separation must be nonnegative.')
    if args.polish_only:
        args.cycles = args.delete_checks = args.steps = 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pool = None
    try:
        validate_args(args)
        g, witness, colors, saved_k = load_source(args.start)
        k = args.k if args.k is not None else (saved_k or g.dim+1)
        if k < 2 or g.m <= k:
            raise ValueError('Tuning requires k>=2 and more than k points.')
        if saved_k is not None and saved_k != k:
            witness, colors = None, None  # Re-evaluate for the newly requested k.
        args.k = k
        if args.max_points is None:
            args.max_points = g.m+24
        if args.max_points < k+1 or args.min_points > args.max_points:
            raise ValueError('Invalid point-count limits.')
        out = args.out
        if out.exists() and any(out.iterdir()):
            raise ValueError('Output directory is not empty. Use a new --out; resume via the old best as input.')
        out.mkdir(parents=True, exist_ok=True)
        config = {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()}
        (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        started = time.monotonic()
        def log(event: str, **data: Any) -> None:
            record = {'event': event, 'elapsed_seconds': round(time.monotonic()-started, 3), **data}
            with (out/'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            if not args.quiet:
                details = ' '.join(f'{key}={value:.12g}' if isinstance(value, float)
                                   else f'{key}={value}' for key, value in data.items()
                                   if key not in ('rejection_counts',))
                print(f'[{event}] {details}', flush=True)
        rng = np.random.default_rng(args.seed)
        pool = OraclePool(k, args)
        log('initial_evaluation', points=g.m, dimension=g.dim, parts=k,
            backend=pool.get(g.m).backend)
        initial = assess(g, pool, witness, colors)
        checkpoint = Checkpoints(out, k, log)
        checkpoint.consider(initial, 'input_rechecked')
        if not initial.exact:
            log('unresolved_initial_bracket', lower=initial.lower, upper=initial.upper,
                note='Timeout is UNKNOWN. Certified acceptance may be conservative; raise --sat-timeout.')
        reduced, reduction = reduce_vertices(initial, pool, rng, args.delete_checks,
                                              args.min_points, log)
        (out/'initial_reduction.json').write_text(json.dumps(reduction, indent=2)+'\n')
        checkpoint.consider(reduced, 'initial_ratio_preserving_reduction')
        polished = local_tune(checkpoint.best.g, checkpoint.best.witness, pool, rng, args,
                              log, smooth=not args.polish_only, colors=checkpoint.best.colors)
        checkpoint.consider(polished, 'initial_polishing')
        archive: list[State] = []
        for cycle in range(args.cycles):
            log('cycle_start', cycle=cycle+1, best=checkpoint.best.lower, points=checkpoint.best.g.m)
            # Every third cycle can explore a nearly-as-good distinct graph.
            # It is not reported as best unless the exact acceptance test passes.
            threshold = checkpoint.best.lower*(1-args.explore_loss)
            archive = [s for s in archive if s.lower >= threshold]
            source = (archive[int(rng.integers(len(archive)))] if archive and cycle % 3 == 2
                      else checkpoint.best)
            base, reduction = reduce_vertices(source, pool, rng, args.delete_checks,
                                               args.min_points, log)
            checkpoint.consider(base, f'cycle_{cycle+1}_ratio_preserving_reduction')
            proposals: list[Proposal] = []
            for trial in range(args.witness_trials):
                slack = args.slacks[int(rng.integers(len(args.slacks)))]
                factor = Fraction(str(1-slack))**2
                ids = base.g.threshold(base.lower2*factor)
                oracle = pool.get(base.g.m)
                ids = bs.prune_witness(oracle, ids, bs.distances2(base.g.x, base.g.pairs),
                                       rng, checks=args.edge_prune, noise=0.025)
                pg = base.g
                if args.compact_trials:
                    pg, ids = witness_compaction(pg, ids, args.min_points)
                proposals.append(Proposal(pg, ids, {'kind': 'old_points_new_witness', 'slack': slack}))
            proposals += insertion_proposals(base, pool, rng, args, log)
            for index, proposal in enumerate(proposals):
                folder = out/'trials'/f'cycle_{cycle+1:04d}_trial_{index+1:03d}'
                folder.mkdir(parents=True, exist_ok=True)
                (folder/'proposal.json').write_text(json.dumps(proposal.meta, indent=2)+'\n')
                log('trial_start', cycle=cycle+1, trial=index+1, points=proposal.g.m,
                    kind=proposal.meta['kind'])
                trial_initial = assess(proposal.g, pool, proposal.ids, witness_known=True)
                checkpoint.consider(trial_initial, f'cycle_{cycle+1}_trial_{index+1}_before_polish')
                try:
                    result = local_tune(proposal.g, proposal.ids, pool, rng, args, log)
                except ValueError as exc:
                    log('trial_rejected', reason=str(exc))
                    continue
                save_state(folder/'result', result, k, proposal.meta)
                checkpoint.consider(result, f'cycle_{cycle+1}_trial_{index+1}')
                if result.lower >= checkpoint.best.lower*(1-args.explore_loss):
                    archive.append(result)
                    archive = sorted(archive, key=lambda s: (-s.lower, s.g.m))[:8]
            log('cycle_finished', cycle=cycle+1, best=checkpoint.best.lower,
                points=checkpoint.best.g.m, **pool.stats())
        # Never finish immediately after a discrete move: polish the reported
        # incumbent itself and then audit its exported decimal coordinates.
        final_reduced, final_reduction = reduce_vertices(
            checkpoint.best, pool, rng, args.delete_checks, args.min_points, log)
        checkpoint.consider(final_reduced, 'final_ratio_preserving_reduction')
        (out/'final_reduction.json').write_text(json.dumps(final_reduction, indent=2)+'\n')
        log('final_polish', points=checkpoint.best.g.m, ratio=checkpoint.best.lower)
        final = local_tune(checkpoint.best.g, checkpoint.best.witness, pool, rng, args,
                           log, smooth=False, colors=checkpoint.best.colors)
        checkpoint.consider(final, 'final_polish_and_audit')
        best = checkpoint.best
        summary = {'initial_points': initial.g.m, 'initial_lower': initial.lower,
                   'initial_upper': initial.upper, 'best_points': best.g.m,
                   'best_lower': best.lower, 'best_upper': best.upper,
                   'exact_ratio_squared': (str(best.lower2) if best.exact else None),
                   'best_directory': str(out/'best'),
                   'polish_status': (best.polish or {}).get('status', 'not_polished'),
                   'oracle': pool.stats(), 'elapsed_seconds': time.monotonic()-started}
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2))
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Saved best, best.previous and immutable history snapshots remain intact.', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, ImportError, InvalidOperation) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    finally:
        if pool is not None:
            pool.close()


if __name__ == '__main__':
    raise SystemExit(main())
