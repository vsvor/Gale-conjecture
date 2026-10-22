#!/usr/bin/env python3
"""Search for finite sets with a large optimal k-part diameter ratio.

Python >=3.10; numpy and scipy required. Kissat is supported directly.
Optional backends: incremental PySAT (Glucose/Minisat), native DSATUR,
and a slower dependency-free Python DSATUR fallback.

Examples:
  python borsuk_search.py search --dim 3 --points 30:50 --restarts 40 \
      --workers 4 --backend external --solver './kissat' --out results3
  python borsuk_search.py verify results3/best --backend external --solver './kissat'

All reported lower bounds use a graph known NOT to be k-colourable.
UNKNOWN/TIMEOUT is never interpreted as UNSAT. The continuous search is
heuristic; the evaluation and exported integer-distance checks are separate.
"""
from __future__ import annotations

# Avoid BLAS oversubscription when several independent restarts run in parallel.
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = os.environ.get('BORSUK_BLAS_THREADS', '1')

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from dataclasses import dataclass
from enum import Enum
import importlib.util
from fractions import Fraction
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

VERSION = '1.1'


class Status(Enum):
    SAT = 'SAT'
    UNSAT = 'UNSAT'
    UNKNOWN = 'UNKNOWN'


@dataclass
class Answer:
    status: Status
    colors: np.ndarray | None = None
    core: np.ndarray | None = None
    reason: str = ''


@dataclass
class Evaluation:
    lower: float
    upper: float
    exact: bool
    witness: np.ndarray
    colors: np.ndarray | None
    lo: int
    hi: int
    unknown: bool = False


def pairs_for(m: int) -> np.ndarray:
    return np.column_stack(np.triu_indices(m, 1)).astype(np.int32)


def distances2(x: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    z = x[pairs[:, 0]] - x[pairs[:, 1]]
    return np.einsum('ij,ij->i', z, z)


def validate_points(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError('Points must be a nonempty finite m-by-d array.')
    if len(x) < 2 or not np.any(x != x[0]):
        raise ValueError('The set must have positive diameter.')
    return x


def normalize(x: np.ndarray, sphere: bool = False) -> np.ndarray:
    x = np.array(x, dtype=np.float64, copy=True)
    if sphere:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        if np.any(norms < 1e-15):
            raise ValueError('Cannot project a zero vector onto the sphere.')
        return x / norms
    x -= x.mean(axis=0)
    diam = math.sqrt(float(distances2(x, pairs_for(len(x))).max()))
    if not math.isfinite(diam) or diam <= 0:
        raise ValueError('Degenerate point set.')
    return x / diam


def base_cnf(m: int, k: int) -> list[list[int]]:
    """Exactly one colour per vertex, with safe value-precedence symmetry."""
    cnf: list[list[int]] = []
    for v in range(m):
        lits = [v*k+c+1 for c in range(k)]
        cnf.append(lits)
        cnf.extend([[-a, -b] for a, b in itertools.combinations(lits, 2)])
        # Every colouring admits a relabelling in order of first occurrence.
        for c in range(1, k):
            if c > v:
                cnf.append([-lits[c]])
            else:
                cnf.append([-lits[c]] + [u*k+c for u in range(v)])
    return cnf


def graph_cnf(m: int, k: int, edges: np.ndarray) -> list[list[int]]:
    cnf = base_cnf(m, k)
    for u, v in edges:
        for c in range(k):
            cnf.append([-int(u)*k-c-1, -int(v)*k-c-1])
    return cnf


def write_cnf(path: Path, nvars: int, clauses: list[list[int]]) -> None:
    with path.open('w', encoding='ascii') as f:
        f.write(f'p cnf {nvars} {len(clauses)}\n')
        for clause in clauses:
            f.write(' '.join(map(str, clause)) + ' 0\n')


def model_colors(model: list[int] | None, m: int, k: int) -> np.ndarray | None:
    if model is None:
        return None
    positive = {int(a) for a in model if 0 < a <= m*k}
    colors = np.full(m, -1, dtype=np.int32)
    for v in range(m):
        chosen = [c for c in range(k) if v*k+c+1 in positive]
        if len(chosen) != 1:
            return None
        colors[v] = chosen[0]
    return colors


def parse_external(text: str, returncode: int, m: int, k: int) -> Answer:
    """Accept standard Kissat 's/v' output and MiniSat result-file output."""
    statuses: set[Status] = set()
    literals: list[int] = []
    model_file = False
    for raw in text.splitlines():
        line = raw.strip()
        if line in ('s SATISFIABLE', 'SATISFIABLE', 'SAT'):
            statuses.add(Status.SAT)
            model_file = line == 'SAT'
        elif line in ('s UNSATISFIABLE', 'UNSATISFIABLE', 'UNSAT'):
            statuses.add(Status.UNSAT)
        elif line.startswith('v '):
            try:
                literals.extend(map(int, line[2:].split()))
            except ValueError:
                return Answer(Status.UNKNOWN, reason='Malformed SAT model.')
        elif model_file and line and all(c in ' -0123456789\t' for c in line):
            literals.extend(map(int, line.split()))
    if returncode in (10, 20):
        statuses.add(Status.SAT if returncode == 10 else Status.UNSAT)
    elif returncode not in (0,):
        return Answer(Status.UNKNOWN, reason=f'Solver exit code {returncode}.')
    if len(statuses) != 1:
        return Answer(Status.UNKNOWN, reason='Missing or contradictory solver status.')
    status = next(iter(statuses))
    return Answer(status, model_colors(literals, m, k) if status is Status.SAT else None)


def dsatur_python(m: int, k: int, edges: np.ndarray, timeout: float) -> Answer:
    """Exact fallback, mainly for small tests; timeout preserves UNKNOWN."""
    adj = [set() for _ in range(m)]
    for u, v in edges:
        adj[int(u)].add(int(v)); adj[int(v)].add(int(u))
    colors = [-1] * m
    degree = list(map(len, adj))
    removed: list[int] = []
    queue = [v for v in range(m) if degree[v] < k]
    for v in queue:
        if colors[v] == -2:
            continue
        colors[v] = -2
        removed.append(v)
        for u in adj[v]:
            if colors[u] != -2:
                degree[u] -= 1
                if degree[u] < k:
                    queue.append(u)
    counts = [[0] * k for _ in range(m)]
    masks = [0] * m
    deadline = time.monotonic() + timeout if timeout > 0 else math.inf
    nodes = 0

    def dfs(left: int, used: int) -> bool:
        nonlocal nodes
        if left == 0:
            return True
        nodes += 1
        if nodes % 256 == 1 and time.monotonic() >= deadline:
            raise TimeoutError
        v = max((u for u in range(m) if colors[u] == -1),
                key=lambda u: (masks[u].bit_count(), degree[u]))
        if masks[v].bit_count() >= k:
            return False
        for c in range(min(k, used + 1)):
            if masks[v] & (1 << c):
                continue
            colors[v] = c
            for u in adj[v]:
                if colors[u] == -1:
                    counts[u][c] += 1
                    masks[u] |= 1 << c
                    degree[u] -= 1
            if dfs(left - 1, max(used, c + 1)):
                return True
            for u in adj[v]:
                if colors[u] == -1:
                    counts[u][c] -= 1
                    if counts[u][c] == 0:
                        masks[u] &= ~(1 << c)
                    degree[u] += 1
            colors[v] = -1
        return False

    try:
        sat = dfs(m - len(removed), 0)
    except (TimeoutError, RecursionError):
        return Answer(Status.UNKNOWN, reason='DSATUR timeout/recursion limit.')
    if not sat:
        return Answer(Status.UNSAT)
    for v in reversed(removed):
        forbidden = {colors[u] for u in adj[v] if colors[u] >= 0}
        colors[v] = next(c for c in range(k) if c not in forbidden)
    return Answer(Status.SAT, np.array(colors, dtype=np.int32))


class GraphOracle:
    """A graph-only oracle, reusable while the point coordinates change.

    PySAT uses one activation literal per potential edge, so learnt clauses
    and assumption cores can be reused during threshold search and pruning.
    External Kissat receives a fresh DIMACS file for each uncached query.
    """
    def __init__(self, m: int, k: int, backend: str = 'auto',
                 solver: str = 'kissat', timeout: float = 10.0,
                 pysat_solver: str = 'g4'):
        if m < 1 or k < 1:
            raise ValueError('m and k must be positive.')
        self.m, self.k, self.pairs = m, k, pairs_for(m)
        self.timeout, self.command = timeout, solver
        self.cache: OrderedDict[tuple[int, ...], Answer] = OrderedDict()
        self.calls = self.unknowns = self.cache_hits = 0
        self.sat = None
        native = Path(__file__).resolve().with_name('dsatur_native.so')
        if backend == 'auto':
            exe = shlex.split(solver)[0] if shlex.split(solver) else ''
            if exe and (shutil.which(exe) or Path(exe).is_file()):
                backend = 'external'
            elif importlib.util.find_spec('pysat') is not None:
                backend = 'pysat'
            elif native.is_file():
                backend = 'native'
            else:
                backend = 'python'
        self.backend = backend
        if backend == 'external':
            parts = shlex.split(solver)
            if not parts or not (shutil.which(parts[0]) or Path(parts[0]).is_file()):
                raise ValueError(f'SAT solver not found: {solver!r}')
        elif backend == 'pysat':
            # Restrict to backends with a documented interruptible limited call.
            if pysat_solver not in ('g3', 'g4', 'm22', 'mgh'):
                raise ValueError('Use g3, g4, m22 or mgh for interruptible PySAT calls.')
            from pysat.solvers import Solver
            clauses = base_cnf(m, k)
            for i, (u, v) in enumerate(self.pairs):
                sel = m*k+i+1
                for c in range(k):
                    clauses.append([-sel, -int(u)*k-c-1, -int(v)*k-c-1])
            self.sat = Solver(name=pysat_solver, bootstrap_with=clauses)
        elif backend == 'native':
            if not native.is_file():
                raise ValueError(f'Native backend not found: {native}. '
                                 'Build dsatur_native.so as described in README.md.')
            if k > 63:
                raise ValueError('The optional native backend supports k <= 63.')
            self.library = ctypes.CDLL(str(native))
            self.native = self.library.graph_color
            arr = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
            self.native.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    arr, ctypes.c_double, arr]
            self.native.restype = ctypes.c_int
        elif backend != 'python':
            raise ValueError(f'Unknown backend {backend!r}.')

    def close(self) -> None:
        if self.sat is not None:
            self.sat.delete()
            self.sat = None

    def __enter__(self) -> GraphOracle:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def solve(self, ids: np.ndarray | list[int]) -> Answer:
        key = tuple(sorted(set(map(int, ids))))
        if key and (key[0] < 0 or key[-1] >= len(self.pairs)):
            raise ValueError('Invalid edge index.')
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.calls += 1
        edges = np.ascontiguousarray(self.pairs[list(key)], dtype=np.int32)
        if not key:
            ans = Answer(Status.SAT, np.zeros(self.m, dtype=np.int32))
        elif self.backend == 'external':
            ans = self._external(edges)
        elif self.backend == 'pysat':
            assert self.sat is not None
            assumptions = [self.m*self.k+i+1 for i in key]
            timer = None
            self.sat.clear_interrupt()
            if self.timeout > 0:
                timer = threading.Timer(self.timeout, self.sat.interrupt)
                timer.daemon = True
                timer.start()
            try:
                res = self.sat.solve_limited(assumptions=assumptions, expect_interrupt=True)
                if res is True:
                    ans = Answer(Status.SAT, model_colors(self.sat.get_model(), self.m, self.k))
                elif res is False:
                    core = self.sat.get_core()
                    ans = Answer(Status.UNSAT, core=np.array(
                        [a-self.m*self.k-1 for a in core], dtype=np.int32) if core else None)
                else:
                    ans = Answer(Status.UNKNOWN, reason='PySAT interrupted.')
            finally:
                if timer is not None:
                    timer.cancel()
                    timer.join()
                self.sat.clear_interrupt()
        elif self.backend == 'native':
            colors = np.full(self.m, -1, dtype=np.int32)
            code = self.native(self.m, self.k, len(edges), edges, self.timeout, colors)
            if code == -2:
                raise RuntimeError('Invalid input or internal failure in native DSATUR.')
            ans = Answer({1: Status.SAT, 0: Status.UNSAT, -1: Status.UNKNOWN}[code],
                         colors if code == 1 else None,
                         reason='Native DSATUR timeout.' if code == -1 else '')
        else:
            ans = dsatur_python(self.m, self.k, edges, self.timeout)
        if ans.status is Status.SAT and ans.colors is not None:
            c = ans.colors
            if (len(c) != self.m or np.any(c < 0) or np.any(c >= self.k)
                    or np.any(c[edges[:, 0]] == c[edges[:, 1]])):
                raise RuntimeError('Solver returned an invalid colouring.')
        if ans.status is Status.UNKNOWN:
            self.unknowns += 1
        else:
            self.cache[key] = ans
            if len(self.cache) > 2048:
                self.cache.popitem(last=False)
        return ans

    def _external(self, edges: np.ndarray) -> Answer:
        with tempfile.TemporaryDirectory(prefix='borsuk_sat_') as tmp:
            cnf_path = Path(tmp) / 'graph.cnf'
            model_path = Path(tmp) / 'model.txt'
            write_cnf(cnf_path, self.m*self.k, graph_cnf(self.m, self.k, edges))
            command = shlex.split(self.command)
            has_input = any('{cnf}' in item for item in command)
            command = [item.replace('{cnf}', str(cnf_path)).replace('{model}', str(model_path))
                       for item in command]
            if not has_input:
                command.append(str(cnf_path))
            try:
                proc = subprocess.run(command, capture_output=True, text=True,
                                      timeout=self.timeout if self.timeout > 0 else None,
                                      check=False)
            except subprocess.TimeoutExpired:
                return Answer(Status.UNKNOWN, reason='External solver timeout.')
            except OSError as exc:
                raise RuntimeError(f'Cannot run SAT solver: {exc}') from exc
            text = proc.stdout + '\n' + proc.stderr
            if model_path.exists():
                text += '\n' + model_path.read_text(encoding='ascii')
            return parse_external(text, proc.returncode, self.m, self.k)


def evaluate(x: np.ndarray, oracle: GraphOracle) -> Evaluation:
    """Bracket the true finite-set objective; exact unless a query is UNKNOWN.

    For distances d[0] >= ... >= d[N-1], if q is the FIRST UNSAT
    prefix LENGTH, the answer is d[q-1]/d[0], NOT d[q]/d[0].
    Equal distances need not be perturbed or artificially separated.
    """
    x = validate_points(x)
    m, k = oracle.m, oracle.k
    if len(x) != m:
        raise ValueError('Point count does not match the oracle.')
    if m <= k:
        return Evaluation(0., 0., True, np.empty(0, dtype=np.int32),
                          np.arange(m, dtype=np.int32), 0, 0)
    d2 = distances2(x, oracle.pairs)
    order = np.argsort(-d2, kind='stable')
    sd = d2[order]
    lo, hi = 0, len(order)  # empty graph SAT; K_m UNSAT when m > k
    colors: np.ndarray | None = np.zeros(m, dtype=np.int32)
    unknown = False
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ans = oracle.solve(order[:mid])
        if ans.status is Status.SAT:
            lo = mid
            colors = ans.colors
            if colors is not None:
                mono = colors[oracle.pairs[:, 0]] == colors[oracle.pairs[:, 1]]
                ub2 = float(d2[mono].max(initial=0.0))
                # This same colouring certifies every edge strictly longer than ub.
                lo = max(lo, int(np.searchsorted(-sd, -ub2, side='left')))
        elif ans.status is Status.UNSAT:
            hi = mid
            if ans.core is not None and len(ans.core):
                pos = np.empty(len(order), dtype=np.int32)
                pos[order] = np.arange(len(order))
                hi = min(hi, int(pos[ans.core].max()) + 1)
        else:
            unknown = True
            break
    lower = math.sqrt(float(sd[hi-1] / sd[0]))
    upper = math.sqrt(float(sd[lo] / sd[0]))
    if colors is not None:
        mono = colors[oracle.pairs[:, 0]] == colors[oracle.pairs[:, 1]]
        upper = min(upper, math.sqrt(float(d2[mono].max(initial=0.0) / sd[0])))
    # Equality here means an actual shared floating-point distance, not a tolerance.
    exact = hi == lo+1 or lower == upper
    return Evaluation(lower, upper, exact, order[:hi].copy(), colors, lo, hi, unknown)


def k_core_edges(m: int, k: int, pairs: np.ndarray, ids: np.ndarray) -> np.ndarray:
    alive = np.ones(m, dtype=bool)
    ids = np.array(ids, dtype=np.int32, copy=True)
    while len(ids):
        deg = np.bincount(pairs[ids].ravel(), minlength=m)
        bad = alive & (deg < k)
        if not np.any(bad):
            break
        alive[bad] = False
        ids = ids[alive[pairs[ids]].all(axis=1)]
    return ids


def prune_witness(oracle: GraphOracle, ids: np.ndarray, d2: np.ndarray,
                  rng: np.random.Generator, checks: int = 100,
                  noise: float = 0.03) -> np.ndarray:
    """Delete edges only on a definite UNSAT answer; may return a nonminimal core."""
    current = k_core_edges(oracle.m, oracle.k, oracle.pairs, ids)
    # Input must be an already-known UNSAT graph. k-core peeling preserves that.
    if not len(current):
        raise ValueError('Cannot prune an empty/noncritical witness.')
    if checks <= 0:
        return current
    ans = oracle.solve(current)
    if ans.status is Status.SAT:
        raise ValueError('The proposed witness is colourable.')
    if ans.status is Status.UNSAT and ans.core is not None and len(ans.core):
        current = k_core_edges(oracle.m, oracle.k, oracle.pairs, ans.core)
    scores = d2[current] / d2.max() + noise * rng.standard_normal(len(current))
    queue = current[np.argsort(scores)]
    active = set(map(int, current))
    calls = 0
    for edge in queue:
        edge = int(edge)
        if edge not in active:
            continue
        if calls >= checks:
            break
        trial = np.array(sorted(active - {edge}), dtype=np.int32)
        ans = oracle.solve(trial)
        calls += 1
        if ans.status is Status.UNSAT:
            if ans.core is not None and len(ans.core):
                trial = ans.core
            trial = k_core_edges(oracle.m, oracle.k, oracle.pairs, trial)
            active = set(map(int, trial))
        # SAT or UNKNOWN: retain the edge.
    return np.array(sorted(active), dtype=np.int32)


def smooth_value_grad(flat: np.ndarray, shape: tuple[int, int], pairs: np.ndarray,
                      ids: np.ndarray, power: float, sphere: bool = False
                      ) -> tuple[float, np.ndarray]:
    """Smooth scale-invariant upper approximation to log(D^2/min_edge d^2)."""
    y = np.asarray(flat).reshape(shape)
    if sphere:
        radii = np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-15)
        x = y / radii
    else:
        x = y
    diff = x[pairs[:, 0]] - x[pairs[:, 1]]
    d2 = np.einsum('ij,ij->i', diff, diff)
    clipped = np.maximum(d2, 1e-24)
    logs = np.log(clipped)
    a = power * logs
    b = -power * logs[ids]
    lsa, lsb = logsumexp(a), logsumexp(b)
    weights = np.exp(a-lsa)
    weights[ids] -= np.exp(b-lsb)
    # Where clipped, the derivative of the clipping operation is zero.
    weights[d2 < 1e-24] = 0.0
    forces = (2.0 * weights / clipped)[:, None] * diff
    grad = np.empty_like(x)
    for j in range(shape[1]):
        grad[:, j] = (np.bincount(pairs[:, 0], weights=forces[:, j], minlength=shape[0])
                      - np.bincount(pairs[:, 1], weights=forces[:, j], minlength=shape[0]))
    if sphere:
        grad = (grad - x * np.sum(x*grad, axis=1, keepdims=True)) / radii
    return float((lsa+lsb)/power), grad.ravel()


def witness_ratio(x: np.ndarray, pairs: np.ndarray, ids: np.ndarray) -> float:
    if not len(ids):
        return 0.0
    d2 = distances2(x, pairs)
    return math.sqrt(float(max(0.0, d2[ids].min()) / d2.max()))


def optimize_witness(x: np.ndarray, pairs: np.ndarray, ids: np.ndarray,
                     steps: int = 180, polish: int = 0,
                     sphere: bool = False,
                     powers: tuple[float, ...] = (8., 24., 80., 256., 1024.)
                     ) -> tuple[np.ndarray, float]:
    """Annealed L-BFGS, optional exact-constraint SLSQP; keep the best TRUE ratio."""
    x = normalize(x, sphere)
    if not len(ids):
        return x, 0.0
    shape = x.shape
    best_x, best_r = x.copy(), witness_ratio(x, pairs, ids)

    def record(flat: np.ndarray) -> None:
        nonlocal best_x, best_r
        candidate = np.asarray(flat).reshape(shape)
        if not np.isfinite(candidate).all():
            return
        if sphere:
            candidate = normalize(candidate, True)
        r = witness_ratio(candidate, pairs, ids)
        if math.isfinite(r) and r > best_r:
            best_x, best_r = candidate.copy(), r

    for power in powers:
        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            record(flat)
            return smooth_value_grad(flat, shape, pairs, ids, power, sphere)
        res = minimize(objective, x.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': steps, 'maxls': 35, 'ftol': 1e-13, 'gtol': 1e-8})
        if np.isfinite(res.x).all():
            x = normalize(res.x.reshape(shape), sphere)
            record(x.ravel())
    best_x = normalize(best_x, sphere)
    if polish > 0 and not sphere:
        polished, rp = slsqp_polish(best_x, pairs, ids, polish)
        if rp > best_r:
            best_x, best_r = polished, rp
    return normalize(best_x, sphere), best_r


def slsqp_polish(x: np.ndarray, pairs: np.ndarray, ids: np.ndarray,
                 steps: int = 100) -> tuple[np.ndarray, float]:
    """Maximise t with all pair distances squared <=1 and witness distances >=t.

    Feasibility tolerances are NOT used as a certificate: the returned graph
    ratio is always recomputed directly, including after an unsuccessful solve.
    """
    x = normalize(x)
    m, d = x.shape
    nv, ne = x.size, len(pairs)
    best_x, best_r = x.copy(), witness_ratio(x, pairs, ids)
    z0 = np.r_[x.ravel(), best_r**2]
    row = np.arange(ne)[:, None]
    col_u = pairs[:, 0, None]*d + np.arange(d)[None, :]
    col_v = pairs[:, 1, None]*d + np.arange(d)[None, :]

    def cons(z: np.ndarray) -> np.ndarray:
        ds = distances2(z[:-1].reshape(m, d), pairs)
        return np.r_[1.0-ds, ds[ids]-z[-1]]

    def jac(z: np.ndarray) -> np.ndarray:
        xx = z[:-1].reshape(m, d)
        diff = 2*(xx[pairs[:, 0]]-xx[pairs[:, 1]])
        base = np.zeros((ne, nv+1))
        base[row, col_u] = diff
        base[row, col_v] = -diff
        out = np.vstack((-base, base[ids]))
        out[ne:, -1] = -1
        return out

    def record(z: np.ndarray) -> None:
        nonlocal best_x, best_r
        xx = z[:-1].reshape(m, d)
        if np.isfinite(xx).all():
            r = witness_ratio(xx, pairs, ids)
            if r > best_r:
                best_x, best_r = xx.copy(), r
    grad = np.zeros(nv+1); grad[-1] = -1
    res = minimize(lambda z: -z[-1], z0, jac=lambda z: grad,
                   method='SLSQP', callback=record,
                   constraints=[{'type': 'ineq', 'fun': cons, 'jac': jac}],
                   bounds=[(-2., 2.)]*nv + [(0., 1.)],
                   options={'maxiter': steps, 'ftol': 1e-12, 'disp': False})
    record(res.x)
    return normalize(best_x), best_r


def regular_simplex(d: int) -> np.ndarray:
    """d+1 unit vertices in R^d, with every off-diagonal inner product -1/d."""
    if d < 1:
        raise ValueError('Simplex dimension must be positive.')
    h = np.zeros((d+1, d))
    for j in range(d):
        h[:j+1, j] = 1 / math.sqrt((j+1)*(j+2))
        h[j+1, j] = -(j+1) / math.sqrt((j+1)*(j+2))
    return h * math.sqrt((d+1)/d)


def sample_edges(vertices: np.ndarray, edges: list[tuple[int, int]], m: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Include vertices, then allocate stratified samples approximately evenly to edges."""
    if m <= len(vertices):
        return vertices[rng.choice(len(vertices), m, replace=False)].copy()
    out = [v.copy() for v in vertices]
    counts = np.full(len(edges), (m-len(vertices))//len(edges), dtype=int)
    counts[rng.permutation(len(edges))[:(m-len(vertices)) % len(edges)]] += 1
    for (u, v), count in zip(edges, counts):
        for j in range(count):
            t = (j + 0.5 + rng.uniform(-0.3, 0.3)) / count
            out.append((1-t)*vertices[u] + t*vertices[v])
    return np.array(out)


SEEDS = ('simplex-edges', 'prism-edges', 'sphere', 'simplex-chords',
         'prism-sphere', 'simplex-faces')


def make_seed(d: int, m: int, kind: str, rng: np.random.Generator,
              jitter: float = 0.015) -> np.ndarray:
    if kind == 'simplex-product':
        # Fitted R5 proposal family, experimentally extrapolated to R6--R8.
        # This changes initialization only; all evaluation stays in this module.
        from simplex_product_seed import make_seed as make_simplex_product_seed
        return make_simplex_product_seed(d, m, rng, jitter=jitter)
    if kind == 'sphere':
        x = rng.normal(size=(m, d))  # Normalised Gaussian is uniform, unlike a cube sample.
        return normalize(x, True)
    project = kind in ('simplex-edges', 'simplex-faces', 'prism-sphere')
    if kind.startswith('simplex'):
        vertices = regular_simplex(d)
        if kind == 'simplex-faces':
            out = [v.copy() for v in vertices[:m]]
            face_size = min(3, d+1)
            while len(out) < m:
                face = rng.choice(d+1, face_size, replace=False)
                weights = rng.dirichlet(np.full(face_size, 0.65))
                out.append(weights @ vertices[face])
            x = np.array(out)
        else:
            x = sample_edges(vertices, list(itertools.combinations(range(d+1), 2)), m, rng)
    elif kind.startswith('prism'):
        if d < 2:
            raise ValueError('Prism seeds require dimension >=2.')
        base = regular_simplex(d-1)
        height = math.exp(rng.uniform(math.log(0.3), math.log(1.2)))
        vertices = np.vstack((np.column_stack((base, np.full(d, -height))),
                              np.column_stack((base, np.full(d, height)))))
        edges = (list(itertools.combinations(range(d), 2))
                 + [(u+d, v+d) for u, v in itertools.combinations(range(d), 2)]
                 + [(u, u+d) for u in range(d)])
        x = sample_edges(vertices, edges, m, rng)
    else:
        raise ValueError(f'Unknown initializer {kind!r}.')
    if project:
        x = normalize(x, True)
    else:
        x = normalize(x)
    if jitter > 0:
        x += jitter * rng.normal(size=x.shape) / math.sqrt(d)
    return normalize(x, project)


def replenish_unused(x: np.ndarray, pairs: np.ndarray, ids: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """Place unused vertices on short segments in the active convex hull.

    This cannot increase the diameter or change a witness-edge length. It
    gives inactive vertices new positions for the next graph-refresh step.
    """
    active = np.unique(pairs[ids])
    inactive = np.setdiff1d(np.arange(len(x)), active)
    if not len(inactive):
        return x
    x = x.copy()
    for v in inactive:
        a, b = rng.choice(active, 2, replace=False)
        t = rng.uniform(0.04, 0.35)
        x[v] = (1-t)*x[a] + t*x[b]
    return x


def mutate(x: np.ndarray, rng: np.random.Generator, magnitude: float,
           sphere: bool = False) -> np.ndarray:
    x = normalize(x, sphere)
    mode = int(rng.integers(3))
    if mode == 0:
        mask = rng.random(len(x)) < 0.25
        if not np.any(mask):
            mask[int(rng.integers(len(x)))] = True
        x[mask] += magnitude*rng.normal(size=(int(mask.sum()), x.shape[1]))
    elif mode == 1:
        a = np.eye(x.shape[1]) + magnitude*rng.normal(size=(x.shape[1], x.shape[1]))
        x = x @ a
    else:
        x += magnitude*rng.normal(size=x.shape) / math.sqrt(x.shape[1])
    return normalize(x, sphere)


def load_points(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.is_dir():
        path = path / 'points.csv'
    if path.suffix == '.npy':
        x = np.load(path, allow_pickle=False)
    elif path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            x = data['points'] if 'points' in data else data['x']
    elif path.suffix == '.json':
        data = json.loads(path.read_text())
        x = data.get('points', data.get('x')) if isinstance(data, dict) else data
    else:
        x = np.loadtxt(path, delimiter=',' if path.suffix == '.csv' else None,
                       ndmin=2)
    return validate_points(x)


def decimal_integer_points(x: np.ndarray, digits: int = 17) -> tuple[list[list[int]], int]:
    """Round to exact decimal rationals; separate any identical rounded rows."""
    rows: list[list[int]] = []
    used: set[tuple[int, ...]] = set()
    adjusted = 0
    for row in x:
        ints = [int(format(float(a), f'.{digits}f').replace('.', '')) for a in row]
        if tuple(ints) in used:
            adjusted += 1
            while tuple(ints) in used:
                ints[0] += 1
        used.add(tuple(ints)); rows.append(ints)
    return rows, adjusted


def format_integer_point(a: int, digits: int = 17) -> str:
    sign = '-' if a < 0 else ''
    a = abs(a)
    scale = 10**digits
    return f'{sign}{a//scale}.{a%scale:0{digits}d}'


def rational_geometry(rows: list[list[int]], edges: np.ndarray,
                      colors: np.ndarray | None, digits: int = 12) -> dict[str, Any]:
    """All squared distances below are Python integers; the scale cancels."""
    m = len(rows)
    distance: dict[tuple[int, int], int] = {}
    for i in range(m):
        for j in range(i+1, m):
            distance[i, j] = sum((a-b)**2 for a, b in zip(rows[i], rows[j]))
    diameter2 = max(distance.values())
    if diameter2 <= 0:
        raise ValueError('Zero diameter in saved coordinates.')
    lower2 = min((distance[tuple(sorted(map(int, e)))] for e in edges), default=0)
    upper2 = (max((v for (i, j), v in distance.items() if colors[i] == colors[j]), default=0)
              if colors is not None else diameter2)
    scale = 10**digits
    low = math.isqrt((lower2 * scale**2) // diameter2)
    up = math.isqrt((upper2 * scale**2) // diameter2)
    if up*up*diameter2 < upper2*scale**2:
        up += 1
    return {'lower_decimal': format_integer_point(low, digits),
            'upper_decimal': format_integer_point(up, digits),
            'bound_decimal_digits': digits,
            'diameter_squared_integer': str(diameter2),
            'min_witness_squared_integer': str(lower2),
            'max_part_squared_integer': str(upper2),
            'distinct_points': len({tuple(r) for r in rows})}


def save_candidate(directory: Path, x: np.ndarray, k: int, ev: Evaluation,
                   pairs: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    rows, adjusted = decimal_integer_points(x)
    # The same UNSAT graph remains a valid witness after coordinate rounding.
    edges = pairs[ev.witness]
    geometry = rational_geometry(rows, edges, ev.colors)
    data = {'version': VERSION, 'dimension': x.shape[1], 'm': len(x), 'k': k,
            'floating_lower': ev.lower, 'floating_upper': ev.upper,
            'threshold_resolved': ev.exact, 'coordinate_decimal_digits': 17,
            'duplicate_rows_separated_by_1e_minus_17': adjusted,
            'witness_edges': len(edges),
            'geometry': geometry, 'run': metadata,
            'qualification': 'Geometry uses exact integer arithmetic. Graph noncolourability '
                             'uses an UNSAT solver/search result or the complete-graph base case, not an independently '
                             'checked SAT proof. This is not a global optimality claim.'}
    text = '\n'.join(','.join(format_integer_point(a) for a in row) for row in rows) + '\n'
    (directory / 'points.csv').write_text(text, encoding='ascii')
    np.save(directory / 'points.npy', np.array(rows, dtype=float) / 1e17)
    np.savetxt(directory / 'witness_edges.csv', edges, fmt='%d', delimiter=',')
    write_cnf(directory / 'witness.cnf', len(x)*k, graph_cnf(len(x), k, edges))
    (directory / 'partition.json').write_text(json.dumps(
        None if ev.colors is None else ev.colors.tolist()) + '\n', encoding='utf-8')
    # Publish the metadata last; it is the completion marker for a snapshot.
    temp = directory / 'certificate.json.tmp'
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temp.replace(directory / 'certificate.json')
    return data


def verify_candidate(directory: Path, backend: str, solver: str,
                     timeout: float, pysat_solver: str) -> dict[str, Any]:
    meta = json.loads((directory / 'certificate.json').read_text())
    precision = int(meta['coordinate_decimal_digits'])
    rows = []
    for line in (directory / 'points.csv').read_text().splitlines():
        if not line.strip():
            continue
        values = line.split(',')
        ints = []
        for value in values:
            scaled = Fraction(value) * (10**precision)
            if scaled.denominator != 1:
                raise ValueError('Coordinates do not have the declared decimal precision.')
            ints.append(int(scaled))
        rows.append(ints)
    m, d, k = int(meta['m']), int(meta['dimension']), int(meta['k'])
    if len(rows) != m or any(len(row) != d for row in rows):
        raise ValueError('Saved dimensions disagree with the coordinate file.')
    edge_text = (directory / 'witness_edges.csv').read_text().strip()
    edges = (np.loadtxt(directory / 'witness_edges.csv', delimiter=',', dtype=np.int32, ndmin=2)
             if edge_text else np.empty((0, 2), dtype=np.int32))
    if edges.shape[1] != 2 or np.any(edges < 0) or np.any(edges >= m) or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError('Invalid saved witness edges.')
    colors_json = json.loads((directory / 'partition.json').read_text())
    colors = None if colors_json is None else np.asarray(colors_json, dtype=np.int32)
    if colors is not None and (colors.shape != (m,) or np.any(colors < 0) or np.any(colors >= k)):
        raise ValueError('Invalid saved partition.')
    with GraphOracle(m, k, backend, solver, timeout, pysat_solver) as oracle:
        lookup = {tuple(map(int, pair)): i for i, pair in enumerate(oracle.pairs)}
        ids = np.array([lookup[tuple(sorted(map(int, e)))] for e in edges], dtype=np.int32)
        ans = oracle.solve(ids) if len(edges) else Answer(Status.SAT)
        if len(edges) and ans.status is not Status.UNSAT:
            raise RuntimeError(f'Witness not verified: {ans.status.value}. {ans.reason}')
        if not len(edges) and m > k:
            raise RuntimeError('A positive lower bound requires a nonempty UNSAT witness.')
        geometry = rational_geometry(rows, edges, colors)
        if geometry['distinct_points'] != m:
            raise ValueError('Saved coordinates are not distinct.')
        old = meta['geometry']
        if any(geometry[key] != old[key] for key in ('lower_decimal', 'upper_decimal',
                 'diameter_squared_integer', 'min_witness_squared_integer', 'max_part_squared_integer')):
            raise ValueError('Certificate bounds disagree with the saved data.')
        return {'verified': True, 'backend': oracle.backend, 'dimension': d, 'm': m, 'k': k,
                'witness_status': ans.status.value, **geometry}



# Installed by the process-pool initializer, NOT passed with every submitted task.
# A synchronized Value must be transferred while workers are being started. This
# works with fork, spawn and forkserver without a manager server or disk polling.
_SEARCH_RECORD: Any = None


def _init_search_worker(record: Any) -> None:
    global _SEARCH_RECORD
    _SEARCH_RECORD = record


def _search_record_lower() -> float:
    if _SEARCH_RECORD is None:
        return 0.0
    with _SEARCH_RECORD.get_lock():
        return float(_SEARCH_RECORD.value)


def _publish_search_record(lower: float) -> None:
    """Publish a saved-coordinate lower bound with an atomic maximum update."""
    if _SEARCH_RECORD is None:
        return
    if not math.isfinite(lower) or not 0 <= lower <= 1:
        raise ValueError('Invalid lower bound for the shared search record.')
    with _SEARCH_RECORD.get_lock():
        if lower > _SEARCH_RECORD.value:
            _SEARCH_RECORD.value = lower


def parse_early_stop(value: str) -> list[tuple[int, float]]:
    """Parse a piecewise-constant ROUND:GAP schedule (absolute ratio gaps)."""
    rules: list[tuple[int, float]] = []
    try:
        for item in value.split(','):
            parts = item.strip().split(':')
            if len(parts) != 2:
                raise ValueError
            completed, gap = int(parts[0]), float(parts[1])
            if completed < 1 or not math.isfinite(gap) or not 0 <= gap <= 1:
                raise ValueError
            if rules and completed <= rules[-1][0]:
                raise ValueError
            rules.append((completed, gap))
        if not rules:
            raise ValueError
    except (ValueError, TypeError, OverflowError) as exc:
        raise argparse.ArgumentTypeError(
            'Use ROUND:GAP[,ROUND:GAP,...], e.g. 2:0.02,5:0.005. '
            'Rounds must be positive and strictly increasing; gaps must be finite in [0,1].'
        ) from exc
    return rules


def early_stop_status(config: dict[str, Any], run: int, rounds_completed: int,
                      best_seen_upper: float, record_lower: float) -> dict[str, Any]:
    """Pure stopping policy. It is a heuristic, not a bound on future improvement.

    Compare the highest evaluation upper bound seen in this restart with a saved
    lower bound attained anywhere in this invocation. In particular, a weak lower
    bound caused by a SAT timeout must not be mistaken for a poor configuration.
    Once active, a rule is checked after EVERY round until the next rule takes over.
    """
    rules = config.get('early_stop')
    status: dict[str, Any] = {
        'stop': False, 'reason': 'disabled', 'run': run,
        'rounds_completed': rounds_completed, 'best_seen_upper': best_seen_upper,
        'record_lower': record_lower, 'rule_round': None, 'gap': None, 'cutoff': None,
    }
    if not rules:
        return status
    warmup = config.get('early_stop_warmup', 1)
    keep_every = config.get('early_stop_keep_every', 10)
    if run < warmup:
        status['reason'] = 'protected_warmup'
        return status
    if keep_every and (run + 1) % keep_every == 0:
        status['reason'] = 'protected_exploration'
        return status
    active = [(int(r), float(g)) for r, g in rules if rounds_completed >= r]
    if not active:
        status['reason'] = 'before_first_check'
        return status
    rule_round, gap = active[-1]
    status.update(rule_round=rule_round, gap=gap)
    if not math.isfinite(record_lower) or record_lower <= 0:
        status['reason'] = 'no_positive_record'
        return status
    if not math.isfinite(best_seen_upper) or best_seen_upper < 0:
        status['reason'] = 'unresolved_upper_bound'
        return status
    cutoff = record_lower - gap
    status['cutoff'] = cutoff
    # Do not make stop decisions from tiny floating-point discrepancies.
    status['stop'] = best_seen_upper + 1e-12 < cutoff
    status['reason'] = 'behind_record' if status['stop'] else 'within_gap'
    return status


def _run_restart(config: dict[str, Any], run: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    d, k = config['dim'], config['k']
    m = int(rng.integers(config['m_min'], config['m_max'] + 1))
    kind = SEEDS[run % len(SEEDS)] if config['init'] == 'mix' else config['init']
    sphere = config['sphere']
    if config['start']:
        x = load_points(config['start'])
        if x.shape[1] != d:
            raise ValueError('Starting coordinates have the wrong dimension.')
        m = len(x)
        if run > 0:
            x = mutate(x, rng, config['mutation'], sphere)
        kind = 'loaded'
    else:
        x = make_seed(d, m, kind, rng, config['seed_jitter'])
    x = normalize(x, sphere)
    out = Path(config['out']) / f'run_{run:05d}'
    out.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    best_x = x.copy()
    best_ev: Evaluation | None = None
    best_value = -1.0
    best_seen_upper = 0.0
    rounds_completed = 0
    stop_reason = 'round_budget_exhausted'
    stop_detail: dict[str, Any] | None = None
    polish_performed = False
    stagnation = 0
    with GraphOracle(m, k, config['backend'], config['solver'], config['sat_timeout'],
                     config['pysat_solver']) as oracle:
        def consider(candidate: np.ndarray, ev: Evaluation, stage: str, iteration: int) -> bool:
            nonlocal best_value, best_x, best_ev, best_seen_upper
            # Track every evaluated state, even one not saved because its lower
            # bound is weak. A timeout bracket may still contain a good result.
            best_seen_upper = max(best_seen_upper, ev.upper)
            if ev.lower <= best_value + 1e-13:
                return False
            best_value, best_x, best_ev = ev.lower, candidate.copy(), ev
            run_meta = {'run': run, 'seed': seed, 'initializer': kind, 'iteration': iteration,
                        'stage': stage, 'backend': oracle.backend,
                        'elapsed_seconds': time.monotonic()-start_time,
                        'oracle_calls': oracle.calls, 'unknown_answers': oracle.unknowns,
                        'config': config}
            cert = save_candidate(out / 'best', candidate, k, ev, oracle.pairs, run_meta)
            if config.get('early_stop'):
                # Publish only AFTER the completed snapshot is on disk. The decimal
                # endpoint is rounded down; nextafter also avoids rounding it up
                # while converting that endpoint to the shared double.
                lower = float(cert['geometry']['lower_decimal'])
                _publish_search_record(max(0.0, math.nextafter(lower, -math.inf)))
            line = {'run': run, 'm': m, 'iteration': iteration, 'stage': stage,
                    'lower': ev.lower, 'upper': ev.upper, 'exact': ev.exact,
                    'elapsed': run_meta['elapsed_seconds']}
            with (out / 'improvements.jsonl').open('a') as f:
                f.write(json.dumps(line) + '\n')
            if not config['quiet']:
                label = '=' if ev.exact else 'in'
                print(f"run {run:04d} {kind:15s} m={m:3d} round={iteration:2d} "
                      f"R {label} [{cert['geometry']['lower_decimal']}, "
                      f"{cert['geometry']['upper_decimal']}]", flush=True)
            return True

        ev = evaluate(x, oracle)
        consider(x, ev, 'initial', -1)
        for iteration in range(config['rounds']):
            if not len(ev.witness):
                stop_reason = 'no_noncolorable_witness'
                break
            d2 = distances2(x, oracle.pairs)
            slack = 0.0 if iteration == 0 else config['graph_slack'] * rng.uniform(0.25, 1.0)
            floor = float(d2[ev.witness].min()) * (1.0-slack)**2
            ids = np.flatnonzero(d2 >= floor).astype(np.int32)
            # The enlarged graph contains the known UNSAT witness.
            ids = prune_witness(oracle, ids, d2, rng, config['prune_checks'],
                                noise=0.015+slack)
            candidate, _ = optimize_witness(x, oracle.pairs, ids, steps=config['steps'],
                                            sphere=sphere)
            if not sphere:
                candidate = replenish_unused(candidate, oracle.pairs, ids, rng)
            new_ev = evaluate(candidate, oracle)
            improved = consider(candidate, new_ev, 'refresh', iteration)
            stagnation = 0 if improved else stagnation+1
            x, ev = candidate, new_ev
            rounds_completed = iteration + 1
            if config.get('early_stop'):
                check = early_stop_status(config, run, rounds_completed,
                                          best_seen_upper, _search_record_lower())
                check['best_lower'] = best_value
                check['elapsed'] = time.monotonic() - start_time
                with (out / 'early_stop.jsonl').open('a') as f:
                    f.write(json.dumps(check) + '\n')
                if check['stop']:
                    stop_reason, stop_detail = 'early_stop', check
                    if not config['quiet']:
                        print(f"run {run:04d} EARLY STOP after {rounds_completed} rounds: "
                              f"best upper={best_seen_upper:.9f}, "
                              f"record lower={check['record_lower']:.9f}, "
                              f"gap={check['gap']:.6g}; remaining rounds and final polish skipped",
                              flush=True)
                    break
            if stagnation >= 2:
                # Basin hopping: evaluate the mutated point set, rather than
                # inheriting an old ratio after changing the geometry.
                assert best_ev is not None
                strength = config['mutation'] * rng.uniform(0.4, 2.0)
                x = mutate(best_x, rng, strength, sphere)
                ev = evaluate(x, oracle)
                consider(x, ev, 'mutation', iteration)
                stagnation = 0
        assert best_ev is not None
        if (stop_reason != 'early_stop' and config['polish'] > 0
                and not sphere and len(best_ev.witness)):
            polish_performed = True
            ds = distances2(best_x, oracle.pairs)
            ids = prune_witness(oracle, best_ev.witness, ds, rng, config['prune_checks'])
            candidate, _ = slsqp_polish(best_x, oracle.pairs, ids, config['polish'])
            final_ev = evaluate(candidate, oracle)
            consider(candidate, final_ev, 'SLSQP polish', config['rounds'])
        result = {'run': run, 'best': best_value, 'path': str(out / 'best'),
                  'elapsed': time.monotonic()-start_time, 'calls': oracle.calls,
                  'unknown': oracle.unknowns, 'rounds_completed': rounds_completed,
                  'rounds_skipped': config['rounds'] - rounds_completed,
                  'stop_reason': stop_reason, 'early_stopped': stop_reason == 'early_stop',
                  'early_stop_detail': stop_detail, 'best_seen_upper': best_seen_upper,
                  'polish_performed': polish_performed,
                  'polish_skipped_by_early_stop': (stop_reason == 'early_stop'
                      and config['polish'] > 0 and not sphere and bool(len(best_ev.witness)))}
        (out / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
        return result


def parse_point_range(value: str) -> tuple[int, int]:
    parts = value.split(':')
    if len(parts) == 1:
        a = b = int(parts[0])
    elif len(parts) == 2:
        a, b = map(int, parts)
    else:
        raise ValueError('Use --points M or --points MIN:MAX.')
    if a < 2 or b < a:
        raise ValueError('Invalid point-count range.')
    return a, b


def add_backend_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--backend', choices=['auto', 'external', 'pysat', 'native', 'python'], default='auto')
    parser.add_argument('--solver', default='kissat',
                        help="External command, e.g. './kissat', or 'minisat {cnf} {model}'. No shell is used.")
    parser.add_argument('--sat-timeout', type=float, default=10., help='Seconds per oracle call; 0 means unlimited.')
    parser.add_argument('--pysat-solver', default='g4', choices=['g3', 'g4', 'm22', 'mgh'])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='action', required=True)
    search = sub.add_parser('search', help='Multistart geometric optimisation.')
    search.add_argument('--dim', type=int, required=True)
    search.add_argument('--k', type=int, help='Number of parts; default: dimension+1 (NOT dimension+2).')
    search.add_argument('--points', default='40', help='M or MIN:MAX; ignored when --start is supplied.')
    search.add_argument('--restarts', type=int, default=12)
    search.add_argument('--workers', type=int, default=1)
    search.add_argument('--rounds', type=int, default=10)
    search.add_argument('--steps', type=int, default=180, help='L-BFGS iterations per annealing stage, five stages per round.')
    search.add_argument('--polish', type=int, default=100, help='Final SLSQP iterations per restart; 0 disables.')
    search.add_argument('--prune-checks', type=int, default=120)
    search.add_argument('--graph-slack', type=float, default=0.025)
    search.add_argument('--mutation', type=float, default=0.035)
    search.add_argument('--seed-jitter', type=float, default=0.015)
    search.add_argument('--init', choices=['mix', *SEEDS, 'simplex-product'], default='mix')
    search.add_argument('--sphere', action='store_true', help='Constrain optimization to the unit sphere; default releases it.')
    search.add_argument('--start', help='Start from points.csv/.txt/.npy/.npz/.json, or a saved best directory.')
    search.add_argument('--seed', type=int, default=20260908)
    search.add_argument('--out', default='borsuk_results')
    search.add_argument('--quiet', action='store_true')
    search.add_argument('--early-stop', nargs='?', type=parse_early_stop,
                        const='2:0.02,5:0.005', default=None, metavar='ROUND:GAP,...',
                        help='Stop lagging restarts after completed rounds. Optional absolute-gap '
                             'schedule; bare flag means 2:0.02,5:0.005. Disabled when omitted. '
                             'An active rule is checked each subsequent round. Final polish is skipped.')
    search.add_argument('--early-stop-warmup', type=int, default=1, metavar='N',
                        help='Protect the first N numbered restarts from early stopping (default: 1). '
                             'They are not run serially before other workers.')
    search.add_argument('--early-stop-keep-every', type=int, default=10, metavar='N',
                        help='Also protect every Nth restart (one-based) for exploration; '
                             'default: 10; 0 disables this protection.')
    add_backend_options(search)
    verify = sub.add_parser('verify', help='Recheck the UNSAT witness and exact integer geometry.')
    verify.add_argument('directory', type=Path)
    add_backend_options(verify)
    assess = sub.add_parser('evaluate', help='Evaluate an existing point set and export its witness.')
    assess.add_argument('points_file')
    assess.add_argument('--k', type=int)
    assess.add_argument('--out', default='evaluated_set')
    add_backend_options(assess)
    args = parser.parse_args(argv)
    try:
        if args.sat_timeout < 0:
            raise ValueError('--sat-timeout must be nonnegative.')
        if args.action == 'verify':
            result = verify_candidate(args.directory, args.backend, args.solver,
                                      args.sat_timeout, args.pysat_solver)
            print(json.dumps(result, indent=2))
            return 0
        if args.action == 'evaluate':
            x = load_points(args.points_file)
            k = args.k if args.k is not None else x.shape[1]+1
            with GraphOracle(len(x), k, args.backend, args.solver, args.sat_timeout,
                             args.pysat_solver) as oracle:
                ev = evaluate(x, oracle)
                data = save_candidate(Path(args.out), x, k, ev, oracle.pairs,
                                      {'source': args.points_file, 'backend': oracle.backend})
            print(json.dumps(data, indent=2))
            return 0
        config = vars(args).copy()
        config.pop('action')
        config['k'] = args.k if args.k is not None else args.dim+1
        config['m_min'], config['m_max'] = parse_point_range(args.points)
        if args.dim < 2 or config['k'] < 1:
            raise ValueError('Search requires dimension >=2 and k>=1.')
        if min(args.restarts, args.workers, args.steps) < 1 or min(args.rounds, args.polish, args.prune_checks) < 0:
            raise ValueError('Invalid iteration/worker count.')
        if not 0 <= args.graph_slack < 1 or min(args.mutation, args.seed_jitter) < 0:
            raise ValueError('Invalid slack/mutation/jitter parameter.')
        if min(args.early_stop_warmup, args.early_stop_keep_every) < 0:
            raise ValueError('Early-stop warmup and keep-every counts must be nonnegative.')
        if args.early_stop and args.rounds < args.early_stop[0][0]:
            print('Warning: --rounds ends before the first early-stop checkpoint.', file=sys.stderr)
        if args.sphere and args.polish:
            print('Sphere constraint enabled: the free-coordinate SLSQP polish is skipped.', file=sys.stderr)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        # Never silently replace an earlier experiment.
        if (out / 'config.json').exists() or (out / 'best').exists():
            raise ValueError('Output directory contains a prior run. Choose a new --out and use --start to resume.')
        (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        seeds = [int(s.generate_state(1, dtype=np.uint64)[0])
                 for s in np.random.SeedSequence(args.seed).spawn(args.restarts)]
        results: list[dict[str, Any]] = []
        best = -1.0
        def collect(result: dict[str, Any]) -> None:
            nonlocal best
            results.append(result)
            with (out / 'completed.jsonl').open('a') as f:
                f.write(json.dumps(result) + '\n')
            if result['best'] > best:
                best = result['best']
                destination = out / 'best'
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(result['path'], destination)
        # Live record includes improvements from unfinished restarts. Only a double
        # and its short-lived lock are shared; coordinates and solvers remain local.
        # Use the SAME context for the Value and pool, including under spawn.
        context = mp.get_context()
        record = context.Value('d', 0.0) if args.early_stop else None
        if args.workers == 1:
            previous_record = _SEARCH_RECORD
            _init_search_worker(record)
            try:
                for run, seed in enumerate(seeds):
                    collect(_run_restart(config, run, seed))
            finally:
                _init_search_worker(previous_record)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                                     initializer=_init_search_worker, initargs=(record,)) as pool:
                futures = [pool.submit(_run_restart, config, run, seed) for run, seed in enumerate(seeds)]
                for future in as_completed(futures):
                    collect(future.result())
        result = {'best_lower': best, 'best_directory': str(out / 'best'),
                  'restarts': len(results), 'results': results,
                  'early_stopped': sum(r['early_stopped'] for r in results),
                  'rounds_completed': sum(r['rounds_completed'] for r in results),
                  'rounds_skipped': sum(r['rounds_skipped'] for r in results),
                  'early_stop_record_lower': float(record.value) if record is not None else None}
        (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: result[key] for key in ('best_lower', 'best_directory', 'restarts', 'early_stopped', 'rounds_completed', 'rounds_skipped')}, indent=2))
        return 0
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('Interrupted. Per-restart best snapshots remain in the output directory.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
