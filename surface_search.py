#!/usr/bin/env python3
"""Add exact boundary points to a fixed convex hull to seek larger R_k.

Python >= 3.10; NumPy, SciPy, and the accompanying borsuk_search.py.
Search is heuristic. Bounds and boundary membership are checked separately.
Use `python surface_search.py search --help` or `... verify --help`.
"""
from __future__ import annotations

# Import the engine first: it also sets BLAS thread limits before importing NumPy.
import borsuk_search as bs
import argparse
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import bisect
import json
import math
import os
import shutil
import sys
import time
from typing import Any, Callable
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError

VERSION = '1.0'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sqdist(a, b) -> int:
    return sum((int(u)-int(v))**2 for u, v in zip(a, b))


def decimal_scale(denominator: int) -> int:
    """Smallest d such that denominator divides 10**d."""
    a = b = 0
    while denominator % 2 == 0:
        denominator //= 2; a += 1
    while denominator % 5 == 0:
        denominator //= 5; b += 1
    require(denominator == 1, 'Only finite-decimal/binary coordinates are supported.')
    return max(a, b)


def decimal_text(value: int, digits: int) -> str:
    sign = '-' if value < 0 else ''
    v = abs(int(value))
    if not digits:
        return sign + str(v)
    scale = 10**digits
    return f'{sign}{v//scale}.{v%scale:0{digits}d}'


def write_rows(path: Path, rows: list[tuple[int, ...]], scale: int) -> int:
    digits = decimal_scale(scale)
    multiplier = 10**digits // scale
    path.write_text('\n'.join(','.join(decimal_text(a*multiplier, digits)
                         for a in row) for row in rows)+'\n', encoding='ascii')
    return digits


def sqrt_bounds(q: Fraction, digits: int = 15) -> tuple[str, str]:
    scale = 10**digits
    lo = math.isqrt(q.numerator*scale*scale//q.denominator)
    hi = lo + (lo*lo*q.denominator < q.numerator*scale*scale)
    return decimal_text(lo, digits), decimal_text(hi, digits)


def coordinate_file(path: Path) -> Path:
    if path.is_dir():
        if (path/'points.csv').exists():
            return path/'points.csv'
        if (path/'points.npy').exists():
            return path/'points.npy'
        if (path/'best').is_dir():
            return coordinate_file(path/'best')
        raise ValueError(f'No points.csv, points.npy or best/ in {path}')
    return path


def read_exact(path: Path) -> tuple[list[tuple[int, ...]], int]:
    """CSV decimals are exact; NumPy floats use their exact binary values."""
    path = coordinate_file(path)
    if path.suffix.lower() in ('.npy', '.npz'):
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            with loaded:
                key = next((s for s in ('points', 'x') if s in loaded), None)
                require(key is not None, 'NPZ needs a points or x array.')
                x = np.asarray(loaded[key], dtype=np.float64)
        else:
            x = np.asarray(loaded, dtype=np.float64)
        require(x.ndim == 2 and np.isfinite(x).all(), 'Expected a finite m-by-d array.')
        data = [[Fraction.from_float(float(v)) for v in row] for row in x]
    else:
        data = []
        for line in path.read_text(encoding='ascii').splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                data.append([Fraction(s.strip()) for s in
                             (line.split(',') if ',' in line else line.split())])
    require(bool(data) and len(data[0]) in (2, 3, 4, 5), 'Dimensions 2, 3, 4, 5 are supported.')
    d = len(data[0])
    require(all(len(r) == d for r in data), 'Inconsistent coordinate dimensions.')
    scale = math.lcm(*(q.denominator for r in data for q in r))
    decimal_scale(scale)
    rows = [tuple(int(q*scale) for q in r) for r in data]
    require(len(set(rows)) == len(rows), 'Input contains duplicate points; remove duplicates first.')
    require(len(rows) >= d+1, 'A full-dimensional hull needs at least d+1 points.')
    return rows, scale


def determinant(matrix: list[list[int]]) -> int:
    """Fraction-free Bareiss elimination with exact integer pivots."""
    n = len(matrix)
    if n == 0:
        return 1
    a = [list(r) for r in matrix]
    sign, previous = 1, 1
    for k in range(n-1):
        p = next((i for i in range(k, n) if a[i][k]), None)
        if p is None:
            return 0
        if p != k:
            a[p], a[k] = a[k], a[p]; sign = -sign
        pivot = a[k][k]
        for i in range(k+1, n):
            for j in range(k+1, n):
                a[i][j] = (a[i][j]*pivot-a[i][k]*a[k][j])//previous
            a[i][k] = 0
        previous = pivot
    return sign*a[-1][-1]


def supporting_plane(rows: list[tuple[int, ...]], face: tuple[int, ...]):
    """Return primitive outward (normal, offset), or None for an invalid simplex."""
    d = len(rows[0]); a = rows[face[0]]
    if len(set(face)) != d:
        return None
    differences = [[rows[i][j]-a[j] for j in range(d)] for i in face[1:]]
    normal = [(-1)**j*determinant([r[:j]+r[j+1:] for r in differences]) for j in range(d)]
    if not any(normal):
        return None
    offset = sum(normal[j]*a[j] for j in range(d))
    positive = negative = False
    for p in rows:
        value = sum(normal[j]*p[j] for j in range(d))-offset
        positive |= value > 0; negative |= value < 0
        if positive and negative:
            return None
    if not (positive or negative):
        return None  # lower-dimensional input
    if positive:
        normal = [-v for v in normal]; offset = -offset
    common = math.gcd(*(normal+[offset]))
    return tuple(v//common for v in normal), offset//common


def triangulate_plane3(rows, plane) -> list[tuple[int, int, int]]:
    normal, offset = plane
    indices = [i for i, r in enumerate(rows) if sum(a*b for a, b in zip(normal, r)) == offset]
    drop = max(range(3), key=lambda j: abs(normal[j]))
    axes = [j for j in range(3) if j != drop]
    pts = sorted((rows[i][axes[0]], rows[i][axes[1]], i) for i in indices)
    def turn(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    polygon = [p[2] for p in lower[:-1]+upper[:-1]]
    return [tuple(sorted((polygon[0], polygon[i], polygon[i+1]))) for i in range(1, len(polygon)-1)]


def normalized_rows(rows, diameter_squared=None) -> np.ndarray:
    """Normalize exact differences before float conversion; avoid large-scale overflow."""
    origin = rows[0]
    span = max(abs(a-b) for row in rows for a, b in zip(row, origin))
    require(span > 0, 'Zero-diameter input.')
    x = np.array([[float(Fraction(a-b, span)) for a, b in zip(row, origin)] for row in rows])
    if diameter_squared is None:
        diameter_squared = max(sqdist(a, b) for i, a in enumerate(rows) for b in rows[i+1:])
    x -= x.mean(axis=0)
    return x / math.sqrt(float(Fraction(diameter_squared, span*span)))


def find_facets(rows, scale: int, mode: str = 'auto', enumeration_limit: int = 200000):
    """Qhull is a proposal generator; every retained face passes exact signs.

    In 3D an exhaustive fallback repairs near-coplanar Qhull choices. In 4D/5D,
    omitted facets affect exploration only; no surface-cover theorem is claimed.
    """
    d, n = len(rows[0]), len(rows)
    x = normalized_rows(rows)
    if mode == 'exhaustive':
        require(d == 3, '--facets exhaustive currently supports dimension 3 only.')
        proposed = []
        invalid = 0
    else:
        try:
            proposed = sorted({tuple(sorted(map(int, f))) for f in ConvexHull(x).simplices})
        except QhullError:
            proposed = []
        invalid = 0
    valid, planes = [], set()
    for f in proposed:
        plane = supporting_plane(rows, f)
        if plane is None:
            invalid += 1
        else:
            valid.append(f); planes.add(plane)
    exhaustive = bool(mode == 'exhaustive' or (mode == 'auto' and d == 3 and (invalid or not valid)))
    if exhaustive:
        require(math.comb(n, 3) <= enumeration_limit,
                'Exact fallback exceeds --facet-enumeration-limit. Increase it, or use '
                '--facets verified (search only the valid proposed faces).')
        planes = set()
        for f in combinations(range(n), 3):
            plane = supporting_plane(rows, f)
            if plane is not None:
                planes.add(plane)
    if d == 3:
        valid = sorted({f for plane in planes for f in triangulate_plane3(rows, plane)})
    require(bool(valid), 'No exact supporting facets found; check full dimensionality of the input.')
    # Every face is revalidated, including triangulations of coplanar polygons.
    require(all(supporting_plane(rows, f) is not None for f in valid), 'Internal facet error.')
    return valid, {'dimension': d, 'qhull_simplices': len(proposed), 'rejected_simplices': invalid,
                   'valid_simplices': len(valid), 'exhaustive_3d_fallback': exhaustive,
                   'all_retained_faces_exactly_supporting': True,
                   'coverage_not_certified': not exhaustive,
                   'qualification': 'Only boundary membership is used for lower bounds; '
                                    'this search does not certify a cover of the hull.'}


def quantize_weights(weights: np.ndarray, denominator: int) -> tuple[int, ...]:
    weights = np.maximum(np.asarray(weights, float), 0)
    if weights.sum() == 0:
        weights[:] = 1
    weights = weights/weights.sum()*denominator
    q = np.floor(weights).astype(np.int64)
    left = denominator-int(q.sum())
    # Largest-remainder rounding preserves nonnegativity and sum exactly.
    if left:
        q[np.argsort(-(weights-q), kind='stable')[:left]] += 1
    require(int(q.sum()) == denominator, 'Weight quantization failed.')
    return tuple(map(int, q))


@dataclass(frozen=True)
class SurfacePoint:
    vertices: tuple[int, ...]
    weights: tuple[int, ...]

    def as_json(self) -> dict:
        return {'vertices': list(self.vertices), 'weights': list(self.weights)}

    @staticmethod
    def from_json(data):
        return SurfacePoint(tuple(map(int, data['vertices'])), tuple(map(int, data['weights'])))


class Cloud:
    def __init__(self, base: list[tuple[int, ...]], scale: int, denominator: int,
                 facets: list[tuple[int, ...]], specs: list[SurfacePoint] | None = None):
        self.base, self.base_scale, self.Q, self.facets = base, scale, denominator, facets
        self.nbase, self.d = len(base), len(base[0])
        self.scale = scale*denominator
        self.Dbase = max(sqdist(a, b) for i, a in enumerate(base) for b in base[i+1:])
        self.D = self.Dbase*denominator**2
        self.base_float = normalized_rows(base, self.Dbase)
        self.specs: list[SurfacePoint] = []
        self.rows: list[tuple[int, ...]] = []
        self.seen: set[tuple[int, ...]] = set()
        self.pairs: list[tuple[int, int]] = []
        self.q: list[int] = []
        initial = specs or [SurfacePoint((i,), (denominator,)) for i in range(self.nbase)]
        for s in initial:
            require(self.append(s), 'Duplicate point in saved surface state.')
        require(len(self.rows) >= self.nbase, 'Missing original points.')
        require(self.rows[:self.nbase] == [tuple(denominator*v for v in p) for p in base],
                'Original coordinates were changed or reordered.')
        require(max(self.q) == self.D, 'Diameter changed; invalid surface construction.')

    def point_row(self, spec: SurfacePoint) -> tuple[int, ...]:
        require(len(spec.vertices) == len(spec.weights) and len(spec.weights) > 0,
                'Invalid barycentric specification.')
        require(all(0 <= i < self.nbase for i in spec.vertices), 'Bad source vertex index.')
        require(all(w >= 0 for w in spec.weights) and sum(spec.weights) == self.Q,
                'Weights must be nonnegative and sum to their denominator.')
        return tuple(sum(w*self.base[i][j] for i, w in zip(spec.vertices, spec.weights))
                     for j in range(self.d))

    def point_float(self, spec: SurfacePoint) -> np.ndarray:
        return np.asarray(spec.weights, float) @ self.base_float[list(spec.vertices)]/self.Q

    def array(self) -> np.ndarray:
        return np.array([self.point_float(s) for s in self.specs])

    def append(self, spec: SurfacePoint) -> bool:
        row = self.point_row(spec)
        if row in self.seen:
            return False
        j = len(self.rows)
        for i, other in enumerate(self.rows):
            distance = sqdist(row, other)
            require(distance <= self.D, 'New point enlarges the original diameter.')
            self.pairs.append((i, j)); self.q.append(distance)
        self.specs.append(spec); self.rows.append(row); self.seen.add(row)
        return True

    def copy_with(self, specs):
        return Cloud(self.base, self.base_scale, self.Q, self.facets, specs)


@dataclass
class Bounds:
    lower: int
    upper: int
    witness: list[tuple[int, int]]
    colors: list[int] | None
    unknown: bool = False

    def payload(self, cloud: Cloud) -> dict:
        return {'points': len(cloud.rows), 'lower': sqrt_bounds(Fraction(self.lower, cloud.D))[0],
                'upper': sqrt_bounds(Fraction(self.upper, cloud.D))[1],
                'squared_bounds_equal': self.lower == self.upper, 'unknown': self.unknown}


class Session:
    """One capacity-sized oracle, reused across all additions and thresholds."""
    def __init__(self, capacity: int, k: int, backend='auto', solver='kissat',
                 timeout=10., pysat_solver='g4'):
        self.capacity, self.k = capacity, k
        self.oracle = bs.GraphOracle(capacity, k, backend, solver, timeout, pysat_solver)

    def close(self):
        self.oracle.close()

    def ids(self, edges):
        return np.array([u*(2*self.capacity-u-1)//2+v-u-1 for u, v in edges], dtype=np.int32)

    def solve_edges(self, edges, n: int):
        require(n <= self.capacity, 'Point capacity exceeded.')
        ans = self.oracle.solve(self.ids(edges))
        colors = None if ans.colors is None else list(map(int, ans.colors[:n]))
        witness = list(edges)
        if ans.status is bs.Status.UNSAT and ans.core is not None and len(ans.core):
            witness = [tuple(map(int, e)) for e in self.oracle.pairs[ans.core]]
        return ans.status, colors, witness

    def threshold(self, cloud: Cloud, threshold_squared: Fraction):
        a, b = threshold_squared.numerator, threshold_squared.denominator
        edges = [e for e, q in zip(cloud.pairs, cloud.q) if q*b > cloud.D*a]
        status, colors, witness = self.solve_edges(edges, len(cloud.rows))
        return status, colors, witness, edges


def upper_from_colors(cloud: Cloud, colors: list[int] | None) -> int:
    if colors is None:
        return cloud.D
    require(len(colors) == len(cloud.rows), 'Wrong coloring length.')
    return max((q for (u, v), q in zip(cloud.pairs, cloud.q) if colors[u] == colors[v]), default=0)


def lower_from_witness(cloud: Cloud, edges) -> int:
    return min((sqdist(cloud.rows[u], cloud.rows[v]) for u, v in edges), default=0)


def evaluate_exact(cloud: Cloud, session: Session, previous: Bounds | None = None,
                   colors: list[int] | None = None, calls: int = 30) -> Bounds:
    """Exact integer-distance bisection; UNKNOWN retains a valid bracket."""
    n, k = len(cloud.rows), session.k
    if n <= k:
        return Bounds(0, 0, [], list(range(n)))
    witness = (previous.witness if previous and previous.witness
               else list(combinations(range(k+1), 2)))  # K_{k+1} base case
    lower = lower_from_witness(cloud, witness)
    if colors is None and previous and previous.colors is not None and len(previous.colors) == n:
        colors = previous.colors
    upper = upper_from_colors(cloud, colors)
    require(lower <= upper, 'Inconsistent witness and coloring.')
    distances = sorted(set(cloud.q))
    unknown = False
    for _ in range(calls):
        if lower == upper:
            break
        left, right = bisect.bisect_left(distances, lower), bisect.bisect_left(distances, upper)
        require(left < right, 'Invalid exact evaluation bracket.')
        threshold = distances[(left+right-1)//2]
        edges = [e for e, q in zip(cloud.pairs, cloud.q) if q > threshold]
        status, new_colors, w = session.solve_edges(edges, n)
        if status is bs.Status.UNSAT:
            qmin = lower_from_witness(cloud, w)
            require(qmin > threshold, 'Invalid UNSAT core.')
            lower, witness = qmin, w
        elif status is bs.Status.SAT and new_colors is not None:
            colors = new_colors
            upper = upper_from_colors(cloud, colors)
            require(upper <= threshold, 'Invalid SAT coloring.')
        else:
            unknown = True
            break
    require(lower <= upper, 'Invalid evaluation result.')
    return Bounds(lower, upper, witness, colors, unknown)


def target_for(bounds: Bounds, cloud: Cloud, gain: Fraction, fixed: Fraction | None) -> Fraction:
    if fixed is not None:
        return fixed*fixed
    if gain == 0:
        return Fraction(bounds.upper, cloud.D)
    # Rounded upward: a user-specified positive gain is never silently rounded down.
    upper = Fraction(sqrt_bounds(Fraction(bounds.upper, cloud.D), 18)[1])
    return min(Fraction(1), (upper+gain)**2)


def canonical_colors(colors) -> tuple[int, ...]:
    labels = {}
    return tuple(labels.setdefault(int(c), len(labels)) for c in colors)


def coloring_bank(colors: list[int], edges, k: int, count: int,
                  rng: np.random.Generator) -> list[list[int]]:
    """Cheap alternative valid colorings by single-vertex and Kempe moves."""
    n = len(colors)
    adjacency = [set() for _ in range(n)]
    for u, v in edges:
        adjacency[u].add(v); adjacency[v].add(u)
    bank, seen = [list(colors)], {canonical_colors(colors)}
    for _ in range(max(20, count*15)):
        c = list(bank[int(rng.integers(len(bank)))])
        for _ in range(4):
            v = int(rng.integers(n))
            available = [a for a in range(k) if all(c[u] != a for u in adjacency[v])]
            if available and rng.random() < .5:
                c[v] = int(rng.choice(available))
            else:
                a = c[v]; b = int(rng.integers(k))
                component, queue = {v}, [v]
                for u in queue:
                    for w in adjacency[u]:
                        if w not in component and c[w] in (a, b):
                            component.add(w); queue.append(w)
                for u in component:
                    c[u] = b if c[u] == a else a
        key = canonical_colors(c)
        if key not in seen:
            seen.add(key); bank.append(c)
        if len(bank) >= count:
            break
    return bank[:count]


def candidate_pool(cloud: Cloud, size: int, rng: np.random.Generator,
                   hot: list[tuple[int, ...]]) -> list[SurfacePoint]:
    facets = cloud.facets
    d = cloud.d
    # Half uniform by face, half approximately area weighted; hot faces get a
    # modest extra share, never all of the sampling probability.
    areas = []
    for f in facets:
        z = cloud.base_float[list(f)]
        gram = (z[1:]-z[0]) @ (z[1:]-z[0]).T
        areas.append(math.sqrt(max(0., float(np.linalg.det(gram)))))
    area = np.asarray(areas)
    probabilities = .5*np.full(len(facets), 1/len(facets)) + .5*area/max(area.sum(), 1e-300)
    if hot:
        counts = {f: hot.count(f) for f in set(hot)}
        h = np.array([counts.get(f, 0) for f in facets], float)
        if h.sum():
            probabilities = .75*probabilities+.25*h/h.sum()
    probabilities /= probabilities.sum()
    specs = []
    # Deterministic sites complement random samples; exact duplicates are removed.
    for f in facets:
        specs.append(SurfacePoint(f, quantize_weights(np.ones(d), cloud.Q)))
        for a, b in combinations(range(d), 2):
            w = np.zeros(d); w[a] = w[b] = .5
            specs.append(SurfacePoint(f, quantize_weights(w, cloud.Q)))
    for j in rng.choice(len(facets), size=size, p=probabilities):
        f = facets[int(j)]
        if rng.random() < .4:
            a, b = rng.choice(d, size=2, replace=False)
            w = np.zeros(d); w[a] = rng.random(); w[b] = 1-w[a]
        else:
            w = rng.exponential(size=d)
        specs.append(SurfacePoint(f, quantize_weights(w, cloud.Q)))
    return unique_candidates(cloud, specs)


def unique_candidates(cloud: Cloud, specs) -> list[SurfacePoint]:
    seen = set(cloud.seen)
    result = []
    for spec in specs:
        row = cloud.point_row(spec)
        if row not in seen:
            seen.add(row); result.append(spec)
    return result


def farthest_by_color(points: np.ndarray, x: np.ndarray, colors: list[int], k: int):
    q = np.sum((points[:, None, :]-x[None, :, :])**2, axis=2)
    values = np.full((len(points), k), -1.)
    colors = np.asarray(colors)
    for c in range(k):
        if np.any(colors == c):
            values[:, c] = q[:, colors == c].max(axis=1)
    return values


def refine_candidate(cloud: Cloud, spec: SurfacePoint, x: np.ndarray,
                     colors: list[int], k: int, iterations: int) -> SurfacePoint:
    """SLSQP proposes a face point maximizing min_c max_{x in color c} |p-x|^2.

    This piecewise-smooth optimization is not used as a certificate.
    """
    vertices = cloud.base_float[list(spec.vertices)]
    groups = [x[np.asarray(colors) == c] for c in range(k)]
    if any(len(g) == 0 for g in groups):
        return spec
    d = len(vertices)
    weights = np.asarray(spec.weights, float)/cloud.Q
    def fg(z):
        point = z[:d] @ vertices
        values, jac = [], np.zeros((k, d+1))
        for c, group in enumerate(groups):
            delta = point-group
            j = int(np.argmax(np.sum(delta*delta, axis=1)))
            values.append(float(delta[j] @ delta[j])-z[-1])
            jac[c, :d] = 2*vertices @ delta[j]; jac[c, -1] = -1
        return np.array(values), jac
    value = float(farthest_by_color(np.array([weights @ vertices]), x, colors, k).min())
    result = minimize(lambda z: -z[-1], np.r_[weights, value],
                      jac=lambda z: np.r_[np.zeros(d), -1.], method='SLSQP',
                      bounds=[(0., 1.)]*d+[(0., 1.)],
                      constraints=[{'type': 'eq', 'fun': lambda z: z[:d].sum()-1,
                                    'jac': lambda z: np.r_[np.ones(d), 0.]},
                                   {'type': 'ineq', 'fun': lambda z: fg(z)[0],
                                    'jac': lambda z: fg(z)[1]}],
                      options={'maxiter': iterations, 'ftol': 1e-11})
    if not np.isfinite(result.x).all():
        return spec
    new = SurfacePoint(spec.vertices, quantize_weights(result.x[:d], cloud.Q))
    new_score = float(farthest_by_color(np.array([cloud.point_float(new)]), x, colors, k).min())
    return new if new_score > value else spec


def list_colorable(adjacency: list[set[int]], domains: list[int], active: list[int],
                   timeout: float = .05, node_limit: int = 30000) -> bool | None:
    """Small list-coloring CSP used ONLY to choose batches for a fixed old coloring."""
    if any(domains[v] == 0 for v in active):
        return False
    remaining = set(active)
    masks = domains.copy()
    start = time.monotonic(); nodes = 0
    def visit():
        nonlocal nodes
        nodes += 1
        if nodes > node_limit or (nodes % 64 == 0 and time.monotonic()-start > timeout):
            return None
        if not remaining:
            return True
        v = min(remaining, key=lambda u: (masks[u].bit_count(), -len(adjacency[u] & remaining)))
        options = masks[v]
        remaining.remove(v)
        while options:
            bit = options & -options; options -= bit
            changed = []; feasible = True
            for u in adjacency[v] & remaining:
                if masks[u] & bit:
                    changed.append((u, masks[u])); masks[u] ^= bit
                    if not masks[u]:
                        feasible = False; break
            result = visit() if feasible else False
            for u, old in changed:
                masks[u] = old
            if result is not False:
                remaining.add(v)
                return result
        remaining.add(v)
        return False
    return visit()


def blocking_core(adjacency, domains, checks: int, timeout: float):
    active = list(range(len(domains)))
    if list_colorable(adjacency, domains, active, timeout) is not False:
        return None, 1
    used = 1
    size = max(1, len(active)//2)
    while size and used < checks:
        i = 0
        while i < len(active) and used < checks:
            trial = active[:i]+active[i+size:]
            used += 1
            if list_colorable(adjacency, domains, trial, timeout) is False:
                active = trial
            else:
                i += size
        size //= 2
    return active, used


def propose_batch(cloud: Cloud, bank: list[list[int]], target: Fraction, args,
                  rng: np.random.Generator, hot: list[tuple[int, ...]]):
    x = cloud.array(); t2 = float(target)
    candidates = candidate_pool(cloud, args.pool, rng, hot)
    if not candidates:
        return [], {'pool': 0}
    def rank(specs):
        p = np.array([cloud.point_float(s) for s in specs])
        # Chunking keeps candidate-to-old-point arrays bounded in size.
        scores = np.empty((len(bank), len(p)))
        available = np.empty((len(bank), len(p)), dtype=np.int16)
        for b, colors in enumerate(bank):
            for start in range(0, len(p), 256):
                far = farthest_by_color(p[start:start+256], x, colors, args.k)
                scores[b, start:start+256] = far.min(axis=1)-t2
                available[b, start:start+256] = (far <= t2).sum(axis=1)
        priority = (available == 0).sum(axis=0)*10. - available.min(axis=0)
        priority += .25*np.clip(scores.max(axis=0), -1, 1)
        return scores, available, priority
    scores, available, priority = rank(candidates)
    refined = []
    for j, i in enumerate(np.argsort(-priority, kind='stable')[:args.refine]):
        b = int(np.argmax(scores[:, i])) if j % 2 else j % len(bank)
        refined.append(refine_candidate(cloud, candidates[int(i)], x, bank[b], args.k, args.refine_iters))
    if refined:
        candidates = unique_candidates(cloud, candidates+refined)
        scores, available, priority = rank(candidates)
    ordered = list(map(int, np.argsort(-priority, kind='stable')))
    # Reserve a quarter of the block pool for geometric/global diversity.
    n = min(args.block_pool, len(candidates))
    top = ordered[:max(1, 3*n//4)]
    rest = [i for i in rng.permutation(len(candidates)) if i not in set(top)]
    pool_ids = top+rest[:n-len(top)]
    pool = [candidates[i] for i in pool_ids]
    rows = [cloud.point_row(s) for s in pool]
    a, b = target.numerator, target.denominator
    conflict = [[j for j, r in enumerate(cloud.rows) if sqdist(p, r)*b > cloud.D*a] for p in rows]
    domains = []
    for coloring in bank:
        domains.append([((1 << args.k)-1) & ~sum(1 << c for c in {coloring[j] for j in neighbors})
                        for neighbors in conflict])
    adjacency = [set() for _ in pool]
    for i, j in combinations(range(len(pool)), 2):
        if sqdist(rows[i], rows[j])*b > cloud.D*a:
            adjacency[i].add(j); adjacency[j].add(i)
    selected = []
    blocked = set()
    # Prefer one point that eliminates many known colorings.
    while len(selected) < args.batch:
        choices = [(sum(domains[j][i] == 0 for j in range(len(bank)) if j not in blocked), -i, i)
                   for i in range(len(pool)) if i not in selected]
        if not choices or max(choices)[0] == 0:
            break
        _, _, i = max(choices); selected.append(i)
        blocked.update(j for j in range(len(bank)) if domains[j][i] == 0)
    cooperative, calls = [], 0
    for j in range(len(bank)):
        if j in blocked or len(selected) >= args.batch:
            continue
        core, used = blocking_core(adjacency, domains[j], args.block_checks, args.block_timeout)
        calls += used
        if core is not None:
            extra = [i for i in core if i not in selected]
            if len(selected)+len(extra) <= args.batch:
                selected += extra; blocked.add(j)
                cooperative.append(len(core))
    # Accumulate informative points even when none blocks the present coloring.
    # Do not require a single insertion to improve the record or reject twins.
    for i in range(len(pool)):
        if len(selected) >= args.batch:
            break
        if i not in selected:
            selected.append(i)
    return [pool[i] for i in selected], {
        'pool': len(candidates), 'colorings': len(bank), 'block_pool': len(pool),
        'known_colorings_blocked': len(blocked), 'cooperative_core_sizes': cooperative,
        'list_csp_calls': calls, 'maximum_sampled_blocking_margin_squared': float(scores.max()),
        'note': 'Blocking one old coloring is a proposal screen, NOT a noncolorability certificate.'}


def polish_added(cloud: Cloud, bounds: Bounds, iterations: int):
    """Optimize ONLY added points on their fixed faces. No diameter constraints needed."""
    if not iterations or len(cloud.rows) == cloud.nbase or not bounds.witness:
        return cloud, {'attempted': False}
    movable = list(range(cloud.nbase, len(cloud.rows)))
    sizes = [len(cloud.specs[i].weights) for i in movable]
    offsets = np.r_[0, np.cumsum(sizes)]
    count = int(offsets[-1])
    matrices = [cloud.base_float[list(cloud.specs[i].vertices)] for i in movable]
    start = np.concatenate([np.asarray(cloud.specs[i].weights, float)/cloud.Q for i in movable])
    edges = np.asarray(bounds.witness, dtype=np.int32)
    x0 = cloud.array()
    cached_z = None; cached = None
    def unpack(z):
        x = x0.copy()
        for j, i in enumerate(movable):
            x[i] = z[offsets[j]:offsets[j+1]] @ matrices[j]
        return x
    def fg(z):
        nonlocal cached_z, cached
        if cached_z is not None and np.array_equal(z, cached_z):
            return cached
        x = unpack(z); delta = x[edges[:, 0]]-x[edges[:, 1]]
        values = np.sum(delta*delta, axis=1)-z[-1]
        jac = np.zeros((len(edges), count+1)); jac[:, -1] = -1
        for j, i in enumerate(movable):
            left = np.flatnonzero(edges[:, 0] == i)
            right = np.flatnonzero(edges[:, 1] == i)
            jac[left, offsets[j]:offsets[j+1]] = 2*delta[left] @ matrices[j].T
            jac[right, offsets[j]:offsets[j+1]] = -2*delta[right] @ matrices[j].T
        cached_z = z.copy(); cached = values, jac
        return cached
    eqjac = np.zeros((len(movable), count+1))
    for j in range(len(movable)):
        eqjac[j, offsets[j]:offsets[j+1]] = 1
    z0 = np.r_[start, bounds.lower/cloud.D]
    best_z, best = z0.copy(), bounds.lower/cloud.D
    def keep(z):
        nonlocal best_z, best
        zz = z.copy()
        for j in range(len(movable)):
            a, b = offsets[j], offsets[j+1]
            w = np.maximum(zz[a:b], 0)
            zz[a:b] = w/w.sum() if w.sum() else 1/len(w)
        x = unpack(zz)
        q = bs.distances2(x, edges).min()
        if q > best:
            best_z, best = zz.copy(), float(q)
    result = minimize(lambda z: -z[-1], z0, jac=lambda z: np.r_[np.zeros(count), -1.],
                      method='SLSQP', bounds=[(0., 1.)]*(count+1),
                      constraints=[{'type': 'eq', 'fun': lambda z: eqjac @ z-1,
                                    'jac': lambda z: eqjac},
                                   {'type': 'ineq', 'fun': lambda z: fg(z)[0],
                                    'jac': lambda z: fg(z)[1]}], callback=keep,
                      options={'maxiter': iterations, 'ftol': 1e-12})
    if np.isfinite(result.x).all():
        keep(result.x)
    specs = list(cloud.specs)
    for j, i in enumerate(movable):
        specs[i] = SurfacePoint(specs[i].vertices,
                                quantize_weights(best_z[offsets[j]:offsets[j+1]], cloud.Q))
    report = {'attempted': True, 'slsqp_success': bool(result.success), 'iterations': int(result.nit),
              'message': str(result.message), 'accepted': False,
              'qualification': 'Proposal optimization, not a stationarity or optimality certificate.'}
    try:
        trial = cloud.copy_with(specs)
    except ValueError:  # coincident points after quantization: retain the old cloud
        return cloud, report
    if lower_from_witness(trial, bounds.witness) > bounds.upper:
        report['accepted'] = True
        return trial, report
    return cloud, report


def save_snapshot(directory: Path, cloud: Cloud, bounds: Bounds, k: int,
                  baseline_upper: Fraction, metadata: dict) -> None:
    """Publish a complete checkpoint by directory rename; CSV is authoritative."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    temp = directory.with_name(directory.name+f'.tmp-{os.getpid()}')
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    precision = write_rows(temp/'points.csv', cloud.rows, cloud.scale)
    write_rows(temp/'source_points.csv', cloud.base, cloud.base_scale)
    np.save(temp/'points.npy', np.array([[float(Fraction(a, cloud.scale)) for a in row]
                                      for row in cloud.rows], dtype=np.float64))
    edges = np.asarray(bounds.witness, dtype=np.int32).reshape(-1, 2)
    np.savetxt(temp/'witness_edges.csv', edges, fmt='%d', delimiter=',')
    bs.write_cnf(temp/'witness.cnf', len(cloud.rows)*k,
                 bs.graph_cnf(len(cloud.rows), k, edges))
    (temp/'partition.json').write_text(json.dumps(bounds.colors)+'\n')
    multiplier = 10**precision//cloud.scale
    rows = [[v*multiplier for v in r] for r in cloud.rows]
    geometry = bs.rational_geometry(rows, edges, bounds.colors)
    certificate = {'version': 'surface-'+VERSION, 'dimension': cloud.d, 'm': len(cloud.rows),
                   'k': k, 'floating_lower': math.sqrt(bounds.lower/cloud.D),
                   'floating_upper': math.sqrt(bounds.upper/cloud.D),
                   'threshold_resolved': bounds.lower == bounds.upper,
                   'coordinate_decimal_digits': precision, 'witness_edges': len(edges),
                   'duplicate_rows_separated_by_1e_minus_17': 0, 'geometry': geometry,
                   'run': metadata, 'qualification': 'Exact geometry; noncolorability relies on '
                   'a solver UNSAT result or a complete-graph base case, not an independent proof checker. '
                   'The upper bound applies to these finite points, NOT to the entire hull.'}
    (temp/'certificate.json').write_text(json.dumps(certificate, indent=2)+'\n')
    surface = {'version': VERSION, 'k': k, 'dimension': cloud.d, 'source_points': cloud.nbase,
               'weight_denominator': cloud.Q, 'facets': cloud.facets,
               'barycentric_points': [s.as_json() for s in cloud.specs],
               'initial_upper_squared': [str(baseline_upper.numerator), str(baseline_upper.denominator)],
               'strict_improvement_over_initial': Fraction(bounds.lower, cloud.D) > baseline_upper,
               'canonical_coordinates': 'points.csv; points.npy is a rounded convenience copy',
               'metadata': metadata}
    (temp/'surface.json').write_text(json.dumps(surface, indent=2)+'\n')
    old = directory.with_name(directory.name+'.previous')
    if old.exists():
        shutil.rmtree(old)
    if directory.exists():
        directory.rename(old)
    temp.rename(directory)
    if old.exists():
        shutil.rmtree(old)


def load_start(path: Path, bits: int, facet_mode: str, enumeration_limit: int):
    file = coordinate_file(path)
    folder = file.parent
    if (folder/'surface.json').exists():
        data = json.loads((folder/'surface.json').read_text())
        base, scale = read_exact(folder/'source_points.csv')
        require(int(data['source_points']) == len(base) and int(data['dimension']) == len(base[0]),
                'Saved source dimensions disagree with source_points.csv.')
        require(int(data['weight_denominator']) > 0, 'Invalid weight denominator.')
        facets = [tuple(map(int, f)) for f in data['facets']]
        require(all(supporting_plane(base, f) is not None for f in facets),
                'Saved surface has an invalid supporting face.')
        cloud = Cloud(base, scale, int(data['weight_denominator']), facets,
                      [SurfacePoint.from_json(s) for s in data['barycentric_points']])
        saved, saved_scale = read_exact(file)
        require(len(saved) == len(cloud.rows) and
                all(all(a*cloud.scale == b*saved_scale for a, b in zip(p, q))
                    for p, q in zip(saved, cloud.rows)), 'Saved coordinates disagree with barycentric data.')
        for s in cloud.specs[cloud.nbase:]:
            require(tuple(s.vertices) in set(facets), 'Added point uses an unlisted face.')
        return cloud, data, {'resumed_surface_state': True, 'valid_simplices': len(facets)}
    base, scale = read_exact(file)
    facets, report = find_facets(base, scale, facet_mode, enumeration_limit)
    return Cloud(base, scale, 2**bits, facets), None, report


def read_partition(path: Path, n: int, k: int):
    folder = coordinate_file(path).parent
    p = folder/'partition.json'
    if p.is_file():
        colors = json.loads(p.read_text())
        if (isinstance(colors, list) and len(colors) == n and
                all(isinstance(c, int) and 0 <= c < k for c in colors)):
            return colors
    return None


def extend_partition(cloud: Cloud, colors: list[int] | None, k: int) -> list[int]:
    """Any returned partition supplies an upper bound, even if it exceeds the target."""
    c = list(colors) if colors is not None else []
    class_max = [0]*k
    for (i, j), q in zip(cloud.pairs, cloud.q):
        if j < len(c) and c[i] == c[j]:
            class_max[c[i]] = max(class_max[c[i]], q)
    for j in range(len(c), len(cloud.rows)):
        far = [0]*k
        for i in range(j):
            far[c[i]] = max(far[c[i]], sqdist(cloud.rows[i], cloud.rows[j]))
        chosen = min(range(k), key=lambda a: max(class_max[a], far[a]))
        c.append(chosen); class_max[chosen] = max(class_max[chosen], far[chosen])
    return c


def verify_surface(folder: Path, args) -> dict:
    folder = coordinate_file(folder).parent
    cloud, state, report = load_start(folder, 24, 'verified', 0)
    require(state is not None, 'This is not a surface_search checkpoint.')
    require(len(cloud.specs) == len(set(cloud.rows)), 'Coincident points.')
    result = bs.verify_candidate(folder, args.backend, args.solver, args.sat_timeout, args.pysat_solver)
    require(int(state['k']) == int(result['k']), 'Surface and finite-set certificates disagree on k.')
    result.update({'exact_boundary_membership_of_all_added_points': True,
                   'original_points_unchanged': True, 'original_hull_and_diameter_unchanged': True,
                   'source_points': cloud.nbase, 'added_points': len(cloud.rows)-cloud.nbase,
                   'qualification': 'The upper bound is for the finite set, not for the whole hull. '
                                    'UNSAT is solver checked, not DRAT/LRAT independently verified.'})
    q = Fraction(int(result['min_witness_squared_integer']), int(result['diameter_squared_integer']))
    baseline = Fraction(*map(int, state['initial_upper_squared']))
    require(bool(state['strict_improvement_over_initial']) == (q > baseline),
            'Incorrect improvement flag.')
    result['strict_improvement_over_initial'] = q > baseline
    return result


def run_search(args) -> dict:
    start_time = time.monotonic()
    require(not args.out.exists(), 'Output directory already exists. Use a new --out directory; '
                                  'resume by giving the previous latest/ as input.')
    cloud, old_state, hull_report = load_start(args.input, args.weight_bits, args.facets,
                                               args.facet_enumeration_limit)
    args.k = args.k if args.k is not None else (old_state['k'] if old_state else cloud.d+1)
    require(old_state is None or args.k == old_state['k'],
            'A resumed surface state must use the same k. For a different k, copy its coordinates '
            'to a fresh input file without the surface sidecar.')
    require(args.k >= 1, 'k must be positive.')
    if args.max_points is None:
        args.max_points = max(256, len(cloud.rows)+args.rounds*args.batch)
    require(args.max_points >= len(cloud.rows), '--max-points is smaller than the input.')
    args.out.mkdir(parents=True)
    events_path = args.out/'progress.jsonl'
    def log(event: str, **data):
        payload = {'event': event, 'elapsed_seconds': round(time.monotonic()-start_time, 3), **data}
        with events_path.open('a') as f:
            f.write(json.dumps(payload, default=str)+'\n')
        print(json.dumps(payload, default=str), flush=True)
    (args.out/'settings.json').write_text(json.dumps(vars(args), indent=2, default=str)+'\n')
    (args.out/'hull.json').write_text(json.dumps(hull_report, indent=2)+'\n')
    rng = np.random.default_rng(args.seed)
    session = Session(args.max_points, args.k, args.backend, args.solver, args.sat_timeout, args.pysat_solver)
    completed, stop, hot = 0, 'round_budget_exhausted', []
    try:
        colors = read_partition(args.input, len(cloud.rows), args.k)
        bounds = evaluate_exact(cloud, session, colors=colors, calls=args.eval_checks)
        baseline = (Fraction(*map(int, old_state['initial_upper_squared'])) if old_state
                    else Fraction(bounds.upper, cloud.D))
        best_lower, best_upper = bounds.lower, bounds.upper
        save_snapshot(args.out/'best', cloud, bounds, args.k, baseline, {'stage': 'initial'})
        save_snapshot(args.out/'latest', cloud, bounds, args.k, baseline, {'stage': 'initial'})
        log('initial', backend=session.oracle.backend, **bounds.payload(cloud), hull=hull_report)
        target = target_for(bounds, cloud, args.gain, args.target)
        for round_no in range(1, args.rounds+1):
            if len(cloud.rows) >= args.max_points:
                stop = 'point_budget_exhausted'; break
            if target >= 1:
                stop = 'target_at_diameter'; break
            # A SAT model at this target is needed to generate coloring-directed proposals.
            status, colors, w, edges = session.threshold(cloud, target)
            if status is bs.Status.UNSAT:
                qmin = lower_from_witness(cloud, w)
                if qmin > bounds.lower:
                    bounds = Bounds(qmin, bounds.upper, w, bounds.colors)
                bounds = evaluate_exact(cloud, session, previous=bounds, calls=args.eval_checks)
                if args.target is not None:
                    stop = 'requested_target_exceeded'; break
                target = target_for(bounds, cloud, args.gain, None)
                status, colors, w, edges = session.threshold(cloud, target)
            if status is not bs.Status.SAT or colors is None:
                stop = 'threshold_unresolved'; log('unresolved', round=round_no, status=status.value); break
            bank = coloring_bank(colors, edges, args.k, args.colorings, rng)
            log('round_start', round=round_no, target=sqrt_bounds(target)[1],
                points=len(cloud.rows), colorings=len(bank))
            proposals, info = propose_batch(cloud, bank, target, args, rng, hot)
            proposals = proposals[:args.max_points-len(cloud.rows)]
            old_n = len(cloud.rows)
            for p in proposals:
                if cloud.append(p):
                    hot.append(p.vertices)
            hot = hot[-32:]
            if len(cloud.rows) == old_n:
                stop = 'candidate_pool_exhausted'; break
            status, new_colors, witness, _ = session.threshold(cloud, target)
            fallback_colors = extend_partition(cloud, colors, args.k)
            if new_colors is not None:
                fallback_colors = new_colors
            upper = upper_from_colors(cloud, fallback_colors)
            if status is bs.Status.UNSAT:
                lower = lower_from_witness(cloud, witness)
                if lower > bounds.lower:
                    bounds = Bounds(lower, upper, witness, fallback_colors)
                else:
                    bounds = Bounds(bounds.lower, upper, bounds.witness, fallback_colors)
            else:
                bounds = Bounds(bounds.lower, upper, bounds.witness, fallback_colors,
                                status is bs.Status.UNKNOWN)
            if (round_no % args.evaluate_every == 0 or status is not bs.Status.SAT):
                bounds = evaluate_exact(cloud, session, previous=bounds, calls=args.eval_checks)
            polish = {'attempted': False}
            if bounds.lower > best_lower and args.polish_iters:
                cloud, polish = polish_added(cloud, bounds, args.polish_iters)
                if polish.get('accepted'):
                    bounds = evaluate_exact(cloud, session, previous=bounds, calls=args.eval_checks)
            completed = round_no
            metadata = {'round': round_no, 'target_squared': [str(target.numerator), str(target.denominator)],
                        'generation': info, 'polish': polish, 'threshold_status': status.value}
            save_snapshot(args.out/'latest', cloud, bounds, args.k, baseline, metadata)
            if bounds.lower > best_lower:
                best_lower, best_upper = bounds.lower, bounds.upper
                save_snapshot(args.out/'best', cloud, bounds, args.k, baseline, metadata)
                save_snapshot(args.out/'history'/f'round_{round_no:04d}', cloud, bounds, args.k, baseline, metadata)
                log('improvement', round=round_no, strict_over_initial=Fraction(bounds.lower, cloud.D) > baseline,
                    **bounds.payload(cloud))
            log('round_end', round=round_no, added=len(cloud.rows)-old_n, threshold_status=status.value,
                **bounds.payload(cloud), proposal=info, polish=polish)
            if status is bs.Status.UNKNOWN:
                stop = 'threshold_unresolved'; break
            if status is bs.Status.UNSAT:
                if args.target is not None:
                    stop = 'requested_target_exceeded'; break
                target = target_for(bounds, cloud, args.gain, None)
        # Tighten the last checkpoint, even when the target was not crossed.
        bounds = evaluate_exact(cloud, session, previous=bounds, calls=args.eval_checks)
        save_snapshot(args.out/'latest', cloud, bounds, args.k, baseline, {'stage': 'final', 'stop': stop})
        if bounds.lower > best_lower:
            best_lower, best_upper = bounds.lower, bounds.upper
            save_snapshot(args.out/'best', cloud, bounds, args.k, baseline, {'stage': 'final', 'stop': stop})
        summary = {'version': VERSION, 'stop': stop, 'rounds_completed': completed,
                   'latest_points': len(cloud.rows), 'original_points': cloud.nbase,
                   'latest': bounds.payload(cloud),
                   'best_lower': sqrt_bounds(Fraction(best_lower, cloud.D))[0],
                   'best_upper': sqrt_bounds(Fraction(best_upper, cloud.D))[1],
                   'initial_upper': sqrt_bounds(baseline)[1],
                   'strict_improvement_over_initial': Fraction(best_lower, cloud.D) > baseline,
                   'backend': session.oracle.backend, 'solver_calls': session.oracle.calls,
                   'unknown_calls': session.oracle.unknowns, 'elapsed_seconds': time.monotonic()-start_time,
                   'qualification': 'Lower bounds apply to the hull; upper bounds apply ONLY to finite samples. '
                                    'Failure to improve is not an equality or optimality proof.'}
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        return summary
    finally:
        session.close()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('search', help='Add boundary points; preserve the original hull exactly.')
    s.add_argument('input', type=Path, help='Coordinates, saved best directory, or previous latest/ to continue.')
    s.add_argument('--out', type=Path, required=True, help='New output directory.')
    s.add_argument('--k', type=int, default=None, help='Number of parts; default dimension + 1.')
    s.add_argument('--rounds', type=int, default=40)
    s.add_argument('--batch', type=int, default=6, help='Points accumulated per round, including neutral additions.')
    s.add_argument('--max-points', type=int, help='Default max(256, starting size + rounds*batch).')
    s.add_argument('--gain', type=Fraction, default=Fraction('0.000001'),
                   help='Absolute ratio step above the current upper bound; 0 seeks any strict improvement.')
    s.add_argument('--target', type=Fraction, help='Fixed ratio to exceed; stop when proved. Overrides --gain.')
    s.add_argument('--pool', type=int, default=2400, help='Random candidate sites per round, in addition to deterministic sites.')
    s.add_argument('--refine', type=int, default=16, help='Candidate sites locally optimized on faces per round.')
    s.add_argument('--refine-iters', type=int, default=45)
    s.add_argument('--colorings', type=int, default=4)
    s.add_argument('--block-pool', type=int, default=96, help='Top/diverse candidate sites used for cooperative blocking tests.')
    s.add_argument('--block-checks', type=int, default=90, help='List-coloring checks to reduce one cooperative blocker.')
    s.add_argument('--block-timeout', type=float, default=.08, help='Time budget for a small fixed-coloring CSP; UNKNOWN is not UNSAT.')
    s.add_argument('--evaluate-every', type=int, default=4)
    s.add_argument('--eval-checks', type=int, default=35, help='Maximum SAT calls in each exact-distance bisection.')
    s.add_argument('--polish-iters', type=int, default=80,
                   help='SLSQP iterations on ADDED points only after a lower-bound improvement; 0 disables.')
    s.add_argument('--weight-bits', type=int, default=24, help='New-input weights have denominator 2**bits; a resumed state keeps its grid.')
    s.add_argument('--facets', choices=('auto', 'verified', 'exhaustive'), default='auto')
    s.add_argument('--facet-enumeration-limit', type=int, default=200000)
    s.add_argument('--seed', type=int, default=20260909)
    v = sub.add_parser('verify', help='Verify finite-set bounds AND exact boundary membership.')
    v.add_argument('directory', type=Path)
    for q in (s, v):
        q.add_argument('--backend', choices=('auto', 'native', 'external', 'pysat', 'python'), default='auto')
        q.add_argument('--solver', default='kissat', help='Kissat command or another DIMACS solver supported by borsuk_search.py.')
        q.add_argument('--sat-timeout', type=float, default=15.)
        q.add_argument('--pysat-solver', default='g4', choices=('g3', 'g4', 'm22', 'mgh'))
    return p


def main(argv=None):
    parser = build_parser(); args = parser.parse_args(argv)
    try:
        require(math.isfinite(args.sat_timeout) and args.sat_timeout >= 0, '--sat-timeout must be nonnegative.')
        if args.command == 'verify':
            print(json.dumps(verify_surface(args.directory, args), indent=2))
            return 0
        for key in ('batch', 'colorings', 'block_pool', 'block_checks', 'evaluate_every', 'eval_checks'):
            require(getattr(args, key) >= 1, f'--{key.replace("_", "-")} must be positive.')
        for key in ('rounds', 'pool', 'refine', 'refine_iters', 'polish_iters', 'facet_enumeration_limit'):
            require(getattr(args, key) >= 0, f'--{key.replace("_", "-")} must be nonnegative.')
        require(2 <= args.weight_bits <= 40, '--weight-bits must be between 2 and 40.')
        require(0 <= args.gain < 1, '--gain must be in [0, 1).')
        require(args.target is None or 0 <= args.target < 1, '--target must be in [0, 1).')
        require(math.isfinite(args.block_timeout) and args.block_timeout > 0, '--block-timeout must be positive.')
        require(args.block_pool >= args.batch, '--block-pool must be at least --batch.')
        print(json.dumps(run_search(args), indent=2))
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Completed best/ and latest/ checkpoints are preserved; '
              'a *.previous directory is the fallback during a checkpoint rename.', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
