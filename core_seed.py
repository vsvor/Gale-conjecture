#!/usr/bin/env python3
"""Fresh structured seeds, distance-signature sampling, and core-aware screening.

This is an experimental initializer, not a guarantee of larger minimum cores.
It retains the supplied geometric exceptions, NOT all points of the old record.
The fitted curves/patches guide sampling only. All coordinates are free in warmup.

Examples:
  python core_seed.py generate --dim 4 --points 81 --attempts 12 --backend native --out seeds4
  python core_seed.py inspect saved_best --k 5 --backend native --out audit

The exact CSV geometry and backend-confirmed colouring brackets are separated
from heuristic scoring. UNKNOWN is never SAT or UNSAT. All graph-core quantities
are explicitly distinguished from minimum-cardinality and renormalized cores.
"""
from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path
import time
from typing import Any

import borsuk_search as bs  # Also limits BLAS threading before importing NumPy.
import tune_solution as ts
import structured_seed as ss
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent


def quotas(weights: np.ndarray, count: int) -> np.ndarray:
    if count < 0 or np.any(weights <= 0):
        raise ValueError('Invalid allocation.')
    raw = weights / weights.sum() * count
    q = np.floor(raw).astype(int)
    for i in np.argsort(-(raw-q), kind='stable')[:count-int(q.sum())]:
        q[i] += 1
    return q


def template_pool(dim: int, count: int, rng: np.random.Generator,
                  pool_factor: int = 20, jitter: float = .002,
                  shape_noise: float = .004, patch_domain: str = 'empirical') -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray, list[str]]:
    """Candidates, group labels, fixed exceptions, reference landmarks, quotas, names.

    The old regular vertices are used only as distance landmarks; they are NOT
    inserted. Radius/center perturbations are shared within a component. No exact
    reflection or orthogonal-triangle constraints are imposed on an eventual seed.
    """
    candidates, labels, names, weights = [], [], [], []
    if dim == 3:
        template = json.loads((ROOT/'templates/curves3.json').read_text())
        comparison = json.loads((ROOT/'templates/curves3_comparison.json').read_text())
        source = np.load(ROOT/'templates/source3.npy', allow_pickle=False)
        frame = template['frame']; B = np.array(frame['basis_columns_a_b_n'])
        landmarks = (source-np.array(frame['origin'])) @ B / frame['diameter']
        exceptions = comparison['fits']['minus_near_apex']['excluded']
        fixed = landmarks[exceptions].copy()
        for comp in ss.components(template, comparison, 'circle'):
            for flip in ((1, -1) if comp['paired'] else (1,)):
                names.append(comp['name'] + ('_mirror' if flip == -1 else ''))
                weights.append(comp['length'])
                pool_n = max(64, pool_factor*count//9)
                f = (np.arange(pool_n)+rng.uniform(.05, .95, pool_n))/pool_n
                rng.shuffle(f)
                p = ss.sample_curve(comp, f)
                p[:, 2] *= flip
                middle = p.mean(axis=0)
                p = middle + math.exp(rng.normal(0, shape_noise))* (p-middle)
                p += rng.normal(0, shape_noise, (1, dim))
                p += rng.normal(0, jitter, p.shape)
                candidates.extend(p); labels.extend([len(names)-1]*len(p))
    elif dim == 4:
        template = json.loads((ROOT/'templates/patches4.json').read_text())
        source = np.loadtxt(ROOT/'templates/source4.csv', delimiter=',')
        landmarks = (source - np.array(template['origin']))/template['D']
        modeled = {int(i) for p in template['models'] for i in p['indices']}
        fixed = landmarks[[i for i in range(len(source)) if i not in modeled]].copy()
        for patch in template['models']:
            names.append('patch_'+str(patch['group'])); weights.append(1.)
            center = np.array(patch['center']); normal = np.array(patch['normal'])
            normal /= np.linalg.norm(normal)
            pole = (np.array(patch['pole']) - center) / patch['radius']
            pole -= pole.dot(normal)*normal; pole /= np.linalg.norm(pole)
            # Tangent basis within the supporting 3-plane at the fitted pole.
            _, _, vh = np.linalg.svd(np.vstack((normal, pole)), full_matrices=True)
            transverse = vh[2:].T
            dirs = np.array(patch['observed_spherical_directions'])
            obs = dirs @ transverse
            cov = np.cov(obs.T) + 1e-5*np.eye(2)
            pool_n = max(96, pool_factor*count//6)
            if patch_domain == 'gaussian':
                z = rng.multivariate_normal(obs.mean(axis=0), cov, pool_n)
                z *= rng.uniform(.85, 1.15)
            else:
                # Uniform area sampling inside the observed tangent-coordinate hull,
                # plus a small explicit boundary quota; do not extrapolate a full cap.
                vertices = obs[ConvexHull(obs).vertices]
                middle = vertices.mean(axis=0)
                nxt = np.roll(vertices, -1, axis=0)
                va, vb = vertices-middle, nxt-middle
                area = np.abs(va[:,0]*vb[:,1]-va[:,1]*vb[:,0])
                fan = rng.choice(len(vertices), pool_n, p=area/area.sum())
                w = rng.dirichlet(np.ones(3), pool_n)
                z = w[:,0,None]*middle + w[:,1,None]*vertices[fan] + w[:,2,None]*nxt[fan]
                boundary = pool_n//4
                f = (np.arange(boundary)+rng.random(boundary))/max(1, boundary)
                side = rng.integers(len(vertices), size=boundary)
                z[:boundary] = (1-f[:,None])*vertices[side]+f[:,None]*nxt[side]
                z = middle + rng.uniform(.97, 1.05)*(z-middle)
            # Sample a broad but bounded cap; model fitting does not justify a full sphere.
            norm = np.linalg.norm(z, axis=1)
            z *= np.minimum(1., .93/np.maximum(norm, 1e-12))[:, None]
            direction = np.sqrt(1-np.sum(z*z, axis=1))[:, None]*pole + z@transverse.T
            radius = patch['radius']*math.exp(rng.normal(0, shape_noise))
            p = center + rng.normal(0, shape_noise, dim) + radius*direction
            p += rng.normal(0, jitter, p.shape)
            candidates.extend(p); labels.extend([len(names)-1]*len(p))
    else:
        raise ValueError('Fitted geometric proposals support dimensions 3 and 4 only.')
    if count <= len(fixed):
        raise ValueError(f'--points must exceed {len(fixed)} retained exceptions.')
    return (np.array(candidates), np.array(labels), fixed, landmarks,
            quotas(np.array(weights), count-len(fixed)), names)


def distance_features(points: np.ndarray, landmarks: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    """Binary old-neighbour signatures at several distance thresholds, plus geometry.

    Squared Euclidean distances of these feature vectors equal mean Hamming
    distance plus a small spatial term. This is only a proposal heuristic: the
    actual graph after resampling/warmup includes new-new edges as well.
    """
    q = cdist(points, landmarks, 'sqeuclidean')
    bits = np.concatenate([q >= t*t for t in thresholds], axis=1)
    return np.column_stack((bits/np.sqrt(bits.shape[1]), .25*points))


def select_pool(candidates: np.ndarray, labels: np.ndarray, fixed: np.ndarray,
                landmarks: np.ndarray, counts: np.ndarray, method: str,
                rng: np.random.Generator, min_separation: float = .003,
                thresholds: tuple[float, ...] = (.86, .885, .91)) -> tuple[np.ndarray, list[int]]:
    if method not in ('iid', 'spatial', 'signature'):
        raise ValueError('Unknown candidate selection method.')
    left = np.array(counts, copy=True)
    available = np.ones(len(candidates), bool)
    separation2 = min_separation**2
    geo_dist = cdist(candidates, fixed, 'sqeuclidean').min(axis=1)
    if method == 'signature':
        features = distance_features(candidates, landmarks, thresholds)
        fixed_features = distance_features(fixed, landmarks, thresholds)
        diversity = cdist(features, fixed_features, 'sqeuclidean').min(axis=1)
    else:
        features = candidates
        diversity = geo_dist.copy()
    chosen = []
    for _ in range(int(left.sum())):
        admissible = available & (left[labels] > 0) & (geo_dist > separation2)
        ids = np.flatnonzero(admissible)
        if not len(ids):
            raise ValueError('Not enough separated proposals; increase --pool-factor or reduce --min-separation.')
        if method == 'iid':
            # Pick a group in proportion to its remaining quota, then a uniform candidate.
            valid_groups = np.unique(labels[ids])
            probs = left[valid_groups]/left[valid_groups].sum()
            group = int(rng.choice(valid_groups, p=probs))
            j = int(rng.choice(ids[labels[ids] == group]))
        else:
            # Tiny random tie-breaking does not overwhelm distinct signature distances.
            j = int(ids[np.argmax(diversity[ids] + rng.uniform(0, 1e-12, len(ids)))])
        chosen.append(j); available[j] = False; left[labels[j]] -= 1
        geo_dist = np.minimum(geo_dist, np.sum((candidates-candidates[j])**2, axis=1))
        diversity = np.minimum(diversity, np.sum((features-features[j])**2, axis=1))
    return np.vstack((fixed, candidates[chosen])), chosen


def propose(dim: int, count: int, method: str, seed: int, args: argparse.Namespace):
    rng = np.random.default_rng(seed)
    if method == 'sphere':
        return bs.normalize(bs.make_seed(dim, count, 'sphere', rng)), {'family': 'sphere', 'exceptions_retained': 0}
    p, labels, fixed, landmarks, counts, names = template_pool(
        dim, count, rng, args.pool_factor, args.jitter, args.shape_noise, args.patch_domain)
    mid = .888 if dim == 3 else .913
    x, selected = select_pool(p, labels, fixed, landmarks, counts, method, rng,
                              args.min_separation, (mid-.035, mid-.015, mid, mid+.015))
    # Canonical similarity normalization. It does not preserve a source ratio;
    # the exceptions are retained up to this common translation and scaling.
    return bs.normalize(x), {'family': 'fitted_'+method, 'exceptions_retained': len(fixed),
                             'group_names': names, 'group_counts': counts.tolist(),
                             'candidate_pool': len(p), 'selected_pool_indices': selected,
                             'old_record_is_NOT_retained': True}


def warmup(x: np.ndarray, k: int, pool: ts.OraclePool, args: argparse.Namespace,
           seed: int) -> tuple[ts.State, list[dict]]:
    rng = np.random.default_rng(seed)
    g = ts.Geometry.from_float(bs.normalize(x))
    state = ts.assess(g, pool)
    record = [{'round': 0, 'lower': state.lower, 'upper': state.upper}]
    oracle = pool.get(g.m)
    for j in range(args.warmup_rounds):
        x = bs.normalize(state.g.x)
        q = bs.distances2(x, oracle.pairs)
        threshold = state.lower2 if not j else state.lower2*(1-Fraction(str(args.graph_slack)))**2
        ids = state.g.threshold(threshold)
        ids = bs.prune_witness(oracle, ids, q, rng, checks=args.prune_checks, noise=.035)
        if args.steps > 0:
            y, _ = bs.optimize_witness(x, oracle.pairs, ids, steps=args.steps, polish=0)
        else:
            y = x.copy()
        candidate = ts.assess(ts.Geometry.from_float(y), pool, ids, state.colors, True)
        if candidate.lower2 > state.lower2:
            state = candidate
        record.append({'round': j+1, 'lower': state.lower, 'upper': state.upper})
    return state, record


def fixed_threshold_audit(state: ts.State, pool: ts.OraclePool, orders: int, seed: int) -> dict:
    """Exact graph diagnostics at r=lower. Equal to the optimum if state.exact.

    Preserve a diameter pair, so every retained witness subset has the same
    denominator. Random deletion cores are inclusion-minimal, not necessarily
    minimum-cardinality. Required vertices (SAT after deleting v) are necessary
    in EVERY induced non-k-colourable subset at this threshold.
    """
    if not len(state.witness) or state.g.m <= pool.k:
        raise ValueError('A positive non-k-colourability witness is required for core screening.')
    g = state.g; m = g.m; oracle = pool.get(m); rng = np.random.default_rng(seed)
    ids = g.threshold(state.lower2)
    edges = g.pairs[ids]
    diameter_pair = list(map(int, g.pairs[g.q.index(g.diameter2)]))
    required, essential_unknown = [], []
    for v in range(m):
        ans = oracle.solve(ids[(edges != v).all(axis=1)])
        if ans.status is bs.Status.SAT:
            required.append(v)
        elif ans.status is bs.Status.UNKNOWN:
            essential_unknown.append(v)
    cores = []
    for j in range(orders):
        keep = np.ones(m, bool); unknown = []
        for vertex in rng.permutation(m):
            v = int(vertex)
            if v in diameter_pair or v in required:
                continue
            keep[v] = False
            trial = ids[keep[edges].all(axis=1)]
            ans = oracle.solve(trial)
            if ans.status is not bs.Status.UNSAT:
                keep[v] = True
                if ans.status is bs.Status.UNKNOWN: unknown.append(v)
        cores.append({'indices': np.flatnonzero(keep).tolist(),
                      'count': int(keep.sum()), 'unknown': unknown,
                      'inclusion_minimal_subject_to_diameter_pair': not unknown})
    adjacency = np.zeros((m, m), bool)
    adjacency[edges[:, 0], edges[:, 1]] = True
    adjacency |= adjacency.T
    low_degree = np.flatnonzero(adjacency.sum(axis=1) < pool.k).tolist()
    folded = set()
    for v in range(m):
        for u in range(m):
            if u != v and not adjacency[u, v] and np.all(~adjacency[v] | adjacency[u]):
                folded.add(v); break
    return {'points': m, 'lower': state.lower, 'upper': state.upper,
            'threshold_is_exact_optimum': state.exact,
            'diameter_pair': diameter_pair,
            'required_vertices': required, 'required_count': len(required),
            'required_check_unknowns': essential_unknown,
            'anchored_minimum_core_lower_bound': len(set(required+diameter_pair)),
            'anchored_minimum_core_upper_bound': min(c['count'] for c in cores),
            'anchored_random_deletion_counts': [c['count'] for c in cores],
            'cores': cores, 'low_degree_vertices': low_degree,
            'dominated_vertices_in_full_threshold_graph': sorted(folded),
            'qualification': 'Core bounds concern induced non-k-colourable subgraphs '
                             'at the stated lower threshold, with the diameter pair retained. '
                             'The threshold equals the exact partition optimum only if resolved. '
                             'Without retaining the diameter pair, normalization changes.'}


def normalized_reductions(state: ts.State, pool: ts.OraclePool, orders: int, seed: int) -> dict:
    """Run the EXISTING normalized non-degrading reducer in several orders.

    Recheck to a fixed point because unlike graph deletion, a changing diameter
    can make earlier failed deletions valid later. UNKNOWN blocks minimality.
    """
    runs = []
    for j in range(orders):
        current = state; rng = np.random.default_rng(seed+104729*j); all_indices = np.arange(state.g.m)
        unknowns = 0; passes = 0
        while True:
            passes += 1
            result, log = ts.reduce_vertices(current, pool, rng, current.g.m, pool.k+1)
            removed = log['deleted_original_indices']; unknowns += log['unknowns']
            if not removed: break
            all_indices = np.delete(all_indices, removed)
            current = result
        runs.append({'count': current.g.m, 'indices': all_indices.tolist(),
                     'lower': current.lower, 'upper': current.upper,
                     'unknowns': unknowns, 'passes': passes})
    return {'runs': runs, 'counts': [r['count'] for r in runs],
            'min_found': min(r['count'] for r in runs),
            'qualification': 'Sizes found by normalized non-degrading greedy deletion. '
                             'They are NOT minimum-cardinality guarantees. '
                             'A deletion can strictly INCREASE the normalized ratio.'}


def screen_state(state: ts.State, pool: ts.OraclePool, args, seed: int) -> dict:
    report = fixed_threshold_audit(state, pool, args.core_orders, seed)
    if args.normalized_audit:
        report['normalized_reductions'] = normalized_reductions(state, pool, args.core_orders, seed)
    return report


def choose_winner(results: list[dict], ratio_window: float, min_core: int) -> int | None:
    valid = [(i, r) for i, r in enumerate(results) if r['audit']['threshold_is_exact_optimum']]
    if not valid:
        return None
    top_ratio = max(r['audit']['lower'] for i, r in valid)
    eligible = [(i, r) for i, r in valid if r['audit']['lower'] >= top_ratio-ratio_window
                and r['audit']['anchored_minimum_core_upper_bound'] >= min_core
                and not r['audit']['required_check_unknowns']
                and all(not c['unknown'] for c in r['audit']['cores'])]
    if not eligible:
        return None
    return max(eligible, key=lambda ir: (ir[1]['audit']['anchored_minimum_core_upper_bound'],
                                        ir[1]['audit']['required_count'],
                                        ir[1]['audit']['lower']))[0]


def backend_options(p):
    bs.add_backend_options(p)
    p.set_defaults(backend='auto', sat_timeout=5.)
    p.add_argument('--core-orders', type=int, default=3)
    p.add_argument('--normalized-audit', action='store_true',
                   help='Also run normalized non-degrading vertex reductions; extra SAT calls.')


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate', help='Generate, briefly optimize, and screen fresh seeds.')
    g.add_argument('--dim', type=int, choices=(3, 4), required=True)
    g.add_argument('--points', type=int, default=81)
    g.add_argument('--k', type=int)
    g.add_argument('--method', choices=('sphere', 'iid', 'spatial', 'signature'), default='spatial')
    g.add_argument('--attempts', type=int, default=8)
    g.add_argument('--seed', type=int, default=20260910)
    g.add_argument('--pool-factor', type=int, default=20)
    g.add_argument('--jitter', type=float, default=.002)
    g.add_argument('--shape-noise', type=float, default=.004)
    g.add_argument('--patch-domain', choices=('empirical','gaussian'), default='empirical',
                   help='R4: sample the observed patch domain (recommended), not an entire sphere.')
    g.add_argument('--min-separation', type=float, default=.003)
    g.add_argument('--warmup-rounds', type=int, default=2)
    g.add_argument('--steps', type=int, default=40)
    g.add_argument('--prune-checks', type=int, default=120)
    g.add_argument('--graph-slack', type=float, default=.015)
    g.add_argument('--ratio-window', type=float, default=.015,
                   help='Eligible seeds must be within this absolute gap of the best screened ratio.')
    g.add_argument('--min-core', type=int, default=0,
                   help='Reject if any extracted anchored core has fewer vertices; heuristic, not a guarantee.')
    g.add_argument('--out', type=Path, required=True)
    backend_options(g)
    i = sub.add_parser('inspect', help='Audit an existing point set without optimization.')
    i.add_argument('start', type=Path)
    i.add_argument('--k', type=int)
    i.add_argument('--seed', type=int, default=20260910)
    i.add_argument('--out', type=Path, required=True)
    backend_options(i)
    return p


def main(argv=None):
    p = build_parser(); args = p.parse_args(argv)
    if args.out.exists(): p.error('Use a new output directory.')
    if args.core_orders < 1 or args.sat_timeout < 0: p.error('Invalid screening budget.')
    if args.command == 'generate':
        if (args.attempts < 1 or args.points < 2 or args.pool_factor < 1 or args.jitter < 0
            or args.shape_noise < 0 or args.warmup_rounds < 0 or args.steps < 0
            or args.prune_checks < 0 or args.min_separation < 0 or args.ratio_window < 0
            or args.min_core < 0 or not 0 <= args.graph_slack < 1):
            p.error('Invalid generation parameters.')
        k = args.dim+1 if args.k is None else args.k
        if not 1 <= k < args.points: p.error('Core screening requires 1 <= k < points.')
    else:
        g, w, c, saved_k = ts.load_source(args.start)
        k = (saved_k or g.dim+1) if args.k is None else args.k
        if not 1 <= k < g.m: p.error('Core screening requires 1 <= k < points.')
    args.out.mkdir(parents=True)
    (args.out/'settings.json').write_text(json.dumps(vars(args), indent=2, default=str)+'\n')
    pool = ts.OraclePool(k, args)
    start = time.monotonic()
    try:
        if args.command == 'inspect':
            state = ts.assess(g, pool, w, c)
            report = screen_state(state, pool, args, args.seed)
            (args.out/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(report, indent=2)); return 0
        results = []
        total_stats = {name: 0 for name in ('calls', 'unknowns', 'cache_hits')}
        for attempt in range(args.attempts):
            seed = args.seed + 1000003*attempt
            pool.close(); pool = ts.OraclePool(k, args)
            t = time.monotonic()
            x, meta = propose(args.dim, args.points, args.method, seed, args)
            folder = args.out/f'seed_{attempt:04d}'
            folder.mkdir(); np.save(folder/'initial.npy', x)
            state, trace = warmup(x, k, pool, args, seed+1)
            ts.save_state(folder, state, k, {'initializer': meta, 'seed': seed})
            audit = screen_state(state, pool, args, seed+2)
            for name, value in pool.stats().items(): total_stats[name] += value
            result = {'attempt': attempt, 'seed': seed, 'path': str(folder.resolve()), 'proposal': meta,
                      'warmup': trace, 'audit': audit, 'elapsed': time.monotonic()-t}
            (folder/'screen.json').write_text(json.dumps(result, indent=2)+'\n')
            results.append(result)
            print(json.dumps({'attempt': attempt, 'R': [state.lower, state.upper],
                              'required': audit['required_count'],
                              'anchored_core_sizes': audit['anchored_random_deletion_counts'],
                              'elapsed': result['elapsed']}), flush=True)
            (args.out/'partial_results.json').write_text(json.dumps(results, indent=2)+'\n')
        winner = choose_winner(results, args.ratio_window, args.min_core)
        if winner is not None:
            import shutil
            shutil.copytree(results[winner]['path'], args.out/'selected')
        report = {'selected_attempt': winner, 'elapsed': time.monotonic()-start,
                  'oracle_stats': total_stats, 'results': results,
                  'qualification': 'Experimental core-aware screening. Largest found cores '
                                   'are heuristic upper bounds on MINIMUM cores, not guarantees. '
                                   'Expectations require a controlled multi-run benchmark.'}
        (args.out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'selected_attempt': winner, 'output': str(args.out), 'elapsed': report['elapsed']}))
        return 0 if winner is not None else 3
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        p.error(str(exc))
    except KeyboardInterrupt:
        print('Interrupted; completed seed folders and partial_results.json remain available.')
        return 130
    finally:
        pool.close()

if __name__ == '__main__':
    raise SystemExit(main())
