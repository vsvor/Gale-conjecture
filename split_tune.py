#!/usr/bin/env python3
"""Improve Borsuk min-max configurations by essential vertex splitting.

Put beside borsuk_search.py and tune_solution.py. Requires numpy and scipy;
uses the same SAT/DSATUR backends and exact-decimal certificates as the tuner.
An auxiliary split may START with coincident coordinates. Only distinct,
re-evaluated configurations with certified improved bounds are published.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = os.environ.get('BORSUK_BLAS_THREADS', '1')

import argparse
from dataclasses import dataclass
from fractions import Fraction
import itertools
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterator

import numpy as np
try:
    import borsuk_search as bs
    import tune_solution as ts
except ImportError as exc:
    raise SystemExit('Place this script beside borsuk_search.py and tune_solution.py.') from exc

VERSION = '1.0'


def split_edge_ids(m: int, edges: np.ndarray, vertex: int,
                   left: tuple[int, ...], right: tuple[int, ...]) -> np.ndarray:
    """Duplicate v as m, omit v-left and m-right, with no edge between copies.

    Disjoint nonempty omissions make the two graph neighborhoods different.
    This function constructs a graph; it does NOT assert non-colorability.
    """
    edges = np.asarray(edges, dtype=np.int32).reshape(-1, 2)
    if not 0 <= vertex < m or not left or not right or set(left) & set(right):
        raise ValueError('Require a valid vertex and disjoint nonempty omission sets.')
    # Validate endpoints via the engine's index conversion.
    ts.edge_ids(m, edges)
    neighbors = set(map(int, edges[edges[:, 0] == vertex, 1]))
    neighbors.update(map(int, edges[edges[:, 1] == vertex, 0]))
    if not (set(left) | set(right)) <= neighbors:
        raise ValueError('An omitted endpoint is not a neighbor of the split vertex.')
    enlarged = np.vstack((edges, np.array([(u, m) for u in sorted(neighbors)], np.int32)))
    removed = ts.edge_ids(m+1, np.array([(vertex, a) for a in left] +
                                        [(m, b) for b in right], np.int32))
    return np.setdiff1d(ts.edge_ids(m+1, enlarged), removed).astype(np.int32)


def omission_pairs(neighbors: list[int], essential: set[int],
                   max_omit: int) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Each omission contains an edge whose deletion from the old graph is SAT.

    Swapping the two copies gives an isomorphic split, so unordered pairs of
    omission sets suffice. Intersecting omissions cannot help: the result would
    be a subgraph of a duplication of a colorable edge-deleted old graph when
    a shared omitted edge is essential; in all cases they are excluded here.
    """
    groups = [a for size in range(1, min(max_omit, len(neighbors))+1)
              for a in itertools.combinations(sorted(neighbors), size)
              if set(a) & essential]
    for a, b in itertools.combinations(groups, 2):
        if not set(a) & set(b):
            yield a, b


@dataclass
class Split:
    vertex: int
    left: tuple[int, ...]
    right: tuple[int, ...]
    ids: np.ndarray
    lp_gain: float
    audit: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        return {'vertex': self.vertex, 'omit_from_original': list(self.left),
                'omit_from_copy': list(self.right), 'linearized_gain': self.lp_gain,
                'audit': self.audit}


def discover_splits(state: ts.State, pool: ts.OraclePool, args: argparse.Namespace,
                    log: Callable[..., None]) -> tuple[list[Split], dict[str, Any]]:
    g = state.g
    x = bs.normalize(g.x)
    m = g.m
    # The complete threshold graph contains the verified witness, so is UNSAT.
    ids = g.threshold(state.lower2)
    q = bs.distances2(x, g.pairs)
    oracle = pool.get(m)
    if args.prune_checks:
        ids = bs.prune_witness(oracle, ids, q,
                              np.random.default_rng(args.seed + m),
                              checks=args.prune_checks, noise=0.04)
    edges = g.pairs[ids]
    active = ids[q[ids] <= q[ids].min()+args.active_tol]
    critical: set[int] = set()
    stats: dict[str, Any] = {'vertices': m, 'witness_edges': len(ids),
                            'near_shortest_edges': len(active),
                            'edge_deletion_UNKNOWN': 0, 'split_SAT': 0,
                            'split_UNSAT': 0, 'split_UNKNOWN': 0,
                            'LP_failed': 0, 'positive_LP': 0,
                            'screen_limit_reached': False}
    for edge in active:
        ans = oracle.solve(ids[ids != edge])
        if ans.status is bs.Status.SAT:
            critical.add(int(edge))
        elif ans.status is bs.Status.UNKNOWN:
            stats['edge_deletion_UNKNOWN'] += 1
    stats['critical_near_shortest_edges'] = len(critical)
    new_oracle = pool.get(m+1)
    splits: list[Split] = []
    tested = 0
    for v in range(m):
        near_neighbors: list[int] = []
        essential: set[int] = set()
        for i in active:
            u, z = map(int, g.pairs[i])
            if v not in (u, z):
                continue
            other = z if u == v else u
            near_neighbors.append(other)
            if int(i) in critical:
                essential.add(other)
        for a, b in omission_pairs(near_neighbors, essential, args.max_omit):
            if args.screen_limit and tested >= args.screen_limit:
                stats['screen_limit_reached'] = True
                break
            candidate = split_edge_ids(m, edges, v, a, b)
            answer = new_oracle.solve(candidate)
            tested += 1
            stats['split_'+answer.status.value] += 1
            if answer.status is not bs.Status.UNSAT:
                continue
            # Deleting either copy gives a subgraph of a verified SAT
            # edge-deleted old graph. Thus both copies are essential.
            # Coincidence is ONLY an auxiliary optimization seed.
            seed = np.vstack((x, x[v]))
            audit, _ = ts.linearized_audit(seed, candidate, args.active_tol, args.lp_timeout)
            if not audit['success']:
                stats['LP_failed'] += 1
                continue
            gain = float(audit['linearized_gain'])
            if gain > args.lp_gain_tol:
                stats['positive_LP'] += 1
            splits.append(Split(v, a, b, candidate, gain, audit))
        if stats['screen_limit_reached']:
            break
    stats['split_tests'] = tested
    log('screen', **stats)
    return splits, stats


def select_splits(splits: list[Split], args: argparse.Namespace) -> list[Split]:
    """Rank by LP gain, but keep one vertex's variants from using all the budget."""
    positive = sorted((s for s in splits if s.lp_gain > args.lp_gain_tol),
                      key=lambda s: -s.lp_gain)
    counts: dict[int, int] = {}
    chosen: list[Split] = []
    for s in positive:
        if args.per_vertex and counts.get(s.vertex, 0) >= args.per_vertex:
            continue
        chosen.append(s)
        counts[s.vertex] = counts.get(s.vertex, 0)+1
        if args.trials and len(chosen) >= args.trials:
            break
    zero = [s for s in splits if s.lp_gain <= args.lp_gain_tol]
    if args.zero_gain_trials and zero:
        rng = np.random.default_rng(args.seed)
        chosen += [zero[int(i)] for i in rng.permutation(len(zero))[:args.zero_gain_trials]]
    return chosen


def split_diagnostics(state: ts.State, vertex: int, copy: int,
                      pool: ts.OraclePool) -> dict[str, Any]:
    """Describe the actual saved geometric threshold, not just its scaffold."""
    g = state.g
    ids = g.threshold(state.lower2)
    A = ts.adjacency(g.m, g.pairs[ids])
    outside = np.ones(g.m, bool)
    outside[[vertex, copy]] = False
    same = bool(np.array_equal(A[vertex, outside], A[copy, outside]))
    status = {}
    for v in (vertex, copy):
        trial = ids[np.all(g.pairs[ids] != v, axis=1)]
        status[str(v)] = pool.get(g.m).solve(trial).status.value
    return {'copy_index': copy,
            'separation': float(np.linalg.norm(g.x[vertex]-g.x[copy])),
            'twins_at_saved_threshold': same,
            'delete_copy_vertex_status': status,
            'note': 'SAT means this threshold obstruction disappears; UNKNOWN is unresolved.'}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('start', type=Path, help='Coordinate file, saved best directory, or search root.')
    p.add_argument('--out', type=Path, default=Path('split_results'), help='New, empty output directory.')
    p.add_argument('--k', type=int, help='Number of parts; saved k, otherwise dimension+1.')
    p.add_argument('--rounds', type=int, default=20, help='Maximum accepted-split rounds.')
    p.add_argument('--max-points', type=int, help='Default: reduced starting size plus --rounds.')
    p.add_argument('--max-omit', type=int, default=2, choices=(1, 2, 3),
                   help='Maximum number of neighbors omitted from each copy.')
    p.add_argument('--trials', type=int, default=24, help='Positive-LP splits optimized per round; 0 = all.')
    p.add_argument('--per-vertex', type=int, default=4, help='Maximum positive-LP trials for one vertex; 0 = unlimited.')
    p.add_argument('--screen-limit', type=int, default=0, help='Maximum split SAT tests per round; 0 = all.')
    p.add_argument('--zero-gain-trials', type=int, default=0, help='Also try this many numerical zero-LP splits.')
    p.add_argument('--zero-gain-jitter', type=float, default=0.002, help='Opposite perturbations of the copies in zero-LP trials, in diameter units.')
    p.add_argument('--prune-checks', type=int, default=0, help='Optional witness edge pruning before splitting.')
    p.add_argument('--seed', type=int, default=20260908)
    p.add_argument('--no-reduce', action='store_true', help='Skip exact ratio-preserving vertex reductions.')
    p.add_argument('--min-gain', type=float, default=1e-10, help='Minimum numerical gain for continuing; exact bound test also required.')
    p.add_argument('--active-tol', type=float, default=1e-8, help='Squared-distance tolerance for near-shortest edges.')
    p.add_argument('--lp-gain-tol', type=float, default=1e-7, help='Numerical LP-gain screening threshold; not an exact theorem.')
    p.add_argument('--lp-timeout', type=float, default=30.)
    p.add_argument('--slsqp-iters', type=int, default=300)
    p.add_argument('--polish-passes', type=int, default=8, help='Repeated SLSQP passes for the final incumbent.')
    p.add_argument('--ftol', type=float, default=1e-12)
    p.add_argument('--full-constraints', action='store_true', help='Use all diameter constraints in trial SLSQP calls.')
    p.add_argument('--save-trials', action='store_true', help='Also save every evaluated trial configuration.')
    p.add_argument('--quiet', action='store_true')
    bs.add_backend_options(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    pool = None
    try:
        for name in ('rounds', 'trials', 'per_vertex', 'screen_limit', 'zero_gain_trials', 'prune_checks'):
            if getattr(args, name) < 0:
                raise ValueError(f'--{name.replace("_", "-")} must be nonnegative.')
        if args.k is not None and args.k < 2:
            raise ValueError('Require k >= 2.')
        if args.slsqp_iters < 1 or args.polish_passes < 1:
            raise ValueError('SLSQP iteration/pass budgets must be positive.')
        if any(not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0
               for name in ('min_gain', 'active_tol', 'lp_gain_tol', 'lp_timeout', 'ftol', 'zero_gain_jitter')):
            raise ValueError('Tolerances and LP timeout must be finite and positive.')
        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError('Output is not empty; choose a new --out directory.')
        args.out.mkdir(parents=True, exist_ok=True)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        started = time.monotonic()
        def log(event: str, **data: Any) -> None:
            record = {'event': event, 'seconds': round(time.monotonic()-started, 4), **data}
            with (args.out/'progress.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            if not args.quiet:
                print(event, json.dumps(data), flush=True)
        # Retain the established tuner's constraint generation, round-trip
        # geometry checks and polishing diagnostics without changing that file.
        numerical = ts.build_parser().parse_args([str(args.start)])
        for key, value in vars(args).items():
            if hasattr(numerical, key):
                setattr(numerical, key, value)
        numerical.steps = 0
        numerical.refreshes = 3
        numerical.polish_kicks = 1
        numerical.stationarity_tol = args.lp_gain_tol
        g, witness, colors, saved_k = ts.load_source(args.start)
        k = args.k if args.k is not None else (saved_k if saved_k is not None else g.dim+1)
        if k < 2 or g.m <= k:
            raise ValueError('Require at least k+1 distinct points and k >= 2.')
        if saved_k is not None and k != saved_k:
            witness, colors = None, None
        pool = ts.OraclePool(k, numerical)
        initial = ts.assess(g, pool, witness, colors)
        checkpoint = ts.Checkpoints(args.out, k, log)
        checkpoint.consider(initial, 'input_rechecked')
        rng = np.random.default_rng(args.seed)
        if not args.no_reduce:
            reduced, report = ts.reduce_vertices(initial, pool, rng, initial.g.m, 0, log)
            (args.out/'reduction.json').write_text(json.dumps(report, indent=2)+'\n')
            checkpoint.consider(reduced, 'exact_initial_reduction')
        if args.max_points is None:
            args.max_points = checkpoint.best.g.m+args.rounds
        if args.max_points < checkpoint.best.g.m:
            raise ValueError('--max-points is smaller than the retained starting configuration.')
        stop = 'round_budget_exhausted'
        accepted_rounds = 0
        for round_no in range(1, args.rounds+1):
            source = checkpoint.best
            if source.g.m >= args.max_points:
                stop = 'point_budget_exhausted'
                break
            log('round', number=round_no, points=source.g.m, lower=source.lower, upper=source.upper)
            candidates, screen = discover_splits(source, pool, args, log)
            folder = args.out/'rounds'/f'{round_no:03d}'
            folder.mkdir(parents=True, exist_ok=True)
            selected = select_splits(candidates, args)
            (folder/'screen.json').write_text(json.dumps({'summary': screen,
                'selected': [s.metadata() for s in selected]}, indent=2)+'\n')
            if not selected:
                stop = 'no_selected_split_in_tested_family'
                break
            improved = False
            for i, split in enumerate(selected):
                x = bs.normalize(source.g.x)
                seed = np.vstack((x, x[split.vertex]))
                if split.lp_gain <= args.lp_gain_tol:
                    direction = rng.normal(size=x.shape[1])
                    direction /= max(float(np.linalg.norm(direction)), 1e-15)
                    seed[split.vertex] -= args.zero_gain_jitter*direction
                    seed[-1] += args.zero_gain_jitter*direction
                # Use one SLSQP solve (with generated-constraint repetitions)
                # to scout. Retain actual nonlinear distances, not result.fun.
                y, report = ts.slsqp_once(seed, split.ids, numerical, args.ftol)
                try:
                    trial = ts.assess(ts.Geometry.from_float(bs.normalize(y)), pool,
                                      split.ids, witness_known=True)
                except ValueError as exc:
                    log('trial_rejected', round=round_no, trial=i+1, reason=str(exc))
                    continue
                meta = {'round': round_no, 'trial': i+1, **split.metadata(),
                        'slsqp': report, 'lower': trial.lower, 'upper': trial.upper,
                        'exact': trial.exact}
                log('trial', round=round_no, trial=i+1, vertex=split.vertex,
                    lower=trial.lower, upper=trial.upper, lp_gain=split.lp_gain)
                if args.save_trials:
                    ts.save_state(folder/f'trial_{i+1:03d}', trial, k, meta)
                best = checkpoint.best
                if trial.lower > best.lower+args.min_gain and trial.lower2 > best.upper2:
                    meta['geometric_split'] = split_diagnostics(trial, split.vertex, source.g.m, pool)
                    (folder/f'accepted_{i+1:03d}.json').write_text(json.dumps(meta, indent=2)+'\n')
                    trial.polish = {'status': 'scout_only', 'split': meta}
                    checkpoint.consider(trial, f'essential_split_round_{round_no}_trial_{i+1}')
                    improved = True
            if not improved:
                stop = 'no_improvement_in_selected_splits'
                break
            accepted_rounds += 1
            # Removal may simplify the new result but cannot lower its ratio.
            if not args.no_reduce:
                best = checkpoint.best
                reduced, _ = ts.reduce_vertices(best, pool, rng, best.g.m, 0, log)
                checkpoint.consider(reduced, f'after_split_round_{round_no}_reduction')
        # The final result always gets a separate, full-constraint repeated
        # polish plus a numerical first-order audit on its saved coordinates.
        numerical.full_constraints = True
        best = checkpoint.best
        final = ts.local_tune(best.g, best.witness, pool, rng, numerical,
                              log, smooth=False, colors=best.colors)
        checkpoint.consider(final, 'final_full_constraint_polish')
        best = checkpoint.best
        summary = {'version': VERSION, 'initial_points': initial.g.m,
                   'initial_lower': initial.lower, 'initial_upper': initial.upper,
                   'best_points': best.g.m, 'best_lower': best.lower, 'best_upper': best.upper,
                   'exact_ratio_squared': str(best.lower2) if best.exact else None,
                   'accepted_rounds': accepted_rounds, 'search_stop': stop,
                   'polish_status': (best.polish or {}).get('status', 'not_polished'),
                   'backend': pool.get(best.g.m).backend, 'oracle': pool.stats(),
                   'seconds': time.monotonic()-started,
                   'qualification': 'No global or exact local optimality claim. Zero LP gain is numerical and only concerns the tested split family.'}
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2))
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Completed best/history snapshots remain available.')
        return 130
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'Error: {exc}')
        return 2
    finally:
        if pool is not None:
            pool.close()


if __name__ == '__main__':
    raise SystemExit(main())
