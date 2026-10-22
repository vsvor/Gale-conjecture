import argparse
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import borsuk_search as bs
import tune_solution as ts
import split_tune as sp


def params(*more):
    return sp.parser().parse_args(['unused', '--backend', 'python', '--sat-timeout', '5', *more])


class SplitTests(unittest.TestCase):
    def test_split_edge_mapping(self):
        # Wheel over a five-cycle: center 0, cycle vertices 1..5.
        edges = np.array([(0, i) for i in range(1, 6)] +
                         [(1+i, 1+(i+1)%5) for i in range(5)], np.int32)
        ids = sp.split_edge_ids(6, edges, 0, (1,), (2,))
        pairs = bs.pairs_for(7)
        es = set(map(tuple, pairs[ids]))
        self.assertNotIn((0, 1), es)
        self.assertNotIn((2, 6), es)
        self.assertNotIn((0, 6), es)
        self.assertIn((0, 2), es)
        self.assertIn((1, 6), es)
        self.assertEqual(len(es), 13)

    def test_essential_noncolorable_split(self):
        edges = np.array([(0, i) for i in range(1, 6)] +
                         [(1+i, 1+(i+1)%5) for i in range(5)], np.int32)
        ids = sp.split_edge_ids(6, edges, 0, (1,), (2,))
        with bs.GraphOracle(7, 3, 'python', timeout=5) as oracle:
            self.assertIs(oracle.solve(ids).status, bs.Status.UNSAT)
            for vertex in (0, 6):
                kept = ids[np.all(oracle.pairs[ids] != vertex, axis=1)]
                self.assertIs(oracle.solve(kept).status, bs.Status.SAT)
            A = ts.adjacency(7, oracle.pairs[ids])
            self.assertFalse(np.array_equal(A[0, 1:6], A[6, 1:6]))

    def test_colorable_split_must_be_screened_out(self):
        ids = sp.split_edge_ids(5, bs.pairs_for(5), 0, (1,), (2,))
        with bs.GraphOracle(6, 4, 'python', timeout=5) as oracle:
            self.assertIs(oracle.solve(ids).status, bs.Status.SAT)

    def test_invalid_split(self):
        edges = bs.pairs_for(5)
        for a, b in (((), (1,)), ((1,), (1,)), ((6,), (2,))):
            with self.assertRaises(ValueError):
                sp.split_edge_ids(5, edges, 0, a, b)

    def test_group_pairs_disjoint_and_symmetric_duplicates_removed(self):
        pairs = list(sp.omission_pairs([1, 2, 3, 4], {1, 3}, 2))
        seen = set()
        for a, b in pairs:
            self.assertFalse(set(a) & set(b))
            self.assertTrue(set(a) & {1, 3})
            self.assertTrue(set(b) & {1, 3})
            key = frozenset((a, b))
            self.assertNotIn(key, seen)
            seen.add(key)
        self.assertIn(((1, 2), (3, 4)), pairs)

    def test_unknown_is_not_critical(self):
        class Unknown:
            def solve(self, ids):
                return bs.Answer(bs.Status.UNKNOWN)
        class Pool:
            def get(self, m):
                return Unknown()
        g = ts.Geometry([[0, 0], [1, 0], [0, 1]], 0)
        s = ts.make_state(g, np.arange(3), np.array([0, 0, 1]))
        splits, stats = sp.discover_splits(s, Pool(), params(), lambda *a, **kw: None)
        self.assertEqual(splits, [])
        self.assertEqual(stats['critical_near_shortest_edges'], 0)
        self.assertEqual(stats['edge_deletion_UNKNOWN'], 2)

    def test_lp_filter_and_vertex_diversity(self):
        ids = np.array([0, 1], np.int32)
        splits = [sp.Split(0, (1,), (2,), ids, float(10-i), {}) for i in range(5)]
        splits += [sp.Split(1, (0,), (2,), ids, 1., {}),
                   sp.Split(2, (0,), (1,), ids, 0., {})]
        selected = sp.select_splits(splits, params('--per-vertex', '1', '--trials', '4'))
        self.assertEqual([s.vertex for s in selected], [0, 1])
        selected = sp.select_splits(splits, params('--per-vertex', '1', '--zero-gain-trials', '1'))
        self.assertEqual([s.vertex for s in selected], [0, 1, 2])

    def test_actual_user_configuration_split(self):
        path = Path(__file__).resolve().parents[1]/'best_known'/'d3_m46'
        args = ts.build_parser().parse_args([str(path), '--backend', 'python',
                                           '--sat-timeout', '10', '--slsqp-iters', '200'])
        pool = ts.OraclePool(4, args)
        try:
            g, w, c, k = ts.load_source(path)
            before = ts.assess(g, pool, w, c)
            ids = sp.split_edge_ids(46, g.pairs[g.threshold(before.lower2)], 44, (25,), (30,))
            self.assertIs(pool.get(47).solve(ids).status, bs.Status.UNSAT)
            x = bs.normalize(g.x)
            seed = np.vstack((x, x[44]))
            audit, _ = ts.linearized_audit(seed, ids)
            self.assertGreater(audit['linearized_gain'], .01)
            y, report = ts.slsqp_once(seed, ids, args, 1e-12)
            after = ts.assess(ts.Geometry.from_float(y), pool, ids, witness_known=True)
            self.assertTrue(after.exact)
            self.assertGreater(after.lower2, before.upper2)
            self.assertGreater(after.lower, .8852)
            self.assertGreater(np.linalg.norm(y[44]-y[46]), .01)
            d = sp.split_diagnostics(after, 44, 46, pool)
            self.assertFalse(d['twins_at_saved_threshold'])
            self.assertEqual(set(d['delete_copy_vertex_status'].values()), {'SAT'})
        finally:
            pool.close()


if __name__ == '__main__':
    unittest.main()
