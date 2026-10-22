"""Regression tests for structural tuning; run python -m unittest -v test_tune_solution."""
import itertools
import io
import json
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import borsuk_search as bs
import tune_solution as t


def arguments(*extra):
    a = t.build_parser().parse_args(['unused', '--backend', 'python',
        '--sat-timeout', '3', '--polish-kicks', '0', '--slsqp-iters', '80',
        '--polish-passes', '5', '--refreshes', '1', '--steps', '4', *extra])
    t.validate_args(a)
    return a


def brute_ratio2(g, k):
    best = g.diameter2
    for tail in itertools.product(range(k), repeat=g.m-1):
        c = np.asarray((0,)+tail)
        best = min(best, g.partition_max(c))
    return Fraction(best, g.diameter2)


class UnknownOracle:
    def __init__(self, m, k):
        self.m, self.k, self.pairs = m, k, bs.pairs_for(m)
    def solve(self, _):
        return bs.Answer(bs.Status.UNKNOWN, reason='simulated timeout')


class UnknownPool:
    def __init__(self, k):
        self.k = k
    def get(self, m):
        return UnknownOracle(m, self.k)


class GeometryTests(unittest.TestCase):
    def test_edge_index_mapping(self):
        for m in range(2, 18):
            e = bs.pairs_for(m)
            np.testing.assert_array_equal(t.edge_ids(m, e), np.arange(len(e)))
            np.testing.assert_array_equal(t.edge_ids(m, e[:, ::-1]), np.arange(len(e)))

    def test_exact_decimal_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'points.csv'
            p.write_text('0.00000000000000000001,0\n1,0\n0,1\n')
            g, _, _, _ = t.load_source(p)
            self.assertEqual(g.digits, 20)
            self.assertEqual(g.rows[0][0], 1)
            self.assertEqual(g.rows[1][0], 10**20)

    def test_tied_distance_evaluation(self):
        g = t.Geometry([[0, 0], [1, 0], [0, 1], [1, 1]], 0)
        p = t.OraclePool(2, arguments())
        try:
            s = t.assess(g, p)
            self.assertTrue(s.exact)
            self.assertEqual(s.lower2, Fraction(1, 2))
        finally:
            p.close()

    def test_integer_evaluator_against_bruteforce(self):
        rng = np.random.default_rng(99)
        for m, k in ((5, 2), (6, 2), (7, 3), (8, 3)):
            rows = rng.choice(100, (m, 2), replace=False).tolist()
            g = t.Geometry(rows, 0)
            p = t.OraclePool(k, arguments())
            try:
                s = t.assess(g, p)
                self.assertTrue(s.exact)
                self.assertEqual(s.lower2, brute_ratio2(g, k))
            finally:
                p.close()

    def test_threshold_inclusion_is_exact(self):
        g = t.Geometry([[0, 0], [1, 0], [0, 1]], 0)
        np.testing.assert_array_equal(g.threshold(Fraction(1, 2)), [0, 1, 2])
        np.testing.assert_array_equal(g.threshold(Fraction(1, 2), False), [2])

    def test_unknown_keeps_bounds(self):
        g = t.Geometry([[0, 0], [1, 0], [0, 3], [3, 3]], 0)
        s = t.assess(g, UnknownPool(2))
        self.assertTrue(s.unresolved)
        truth = brute_ratio2(g, 2)
        self.assertLessEqual(s.lower2, truth)
        self.assertGreaterEqual(s.upper2, truth)

    def test_duplicate_coordinates_rejected(self):
        with self.assertRaises(ValueError):
            t.Geometry([[0, 0], [0, 0], [1, 0]], 0)


class DeletionTests(unittest.TestCase):
    def test_deletion_preserves_ratio_exactly(self):
        # Four corners plus a point on the bottom side. Removing it is safe.
        g = t.Geometry([[0, 0], [4, 0], [0, 4], [4, 4], [2, 0]], 0)
        p = t.OraclePool(2, arguments())
        try:
            s = t.assess(g, p)
            reduced, _ = t.reduce_vertices(s, p, np.random.default_rng(1), 5, 3)
            self.assertLess(reduced.g.m, g.m)
            self.assertGreaterEqual(reduced.lower2, s.upper2)
            self.assertEqual(reduced.lower2, brute_ratio2(reduced.g, 2))
        finally:
            p.close()

    def test_unknown_never_deletes(self):
        g = t.Geometry([[0, 0], [4, 0], [0, 4], [4, 4], [2, 0]], 0)
        p = t.OraclePool(2, arguments())
        try:
            s = t.assess(g, p)
        finally:
            p.close()
        reduced, report = t.reduce_vertices(s, UnknownPool(2), np.random.default_rng(1), 5, 3)
        self.assertEqual(reduced.g.rows, g.rows)
        self.assertEqual(report['unknowns'], 5)

    def test_subset_diameter_is_recomputed(self):
        g = t.Geometry([[0, 0], [10, 0], [0, 1], [1, 1]], 0)
        sub = g.subset(np.array([0, 2, 3]))
        self.assertEqual(sub.diameter2, 2)
        self.assertGreater(g.diameter2, sub.diameter2)


class InsertionTests(unittest.TestCase):
    def setUp(self):
        self.g = t.Geometry([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], 0)
        self.cycle = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])
        self.ids = t.edge_ids(5, self.cycle)

    def test_essential_single_insertion(self):
        p = t.OraclePool(2, arguments())
        try:
            w, reason = t.essential_extension(np.array([0, 1, 0, 1]), self.g, self.ids, p)
            self.assertIsNotNone(w)
            self.assertEqual(reason, 'essential_UNSAT_extension')
        finally:
            p.close()

    def test_cooperative_pair_insertion(self):
        colors = np.array([0, 1, 0])
        A = t.adjacency(5, self.cycle)
        self.assertFalse(t.coloring_extends(colors, A, 2))
        self.assertIsNone(t.redundant_new_vertex(A, A, 3, 2))
        p = t.OraclePool(2, arguments())
        try:
            w, _ = t.essential_extension(colors, self.g, self.ids, p)
            self.assertIsNotNone(w)
        finally:
            p.close()

    def test_noncolorable_old_skeleton_is_rejected(self):
        edges = np.array([[0, 1], [1, 2], [0, 2], [2, 4]])
        p = t.OraclePool(2, arguments())
        try:
            with self.assertRaises(ValueError):
                t.essential_extension(np.array([0, 1, 0, 1]), self.g, t.edge_ids(5, edges), p)
        finally:
            p.close()

    def test_false_twin_filtered(self):
        edges = np.array([[0, 1], [1, 2], [0, 3], [2, 3]])
        A = t.adjacency(4, edges)
        self.assertEqual(t.redundant_new_vertex(A, A, 3, 2), 'geometric_false_twin')

    def test_true_twin_filtered_as_preference(self):
        A = t.adjacency(3, bs.pairs_for(3))
        self.assertEqual(t.redundant_new_vertex(A, A, 2, 2), 'geometric_true_twin')

    def test_dominated_neighborhood_filtered(self):
        A = t.adjacency(5, np.array([[1, 0], [1, 2], [1, 3], [4, 0], [4, 2]]))
        self.assertEqual(t.redundant_new_vertex(A, A, 4, 2), 'nonadjacent_dominated_neighbourhood')

    def test_list_extension(self):
        A = t.adjacency(5, np.array([[0, 1], [1, 2], [2, 3], [0, 4]]))
        self.assertTrue(t.coloring_extends(np.array([0, 1, 0]), A, 2))

    def test_unknown_does_not_certify_insertion(self):
        w, reason = t.essential_extension(np.array([0, 1, 0, 1]), self.g, self.ids, UnknownPool(2))
        self.assertIsNone(w)
        self.assertEqual(reason, 'UNKNOWN')

    def test_completion_candidates_inside_all_balls(self):
        x = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        pts = t.completion_candidates(x, 300, np.random.default_rng(77))
        q = np.sum((pts[:, None]-x[None])**2, axis=2)
        self.assertLessEqual(q.max(), 2+1e-12)
        # The completion body is genuinely larger than the convex hull.
        self.assertTrue(np.any((pts < 0) | (pts > 1)))

    def test_insertion_gradient_refinement_feasible(self):
        x = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        colors = np.array([0, 1, 0, 1])
        start = np.array([0.5, 1.2])
        out = t.refine_insert_point(start, x, colors, 2, 30)
        self.assertLessEqual(np.sum((x-out)**2, axis=1).max(), 2+1e-12)
        self.assertGreaterEqual(t.coverage_score(out, x, colors, 2),
                                t.coverage_score(start, x, colors, 2)-1e-12)


class OptimizerTests(unittest.TestCase):
    def test_constraint_jacobian(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(6, 3))
        p = t.ConstrainedProblem(x, np.array([0, 2, 4, 7, 10]), True)
        z = p.pack(x)
        J = p.jac(z)
        for j in range(len(z)):
            v = np.zeros(len(z)); v[j] = 1e-6
            approx = (p.cons(z+v)-p.cons(z-v))/2e-6
            np.testing.assert_allclose(J[:, j], approx, atol=2e-9, rtol=2e-7)

    def test_audit_regular_simplex(self):
        x = np.array([[0., 0.], [1., 0.], [.5, np.sqrt(3)/2]])
        report, _ = t.linearized_audit(x, np.arange(3))
        self.assertTrue(report['success'])
        self.assertLess(report['linearized_gain'], 1e-8)

    def test_audit_detects_improving_direction(self):
        x = np.array([[0., 0.], [1., 0.], [.3, .6]])
        report, direction = t.linearized_audit(x, np.arange(3))
        self.assertGreater(report['linearized_gain'], 1e-3)
        self.assertIsNotNone(direction)

    def test_slsqp_improves_triangle(self):
        x = np.array([[0., 0.], [1., 0.], [.3, .6]])
        y, report = t.polish_fixed(x, np.arange(3), np.random.default_rng(1), arguments())
        self.assertGreater(bs.witness_ratio(y, bs.pairs_for(3), np.arange(3)), .99999999)
        self.assertEqual(report['status'], 'numerically_stationary')

    def test_polish_budget_is_not_success(self):
        x = np.array([[0., 0.], [1., 0.], [.3, .6]])
        a = arguments('--polish-passes', '1', '--slsqp-iters', '1')
        _, report = t.polish_fixed(x, np.arange(3), np.random.default_rng(1), a)
        self.assertEqual(report['status'], 'polish_budget_exhausted')


class ExportTests(unittest.TestCase):
    def test_export_verified_by_original_engine(self):
        g = t.Geometry([[0, 0], [1, 0], [0, 1], [1, 1]], 0)
        p = t.OraclePool(2, arguments())
        try:
            s = t.assess(g, p)
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                t.save_state(directory, s, 2, {'test': True})
                data = bs.verify_candidate(directory, 'python', '', 3, 'g4')
                self.assertTrue(data['verified'])
                loaded, w, colors, k = t.load_source(directory)
                self.assertEqual(loaded.rows, g.rows)
                self.assertEqual(k, 2)
                self.assertEqual(len(w), len(s.witness))
        finally:
            p.close()

    def test_checkpoint_rejects_weaker_candidate(self):
        p = t.OraclePool(2, arguments())
        try:
            a = t.assess(t.Geometry([[0, 0], [1, 0], [0, 1], [1, 1]], 0), p)
            b = t.assess(t.Geometry([[0, 0], [1, 0], [100, 0], [101, 0]], 0), p)
            with tempfile.TemporaryDirectory() as tmp:
                saver = t.Checkpoints(Path(tmp), 2, lambda *a, **k: None)
                self.assertTrue(saver.consider(a, 'first'))
                self.assertFalse(saver.consider(b, 'weaker'))
                self.assertIs(saver.best, a)
        finally:
            p.close()

    def test_original_49_point_reduction(self):
        source = Path(__file__).resolve().parents[1]/'best_known'/'d3_m49'
        p = t.OraclePool(4, arguments())
        try:
            g, w, c, _ = t.load_source(source)
            s = t.assess(g, p, w, c)
            reduced, report = t.reduce_vertices(s, p, np.random.default_rng(7), 100, 5)
            self.assertEqual(reduced.g.m, 23)
            self.assertEqual(reduced.lower2, s.upper2)
            self.assertEqual(len(report['deleted_original_indices']), 26)
        finally:
            p.close()


class IntegrationTests(unittest.TestCase):
    def test_geometric_insertion_proposals(self):
        source = Path(__file__).resolve().parents[1]/'best_known'/'d3_m23'
        a = arguments('--candidates', '200', '--refine-candidates', '4',
                      '--trials', '1', '--candidate-attempts', '500',
                      '--slacks', '0.15', '--edge-prune', '10', '--add', '1:2')
        a.max_points = 40
        pool = t.OraclePool(4, a)
        try:
            g, w, c, _ = t.load_source(source)
            state = t.assess(g, pool, w, c)
            proposals = t.insertion_proposals(state, pool, np.random.default_rng(6), a,
                                              lambda *a, **k: None)
            self.assertGreater(len(proposals), 0)
            for proposal in proposals:
                self.assertEqual(pool.get(proposal.g.m).solve(proposal.ids).status, bs.Status.UNSAT)
                self.assertIn('every new-point deletion SAT', proposal.meta['screen'])
        finally:
            pool.close()

    def test_full_cycle_checkpoint_monotonicity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root/'start.npy', np.random.default_rng(127).normal(size=(12, 3)))
            out = root/'output'
            with patch('sys.stdout', new=io.StringIO()):
                code = t.main([str(root/'start.npy'), '--out', str(out), '--k', '4',
                    '--backend', 'python', '--cycles', '1', '--trials', '1',
                    '--steps', '5', '--edge-prune', '6', '--candidates', '60',
                    '--refine-candidates', '2', '--candidate-attempts', '100',
                    '--polish-kicks', '0', '--polish-passes', '4',
                    '--slsqp-iters', '70', '--refreshes', '1', '--quiet'])
            self.assertEqual(code, 0)
            previous_upper = None
            for directory in sorted((out/'history').iterdir()):
                meta = json.loads((directory/'certificate.json').read_text())
                geo = meta['geometry']
                lower = Fraction(int(geo['min_witness_squared_integer']), int(geo['diameter_squared_integer']))
                upper = Fraction(int(geo['max_part_squared_integer']), int(geo['diameter_squared_integer']))
                if previous_upper is not None:
                    self.assertGreaterEqual(lower, previous_upper)
                previous_upper = upper
            self.assertTrue(bs.verify_candidate(out/'best', 'python', '', 3, 'g4')['verified'])

    def test_constraint_generation_matches_full_problem(self):
        x = np.random.default_rng(23).normal(size=(8, 3))
        ids = t.edge_ids(8, bs.pairs_for(5))
        working, a = t.slsqp_once(x, ids, arguments('--slsqp-iters', '150'), 1e-12)
        full, b = t.slsqp_once(x, ids, arguments('--full-constraints', '--slsqp-iters', '150'), 1e-12)
        pairs = bs.pairs_for(8)
        self.assertTrue(a['all_diameter_constraints_checked'])
        self.assertTrue(b['all_diameter_constraints_checked'])
        self.assertGreater(bs.witness_ratio(working, pairs, ids), .7637)
        self.assertGreater(bs.witness_ratio(full, pairs, ids), .7637)


if __name__ == '__main__':
    unittest.main(verbosity=2)
