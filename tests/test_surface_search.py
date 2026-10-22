"""Regression tests for fixed-hull surface insertion. Run with unittest."""
import contextlib
import io
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from fractions import Fraction
import numpy as np
import borsuk_search as bs
import surface_search as ss

ROOT = Path(__file__).resolve().parents[1]
SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]


def square_cloud():
    f, _ = ss.find_facets(SQUARE, 1)
    return ss.Cloud(SQUARE, 1, 2**12, f)


def square_midpoints(cloud):
    for f in cloud.facets:
        cloud.append(ss.SurfacePoint(f, (cloud.Q//2, cloud.Q//2)))


def brute_ratio_squared(cloud, k):
    best = cloud.D
    for colors in itertools.product(range(k), repeat=len(cloud.rows)-1):
        best = min(best, ss.upper_from_colors(cloud, [0]+list(colors)))
    return Fraction(best, cloud.D)


class ExactGeometryTests(unittest.TestCase):
    def test_binary_input_is_not_decimalized(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'x.npy'
            np.save(p, np.array([[.1, 0.], [1., 0.], [0., 1.]]))
            rows, scale = ss.read_exact(p)
            self.assertEqual(Fraction(rows[0][0], scale), Fraction.from_float(.1))
            self.assertNotEqual(Fraction(rows[0][0], scale), Fraction('0.1'))

    def test_decimal_input_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'x.csv'
            p.write_text('.1,0\n1,0\n0,.125\n')
            rows, scale = ss.read_exact(p)
            self.assertEqual(Fraction(rows[0][0], scale), Fraction(1, 10))
            q = Path(td)/'y.csv'; ss.write_rows(q, rows, scale)
            rr, s = ss.read_exact(q)
            self.assertEqual([[Fraction(a, scale) for a in r] for r in rows],
                             [[Fraction(a, s) for a in r] for r in rr])

    def test_duplicate_input_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'x.csv'; p.write_text('0,0\n1,0\n0,0\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                ss.read_exact(p)

    def test_outward_square_root_rounding(self):
        for q in [Fraction(0), Fraction(1), Fraction(5, 8), Fraction(7, 99)]:
            lo, hi = map(Fraction, ss.sqrt_bounds(q))
            self.assertLessEqual(lo*lo, q); self.assertGreaterEqual(hi*hi, q)
            self.assertLessEqual(hi-lo, Fraction(1, 10**15))

    def test_finite_decimal_scale(self):
        self.assertEqual(ss.decimal_scale(2**24*10**17), 41)
        with self.assertRaises(ValueError):
            ss.decimal_scale(3)

    def test_bareiss_matches_permutation_determinant(self):
        rng = np.random.default_rng(2)
        for n in range(1, 5):
            for _ in range(8):
                a = rng.integers(-4, 5, size=(n, n)).tolist()
                exact = 0
                for p in itertools.permutations(range(n)):
                    inv = sum(p[i] > p[j] for i in range(n) for j in range(i+1, n))
                    exact += (-1)**inv*math.prod(a[i][p[i]] for i in range(n))
                self.assertEqual(ss.determinant(a), exact)

    def test_extreme_coordinate_scale_normalized_before_float(self):
        scale = 10**350
        rows = [(0, 0), (scale, 0), (scale, scale), (0, scale)]
        x = ss.normalized_rows(rows)
        self.assertTrue(np.isfinite(x).all())
        self.assertAlmostEqual(float(bs.distances2(x, bs.pairs_for(4)).max()), 1.)

    def test_nonsupporting_diagonal_rejected(self):
        self.assertIsNone(ss.supporting_plane(SQUARE, (0, 2)))
        self.assertIsNotNone(ss.supporting_plane(SQUARE, (0, 1)))

    def test_cube_coplanar_faces_triangulate_exactly(self):
        rows = list(itertools.product((0, 1), repeat=3))
        faces, report = ss.find_facets(rows, 1, 'exhaustive')
        self.assertEqual(len(faces), 12)
        self.assertIs(report['exhaustive_3d_fallback'], True)
        self.assertTrue(all(ss.supporting_plane(rows, f) for f in faces))

    def test_record58_exact_facets(self):
        rows, scale = ss.read_exact(ROOT/'best_known/record58')
        faces, report = ss.find_facets(rows, scale, 'exhaustive')
        self.assertEqual(len(rows), 58); self.assertEqual(len(faces), 112)
        self.assertTrue(all(ss.supporting_plane(rows, f) for f in faces))

    def test_full_dimension_required(self):
        with self.assertRaisesRegex(ValueError, 'full dimensionality'):
            ss.find_facets([(0, 0), (1, 0), (2, 0)], 1)

    def test_high_dimensional_simplex_facets(self):
        for d in (4, 5):
            rows = [tuple([0]*d)]+[tuple(int(i == j) for j in range(d)) for i in range(d)]
            faces, _ = ss.find_facets(rows, 1)
            self.assertEqual(len(faces), d+1)
            self.assertTrue(all(ss.supporting_plane(rows, f) for f in faces))

    def test_weights_nonnegative_and_sum_exactly(self):
        rng = np.random.default_rng(6)
        for d in range(2, 6):
            for _ in range(20):
                w = ss.quantize_weights(rng.normal(size=d), 2**24)
                self.assertEqual(sum(w), 2**24); self.assertGreaterEqual(min(w), 0)

    def test_source_points_and_diameter_preserved(self):
        c = square_cloud(); before = c.rows.copy()
        square_midpoints(c)
        self.assertEqual(c.rows[:4], before)
        self.assertEqual(max(c.q), c.D)
        self.assertEqual(len(c.rows), 8)
        self.assertFalse(c.append(c.specs[-1]))

    def test_invalid_barycentric_weights_rejected(self):
        c = square_cloud()
        with self.assertRaises(ValueError):
            c.append(ss.SurfacePoint((0, 1), (-1, c.Q+1)))


class ColoringTests(unittest.TestCase):
    def test_exact_bisection_against_all_partitions(self):
        rng = np.random.default_rng(23)
        for _ in range(8):
            c = square_cloud()
            for f in c.facets[:2]:
                a = int(rng.integers(1, c.Q))
                c.append(ss.SurfacePoint(f, (a, c.Q-a)))
            session = ss.Session(15, 2, 'python')
            try:
                b = ss.evaluate_exact(c, session)
                self.assertEqual(b.lower, b.upper)
                self.assertEqual(Fraction(b.lower, c.D), brute_ratio_squared(c, 2))
            finally:
                session.close()

    def test_square_midpoints_strictly_improve(self):
        c = square_cloud(); session = ss.Session(16, 2, 'python')
        try:
            initial = ss.evaluate_exact(c, session)
            square_midpoints(c)
            final = ss.evaluate_exact(c, session, initial)
            self.assertEqual(Fraction(initial.lower, c.D), Fraction(1, 2))
            self.assertEqual(Fraction(final.lower, c.D), Fraction(5, 8))
            self.assertEqual(final.lower, final.upper)
        finally:
            session.close()

    def test_strict_threshold_and_ties(self):
        c = square_cloud(); session = ss.Session(10, 2, 'python')
        try:
            status, _, _, edges = session.threshold(c, Fraction(1, 2))
            self.assertIs(status, bs.Status.SAT)
            self.assertEqual(len(edges), 2)
            status, _, _, _ = session.threshold(c, Fraction(1, 2)-Fraction(1, 10**20))
            self.assertIs(status, bs.Status.UNSAT)
        finally:
            session.close()

    def test_unused_capacity_does_not_change_coloring(self):
        c = square_cloud(); square_midpoints(c)
        out = []
        for capacity in (8, 31):
            session = ss.Session(capacity, 2, 'python')
            try:
                b = ss.evaluate_exact(c, session); out.append((b.lower, b.upper))
            finally:
                session.close()
        self.assertEqual(out[0], out[1])

    def test_unknown_retains_valid_bounds(self):
        c = square_cloud()
        class UnknownSession:
            k = 2
            def solve_edges(self, edges, n):
                return bs.Status.UNKNOWN, None, []
        b = ss.evaluate_exact(c, UnknownSession())
        actual = brute_ratio_squared(c, 2)*c.D
        self.assertTrue(b.unknown)
        self.assertLessEqual(b.lower, actual); self.assertGreaterEqual(b.upper, actual)

    def test_single_vertex_domain_and_pair_blocker(self):
        adj = [{1}, {0}]
        self.assertTrue(ss.list_colorable(adj, [1, 1], [0]))
        self.assertTrue(ss.list_colorable(adj, [1, 1], [1]))
        self.assertFalse(ss.list_colorable(adj, [1, 1], [0, 1]))
        core, _ = ss.blocking_core(adj, [1, 1], 30, .1)
        self.assertEqual(set(core), {0, 1})

    def test_cooperative_three_point_blocker(self):
        adj = [{1, 2}, {0, 2}, {0, 1}]
        for i in range(3):
            self.assertTrue(ss.list_colorable(adj, [3, 3, 3], [j for j in range(3) if j != i]))
        self.assertFalse(ss.list_colorable(adj, [3, 3, 3], [0, 1, 2]))

    def test_list_csp_unknown_is_not_unsat(self):
        self.assertIsNone(ss.list_colorable([set()], [3], [0], node_limit=0))

    def test_list_csp_matches_bruteforce(self):
        rng = np.random.default_rng(17)
        for n in range(1, 7):
            for _ in range(8):
                edges = [e for e in itertools.combinations(range(n), 2) if rng.random() < .45]
                adj = [set() for _ in range(n)]
                for i, j in edges:
                    adj[i].add(j); adj[j].add(i)
                domains = list(map(int, rng.integers(1, 8, size=n)))
                brute = any(all(domains[i] & (1 << col[i]) for i in range(n)) and
                            all(col[i] != col[j] for i, j in edges)
                            for col in itertools.product(range(3), repeat=n))
                self.assertEqual(ss.list_colorable(adj, domains, list(range(n)), timeout=1), brute)

    def test_kempe_bank_is_valid(self):
        edges = [(i, j) for i in range(9) for j in range(i+1, 9) if i % 3 != j % 3 and (i*j) % 4]
        bank = ss.coloring_bank([i % 3 for i in range(9)], edges, 3, 8, np.random.default_rng(7))
        for colors in bank:
            self.assertTrue(all(colors[i] != colors[j] for i, j in edges))

    def test_zero_gain_target_is_exact_upper(self):
        c = square_cloud(); b = ss.Bounds(c.D//2, c.D//2, [], None)
        self.assertEqual(ss.target_for(b, c, Fraction(0), None), Fraction(1, 2))
        self.assertGreater(ss.target_for(b, c, Fraction(1, 10**6), None), Fraction(1, 2))


class SearchTests(unittest.TestCase):
    def make_args(self, folder, rounds=1, extra=()):
        return ss.build_parser().parse_args(['search', str(ROOT/'best_known/square'), '--k', '2',
            '--rounds', str(rounds), '--batch', '4', '--pool', '80', '--refine', '2',
            '--block-pool', '24', '--polish-iters', '10', '--backend', 'python', '--out', str(folder)]+list(extra))

    def test_candidate_refinement_stays_on_face(self):
        c = square_cloud(); spec = ss.SurfacePoint(c.facets[0], (c.Q//4, 3*c.Q//4))
        refined = ss.refine_candidate(c, spec, c.array(), [0, 0, 1, 1], 2, 20)
        self.assertEqual(refined.vertices, spec.vertices)
        self.assertEqual(sum(refined.weights), c.Q)
        self.assertIsNotNone(ss.supporting_plane(c.base, refined.vertices))

    def test_search_checkpoint_and_independent_verification(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            out = Path(td)/'run'
            summary = ss.run_search(self.make_args(out))
            self.assertTrue(summary['strict_improvement_over_initial'])
            args = ss.build_parser().parse_args(['verify', str(out/'best'), '--backend', 'python'])
            result = ss.verify_surface(args.directory, args)
            self.assertTrue(result['exact_boundary_membership_of_all_added_points'])
            self.assertTrue(result['strict_improvement_over_initial'])
            # Old engine can independently verify the ordinary finite-set certificate.
            result2 = bs.verify_candidate(out/'best', 'python', 'kissat', 5, 'g4')
            self.assertTrue(result2['verified'])

    def test_resume_preserves_original_hull(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            first, second = Path(td)/'one', Path(td)/'two'
            ss.run_search(self.make_args(first))
            args = self.make_args(second)
            args.input = first/'latest'
            summary = ss.run_search(args)
            self.assertEqual(summary['original_points'], 4)
            self.assertGreater(summary['latest_points'], 8)
            original, old, _ = ss.load_start(first/'latest', 24, 'verified', 0)
            resumed, state, _ = ss.load_start(second/'latest', 24, 'verified', 0)
            self.assertEqual(resumed.base, original.base)
            self.assertEqual(state['initial_upper_squared'], old['initial_upper_squared'])

    def test_changed_k_on_resume_rejected(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            first, second = Path(td)/'one', Path(td)/'two'
            ss.run_search(self.make_args(first, 0))
            args = self.make_args(second); args.input = first/'latest'; args.k = 3
            with self.assertRaisesRegex(ValueError, 'same k'):
                ss.run_search(args)

    def test_tampered_coordinates_rejected(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            out = Path(td)/'run'; ss.run_search(self.make_args(out))
            p = out/'best/points.csv'
            lines = p.read_text().splitlines(); lines[-1] = '0.125,0.125'; p.write_text('\n'.join(lines)+'\n')
            with self.assertRaises(ValueError):
                ss.load_start(out/'best', 24, 'verified', 0)

    def test_tampered_supporting_face_rejected(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            out = Path(td)/'run'; ss.run_search(self.make_args(out))
            p = out/'best/surface.json'; data = json.loads(p.read_text())
            data['facets'].append([0, 2]); p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'supporting face'):
                ss.load_start(out/'best', 24, 'verified', 0)

    def test_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, 'already exists'):
                ss.run_search(self.make_args(Path(td)))

    def test_native_and_python_agree(self):
        if not (ROOT/'dsatur_native.so').exists():
            self.skipTest('native solver not compiled')
        c = square_cloud(); square_midpoints(c)
        values = []
        for backend in ('python', 'native'):
            session = ss.Session(20, 2, backend)
            try:
                b = ss.evaluate_exact(c, session); values.append((b.lower, b.upper))
            finally:
                session.close()
        self.assertEqual(values[0], values[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
