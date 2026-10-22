"""Focused checks for the paired move, in addition to the existing regression suite."""
from pathlib import Path
from fractions import Fraction
import itertools
import json
import unittest
import numpy as np
import borsuk_search as bs
import tune_solution as ts
import split_tune as st
from replay_pair73 import exact_binary_geometry

class PairReplayTests(unittest.TestCase):
    def test_exact_binary_geometry(self):
        x = np.array([[.1, -.2, 1.], [.3, .4, .5]])
        g = exact_binary_geometry(x)
        for row, vals in zip(g.rows, x):
            for integer, value in zip(row, vals):
                self.assertEqual(Fraction(integer, 10**g.digits), Fraction(float(value)))
        np.testing.assert_array_equal(g.x, x)

    def test_second_split_matches_explicit_edges(self):
        rng = np.random.default_rng(733)
        for m in range(5, 10):
            e = bs.pairs_for(m)
            e = e[rng.random(len(e)) < .8]
            v = 0
            ns = sorted({int(b if a == v else a) for a, b in e if v in (a, b)})
            if len(ns) < 2:
                continue
            w1 = st.split_edge_ids(m, e, v, (ns[0],), (ns[1],))
            first = bs.pairs_for(m+1)[w1]
            v2 = 1
            ns2 = sorted({int(b if a == v2 else a) for a, b in first if v2 in (a,b)})
            if len(ns2) < 2:
                continue
            w2 = st.split_edge_ids(m+1, first, v2, (ns2[0],), (ns2[1],))
            expected = {tuple(map(int, ab)) for ab in first}
            expected.update(tuple(sorted((m+1,j))) for j in ns2)
            expected.remove(tuple(sorted((v2,ns2[0]))))
            expected.remove(tuple(sorted((m+1,ns2[1]))))
            self.assertEqual(set(map(tuple,bs.pairs_for(m+2)[w2])),expected)

    def test_actual_pair_scaffold_noncolorability_and_essential_copies(self):
        root = Path(__file__).resolve().parents[1]
        args=ts.build_parser().parse_args([str(root/'tests/fixtures/distinct70'), '--backend','python','--sat-timeout','20'])
        pool=ts.OraclePool(4,args)
        try:
            g,w,c,k=ts.load_source(root/'tests/fixtures/distinct70')
            s=ts.assess(g,pool,w,c)
            w1=st.split_edge_ids(70,g.pairs[g.threshold(s.lower2)],26,(45,),(64,))
            self.assertIs(pool.get(71).solve(w1).status,bs.Status.UNSAT)
            w2=st.split_edge_ids(71,bs.pairs_for(71)[w1],55,(26,),(15,56))
            oracle=pool.get(72)
            self.assertIs(oracle.solve(w2).status,bs.Status.UNSAT)
            for v in (26,70,55,71):
                deletion=w2[np.all(oracle.pairs[w2] != v,axis=1)]
                self.assertIs(oracle.solve(deletion).status,bs.Status.SAT)
        finally:
            pool.close()

    def test_input_file_unchanged(self):
        import hashlib
        from replay_pair73 import INPUT_SHA256
        p=Path(__file__).resolve().parents[1]/'tests/fixtures/original73.npy'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),INPUT_SHA256)

if __name__=='__main__':unittest.main()
