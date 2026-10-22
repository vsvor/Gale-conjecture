"""Run with: python -m unittest -v test_borsuk_search.py"""
import itertools
import math
from pathlib import Path
import tempfile
import unittest
import numpy as np
import borsuk_search as b


def brute_coloring(m, k, edges):
    adj = [[] for _ in range(m)]
    for u, v in edges:
        adj[max(u, v)].append(min(u, v))
    colors = [-1]*m
    def rec(v, used):
        if v == m:
            return colors.copy()
        for c in range(min(k, used+1)):
            if any(colors[u] == c for u in adj[v]):
                continue
            colors[v] = c
            ans = rec(v+1, max(used, c+1))
            if ans is not None:
                return ans
        colors[v] = -1
        return None
    return rec(0, 0)


def brute_ratio(x, k):
    m = len(x)
    ds = np.sum((x[:, None]-x[None, :])**2, axis=2)
    colors = [-1]*m
    best = float(ds.max())
    def rec(v, used, cost):
        nonlocal best
        if v == m:
            best = min(best, cost)
            return
        for c in range(min(k, used+1)):
            new_cost = max([cost]+[float(ds[u,v]) for u in range(v) if colors[u] == c])
            if new_cost <= best:
                colors[v] = c
                rec(v+1, max(used,c+1),new_cost)
        colors[v] = -1
    rec(0,0,0.0)
    return math.sqrt(best/ds.max())


BACKENDS = ['python']
if Path(b.__file__).with_name('dsatur_native.so').exists():
    BACKENDS.append('native')
try:
    import pysat
    BACKENDS.append('pysat')
except ImportError:
    pass


class GraphTests(unittest.TestCase):
    def test_all_graphs_on_four_vertices(self):
        pairs = b.pairs_for(4)
        for backend in BACKENDS:
            for k in range(1,5):
                with b.GraphOracle(4,k,backend=backend) as oracle:
                    for mask in range(1<<len(pairs)):
                        ids=[i for i in range(len(pairs)) if mask & (1<<i)]
                        expected=brute_coloring(4,k,pairs[ids]) is not None
                        ans=oracle.solve(ids)
                        self.assertEqual(ans.status, b.Status.SAT if expected else b.Status.UNSAT)

    def test_random_graphs(self):
        rng=np.random.default_rng(404)
        for _ in range(200):
            m=int(rng.integers(5,10)); k=int(rng.integers(1,6)); pairs=b.pairs_for(m)
            ids=np.flatnonzero(rng.random(len(pairs)) < rng.uniform(.1,.95))
            expected=brute_coloring(m,k,pairs[ids]) is not None
            for backend in BACKENDS:
                with b.GraphOracle(m,k,backend=backend) as oracle:
                    ans=oracle.solve(ids)
                    self.assertEqual(ans.status, b.Status.SAT if expected else b.Status.UNSAT)

    def test_cnf_encoding_exhaustive(self):
        m,k=4,2; pairs=b.pairs_for(m)
        for mask in range(1<<len(pairs)):
            edges=pairs[[i for i in range(len(pairs)) if mask & (1<<i)]]
            cnf=b.graph_cnf(m,k,edges)
            expected=brute_coloring(m,k,edges) is not None
            sat=False
            for bits in range(1<<(m*k)):
                if all(any(bool(bits & (1 << (abs(lit)-1))) == (lit>0) for lit in clause) for clause in cnf):
                    sat=True;break
            self.assertEqual(sat,expected)

    def test_external_parser(self):
        ans=b.parse_external('s SATISFIABLE\nv 1 -2 -3 4 0\n',10,2,2)
        self.assertIs(ans.status,b.Status.SAT)
        self.assertEqual(ans.colors.tolist(),[0,1])
        self.assertIs(b.parse_external('s UNSATISFIABLE\n',20,2,2).status,b.Status.UNSAT)
        self.assertIs(b.parse_external('UNSATISFIABLE\n',0,2,2).status,b.Status.UNSAT)
        self.assertIs(b.parse_external('SAT\n1 -2 -3 4 0\n',10,2,2).status,b.Status.SAT)
        self.assertIs(b.parse_external('s SATISFIABLE',20,2,2).status,b.Status.UNKNOWN)
        self.assertIs(b.parse_external('',0,2,2).status,b.Status.UNKNOWN)
        self.assertIs(b.parse_external('s UNSATISFIABLE',-9,2,2).status,b.Status.UNKNOWN)

    def test_external_subprocess_adapter(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            script=Path(tmp)/'fake_solver.py'
            script.write_text("import sys\nassert 'p cnf' in open(sys.argv[1]).read()\nprint('s UNSATISFIABLE')\nsys.exit(20)\n")
            command=f'{sys.executable} {script}'
            with b.GraphOracle(3,2,backend='external',solver=command) as oracle:
                self.assertIs(oracle.solve([0,1,2]).status,b.Status.UNSAT)

    def test_external_timeout(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            script=Path(tmp)/'sleep.py';script.write_text('import time\ntime.sleep(2)\n')
            with b.GraphOracle(3,2,backend='external',solver=f'{sys.executable} {script}',timeout=.02) as oracle:
                self.assertIs(oracle.solve([0,1,2]).status,b.Status.UNKNOWN)

    def test_pruning_preserves_unsat(self):
        rng=np.random.default_rng(5)
        with b.GraphOracle(8,3,backend=BACKENDS[-1]) as oracle:
            ids=np.arange(len(oracle.pairs))
            out=b.prune_witness(oracle,ids,np.arange(1.,len(ids)+1),rng,1000)
            self.assertIsNone(brute_coloring(8,3,oracle.pairs[out]))
            for e in out:
                self.assertIsNotNone(brute_coloring(8,3,oracle.pairs[out[out!=e]]))


class ObjectiveTests(unittest.TestCase):
    def test_against_all_partitions(self):
        rng=np.random.default_rng(1234)
        for m in range(3,9):
            for k in range(1,min(m+1,5)):
                for _ in range(3):
                    x=rng.normal(size=(m,3))
                    expected=brute_ratio(x,k)
                    for backend in BACKENDS:
                        with b.GraphOracle(m,k,backend=backend) as oracle:
                            ev=b.evaluate(x,oracle)
                        self.assertTrue(ev.exact)
                        self.assertAlmostEqual(ev.lower,expected,places=12)
                        self.assertAlmostEqual(ev.upper,expected,places=12)

    def test_tied_distances_and_degenerate_clusters(self):
        cases=[(np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]]),2,1/math.sqrt(2)),
               (b.regular_simplex(3),3,1.0),
               (b.regular_simplex(3),4,0.0),
               (np.array([[0.],[1.],[3.]]),2,1/3),
               (np.array([[0.,0.],[0.,0.],[1.,0.],[1.,0.]]),2,0.)]
        for x,k,expected in cases:
            with b.GraphOracle(len(x),k,backend=BACKENDS[-1]) as oracle:
                ev=b.evaluate(x,oracle)
                self.assertAlmostEqual(ev.lower,expected,places=12)
                self.assertAlmostEqual(ev.upper,expected,places=12)

    def test_unknown_is_a_bracket(self):
        class Unknown:
            m,k,pairs=6,3,b.pairs_for(6)
            def solve(self,ids):return b.Answer(b.Status.UNKNOWN)
        x=np.random.default_rng(12).normal(size=(6,2))
        ev=b.evaluate(x,Unknown())
        true=brute_ratio(x,3)
        self.assertTrue(ev.unknown)
        self.assertFalse(ev.exact)
        self.assertLessEqual(ev.lower,true)
        self.assertGreaterEqual(ev.upper,true)

    def test_zero_diameter_rejected(self):
        with b.GraphOracle(4,2,backend='python') as oracle:
            with self.assertRaises(ValueError):b.evaluate(np.zeros((4,3)),oracle)


class GeometryTests(unittest.TestCase):
    def test_simplex_and_seeds(self):
        rng=np.random.default_rng(89)
        for d in (2,3,4,5):
            v=b.regular_simplex(d)
            self.assertTrue(np.allclose(v@v.T,np.eye(d+1)*(1+1/d)-np.ones((d+1,d+1))/d))
            for seed in b.SEEDS:
                x=b.make_seed(d,40,seed,rng)
                self.assertEqual(x.shape,(40,d))
                self.assertTrue(np.isfinite(x).all())
                if seed in ('simplex-edges','simplex-faces','prism-sphere','sphere'):
                    self.assertTrue(np.allclose(np.linalg.norm(x,axis=1),1))

    def test_analytic_gradient(self):
        rng=np.random.default_rng(71);x=rng.normal(size=(7,3));pairs=b.pairs_for(7)
        ids=np.array([0,2,5,7,12,18])
        for sphere in (False,True):
            f,g=b.smooth_value_grad(x.ravel(),x.shape,pairs,ids,12.,sphere)
            numeric=[];h=1e-6
            for j in range(x.size):
                z=x.ravel().copy();z[j]+=h
                plus=b.smooth_value_grad(z,x.shape,pairs,ids,12.,sphere)[0]
                z[j]-=2*h
                minus=b.smooth_value_grad(z,x.shape,pairs,ids,12.,sphere)[0]
                numeric.append((plus-minus)/(2*h))
            self.assertTrue(np.allclose(g,numeric,rtol=2e-5,atol=1e-7))

    def test_optimizer_never_loses_witness_bound(self):
        rng=np.random.default_rng(78);x=rng.normal(size=(8,3));pairs=b.pairs_for(8)
        ids=np.arange(len(pairs))
        for sphere in (False,True):
            start=b.normalize(x,sphere)
            r0=b.witness_ratio(start,pairs,ids)
            y,r=b.optimize_witness(start,pairs,ids,steps=15,polish=8,sphere=sphere,powers=(8.,40.))
            self.assertGreaterEqual(r+1e-12,r0)
            self.assertAlmostEqual(r,b.witness_ratio(y,pairs,ids),places=10)

    def test_replenishing_cannot_hurt_witness(self):
        x=b.normalize(np.random.default_rng(18).normal(size=(9,3)))
        pairs=b.pairs_for(9);ids=np.flatnonzero((pairs<5).all(axis=1))
        y=b.replenish_unused(x,pairs,ids,np.random.default_rng(19))
        self.assertTrue(np.allclose(x[:5],y[:5]))
        self.assertLessEqual(b.distances2(y,pairs).max(),b.distances2(x,pairs).max()+1e-12)

    def test_exact_export_and_verification(self):
        x=np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.]])
        with tempfile.TemporaryDirectory() as tmp:
            with b.GraphOracle(4,2,backend=BACKENDS[-1]) as oracle:
                ev=b.evaluate(x,oracle)
                b.save_candidate(Path(tmp),x,2,ev,oracle.pairs,{'test':True})
            result=b.verify_candidate(Path(tmp),BACKENDS[-1],'kissat',10.,'g4')
            self.assertTrue(result['verified'])
            self.assertLessEqual(float(result['lower_decimal']),1/math.sqrt(2))
            self.assertGreaterEqual(float(result['upper_decimal']),1/math.sqrt(2))

    def test_integer_rounding_and_distinctness(self):
        rows,adjusted=b.decimal_integer_points(np.array([[0.,0.],[0.,0.],[1.,0.]]))
        self.assertEqual(adjusted,1)
        self.assertEqual(len(set(map(tuple,rows))),3)
        data=b.rational_geometry(rows,np.array([[0,2],[1,2]]),np.array([0,1,0]))
        a=int(data['min_witness_squared_integer']);den=int(data['diameter_squared_integer'])
        low=int(data['lower_decimal'].replace('.',''));scale=10**12
        self.assertLessEqual(low**2*den,a*scale**2)
        self.assertGreater((low+1)**2*den,a*scale**2)

if __name__=='__main__':
    unittest.main()
