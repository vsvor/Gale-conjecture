import ctypes
from pathlib import Path
import unittest
import numpy as np
import borsuk_search as bs

ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless((ROOT/'dsatur_native.so').is_file(),'Build the optional native backend to run these checks.')
class NativeABI(unittest.TestCase):
    def test_random_graphs_match_python(self):
        rng=np.random.default_rng(20260911)
        for n in range(3,10):
            for k in (2,3,4):
                with bs.GraphOracle(n,k,'native',timeout=3) as fast,bs.GraphOracle(n,k,'python',timeout=3) as slow:
                    ids=np.flatnonzero(rng.random(len(fast.pairs))<.55)
                    a,b=fast.solve(ids),slow.solve(ids)
                    self.assertEqual(a.status,b.status)
                    self.assertIsNot(a.status,bs.Status.UNKNOWN)
    def test_invalid_input_status(self):
        with bs.GraphOracle(4,3,'native',timeout=3) as o:
            colors=np.full(4,-1,dtype=np.int32)
            self_loop=np.ascontiguousarray([[0,0]],dtype=np.int32)
            self.assertEqual(o.native(4,3,1,self_loop,1.,colors),-2)
            self.assertEqual(o.native(4,64,1,self_loop,1.,colors),-2)
    def test_duplicate_edges_and_peeling(self):
        with bs.GraphOracle(5,2,'native',timeout=3) as o:
            colors=np.full(5,-1,dtype=np.int32)
            edges=np.ascontiguousarray([[0,1],[0,1],[1,2],[2,3]],dtype=np.int32)
            self.assertEqual(o.native(5,2,len(edges),edges,3.,colors),1)
            self.assertTrue(np.all(colors>=0));self.assertTrue(np.all(colors<2))
            self.assertTrue(np.all(colors[edges[:,0]]!=colors[edges[:,1]]))

if __name__=='__main__':unittest.main()
