import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import simplex_product_seed as sp


class SimplexProductTests(unittest.TestCase):
    def test_simplex_gram(self):
        for d in range(2,7):
            v=sp.regular_simplex(d)
            target=np.full((d+1,d+1),-1/d);np.fill_diagonal(target,1.)
            np.testing.assert_allclose(v@v.T,target,atol=1e-14)
            np.testing.assert_allclose(v.sum(axis=0),0.,atol=1e-14)

    def test_five_dimensional_calibration(self):
        s,swap=sp.sample_shape(2,3,np.random.default_rng(1),0.,False)
        np.testing.assert_allclose([s.a,s.b,s.c,s.d,s.e,s.f],sp.FIT5,atol=1e-15)
        self.assertFalse(swap)
        self.assertAlmostEqual(s.hE,-.030998715104262525,places=13)
        self.assertAlmostEqual(s.rF,.5211198043298698,places=13)

    def test_product_site_counts(self):
        for dim in range(5,9):
            p,q=dim//2,dim-dim//2
            s,_=sp.sample_shape(p,q,np.random.default_rng(1),0.,False)
            x,t=sp.sites(s,sp.regular_simplex(p),sp.regular_simplex(q))
            self.assertEqual(x.shape,(3*(p+1)*(q+1)+dim+2,dim))
            self.assertEqual(len(t),len(x))
            self.assertEqual(len(np.unique(x,axis=0)),len(x))

    def test_endpoints_and_common_intersection(self):
        for p,q in [(2,3),(3,3),(3,4),(4,4)]:
            s,_=sp.sample_shape(p,q,np.random.default_rng(2),.01,False)
            u,v=sp.regular_simplex(p),sp.regular_simplex(q)
            for i in range(p+1):
                for j in range(q+1):
                    e=sp.meridian(s,u,v,i,j,[s.alpha0,0.,s.alpha1],'E')
                    f=sp.meridian(s,u,v,i,j,[s.beta0,0.,s.beta1],'F')
                    np.testing.assert_allclose(e[0],np.r_[s.a*u[i],-s.b*v[j]],atol=1e-13)
                    np.testing.assert_allclose(f[0],np.r_[-s.c*u[i],s.d*v[j]],atol=1e-13)
                    np.testing.assert_allclose(e[-1],f[-1],atol=1e-13)
                    np.testing.assert_allclose(e[-1],np.r_[s.e*u[i],s.f*v[j]],atol=1e-13)

    def test_meridian_sphere_identity(self):
        s,_=sp.sample_shape(3,4,np.random.default_rng(2),.01,False)
        u,v=sp.regular_simplex(3),sp.regular_simplex(4)
        x=sp.meridian(s,u,v,1,2,np.linspace(s.alpha0,s.alpha1,20),'E')
        np.testing.assert_allclose(np.sum((x-np.r_[s.hE*u[1],np.zeros(4)])**2,axis=1),s.rE**2,atol=1e-13)

    def test_generation_all_dimensions(self):
        for d,m in [(5,65),(6,75),(7,85),(8,95)]:
            x,meta=sp.generate(d,m,np.random.default_rng(42),pool_factor=8)
            self.assertEqual(x.shape,(m,d))
            self.assertAlmostEqual(sp.diameter(x),1.,places=12)
            self.assertEqual(np.linalg.matrix_rank(x),d)
            self.assertGreaterEqual(meta['minimum_separation_float'],.003-1e-12)
            self.assertLessEqual(meta['diameter_ratio_to_site_cloud'],1.02+1e-12)
            self.assertEqual(sum(meta['sample_counts'].values()),m)

    def test_small_budget(self):
        for d in range(5,9):
            x,info=sp.generate(d,d+2,np.random.default_rng(42),pool_factor=3)
            self.assertEqual(len(x),d+2)
            self.assertEqual(info['sample_counts']['site'],d+2)

    def test_partial_scaffold(self):
        x,m=sp.generate(6,80,np.random.default_rng(1),site_fraction=.6,pool_factor=8)
        self.assertEqual(len(x),80)
        self.assertEqual(m['sample_counts']['site'],math.ceil(.6*56))

    def test_repeatable(self):
        kw={'selection':'signature','pool_factor':4}
        a,ma=sp.generate(5,65,np.random.default_rng(3),**kw)
        b,mb=sp.generate(5,65,np.random.default_rng(3),**kw)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(ma,mb)

    def test_seed_changes_geometry(self):
        a=sp.make_seed(5,65,np.random.default_rng(1),pool_factor=4)
        b=sp.make_seed(5,65,np.random.default_rng(2),pool_factor=4)
        da=np.sort(np.sum((a[:,None]-a[None,:])**2,axis=2).ravel())
        db=np.sort(np.sum((b[:,None]-b[None,:])**2,axis=2).ravel())
        self.assertFalse(np.allclose(da,db))

    def test_role_swap_consistency(self):
        a,_=sp.sample_shape(2,3,np.random.default_rng(1),0.,False)
        b,_=sp.sample_shape(3,2,np.random.default_rng(1),0.,True)
        np.testing.assert_allclose([a.a,a.b,a.c,a.d,a.e,a.f],[b.d,b.c,b.b,b.a,b.f,b.e])
        self.assertAlmostEqual(a.rE,b.rF)

    def test_other_splits(self):
        for d,pq in [(6,(2,4)),(7,(2,5)),(8,(3,5)),(8,(2,6))]:
            x,meta=sp.generate(d,70,np.random.default_rng(2),split=pq,pool_factor=5)
            self.assertEqual(x.shape,(70,d)); self.assertEqual(meta['split'],list(pq))

    def test_invalid_input(self):
        for kw in [{'dim':4,'points':80},{'dim':6,'points':4},
                   {'dim':6,'points':70,'split':(2,3)},
                   {'dim':6,'points':70,'jitter':float('nan')},
                   {'dim':6,'points':70,'site_fraction':1.1}]:
            with self.assertRaises(ValueError):sp.generate(rng=np.random.default_rng(2),**kw)

    def test_spacing_failure_explicit(self):
        with self.assertRaises(ValueError):
            sp.generate(5,80,np.random.default_rng(1),min_separation=.8)

    def test_cli_roundtrip_no_clobber(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'seed'
            cmd=[sys.executable,str(Path(sp.__file__)), '--dim','5','--points','63',
                 '--pool-factor','4','--out',str(out)]
            ans=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
            self.assertEqual(ans.returncode,0,ans.stderr)
            x=np.load(out/'points.npy'); csv=np.loadtxt(out/'points.csv',delimiter=',')
            np.testing.assert_array_equal(x,csv)
            self.assertEqual(json.loads((out/'generator.json').read_text())['points'],63)
            second=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
            self.assertNotEqual(second.returncode,0)

if __name__=='__main__':
    unittest.main()
