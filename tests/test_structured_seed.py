import json, math, unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
import compare_curves as cc
import structured_seed as ss

R=Path(__file__).resolve().parents[1]

class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=np.load(R/'templates/source3.npy')
        cls.template=json.loads((R/'templates/curves3.json').read_text())
        cls.comparison=json.loads((R/'templates/curves3_comparison.json').read_text())

    def test_frame(self):
        B=np.array(self.template['frame']['basis_columns_a_b_n'])
        assert_allclose(B.T@B,np.eye(3),atol=1e-14)

    def test_core_definition(self):
        c=self.comparison['fits']['minus_near_apex']
        self.assertEqual(len(c['indices']),10)
        self.assertEqual(c['excluded'],[9,12,16,44,50,51,52,53,54,56])
        perm=self.template['frame']['mirror_permutation']
        self.assertEqual({perm[i] for i in c['indices']},set(c['indices']))

    def test_projection_on_exact_curves(self):
        for sign in [1,-1]:
            for side in [False,True]:
                p=np.array([.06,1.,-.1,np.log(2.)]) if side else np.array([.03,np.log(2.)])
                d,n=cc.basis(p,sign,side)
                for kind in ['circle','parabola']:
                    k=np.exp(p[-1]);t=np.linspace(.001,.21,23)
                    if kind=='circle':y=np.sin(k*t)[:,None]/k*d+(1-np.cos(k*t))[:,None]/k*n
                    else:y=t[:,None]*d+.5*k*t[:,None]**2*n
                    foot,_=cc.project(y,p,sign,side,kind)
                    assert_allclose(foot,y,atol=2e-12)

    def test_parabolic_global_projection(self):
        rng=np.random.default_rng(15)
        d=np.array([0.,1.,0.]);n=np.array([1.,0.,0.]);k=4.
        q=rng.normal(0,.4,(30,3));t=cc.parabola_parameters(q,d,n,k)
        for p,s in zip(q,t):
            roots=np.roots([.5*k*k,0.,1-k*(p@n),-(p@d)])
            options=[0.]+[float(v.real) for v in roots if abs(v.imag)<1e-9 and v.real>=0]
            f=lambda r:np.sum((p-r*d-.5*k*r*r*n)**2)
            self.assertAlmostEqual(f(s),min(map(f,options)),places=11)

    def test_pair_constraint_not_just_completion(self):
        old=np.array([[-.5,0,0],[.5,0,0]])
        candidate=np.array([[0.,.85,0.],[0.,-.85,0.]])
        self.assertLess(np.max(np.sum((old[:,None]-candidate[None])**2,axis=2)),1.)
        z,alpha=ss.contract_pair(candidate,old,1-1e-11)
        self.assertLess(alpha,1.)
        self.assertLessEqual(np.sum((z[0]-z[1])**2),1.)

    def test_zero_additions(self):
        x,r=ss.make_seed(self.template,self.comparison,self.source,81)
        self.assertTrue(np.array_equal(x,self.source));self.assertEqual(r,[])
        self.assertTrue(ss.exact_geometry(x,self.source)['diameter_exactly_unchanged'])

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):ss.make_seed(self.template,self.comparison,self.source,80)

    def test_integer_check_rejects_diameter_growth(self):
        x=np.vstack((self.source,[10,0,0]))
        with self.assertRaises(ValueError):ss.exact_geometry(x,self.source)

    def test_integer_check_rejects_original_change(self):
        x=self.source.copy();x[0,0]+=1e-16
        with self.assertRaises(ValueError):ss.exact_geometry(x,self.source)

    def test_integer_check_rejects_duplicates(self):
        with self.assertRaises(ValueError):ss.exact_geometry(np.vstack((self.source,self.source[:1])),self.source)

    def test_reflection_and_count_and_determinism(self):
        for count in [82,83,121,122]:
            x,r=ss.make_seed(self.template,self.comparison,self.source,count,seed=81)
            xx,rr=ss.make_seed(self.template,self.comparison,self.source,count,seed=81)
            self.assertTrue(np.array_equal(x,xx));self.assertEqual(r,rr)
            self.assertEqual(len(x),count)
            ss.exact_geometry(x,self.source)
            f=self.template['frame'];y=(x-np.array(f['origin']))@np.array(f['basis_columns_a_b_n'])
            for rec in r:
                ids=rec['indices']
                if len(ids)==2:assert_allclose(y[ids[0]]*ss.REF,y[ids[1]],atol=3e-14)
                else:self.assertLess(abs(y[ids[0],2]),3e-14)

    def test_parabola_sampling(self):
        comps=ss.components(self.template,self.comparison,'parabola')
        fractions=np.linspace(.01,.99,19)
        for component in comps:
            if component['curve']!='branch':continue
            b=component['data'];y=ss.sample_curve(component,fractions)
            t=(y-np.array(b['junction']))@np.array(b['tangent'])
            lo,hi=b['parameter_interval'];k=b['curvature']
            actual=(ss.arclength(t,k,'parabola')-ss.arclength(lo,k,'parabola'))/(ss.arclength(hi,k,'parabola')-ss.arclength(lo,k,'parabola'))
            assert_allclose(actual,fractions,atol=1e-11)
        x,r=ss.make_seed(self.template,self.comparison,self.source,101,kind='parabola',seed=11)
        self.assertTrue(ss.exact_geometry(x,self.source)['diameter_exactly_unchanged'])

    def test_comparison_reconstruction(self):
        f=self.template['frame'];y=(self.source-np.array(f['origin']))@np.array(f['basis_columns_a_b_n'])
        for label,fit in self.comparison['fits'].items():
            sign=1 if label.startswith('plus') else -1
            for kind,data in fit['models'].items():
                errors=[]
                for b in data['branches']:
                    ids=b['indices'];side=b['branch']=='side';q=y[ids]-b['junction']
                    foot,_=cc.project(q,np.array(b['shape_parameters']),sign,side,kind)
                    err=np.linalg.norm(foot-q,axis=1)
                    assert_allclose(err,b['distance_residuals'],atol=5e-13)
                    errors.extend(err.tolist()*(2 if side else 1))
                errors.append(0.)
                self.assertAlmostEqual(np.sqrt(np.mean(np.array(errors)**2)),data['rms'],places=12)

if __name__=='__main__':unittest.main(verbosity=2)
