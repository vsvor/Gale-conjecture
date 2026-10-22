"""Tests for geometry, exception protection, checkpoint semantics, and orchestration."""
from __future__ import annotations
import copy,json,math,tempfile,unittest
from pathlib import Path
from fractions import Fraction
from unittest.mock import patch
import subprocess,sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
import borsuk_search as bs
import tune_solution as ts
import curve_model as cm
import distill_search as ds
ROOT=Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g,cls.w,cls.colors,_=ts.load_source(ROOT/'best_known/3d_81p_088871')
        cls.state=ts.make_state(cls.g,cls.w,cls.colors)
        cls.template=json.loads((ROOT/'templates/circle.json').read_text())
        cls.protected=cls.template['protected']
        cls.model,cls.report=cm.refit(cls.template,cls.g.x,cls.protected,iterations=12)
    def test_circle_projection(self):
        c={'kind':'circle','center':[0,0,0],'radius':1.,'u':[1,0,0],'v':[0,1,0],'interval':[0,np.pi/2]}
        t,y,d=cm.closest(c,np.array([[2,0,0],[1,1,3],[-1,0,0]]))
        np.testing.assert_allclose(t,[0,np.pi/4,np.pi/2]);np.testing.assert_allclose(d,[1,math.sqrt((math.sqrt(2)-1)**2+9),math.sqrt(2)])
    def test_wrapped_circle(self):
        c={'kind':'circle','center':[0,0,0],'radius':1.,'u':[1,0,0],'v':[0,1,0],'interval':[5.5,7.]}
        x=cm.value(c,[6.2,6.9]);t,y,d=cm.closest(c,x)
        np.testing.assert_allclose(t,[6.2,6.9]);self.assertLess(max(d),1e-14)
    def test_quadratic_projection(self):
        c={'kind':'parabola','coefficients':[[0,0,0],[1,0,0],[0,1,0]],'interval':[0,1]}
        x=np.array([[.1,.3,.2],[.7,.1,0],[2,2,0]])
        t,y,d=cm.closest(c,x)
        for i,row in enumerate(x):
            f=lambda v:np.sum((cm.value(c,[v])[0]-row)**2)
            r=minimize_scalar(f,bounds=(0,1),method='bounded',options={'xatol':1e-14})
            self.assertAlmostEqual(d[i]**2,min(f(0),f(1),r.fun),places=10)
    def test_reflection_involution(self):
        s=self.model['symmetry'];np.testing.assert_allclose(cm.reflect(cm.reflect(self.g.x,s),s),self.g.x,atol=1e-14)
    def test_reflected_curve(self):
        c=self.model['curves'][0];s=self.model['symmetry'];r=cm.reflected_curve(c,s,'reflected')
        np.testing.assert_allclose(cm.value(r,[.9,1.4]),cm.reflect(cm.value(c,[.9,1.4]),s),atol=1e-14)
    def test_fit_does_not_move_input(self):
        x=self.g.x.copy();old=x.copy();cm.refit(self.template,x,self.protected,iterations=4)
        np.testing.assert_array_equal(old,x)
    def test_exceptions_excluded_not_deleted(self):
        self.assertEqual(len(self.model['reference']),81)
        self.assertTrue(all(self.model['labels'][i]==-1 for i in self.protected))
        self.assertEqual(self.report['inliers'],71)
    def test_arclength_sampling_circle(self):
        c=self.model['curves'][0];t,y=cm.sample(c,np.linspace(0,1,101))
        ds_=np.linalg.norm(np.diff(y,axis=0),axis=1)
        self.assertLess(np.ptp(ds_),1e-14)
    def test_arclength_sampling_parabola(self):
        c={'kind':'parabola','coefficients':[[0,0,0],[1,0,0],[0,2,0]],'interval':[0,1]}
        t,y=cm.sample(c,np.linspace(0,1,101));ds_=np.linalg.norm(np.diff(y,axis=0),axis=1)
        self.assertLess(np.ptp(ds_)/np.mean(ds_),.003)
    def test_similarity_registration(self):
        r=Rotation.from_rotvec([.2,-.3,.1]).as_matrix();x=1.7*self.g.x@r+[3,2,-1]
        model=cm.transform(self.template,x)
        for c0,c1 in zip(self.template['curves'],model['curves']):
            ts_=np.linspace(*c0['interval'],3)
            np.testing.assert_allclose(cm.value(c1,ts_),1.7*cm.value(c0,ts_)@r+[3,2,-1],atol=2e-14)
    def test_asymmetric_addition_disables_symmetry(self):
        x=np.vstack((self.g.x,[.1,.2,.05]));m=copy.deepcopy(self.model);m['reference']=x.tolist()
        self.assertGreater(cm.symmetry_error(m,x,1),.001)
    def test_same_seed_reproducible(self):
        a,am,ar=cm.resample(self.state,self.model,7,np.random.default_rng(42))
        b,bm,br=cm.resample(self.state,self.model,7,np.random.default_rng(42))
        self.assertEqual(a.rows,b.rows);self.assertEqual(ar,br)
    def test_seed_exact_geometry(self):
        a,m,r=cm.resample(self.state,self.model,9,np.random.default_rng(17))
        self.assertEqual(a.m,90);self.assertEqual(a.rows[:81],self.g.rows)
        self.assertEqual(a.diameter2,self.g.diameter2);self.assertEqual(len(set(map(tuple,a.rows))),90)
    def test_coarse_decimal_grid_promoted_exactly(self):
        g=ts.Geometry([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,1]],0)
        c=np.array([0,1,2,3,0],np.int32);state=ts.make_state(g,np.arange(len(g.q)),c)
        model={'reference':g.x.tolist(),'curves':[{'name':'circle','kind':'circle','center':[0,0,0],
                   'radius':1.,'u':[1,0,0],'v':[0,1,0],'interval':[0,6.28],'support':4,'mirror':None}],
               'labels':[0,0,0,0,-1],'symmetry':None}
        ng,m,r=cm.resample(state,model,2,np.random.default_rng(1),exception_fraction=0)
        self.assertEqual(ng.m,7);self.assertEqual(ng.digits,17)
        for old,new in zip(g.rows,ng.rows):
            self.assertEqual(new,[v*10**17 for v in old])
        self.assertEqual(ng.diameter2,g.diameter2*10**34)
    def test_new_new_distance_checked(self):
        g=ts.Geometry.from_float(np.array([[-.5,0,0],[.5,0,0],[0,0,.1]]))
        ans=cm.add_exact(g,np.array([[0,.8,0],[0,-.8,0]]),g.rows,np.zeros(3),.001)
        self.assertIsNotNone(ans);rows,alpha=ans
        self.assertLess(alpha,.626);self.assertGreater(alpha,.62)
        self.assertEqual(ts.Geometry(g.rows+rows,g.digits).diameter2,g.diameter2)
    def test_duplicate_insertion_rejected(self):
        self.assertIsNone(cm.add_exact(self.g,self.g.x[0:1],self.g.rows,self.g.x.mean(0),.001))
    def test_zero_additions_no_changes(self):
        g,m,r=cm.resample(self.state,self.model,0,np.random.default_rng(1));self.assertEqual(g.rows,self.g.rows);self.assertEqual(r['added'],0)
    def test_requested_spacing_can_reduce_batch(self):
        g,m,r=cm.resample(self.state,self.model,1,np.random.default_rng(1),min_sep=2)
        self.assertEqual(g.m,81);self.assertEqual(r['added'],0)
    def test_greedy_partition_upper(self):
        g,m,r=cm.resample(self.state,self.model,4,np.random.default_rng(11))
        c=cm.greedy_partition(g,self.colors,4);self.assertEqual(len(c),85);self.assertTrue(np.all((c>=0)&(c<4)))
        w=ts.edge_ids(g.m,self.g.pairs[self.w]);s=ts.make_state(g,w,c)
        self.assertEqual(s.lower2,self.state.lower2);self.assertGreaterEqual(s.upper2,s.lower2)
    def test_parabolic_template_cycle(self):
        template=json.loads((ROOT/'templates/parabola.json').read_text())
        m,r=cm.refit(template,self.g.x,self.protected,iterations=3)
        g,m,r=cm.resample(self.state,m,4,np.random.default_rng(91));self.assertEqual(g.m,85);self.assertEqual(g.diameter2,self.g.diameter2)
    def test_circle_from_custom_group(self):
        c={'kind':'circle','center':[.2,.1,.3],'radius':.7,'u':[1,0,0],'v':[0,1,0],'interval':[0,2]}
        x=cm.value(c,np.linspace(0,2,20));fit=ds.circle_from_group(x,'test')
        self.assertLess(cm.closest(fit,x)[2].max(),1e-13)
    def test_default_template_mismatch_rejected(self):
        a=ds.parser().parse_args(['search','--out','unused']);x=np.random.default_rng(91).normal(size=(81,3))
        g=ts.Geometry.from_float(bs.normalize(x))
        with self.assertRaises(ValueError):ds.initialize_model(g,a)
    def test_saved_node_hash_new_not_parent(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test';ds.save_node(p,self.state,self.model,self.protected,{'k':4,'coordinates_sha256':'OLD'})
            self.assertEqual(json.loads((p/'lineage.json').read_text())['coordinates_sha256'],cm.digest(self.g))
            m,e=ds.read_lineage(p);self.assertEqual(e,self.protected)
    def test_exact_acceptance_guard(self):
        a=copy.copy(self.state);b=copy.copy(self.state)
        a.lower2=Fraction(49,100);a.upper2=Fraction(64,100)
        b.lower2=Fraction(56,100);b.upper2=Fraction(60,100)
        self.assertFalse(ds.strictly_improves(b,a,0))
        b.lower2=Fraction(65,100);b.upper2=Fraction(66,100)
        self.assertTrue(ds.strictly_improves(b,a,0))
    def test_timeout_does_not_mean_unsat(self):
        a=ds.local_args({'backend':'python','sat_timeout':.01});pool=ts.OraclePool(4,a)
        try:
            with patch.object(bs.GraphOracle,'solve',return_value=bs.Answer(bs.Status.UNKNOWN)):
                s=ts.assess(self.g,pool)
                self.assertTrue(s.unresolved);self.assertGreaterEqual(s.upper2,s.lower2)
                self.assertLessEqual(s.lower2,self.state.lower2)
        finally:pool.close()
    def test_neutral_descendant_can_enter_beam(self):
        with tempfile.TemporaryDirectory() as d:
            old=Path(d)/'old';new=Path(d)/'new'
            ng,nm,r=cm.resample(self.state,self.model,2,np.random.default_rng(31))
            c=cm.greedy_partition(ng,self.colors,4);w=ts.edge_ids(ng.m,self.g.pairs[self.w]);ns=ts.make_state(ng,w,c)
            ds.save_node(old,self.state,self.model,self.protected,{'k':4,'generation':0})
            ds.save_node(new,ns,nm,self.protected,{'k':4,'generation':1})
            chosen=ds.choose_beam([old,new],{str(old):self.state,str(new):ns},1,.002,np.random.default_rng(3))
            self.assertEqual(chosen,[new])
    def test_checkpoint_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test';ds.save_node(p,self.state,self.model,self.protected,{'k':4})
            md=json.loads((p/'model.json').read_text());md['labels'][9]=0;ds.write_json(p/'model.json',md)
            a=ds.local_args({'backend':'python','sat_timeout':2})
            with self.assertRaisesRegex(ValueError,'protected exception'):ds.verify(p,a)


class IntegrationTests(unittest.TestCase):
    def test_python_cycle_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            first=Path(d)/'one';second=Path(d)/'two'
            base=[sys.executable,str(ROOT/'distill_search.py'),'search']
            flags=['--children','1','--beam','1','--add','2','--local-rounds','0','--final-polish-passes','0',
                   '--fit-iters','4','--backend','python','--sat-timeout','2']
            a=subprocess.run(base+['--out',str(first),'--cycles','1']+flags,text=True,capture_output=True,timeout=35)
            self.assertEqual(a.returncode,0,a.stderr)
            b=subprocess.run(base+[str(first),'--out',str(second),'--cycles','1']+flags,text=True,capture_output=True,timeout=35)
            self.assertEqual(b.returncode,0,b.stderr)
            for root,m in [(first,83),(second,85)]:
                manifest=json.loads((root/'manifest.json').read_text());path=root/manifest['beam'][0]
                g,_,_,_=ts.load_source(path);self.assertEqual(g.m,m)
                protected=json.loads((path/'lineage.json').read_text())['protected_exceptions'];self.assertEqual(len(protected),10)
                a=ds.local_args({'backend':'python','sat_timeout':3});report=ds.verify(path,a);self.assertTrue(report['verified'])
    @unittest.skipUnless((ROOT/'dsatur_native.so').exists(),'Native library not compiled')
    def test_parallel_spawn_cycle(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'parallel'
            cmd=[sys.executable,str(ROOT/'distill_search.py'),'search','--out',str(out),'--cycles','1',
                 '--children','2','--workers','2','--add','2','--local-rounds','0','--final-polish-passes','0',
                 '--fit-iters','3','--backend','native']
            r=subprocess.run(cmd,text=True,capture_output=True,timeout=40);self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(len(list(out.glob('generations/*/*/optimized/worker.json'))),2)
    def test_invalid_cli(self):
        r=subprocess.run([sys.executable,str(ROOT/'distill_search.py'),'search','--out','not_created','--workers','0'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2);self.assertIn('positive',r.stderr)


if __name__=='__main__':unittest.main(verbosity=2)
