import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import core_seed as c
import numpy as np


def args(dim=3, method='spatial'):
    return c.build_parser().parse_args(['generate','--dim',str(dim),'--points','81',
        '--method',method,'--backend','python','--sat-timeout','2','--out','unused'])


def brute_sat(n,k,edges):
    return any(all(a[i]!=a[j] for i,j in edges) for a in itertools.product(range(k),repeat=n))


class SamplingTests(unittest.TestCase):
    def test_quotas(self):
        for n in range(40):
            q=c.quotas(np.array([1.,3.,2.]),n)
            self.assertEqual(sum(q),n); self.assertTrue(np.all(q>=0))
    def test_features(self):
        x=np.array([[0.,0.],[.1,0.],[1.,0.]])
        f=c.distance_features(x,x,(.5,))
        self.assertAlmostEqual(np.sum((f[0]-f[1])**2),.25**2*.1**2)
    def test_all_selectors_3d(self):
        for meth in ('sphere','iid','spatial','signature'):
            x,meta=c.propose(3,81,meth,246,args(method=meth))
            self.assertEqual(x.shape,(81,3)); self.assertTrue(np.isfinite(x).all())
            self.assertEqual(len(np.unique(x,axis=0)),81)
            self.assertAlmostEqual(c.bs.distances2(x,c.bs.pairs_for(81)).max(),1.)
    def test_all_selectors_4d(self):
        for meth in ('iid','spatial','signature'):
            x,meta=c.propose(4,81,meth,246,args(4,meth))
            self.assertEqual(x.shape,(81,4)); self.assertEqual(meta['exceptions_retained'],16)
            self.assertEqual(sum(meta['group_counts']),65)
            self.assertEqual(len(np.unique(x,axis=0)),81)
    def test_reproducible(self):
        a=args();x,m=c.propose(3,81,'spatial',32,a);y,n=c.propose(3,81,'spatial',32,a)
        self.assertTrue(np.array_equal(x,y)); self.assertEqual(m,n)
    def test_exception_geometry(self):
        for dim in (3,4):
            a=args(dim); seed=4
            p,l,fixed,lm,q,names=c.template_pool(dim,81,np.random.default_rng(seed))
            x,meta=c.propose(dim,81,'spatial',seed,a)
            old=c.bs.distances2(fixed,c.bs.pairs_for(len(fixed)))
            new=c.bs.distances2(x[:len(fixed)],c.bs.pairs_for(len(fixed)))
            self.assertTrue(np.allclose(new/old,(new/old)[0],rtol=1e-12,atol=1e-12))
    def test_fresh_not_anchor_all(self):
        p,l,fixed,lm,q,names=c.template_pool(3,81,np.random.default_rng(9))
        x,chosen=c.select_pool(p,l,fixed,lm,q,'spatial',np.random.default_rng(9))
        self.assertEqual(len(fixed),10)
        self.assertTrue(np.array_equal(x[:10],fixed))
        self.assertLess(np.sum(c.cdist(lm,x).min(axis=1)<1e-12),len(lm))
    def test_infeasible_spacing(self):
        p=np.array([[0.,0.],[.001,0.],[.002,0.]])
        with self.assertRaises(ValueError):
            c.select_pool(p,np.zeros(3,int),p[:1],p,np.array([2]),'spatial',np.random.default_rng(1),.1)
    def test_small_count_rejected(self):
        with self.assertRaises(ValueError):c.template_pool(4,16,np.random.default_rng(1))


class CoreTests(unittest.TestCase):
    def make(self,x,k):
        a=args();p=c.ts.OraclePool(k,a)
        g=c.ts.Geometry.from_float(np.array(x,dtype=float))
        s=c.ts.assess(g,p)
        self.addCleanup(p.close)
        return s,p
    def test_triangle_all_mandatory(self):
        s,p=self.make([[0,0],[2,0],[1,1]],2)
        a=c.fixed_threshold_audit(s,p,3,1)
        self.assertEqual(a['required_count'],3)
        self.assertEqual(a['anchored_minimum_core_upper_bound'],3)
    def test_tied_square(self):
        s,p=self.make([[0,0],[1,0],[1,1],[0,1]],2)
        a=c.fixed_threshold_audit(s,p,3,1)
        self.assertEqual(s.lower2,c.Fraction(1,2))
        self.assertEqual(a['required_count'],0)
        self.assertEqual(a['anchored_minimum_core_upper_bound'],3)
    def test_core_bracket_against_exhaustion(self):
        rng=np.random.default_rng(11)
        for n in (5,6):
            for _ in range(4):
                s,p=self.make(rng.normal(size=(n,2)),2)
                report=c.fixed_threshold_audit(s,p,4,2)
                edges=s.g.pairs[s.g.threshold(s.lower2)]
                anchor=report['diameter_pair']; minimum=n
                for count in range(2,n+1):
                    for chosen in itertools.combinations(range(n),count):
                        if not set(anchor)<=set(chosen):continue
                        lookup={u:j for j,u in enumerate(chosen)}
                        e=[(lookup[u],lookup[v]) for u,v in edges if u in lookup and v in lookup]
                        if not brute_sat(count,2,e):minimum=min(minimum,count)
                self.assertLessEqual(report['anchored_minimum_core_lower_bound'],minimum)
                self.assertGreaterEqual(report['anchored_minimum_core_upper_bound'],minimum)
    def test_found_cores_inclusion_minimal(self):
        s,p=self.make(np.random.default_rng(71).normal(size=(7,2)),2)
        report=c.fixed_threshold_audit(s,p,3,42)
        es=s.g.pairs[s.g.threshold(s.lower2)]
        for core in report['cores']:
            chosen=core['indices'];lookup={u:j for j,u in enumerate(chosen)}
            edges=[(lookup[u],lookup[v]) for u,v in es if u in lookup and v in lookup]
            self.assertFalse(brute_sat(len(chosen),2,edges))
            for v in chosen:
                if v in report['diameter_pair']: continue
                self.assertTrue(brute_sat(len(chosen),2,[(u,w) for u,w in edges if lookup[v] not in (u,w)]))
    def test_timeouts_retain_vertices(self):
        s,p=self.make([[0,0],[1,0],[1,1],[0,1]],2)
        with patch.object(p.get(4),'solve',return_value=c.bs.Answer(c.bs.Status.UNKNOWN)):
            a=c.fixed_threshold_audit(s,p,2,0)
        self.assertEqual(a['required_count'],0)
        self.assertEqual(a['anchored_random_deletion_counts'],[4,4])
        self.assertTrue(a['required_check_unknowns'])
        self.assertFalse(a['cores'][0]['inclusion_minimal_subject_to_diameter_pair'])
    def test_normalized_reducer_nondegrading(self):
        s,p=self.make(np.random.default_rng(71).normal(size=(9,2)),2)
        r=c.normalized_reductions(s,p,2,101)
        for run in r['runs']:self.assertGreaterEqual(run['lower'],s.lower-1e-14)
    def test_save_verify(self):
        s,p=self.make([[0,0],[2,0],[1,1]],2)
        with tempfile.TemporaryDirectory() as folder:
            c.ts.save_state(Path(folder),s,2,{})
            result=c.bs.verify_candidate(Path(folder),'python','kissat',2.,'g4')
            self.assertTrue(result['verified'])
    def test_selection_gate(self):
        def r(score,size,unknown=False):
            return {'audit':{'threshold_is_exact_optimum':True,'lower':score,
                            'anchored_minimum_core_upper_bound':size,'required_count':size-2,
                            'required_check_unknowns':[0] if unknown else [],'cores':[{'unknown':[]}]}}
        records=[r(.88,25),r(.82,65),r(.875,40),r(.887,71,True)]
        self.assertEqual(c.choose_winner(records,.02,0),2)
        self.assertIsNone(c.choose_winner(records,.02,60))
    def test_warmup_keeps_point_count(self):
        a=args();a.warmup_rounds=1;a.steps=2;a.prune_checks=3
        p=c.ts.OraclePool(2,a);self.addCleanup(p.close)
        s,log=c.warmup(np.random.default_rng(11).normal(size=(8,2)),2,p,a,1)
        self.assertEqual(s.g.m,8);self.assertEqual(len(log),2)
        self.assertGreaterEqual(log[1]['lower'],log[0]['lower'])

if __name__=='__main__':unittest.main(verbosity=2)
