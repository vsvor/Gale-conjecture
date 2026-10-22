"""Imports, relocated assets, and explicit prototype limitations."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

class WorkspaceTests(unittest.TestCase):
    def test_one_canonical_engine_import(self):
        engine=importlib.import_module('borsuk_search')
        for module in ['tune_solution','split_tune','paired_splits','distill_search','core_seed']:
            obj=importlib.import_module(module)
            self.assertEqual(Path(obj.__file__).resolve().parent,ROOT)
            if hasattr(obj,'bs'):self.assertIs(obj.bs,engine)
    def test_template_identity(self):
        data=json.loads((ROOT/'templates/curves3.json').read_text())
        x=np.load(ROOT/'templates/source3.npy',allow_pickle=False)
        self.assertEqual(hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest(),data['input_array_sha256'])
    def test_paired_splits_rejects_saved_k5_before_writing(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run'
            p=subprocess.run([sys.executable,str(ROOT/'paired_splits.py'),str(ROOT/'best_known/4d_61p_091314'),
                 '--out',str(out)],capture_output=True,text=True,timeout=15)
            self.assertEqual(p.returncode,2,p.stdout+p.stderr)
            self.assertIn('k=4 only',p.stderr);self.assertFalse(out.exists())
    def test_standalone_structured_seed_resolves_resources_from_elsewhere(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'seed'
            p=subprocess.run([sys.executable,str(ROOT/'structured_seed.py'),'--points','83','--out',str(out)],
                 cwd=td,capture_output=True,text=True,timeout=25)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            x=np.load(out/'points.npy',allow_pickle=False)
            self.assertEqual(x.shape,(83,3))
            np.testing.assert_array_equal(x[:81],np.load(ROOT/'templates/source3.npy',allow_pickle=False))
            record=json.loads((out/'sampling.json').read_text())
            self.assertIsNotNone(record['exact_geometry'])
    def test_presets_are_explicit_argv_and_bounded(self):
        files=list((ROOT/'configs').glob('smoke_*.json'));self.assertGreaterEqual(len(files),6)
        for f in files:
            d=json.loads(f.read_text());self.assertLessEqual(d['timeout_seconds'],600)
            self.assertIsInstance(d['command'],list);self.assertEqual(d['command'][0],'{python}')
            self.assertIn('{out}',d['command'])
            for p in d.get('inputs',[]):self.assertTrue((ROOT/p).is_file(),p)

if __name__=='__main__':unittest.main()
