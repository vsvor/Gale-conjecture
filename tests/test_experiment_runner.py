"""Workspace integration tests; no research optimization or network access."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from scripts import run_experiment as runner

ROOT = Path(__file__).resolve().parents[1]

class ExperimentRunnerTests(unittest.TestCase):
    def run_cli(self, args, cwd=None):
        return subprocess.run([sys.executable,str(ROOT/'scripts/run_experiment.py'),*args],
                              cwd=cwd or ROOT,capture_output=True,text=True,timeout=20)
    def test_expand_keeps_arguments_literal(self):
        value=runner.expand_command(['{python}','a b','$(touch no)','{out}/x'],
                                    {'python':'python3','out':'dir with spaces'})
        self.assertEqual(value,['python3','a b','$(touch no)','dir with spaces/x'])
    def test_expansion_rejects_empty_command(self):
        with self.assertRaises(ValueError):runner.expand_command([], {})
    def test_dry_run_is_read_only(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run'
            p=self.run_cli(['--out',str(out),'--dry-run','--','{python}','-c','print(1)'],td)
            self.assertEqual(p.returncode,0,p.stderr);self.assertFalse(out.exists())
            self.assertEqual(json.loads(p.stdout)['cwd'],str(ROOT))
    def test_success_records_hashes_without_inventing_results(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run';source=Path(td)/'input.txt';source.write_text('source')
            code="import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(); (p/'ok.txt').write_text('ok'); print('finished')"
            p=self.run_cli(['--out',str(out),'--input',str(source),'--','{python}','-c',code,'{out}'])
            self.assertEqual(p.returncode,0,p.stderr)
            report=json.loads((out/'run.json').read_text())
            self.assertEqual(report['status'],'completed');self.assertTrue(report['inputs_unchanged'])
            self.assertFalse(report['source_changed_during_run'])
            self.assertIn('borsuk_search.py',report['source_hashes_before'])
            self.assertNotIn('best_ratio',report)
            self.assertEqual((out/'result/ok.txt').read_text(),'ok')
            self.assertIn('finished',(out/'console.log').read_text())
    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            p=self.run_cli(['--out',td,'--','{python}','-c','print(1)'])
            self.assertEqual(p.returncode,2);self.assertEqual(list(Path(td).iterdir()),[])
    def test_failed_child_is_not_success(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run'
            p=self.run_cli(['--out',str(out),'--','{python}','-c','raise SystemExit(7)'])
            self.assertEqual(p.returncode,7)
            self.assertEqual(json.loads((out/'run.json').read_text())['status'],'failed')
    def test_timeout_is_logged_and_stops_child(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run'
            p=self.run_cli(['--out',str(out),'--timeout','0.15','--','{python}','-c','import time; time.sleep(30)'])
            self.assertEqual(p.returncode,124,p.stderr)
            self.assertEqual(json.loads((out/'run.json').read_text())['status'],'timed_out')
    def test_long_budget_requires_explicit_gate(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run'
            p=self.run_cli(['--out',str(out),'--timeout','601','--dry-run','--','{python}','-c','pass'])
            self.assertEqual(p.returncode,2);self.assertFalse(out.exists())
    def test_launch_failure_retains_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run'
            p=self.run_cli(['--out',str(out),'--',str(Path(td)/'missing-executable')])
            self.assertEqual(p.returncode,2)
            self.assertEqual(json.loads((out/'run.json').read_text())['status'],'launch_failed')
    def test_input_mutation_is_reported(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run';source=Path(td)/'input.txt';source.write_text('before')
            p=self.run_cli(['--out',str(out),'--input',str(source),'--','{python}','-c',
                'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("after")',str(source)])
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertFalse(json.loads((out/'run.json').read_text())['inputs_unchanged'])

if __name__=='__main__':unittest.main()
