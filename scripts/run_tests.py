#!/usr/bin/env python3
"""Run unittest suites from any working directory, without silent zero-test success."""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('BORSUK_BLAS_THREADS','1')
sys.path.insert(0, str(ROOT))

SUITES = {
 'quick': ['tests.test_borsuk_search','tests.test_workspace','tests.test_experiment_runner'],
 'engine': ['tests.test_borsuk_search','tests.test_early_stop','tests.test_native_backend'],
 'tuning': ['tests.test_tune_solution','tests.test_split_tune','tests.test_paired_replay'],
 'geometry': ['tests.test_structured_seed','tests.test_core_seed','tests.test_distillation'],
 'boundary': ['tests.test_surface_search','tests.test_boundary_workspace'],
 'viewer': ['tests.test_viewer_workspace'],
}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=['all',*SUITES],default='quick')
    p.add_argument('--pattern',help='Override suite with a test filename glob, e.g. test_core_seed.py.')
    p.add_argument('--report',type=Path,help='JSON test result (optional).')
    args=p.parse_args(argv)
    # Legacy tests deliberately launch subprocess CLIs relative to the repository.
    os.chdir(ROOT)
    loader=unittest.TestLoader()
    suite=(loader.discover(str(ROOT/'tests'),pattern=args.pattern or 'test_*.py',top_level_dir=str(ROOT))
           if args.suite=='all' or args.pattern else loader.loadTestsFromNames(SUITES[args.suite]))
    count=suite.countTestCases()
    if count==0: p.error('No tests discovered; check the suite/pattern.')
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
    data={'suite':args.suite,'pattern':args.pattern,'tests_run':result.testsRun,
          'failures':len(result.failures),'errors':len(result.errors),
          'skipped':[{'test':str(t),'reason':r} for t,r in result.skipped],
          'elapsed_seconds':time.monotonic()-start,'successful':result.wasSuccessful(),
          'native_present':(ROOT/'dsatur_native.so').is_file(), 'python':sys.version}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(data,indent=2)+'\n')
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
