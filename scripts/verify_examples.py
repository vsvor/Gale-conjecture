#!/usr/bin/env python3
"""Recheck best_known/ finite-set certificates (legacy command name retained)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import borsuk_search as bs

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['python','native','external','pysat'],default='python')
    p.add_argument('--solver',default='kissat');p.add_argument('--sat-timeout',type=float,default=30)
    p.add_argument('--report',type=Path);a=p.parse_args(argv)
    records=[];fail=False
    for folder in sorted((ROOT/'best_known').iterdir()):
        if not (folder/'certificate.json').is_file():continue
        start=time.monotonic()
        try:
            report=bs.verify_candidate(folder,a.backend,a.solver,a.sat_timeout,'g4')
            records.append({'path':str(folder.relative_to(ROOT)),'seconds':time.monotonic()-start,**report})
            print(f'{folder.name}: verified [{report["lower_decimal"]}, {report["upper_decimal"]}]',flush=True)
        except (ValueError,RuntimeError,OSError,ImportError) as exc:
            fail=True;records.append({'path':str(folder.relative_to(ROOT)),'verified':False,'error':str(exc)})
            print(f'{folder.name}: NOT VERIFIED: {exc}',flush=True)
    data={'all_verified':bool(records) and not fail,'records':records,
          'qualification':'Finite CSV geometry and solver witness checks; not DRAT/LRAT proof checking or hull-cover verification.'}
    if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(data,indent=2)+'\n')
    return 0 if data['all_verified'] else 1
if __name__=='__main__':raise SystemExit(main())
