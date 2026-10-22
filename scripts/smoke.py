#!/usr/bin/env python3
"""Bounded end-to-end searches in R3 and R4, followed by Python witness verification."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_experiment import stop_group

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['native','python'],default='native')
    p.add_argument('--out',type=Path);a=p.parse_args(argv)
    out=(a.out or ROOT/'experiments'/('smoke_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))).resolve()
    if out.exists():p.error('Use a new output directory.')
    out.mkdir(parents=True)
    env=os.environ.copy();env['BORSUK_BLAS_THREADS']='1';env['MPLBACKEND']='Agg';env['PYTHONUNBUFFERED']='1'
    runs=[];ok=True
    for d in (3,4):
        result=out/f'd{d}'
        stages=[('search',[sys.executable,str(ROOT/'borsuk_search.py'),'search','--dim',str(d),'--points','12',
                  '--k',str(d+1),'--restarts','2','--workers','2','--rounds','2','--steps','6','--prune-checks','6',
                  '--polish','4','--seed','20260911','--early-stop','--backend',a.backend,'--sat-timeout','3','--quiet','--out',str(result)]),
                ('verify',[sys.executable,str(ROOT/'borsuk_search.py'),'verify',str(result/'best'),'--backend','python','--sat-timeout','5'])]
        for name,cmd in stages:
            start=time.monotonic()
            try:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                      text=True,start_new_session=(os.name=='posix'))
                try:
                    stdout,stderr=proc.communicate(timeout=60);code=proc.returncode
                except subprocess.TimeoutExpired:
                    stop_group(proc);stdout,stderr=proc.communicate();code=124
                except KeyboardInterrupt:
                    stop_group(proc);raise
            except OSError as e:
                stdout='';stderr=str(e);code=2
            (out/f'd{d}_{name}.log').write_text(stdout+'\n'+stderr)
            rec={'dimension':d,'stage':name,'command':cmd,'returncode':code,'seconds':time.monotonic()-start}
            runs.append(rec);print(json.dumps(rec),flush=True)
            if code!=0:ok=False;break
    report={'successful':ok,'backend':a.backend,'runs':runs,'output':str(out)}
    (out/'smoke.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if ok else 1
if __name__=='__main__':raise SystemExit(main())
