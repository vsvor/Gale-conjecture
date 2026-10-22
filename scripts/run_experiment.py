#!/usr/bin/env python3
"""Run one bounded experiment with local provenance, logs and input hashes.

No shell expansion, automatic parameter sweeps, credential logging or record claims.
Commands run from the repository root. {python}, {root}, {out}, {seed}, {backend}
are replaced as literal tokens/substrings; {out} is the child's fresh result folder.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS={'.git','.venv','venv','__pycache__','.pytest_cache','experiments','validation','.mypy_cache'}
SUFFIXES={'.py','.cpp','.h','.hpp','.json','.md','.txt','.toml','.sh','.yml','.yaml','.csv','.npy','.npz'}

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def source_hashes(root:Path=ROOT)->dict[str,str]:
    result={}
    for current,dirs,files in os.walk(root):
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.native-build-'))
        for name in sorted(files):
            path=Path(current)/name
            if path.is_symlink():continue
            if path.suffix in SUFFIXES or name in {'Makefile','.gitignore','.gitattributes'}:
                result[str(path.relative_to(root))]=sha256(path)
    return result

def git_info(root:Path=ROOT)->dict[str,Any]:
    def run(*args):
        try:
            p=subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True,timeout=8)
            return p.stdout.strip() if p.returncode==0 else None
        except (OSError,subprocess.TimeoutExpired):return None
    head=run('rev-parse','HEAD');status=run('status','--porcelain')
    return {'commit':head,'dirty':bool(status) if status is not None else None,
            'status_porcelain':status,'branch':run('branch','--show-current')}

def resolve_inputs(items,root:Path=ROOT):
    hashes={}
    for value in items:
        p=Path(value).expanduser();p=p if p.is_absolute() else root/p
        if not p.is_file():raise ValueError(f'Input is not a file: {p}')
        hashes[str(p.resolve())]=sha256(p)
    return hashes

def expand_command(items, substitutions):
    if not items or not all(isinstance(v,str) and v for v in items):
        raise ValueError('Command must be a nonempty list of strings.')
    result=[]
    for arg in items:
        for key,value in substitutions.items(): arg=arg.replace('{'+key+'}',str(value))
        result.append(arg)
    return result

def write_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def stop_group(proc):
    if os.name=='posix':
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGKILL):
            try:os.killpg(proc.pid,sig)
            except ProcessLookupError:return
            try:proc.wait(timeout=2)
            except subprocess.TimeoutExpired:continue
            # The group may still contain grandchildren after the parent exits.
            time.sleep(.1)
        return
    # No full process-tree cleanup promise for native Windows; use WSL for workers.
    proc.terminate()
    try:proc.wait(timeout=2)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()

def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset',type=Path)
    p.add_argument('--name',help='Simple new folder name beneath experiments/.')
    p.add_argument('--out',type=Path,help='Alternative new run folder (never overwritten).')
    p.add_argument('--seed',type=int,default=20260911)
    p.add_argument('--backend',choices=['native','python','external','pysat','auto'],default='native')
    p.add_argument('--timeout',type=float,help='Overall wall-clock seconds (preset default, otherwise 300).')
    p.add_argument('--allow-long',action='store_true',help='Explicitly permit budgets exceeding 600 seconds.')
    p.add_argument('--input',action='append',default=[],help='Input file to hash; repeat as needed.')
    p.add_argument('--dry-run',action='store_true',help='Print expanded argv and inputs without running/writing.')
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args(argv)
    preset={}
    try:
        if a.preset:
            path=a.preset if a.preset.is_absolute() else ROOT/a.preset
            preset=json.loads(path.read_text())
            if not isinstance(preset,dict):raise ValueError('Preset must be a JSON object.')
        command=a.command[1:] if a.command[:1]==['--'] else a.command
        if command and a.preset:raise ValueError('Use --preset OR a command after --, not both.')
        command=command or preset.get('command',[])
        timeout=a.timeout if a.timeout is not None else preset.get('timeout_seconds',300)
        if not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
            raise ValueError('Timeout must be finite and positive.')
        if timeout>600 and not a.allow_long:raise ValueError('Budgets above 600 s require --allow-long.')
        if a.seed<0:raise ValueError('Seed must be nonnegative.')
        if a.name and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.name):raise ValueError('Use a simple --name, not a path.')
        if a.out and a.name:raise ValueError('Use --out or --name, not both.')
        default_name='run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        out=(a.out.expanduser() if a.out else ROOT/'experiments'/(a.name or default_name)).resolve()
        if out.exists():raise ValueError(f'Output already exists: {out}')
        inputs=resolve_inputs(preset.get('inputs',[])+a.input)
        command=expand_command(command,{'python':sys.executable,'root':ROOT,'out':out/'result','seed':a.seed,'backend':a.backend})
    except (OSError,ValueError,TypeError,json.JSONDecodeError) as exc:p.error(str(exc))
    plan={'cwd':str(ROOT),'command':command,'timeout_seconds':timeout,'inputs':inputs,'output':str(out)}
    if a.dry_run:print(json.dumps(plan,indent=2));return 0
    out.mkdir(parents=True,exist_ok=False)
    data={**plan,'schema_version':1,'started_utc':datetime.now(timezone.utc).isoformat(),
          'seed_metadata':a.seed,'backend_metadata':a.backend,'git_before':git_info(),
          'source_hashes_before':source_hashes(),'python':sys.version,'executable':sys.executable,
          'packages':{},'native_library_sha256':sha256(ROOT/'dsatur_native.so') if (ROOT/'dsatur_native.so').is_file() else None,
          'status':'running','qualification':'Process provenance only; successful execution is not certificate verification.'}
    for name in ['numpy','scipy','matplotlib','Pillow','python-sat']:
        try:data['packages'][name]=md.version(name)
        except md.PackageNotFoundError:pass
    write_json(out/'run.json',data)
    (out/'EXPERIMENT.md').write_text('# Experiment report\n\n## Hypothesis\nTODO\n\n## Method\nSee run.json for argv, seed, inputs, source hashes and environment.\n\n## Result\nTODO: report lower AND upper bounds, timeouts and verification command.\n\n## Limitations\nAn exit code of zero is not a proof or a new record.\n')
    print(json.dumps(plan,indent=2),flush=True)
    print(f'Console output is recorded in {out/"console.log"}',flush=True)
    env=os.environ.copy();env['BORSUK_BLAS_THREADS']=env.get('BORSUK_BLAS_THREADS','1');env['MPLBACKEND']='Agg';env['PYTHONUNBUFFERED']='1'
    proc=None;start=time.monotonic();exit_code=2
    try:
        with (out/'console.log').open('wb') as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=(os.name=='posix'))
            try:
                rc=proc.wait(timeout=timeout);exit_code=rc if rc>=0 else 128-rc
                data['status']='completed' if rc==0 else 'failed'
            except subprocess.TimeoutExpired:
                stop_group(proc);data['status']='timed_out';exit_code=124
            except KeyboardInterrupt:
                stop_group(proc);data['status']='interrupted';exit_code=130
    except OSError as exc:
        data['status']='launch_failed';data['error']=str(exc)
    finally:
        if proc is not None and proc.poll() is None:stop_group(proc)
        data.update(exit_code=exit_code,elapsed_seconds=time.monotonic()-start,
                    finished_utc=datetime.now(timezone.utc).isoformat(),git_after=git_info())
        data['source_hashes_after']=source_hashes()
        data['source_changed_during_run']=data['source_hashes_before']!=data['source_hashes_after']
        data['input_hashes_after']={path:sha256(Path(path)) if Path(path).is_file() else None for path in inputs}
        data['inputs_unchanged']=inputs==data['input_hashes_after']
        write_json(out/'run.json',data)
    print(json.dumps({'status':data['status'],'exit_code':exit_code,'elapsed_seconds':data['elapsed_seconds'],
                     'run':str(out),'source_changed_during_run':data['source_changed_during_run']},indent=2))
    return exit_code

if __name__=='__main__':raise SystemExit(main())
