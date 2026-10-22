#!/usr/bin/env python3
"""Local environment and native-backend diagnostic. Does not install anything."""
from __future__ import annotations
import argparse
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def inspect():
    result = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'repository': str(ROOT), 'packages': {}, 'codex_path': shutil.which('codex'),
              'compiler': shutil.which(os.environ.get('CXX', 'c++')), 'issues': []}
    for name in ['numpy', 'scipy', 'matplotlib', 'Pillow']:
        try: result['packages'][name] = md.version(name)
        except md.PackageNotFoundError: result['issues'].append(f'Missing {name}; install requirements.txt.')
    if sys.version_info < (3,10): result['issues'].append('Python >=3.10 required.')
    for name in ['templates/curves3.json', 'templates/curves3_comparison.json', 'templates/patches4.json',
                 'templates/source3.npy', 'templates/source4.csv', 'templates/circle.json', 'templates/parabola.json']:
        if not (ROOT/name).is_file(): result['issues'].append('Missing asset: '+name)
    result['native'] = {'present': (ROOT/'dsatur_native.so').is_file(), 'validated': False}
    if not result['issues']:
        import borsuk_search as bs
        for backend in ['python'] + (['native'] if result['native']['present'] else []):
            try:
                with bs.GraphOracle(5, 4, backend, timeout=3) as oracle:
                    ans = oracle.solve(list(range(10)))  # K5 is not 4-colourable.
                    if ans.status is not bs.Status.UNSAT: raise RuntimeError(f'K5 status: {ans.status}')
                with bs.GraphOracle(4, 4, backend, timeout=3) as oracle:
                    ans = oracle.solve(list(range(6)))
                    if ans.status is not bs.Status.SAT: raise RuntimeError(f'K4 status: {ans.status}')
                if backend == 'native': result['native']['validated'] = True
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                result['issues'].append(f'{backend} backend: {exc}')
    result['ready'] = not result['issues']
    if not result['native']['present']:
        result['note'] = 'Python backend is available without compilation; build native for serious searches.'
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--json', type=Path)
    args = p.parse_args(); data = inspect()
    text = json.dumps(data, indent=2) + '\n'; print(text, end='')
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True); args.json.write_text(text)
    return 0 if data['ready'] else 1

if __name__ == '__main__': raise SystemExit(main())
