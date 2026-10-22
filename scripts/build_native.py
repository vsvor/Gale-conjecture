#!/usr/bin/env python3
"""Build the local DSATUR library; no downloads or system installation."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler', default=os.environ.get('CXX', 'c++'))
    args = p.parse_args(argv)
    if os.name != 'posix':
        p.error('This build helper targets Linux/macOS. Use WSL for the native backend on Windows.')
    compiler = shlex.split(args.compiler)
    if not compiler:
        p.error('Compiler command must not be empty.')
    target = ROOT/'dsatur_native.so'
    # Stage beside the final library so replacement is on the same filesystem.
    with tempfile.TemporaryDirectory(prefix='.native-build-', dir=ROOT) as tmp:
        staged = Path(tmp)/target.name
        flag = '-dynamiclib' if sys.platform == 'darwin' else '-shared'
        command = compiler + ['-O3', '-std=c++17', flag, '-fPIC', str(ROOT/'dsatur_native.cpp'), '-o', str(staged)]
        print(shlex.join(command), flush=True)
        try:
            subprocess.run(command, check=True, cwd=ROOT, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            p.error(f'Native build failed; any existing library was left unchanged: {exc}')
        staged.replace(target)
    print(f'Built {target}. Restart Python processes that had already loaded an older library.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
