#!/usr/bin/env python3
"""Verify versioned scientific assets. Does not regenerate/overwrite the manifest."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,default=ROOT/'DATA_MANIFEST.json')
    a=p.parse_args(argv)
    try:
        data=json.loads(a.manifest.read_text());bad=[]
        for relative,expected in data['files'].items():
            path=(ROOT/relative).resolve()
            if ROOT not in path.parents:raise ValueError('Manifest path escapes repository.')
            observed=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if observed!=expected:bad.append(relative)
        print(json.dumps({'checked':len(data['files']),'successful':not bad,'mismatches':bad},indent=2))
        return 1 if bad else 0
    except (OSError,ValueError,KeyError) as e:p.error(str(e))
if __name__=='__main__':raise SystemExit(main())
