#!/usr/bin/env bash
# Human-invoked setup. Installs Python packages into .venv, not globally.
set -euo pipefail
if (( $# > 1 )) || [[ "${1:-}" != "" && "${1:-}" != "--python-only" ]]; then
    printf 'Usage: bash scripts/setup.sh [--python-only]\n' >&2
    exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3,10), "Python >=3.10 is required"'
if [[ ! -x .venv/bin/python ]]; then "$PYTHON" -m venv .venv; fi
.venv/bin/python -m pip install -r requirements.txt
if [[ "${1:-}" != "--python-only" ]]; then
    .venv/bin/python scripts/build_native.py
fi
.venv/bin/python scripts/doctor.py
printf '\nNext: source .venv/bin/activate\nThen: python scripts/run_tests.py --suite all\n'
