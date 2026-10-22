# Validation of the consolidated workspace

Assembly checks performed on 2026-09-11, Linux x86_64, Python 3.13.5, NumPy 2.3.5,
SciPy 1.17.0, Matplotlib 3.10.8 and Pillow 12.3.0. The native solver was built locally
with Debian g++ 14.2.0. No compiled solver is included in the ZIP.

## Results actually obtained

| Check | Outcome | Record |
|---|---|---|
| Integrated unittest suite, native present | 206 tests, 0 failures/errors/skips | `assembly_tests_native.json`, `.log` |
| Integrated suite, native removed | 206 discovered/run, 200 passed, 6 explicit native-related skips | `assembly_tests_python.json`, `.log` |
| Tiny parallel baseline searches in R3 and R4, then Python verification | All four stages completed | `smoke.json`, `smoke.log` |
| Same smoke with Python-only backend | All stages completed | `smoke_python.json`, `.log` |
| Six bounded presets: search3/search4/core3/core4/tune3/distill3 | All exited 0; inputs/source unchanged during each run | `preset_smokes.json`, individual provenance/console logs |
| Saved results of the six presets | Each finite candidate verified with Python backend | `preset_result_verification.json` |
| Small split, paired-split, surface and edge CLI runs | All exited 0 | `additional_cli_smokes.json`, individual logs |
| Finite certificate examples | 11 verified with Python backend | `example_verification.json` |
| Surface sample membership and skeleton continuous cover | Separate verifiers passed | `surface_membership_verification.json`, `skeleton_verification.json` |
| Canonical CLI help | 13 entry points exited 0 | `cli_help.json` |
| Immutable template/example/fixture hashes | 98 files matched | `relocated_assets.json` |
| Fresh ZIP extraction, local rebuild, integrated suite | 206 tests passed without skips | `relocated_tests.json`, `.log` |
| Fresh extraction: tiny search/verify smoke and all finite examples | Passed | `relocated_smoke.json`, `relocated_examples.json` |

The 206 tests comprise 181 relocated inherited tests and 25 new integration tests.
They are not obtained by summing old README claims. Tests exercise small problems and
selected deterministic scientific fixtures, not global correctness of every search path.
Headless PNG/GIF output is tested; no GUI-window interaction was tested.

## Relocation and setup qualification

The extracted copy was placed under a different working directory and its native
library rebuilt. Setup was exercised **offline against the environment's already
installed scientific dependencies**, not with a fresh network download from PyPI.

An initial test-only virtualenv with `--system-site-packages` did not inherit SciPy
from the parent virtualenv; the offline dependency install therefore failed. The initial
setup and consequent import-failure logs are preserved with `initial_offline_` names.
No scientific-source fix or dependency downgrade was made to hide those failures.
For the successful check, a test-only `.pth` inside the extracted copy's venv made the
existing `/opt/pyvenv/lib/python3.13/site-packages` available. Setup, all 206 tests, smoke
searches and example verification then passed. That venv, its absolute dependency bridge,
and all binaries are excluded from the distributed archive.

Users normally run `bash scripts/setup.sh` with package-index access, or provision the
listed dependencies themselves. Actual network dependency installation on a new machine
remains untested here. `requirements-tested.txt` records the successful environment.

## Not performed

No Codex binary was installed, invoked or authenticated (`doctor.json` reports its
absence). The project configuration and instructions were checked against official
OpenAI documentation, not exercised in a live agent session. No actual Kissat/PySAT
backend execution, independent DRAT/LRAT proof checking, macOS/native-Windows test or
long performance/record search was done during assembly. The external-process adapter
is covered by inherited synthetic tests, which is a different claim.

Historical record values are supplied examples, not improvements found in this release.
A zero process exit code does not certify its result; the separate checks above do.
Source-manifest/asset hashes provide provenance, not mathematical proof.

Paths and interpreter locations in logs describe this build environment. They are not
runtime dependencies of the source or instructions to recreate `/mnt/data` locally.
