# Local tooling

| Script | Action |
|---|---|
| `setup.sh` | Human-invoked venv/dependency setup; native build unless --python-only |
| `build_native.py` | Stage/build/replaces the local library, no downloads |
| `doctor.py` | Report dependencies, compiler/Codex discovery, tiny Python/native graph checks |
| `check_assets.py` | Compare immutable scientific input hashes, no rewriting |
| `run_tests.py` | Consolidated unittest suites with explicit skips and optional JSON report |
| `smoke.py` | Two tiny R3/R4 search jobs, each followed by Python witness verification |
| `run_experiment.py` | One bounded explicit-argv child with provenance and process-group cleanup |
| `verify_examples.py` | Verify finite certificates in `best_known/` with the chosen backend; directories without a certificate are omitted |

Run from any directory using an absolute script path; ordinary research CLIs still accept
normal relative input/output paths. The runner executes child argv from the repository
root. Setup uses a local venv, not global installation. None of these tools installs
Codex, logs in, publishes a repository or sets an API key.
