# Development and testing

## Environment

Use Python 3.10+ and the root dependency files. `scripts/setup.sh` creates a local venv,
installs requirements, compiles native DSATUR, and runs the doctor. Use `--python-only`
to omit compilation. Activate `.venv` before `make` or pass `PYTHON=.venv/bin/python`.
The bounded scientific CLIs do not require internet once dependencies are installed.

`requirements-tested.txt` pins the assembly's Python-3.13 Linux package versions for
comparison. On Python 3.10, use `requirements.txt` and let pip select compatible wheels;
do not force incompatible tested-environment pins. The optional PySAT dependency is
`python-sat`, not the unrelated `pysat` package. External Kissat is user-supplied.

## Canonical sources and assets

Edit the root source modules and tests. Do not edit imported historical documentation
as if it were a current CLI guide. There are no duplicate engines inside `best_known/`.
Templates and best-known configurations are immutable inputs during a run. `scripts/check_assets.py`
checks their recorded hashes; when deliberately changing a fixture, update the
manifest in the same reviewed commit and document why.

The code is intentionally still a set of importable scripts. Moving it to an installable
package is a separate refactoring project, not a prerequisite for Codex work.

## Tests

```bash
python scripts/run_tests.py --suite quick
python scripts/run_tests.py --suite engine
python scripts/run_tests.py --suite tuning
python scripts/run_tests.py --suite geometry
python scripts/run_tests.py --suite boundary
python scripts/run_tests.py --suite viewer
python scripts/run_tests.py --suite all --report validation/local-all.json

# A single file:
python scripts/run_tests.py --pattern test_core_seed.py

# End-to-end R3 and R4 baseline search plus Python verification:
python scripts/smoke.py

# Recheck all certificates in best_known/ (legacy script name):
python scripts/verify_examples.py --backend python
```

The runner works from other current directories, sets the repository as the test cwd,
and fails when discovery finds zero tests. Missing-native tests are explicitly skipped;
absence of a compiled backend is not a native-test success. `tests/README.md` maps test
ownership. Baseline evaluation, tuner, split, and model tests come from the supplied
packages, with path changes for consolidation; workspace/runner/viewer tests are new.

After a C++ edit:

```bash
python scripts/build_native.py
python scripts/run_tests.py --suite engine
```

Compilation is staged and atomically replaces the library only on success. Restart
processes that have already loaded it. Native ABI status codes are 1 SAT, 0 UNSAT,
-1 timeout, -2 invalid input/internal failure. The current implementation accepts k<=63
and n<=10000; it is not a geometry solver and need not gain one.

## CLI pitfalls

- `--k` is the number of parts, not the desired chromatic number of the witness.
- `tune_solution.py --trials 0` disables insertion proposals; witness trials still run.
  `split_tune.py --trials 0` instead removes the cap on positive-LP split trials.
- `--polish-only` in the tuner disables topology exploration. It will not reproduce an
  improvement whose cause was witness pruning.
- An unsuccessful split round may terminate the run. Increasing `--rounds` alone does
  not mean independent restarts. `--zero-gain-trials` defaults to zero in the splitter.
- `paired_splits.py` assumes k=4. Assembly adds an early rejection for saved inputs
  declaring another k; metadata-free arrays still require the user to choose this
  script only for a k=4 task. Its first-split enumeration can be expensive before the
  `--first-limit` slice applies. Use an overall wall-time limit.
- `distill_search.py` currently supports the 3D curve model. Use `core_seed.py` for the
  fitted 4D patch initializer; a 4D distillation loop is not implemented here.
- The core generator may return code 3 when no candidate passes its gates. That is a
  legitimate no-selection outcome, not an instruction to weaken those gates silently.
- Continue distillation from its run root for the complete beam. For surface/edge
  sampling, `latest/` keeps neutral additions; `best/` may discard them on restart.
- Supply `structured_seed.py` output as an explicit `points.npy` path when relying on
  its exact binary-diameter check. Other standard solution directories prefer CSV.
- All outputs require a fresh directory. Do not reuse a completed `--out` path.
- Viewer `--no-show` selects a headless backend; saving without that flag preserves the
  possibility of a GUI. GIF writing requires Pillow, not ffmpeg. Axes are hidden unless
  `--axes` is passed. 4D/5D views are PCA projections, not distance-preserving embeddings.

## Small feature workflow

Read the relevant invariants and routing table. Reproduce the failure on a small input.
Patch only the owning layer and its direct interface/tests. Run focused tests, then the
full affected dependency suites, then a tiny recorded smoke. Review numerical tolerances,
CLI defaults, output schema changes and input mutation explicitly. Update documentation
when the interface changes. Never regenerate all fixtures to hide a failing regression.

## Debugging

A loader error: inspect `scripts/doctor.py`; rebuild next to the canonical engine and
confirm its absolute path. A hard solver instance: preserve UNKNOWN and inspect logs;
raise a timeout only as an explicit experiment decision. Changed output coordinates:
compare seed, full argv, input hash, canonical version, compiler and NumPy/SciPy/BLAS
versions before claiming a regression. Bitwise equality across different platforms is
not promised. An empty selected folder: check seed-selection completion/exit code first.

## Scope of integration

The main algorithms are preserved. Module-path changes, a k guard and a viewer backend
fix are documented in `assembly.md`; no generalizing of paired splitting, 4D distillation,
proof production, global certification or core-minimization theorem was added.
