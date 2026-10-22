# Test ownership

Run from any directory with `python /path/to/repo/scripts/run_tests.py --suite all`.
The runner sets the repository cwd/import path, reports skips, and fails on zero tests.
The package's validation directory records actual assembly results, not historic sums.

The current checkout lacks six legacy fixture directories: `cover49`, `d3_m23`,
`d3_m46`, `d3_m49`, `record58`, and `square`. They were already absent before the
`examples/` to `best_known/` rename. Eleven fixture-specific tests in the tuning and
boundary suites require those original assets and currently error. Their mathematical
expectations have been preserved; restore the original data before claiming a clean
full-suite run. Generic viewer and k-guard tests use the retained 81-point 3D and
61-point 4D configurations.

| Suite | Files / principal contracts |
|---|---|
| `engine` | `test_borsuk_search.py`, `test_early_stop.py`, `test_native_backend.py`: threshold/ties, model parsing, bounds, gradients, shared stopping, native ABI |
| `tuning` | `test_tune_solution.py`, `test_split_tune.py`, `test_paired_replay.py`: exact state, deletion/insertion, polishing and split scaffolds |
| `geometry` | `test_distillation.py`, `test_core_seed.py`, `test_structured_seed.py`: exceptions, lineage, sampling, exact seed diameter, core metrics |
| `boundary` | `test_surface_search.py`, `test_boundary_workspace.py`: rational boundary membership and complete skeleton-cover checks |
| `viewer` | `test_viewer_workspace.py`: headless PNG/GIF, axes-off setting, PCA labeling and partitions |
| `quick` | engine foundation + `test_workspace.py` + `test_experiment_runner.py`: imports, resources, k guard, provenance/timeouts |

The inherited suites use unittest. Added tests follow the same runner; pytest is not
required. `tests/fixtures/original73.npy` and `distinct70/` are frozen replay data.
Tests create temporary directories and must not rewrite `best_known/` or `templates/`. They do
not rely on a preinstalled Kissat/PySAT; external adapter tests may use simulated
processes. Actual external solver integration remains a separate environment check.

Native-only tests skip explicitly without dsatur_native.so. A fallback test run is
useful but must not be described as a complete native validation. Randomized tests use
fixed seeds. When subprocess workloads change, keep them within the documented small
budget; put expensive statistical campaigns in experiments, not default tests.
