# Consolidation and compatibility notes

## Scope

Assembled on 2026-09-11 from the conversation's supplied archives and files. This is a
single-source local workspace with the requested engine, tuner, splitting, distillation,
core-aware generation, native backend, tests and docs. It additionally includes structured
sampling, surface/edge search and the GIF viewer. It is not a merge into an unseen local
checkout. Extract into a new directory and compare local modifications before replacing
an existing project.

The full historical trajectory of every record, large experiment output trees and
several one-off fitting studies are not duplicated. Active helpers, their matching data,
small deterministic fixtures and representative certificates are present. Existing
mathematical search algorithms were not replaced by a new implementation during assembly.

## Canonical version selection

`borsuk_search.py` is the early-stopping version with the absolute native-loader fix,
identical to the engine subsequently bundled with distillation and core-aware generation.
`tune_solution.py` and `dsatur_native.cpp` were identical across the selected packages.
The single/paired splitter comes from the paired73 lineage. Distillation/model code comes
from its own release; core_seed and its matching templates come from the core-aware release.

`SOURCE_MANIFEST.json` records original and assembled hashes for imports and hashes of
new workspace source/documentation files. `validation/imported_sources.json` preserves
the original import list. `validation/assembly.patch` shows textual compatibility changes
to imported Python sources/tests and the updated architecture document. `DATA_MANIFEST.json`
protects 98 example/template/fixture files and can be checked with `scripts/check_assets.py`.
These are provenance manifests, not replacement mathematical certificates.

## Changes to imported working files

| File(s) | Integration change |
|---|---|
| `structured_seed.py` | Point source/model/comparison resources to `templates/`; binary-NPY seed guarantee unchanged |
| `compare_curves.py` | Use consolidated template paths; write default refit under experiments and reject overwriting an existing fit |
| `paired_splits.py` | Reject saved inputs declaring k other than 4 before creating output; retain prototype's k4 algorithms |
| `view_solution.py` | Select Agg only with `--no-show`, not merely because PNG/GIF output is requested; preserve interactive save use |
| Inherited `tests/test_*.py` | Move root/fixture resource paths to consolidated tree, remove session-specific input path; assertions retained |
| `docs/architecture.md` | Match actual test/tooling layout and distinguish baseline lower-bound selection from stricter tuner incumbent checks |

No evaluation, witness-pruning, exact-geometry, SLSQP, distillation or core-selection
algorithm was silently altered. In particular, the baseline's lower-endpoint record
policy was **documented**, not replaced by the tuner policy. The prototype was not
advertised as generic k splitting or as a new 4D distillation implementation.

The d4_m61 example lacked a standard candidate directory in its source analysis archive.
Its original decimal coordinates, witness and partition were assessed with the canonical
exact-state tuner and Python coloring backend, and a compatible certificate was exported.
This is a re-evaluation of the supplied set, not an optimization improvement.

## New integration tooling

The new `scripts/` helpers provide local build/setup, environment checks, a unified unittest
runner, finite-example verification, asset checks, tiny integration smokes and a bounded
experiment launcher. The launcher uses explicit argv, no shell, fresh outputs, hashes and
process-group termination. It is not a sandbox or a mathematical verifier. New tests
exercise those boundaries and GIF output, plus the native ABI and saved skeleton cover.

The root AGENTS file is a concise repository map with invariant/access/compute rules.
The original longer template remains under docs/history. The project Codex TOML contains
no authentication state, model pin, unsafe approval setting, remote hook or automatic task.
Setup does not install Codex or alter user/global Codex settings.

## Checks and limits

See `validation/README.md` for this assembly's exact test counts and logs. The checks
include the integrated regression suite with/without a native library, finite-example
Python verification, bounded CLI runs, headless graphics, and relocation into a clean
extracted directory. Historical README claims are not counted as newly run tests.

No actual Codex login/session, live Kissat/PySAT integration, independently checked SAT
proofs, native Windows test, macOS test or hours-long performance campaign was performed.
No prebuilt shared library, virtualenv, Git identity/remote, credential, or font file is
shipped. Build the native code with the compiler on the target machine.
