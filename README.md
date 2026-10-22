# Minimax diameter ratio search and best-known configurations

Research software for finding finite point sets that are difficult to partition into parts of small diameter. The repository includes configurations in R³–R⁶, a dimension-independent evaluator, search and tuning tools, exact-decimal geometry certificates, fitted templates, and regression tests.

For a finite set X and k parts, the objective is

    R_k(X) = minimum over k-partitions of X of
             (largest part diameter) / diameter(X).

We maximize this ratio. A heuristic search score is not an exact value: a non-k-colorable witness gives a lower bound, and an explicit partition gives an upper bound. See [mathematics](docs/mathematics.md).

The [algorithms companion paper (PDF, v5)](paper_v5.pdf) describes the search,
continuous optimization, vertex splitting, and verification procedures.
The paper accompanies
[*A counterexample to Gale's conjecture in R³*](https://arxiv.org/abs/2610.28828).

## Best-known results in this repository

The strongest stored configurations for each dimension are listed below, all with
k = d + 1 parts. “Best known” refers to the configurations collected here; it does
not assert a literature-wide record or global optimality. Larger ratios give stronger
lower bounds for the worst-case partition problem in that dimension.

| Dimension d | Parts k | Points | Lower bound on R_k(X) | Upper bound on R_k(X) | Coordinates | Certificate |
|---:|---:|---:|---:|---:|---|---|
| 3 | 4 | 101 | 0.889281162393 | 0.889281162394 | [CSV](best_known/3d_101p_088928/points.csv) | [JSON](best_known/3d_101p_088928/certificate.json) |
| 4 | 5 | 61 | 0.913148536990 | 0.913148536991 | [CSV](best_known/4d_61p_091314/points.csv) | [JSON](best_known/4d_61p_091314/certificate.json) |
| 5 | 6 | 87 | 0.925763634680 | 0.925763634681 | [CSV](best_known/5d_87p_092576/points.csv) | [JSON](best_known/5d_87p_092576/certificate.json) |
| 6 | 7 | 64 | 0.936967247964 | 0.936967247965 | [CSV](best_known/6d_64p_093696/points.csv) | [JSON](best_known/6d_64p_093696/certificate.json) |

These are outward-rounded bounds for each **specific saved finite set**, not upper
bounds over all sets in R^d. In all seven stored certificates, the exact squared lower
and upper endpoints coincide; the displayed gap comes from decimal rounding. The
non-colorability check remains part of the trusted computation: no independent
DRAT/LRAT proof checking is provided.

Earlier 3D configurations are retained for comparison and reproduction:

| Points | Lower bound on R_4(X) | Upper bound on R_4(X) | Files |
|---:|---:|---:|---|
| 65 | 0.888111502101 | 0.888111502102 | [Coordinates and certificate](best_known/3d_65p_088808/) |
| 81 | 0.888711925898 | 0.888711925899 | [Coordinates and certificate](best_known/3d_81p_088871/) |
| 89 | 0.888892346465 | 0.888892346466 | [Coordinates and certificate](best_known/3d_89p_088889/) |

For the [65-point 3D configuration](best_known/3d_65p_088808/), canonical `points.csv`
is a byte-identical copy of `points65_precise.csv`. The original five-decimal
`points65.csv` is retained as a separate representation and is not covered by this
certificate. The verified value above supersedes the approximate directory-name score;
[provenance](best_known/3d_65p_088808/provenance.json) and
[native/Python verification](best_known/3d_65p_088808/verification.json) are included.

### Inspect and verify a result

In each certified directory, `points.csv` defines the exact rational coordinates;
`points.npy` is a numerical convenience copy. `witness_edges.csv` specifies the
non-k-colorable graph establishing the lower bound, and `partition.json` supplies
the partition establishing the upper bound. `certificate.json` records the integer
squared-distance bounds; `witness.cnf` is the coloring instance, not an UNSAT proof.
Additional optimization and lineage files describe how a configuration was obtained.

After setup, recheck a result or all seven certificates with:

```bash
python borsuk_search.py verify best_known/3d_101p_088928 \
  --backend native --sat-timeout 30
python scripts/verify_examples.py --backend native --sat-timeout 30
python scripts/check_assets.py
```

The batch verifier retains its original script name and checks `best_known/`.
Use `--backend python` when the native
library is unavailable. A timeout means the witness was not verified, not that it
is colorable. See [mathematical contracts](docs/mathematics.md) for the exact scope
of the certificates.

The [assembly notes](docs/assembly.md) and [archived validation](validation/README.md)
describe an older collection. Historical paths and test counts there are provenance,
not the inventory of the current `best_known/` directory.

Some legacy regression fixtures are absent from this checkout; see
[test prerequisites](tests/README.md) before running the full suite. These missing
fixtures do not affect verification of the seven certificates listed above.

## Start here: Linux or a Linux environment under WSL

Prerequisites: Python 3.10+, Git, and a C++17 compiler for the optional native backend. Python's `venv` and `pip` must be installed. Codex is a separate, user-installed tool; the research code does not require an OpenAI API key.

```bash
# Clone or download the repository, then enter its directory.
cd borsuk_project

# Creates .venv, installs scientific dependencies, and compiles DSATUR locally.
bash scripts/setup.sh
source .venv/bin/activate

python scripts/run_tests.py --suite quick
python scripts/smoke.py
```

The setup script downloads Python dependencies; review and run it yourself before an offline agent session. It does not install Codex, sign in, alter your global configuration, or initialize a Git repository.

Without a compiler, use `bash scripts/setup.sh --python-only` and pass `--backend python` to experiments and smoke tests. The Python backend is useful for checking small instances but may be much slower on large ones. The optional PySAT requirements are separate. `requirements-tested.txt` records the assembly environment, not a universal lock file.

The assembly tests ran on Linux x86_64 with Python 3.13.5. Native Windows and macOS were not exercised. The build helper has a macOS branch, but that is not a portability-test claim. Use a Linux/WSL environment for the documented process-group timeout behavior.

## Open this repository in Codex

Install/authenticate Codex using the [official CLI instructions](https://developers.openai.com/codex/cli/). Then, from this directory:

```bash
codex
```

Use your own Git identity; no identity, remote, or credentials are supplied.

First task to paste into Codex:

> Read AGENTS.md and docs/architecture.md. Run the quick tests and the tiny smoke run using the existing virtual environment. Do not install dependencies or change source code. Report the module map, test results, and any unresolved failures. Do not launch a record search.

[Codex setup and permissions](docs/codex.md) explains the included conservative `.codex/config.toml`. [Task briefs](docs/tasks/README.md) provide bounded follow-on tasks.

## What is included

```text
AGENTS.md                       Agent map, mathematical invariants, compute policy
borsuk_search.py                 Evaluator, solver adapters, baseline parallel search
 dsatur_native.cpp               Optional graph-only C++ coloring backend
 tune_solution.py                Exact-decimal tuning, deletion/insertion, SLSQP/LP
 split_tune.py                   One-step essential vertex splits
 paired_splits.py                Two-step split prototype: k=4 only
 distill_search.py               Exception-preserving 3D fit/resample/optimize loop
 curve_model.py                  Distillation model fitting and sampling
 core_seed.py                    Fresh structured seeds and core screening: 3D/4D
 structured_seed.py              Anchored 3D curve-based seed generator
 compare_curves.py               Refit comparison for the original 81-point template
 surface_search.py              Fixed-hull surface insertion experiments
 edge_search.py                  3D hull-edge insertion and continuous edge covers
 verify_skeleton.py              Independent segment-cover geometry checker
 view_solution.py                Static plots and rotating GIFs; axes hidden by default
scripts/                        Setup, build, diagnostics, tests, experiment runner
configs/                        Small, explicit JSON experiment presets
tests/                          Consolidated inherited tests + workspace tests
templates/                      Fitted models and matching source coordinates
best_known/                     Best stored configurations and earlier comparison sets
docs/                           Architecture, mathematics, development, experiments
validation/                     This assembly's test logs and verification reports
experiments/                    Fresh local outputs (Git-ignored, except guidance)
```

Each working source module has one canonical copy. Historical documentation lives under `docs/history/`; old runnable code copies and bulk optimization trajectories are not shipped.

## Choose a workflow

| Goal | Entry point | Output to use next |
|---|---|---|
| Ordinary multistart search | `borsuk_search.py search` | `run/best/` |
| Evaluate/verify finite coordinates | `borsuk_search.py evaluate` / `verify` | Candidate directory |
| Tune a good existing set | `tune_solution.py` | `run/best/` |
| Essential single split | `split_tune.py` | `run/best/` |
| Two neutral/successive splits | `paired_splits.py` | Prototype output; read limitations first |
| Retain exceptions and alternate fitting/optimization | `distill_search.py search` | **Run root** to retain the whole beam |
| Generate fresh large-core candidates | `core_seed.py generate` | `run/selected/` |
| Audit a fixed-scale/normalized core | `core_seed.py inspect` | Diagnostic JSON, not a new minimum-core proof |
| Add surface or edge samples | `surface_search.py`, `edge_search.py` | `latest/` to retain neutral additions |
| View or animate | `view_solution.py` | PNG/GIF, not a certificate |

Run `python MODULE.py --help` before adapting commands. Flags with the same name can have different meanings across scripts; [CLI pitfalls](docs/development.md#cli-pitfalls) lists the important ones.

## A bounded first experiment

```bash
python scripts/run_experiment.py \
  --preset configs/smoke_search4.json --name first-r4 --backend native

python borsuk_search.py verify experiments/first-r4/result/best \
  --backend python --sat-timeout 30
```

The runner records command arguments, input/source hashes, Git state, dependency versions, exit status, wall time, and console output. It does **not** infer a scientific result from a zero exit code. The default overall limit is 300 seconds; budgets above 600 seconds require the explicit `--allow-long` gate as well as human authorization for an agent-run experiment.

```bash
# A core-aware R4 initializer with a deliberately tiny screening budget:
python scripts/run_experiment.py \
  --preset configs/smoke_core4.json --name first-core4 --backend native

# Animate the best stored 3D configuration:
python view_solution.py best_known/3d_101p_088928 \
  --hull --gif experiments/record101.gif --frames 72 --no-show
```

These presets are functionality checks, not competitive search budgets.

## Documentation map

- [Architecture and change routing](docs/architecture.md)
- [Mathematical and certificate contracts](docs/mathematics.md)
- [Development and testing](docs/development.md)
- [Experiment runner, budgets, and reproducibility](docs/experiments.md)
- [Experiment report template](docs/EXPERIMENT.md)
- [Geometric model scope and exception handling](docs/geometric-models.md)
- [Codex installation handoff and local workflow](docs/codex.md)
- [Best-known results](#best-known-results-in-this-repository) and [template catalog](templates/README.md)
- [Assembly provenance / compatibility changes](docs/assembly.md)
- [Validation performed for this package](validation/README.md)

A second coloring implementation helps catch bugs; it is not independent DRAT/LRAT proof checking. No SAT proof checker or proof-producing backend has been added. No global optimality claim is implied by any example or successful test.
