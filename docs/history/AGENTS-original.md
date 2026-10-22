# AGENTS.md

## Purpose

This repository is experimental research software for finite Borsuk/minimax-diameter problems.
For a finite set X in R^d and k parts,

    R_k(X) = min over k-partitions of X
             (maximum part diameter) / diameter(X).

Main cases: (d,k) = (3,4), (4,5), and sometimes (5,6).

The code combines exact graph-coloring checks, non-k-colorable witness graphs,
continuous geometry optimization, point deletion/insertion/splitting, structured
geometric generators, distillation cycles, and parallel experiments.

Correctness of reported bounds is more important than code brevity or speed.

## Repository map

Inspect the repository before assuming every file below exists.

- `borsuk_search.py`: main evaluator/search engine and coloring backends.
- `dsatur_native.cpp`: native exact coloring backend.
- `tune_solution.py`: witness pruning, deletion/insertion, SLSQP tuning.
- `split_tune.py`, `paired_splits.py`: vertex-splitting experiments.
- `distill_search.py`: fit -> resample -> optimize cycles.
- `core_seed.py`, `structured_seed.py`: structured initial-set generators.
- `surface_search.py`, `edge_search.py`: convex-hull boundary experiments.
- `view_solution.py`: visualization and GIF export.
- `examples/`, `validation/`, `experiments/`: reproducibility artifacts.

If detailed docs exist under `docs/`, treat them as the source of truth and keep
this file as a map plus the most important invariants.

## Non-negotiable mathematical invariants

### SAT semantics

Coloring has three logical outcomes: SAT, UNSAT, UNKNOWN/TIMEOUT.

- Only definite UNSAT may justify non-k-colorability.
- UNKNOWN/TIMEOUT must remain unresolved.
- Never convert a timeout into UNSAT for binary search, pruning, deletion, or splitting.

### Bounds are not optimizer objectives

A non-k-colorable witness H gives

    R_k(X) >= min_{uv in E(H)} dist(u,v) / diameter(X).

A valid k-coloring/partition gives an upper bound from the largest same-color distance.

Smooth losses, LP gains, SLSQP objectives, fitting residuals, and visual structure are
heuristics until the resulting point set is reevaluated by the discrete evaluator.

### Threshold indexing

If distances satisfy d[0] >= d[1] >= ... and q is the FIRST prefix length whose
graph is non-k-colorable, the threshold value is

    d[q-1] / d[0]

not `d[q] / d[0]`. Handle ties without artificial perturbation.

### Saved coordinates and certificates

- Final decimal CSV coordinates may represent exact rationals; preserve that semantics.
- Prefer integer squared-distance comparisons for final saved bounds.
- `.npy` files are convenience copies unless a check explicitly concerns binary floats.
- Do not replace an exact-coordinate certificate with tolerance-only floating arithmetic.

### Inputs and records

- Never silently modify an input configuration.
- Write new experiments to new output directories.
- Preserve enough metadata to reproduce a claimed result.
- A heuristic candidate replaces the verified incumbent only after certified comparison.

### Geometric fitting

A smaller fitting residual is not a better Borsuk configuration.
Structured geometry is a proposal prior unless the task explicitly studies the model itself.
Do not project all points onto a fitted model and then claim the old ratio is preserved.
Keep poorly fitted/exceptional points unless a separate ratio-preserving deletion proves
that they are unnecessary.

## Coloring backends

Supported backends may include external SAT (e.g. Kissat), PySAT, native DSATUR,
and pure Python.

Resolve the native library relative to the Python module, not the current directory:

```python
native = Path(__file__).resolve().with_name("dsatur_native.so")
```

Typical Linux build:

```bash
c++ -O3 -std=c++17 -shared -fPIC dsatur_native.cpp -o dsatur_native.so
```

When practical, independently verify an important final witness with a second backend.

## Core terminology

Be explicit about which "core" is meant:

- subset preserving the exact normalized ratio;
- fixed-diameter threshold core;
- inclusion-minimal non-k-colorable graph;
- vertex-critical graph;
- core merely found by one greedy deletion order.

A greedy reduction does not generally prove minimum core size.

## Search and tuning rules

When deleting a vertex, recompute the relevant diameter and prove the required bound.

When inserting a point, remember that adding an isolated vertex to an already UNSAT graph
proves nothing about the usefulness of the new vertex.

Temporary coincident copies are allowed in vertex-splitting scaffolds when their graph
constraints differ. Accepted final point sets should be checked for distinctness.

Neutral intermediate graph moves may be retained in exploratory branches. Keep the
verified incumbent separate so such exploration cannot degrade it.

Witness pruning is useful: removing an edge only after a definite UNSAT check can expose
new geometric motions while preserving a valid lower-bound witness.

## Structured geometry currently under study

Empirical models, not theorems:

- R^3: several long circular arcs, short curved tripod branches, reflection-related
  structure, plus essential exceptions.
- R^4: six spherical-patch-like regions arranged approximately as two orthogonal triples,
  plus an irregular/exceptional component.

For distillation cycles:

1. optimize a finite configuration;
2. fit only the well-modelled subset;
3. keep exceptions in the configuration;
4. resample from fitted geometry and near exceptions;
5. release coordinates again during optimization;
6. judge progress only by reevaluated partition bounds.

## Development workflow

Before editing:

1. Inspect relevant implementation and tests.
2. Identify which mathematical invariant the change touches.
3. Prefer a localized patch unless refactoring is the task.
4. Preserve CLI compatibility when practical.

After editing:

1. Run focused tests.
2. Run the relevant regression suite.
3. Run a small end-to-end smoke experiment.
4. If evaluation/certification changed, cross-check with another backend when feasible.
5. Report exactly what was and was not tested.

## Experiment policy

Before a long run, first test the exact command on a small budget.

Record for substantial experiments:

- command line;
- random seed;
- d, k, point count;
- coloring backend and timeout;
- optimizer/pruning budgets;
- input filename/hash;
- best lower and upper bounds;
- stop reason;
- elapsed time when available.

Prefer JSON/JSONL logs in addition to console output.
Never claim global optimality from heuristic experiments.

## Parallel searches

For multiprocessing changes:

- use deterministic, distinct worker seeds;
- avoid hidden mutable globals;
- avoid BLAS oversubscription;
- make output publication race-safe;
- preserve per-worker checkpoints after interruption.

Early stopping is a heuristic throughput optimization. It must not alter SAT semantics or
certificate logic, and protected full-budget restarts should remain possible.

## Testing priorities

Include regression cases for:

- tied distances and threshold off-by-one errors;
- SAT/UNSAT/UNKNOWN behavior;
- solver timeouts;
- duplicate or near-duplicate points;
- diameter changes after deletion/insertion;
- checkpoint continuation;
- native-library path resolution;
- multiprocessing reproducibility;
- invalid witness/partition files.

For small graphs or point sets, use exhaustive enumeration when practical.
For a new optimizer or geometric move, include at least one synthetic case with an
independently known expected outcome.

## Performance

Correctness first. Safe performance work includes:

- reusing graph oracles and caches;
- NumPy-vectorized distance calculations;
- UNSAT-preserving witness pruning;
- lazy/working-set geometric constraints;
- early stopping of clearly uncompetitive restarts;
- parallel independent restarts.

For performance patches, benchmark on a deterministic small workload and verify that the
scientific result is unchanged or logically equivalent.

## Style

- Python 3.10+.
- Prefer NumPy/SciPy over adding large dependencies without clear benefit.
- Use explicit status objects/enums when UNKNOWN is possible.
- Use zero-based vertex indices in diagnostics and files unless a format requires otherwise.
- Avoid broad exception handlers around solver logic.
- Comments should explain mathematical/algorithmic reasons, not obvious syntax.

## Definition of done

An algorithmic task normally requires:

- relevant invariants preserved;
- focused and regression tests passing;
- one end-to-end smoke run;
- reproducible command/seed;
- heuristic and certified quantities clearly distinguished.

A claimed new record additionally requires saved coordinates, a definite non-k-colorable
lower-bound witness, an explicit upper-bound coloring/partition when claiming the exact
finite-set value, and an independent backend check when feasible. Never describe such a
result as a proof of global optimality.
