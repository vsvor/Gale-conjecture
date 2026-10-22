# Architecture

This document describes the current architecture of the finite Borsuk / minimax-diameter research code and gives routing guidance for Codex agents. It is intentionally about **responsibility boundaries**: which module owns which invariant, which modules are allowed to depend on which lower layers, and where a proposed change should be implemented.

For the mathematical problem, a finite set \(X\subset\mathbb R^d\) and an integer \(k\) are scored by

\[
R_k(X)=\min_{X=X_1\sqcup\cdots\sqcup X_k}
\frac{\max_i\operatorname{diam}(X_i)}{\operatorname{diam}(X)}.
\]

The project is experimental optimization software, but its saved lower and upper bounds are intended to have a much stricter semantics than its heuristic search steps. Keep that distinction visible in the architecture.

**Consolidated-workspace edition.** This file updates the previously supplied architecture
map to match the assembled source tree. Setup/tests are in `scripts/`; current commands
are in [development.md](development.md). Imported historical READMEs are under
`history/` and are not competing source trees. See [assembly.md](assembly.md) for the
small compatibility edits made during consolidation.

## 1. Architectural layers

The code is easiest to reason about as four layers.

```text
Layer 4: research workflows
    distill_search.py       core_seed.py       paired_splits.py
             |                   |                    |
             +-------------------+--------------------+
                                 |
Layer 3: topology / local-tuning operations
                     split_tune.py
                           |
                     tune_solution.py
                           |
Layer 2: finite-set engine and certification
                     borsuk_search.py
                           |
Layer 1: optional graph-coloring accelerator
                     dsatur_native.cpp
```

The dependency direction should normally remain downward. Higher-level research scripts may orchestrate lower-level operations, but lower-level code should not import a specialized experiment.

More explicitly:

```text
borsuk_search.py
    └── optionally loads dsatur_native.so built from dsatur_native.cpp

tune_solution.py
    └── imports borsuk_search.py

split_tune.py
    ├── imports borsuk_search.py
    └── imports tune_solution.py

paired_splits.py
    ├── imports borsuk_search.py
    ├── imports tune_solution.py
    └── imports split_tune.py

distill_search.py
    ├── imports borsuk_search.py
    ├── imports tune_solution.py
    └── imports geometry/model helpers such as curve_model.py

core_seed.py
    ├── imports borsuk_search.py
    ├── imports tune_solution.py
    └── imports structured-generator helpers such as structured_seed.py
```

In a clean local repository there should be **one canonical copy** of each source module. Experiment output directories may contain frozen copies for reproduction, but Codex agents should not edit those copies as if they were source files.

## 2. Cross-cutting invariants

Before changing any module, identify which of these invariants the change touches.

### 2.1 SAT / UNSAT / UNKNOWN is a three-valued interface

Graph-coloring calls can return:

- `SAT`: a valid \(k\)-coloring was found;
- `UNSAT`: non-\(k\)-colorability was established by the selected backend;
- `UNKNOWN`: typically a timeout or interrupted solver call.

Only `UNSAT` may justify a lower-bound witness. `UNKNOWN` must never be silently converted to `UNSAT` in threshold search, witness pruning, deletion, insertion, splitting, or core extraction.

### 2.2 A heuristic objective is not a certified Borsuk ratio

The following are proposal or optimization quantities only:

- smooth L-BFGS losses;
- fixed-witness ratios during a continuous solve;
- LP first-order gains;
- SLSQP objective values and termination flags;
- curve/surface fitting residuals;
- geometric similarity scores.

A candidate becomes scientifically meaningful only after the finite set is reevaluated through the discrete coloring logic and, for saved results, through the exact-coordinate geometry path.

### 2.3 Threshold indexing and ties are part of correctness

If pair distances are ordered

\[
d_0\ge d_1\ge\cdots
\]

and \(q\) is the first **prefix length** whose graph is not \(k\)-colorable, the threshold distance is `d[q-1]`, not `d[q]`.

Equal-distance levels must not be broken by artificial random perturbations in certification code. The integer evaluator in `tune_solution.py` deliberately treats a tied distance level as a whole.

### 2.4 Saved decimal coordinates are the canonical certificate geometry

The normal saved-result directory contains files such as:

```text
points.csv
points.npy
witness_edges.csv
witness.cnf
partition.json
certificate.json
```

`tune_solution.py` additionally writes `polish.json`. Workflow checkpoints can add
surface-membership, model, lineage and manifest files; those are not interchangeable.

For engine/tuner and ordinary workflow certificates, `points.csv` is the canonical
decimal representation used for integer squared-distance comparisons. `points.npy`
is a convenient numerical copy and may have different exact binary arithmetic.
**Exception:** standalone `structured_seed.py` validates its binary NPY values and
explicitly labels its CSV as a convenience representation. Do not transfer that seed
promise to a later CSV export without recomputing it.

### 2.5 The verified incumbent is separate from exploratory state

Research scripts can keep neutral, temporarily worse, or incompletely polished branches.
The tuner and distillation have strict rational incumbent-comparison policies with
explicit tie/refinement cases. **The baseline `borsuk_search.py` instead selects the
largest lower endpoint**, including its floating search score; it does not prove that
new.lower > every previous.upper. With unresolved brackets, its directory name `best`
is not a proved ordering of exact objectives. Do not silently change this policy during
an unrelated refactor, and do not describe it as the stronger tuner guarantee. See
[mathematics.md](mathematics.md#5-acceptance-policies-are-not-identical-across-workflows).

## 3. `borsuk_search.py`: foundational finite-set engine

### Role

`borsuk_search.py` is the lowest Python layer and should remain the **source of truth for generic finite-set evaluation, graph-coloring backends, baseline multistart search, and the base saved-candidate format**.

It owns four main responsibilities.

#### A. Generic graph-coloring interface

`GraphOracle` presents one interface over:

- external DIMACS solvers such as Kissat / MiniSat;
- incremental PySAT backends;
- the native DSATUR shared library;
- the pure-Python DSATUR fallback.

Backend-specific parsing, timeouts, model validation, and status mapping belong here.

#### B. Logical evaluation using floating-point distance ordering

`evaluate()` orders pair distances and searches for the transition between colorable and non-colorable threshold graphs. It returns an `Evaluation` bracket containing a lower bound, an upper bound, an UNSAT witness prefix, and an upper-bound coloring when available.

This is the generic floating-point evaluator used by the baseline search. Do not duplicate threshold-search logic in higher-level scripts unless the exact-decimal representation requires a genuinely different implementation.

#### C. Generic witness optimization

The module provides reusable operations such as:

- `k_core_edges()`;
- `prune_witness()`;
- `witness_ratio()`;
- smooth fixed-witness optimization with L-BFGS;
- final fixed-witness SLSQP polishing;
- generic seed construction, mutation, and inactive-point replenishment.

These routines know about point geometry and a non-colorable witness, but they should not know about vertex splitting, fitted arcs, beam search, or a particular record configuration.

#### D. Search orchestration and base certification

The `search` action runs independent restarts, refreshes witnesses, optimizes coordinates, applies basin-hopping mutations, optionally performs live shared-record early stopping, and publishes per-restart checkpoints.

The `evaluate` action evaluates an existing coordinate file and exports a candidate. The `verify` action rechecks a saved witness and recomputes the integer geometry in a saved result directory.

### Changes that belong here

Put a change in `borsuk_search.py` when it affects **all or most search/tuning workflows**, for example:

- a new coloring backend or a change to solver status semantics;
- the pair-index convention;
- threshold evaluation or tie handling;
- base witness pruning;
- generic smooth fixed-witness optimization;
- generic save/verify certificate semantics;
- baseline seed families used by `borsuk_search search`;
- multiprocessing of independent baseline restarts;
- shared-record early stopping for baseline restarts.

### Changes that do not belong here

Do not put specialized research moves here merely because several experiments call the engine. In particular, avoid adding:

- vertex-splitting combinatorics;
- model fitting for arcs, tripods, or 4D patches;
- beam/population management;
- core-aware seed ranking;
- special logic for a particular saved record.

If a specialized module discovers a utility that is genuinely generic and used by several sibling modules, promote only that utility after adding engine-level tests.

## 4. `tune_solution.py`: exact-decimal state and local/topological tuning toolkit

### Role

`tune_solution.py` sits immediately above the engine. It is the main reusable toolkit for **post-processing an existing good configuration**.

Its key architectural addition is an exact-decimal state model:

- `Geometry` stores integer coordinate rows plus their decimal scale and precomputes exact squared distances;
- `State` combines geometry, witness, upper-bound coloring, exact rational lower/upper squared ratios, and numerical polish diagnostics;
- `OraclePool` reuses `GraphOracle` instances for several point counts.

The central `assess()` function performs integer-distance threshold search, seeded when possible by a known witness and a known coloring. It keeps a rigorous bracket if a coloring call becomes `UNKNOWN`.

### Main operations

`tune_solution.py` owns the reusable post-search operations:

- **ratio-preserving vertex deletion** with recomputation of the subset diameter;
- **alternative witness construction and pruning**;
- **SLSQP constraint formulation** for a known non-colorable witness;
- **numerical first-order LP audit**;
- **repeated polishing / witness refresh**;
- **geometric point insertion proposals**, including color-extension and redundancy screening;
- **checkpoint acceptance and history publication**;
- export in the same candidate/certificate family used by the base engine.

The module is allowed to be more conservative and exact than `borsuk_search.py`; this is where expensive tuning logic for already-promising configurations belongs.

### Changes that belong here

Use `tune_solution.py` for changes such as:

- a better generic ratio-preserving deletion rule;
- a new generic insertion proposal inside the diameter-completion body;
- stronger insertion nonredundancy / list-coloring checks;
- generic alternative-witness generation;
- improved SLSQP constraint generation, polishing, or stationarity diagnostics;
- exact-decimal state/certificate handling shared by tuning scripts;
- checkpoint rules shared by several post-processing workflows.

### Changes that do not belong here

Keep out combinatorics that are specific to one graph surgery, such as the definition of a split vertex and omission sets. Keep out fitted geometric models and population/beam policies.

A useful test is: **could the operation sensibly be applied to an arbitrary saved configuration without knowing how that configuration was generated?** If yes, `tune_solution.py` is a plausible home.

## 5. `split_tune.py`: one-step essential vertex splitting

### Role

`split_tune.py` implements a specific graph-topology move: replace a vertex by two initially coincident copies whose incident witness constraints differ.

It is deliberately above `tune_solution.py` because it relies on the tuner for:

- pair/edge ID conversion;
- oracle management;
- exact state assessment;
- SLSQP polishing;
- first-order diagnostics;
- saving/checkpointing.

The split-specific responsibilities are:

- mapping an old edge set to the enlarged graph (`split_edge_ids`);
- enumerating disjoint omission sets (`omission_pairs`);
- detecting which near-active edges are color-critical;
- testing candidate split graphs for definite non-colorability;
- ranking splits by the numerical LP gain;
- allowing selected zero-gain candidates to receive a finite perturbation;
- reevaluating distinct optimized configurations before publication.

### Changes that belong here

Use `split_tune.py` for **single-split combinatorics or screening**, for example:

- alternative omission-set enumeration;
- stronger essentiality conditions for the two copies;
- better one-step split ranking;
- split-specific diagnostics;
- genericization of a single split to other `k` values.

### Escalation rule

If a proposed feature needs a sequence of graph surgeries before any geometric optimization, it probably belongs in a higher-level split workflow rather than in this module's core one-step search loop.

## 6. `paired_splits.py`: multi-step / neutral split experiment

### Role

`paired_splits.py` is a specialized research script created to test a phenomenon that the one-step splitter misses: a first split may have zero numerical gain but enable a productive second split.

It composes the public machinery from:

- `borsuk_search.py` for geometry and witness ratios;
- `tune_solution.py` for state/oracles/SLSQP;
- `split_tune.py` for split edge mapping and omission enumeration.

The current script searches two successive essential splits, retains some zero-LP-gain paired scaffolds, optionally jitters them, and optimizes the combined graph.

### Important current limitation

The implementation is experiment-oriented and assumes `k=4` in several places. It is
not a generic engine primitive. Assembly adds an early rejection for a saved input
explicitly declaring another k; raw arrays still assume k=4. The first-split family is
enumerated before `--first-limit` slicing, so that option is not a total wall-time bound.
Use the external bounded runner for exploratory calls.

### Changes that belong here

Put changes here when they concern **multi-step split exploration**, for example:

- two-step or shallow split trees;
- retaining neutral first moves;
- search ordering over first/second splits;
- branch budgets for paired-split experiments.

If multi-step splitting becomes a stable, reusable family of operations, refactor the reusable combinatorics into a dedicated module and keep `paired_splits.py` as an orchestration CLI. Do not push a large search tree into `borsuk_search.py`.

## 7. `distill_search.py`: fit → resample → optimize research workflow

### Role

`distill_search.py` is a **high-level population workflow**, not a new evaluator.

Its purpose is to alternate:

1. fitting a structured geometric model to the current configuration;
2. keeping exceptional / poorly modelled points as actual vertices;
3. resampling additional points from the fitted model in a diameter-safe way;
4. releasing coordinates again for free geometric optimization;
5. refitting the descendants;
6. maintaining a verified incumbent separately from a beam of exploratory states.

It delegates finite-set correctness to `borsuk_search.py` and exact-decimal tuning to `tune_solution.py`. Model-specific geometry is supplied by helper modules such as `curve_model.py` rather than implemented in the graph engine.

The module also owns lineage/checkpoint integrity for the distillation process: model files, protected exception indices, coordinate hashes, and continuation of a beam run.

### Changes that belong here

Use `distill_search.py` for changes involving the **alternating workflow**, such as:

- beam size and child selection;
- how many growing versus control children are created;
- when to refit;
- how fitted-model proposals are mixed with exception-neighborhood proposals;
- lineage and continuation of a distillation run;
- parallel evaluation of independent children;
- acceptance policy for neutral descendants into the exploratory beam.

### Changes that should be elsewhere

- A new curve/surface fitting primitive belongs in the geometry/model helper layer.
- A generic exact evaluator change belongs in `borsuk_search.py` or `tune_solution.py`.
- A generic insertion validity test belongs in `tune_solution.py`.
- A fresh-start initializer whose main purpose is screening core size belongs in `core_seed.py`.

The model is always a **proposal prior**. Never make `distill_search.py` equate a smaller model residual with a better Borsuk ratio.

## 8. `core_seed.py`: fresh structured initialization and core-aware screening

### Role

`core_seed.py` addresses a different question from distillation: **how to generate fresh starting configurations that are likely to retain a large obstruction after a short warm-up**.

It intentionally does not anchor all points of an old record. Instead it uses learned geometric templates as proposal distributions while retaining only designated exceptional points. Current structured proposal families cover the empirical 3D curve model and 4D patch model.

The module provides:

- component quota allocation;
- candidate pools from fitted templates;
- spatial maximin selection;
- distance-signature feature selection;
- a short no-deletion warm-up using the generic witness optimizer;
- fixed-threshold core diagnostics;
- mandatory-vertex tests;
- repeated greedy core extractions;
- optional normalized non-degrading reductions using `tune_solution.py`;
- ranking of candidate seeds within a ratio window.

### Core terminology matters here

This module deliberately distinguishes:

- a vertex proven mandatory for a particular threshold graph;
- a core found by one or more greedy deletion orders;
- a fixed-diameter induced non-colorable core;
- a subset preserving the normalized Borsuk ratio after its own diameter is recomputed.

Do not collapse these into one field called simply `core_size`.

### Changes that belong here

Use `core_seed.py` for:

- new fresh structured proposal distributions;
- alternative spatial or graph-signature diversity features;
- seed quotas and minimum separation rules;
- short warm-up policy used only for seed screening;
- fixed-threshold core metrics and seed-selection policy;
- benchmarking the expectation of surviving core size across initializer families.

### Changes that do not belong here

Do not turn `core_seed.py` into the main optimizer. Once a seed has been selected, long searches should be handed to `borsuk_search.py`, `tune_solution.py`, or a higher-level workflow.

## 9. `dsatur_native.cpp`: optional exact coloring accelerator

### Role

`dsatur_native.cpp` is deliberately narrow. It implements an exact DSATUR-style \(k\)-colorability search with a C ABI:

```cpp
extern "C" int graph_color(
    int n,
    int k,
    int ne,
    const int* edges,
    double seconds,
    int* colors
);
```

The return-code contract is:

```text
 1  SAT / colorable; output colors are filled
 0  UNSAT / not k-colorable
-1  timeout
-2  invalid input or internal failure
```

The current ABI limits are `0 <= n <= 10000` and `1 <= k <= 63` (the Python oracle requires positive n). The implementation performs graph-level preprocessing such as low-degree peeling and uses DSATUR-style branching. It has no knowledge of Euclidean points, distances, witnesses, Borsuk ratios, or certificates.

`borsuk_search.py` is responsible for loading the compiled shared library using a path resolved relative to the Python module, mapping return codes into `Status`, and validating any returned coloring.

### Changes that belong here

Use `dsatur_native.cpp` only for **graph-coloring performance or correctness**:

- better DSATUR branching;
- safe graph preprocessing;
- cache-friendly adjacency structures;
- timeout checking;
- ABI-preserving performance work.

Do not add geometry or experiment logic to C++ merely for speed. If the ABI changes, update the loader and add explicit cross-backend tests in the same change.

## 10. Tests are part of the architecture

The test suite is not just regression coverage; it defines boundaries between heuristic research code and correctness-critical code.

The consolidated test layout includes:

```text
tests/
    test_borsuk_search.py
    test_early_stop.py
    test_tune_solution.py
    test_split_tune.py
    test_paired_replay.py
    test_distillation.py
    test_core_seed.py
    test_structured_seed.py
    test_surface_search.py
    test_native_backend.py
    test_workspace.py
    test_experiment_runner.py
    test_boundary_workspace.py
    test_viewer_workspace.py
```

Inherited tests were relocated and their fixture paths adjusted without replacing
assertions. `scripts/run_tests.py` is the unified unittest entry point; it also supports
focused suites and rejects zero-test discovery. `tests/README.md` maps those suites.
`test_workspace.py` and `test_experiment_runner.py` own assembly/resource/provenance
behavior rather than mathematical search. Native and viewer checks are explicit.
Current execution results are in `validation/`, not inferred from historical test counts.

### `test_borsuk_search.py`

This is the foundational suite. It should cover:

- graph coloring against exhaustive small graphs;
- CNF encoding;
- external solver parsing and timeouts;
- witness pruning preserving definite UNSAT;
- evaluator comparison with exhaustive partitions;
- tied distances and threshold indexing;
- UNKNOWN returning a bracket rather than a false exact result;
- analytic gradients;
- optimizer non-degradation of a known witness;
- exact export and verification.

Any change to evaluator semantics, pair indexing, solver adapters, or base certificate format requires this suite.

### `test_early_stop.py`

This suite owns the throughput heuristic around independent restarts:

- schedule parsing;
- protection/warmup rules;
- record-holder behavior;
- live shared-record publication;
- UNKNOWN upper-bound protection;
- skipping remaining rounds and final polish only when intended;
- multiprocessing under available start methods;
- native-library path resolution.

Early stopping must remain orthogonal to certificate correctness.

### `test_tune_solution.py`

This suite should exercise:

- exact-decimal loading and edge-ID mapping;
- integer tied-level evaluation;
- ratio-preserving deletion and diameter recomputation;
- UNKNOWN blocking deletion/insertion claims;
- insertion essentiality, twins/domination, and cooperative insertions;
- SLSQP Jacobians and first-order audits;
- checkpoint monotonicity;
- compatibility of tuner exports with the base verifier.

### `test_split_tune.py`

This suite owns one-step split correctness:

- mapping edges under a split;
- valid/invalid omission sets;
- essential non-colorable examples;
- rejection of colorable or UNKNOWN split candidates;
- LP selection behavior;
- at least one regression based on an actual research configuration.

### Paired-split tests

Tests for `paired_splits.py` should check the graph scaffold independently of whether a particular stochastic run finds a record. In particular:

- the second split should match an explicitly reconstructed edge set;
- the combined scaffold should have the intended non-colorability;
- copy essentiality should be checked on a small deterministic case;
- input files must remain unchanged;
- neutral first moves should be representable without being falsely published as improvements.

### `test_distillation.py`

This suite should focus on workflow integrity:

- curve/model projection primitives;
- reflection or other declared symmetry transformations;
- exceptions excluded from fitting but not deleted;
- arclength sampling;
- exact diameter-safe insertion, including new-new distances;
- duplicate rejection;
- exact acceptance guards;
- neutral descendants entering the exploratory beam;
- checkpoint hash/lineage validation;
- deterministic continuation and multiprocessing smoke tests.

### `test_core_seed.py`

This suite should check:

- quota allocation and proposal selectors;
- reproducibility;
- retention of designated exceptions;
- fresh sampling rather than accidental anchoring of all old points;
- core diagnostics on small graphs where exhaustive answers are known;
- timeout-conservative mandatory/core tests;
- normalized reducer non-degradation;
- seed-selection gates;
- warm-up preserving the intended point count.

## 11. Where should a Codex change go?

Use this routing table before editing.

| Requested change | Primary module | Usually also test |
|---|---|---|
| New SAT solver / parser | `borsuk_search.py` | `test_borsuk_search.py` |
| Native DSATUR speedup | `dsatur_native.cpp` | cross-backend tests in `test_borsuk_search.py` |
| Threshold/tie/evaluation fix | `borsuk_search.py` | exhaustive objective tests |
| Saved certificate semantics | `borsuk_search.py` and, if exact-state format changes, `tune_solution.py` | base + tuner export tests |
| Baseline initializer for ordinary multistart search | `borsuk_search.py` | seed/geometry tests |
| Baseline restart scheduling / early stopping | `borsuk_search.py` | `test_early_stop.py` |
| Generic witness pruning | `borsuk_search.py` | base pruning tests plus tuner smoke test |
| Exact-decimal post-processing | `tune_solution.py` | `test_tune_solution.py` |
| Ratio-preserving deletion | `tune_solution.py` | deletion tests |
| Generic point insertion | `tune_solution.py` | insertion tests |
| SLSQP/LP polishing | `tune_solution.py` | optimizer/Jacobian tests |
| One essential vertex split | `split_tune.py` | `test_split_tune.py` |
| Neutral first split / multi-step split tree | `paired_splits.py` or a new higher-level split workflow | paired-split deterministic tests |
| Fit/resample/beam alternation | `distill_search.py` | `test_distillation.py` |
| New geometric primitive used by distillation | geometry helper (`curve_model.py`, etc.) | primitive tests + distillation smoke test |
| Fresh learned initializer | `core_seed.py` | `test_core_seed.py` |
| New definition/diagnostic of “core” | `core_seed.py` if screening-only; `tune_solution.py` if generic post-processing | exhaustive small-core tests |
| Experiment-only parameter sweep | experiment runner/config, not a core source module | smoke command + recorded metadata |

## 12. Rules for shared code and refactoring

### Promote downward only when the abstraction is genuinely shared

If `split_tune.py`, `distill_search.py`, and `core_seed.py` all need the same exact state operation, that operation probably belongs in `tune_solution.py`.

If several modules need the same solver/evaluator primitive, it probably belongs in `borsuk_search.py`.

Do not move a function downward solely to avoid one import. Lower layers should stay conceptually smaller and more stable.

### Keep orchestration out of primitives

A helper should not write global experiment output, mutate a verified incumbent, or launch worker pools unless orchestration is explicitly its role. In particular:

- `GraphOracle.solve()` should answer a graph query, not decide which experiment branch wins;
- `prune_witness()` should preserve UNSAT, not publish a candidate;
- a geometric model function should generate/fit geometry, not certify a Borsuk ratio.

### Avoid code-copy drift

Several historical experiment archives include copies of the engine and tuner for reproducibility. In the local development repository:

1. keep canonical source modules at the root (or in a package);
2. store experiment metadata and commit hashes instead of editable source copies;
3. if a frozen source snapshot is needed for a published result, put it under an explicitly immutable reproduction directory;
4. never patch both the canonical file and an old reproduction copy in the same feature change unless the task explicitly asks to refresh that reproduction bundle.

## 13. Suggested development workflow for Codex agents

For an algorithmic change:

1. **Read `AGENTS.md` and this file.**
2. Inspect the target module and its direct dependencies before editing.
3. State which invariant is affected: solver semantics, evaluator, exact geometry, graph surgery, optimizer, or experiment orchestration.
4. Implement the narrowest change in the owning module.
5. Add or update the focused tests listed above.
6. Run the focused suite.
7. Run `test_borsuk_search.py` whenever the change can affect graph/evaluator semantics.
8. Run a small end-to-end smoke experiment with a fixed seed.
9. For an important claimed improvement, verify the saved candidate with another coloring backend when feasible.
10. Report separately:
   - code/test success;
   - heuristic search outcome;
   - certified lower/upper bounds;
   - anything not independently checked.

Do not launch a many-hour parameter sweep before the deterministic tests and a small smoke run pass.

## 14. Architectural anti-patterns

Avoid these changes even if they make a short experiment appear simpler.

### Reimplementing evaluation in an experiment script

If a script needs \(R_k(X)\), call the engine/tuner evaluator. A local binary search with slightly different timeout or tie semantics is a correctness hazard.

### Treating solver timeout as evidence

A timeout is an absence of an answer, not non-colorability. Never use it to delete an edge, delete a point, certify a split, or raise a lower bound.

### Letting a model fit own the incumbent

A curve/surface model may propose points or organize a population. It must not replace the exact Borsuk acceptance check.

### Hiding experiment-specific constants in low-level code

A threshold or grouping learned from one 3D/4D record belongs in the experiment/model layer or a template file, not in `GraphOracle`, `evaluate()`, or generic exact geometry.

### Expanding the C++ backend into a second engine

`dsatur_native.cpp` should stay a graph-coloring service. Geometry, certificate files, and experiment policy belong in Python.

### Calling a greedy core “the minimum core”

Greedy deletion gives a found inclusion-minimal or non-degrading subset under its exact procedure. Unless exhaustive or otherwise proved, it does not determine minimum core cardinality.

## 15. Practical source-of-truth rule

When two modules seem to disagree, use this hierarchy:

1. mathematical invariants documented in `AGENTS.md` and correctness tests;
2. `borsuk_search.py` for generic coloring/evaluation/certificate semantics;
3. `tune_solution.py` for exact-decimal post-processing state and generic tuning operations;
4. specialized workflow modules for their own experimental search policy;
5. historical experiment scripts and frozen reproduction bundles only as records of what was run.

A specialized script may intentionally be less generic than the lower layer. That is acceptable. What it must not do is weaken lower-layer correctness semantics.


## 16. Repository tooling and data ownership

`AGENTS.md` is the short entry map and invariant/compute policy. `.codex/config.toml`
contains only conservative project permissions, no credentials or model pin. Setup,
build and diagnostic commands live in `scripts/`; dependency installation is a human
step separate from bounded agent tasks. Current Codex guidance is in `docs/codex.md`.

`scripts/run_experiment.py` owns overall process timeouts, new output directory creation,
command/input/source hashes, Git state and console logging. It does not evaluate R_k or
choose a scientific incumbent. `configs/` holds explicit small-budget argv presets;
change preset budgets there, not by altering core defaults. `scripts/smoke.py` owns two
tiny integration searches and follow-up Python verification. Process budget outcomes
remain distinct from graph-query UNKNOWN. `scripts/verify_examples.py` checks ordinary
finite certificates; use surface/skeleton verifiers for their additional claims.

`templates/` is versioned fitted geometry with matching source coordinates; `best_known/`
contains historical inputs/checkpoints and `tests/fixtures/` regression inputs. None
is a scratch output directory. `DATA_MANIFEST.json` records their hashes. New scientific
outputs belong in `experiments/`, whose contents are normally ignored by Git. A promoted
example requires explicit review, provenance, and verification. `SOURCE_MANIFEST.json`
records assembled-versus-imported source hashes; update this map when making a new release,
not after every experiment.

The assembled repository intentionally does not migrate the source to an installable
Python package, generalize the paired splitter to k=5, add R4 distillation, or integrate
a SAT proof checker. Those are separate tasks with their own tests and review.
