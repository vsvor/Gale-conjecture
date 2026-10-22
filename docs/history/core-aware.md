# Core-aware initial configurations

Experimental initializers for the Borsuk minimax partition-ratio search in R3
(k=4) and R4 (k=5). This release proposes FRESH configurations using the fitted
geometric families from the previous analysis. It does not change the original
search engine or claim a new ratio record.

## What is different from anchored enlargement?

Keeping a known record X inside S, keeping its diameter, and obtaining the same
ratio means the old core is STILL a core of S. Such enlargement is a safe way to
continue a record but cannot, by itself, force a larger minimum core.

Here the regular part is freshly resampled. Only the previously identified
exceptions are retained, up to a common normalization: 10 source points in R3,
16 in R4. The complete old record is not present. No old lower bound is inherited.
All points, including exceptions, can move during warmup. No projection or
symmetry constraint is imposed on the continuous optimization.

## Installation and use

Python >=3.10; NumPy and SciPy. The optional native DSATUR backend needs a C++ compiler.

```bash
python -m pip install -r requirements.txt
c++ -O3 -std=c++17 -shared -fPIC dsatur_native.cpp -o dsatur_native.so

# Recommended R3 proposal: quota-balanced geometric maximin sampling.
python core_seed.py generate --dim 3 --points 81 --method spatial \
  --attempts 8 --warmup-rounds 2 --steps 40 --core-orders 3 \
  --normalized-audit --min-core 50 --seed 20261010 \
  --backend native --out core3

# R4 alternative: distance-neighbourhood diversity within the six fitted patches.
python core_seed.py generate --dim 4 --points 81 --method signature \
  --attempts 8 --warmup-rounds 2 --steps 40 --core-orders 3 \
  --normalized-audit --min-core 30 --seed 20261010 \
  --backend native --out core4

# Hand a screened, already briefly optimized configuration to the usual optimizer.
python borsuk_search.py search --dim 4 --k 5 --start core4/selected \
  --restarts 12 --workers 4 --rounds 12 --early-stop \
  --backend native --out optimized_core4
```

For Kissat replace `--backend native` by
`--backend external --solver /absolute/path/to/kissat --sat-timeout 60`.
Native and Python were exercised here; actual Kissat/PySAT execution was not.

`generate` uses the bundled fitted templates. `inspect` can audit arbitrary
existing point configurations and does not optimize their coordinates:

```bash
python core_seed.py inspect /path/to/best --k 5 --normalized-audit \
  --core-orders 5 --backend native --out core_audit
```

The source4 audit in `validation/source4_audit.json` finds all 61 vertices
mandatory in its exact threshold graph; the normalized one-vertex reducer also
removes none. This is not a proof of globally minimum cardinality when a subset's
diameter may change.

## Proposal models

R3: the fitted three outer circular arcs and six short circular branches, plus
the ten unresolved points. Component quotas are proportional to arc length.

R4: six S2 patches in affine 3-planes, with the two nearly orthogonal triples
encoded by the preceding fitted template, plus the sixteen unassigned points.
Each patch receives an approximately equal quota. Sampling uses the convex hull
of its OBSERVED local two-dimensional coordinates, with a small expansion. It
never interprets the fitted supporting sphere as evidence for sampling the full
sphere. `--patch-domain gaussian` is an exploratory, more extrapolative variant.

A large candidate pool is generated with small componentwise shape perturbations
and pointwise jitter. Selection methods:

* `sphere`: original uniform-spherical baseline, with no retained exceptions.
* `iid`: uniform random selection within quotas, with a spacing rejection. Despite
  the short option name, quotas and rejection mean the full configuration is not IID.
* `spatial`: greedy maximum minimum Euclidean separation with the same quotas.
* `signature`: greedy maximum minimum distance in a feature space of distance
  threshold indicators to source landmarks, plus a small spatial term. Four nearby
  thresholds are used; this does NOT certify graph nonredundancy. New-new edges and
  the actual normalized threshold are evaluated after sampling/warmup.

Default: `spatial`. Both `spatial` and `signature` merit testing in R4. In the pilot,
`signature` was worse than plain spatial sampling in R3.

`--pool-factor 20` controls candidate oversampling; `--jitter .002` and
`--shape-noise .004` control exploratory perturbations. `--min-separation .003`
applies during insertion before final normalization; it is not a hard constraint
on later optimization. If the requested count cannot meet it, the program fails
explicitly rather than inserting duplicates. There is no claim that core size
will keep increasing indefinitely with the requested point count.

## What is being measured?

Let D be the whole-set diameter and a=d_k(X). For an exactly resolved finite set,
construct the inclusive threshold graph G with edges at distances >=a. It is not
k-colourable. Its k-colourable strict predecessor provides the upper bound.

### Diameter-anchored graph core

A diameter pair is retained during deletion. Then every retained subset has the
same denominator D and can preserve exactly the same ratio. A vertex v for which
G-v is k-colourable is mandatory in EVERY non-k-colourable induced subgraph at
this threshold. These tests give a genuine lower bound on the anchored minimum
core size. Randomized deletion gives an upper bound. The JSON reports both.

Three deletion orders are used by default. A completed order is inclusion-minimal
subject to the diameter anchors, NOT necessarily minimum-cardinality.

### Normalized non-degrading reductions

With `--normalized-audit`, the original `tune_solution.reduce_vertices` is also
run in several orders, now recomputing the subset diameter at every deletion.
An accepted deletion must prove new.lower >= old.upper with exact rational
comparisons. Reduction is repeated to a fixed point. It can INCREASE the ratio.
The smallest retained size found is a practical diagnostic, not a theorem of
minimum cardinality or equality with the original normalized ratio.

These diagnostics do not delete points from the exported seed. The full sampled
configuration remains available for subsequent topology changes.

### Candidate screening

After two brief optimization rounds, exactly resolved candidates within
`--ratio-window .015` of the largest screened ratio are eligible. Among them the
program prefers the largest minimum-found anchored core, then mandatory-vertex
count, then ratio. `--min-core` rejects candidates when a smaller anchored core
has already been found. Passing that threshold is NOT a proof that the minimum
core is at least that large; consult `anchored_minimum_core_lower_bound` for the
actual lower bound.

UNKNOWN never permits a deletion, never proves mandatory status, and never makes
a core 'minimal'. Unresolved evaluations or incomplete core checks are not
selected. A run with no eligible candidate returns exit status 3, leaving all
completed folders. Interrupted runs retain complete per-seed checkpoints.

## Warmup and export

Each warmup round prunes a non-k-colourable witness and runs the existing annealed
L-BFGS optimizer: default 40 iterations per each of five annealing stages, two
rounds, and at most 120 witness-pruning checks per round. No SLSQP final polish is
performed by this initializer. The point count does not change.

`initial.npy` is the fresh proposal. `points.csv` is the CANONICAL exact-decimal
warmup output; `points.npy` is its floating convenience copy. The standard
certificate, explicit partition, and witness graph are written beside it.
`selected/` contains the chosen seed, not a record improvement. Use a new output
directory; no prior user run is overwritten.

The included engine retains the native-loader absolute-path fix and early stop.
No engine implementation is modified by this release.

## Pilot results (not a long-run performance claim)

Eight independent seeds per method/dimension, all with 81 points, then two equal
warmup rounds. Ratios and all deletion checks resolved, with zero UNKNOWN answers.
Each retained-size statistic is the minimum found over three normalized
non-degrading reduction orders, averaged across the eight seeds.

| Dimension | Proposal | Mean raw core | Mean core after warmup | Mean ratio after warmup |
|---|---|---:|---:|---:|
| 3 | uniform sphere | 32.250 | 17.750 | 0.85034684 |
| 3 | random fitted components | 18.375 | 25.000 | 0.86262547 |
| 3 | spatial fitted components | 48.000 | 54.875 | 0.88265099 |
| 3 | signature fitted components | 24.875 | 31.000 | 0.86055474 |
| 4 | uniform sphere | 40.250 | 19.000 | 0.87185401 |
| 4 | random fitted components | 19.875 | 22.125 | 0.87815557 |
| 4 | spatial fitted components | 30.375 | 36.500 | 0.89550384 |
| 4 | signature fitted components | 30.125 | 37.250 | 0.89780763 |

Thus raw large cores alone can be misleading. For example the random R4 sphere
initially has a larger found core than the learned proposals, but a MUCH lower
ratio and a much smaller surviving core after optimization.

These are small-sample means, not established population expectations. They do
not compare all original initializers, prove minimum core cardinality, or establish
superior final records at equal wall-clock cost. Fitted proposals also have prior
information from successful optimized configurations; this is deliberate.

Raw per-seed results and exact-geometry certificates are included. Run a new pilot:

```bash
python benchmark_core.py --replicates 8 --out pilot
# Or restrict a batch:
python benchmark_core.py --replicates 8 --dim 4 --method signature --out pilot4
```

The benchmark uses 0.15-second SAT query limits (configurable). Normal generator
runs default to 5 seconds. The zero-UNKNOWN statement concerns the recorded pilot,
not a promise for different machines or seeds. Native DSATUR supplied the pilot
answers; the two selected example certificates and all 61 mandatory-vertex
checks for the source4 set were cross-checked with Python. No DRAT/LRAT checker
is included.

## Tests

```bash
python test_core_seed.py
```

18 focused tests cover proposal counts, exception geometry, spacing, reproduction,
strictly tied thresholds, core bounds against exhaustive small-subset enumeration,
inclusion minimality, UNKNOWN handling, normalized reductions, export verification,
and the selection gate. The saved tests.log records this run.

## Promising next algorithm, not implemented here

Starting from an edge- and vertex-critical non-k-colourable scaffold, split a
vertex into two nonadjacent copies, give each a proper subset of the old neighbours,
and retain a split only if the result is still non-k-colourable. The result is
vertex-critical: old-vertex deletions extend the old colouring by giving the two
copies the old vertex's colour; either new-copy deletion is a subgraph of an
edge-deleted old critical graph. This can grow the scaffold one vertex at a time.

This does not guarantee that its eventual geometric threshold SUPERGRAPH has
no smaller obstruction. Re-evaluate the full geometric core after realization.
The separate split_tune/paired_splits programs are the natural basis for that
extension; they are not integrated into this generator.
