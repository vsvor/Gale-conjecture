# Essential vertex splitting for the Borsuk min-max ratio

This package is an extension of `borsuk_search.py` / `tune_solution.py`, not a
replacement for their graph or certificate engines. It was developed and tested
on the supplied 95-point configuration in R^3.

## Run

Python 3.10+, NumPy and SciPy are required. No PyTorch is used.

```bash
python -m pip install numpy scipy
python split_tune.py /path/to/points.npy --k 4 --max-omit 2 --rounds 20 \
    --backend external --solver /path/to/kissat_bin --sat-timeout 60 \
    --out split_results
```

A coordinate file, a saved solution folder, or a search root containing `best/`
can be passed as the input. The original is never overwritten. Use a new output
directory for every run. The result is `split_results/best/`.

Use the more extensive split family in a continuation run:

```bash
python split_tune.py split_results/best --max-omit 3 --rounds 20 \
    --trials 40 --per-vertex 6 \
    --backend external --solver /path/to/kissat_bin --sat-timeout 60 \
    --out split_results_3
```

The supplied best has no positive-LP candidates in the last tested unpruned
family. Repeating that identical screen is therefore unlikely to help. To
explore different witnesses and finite split perturbations instead (not a
guaranteed further improvement):

```bash
python split_tune.py best --max-omit 3 --rounds 20 \
    --trials 40 --per-vertex 6 --prune-checks 800 \
    --zero-gain-trials 12 --zero-gain-jitter 0.01 --seed 20260909 \
    --backend external --solver /path/to/kissat_bin --sat-timeout 60 \
    --out continued
```

The search is deterministic given the same numerical libraries, solver, flags,
and seed, but small floating-point changes can change the subsequent branch.
Different settings can therefore reach different configurations. A larger family
is not guaranteed to produce a better final result: greedy path dependence remains.

## Without a SAT executable

The pure-Python graph-coloring backend works without compilation:

```bash
python split_tune.py examples/d3_m46 --max-omit 1 --rounds 1 \
    --backend python --sat-timeout 30 --out one_split
```

The optional native DSATUR backend is faster on many of these examples:

```bash
c++ -O3 -std=c++17 -shared -fPIC dsatur_native.cpp -o dsatur_native.so
python split_tune.py /path/to/points.npy --backend native \
    --max-omit 2 --rounds 20 --out native_run
```

A PySAT backend is also inherited from the original engine (`pip install
python-sat`, `--backend pysat`). The native and pure-Python backends were exercised
in this investigation. Actual external Kissat and PySAT execution was not tested
here; those adapters are unchanged from the earlier engine.

## What the move does

Let H be a verified non-k-colorable graph whose edges are long in the current
configuration. Choose v and disjoint nonempty subsets A, B of its neighbors.
Replace v by nonadjacent copies v0, v1 with neighborhoods

    N(v0) = N_H(v) \ A,
    N(v1) = N_H(v) \ B.

All other old edges remain. Each of A and B must contain a neighbor u for which
H - vu was explicitly verified k-colorable. The split graph itself must be
explicitly verified NON-k-colorable. Consequently both copies are essential:
deleting either leaves a subgraph of a known colorable edge-deleted H.

An intermediate optimization seed places v0 and v1 at the same coordinates.
This is NOT published as a new finite point set. Their GRAPH constraints are
different, so they can move in different directions. Only distinct final points
with a rigorously improved saved-coordinate lower bound are accepted.

The use of temporary coincidence is deliberate. Rejecting every insertion that
starts as a geometric twin prevents this move even though the optimized copies
can be non-twins and essential. In the supplied first successful split, their
final separation is about 0.04585 times the diameter; both are essential and
non-twins in the final threshold graph.

A numerical first-order linear program screens split graphs for an improving
motion. The solver then maximizes t under:

    all pair squared distances <= 1,
    all split-witness edge squared distances >= t.

SLSQP uses analytic Jacobians and checks all omitted diameter constraints during
constraint generation. The objective ultimately reported is recomputed from
actual coordinates, not read from the optimizer's `fun` or epigraph variable.

## Parameters worth varying

`--max-omit 1` is the smallest family: omit one neighbor from each copy. Values
2 and 3 allow larger disjoint omission sets. Neighbors are selected among the
numerically near-shortest edges (`--active-tol`, squared-distance units after
normalization). Thus these flags do not enumerate every possible graph splitting.

`--trials` caps the positive-LP candidates optimized per round; 0 means all.
`--per-vertex` prevents many variants of one vertex from consuming the whole
budget; 0 removes this cap. `--screen-limit 0` (default) tests every generated
split graph; a positive value limits the SAT screen.

`--zero-gain-trials 8` additionally tries some candidates with a numerically zero
first-order gain, starting with opposite random perturbations of the two
copies (`--zero-gain-jitter`, default 0.002 in diameter units). These are not
proved unimprovable; second-order effects or a finite displacement may still help. `--prune-checks 800` first extracts a sparser
UNSAT witness, allowing exploration of a different constraint graph. Both are
exploratory options, not guaranteed improvements.

`--no-reduce` keeps the original point count during reduction stages. By default,
vertices are removed only when exact integer-distance comparisons and an UNSAT
query prove that the ratio cannot decrease.

## Stopping and polishing

The reported best never uses an UNKNOWN answer as an UNSAT answer. A proposed new
configuration must have its exact rational lower squared bound strictly above
the incumbent's exact rational upper squared bound, as well as exceed the
numerical `--min-gain` screen (default 1e-10 in ratio units).

After searching, the actual saved best receives a separate repeated SLSQP polish
with ALL diameter constraints, small perturbed starts, witness reevaluation and
a numerical first-order audit. See `best/polish.json`.

`numerically_stationary` is NOT a mathematical certificate of local or global
optimality. It concerns the current witness and the configured numerical checks.
Likewise, `no_selected_split_in_tested_family` is not a claim that no larger,
different or finite-amplitude split can improve the result.

A polishing-only check, without vertex removal, is:

```bash
python split_tune.py best --rounds 0 --no-reduce --polish-passes 16 \
    --slsqp-iters 1000 --backend native --out repolished
```

## Files and verification

The output is compatible with the previous viewer and verifier:

```bash
python view_solution.py best --hull
python borsuk_search.py verify best --backend python --sat-timeout 120
# Or cross-check with your own Kissat:
python borsuk_search.py verify best --backend external \
    --solver /path/to/kissat_bin --sat-timeout 120
```

`points.csv` is the canonical 17-decimal coordinate file. Squared distances are
computed as integers after multiplying coordinates by 10^17. `witness_edges.csv`
gives the lower bound; `partition.json` gives the upper bound. Equal integer
squared bounds determine the ratio of that saved configuration exactly, up to
taking a square root. The graph UNSAT result is not independently DRAT/LRAT checked.

`progress.jsonl` records every screened stage and optimized candidate.
`rounds/NNN/screen.json` records the selected split and its LP audit. Accepted
split details include whether the two resulting points are twins in the saved
threshold graph and whether deleting either makes that graph colorable.
`history/` contains immutable improving snapshots. `--save-trials` also stores
all evaluated trial configurations. In an interrupted run, completed snapshots
remain available, although final polishing may not yet have run.

The attached results and numerical validation are documented in `VALIDATION.md`.

Run the regression suite with:

```bash
python -m unittest -v test_split_tune test_tune_solution test_borsuk_search
```

Original 2026 examples are retained under `examples/d3_m23`, `examples/d3_m49`,
etc., because inherited regression tests use those exact paths. The new
49-point example is `examples/improved_d3_m49`, not `examples/d3_m49`.
