# Fixed-hull surface-point search

`surface_search.py` adds points on verified faces of the **original, fixed convex
hull**, looking for a finite set with a larger minimax partition ratio

    R_k(S) = min_partition max_i diam(S_i) / diam(S).

All original points are retained at exactly their original coordinates. The hull
and its diameter do not change. This is not the free-coordinate optimizer from
`borsuk_search.py`, and a finite-sample coloring is NOT an upper bound for the hull.

## Install and run

Requires Python 3.10+, NumPy, SciPy, and the included `borsuk_search.py`. No changes
to that engine are necessary. Its native-library absolute-path fix and optional
early-stopping implementation are retained.

```bash
python -m pip install -r requirements.txt

# Optional native coloring backend (Linux/Unix):
c++ -O3 -std=c++17 -shared -fPIC dsatur_native.cpp -o dsatur_native.so

python surface_search.py search examples/record58 \
    --k 4 --rounds 50 --batch 6 --max-points 400 \
    --pool 3000 --refine 24 --gain 0.000001 \
    --backend native --sat-timeout 30 \
    --out surface_run
```

Replace `examples/record58` with your saved best directory, search root containing
`best/`, or a `.csv`, whitespace-separated text, `.npy`, or `.npz` coordinate file.
An NPZ needs a `points` or `x` array. CSV files must have no header (comment lines
beginning with `#` are permitted). An existing directory's `points.csv` takes
precedence over `points.npy`. Input points must be distinct and full-dimensional.

Dimensions 2, 3, 4 and 5 are supported. The main tested application is R^3, k=4.
`--k` means the number of parts and defaults to dimension+1. Basic 4D and 5D
search/verification tests are included, not large optimization benchmarks.

To use Kissat, replace the backend arguments with:

```bash
--backend external --solver /absolute/path/to/kissat_bin --sat-timeout 60
```

The external adapter is inherited from the existing engine. Actual Kissat was not
available in the validation environment. For MiniSat-style result files, the
engine accepts `--solver 'minisat {cnf} {model}'`.

An optional incremental backend is available:

```bash
python -m pip install python-sat
python surface_search.py search examples/record58 \
    --backend pysat --pysat-solver g4 --max-points 400 --out surface_pysat
```

The package is `python-sat`, not `pysat`. PySAT was not installed/tested in this
session. Its edge-activation implementation is inherited from the engine.
One capacity-sized oracle is reused throughout a run. Reserve vertices are
isolated until populated. This permits reuse of learnt clauses with PySAT;
external Kissat still starts a new process for each uncached query.
Memory scales quadratically with `--max-points`, and the activation CNF additionally
scales with k. Avoid setting an enormous capacity unnecessarily.

The pure-Python backend requires no SAT executable or C++ compilation:

```bash
python surface_search.py search examples/record58 --backend python --out slow_run
```

## Continue from the WORKING set, not necessarily the best set

```bash
python surface_search.py search surface_run/latest \
    --rounds 70 --batch 8 --max-points 900 \
    --pool 5000 --refine 32 --block-pool 160 \
    --colorings 6 --gain 0 \
    --seed 123456 --backend native --sat-timeout 60 \
    --out surface_continued
```

**Use `latest/` to retain additions that have not yet increased the ratio.**
`best/` remains the earlier, smaller set until its certified lower bound increases.
Several individually neutral points can obstruct a coloring jointly. Throwing
away all neutral additions can prevent the method from reaching that obstruction.

The source hull, barycentric coordinates, and original comparison bound are read
from `surface.json` when continuing. Existing points and their face assignments
are retained. This is a new continuation run, not an exact replay of the old RNG
state; choose a new `--seed` for different sampling. The resumed run must use the
same k. A fresh output directory is required; earlier experiments are not silently
overwritten. To change k, put a copy of the desired coordinates in a fresh input
location without the old surface sidecar.

The example `examples/record58_370` is an already populated working set containing
312 new surface points, with the same exact ratio as the original 58 points:

```bash
python surface_search.py search examples/record58_370 \
    --rounds 60 --batch 8 --max-points 900 \
    --gain 0 --seed 7654321 --backend native --out continued_example
```

These continuation commands are exploratory settings, not demonstrated
improvements beyond 0.8872505036444075.

## Search logic

### 1. Choose a target just above a known finite-set upper bound

For target ratio t and original diameter D, create the **point graph**

    pq is an edge iff |p-q|^2 > t^2 D^2.

Every comparison uses Python integers and cross multiplication. No floating-point
threshold tolerance changes the graph. A k-coloring yields a partition of the
finite working set. An UNSAT answer yields a non-k-colorable witness; its shortest
edge provides a lower bound exceeding t for both the finite set and the hull.
A solver timeout never counts as UNSAT.

The default `--gain 0.000001` seeks an absolute ratio increase of at least 1e-6
above the current upper endpoint. This is not a relative percentage. The target
is advanced after it is crossed. `--gain 0` uses the exact squared upper endpoint
and seeks *any* strict increase. Very tiny gains need not be scientifically useful.
A fixed target can be used instead:

```bash
python surface_search.py search examples/record58 \
    --target 0.888 --max-points 500 --backend native --out target_0888
```

That run stops once it has a proof that its finite-set ratio exceeds 0.888, or
when a budget or unresolved solver call stops the search. Reaching that value is
not guaranteed.

### 2. Find points that obstruct current colorings

For an existing coloring S_1,...,S_k and a potential new point p, define

    F(p) = min_i max_{s in S_i} |p-s|.

If F(p)>tD, no old color can be assigned to p without changing the old coloring.
Candidates are sampled on hull faces and edges, with both global coverage and a
modest bias toward recently selected faces. Face centers and edge midpoints are
also considered. Selected candidates are refined by SLSQP on their original face.
This local nonsmooth optimization proposes candidates only; it is not a proof.

Several valid colorings are obtained through single-vertex and Kempe moves. They
are a small heuristic bank, not an enumeration of all colorings. Candidate ranking
seeks to obstruct several bank members at once.

For cooperative insertion, each candidate receives the list of colors permitted
by one fixed old coloring. A small list-coloring CSP also includes new-new
long-distance edges. An UNSAT block can show that several points jointly block
that particular old coloring even when each is individually admissible. Bounded
block reduction tries to fit it in the requested batch size. UNKNOWN checks do
not justify deleting a point from a blocking core.

**A blocker for one coloring is not a lower certificate.** After each batch,
all original and added points are freely recolored by the global graph oracle.
Only global noncolorability supports a new lower bound.

The search retains informative candidates even when no individual/block screen
succeeds or no immediate gain results. There is no hard graph-twin rejection.
Only exact coordinate duplicates are discarded. Different future neighbors can
make presently similar points useful.

### 3. Evaluate and optionally polish

At every `--evaluate-every` rounds (default 4), and after a target crossing, exact
integer-distance bisection tightens the finite-set lower/upper bracket. Incomplete
solving leaves a bracket instead of manufacturing an exact answer.

After a lower-bound improvement, `--polish-iters` SLSQP iterations can optimize
only the added points, each confined to its existing supporting simplex. For a
noncolorable witness H, this maximizes its shortest edge. Since every point stays
in the fixed hull and all original points remain, there is no need to impose the
quadratic number of global diameter constraints from the free-coordinate tuner.
Analytic witness Jacobians are used.

The continuous proposal is rounded back to the exact barycentric grid. It is
accepted only if its exact witness lower bound exceeds the preceding finite-set
upper bound. Coincident points or insufficient improvement cause rejection.
`--polish-iters 0` disables this step. There is no claim of global or second-order
optimality, or of stationarity across all possible witnesses/faces.

## Exact surface membership

A floating-point convex-hull triangulation is only a proposal. Every retained
simplex is checked with exact integer supporting-plane signs. For the supplied
58-point input, Qhull proposes some triangles that do not support the exact
17-decimal point set because of near-coplanarity. The default `--facets auto`
repairs this in 3D by enumerating supporting triples, grouping coplanar faces,
and triangulating their polygons with exact orientation tests.

`--facets exhaustive` always performs that enumeration in 3D. The default
`--facet-enumeration-limit 200000` bounds the number of tested triples. In higher
dimensions, proposed simplicial faces are individually checked; invalid ones are
omitted. Thus higher-dimensional runs may explore only part of the boundary.
No claim that Qhull's list covers the exact boundary is needed for the lower bound.

On a face with original vertices v_1,...,v_d, a new point is stored as

    p = (b_1 v_1 + ... + b_d v_d) / Q,
    b_i >= 0 integers, sum b_i = Q, Q = 2**weight_bits.

The default is Q=2**24. Rounding acts on the weights, not the Cartesian
coordinates, so a new point does not drift outside the face. The input decimals
are interpreted exactly. NumPy floating inputs use their exact binary values,
not a silently substituted 17-decimal approximation. Computations use a common
integer coordinate scale.

`points.csv` is authoritative. It has as many decimal places as needed to encode
the exact rational convex combinations. `points.npy` is a float64 convenience
copy; do not use it to certify exact boundary membership. Saved CSV checkpoints
can be verified by the original engine, whose coordinate precision comes from
`certificate.json`.

## Output

- `best/`: the best achieved certified lower bound, with a finite-set upper bound.
- `latest/`: the working set, including neutral additions; use this to continue.
- `history/`: improving checkpoints, when there are improvements.
- `progress.jsonl`: target, SAT statuses, proposal diagnostics, and bounds per round.
- `summary.json`, `settings.json`, `hull.json`: final status, parameters, face checks.

Each point-set checkpoint contains `points.csv`, `points.npy`, `partition.json`,
`witness_edges.csv`, `witness.cnf`, and `certificate.json` in the old engine format.
`surface.json` and `source_points.csv` additionally certify fixed-hull membership.

Completed checkpoints survive interruption. A complete temporary snapshot is
renamed into place; `*.previous` is the previous complete snapshot during the
brief rename window. The last in-progress batch may not have been saved.

## Verify and visualize

```bash
# Recheck both exact surface membership and finite-set bounds:
python surface_search.py verify surface_run/latest \
    --backend python --sat-timeout 120

# A separate check with your solver:
python surface_search.py verify surface_run/latest \
    --backend external --solver /absolute/path/to/kissat_bin --sat-timeout 120

# Old engine finite-set check (does not check surface.json):
python borsuk_search.py verify surface_run/latest --backend python

# Viewer included for convenience; Matplotlib is optional:
python -m pip install matplotlib
python view_solution.py surface_run/latest --hull
```

Geometry is checked by integer arithmetic. Noncolorability uses a solver result,
not an independently checked DRAT/LRAT proof. A SAT timeout leaves verification
unresolved; increase the timeout or use another solver.

## Tests and obtained results

```bash
python -m unittest -v test_surface_search
```

All 34 regression methods passed in the recorded environment. They include exact
evaluation against exhaustive partitions, list-coloring against brute force,
joint blockers, strict thresholds/ties, unused oracle capacity, timeout semantics,
source-coordinate preservation, non-triangular faces, near-coplanar facets,
continuation, and rejection of tampered certificates. Native and pure-Python
backends were exercised. Actual Kissat and PySAT execution was not tested.

For a square with k=2, the program added four boundary midpoints and increased
R_2 from sqrt(1/2) to sqrt(5/8), with matching exact squared bounds. The resulting
8-point checkpoint is in `examples/square_improved`.

For the latest 58-point record hull, a 16-round run followed by a 36-round
continuation accumulated 370 points. The final exact squared lower and upper
bounds equal those of the original 58-point set. No larger ratio was found in
these runs. This does not prove equality with the convex hull's optimum, and no
new upper bound for the entire hull is claimed.

See `VALIDATION.md` and `validation/` for the commands, summaries, and checks.

## References for underlying library interfaces

SciPy ConvexHull documentation (Qhull simplices):
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html

PySAT assumption-based solver interface:
https://pysathq.github.io/docs/html/api/solvers.html

The surface-insertion heuristic, exact membership layer, and certificates in this
package are implemented here; those references do not establish their correctness.
