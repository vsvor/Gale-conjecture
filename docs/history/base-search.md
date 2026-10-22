# Searching for badly partitionable finite point sets

This package searches for large values of

    R_k(X) = min over partitions into k parts of
             (largest part diameter)/(diameter of X).

The main intended cases are dimensions 3, 4 and 5, with k = dimension + 1.
The program is a heuristic outer optimizer with a separate exact discrete
evaluator. It does not prove a global optimum and does not establish a universal
upper bound for all point sets.

## Included results

All four saved configurations below were checked using the independent
pure-Python coloring backend after their construction. Their final diameter
bounds were checked using integer arithmetic on the saved decimal coordinates.
The native C++ backend was used during the searches.

| Directory | Dimension | Points | Parts | Lower bound | Upper bound |
|---|---:|---:|---:|---:|---:|
| `examples/d3_m49` | 3 | 49 | 4 | 0.875161234080 | 0.875161234081 |
| `examples/d3_m23` | 3 | 23 | 4 | 0.875161234080 | 0.875161234081 |
| `examples/d4_m32` | 4 | 32 | 5 | 0.887138937436 | 0.887138937437 |
| `examples/d5_m32` | 5 | 32 | 6 | 0.892304779206 | 0.892304779207 |

The 23-point configuration is an extracted subset of the 49-point configuration.
It already has the same ratio. For both 3D examples, the integer squared
lower and upper bounds actually agree exactly. Their common ratio squared is

    7659071856378174621085296199873074
    -----------------------------------
    10000000000000000738594676645254850

For a fixed set, these matching bounds determine its optimum; this is NOT a
claim that the set is globally extremal. The 4D and 5D results came from small
smoke tests, not an extensive search or a literature comparison.

Adding distinct points inside a set's convex hull does not change its diameter
and cannot reduce the optimum largest-part diameter. Thus the 23-point example
also gives the same lower bound for every larger point count, including 30--50.
The supplied 49-point example additionally has a matching upper-bound partition.

## Version 1.1: optional early stopping

Add `--early-stop` to abandon unpromising restarts after two to five rounds,
using a live record shared across all workers. The default schedule is
`--early-stop 2:0.02,5:0.005`; gaps are absolute differences in the ratio.
The first and every tenth restart are protected by default. Culled restarts
keep their best snapshot and skip final SLSQP. Without the option, the previous
search behavior is retained. The native absolute-path loader fix is included.

See **[EARLY_STOP.md](EARLY_STOP.md)** for the exact stopping rule, timeout handling,
examples, protection controls, and validation.

## Installation and first run

Requires Python 3.10 or later. PyTorch is not needed.

```bash
python -m pip install -r requirements.txt

# Use your existing Kissat executable. --k defaults to --dim + 1.
python borsuk_search.py search \
  --dim 3 --points 30:50 \
  --restarts 32 --workers 4 \
  --backend external --solver /absolute/path/to/kissat_bin \
  --out results3
```

`--points 40` fixes the point count. `--points 30:50` chooses an integer uniformly
in this inclusive range for each restart. `--k` is the number of PARTS directly;
it is not the target chromatic number. In particular, use `--k 4` in dimension 3,
not `--k 5`.

Two higher-dimensional examples of commands are:

```bash
python borsuk_search.py search --dim 4 --k 5 --points 40:60 \
  --restarts 32 --workers 4 --backend external --solver ./kissat_bin --out results4

python borsuk_search.py search --dim 5 --k 6 --points 40:70 \
  --restarts 32 --workers 4 --backend external --solver ./kissat_bin --out results5
```

The same command-line interface supports another solver that reads DIMACS.
For MiniSat, specify its result file so that the upper-bound partition is saved:

```bash
python borsuk_search.py search --dim 3 --points 40 \
  --backend external --solver 'minisat {cnf} {model}' --out minisat_run
```

Commands are split with `shlex` and run without a shell. If `{cnf}` is absent,
the CNF filename is appended. `{model}` is replaced with a temporary model-file
path. Temporary files are removed, including after timeouts.

### Optional faster in-process SAT backend

```bash
python -m pip install python-sat
python borsuk_search.py search --dim 3 --points 30:50 \
  --backend pysat --pysat-solver g4 --workers 4 --out pysat_run
```

The package name is `python-sat`, not `pysat`. This backend uses an activation
literal for each possible edge, assumption-based solving, reuse of learnt
clauses, and UNSAT-core extraction. The available choices here are Glucose 3/4
and MiniSat 2.2/GH, whose limited-solving interface supports interruption.
The subprocess Kissat backend does not pretend that Kissat supports incremental
assumptions: it launches a fresh process for each uncached graph.

### Optional dependency-free native coloring backend

The included C++ file implements exact DSATUR graph coloring. It is particularly
convenient when no SAT executable is installed. Compile it on a Unix-like system:

```bash
c++ -O3 -std=c++17 -shared -fPIC dsatur_native.cpp -o dsatur_native.so

python borsuk_search.py search --dim 3 --points 30:50 \
  --backend native --workers 4 --out native_run
```

The native backend supports k <= 63. Like every exact coloring algorithm, it
can become slow on hard instances. A slower pure-Python implementation is always
available with `--backend python`; it is useful for independent checks and small
examples, rather than large searches.

`--backend auto`, the default, tries the specified external command, then PySAT,
then the native shared library if compiled, then Python. Each saved run records
the backend actually selected. BLAS is limited to one thread per process to avoid
oversubscription; set `BORSUK_BLAS_THREADS` before launching to override this.

## Verify the supplied examples

These commands work without Kissat or PySAT:

```bash
python borsuk_search.py verify examples/d3_m23 --backend python
python borsuk_search.py verify examples/d3_m49 --backend python
python borsuk_search.py verify examples/d4_m32 --backend python
python borsuk_search.py verify examples/d5_m32 --backend python
```

To cross-check with your own SAT solver:

```bash
python borsuk_search.py verify examples/d3_m23 \
  --backend external --solver ./kissat_bin --sat-timeout 120

./kissat_bin examples/d3_m23/witness.cnf
```

The second command should report UNSATISFIABLE. The `verify` command reconstructs
the graph CNF from `witness_edges.csv`, so it does not blindly trust a preexisting
CNF file or the metadata's earlier solver status.

The result is a lower bound backed by a non-k-colorable graph and an upper bound
backed by an explicit partition. The graph's UNSAT result has not been checked
with an independent DRAT/DRUP proof checker. The geometric calculation, however,
is exact integer arithmetic, not a floating-point tolerance test. This distinction
is also recorded in `certificate.json`.

## Continue from a promising configuration

```bash
python borsuk_search.py search --dim 3 --k 4 \
  --start examples/d3_m49 \
  --restarts 24 --workers 4 --rounds 16 --steps 220 \
  --prune-checks 120 --polish 150 \
  --backend external --solver ./kissat_bin --out refined3
```

A starting file can be `.npy`, `.npz` (key `points` or `x`), `.csv`, whitespace
separated `.txt`, or a JSON point array (or an object containing `points`/`x`).
A saved best directory is also accepted. Its point count overrides `--points`;
its dimension must match `--dim`. Restart zero uses the supplied positions;
subsequent restarts perturb them.

The program does not parse the old free-form `log.txt` files automatically.
An existing NumPy array can be exported with `np.save('start.npy', x)`.

Use a new output directory when continuing a search. An existing experiment is
never silently overwritten. Each worker saves its own improving configurations
as it proceeds. If interrupted, inspect `run_00000/best`, `run_00001/best`, etc.;
the top-level `best` directory is updated only when an entire restart completes.

To evaluate existing points without optimizing them:

```bash
python borsuk_search.py evaluate start.npy --k 4 \
  --backend external --solver ./kissat_bin --out start_evaluation
```

## Initialization: simplex edges, prisms, and the sphere

The initializer `simplex-edges` implements radial projections of simplex edges.
Let v_0,...,v_d be regular-simplex vertices with unit norm and pairwise inner
product -1/d. On each edge, generate approximately evenly allocated, stratified
samples of

    ((1-t)*v_i + t*v_j) / norm((1-t)*v_i + t*v_j),    0 < t < 1.

The simplex vertices are included as well, and a small perturbation is applied
before projecting back to the sphere. Sampling is approximately balanced across
edges rather than leaving some edges empty by chance.

```bash
python borsuk_search.py search --dim 3 --points 40 \
  --init simplex-edges --seed-jitter 0.015 --restarts 24 --out simplex_trials
```

Other choices are `prism-edges`, `prism-sphere`, `simplex-chords`,
`simplex-faces`, and `sphere`. In dimension d, the prism is a regular
(d-1)-simplex times an interval, with a randomized aspect ratio. Thus in 3D it
is a triangular prism. `sphere` normalizes Gaussian vectors, giving genuinely
uniform directions; normalizing points uniform in a cube does not do that.

The default `--init mix` cycles through all six families. The supplied 3D winner
originated from a uniform-sphere run and was then refined from saved coordinates.
The present experiments therefore do not establish that the simplex-edge family
is better than uniform spherical sampling. Keeping both is deliberate.

**Spherical initialization is not a spherical constraint.** By default points
are free to move through R^d, as in the original optimization problem. Add
`--sphere` to keep them on the unit sphere during the continuous search:

```bash
python borsuk_search.py search --dim 3 --points 40 \
  --init simplex-edges --sphere --polish 0 --out spherical_trials
```

The free-coordinate SLSQP polish is skipped in this mode. Exported decimal
coordinates are rounded, so the exact certificates still concern finite sets
in R^d; they do not assert exact unit-sphere membership of the decimal rationals.

## Why the evaluator is correct

Normalize the diameter to D. Put an edge ij in G_t precisely when

    distance(x_i,x_j) > t*D.

A partition into at most k parts of diameter at most t*D exists exactly when G_t
is k-colorable. For m >= k, allowing at most k rather than exactly k nonempty
parts does not change the answer, because a part can be split. For k > m the
program permits empty parts and returns zero.

The implementation sorts all pair distances in decreasing order:

    d[0] >= d[1] >= ... >= d[N-1].

If q is the first prefix LENGTH for which the first q edges are not k-colorable,
then the exact ratio for the current point set is

    d[q-1] / d[0].

The q-edge graph forces a monochromatic edge of length at least d[q-1] in every
partition. A coloring of the preceding q-1 edges gives the matching upper bound.
This remains valid when distances tie. No numerical epsilon is used to turn a
tie into a fake strict inequality.

The empty prefix is known to be colorable. When m > k, the complete graph is
known not to be k-colorable. Binary search preserves these two proven endpoint
invariants. SAT models and PySAT cores can tighten the bracket further.

### Timeouts

`--sat-timeout` is a limit per oracle call, not a limit per restart. Zero means
unlimited. The three statuses SAT, UNSAT and UNKNOWN are handled explicitly.

An UNKNOWN answer stops that threshold search with a lower/upper bracket. It
does not replace either endpoint by an unproved claim. During graph pruning,
UNKNOWN means the tested edge is retained. Output uses `R in [lower,upper]` for
an unresolved bracket and `R = [...]` for a resolved threshold. Saved exact
decimal bounds may span one unit in the last printed place even for a resolved
threshold, because square roots are rounded outward.

A solver that deliberately suppresses its model can still provide a useful
lower bound, but the exported upper bound is then conservatively 1 unless a
partition is available. The default Kissat output includes a model.

## What the geometric optimizer does

For any graph H known not to be k-colorable, define

    rho_H(X) = min(edge lengths in H) / max(all pair distances).

Every k-part partition makes some edge of H monochromatic. Therefore rho_H is
always a valid lower bound on R_k(X). The optimizer improves this lower bound,
then reevaluates the actual R_k(X) rather than assuming H is still the critical
graph at the new positions.

A round starts from the critical threshold graph, optionally adds edges slightly
below its threshold, peels vertices of degree less than k, and tries deleting
edges. An edge is deleted only when the remaining graph is definitely UNSAT.
Deletion priorities favor short edges but contain randomness; pruning is bounded,
so the result need not be edge-minimal. The graph enlargement and randomized
pruning allow different obstructions to be explored.

The continuous stage minimizes the smooth, scale-invariant objective

    L_p = [ log(sum over all pairs of (distance^2)^p)
            + log(sum over H of (distance^2)^(-p)) ] / p.

As p increases, this tends to log(D^2/min_H distance^2). It is evaluated using
log-sum-exp with analytic gradients, followed by L-BFGS at p = 8,24,80,256,1024.
The best actual witness ratio encountered is retained; the smooth objective's
value is never reported as the partition ratio.

Vertices unused by the current witness are relocated to convex combinations
of active vertices before refreshing the graph. This cannot increase the
diameter or shorten a witness edge. After two nonimproving rounds, the code
perturbs the best positions and reevaluates the new configuration.

At the end of a free-coordinate restart, SLSQP optionally maximizes t subject to

    distance(x_i,x_j)^2 <= 1          for every pair,
    distance(x_i,x_j)^2 >= t          for each edge of H.

Even if SLSQP reports a failure or has residual constraint error, the candidate
is accepted only on its recomputed geometric ratio and fresh discrete evaluation.
No numerical feasibility tolerance is used as the lower-bound certificate.

### Important controls

`--rounds` controls how many times the critical graph is refreshed. `--steps`
is the maximum L-BFGS iterations PER annealing stage (five stages per round).
`--polish` is the final SLSQP iteration limit; zero disables it.

`--prune-checks 120` is the default deletion budget per round. Zero still peels
the k-core but skips SAT-guided edge deletion; a large value such as 1000 gives
more extensive pruning. These settings explore different topologies and need
not have monotonically improving performance.

`--graph-slack 0.025` controls how far below the current threshold the graph may
be enlarged. `--mutation 0.035` controls perturbation size. Larger values explore
more aggressively but can destroy a promising local configuration.

`--seed` controls per-restart random streams independently of worker scheduling.
Numerical trajectories can still depend on library versions, solver choices,
and floating-point behavior. The saved coordinates and certificate provide the
reproducible result, not a guarantee of bitwise-identical optimization runs.

## Output files

Each `best` directory contains `points.csv`, `points.npy`, `witness_edges.csv`,
`witness.cnf`, `partition.json`, and `certificate.json`.

The CSV is the canonical set of exact decimal-rational coordinates. Edges and
colors use zero-based vertex/color indices. `partition.json` lists one color
per point, or null if the solver supplied no model. `certificate.json` records
parameters, seeds, the backend, floating-point evaluations, and integer-derived
bounds for the saved coordinates.

The squared distances are computed as integers after multiplying all coordinates
by 10^17. The common coordinate scale cancels in the ratio. The lower square-root
bound is rounded down, and the upper bound is rounded up, to 12 decimal places.
Identical rounded coordinate rows, if any arise, are separated by increments of
10^-17 before this final check so that the saved object really is an m-point set.

Top-level `config.json`, `completed.jsonl`, and `summary.json` describe the run;
per-restart `improvements.jsonl` records progress. The original uploaded scripts
and `helper.py` are not imported or modified.

## Checks against the previous scripts

The earlier prefix search returned `l[e1]`, although `el[:e1]` has shortest edge
`l[e1-1]`. This indexing is corrected. More seriously, `gen7.py` used a timeout
to move the UNSAT endpoint. That can select a colorable graph as a supposed
witness; it is not a safe approximation.

The supplied `helper.py` actually invokes MiniSat in its active command; its
Kissat command is commented out. Its parser also does not recognize the standard
Kissat `s SATISFIABLE` / `s UNSATISFIABLE` lines. The new adapter accepts those
lines, validates exit codes, and checks any returned coloring explicitly.

The previous fine-tuning code evaluates distances before its last optimizer
step but returns the coordinates after that step. Also, optimizing one fixed
non-colorable graph yields a lower bound, not automatically the exact partition
ratio at the final coordinates. The new program recomputes both the geometry
and the discrete objective before reporting a candidate.

## Tests and limits of validation

```bash
python -m unittest -v test_borsuk_search test_early_stop
```

The 17 original test methods plus 26 early-stop tests passed (43 total) for
version 1.1. The original tests include all
graphs on four vertices, 200 additional random graphs checked by independent
brute-force coloring, finite-set objectives compared with enumeration of all
partitions on up to eight points, tied distances, duplicate clusters, simulated
external SAT/UNSAT/timeout responses, analytic-gradient checks, optimizer
nondecrease, and integer-certificate round trips.

The pure-Python and native backends were exercised. The external adapter was
exercised using simulated solver processes and output, but an actual Kissat
binary was not available in this environment. PySAT could not be installed here
and its optional backend was not executed. The tests automatically include PySAT
when it is installed. Cross-check a supplied example with your own Kissat binary
before a long campaign.

This is a research search tool. A good value is evidence about one supplied
finite configuration, not a proof of a best possible ratio, not a proof or
disproof of Borsuk's conjecture, and not a claim to improve a published record.
