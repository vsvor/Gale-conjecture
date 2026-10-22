# Structural tuning of badly partitionable point sets

`tune_solution.py` starts from an existing good configuration. It does **not**
restart from an unrelated random set. It alternates certified vertex deletion,
colouring-aware insertion of small point blocks, changes of non-colourable
witness graph, smooth local optimisation, and repeated constrained SLSQP
polishing.

The saved best is protected by exact comparisons of integer squared distances.
Trial configurations are allowed to be worse: otherwise many useful graph
changes would be inaccessible. A worse trial never replaces the saved best.
There is no claim that either a fixed configuration or the reported ratio is
globally optimal.

## Quick start with the existing Kissat executable

Requires Python 3.10 or later. Extract the ZIP, enter its `borsuk_tuner`
directory, and run:

```bash
python -m pip install -r requirements.txt

python tune_solution.py /path/to/results3/best \
    --k 4 --cycles 60 --max-points 120 --add 1:3 \
    --backend external --solver /path/to/kissat_bin \
    --sat-timeout 60 --out tuned3
```

`--k` is the number of parts, not the chromatic number of the obstruction.
It defaults to the saved value of `k`, or to dimension + 1 when no saved
metadata is available. Dimensions 4 and 5 use exactly the same interface.

A search root containing `best/`, a saved result directory, or a coordinate
file is accepted. Examples:

```bash
python tune_solution.py results3 --out tuned3
python tune_solution.py results3/run_00023/best --out tuned_run23
python tune_solution.py points.npy --k 4 --out tuned_numpy
```

Keep `tune_solution.py` beside the accompanying `borsuk_search.py`. The latter
is the compatible engine from the earlier package, unchanged. Existing
modified copies need not be overwritten: the new ZIP can live separately.
The old `helper.py` is not needed. No PyTorch dependency is used.

The ZIP includes the earlier examples and a tuned 23-point example. To try the
whole pipeline without an external SAT solver:

```bash
python tune_solution.py examples/d3_m49 --backend python --cycles 2 --out demo
```

The pure-Python solver is intended for small examples and cross-checks. For
your roughly 95-point configuration, use Kissat or the incremental backend:

```bash
python -m pip install python-sat
python tune_solution.py results3/best --backend pysat --pysat-solver g4 \
    --cycles 60 --max-points 120 --sat-timeout 60 --out tuned_incremental
```

Optional native colouring backend, compiled locally on Linux/macOS:

```bash
c++ -O3 -std=c++17 -shared -fPIC dsatur_native.cpp -o dsatur_native.so
python tune_solution.py examples/d3_m23 --backend native --out tuned_native
```

No compiled binary is distributed. `--backend auto` tries the configured
external solver, PySAT, a locally compiled native library, then Python.

## Separate reduction/polishing from topology exploration

First reduce and polish the input, without insertion cycles:

```bash
python tune_solution.py results3/best --cycles 0 \
    --slsqp-iters 600 --polish-passes 12 --polish-kicks 2 \
    --backend external --solver ./kissat_bin --out reduced_polished
```

Then continue from that result:

```bash
python tune_solution.py reduced_polished/best --cycles 100 \
    --add 1:3 --max-points 120 --seed 8123 \
    --backend external --solver ./kissat_bin --out structural_tuning
```

For a final SLSQP-only check, with every diameter constraint present from the
first iteration:

```bash
python tune_solution.py structural_tuning/best --polish-only \
    --full-constraints --slsqp-iters 1000 \
    --polish-passes 16 --polish-kicks 3 \
    --backend external --solver ./kissat_bin --sat-timeout 120 \
    --out final_polish
```

`--polish-only` disables vertex deletion, insertion and L-BFGS. Small perturbed
SLSQP starts remain enabled; set `--polish-kicks 0` for unperturbed repeats only.

Use a **new** output directory for every invocation. To continue an experiment,
pass its `best` directory as the new input. Input files are never modified.
The program is single-process. Independent invocations with different seeds
and different output directories can run concurrently; never share `--out`.
BLAS defaults to one thread, adjustable through `BORSUK_BLAS_THREADS`.

## 1. What “ratio-preserving deletion” means

For a point set X, let L(X) and U(X) be the lower and upper bounds certified by
an UNSAT witness graph and an explicit partition, respectively. Whenever the
threshold search finishes, L(X) = U(X) exactly as squared rational numbers.

For a candidate deletion Y = X minus one point, recompute D(Y), then form

    E(H) = {ij : |y_i-y_j|^2 / D(Y)^2 >= U(X)^2}.

Delete the point only if H is definitely not k-colourable. This proves

    R_k(Y) >= U(X) >= R_k(X).

The comparison uses integer cross multiplication, not an epsilon test. The
partition upper bound is updated after every deletion, so a gain due to a
shrinking diameter is not silently lost on the next deletion. UNKNOWN retains
the vertex. A bounded number of failed deletion tests does not establish
vertex-minimality.

The input CSV is interpreted as exact decimal coordinates, including its
precision. Non-CSV inputs are normalised and exported to 17 decimal places;
subsequent optimisation results also use 17 decimal places. Every saved bound
concerns the exact decimals in **that saved CSV**, not hypothetical unrounded
coordinates.

## 2. Why the insertion test is meaningful

Testing the original non-colourable graph plus a new vertex is useless: it
remains non-colourable even if the new vertex is isolated.

Instead the tuner constructs a **colourable comparison skeleton B** on the
old vertices. Take an upper-bound partition, keep geometrically long edges
joining different parts, and omit same-part edges. The old partition explicitly
colours B. At the current ratio use a strict threshold; at relaxed thresholds,
this gives alternative colourable skeletons for exploration.

For a proposed block of 1–3 points (configurable up to 6), add all its long
new–old and new–new edges. Require:

* The resulting graph H is definitely not k-colourable.
* Deleting **each** inserted vertex from H gives a colourable graph.

Thus the inserted points genuinely create this obstruction. This does not
assert that each new point is essential in the complete geometric threshold
graph after subsequent optimisation; the comparison graph and its role are
recorded separately.

For one inserted point v, the UNSAT condition says that its neighbours use
all k colours in **every** k-colouring of B. Checking only one saved colouring
would not establish this. The saved-colouring check is used only as a cheap
rejection filter; SAT checks the full condition. For blocks, an exact small
list-colouring precheck also catches cooperative obstructions.

The prefilters reject new vertices with degree below k, geometric twins, and
nonadjacent dominated neighbourhoods. If v is nonadjacent to u and
N(v) is contained in N(u), v can always copy u's colour. False twins are a
special case. **Adjacent true twins are not generally colour-redundant**;
they are also excluded here as an explicit non-twin exploration preference,
not because of a false redundancy theorem.

### Where candidate points are placed

Most candidates sample the diameter-completion body

    intersection over x in X of B(x, D(X)),

particularly its boundary. This includes locations outside the convex hull.
Some candidates are refined to be far from at least one vertex of each
saved colour class. A block's new–new distances are checked as well.

The default `--outer-slack 0.04` also permits exploratory shell candidates
and up to 4% initial diameter expansion. Set `--outer-slack 0` to allow only
blocks that preserve the old diameter numerically before optimisation. Final
acceptance always uses exact saved-coordinate bounds, regardless of this flag.

Default relative threshold relaxations are

    0, 0.004, 0.012, 0.03, 0.08, 0.15.

The larger relaxations are deliberate: a very good configuration often admits
no novel essential insertion at its present threshold. A weaker initial
obstruction can sometimes be geometrically improved beyond the incumbent.
The program does not claim that every admitted trial will do so.

For a narrower exploration:

```bash
python tune_solution.py results3/best --outer-slack 0 \
    --slacks 0,0.004,0.012,0.03 --add 1:2 --out conservative_tuning
```

`--compact-trials` is enabled by default. After extracting a new UNSAT witness,
it removes vertices not used by that witness before the trial optimisation.
This preserves the **trial witness bound**, not necessarily the original full
set's ratio. It is distinct from the certified incumbent deletion in section 1.
Disable with `--no-compact-trials`. Exploratory reductions cannot degrade `best`.

## 3. Repeated SLSQP and first-order diagnostics

For a fixed non-k-colourable graph H, the continuous problem is

    maximise t
    subject to |x_i-x_j|^2 <= 1 for ALL pairs,
               |x_i-x_j|^2 >= t for edges of H.

The realised witness ratio is sqrt(t) at a feasible optimum. In the code the
actual ratio is always recomputed as min witness length / actual diameter;
SLSQP's objective and success flag are never used as geometric certificates.
The first vertex is fixed at the origin, removing translation. Jacobians are
analytic.

Initially, SLSQP uses all witness inequalities and a working set of diameter
inequalities. Every omitted diameter inequality is checked after each solve.
Violated ones are added and SLSQP is restarted. `--full-constraints` puts all
pair constraints into every solve immediately.

Each local trial then:

1. Repeats SLSQP from the best retained coordinates and tries small coordinate
   perturbations.
2. Requires the configured number of successful, feasible, no-gain passes.
3. Uses a separate linear programme to look for a first-order improving motion
   of the near-active diameter and shortest-witness constraints.
4. Re-evaluates the full partition objective, and polishes a refreshed witness
   when evaluation finds a stronger obstruction.
5. Audits the coordinates that will actually be saved.

The LP maximises eta with coordinate velocities bounded in absolute value by 1:

    2(x_i-x_j).(v_i-v_j) <= 0                 for near-diameter pairs,
    eta <= 2(x_i-x_j).(v_i-v_j)               for near-shortest witness edges.

A candidate LP direction is useful only if a line search improves the actual
nonlinear ratio. A numerical zero LP gain with near-active constraints is
**not** an exact stationarity certificate, a second-order test, or a proof
about other witness graphs.

`best/polish.json` and `summary.json` distinguish:

| Status | Meaning |
|---|---|
| `numerically_stationary` | Repeated no-gain successful SLSQP passes, required kicks, and the numerical LP audit passed at the configured tolerances. |
| `polish_budget_exhausted` | The pass budget ended before all stopping conditions were met. |
| `refresh_budget_exhausted` | The full objective/witness changed or the retained point still needed further checking. |
| `audit_unresolved` / `saved_coordinate_audit_unresolved` | The final LP audit did not support the numerical stopping claim. |
| `not_polished` | A saved checkpoint exists, but polishing was not yet completed, for example after interruption. |

No finite collection of SLSQP runs proves that another starting point, a
higher-order motion, or a different graph cannot improve the result. A solver
failure or a first-order residual is reported, not relabelled as convergence.

The SciPy interfaces used are documented at:

* https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
* https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html

## Main tuning controls

| Option | Effect |
|---|---|
| `--cycles 60` | Number of structural search cycles. |
| `--add 1:3` | One-, two-, and three-point insertion blocks. |
| `--trials 6` | Maximum selected insertion trials per cycle. |
| `--candidates 384` | Geometric candidate pool size. |
| `--screen-checks 150` | Budget of full-extension SAT tests, before essentiality checks. |
| `--delete-checks 150` | Maximum vertex tests in each reduction pass. |
| `--edge-prune 80` | Maximum UNSAT-preserving edge deletion tests per witness. |
| `--steps 100` | L-BFGS iterations at each of four smoothing levels. |
| `--slsqp-iters 300` | Iteration budget for each SLSQP solve. |
| `--polish-passes 8` | Repeated polishing calls per witness. |
| `--polish-kicks 1` | Perturbed SLSQP starts before numerical stopping. |
| `--ratio-tol 1e-10` | Numerical no-gain tolerance, never a deletion/acceptance epsilon. |
| `--explore-loss 0.002` | Relative deficit allowed for an internal exploratory branch; saved best is protected. |

After a run with exhausted polishing budgets, continue in `--polish-only`
mode with higher iteration/pass limits rather than treating it as certified
optimal. If SAT repeatedly times out, increase `--sat-timeout` or use an
incremental backend. A bracket is retained on UNKNOWN; false improvements
are never inferred from timeout.

## Saved results and verification

`best/` has the same format as the previous package and works with the earlier
viewer and verifier:

```bash
python borsuk_search.py verify tuned3/best \
    --backend external --solver ./kissat_bin --sat-timeout 120

python view_solution.py tuned3/best --hull
```

The viewer needs Matplotlib, and `--hull` needs SciPy. Matplotlib is not needed
for tuning or verification.

Files include `points.csv`, `points.npy`, `witness_edges.csv`, `witness.cnf`,
`partition.json`, `certificate.json`, and the new `polish.json`.
`progress.jsonl` records deletions, screening, numerical passes and accepted
bests. `history/` keeps immutable best snapshots; `trials/` keeps proposal
details and trial results. A previous best survives as `best.previous/` while
a new best is published. Ctrl+C leaves completed snapshots intact.

Each replacement by a different configuration uses the conservative exact test

    new lower bound >= old upper bound,

with a strict improvement, fewer points, or a stronger polishing report at
an unchanged certified ratio. In particular, improving only an unresolved
lower estimate is not enough to replace a potentially better incumbent. Updating
only the bounds or polishing report of identical coordinates does not change
the configuration and cannot degrade its ratio.

The geometric bounds are exact for the saved decimals. Non-colourability
comes from the SAT solver or exhaustive colouring search; no independent
DRAT/DRUP proof checker is integrated.

## Validation and supplied example

Run both suites:

```bash
python -m unittest -v test_tune_solution test_borsuk_search
```

There are 31 tuner tests and 17 engine tests. They cover integer/tied-distance
evaluation versus exhaustive partition enumeration, deletion preservation,
UNKNOWN handling, single and cooperative insertions, false/true twins,
nonadjacent domination, completion-body sampling, analytic Jacobians, SLSQP
budget flags, full versus generated constraints, end-to-end monotone
checkpoints, and compatibility with the original verifier.

On the supplied 49-point example, a reduction pass removed 26 points without
any change in the exact ratio. A native-backend tuning smoke run improved the
23-point result from approximately 0.875161234080794 to
0.875415792923508. The improved coordinates and matching lower/upper
certificates are in `examples/tuned_d3_m23`. They were also verified with the
independent pure-Python colouring backend.

This does not beat the user's reported 0.884 result. The approximately
95-point configuration was not available in this conversation for testing.
Actual Kissat and PySAT invocations were not exercised in this environment;
they use the earlier engine's adapters. The native and pure-Python backends
were exercised. See `VALIDATION.md` for the validation record.
