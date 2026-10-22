# Exception-preserving geometric distillation

This is an automated **fit -> resample -> optimize -> refit** search for finite
point sets with large minimax part diameter / whole-set diameter. The current
geometric model supports R^3. The number of parts defaults to four.

**No existing point is projected onto a fitted curve or deleted.** Original
exception indices are preserved throughout a lineage, excluded from every fit,
and free to move during geometric optimization. Newly poorly fitted points are
also retained, rather than discarded as noise.

The distribution contains an actual successful two-generation experiment:

| Configuration | Points | Ratio, approximately |
|---|---:|---:|
| Source | 81 | 0.8887119258981534 |
| After first generation | 89 | 0.8888253201903541 |
| Best after second generation | 89 | **0.8888923464654452** |

The saved `best89/` result has exactly matching squared lower/upper bounds on
its canonical decimal coordinates. `validation/independent_best.json` contains
an additional Python-backend verification. These experiments are not a global
optimality claim or an equal-compute comparison with other initialization schemes.
The best has a roundoff-scale near-coincident pair created by FREE optimization;
89 is not a claim of its necessary or minimum cardinality. See the diagnostics.

## Installation and a tested first run

Keep all files in this directory together. The previous optimizer installation
need not be changed. The included engine has the absolute native-library path fix.

```bash
python -m pip install -r requirements.txt

# Unix-like systems, or reuse a compatible .so beside borsuk_search.py:
c++ -O3 -std=c++17 -shared -fPIC dsatur_native.cpp -o dsatur_native.so

# The tested two-generation experiment. A new output directory is required.
python distill_search.py search examples/record81 \
    --cycles 2 --children 3 --beam 2 --workers 2 \
    --add 8 --max-points 105 \
    --local-rounds 2 --steps 30 --prune-checks 250 \
    --slsqp-iters 100 --polish-passes 1 \
    --final-polish-passes 4 --final-slsqp-iters 180 \
    --fit-iters 40 --backend native --sat-timeout 5 \
    --seed 20260910 --out my_distillation
```

`./reproduce.sh my_distillation` supplies those arguments. The actual saved
coordinates/certificates should be used when exact final digits matter. Different
SciPy/BLAS/solver versions, or solver timeouts, can change optimization paths.

For Kissat, replace the backend arguments with:

```bash
--backend external --solver /absolute/path/to/kissat --sat-timeout 60
```

For optional PySAT, install `python-sat` (not `pysat`) and use `--backend pysat`.
The slower exact Python fallback is `--backend python`. Actual Kissat and PySAT
execution were not exercised for this release; their existing engine adapters
are included. Every backend's UNKNOWN/TIMEOUT remains unresolved, never UNSAT.

## Continue the whole population, not just the incumbent

```bash
python distill_search.py search my_distillation \
    --cycles 6 --children 3 --beam 2 --workers 3 \
    --add 8 --max-points 161 \
    --backend native --out continued_distillation
```

Passing a RUN ROOT reads `manifest.json`, restores its certified incumbent and
its retained exploratory parents, and adds the requested number of generations.
All output is written into the NEW directory. Previously completed nodes are
not overwritten. An interrupted run can be used in the same way; the last
committed beam and the most recently published best survive. Partial-generation
worker outputs are retained, but not all unfinished trials are replayed.

To continue only the supplied improved individual:

```bash
python distill_search.py search best89 \
    --cycles 6 --workers 3 --add 8 --max-points 153 \
    --backend native --out beyond89
```

This starts from that individual's saved model/exception lineage. It does not
retain the other beam members from the original experiment. `latest/` similarly
holds one preferred exploratory parent; the run root is the full resume entry.

## What one generation does

1. Every parent has a fitted model for its CURRENT coordinates. Original and
   newly identified exceptions remain present. The model is refitted after each
   child's free-coordinate optimization.
2. With the default `--children 3`, two children receive new points and the last
   is a no-insertion control. With `--children 1`, there is one growing child and
   no control. Children use deterministic individual random seeds.
3. Existing canonical rational coordinates are retained during resampling. New
   points are sampled in arclength gaps of the fitted arcs/branches and near the
   exception points. Near-shortest witness-edge endpoints receive extra sampling
   weight. A minimum spacing prevents repeated near-identical INSERTIONS.
4. All new-old and new-new distances are checked. New points alone are contracted
   toward the midpoint of an existing diameter pair when necessary, followed by
   exact integer checks. Hence the seeded set S contains its parent X and has
   exactly the same diameter. Therefore R_k(S) >= R_k(X).
5. The child's non-colorable witness is changed by threshold relaxation and
   UNSAT-preserving edge pruning. Annealed L-BFGS and budgeted SLSQP then move ALL
   its coordinates, including exceptions. There are no curve constraints in this
   geometric step. The true finite-set objective is evaluated again.
6. One best exploratory parent and additional near-best alternatives form the
   next beam. Neutral additions can survive multiple generations. The certified
   best is kept separately and can never be replaced merely for fitting better.

There is no automatic vertex compaction or merging. A saturated `--max-points`
budget disables additional insertions, but witness/coordinate optimization and
refitting continue. `--min-separation` applies to new insertions only: free
optimization can bring two previously distinct vertices very close. This is
explicitly reported in `geometry_diagnostics.json`.

## Refitting, grouping, and symmetry

The bundled starting template contains three long circular arcs and six short
branches fitted previously to the reliable portions of the 81-point set. The ten
protected source indices are:

```
9, 12, 16, 44, 50, 51, 52, 53, 54, 56
```

They are excluded from every curve fit even if later geometry approaches a
curve; they are never removed from optimization. Other points with current
point-to-curve distance greater than `--fit-tol` (default 0.012 D) become dynamic
exceptions. They can rejoin a curve later if they come close again, but are never
deleted in either case. Fits use a broader 3*tolerance proposal gate followed by
a tighter membership check.

The geometric fitting objective is a locally regularized soft-L1 least-squares
loss on closest-point residuals. There are two fit/reclassify sweeps. Crowded
arclength bins are downweighted so repeated nearby samples do not dominate. A
curve with too few inliers keeps its previous fit; no unsupported new primitive
is invented. Circle projection is analytic, while quadratic-arc projection
checks the real stationary roots and endpoints.

The model is registered to new coordinates by a similarity transformation using
persistent row identities. Optimizer translations/rotations are thereby separated
from actual changes of shape. No observation is changed by this registration.

`--symmetry auto` uses paired fits/sampling only when the registered reflection
correspondence AND the entire current point cloud pass `--symmetry-tol` (default
0.0002 D). This is a numerical prior, not an exact-symmetry theorem. `--symmetry off`
lets the model fit its branches independently. No symmetry is forced on free
optimization. Every fourth generation, one child defaults to unpaired sampling
(`--break-symmetry-every 4`), even when the parent is almost symmetric.

`--branch-kind parabola` uses the earlier parabolic short-branch template for a
FRESH run; long arcs remain circular. After refitting, its branches are planar
quadratic arcs, not sections of a shared prescribed paraboloid. Likewise, the
circular branches need not keep a common junction after refitting. These are
sampling guides, not a rigid global ansatz. On resume the saved model type is
retained; changing this flag does not silently change its meaning.

The method refits the existing curve families. It does NOT automatically discover
new curve families for the unresolved points, enforce second-order stationarity,
or include the separate paired-vertex-split search from earlier packages.

## Input formats and avoiding wrong index assignments

Use the bundled 81-point source, its descendants with sidecars, or a fresh point
set with explicit group indices.

For a fresh standalone input without lineage, the default 81-point template
requires its ordered reference rows to match after similarity registration
(RMS <= 0.02 D). A reordered or unrelated array should not be given the old ten
exception indices blindly. Use an explicit grouping instead:

```json
{
  "groups": {
    "arc_A": [0, 2, 5, 7],
    "branch_B": [1, 3, 8, 9]
  },
  "exceptions": [4, 6]
}
```

Each group needs at least three non-collinear points. Overlap at a junction is
permitted. Unassigned points are also protected exceptions. Custom groups start
with independently fitted circles and no inferred mirror constraint. The supplied
`templates/groups81.json` is a complete example for the source's original order.

```bash
python distill_search.py search my_points.npy \
    --groups my_groups.json --out custom_run --backend native
```

CSV decimals are canonical and read exactly. As in the existing tuner, NPY/NPZ/
other floating inputs are normalized and rounded once to a 17-decimal canonical
snapshot before the initial evaluation. All later preservation guarantees refer
to this evaluated snapshot. Low-precision CSV grids are promoted when adding
points by appending decimal zeros, without changing their rational values.
`points.npy` remains a convenience floating copy. Exact certificates concern CSV.

Checkpoint coordinates, row order, and exception lineage are hash-checked. Do not
manually remove/reorder vertices or replace a checkpoint's coordinate file. To
start after external deletion/reordering, supply a fresh explicit grouping.

## Most useful controls

| Option | Default | Meaning |
|---|---:|---|
| `--cycles` | 8 | Additional generations |
| `--children` | 3 | Children per parent, including a no-insertion control |
| `--beam` | 2 | Exploratory parents retained |
| `--beam-gap` | 0.002 | Allowed absolute ratio deficit in the exploratory beam |
| `--add` | 12 | Maximum additions per growing child |
| `--max-points` | automatic | Point cap; never silently deletes to meet it |
| `--fit-tol` | 0.012 | Residual cutoff divided by diameter |
| `--exception-fraction` | 0.20 | Target share of proposals near exceptions |
| `--jitter` | 0.001 | New-point perturbation / D; scaled across children |
| `--min-separation` | 0.001 | Spacing cutoff for new insertions / D |
| `--local-rounds` | 3 | Witness-change/optimization rounds per child |
| `--prune-checks` | 400 | Graph edge-deletion checks per local round |
| `--slacks` | 0,0.004,0.012 | Relative threshold relaxations cycled locally |
| `--opt-jitter` | 0.001 | Coordinate kicks inside optimization / D |
| `--steps` | 60 | L-BFGS steps at each smoothing level; 0 disables |
| `--polish-passes` | 2 | Budgeted SLSQP calls per local round |
| `--final-polish-passes` | 4 | Separate full-constraint incumbent polish; 0 disables |
| `--workers` | 1 | Parallel independent child jobs |

Early abandonment (`--early-after 2 --early-gap 0.015`) compares a child's
highest upper bound seen with the record lower bound AT JOB DISPATCH. It protects
a previously competitive iterate and a competitive unresolved bound. This is
not the live record mechanism of `borsuk_search.py`. It only acts at local-round
boundaries; it never interprets a SAT timeout as poor quality.

For a model-only check: `--cycles 0 --final-polish-passes 0`. For a sampling-only
check: `--local-rounds 0 --final-polish-passes 0`. Both still evaluate/save the
appropriate finite-set certificates.

## What is accepted as a new best?

Write L(Y) <= R_k(Y) <= U(Y). A geometrically different candidate can replace the
incumbent only if its exact rational squared lower bound exceeds the incumbent's
squared upper bound, and the displayed ratio gain meets `--min-gain` (1e-10 by
default). Neither a lower fit residual nor an optimizer success flag is sufficient.
Bounds for IDENTICAL coordinates can be tightened without changing the set.

The exploratory beam is different: it deliberately retains neutral and near-best
branches so several discrete moves can combine. It is not a certified improvement
sequence. The `best/` sequence is protected.

## Output and verification

- `best/`: certified incumbent, with all ordinary engine/viewer sidecars.
- `latest/`: one preferred exploratory parent; use the root for full beam resume.
- `nodes/`: immutable initial/resumed configurations.
- `generations/####/p##_c##/seed/`: insertion certificate and original-row-preserving seed.
- `generations/####/p##_c##/optimized/`: actual optimized points and newly fitted model.
- `model.json`, `fit.json`, `lineage.json`: evolving curves, residuals, and exceptions.
- `geometry_diagnostics.json`: point spacing and near-coincidence warnings.
- `final_polish/`: separate final full-constraint SLSQP/LP check and its actual coordinates.
- `progress.jsonl`, `manifest.json`, `summary.json`: event log, resume state, final summary.

The final polish may round coordinates differently without exceeding `--min-gain`.
Its numerical stationarity report then applies to `final_polish/`, not necessarily
to the byte-identical coordinate rows in `best/`. This is a fixed-witness numerical
check only, not a proof of optimality. Its candidate is still independently evaluated.

```bash
python distill_search.py verify my_distillation/best --backend python --sat-timeout 120
python borsuk_search.py verify my_distillation/best --backend native
python view_solution.py my_distillation/best --hull
python view_solution.py my_distillation/best --hull --gif turntable.gif --no-show
```

Graph UNSAT results rely on the selected solver/search. There is no independently
checked DRAT/LRAT proof in this package. Geometry and replacement decisions use
integer arithmetic. See `VALIDATION.md` for tests, the actual experiment, and limits.

## Tests and documentation for dependencies

```bash
python -m unittest discover -s . -p 'test_distillation.py' -v
```

Thirty focused tests cover geometry, protection, timeout safety, exact insertion,
coarse coordinate-grid promotion, serialization, resume and spawned workers. The
spawn/native integration test is skipped until the local native library is compiled.

Implementation reference: SciPy's official `least_squares` documentation explains
soft-L1/f_scale and local convergence; `minimize(method='SLSQP')` documents the
finite stopping/feasibility tolerances. These numerical fits and solver termination
flags are intentionally not used as graph/geometry certificates.
