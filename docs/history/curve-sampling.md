# Curve-informed starting configurations from the 81-point record

This is a fitted-template **initializer**, not a claim of a new ratio record and
not yet an automatic optimize-fit-optimize driver. It includes an actual,
equal-flexibility comparison of circles and parabolas, and diameter-preserving
resampling from the fitted geometry. Original source: the saved 81-point
configuration with ratio approximately 0.8887119258981534.

## Findings

We compare one-sided curves through the observed tripod junctions (+/-0.5,0,0)
in the previously supplied orthonormal UVW frame. W -> -W is imposed. Each
tripod has a central curve in W=0 and a mirrored pair of side curves. The
circular and parabolic models each have six supporting-curve degrees of freedom
per tripod: two for the central curve and four for the mirrored pair. Apex
positions, memberships, weights, and reflection are the SAME for both models.
This is NOT the old model that required all three sections to lie on one common
paraboloid; comparison with its reported errors is not apples-to-apples.

RMS orthogonal point-to-curve errors, diameter units:

| Subset | Points | Straight branches | Circular branches | Parabolic branches |
|---|---:|---:|---:|---:|
| Positive tripod | 13 | 0.00757547228 | 0.00237028764 | 0.00254345678 |
| Negative near-apex core | 10 | 0.00246309029 | 0.00018306684 | 0.00018375321 |

Circles are slightly better in this test. This is not proof of an exact circular
construction or global least-squares optimality. Mirrored samples are not
independent evidence; only 3-4 non-apex points occur on each fitted side branch.
Each fit used 12 numerical initializations. The full original negative group is
also tested in comparison.json, using the same fixed W-sign branch assignment;
its large errors describe a poor one-sided model, not a contradiction of the
previous unconstrained, two-sided fit.

The negative core is selected **geometrically before either fit**: distance at
most 0.25 from (-0.5,0,0), not a method-specific residual cutoff. The furthest
retained point is 0.1595275304 away; the nearest excluded point is 0.2857134910
away. All 13 positive points satisfy the same radius criterion. Excluded
zero-based indices:

    9, 12, 16, 44, 50, 51, 52, 53, 54, 56

Removing these ten points from the actual configuration lowers its ratio to
approximately **0.8546268854596762** (71 points). Exact binary-coordinate
squared lower/upper bounds agree; see results/core71_binary.json. The witness
was independently checked with the Python backend; see core71_verified.json.
They are excluded from the SIMPLE TRIPOD FIT, never from the initializer.

## Run

    python -m pip install -r requirements.txt

    # Generate 201 points, retaining the 81 source rows exactly.
    python structured_seed.py --points 201 --kind circle \
        --jitter 0.001 --out seed201

    # Generate a different-size parabola-based template seed.
    python structured_seed.py --points 301 --kind parabola \
        --jitter 0.002 --seed 7 --out seed301

    # Recompute the comparison (specialized to the supplied ordered 81 points).
    python compare_curves.py --restarts 12 --out results/recomputed_comparison.json

--points must be >=81. There is no built-in upper point-count cap, but geometry
checks and coloring constraints grow quadratically, and finite precision puts
practical limits on packing distinct points. Increase the size gradually.
The source fingerprint is checked: these fitted templates must not silently be
applied to unrelated or differently ordered coordinates.

## What is sampled

* The three previously fitted long circular arcs, with their fitted finite ranges.
* Six short near-apex branches: either circles or parabolas from the NEW comparison.
* A residual reservoir around the ten excluded-from-fit points: small local
  convex combinations with nearby source points. Default 15% of additions.

Arclength-stratified sampling, not uniform parabola parameter sampling, is used.
The side families are sampled in reflected pairs; middle families in W=0.
New jitter respects this reflection. Original rows are preserved and already
obey that reflection to floating-point precision, rather than being projected.
The original 81-point anchor set prevents fit compression from losing its
non-colorability obstruction. All points can subsequently move under the
existing free-coordinate optimizer.

The common midpoint is the radial contraction center. Candidates that would
increase diameter are contracted toward it, with the SAME factor for a reflected
pair. The check includes every previous accepted point AND the two new points'
mutual distance. Testing only membership in the intersection of balls around
old points would not suffice. Contraction, jitter, and reservoir sampling mean
the final additions need not lie exactly on the fitted curves or on the old
hull surface. In particular, this is not another edge-only/hull-surface sampler.

A final integer calculation on the exact stored binary values proves:

    original 81 rows unchanged;
    all rows distinct;
    enlarged-set diameter equals original diameter exactly.

Consequently the true finite-set partition ratio cannot decrease, assuming the
source lower bound. This does NOT establish that it increases. Metadata in
sampling.json states this distinction. --skip-exact-check disables the integer
check for very large experiments; floating-point checks still run.

## Coordinates and optimization

points.npy is canonical for the initializer and its exact binary geometry check.
points.csv is a decimal convenience copy, without a partition certificate.
Use the NPY input explicitly:

    c++ -O3 -std=c++17 -shared -fPIC dsatur_native.cpp -o dsatur_native.so
    python borsuk_search.py search --dim 3 --k 4 \
        --start seed201/points.npy --restarts 16 --workers 4 \
        --rounds 12 --steps 180 --prune-checks 600 --polish 120 \
        --backend native --early-stop --out optimized201

This is a proposed optimization command, not a completed performance benchmark.
The copied engine includes the native path fix and early stopping. Existing
external Kissat support can be used instead; actual Kissat execution was not
exercised in this comparison.

## Tests and sample seeds

    python -m unittest test_structured_seed -v

13 focused tests pass. These cover exact projection on sample curves, parabola
projection against independent polynomial roots, reflection, arclength sampling,
counts/parity, determinism, diameter checks, and residual reconstruction.

seeds/ contains circle161, circle201, circle401, and parabola201. All original
81 rows remain unchanged and their diameters are unchanged by exact integer
checks. Initial, UNOPTIMIZED seed evaluations all give the same ratio as the
81-point parent. Matching exact binary squared bounds are recorded in
results/seed_binary_bounds.json. The lower bound is inherited from the parent
witness; explicit upper-bound partitions are in results/evaluated/.
No optimize-fit cycle or speedup benchmark was performed here, and no independent
DRAT/LRAT proof checker was used. Geometry and partitions have separate checks.

    python check_seed_bounds.py
    python borsuk_search.py verify source --backend python --sat-timeout 30

The .so is deliberately not shipped: compile it for the local platform.

## Suggested distillation loop (NOT implemented automatically)

1. Optimize several configurations freely, prune witnesses, and retain a few
   near-best branches; do not retain only immediate-gain proposals.
2. Fit a robust, reflection-aware mixture of curves to the geometrically coherent
   points. Keep a separate unmodelled reservoir. Re-estimate pose and assignments
   after substantial movement rather than assuming source labels remain correct.
3. Spawn arclength-sampled children with original certified anchors retained;
   enrich intervals involved in near-shortest witness edges, and preserve all
   unresolved exceptional points. Limit oversampling of geometrically redundant
   intervals.
4. Optimize and SAT-reevaluate each child. Keep the certified incumbent separate
   from the fitted/resampled approximation. Accept a new record only after the
   new lower bound exceeds the preceding upper bound (exact coordinate checks).
5. Once a small extra cluster is stable across several good configurations,
   promote it to a new curve or patch. An occasional symmetry-breaking child is
   useful; observed symmetry is not a theorem that an optimum must have it.

Useful alternative to a hard curve constraint: x_i = gamma_j(t_i;theta)+r_i,
with a diminishing penalty on r_i for inliers, no such penalty on unresolved
points, and the graph's diameter/lower-distance constraints enforced separately.
A small geometric RMS alone cannot protect a partition ratio.

Software used in this run: Python 3.13, NumPy 2.3.5, SciPy 1.17.0.
External method reference: SciPy's official least_squares documentation,
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
Robust loss functions discussed in the proposal are not secretly substituted
for the ordinary equally weighted loss used in the numerical comparison.
