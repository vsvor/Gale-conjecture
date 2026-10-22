# Geometric analysis of the 61-point R4 configuration

This package analyzes the ORIGINAL `dim4_61points_0913.csv`; it does not optimize
or replace that configuration. It concerns five parts (R_5), not four parts.

## Main findings

The saved source has matching exact squared partition bounds with

    0.913148536990 <= R_5(X) <= 0.913148536991.

The witness was checked by the native and separate Python coloring backends.
`verify_result.py` rechecks the decimal geometry and Python noncolorability.
No independent SAT proof checker is used.

The proposed four-long-arc/eight-short-arc model is NOT established. A stronger
empirical pattern in this particular sample is SIX groups close to two-dimensional
spherical patches in affine three-dimensional hyperplanes of R4. Those hyperplane
normals form two triples close to orthogonal 2D subspaces. This suggests exploring
an R2 + R2 organization rather than imposing the R1 + R3 analogy of two tetrapods.
It is not a proof of an underlying continuous construction or intrinsic dimension.
Sparse portions of these surfaces could themselves consist of several curves.

## Reproduce

    python -m pip install -r requirements.txt
    python fit_geometry.py --out rerun
    python verify_result.py
    python test_analysis.py

The included `reproduced_fit/fit_results.json` was created by the first command.
`analysis.json` additionally contains exploratory symmetry, circle-screening,
grouping-sensitivity, and subset checks. `patch_template.json` is a compact,
32-parameter surface template with original indices, centers, normals and radii.
Coordinates use original coordinate directions, normalized by the source diameter
(which is 1 at displayed floating-point precision). The input centroid is at zero
to floating-point precision. `point_residuals.csv` lists per-point assignments and
4D residuals; its 16 unassigned rows were not silently discarded from the source.

## Selection before fitting

Connect source points at Euclidean distance less than 0.18 times the diameter.
Take connected components with at least six points. This selects six disjoint
components of sizes 7, 7, 9, 7, 6, 9 (45 points), without fitting circles or spheres
first. Six large components also occur at thresholds 0.19--0.21, with five more
points included. The threshold is a diagnostic choice, not a canonical partition.

The unassigned source indices (zero-based) are:

    2,4,8,9,11,14,22,27,34,35,37,38,45,51,54,56

They must not be presumed dispensable. Removing all 16 gives a 45-point subset
with exact matching squared bounds and R_5 approximately 0.864706824375579.
Those checks and the actual witness/partition are retained in `analysis.json`.

## Fits and residuals

A circle is S^1 in an affine 2-plane of R4. A spherical patch here is part of S^2
in an affine 3-plane of R4; it has two intrinsic parameters, not one.
The optimizations use true Euclidean point-to-primitive residuals in R4, jointly
optimizing the plane, center, and radius. Each model uses a finite multistart
least-squares search, not a global optimality proof.

Across the 45 selected points:

    six independent circles:          RMS 0.056984736566486
    six independent S2 surfaces:      RMS 0.003749810745337
    S2 normals in orthogonal planes:  RMS 0.003977783740130 (40 shape parameters)
    additionally shared projections
      of centers in opposite planes: RMS 0.004283595633358 (32 shape parameters)

Leave-one-out RMS for the independent models:

    circles: 0.11986454
    S2:      0.00917320

The increase on held-out points indicates real uncertainty. In particular,
individual patch radii are not accurate to all printed digits. Six surfaces
fitting six local groups better than six circles does NOT mathematically exclude
a different grouping into twelve arcs plus an unspecified irregular set.

## Compact template

For each group, `patch_template.json` supplies a unit normal n, center c and r:

    n dot (x-c) = 0
    ||x-c||^2 = r^2

These are fitted supporting surfaces, not a claim that full spheres belong in
an extremal set. The finite patch domains and safe resampling must be designed
separately. The poles in the JSON are extrapolated points on these surfaces;
they need not be original vertices and their mutual distances slightly exceed 1.
No ratio is claimed for the projected or reconstructed model.

The frame's first two columns span plane A; the last two span plane B. The
sphere normals for groups 2,3,6 lie in A; groups 1,4,5 have normals in B.
The independently fitted plane-normal spans had principal angles
89.21672599 and 89.96430856 degrees before orthogonality was imposed.
The six extrapolated poles form two nearly equilateral triangles in parallel
translates of these mutually orthogonal planes. Their side lengths range from
0.99775656 to 1.00473962. Their cross-triangle distances range from about 0.8027
to 0.8329, near sqrt(2/3). The dashed triangles in the projection plots show
these extrapolated poles, not edges of the original configuration.

## Symmetry

All 61 EXACT sorted squared-distance signatures of the source points differ.
Any distance-preserving permutation must preserve signatures, so it fixes every
point. An exact nonzero affine determinant confirms full affine span R4.
The source set therefore has no nonidentity Euclidean symmetry. This does not
exclude approximate symmetries of a fitted continuous model.
An exploratory reflection search attained RMS correspondence error 0.06042865;
it is not a certified global minimum over reflections.

## Files and checks

- `source_points.csv`: unchanged original coordinates.
- `partition.json`, `witness_edges.csv`, `evaluation.json`, `verification.json`:
  ratio calculation and verification.
- `fit_geometry.py`: geometric reproduction, including leave-one-out checks.
- `patch_template.json`: full-precision compact surface-model parameters.
- `reproduced_fit/fit_results.json`: full fit output.
- `analysis.json`: exploratory diagnostics and all local model parameters.
- `point_residuals.csv`: all 61 point assignments/errors; no hidden exclusions.
- `projection_A.png`, `projection_B.png`: orthogonal 2D projections for inspection.
- `test_analysis.py`, `tests.log`: nine passing focused checks.
- `borsuk_search.py`, `dsatur_native.cpp`: supporting ratio-checking engine.

The package does not benchmark a new initializer or claim a new record. Its
purpose is model diagnosis before choosing a 4D distillation prior.
