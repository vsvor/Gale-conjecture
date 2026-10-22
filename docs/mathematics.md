# Mathematical and certificate contracts

This is the correctness reference for the supplied code. It states the model used by
this project; empirical templates and optimizer behavior belong in other documents.

## 1. Objective and monotonicity

For a finite set X of distinct points with positive diameter D and k >= 1, let

\[
d_k(X)=\min_{X=X_1\sqcup\cdots\sqcup X_k}\max_i\operatorname{diam}(X_i),
\qquad R_k(X)=d_k(X)/D.
\]

Empty classes and singletons have diameter zero; use at most k nonempty parts. The
finite minimum is attained. If |X| <= k, the answer is zero. We maximize R, not its
reciprocal. The principal cases are (dimension,parts)=(3,4),(4,5),(5,6).

For X subset S, d_k(S) >= d_k(X), because a partition restricts. The **normalized**
ratio need not be monotone if the diameter grows. If the diameters are equal, then
R_k(S) >= R_k(X). This is the basis for anchored, diameter-safe insertions.

## 2. Strict threshold graphs

For t >= 0 let G_t have the points as vertices and an edge whenever distance > tD.
A proper k-coloring of G_t is exactly a partition with each part's diameter <= tD:

\[
G_t\text{ is k-colorable}\iff R_k(X)\le t.
\]

Every SAT partition must be checked against the actual graph and its permitted color
range. UNKNOWN means no conclusion. A solver error is not a colorability answer.

For descending pair distances d[0] >= ... >= d[N-1], if q is the first UNSAT prefix
**length**, the value is d[q-1]/d[0]. The q-edge graph supplies the lower bound and
any coloring of the previous prefix supplies the upper bound. Equal distances may
appear in any stable order; the index formula does not change. The exact-state
evaluator groups equal integer squared-distance levels explicitly.

## 3. Witness and partition bounds

For any non-k-colorable graph H on the current vertices,

\[
L_H(X)=\frac{\min_{uv\in E(H)}\|x_u-x_v\|}{D}\le R_k(X).
\]

H need not be the full geometric threshold graph. Removing edges is valid only when
the remaining graph is again definitely UNSAT. A subgraph may expose useful geometric
motions by removing unnecessary lower-distance constraints.

For any explicit assignment c into {0,...,k-1}, regardless of its provenance,

\[
R_k(X)\le U_c(X)=\frac{\max_{c(u)=c(v)}\|x_u-x_v\|}{D}.
\]

A timeout leaves a bracket [L,U]. It cannot make the bracket exact, authorize deletion,
prove essentiality, or justify increasing a lower bound. A complete-graph baseline
witness is a combinatorial base case when |X| > k.

## 4. Exactness and rounding

The baseline `evaluate()` chooses graphs using floating distances. Its logical query
answers are exact when resolved, but floating geometry is not an exact-real certificate.
`save_candidate()` subsequently checks the saved coordinate representation using integers.
The tuner instead stores decimal integer rows in `Geometry`, and `assess()` orders exact
integer squared distances. Both workflows must report what representation was checked.

A standard saved decimal point is z/10^p for an integer row z. Squared distances are
integers divided by a common factor; that factor cancels in ratios. Store L² and U²
as rational values, compare by exact cross multiplication, and outward-round square roots.
Matching squared endpoints determine the optimum of the saved finite set, conditional
on the graph non-colorability result. They are not a global extremality theorem.

`points.csv` is authoritative in engine/tuner/distillation/surface/core exports. A
rounded `points.npy` need not have precisely the same exact squared bounds. In contrast,
standalone `structured_seed.py` checks **binary NPY values** and labels its CSV as a
convenience copy; use its explicit NPY path for that guarantee. Surface checkpoints may
use a finer declared scale than the base engine's usual 17 decimal places.

Do not silently reuse certificates after transforming coordinates, changing row order,
normalizing, projecting onto a fitted curve, or modifying a saved file. Recompute them.
Near-coincident points may be distinct rationals, but report their minimum separation
and do not claim their count is a necessary/minimum cardinality.

## 5. Acceptance policies are not identical across workflows

The baseline multistart engine ranks lower endpoints to choose restart/global output.
It preserves a useful lower-bound objective but does not require a new candidate's lower
bound to exceed every earlier upper bound. With unresolved brackets, “best” does not
mean a proved ordering of exact objectives. Early stopping uses a shared record and
conservative upper bounds; it is a throughput heuristic only.

`tune_solution.Checkpoints` uses rational comparison for a geometrically different
incumbent (new lower > old upper) and may retain no-loss smaller configurations under
its explicit tie/reduction policy. Distillation similarly separates an exact-comparison
incumbent from exploratory beam membership and applies its numerical minimum-gain gate.
Other workflows have explicit policies in their implementations. Do not silently replace
them with one global rule during packaging or refactoring.

For a publication-quality strict improvement, independently require the new saved lower
bound to exceed the old saved upper bound on their stated coordinate representations.

## 6. Deletion and the word “core”

A deletion changes the denominator unless a diameter pair remains. Generic normalized
reduction recomputes the subset diameter and accepts a deletion only after proving the
new ratio is at least the previous partition upper bound. This can **increase** the
ratio; a found non-degrading subset is not automatically an equal-ratio core.

For a resolved value a=d_k(X), the inclusive threshold graph with distances >= a is
UNSAT (when a>0 and |X|>k). With one diameter pair anchored, non-colorable induced
subgraphs retaining the anchors preserve the same ratio. A vertex v is mandatory when
G-v is SAT: every induced non-colorable subgraph must contain v, since a graph omitting
v is a subgraph of G-v. These tests give a lower bound on the anchored minimum size;
a found reduction gives an upper bound. UNKNOWN gives neither mandatory status nor
permission to remove v. Anchor obligations and vertex obligations must be counted as
a union, not double-counted.

“Inclusion-minimal”, “minimum cardinality”, “edge-minimal”, “vertex-critical”, “anchored
core”, and “normalized non-degrading subset” are different claims. Repeated greedy
orders do not prove a global minimum. Core-aware generation rejects unresolved audits
from selection rather than fabricating a metric.

## 7. Insertion and splitting

Adding a point to an already UNSAT graph proves nothing about the new point's usefulness.
Insertion scaffolds therefore may start from a known colorable old skeleton and test a
non-colorable extension. Blocking one saved coloring is only a proposal screen: all
old vertices must be allowed to recolor for a genuine lower bound.

Nonadjacent false twins or a nonadjacent dominated neighborhood can be redundant for
color extension. Adjacent true twins are not generically redundant. A split can begin
with geometrically coincident copies whose **witness constraints differ**; it is an
auxiliary realization, not an accepted point set of distinct vertices.

A neutral first split may enable a productive second split. Keep such scaffolds in
exploratory storage, not in the certified incumbent. The current paired prototype is
k=4-specific; using k=5 requires a separately reviewed genericization.

For diameter-safe multiple insertions, checking each point against the old vertices
is insufficient: new-new pairs must also respect the diameter. The fixed original set
must actually remain present before invoking the anchored monotonicity argument.

## 8. Continuous optimization and fitted geometry

For fixed verified H, SLSQP maximizes t subject to all distances² <= 1 and witness
distances² >= t. Working-set variants must check omitted diameter constraints afterward.
Numerical feasibility tolerances, success flags, first-order LP gains and zero-gain
conditions do not certify exact stationarity or global optimality. Recompute the actual
witness ratio and finite objective on retained/final coordinates.

A fit minimizes distances to curves or patches, not R_k. Treat it as a proposal prior.
Never remove high-residual points merely because they do not fit. Distillation retains
all lineages/exceptions while its free-coordinate optimizer can move every vertex.

## 9. Boundary-specific certificates

Convexification preserves diameter and cannot decrease R. Point-graph UNSAT on boundary
samples gives a lower bound for the hull. Patch-graph SAT can give a covering upper bound
because the maximum distance between convex patches occurs among their vertices.
**Patch-graph UNSAT is not a lower bound**: a true partition can split a patch.

A finite coloring on hull-edge samples is not yet a coloring of the whole 1-skeleton.
`verify_skeleton.py` instead checks a saved continuous segment cover, exact coverage of
all true hull edges, and all same-color endpoint distances. A cover can be made into a
partition by assigning each overlap to one of its covering classes.

Boundary membership, finite-set bounds, and full-hull/edge-cover bounds are separate
claims with separate files/verifiers. A 3D projection of 4D data proves none of them.

## 10. Trust boundary

Native and Python coloring implementations can cross-check each other. External solver
status and valid model parsing remain part of the trusted computation. This repository
does not provide an independent DRAT/LRAT proof-checking stage. The native shared library
contains graph search only; it does not know Euclidean coordinates or certify geometry.
