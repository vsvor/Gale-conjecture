# Geometric priors, exceptions, and scope

These are empirical models of particular saved configurations, not theorems about all
Borsuk extremizers. Fitting/initialization is kept separate from the finite evaluator.

## Three-dimensional data

`templates/source3.npy` and `templates/curves3.json` describe the earlier 81-point
configuration. Its adapted frame records a numerical reflection correspondence.
Three long circular arcs and reliable near-junction sections of two curved tripods
fit many points well. Ten zero-based source indices are explicitly unresolved:

    9, 12, 16, 44, 50, 51, 52, 53, 54, 56

They are not discarded as noise. Comparisons are stored in
`templates/curves3_comparison.json`; circular and parabolic short-branch alternatives
exist. `compare_curves.py` can refit the source comparison into a **new** output file.
The default output is `experiments/curve_comparison.json`, not a template overwrite.

`structured_seed.py` retains the full source and adds diameter-safe curve/neighborhood
samples. `core_seed.py` instead freshly samples regular components while retaining
exceptions, so it does not inherit the old ratio. `curve_model.py` supplies evolving
local models for `distill_search.py`; those models can move away from exact common
junctions, and symmetry is a conditional sampling prior rather than a hard restriction
on the free optimizer. Row lineage and source/model hashes must match.

## Four-dimensional data

`templates/source4.csv` and `templates/patches4.json` concern the 61-point five-part
configuration. Its empirical model contains six small S² patches in affine three-planes,
organized into two approximately orthogonal triples, plus 16 unassigned points. A patch
is a two-dimensional surface in R⁴, not a full S³ hypersphere.

Sampling must be limited to neighborhoods supported by the observed local patch
coordinates; a sphere fit does not justify filling a whole sphere. `core_seed.py`
provides structured R⁴ initialization, spatial separation and distance-signature
selection. The current distillation workflow does **not** refit R⁴ patches automatically.
A future R⁴ implementation belongs in model helpers and the workflow layer, not in the
coloring engine. Do not reuse 3D exception indices or a reflection permutation in R⁴.

## Model versus certificate

Never project every point to a fitted primitive and claim its old ratio survives.
A fit can reduce diameter constraints, introduce duplicates or alter critical edges.
Keep residual components separate and let free optimization discover additional shape.
For anchored additions, verify that every old point remains unchanged and all new-old
and new-new distances stay within the old diameter. For fresh starts, explicitly state
that there is no inherited bound. Inspect exact final saved geometry in either case.

Dense model sampling alone does not ensure a larger core. Component quotas, separation,
short warm-up and core diagnostics are practical selectors, not minimum-core proofs.
Historical fit/pilot details are preserved in `docs/history/`, with their original
version-specific limitations. No new fit or expectation estimate was performed merely
by assembling this repository.
