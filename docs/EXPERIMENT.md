# Experiment: <short descriptive name>

## Question and hypothesis
What specific change is being tested? What would count as evidence against it?

## Provenance
- Source commit / dirty state and source-hash record:
- Exact command / preset and any overrides:
- Input paths, coordinate representation and SHA-256 values:
- Python/dependency/compiler versions; native library hash:
- Dimension, k, number of points; seeds:
- Backend(s), per-query timeouts, optimizer/pruning/fit budgets:
- Worker count and overall wall-time authorization:

## Controls
Baseline command, matched budget, seeds, input family and known limitations.

## Completion and failures
Exit code, stop reason, wall time, SAT/UNSAT/UNKNOWN counts, interrupted stages,
partial checkpoints and changed input/source flags. Do not omit unsuccessful trials.

## Results
Report finite-set lower and upper bounds, whether exact squared endpoints agree,
canonical CSV versus binary NPY semantics, point count and minimum separation.
For cores, specify anchored/normalized, mandatory lower bound and found upper bound.
For fits, report assignments/exceptions, objective and residual distribution.

## Verification
Exact commands, backend(s), output hashes, verification JSON, and unperformed checks.
A completed process or SLSQP success is not a verification. State whether a SAT proof
checker was actually used (none is supplied in this repository).

## Interpretation and next decision
What changed, what did not, confounders, precision effects, and the one next experiment
justified by the evidence. Distinguish an improved lower bound from a resolved exact
finite-set value, and both from a global extremality claim.
