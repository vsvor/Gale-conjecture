# Task 003: plan generalization of paired splits

Initial deliverable is a plan plus deterministic tests, not a large refactor or record
search. Inventory hardcoded k=4 assumptions in paired_splits.py and its use of split_tune.
Identify which are merely CLI defaults and which are combinatorial correctness conditions.
Preserve neutral first moves, exact final comparison, and explicit UNKNOWN outcomes.

Specify a new explicit k interface, compatibility for old k4 commands and checkpoints,
small graph examples with independent expected answers, and enumeration/timeout budgets.
Pay attention to first-split enumeration preceding --first-limit in the current prototype.
Run no R4,k5 optimization before the generic semantics have been reviewed. Do not route
a saved k5 configuration through the present hardcoded k4 script.
