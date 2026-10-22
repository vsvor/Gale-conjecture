# Task 002: optional core-aware replenishment proposal

Scope: inspect `borsuk_search.replenish_unused`, seed/model helpers, and the relevant
engine/core tests. The existing repair puts inactive vertices on short convex combinations
of active ones. Propose an **opt-in** alternative that improves interaction diversity
without moving active witness endpoints or increasing the diameter.

Do not make the low-level engine import a specialized 3D/4D generator. Prefer a generic
proposal/selection interface or a narrowly justified geometry helper. Include new-new
distance checks, deterministic seed handling, and fallback behavior when proposals fail.
Leave default behavior unchanged. Preserve SAT/UNSAT/UNKNOWN and certificate semantics.

Add a small synthetic invariant test, a regression for identical default outputs, and
one bounded R3/R4 smoke comparison. At most two workers and ten minutes combined
experiment time unless a separate benchmark budget is authorized. Report pilot outcomes
without claiming higher expected final record or minimum core size from a tiny sample.
