# Experiments and reproducibility

## Code work and experiment work are different tasks

Fix a source revision before comparing methods. Use separate worktrees for concurrent
editing; never modify code while a long process is running. A source change can affect
new worker imports even while other workers retain old modules. The runner detects
before/after hash differences but does not freeze the source or observe every intermediate
write. Use a clean commit for serious comparisons.

Run the quick tests and a small instance first. A requested experiment should specify
dimension, k, point budget, seeds, backend/timeout, worker count, optimization/pruning
budgets, overall wall time, output path and intended validation. Agent defaults are in
AGENTS.md. An explicit `--allow-long` flag is a runtime gate, not human authorization.

## Runner

```bash
python scripts/run_experiment.py \
  --preset configs/smoke_search3.json --name pilot3 --seed 123 --backend native

tail -f experiments/pilot3/console.log
```

Presets are JSON objects with a command **argument array**, optional input-file list,
and wall-time budget. The runner does not use a shell. Literal substitutions are:

| Token | Meaning |
|---|---|
| `{python}` | The exact interpreter running the runner |
| `{root}` | Absolute repository root |
| `{out}` | Fresh run directory's `result/` path; the child creates it |
| `{seed}` | Runner seed argument |
| `{backend}` | Runner backend argument |

Quote these tokens in custom shell invocations so they reach the runner literally:

```bash
python scripts/run_experiment.py --name tune-small --timeout 180 \
  --input best_known/3d_81p_088871/points.csv --backend native -- \
  '{python}' tune_solution.py best_known/3d_81p_088871 --k 4 \
  --cycles 1 --witness-trials 1 --trials 0 --slacks 0 --edge-prune 20 \
  --steps 3 --delete-checks 0 --no-compact-trials --refreshes 1 \
  --slsqp-iters 10 --polish-passes 2 --polish-kicks 0 \
  --seed '{seed}' --backend '{backend}' --sat-timeout 3 --out '{out}'
```

For an external solver, use a custom command containing its explicit `--solver` path
(or a reviewed preset that includes it). Changing only the runner's `--backend external`
does not automatically choose your Kissat binary.

`--dry-run` prints the expanded plan and checks named inputs without launching a command
or creating output. `--input FILE` is repeatable and accepts files, not directories.
Presets identify main inputs; the general source manifest additionally hashes templates,
best-known configurations and code in the repository. External input sidecars should be named explicitly.

## Recorded output

```text
experiments/pilot3/
    run.json            argv, cwd, hashes, Git state, packages, outcome and timing
    console.log         complete stdout/stderr of the child
    EXPERIMENT.md       hypothesis/result template to fill in
    result/             the research script's own output and certificates
```

`run.json` is process provenance, not a scientific result schema. Source and input hashes
are captured before/after. `source_changed_during_run` and `inputs_unchanged` flag possible
contamination. Source inventory excludes experiments, validation logs, caches, venvs and
Git internals; a local native-library hash is recorded separately. No entire environment,
credentials or authentication state is recorded. Do not pass secrets in argv or print
them in child logs.

For custom commands, seed/backend fields are annotations unless the corresponding token
is actually used by the child. The expanded command is authoritative. Nonzero statuses
and failed launches are retained. Console output is written to a file, not streamed
back line-by-line by the runner; use `tail -f` from another terminal.

A zero exit code means the process completed, not that its finite ratio is resolved,
its core is minimum, or its record is verified. Inspect the child's summary and run
its appropriate verifier.

## Budgets and interruption

The runner defaults to 300 seconds unless a preset supplies a shorter limit. Budgets
above 600 seconds require `--allow-long`. Explicitly authorize a long job before an
agent supplies that flag. A different subprocess tool must not be used to evade limits.

On POSIX, each experiment gets a new process group. On timeout/interruption the runner
sends SIGINT, then SIGTERM, then SIGKILL with short grace periods to stop workers too.
Termination may exceed the nominal budget by those grace periods. The runner returns
124 for a timeout, 130 for interruption, and preserves other child exit statuses.
Direct native-Windows termination is not a full descendant-cleanup guarantee; the
recommended environment for multiprocessing is Linux/WSL. This runner is not a sandbox.

Completed per-run checkpoints can survive a killed search. An in-progress snapshot may
be incomplete. Use only the script's completion markers and verifier, not the mere
existence of `points.npy`. Never interpret forced termination as graph UNSAT.

## Resume deliberately

- Baseline/tuner: start a new run from a selected saved solution; this is a new search,
  not replay of interrupted worker state.
- Distillation: pass the run root to restore its manifest and beam; passing best keeps
  only the incumbent's lineage.
- Surface/edge sampling: use latest to retain neutral additions and membership data.
- Core generator: selected is a screened warm-up seed; no lower bound is inherited from
  the old fitted source simply because its geometry looks similar.

Always use a new output directory for continuation. Do not relabel an exploratory branch
as a verified incumbent just to preserve it; maintain both.

## Scientific comparison

Use equal wall-time and/or explicitly equal computational budgets, multiple deterministic
seeds, both positive and negative runs, and actual timeout rates. A better fit or larger
raw graph is not necessarily a better ratio after tuning. For a claimed core benefit,
report the exact core definition and both mandatory-vertex lower bounds and found-core
upper bounds where applicable. Small pilots do not establish long-run expected gain.

For a ratio improvement record both old and new canonical-coordinate lower/upper bounds,
input/output hashes, a non-colorability witness, an explicit partition for an exact-value
claim, and a second-backend check when feasible. Do not imply DRAT/LRAT checking. Retain
minimum separation and geometric/polish qualifications. Use `docs/EXPERIMENT.md`.
