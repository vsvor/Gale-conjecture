# Optional early stopping of unpromising restarts (version 1.1)

Replace your existing `borsuk_search.py` with this version. The native library ABI
has not changed: keep your existing `dsatur_native.so` beside it. The loader uses
`Path(__file__).resolve()`, retaining the absolute-path fix. No new dependencies
are required. The optimization, evaluation, certificate formats, and the APIs used
by `tune_solution.py` and `split_tune.py` are unchanged.

Early stopping is **disabled unless `--early-stop` is supplied**. It only affects
the `search` command; `evaluate` and `verify` are unchanged.

## Start a search

```bash
python borsuk_search.py search \
    --dim 3 --points 40:100 \
    --restarts 200 --workers 4 --rounds 12 \
    --backend native --early-stop \
    --out fast_search
```

All ordinary options, including `--start`, `--sphere`, and the backend options,
can be combined with early stopping. Use a fresh output directory as before.

The bare flag is equivalent to:

```bash
--early-stop 2:0.02,5:0.005
```

The entries mean **completed optimization rounds : absolute ratio gap**. The
initial point-set evaluation is not a round. Once a checkpoint is reached, its
rule is checked after **every subsequent round**, until the next rule supersedes
it. The final rule remains active through the end of the restart.

| Completed rounds | Stop when the restart's best upper bound is below |
| --- | --- |
| 0 or 1 | No check |
| 2, 3, or 4 | Current series record lower bound minus 0.020 |
| 5 onward | Current series record lower bound minus 0.005 |

For example, with a record of approximately 0.88725, the cutoffs are approximately
0.86725 after two rounds and 0.88225 after five. These are absolute differences in
the ratio, **not percentages**. A margin of 1e-12 avoids decisions from roundoff.

For a single checkpoint, checked after round 3 and every later round:

```bash
--early-stop 3:0.01
```

Round indices must be positive and strictly increasing. Gaps must be finite and
between zero and one. Zero means stop when strictly behind the current record;
it is particularly aggressive and may lose useful later improvements.

## What is compared

Let `L_record` be the greatest saved-coordinate lower bound published by any
restart in the current invocation, and let

```
U_run = max(evaluation.upper for every configuration evaluated in this restart).
```

At an eligible round, stopping occurs when

```
U_run + 1e-12 < L_record - gap
```

When evaluations are exact, this compares the restart's **best ratio so far**
with the current series record. It does not discard a good restart merely because
a later mutation is worse.

When an evaluation times out, its result is a lower/upper bracket. The rule uses
the upper bound, not the possibly weak lower bound. An earlier unresolved bracket
that could contain a competitive ratio can therefore protect a restart for the
rest of its life. This is intentional: UNKNOWN must not be interpreted as a poor
configuration. Early stopping does not add extra SAT calls to tighten brackets.

The shared record is updated only after the corresponding best snapshot has been
saved. It is based on the downward-rounded decimal lower endpoint, with a further
downward floating-point conversion safeguard. No user-supplied or historical
record is silently imported. A `--start` configuration contributes after it is
evaluated by its restart, like any other candidate.

## Parallel execution and saved results

Workers publish improvements immediately, including improvements from unfinished
restarts. A synchronized shared-memory scalar holds the record. The maximum
update is protected by a lock; a weaker late update cannot lower the reference.
No manager server and no repeated scanning of output files are needed. The pool
and shared value use the same multiprocessing context, including with `spawn`.

Stopping is cooperative, **between rounds**. It does not kill a process or
interrupt an in-progress SAT/optimizer call. An early-stopped restart saves its
status, closes its oracle, and returns. Its best snapshot is retained, and the
worker proceeds to another submitted restart. The requested number of restarts
is not reduced, and the whole series is not stopped.

**The remaining rounds and the final SLSQP polish are skipped for a culled
restart.** Competitive and protected restarts retain their ordinary final polish.
If the first applicable checkpoint equals `--rounds`, it can save only the final
polish (and any otherwise pending mutation), not optimization rounds.

Parallel early-stop decisions depend on when other workers discover improvements;
a fixed random seed does not make the stopping decisions independent of worker
timing. No universal speedup or equal-quality final record is guaranteed.

As before, the top-level `best/` snapshot is updated when a restart completes.
The live early-stop record can include a stronger snapshot still stored under an
unfinished worker's `run_XXXXX/best/` directory.

## Keep some full-budget exploration

Defaults:

```bash
--early-stop-warmup 1 --early-stop-keep-every 10
```

The first restart and every tenth restart (10th, 20th, ...) are protected from
this heuristic. They retain the usual round/polish budgets, although existing
termination conditions, such as an empty witness, still apply. Other workers do
not wait for protected restarts to finish. The warmup is a protected set of
restart indices, not a serial preliminary phase.

To make every restart eligible:

```bash
--early-stop-warmup 0 --early-stop-keep-every 0
```

Keeping some protected trials is a safeguard against discarding configurations
that improve only after several rounds or after final SLSQP. Early stopping is a
budget-allocation heuristic, not a mathematical bound on their future potential.

## Logs

A stop prints a line such as:

```text
run 0003 EARLY STOP after 2 rounds: best upper=0.720539548, record lower=0.827474957, gap=0.02; remaining rounds and final polish skipped
```

The following files record the decisions:

- `run_XXXXX/early_stop.jsonl`: one entry per completed round when the option is
  enabled, including protected/before-check/continue/stop decisions, the record,
  bound, cutoff, and elapsed time.
- `run_XXXXX/status.json`: final restart status, completed and skipped rounds,
  and whether final polishing was performed or skipped by this heuristic.
- `completed.jsonl` and `summary.json`: the same per-restart results and aggregate
  counts `early_stopped`, `rounds_completed`, and `rounds_skipped`.

These checks do not change saved geometric or graph certificates.

## Validation in this update

Run:

```bash
python -m unittest -v test_borsuk_search test_early_stop
```

All 43 test methods passed: the 17 original methods and 26 new methods. New tests
cover checkpoint timing, exact boundary comparisons, protection, historical bests,
unsaved timeout candidates, skipped mutation/polish, retained checkpoints, the
native path fix, CLI parsing, and concurrent monotone record updates. Real process
sharing was tested with `fork`, `spawn`, and `forkserver` on Linux; the real search
CLI was also exercised with a spawned two-worker pool. This is not a claim of a
native Windows/macOS test.

A three-restart comparison with early stopping disabled produced bit-for-bit
identical saved coordinate arrays to the original engine using the same seeds and
settings. The metadata adds version/status fields, so metadata files are not
byte-for-byte identical.

A native two-worker smoke search on 18 points with 12 restarts and 8 rounds each,
using the default early-stop schedule, stopped 7 restarts and completed 57 instead
of 96 planned rounds (39 skipped). This is a small functionality test, not a
speed or solution-quality benchmark for the larger configurations. Timing can
change which parallel restarts stop. The saved best passed separate verification
using the Python coloring backend.

See `validation/` for the commands, logs, and summary. Actual Kissat/PySAT execution
was not tested for this update; their backend implementations were not changed.

Implementation references: Python's official `multiprocessing` documentation
(shared `Value` and locks) and `concurrent.futures` documentation (pool initializer
and `mp_context`). The record is passed during worker initialization, not as an
ordinary submitted task argument.
