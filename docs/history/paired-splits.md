# Tuning the supplied 73-point configuration

This package contains actual computations on the user's file `points(3)(1).npy`.
The input's SHA-256 is

    819e91d5440d8030ca390a54302f2ad1c277f95d083cacc2d93f627e67a28f85

The original bytes are retained as `input/original.npy`; they were not modified.
The objective throughout is **minimax part diameter / whole-set diameter**, with
four parts in R^3. The successful coordinate optimization is unrestricted in R^3;
it is NOT a search restricted to the original hull or its skeleton.

## Results

| Directory | Points | Ratio (approximately) | Verified decimal bracket |
|---|---:|---:|---|
| `input` | 73 | 0.8878496018804404 | [0.887849601880, 0.887849601881] |
| `paired_seed` | 72 | 0.8879156006604563 | [0.887915600660, 0.887915600661] |
| `compact71` | 71 | 0.8883064706858724 | [0.888306470685, 0.888306470686] |
| `same73` | 73 | **0.8883487217283716** | [0.888348721728, 0.888348721729] |
| `best` | 81 | **0.8887119258981534** | [0.888711925898, 0.888711925899] |

Every row has exactly matching lower/upper **squared** bounds for its saved
canonical CSV coordinates. The displayed interval width is outward rounding
of the square root. The witnesses were checked with the native and the separate
pure-Python coloring implementations. No independent DRAT/LRAT proof checker was
used, and none of these results is a global extremality or literature-record claim.

The increase over the original input is about 0.0004991198479312 with the same
73 points, and about 0.0008623240177130 for the 81-point result.

Minimum pair separations in diameter units:

- `best`: approximately 0.0016831126438354;
- `same73`: approximately 0.0058220311874251;
- `compact71`: approximately 0.0025900166923574.

The final reported configurations do not rely on roundoff-scale duplicates.
`validation/packaged_verification.json` also checks the exact **binary** values
of each distributed NPY file, separately from the decimal CSV. The NPY binary
bounds need not coincide exactly, but give the same displayed brackets.

## Verify and view

Python >=3.10, NumPy and SciPy are required. Matplotlib is only for the viewer.

    python -m pip install -r requirements.txt
    python borsuk_search.py verify best --backend python --sat-timeout 120
    python borsuk_search.py verify same73 --backend python --sat-timeout 120
    python view_solution.py best --hull

To repeat all packaged checks, including the exact binary NPY arithmetic:

    python verify_report.py --output my_verification.json

To use native optimization:

    c++ -O3 -std=c++17 -shared -fPIC dsatur_native.cpp -o dsatur_native.so

The engine's absolute native-library path fix is included. The engine and tuner
come from the previously supplied vertex-split package; optimization semantics
have not been changed. This is a separate reproduction directory, not an instruction
to overwrite an existing installation or its early-stopping version.

Actual Kissat/PySAT execution was not exercised in these runs. The existing
backends remain available; for an additional independent solver check:

    python borsuk_search.py verify best --backend external \
        --solver /absolute/path/to/kissat_bin --sat-timeout 120

## The missing move: a neutral first split, followed by a second split

Initial full-witness SLSQP and alternative-witness pruning did not yield a
meaningful improvement. The familiar one-step split screen also stalled.

Three input pairs were nearly coincident (zero-based indices):

    (21,68): approximately 1.66e-16
    (39,69): approximately 8.92e-16
    (48,67): approximately 2.69e-15

Two points could be deleted with the exact ratio preserved. For exploration,
we removed indices 67,68,69, retaining 70 points. This changed the original
ratio downward by about 5.6e-16, so **exact preservation is not asserted**.
`distinct70` contains that precise exploratory seed. All reported improvements
were subsequently compared against and verified above the original 73-point
upper bound, not merely this very slightly smaller starting value.

The important new search keeps a first non-4-colorable split scaffold even when
its numerical first-order gain is zero. It applies another split before asking
SLSQP to improve the geometry. This permits a pair of discrete moves that the
one-step positive-gain filter would never optimize.

The particularly useful pair on `distinct70` was:

1. Split vertex 26 into indices 26 and 70. Omit neighbor 45 from the original
   copy, and neighbor 64 from the new copy.
2. Split vertex 55 into indices 55 and 71. Omit neighbor 26 from the original
   copy, and neighbors 15 and 56 from the new copy.

Copies begin at the old coordinates only as auxiliary optimization seeds;
there is no edge between a pair of copies. Both scaffolds are explicitly
checked non-4-colorable. The four copy vertices are essential to the joint
scaffold: deleting any one makes it 4-colorable (also checked with Python).

For this move the first-only numerical LP gain is zero, and a full-constraint
SLSQP call terminates in one iteration at the old value. For the joint scaffold
the LP gain is approximately 0.00664232477452946 in the LP's velocity
normalization. Full-constraint SLSQP reaches about 0.887915600660458 in 24
iterations. The number 0.006642 is NOT the actual ratio increase.

The main exploratory run used generated diameter constraints; the separate
full-constraint check reached the same useful value to floating precision.
The saved `paired_seed` is the actual scout result subsequently used by the
successful branch, not a substituted/reconstructed configuration.

See `validation/pair_move_audit.json`, `validation/paired_tests.log`, and
`paired_seed/optimization_edges.csv` / `optimization_seed.npy`.

## Replay the first successful paired move

    python replay_pair73.py input/original.npy --out replay_pair --backend native

This script deliberately checks the original attachment's fingerprint before
applying index-based changes. It is a replay for THIS input, not a generic tuner.
The replay was actually run twice here and produced bit-for-bit identical saved
NPY coordinates to `paired_seed` on this environment.

## Follow the successful branch

This exact command was run on `paired_seed`:

    python split_tune.py paired_seed \
        --out split_chain --backend native --sat-timeout 2 \
        --rounds 12 --max-omit 3 --trials 24 --per-vertex 4 \
        --polish-passes 6 --save-trials --seed 739

It reached 0.8883336735986498 with 76 points, saved here as `chain76`.
Intermediate exact reductions lowered the cardinality at several stages. The
first reduction was 72 -> 67 without a loss of ratio. The full actual history
and graph-screen reports are in `experiments/branch19/`.

A subsequent continuation from `chain76` used:

    python split_tune.py chain76 \
        --out continued_chain --backend native --sat-timeout 2 \
        --rounds 12 --max-omit 3 --trials 16 --per-vertex 4 \
        --polish-passes 6 --save-trials --seed 740

This continuation was stopped at the experiment time budget during round 12,
after already finding 0.8886861257815922 with 81 points. Its actual saved
incumbent is `continuation_seed81`. Its interrupted log/history are retained;
we do NOT claim the round, family, or optimization was exhausted. Letting the
command finish can change the final trajectory or improve its result further.
The subsequently reported `best` received a SEPARATE completed full-constraint
polish and exact verification, so it is not an unaudited interrupted scout.

## Reproduce the final 81-point improvement

The following completed run starts from the supplied `continuation_seed81`:

    python tune_solution.py continuation_seed81 \
        --out final_tune --backend native --sat-timeout 3 \
        --cycles 1 --witness-trials 4 --trials 0 --slacks 0 \
        --edge-prune 2000 --steps 0 --delete-checks 0 \
        --no-compact-trials --polish-passes 6 --full-constraints --seed 7981

It reached **0.8887119258981534**, still with 81 points. The final cleanup found
no near-duplicate points; it performed another full-constraint polish and audit.
The saved result is `best`.

Here `--trials 0` disables insertions in `tune_solution.py`, not the four witness
trials. Do NOT add `--polish-only` to this reproduction: that would disable the
witness changes responsible for the last improvement.

Numerical libraries, solver choices and thread implementations can affect last
digits and branching. The exact saved coordinates and certificates, rather
than a promise of bitwise reproducibility on every platform, define the results.
The recorded environment was Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, Linux x86-64.

## The same-cardinality and compact results

Two independent postprocessing routes were used. The 73-point intermediate
was optimized with four witness trials (seed 7973, `--edge-prune 1800`,
`--full-constraints`) to make `fixed73_before_cleanup`. Separately, `chain76`
was optimized with the analogous four-witness run (seed 7976). The exact routes are:

- `fixed73_before_cleanup` -> remove indices 66,69 -> `compact71`;
- `chain76` -> seed-7976 four-witness polish -> `pruned76`;
- `pruned76` -> remove indices 65,74,75 -> `same73`.

The postprocessing reductions can lower the ratio by a few 1e-16; exact
preservation is NOT asserted. Each resulting configuration was reassessed and
polished, and its own fresh certificates are provided. Both remain far above
the original input, with exact comparison available in the verification data.

## Generic paired-split exploration

`paired_splits.py` is the exploratory two-step driver used in these computations
(for k=4). Unlike the single-split program it keeps first scaffolds regardless
of their immediate LP gain. For example:

    python paired_splits.py distinct70 --out paired_sweep \
        --max-omit 3 --zero-trials 16 --seed 739 --backend native

The actual run screened 16,496 second-split graphs, found 118 positive-LP
candidates, and optimized them plus 16 zero-gain perturbed candidates. There
were no solver timeouts. It saved improving alternatives, including
`improvement_019` (the successful branch used here). The largest immediate
paired improvement was NOT the branch that gave the best subsequent result.
This is a reason to retain several promising branches, rather than only the
single highest immediate score.

Budget controls: `--first-limit`, `--limit` (second-split tests), `--prune`,
`--zero-trials`, and `--sat-timeout`. This is an experimental driver, not a
complete enumeration of all possible graph changes or simultaneous splits.
It restricts the second original index to exceed the first and uses at most
three omitted near-shortest neighbors by default. UNKNOWN never means UNSAT.

## Validation and limitations

- 56 existing engine/tuner/split tests passed with the native library present.
- Four new focused tests passed: exact binary-coordinate ingestion, explicit
  two-split edge mapping, independent non-colorability/essentiality checks for
  the actual move, and the original input fingerprint.
- Original and final witnesses were independently checked with Python.
- Exact integer squared-distance checks cover both CSV and binary NPY versions.
- The primary `best` has matching lower/upper squared bounds. SLSQP reports are
  NOT used as certificates for the ratio.
- `best/polish.json` and `same73/polish.json` record repeated successful SLSQP
  passes, perturbation checks and numerical near-active LP audits. These are
  fixed-witness numerical stopping checks, not local or global optimality proofs.
- Initial ordinary polishing, 32 alternative-witness trials across original/
  cleaned inputs, and 95 forced critical-edge-replacement trials found no
  meaningful improvement before the paired-move search. Other branches and
  unsuccessful trials are retained under `experiments/`.
- No independent SAT proof log/checker was used. No claim of a published record
  or an exhausted search is made.

Run the regression tests after compiling the native library:

    python -m unittest -v test_borsuk_search test_tune_solution test_split_tune test_paired_replay

Detailed results are in `validation/result_summary.json`,
`validation/packaged_verification.json`, and the saved certificates.
