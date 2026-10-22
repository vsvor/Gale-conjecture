# Bounded presets

Every preset is a small argv-based smoke check, not a recommended record-search budget.
Run through `scripts/run_experiment.py --preset configs/NAME.json --name NEW_RUN`.
Use `--dry-run` to inspect expanded arguments. The `--backend` and `--seed` substitutions
are explicit; changing backend to external requires supplying a reviewed solver command.

| Preset | Workflow | Overall limit |
|---|---|---:|
| `smoke_search3.json`, `smoke_search4.json` | Baseline 3D/4D, early stop and two workers | 180 s |
| `smoke_core3.json`, `smoke_core4.json` | One small fresh structured seed and core audit | 180 s |
| `smoke_tune3.json` | Tiny witness-pruning/tuning run, no insertions | 180 s |
| `smoke_distill3.json` | One 3D child with two anchored additions | 240 s |

The tuning and distillation smoke presets now read the retained 81-point configuration
in `best_known/3d_81p_088871/`; their iteration budgets and seeds are unchanged.
`refine_distilled89.json` reads `best_known/3d_89p_088889/`.

All six completed during assembly. Their scientifically weak, tiny-budget results are
not new records. The runner captures nonzero results, including a legitimate no-selection
code from core screening, rather than weakening the criteria.
