# Reactive-power adjustment reasoner

Corrective control for a solved operating state with voltage violations. The frozen encoder reads the solved
case unmasked. Task-side inputs per bus: under-voltage depth, over-voltage depth, switchable-shunt candidate
mask and a violation-proximity feature (graph distance to the nearest violating bus). The reasoner
(1,343,251 trainable parameters; 16 query tokens, 2 layers) outputs generator-selection scores, one of four
terminal-voltage set-point changes (-0.05, -0.02, +0.02, +0.05 p.u.), shunt-selection scores and a
graph-level unit count.

**Weights:** `reactive_adjustment_reasoner_weights.zip` on the Releases page - one realization
(`best.pt`, variant `v6d`) with `train_report.json`. Supervised labels came from IEEE118, ACTIVSg2000 and
case3120sp.

## Contents

| Path | Purpose |
|---|---|
| `adj2/adj2_train.py` | training (`ReactiveCountReasoner`; variant table incl. `v6d`) |
| `adj2/adj2_eval.py` | PSASP-in-the-loop evaluation: initial action, model-guided ladder (`--ladder ...`), locality prescreen (`--ptr-local-hops H`), rule fallback kept separate |
| `adj2/adj3_scale_diag.py`, `adj2/hop_census.py` | prescreen activation diagnostic and census for the locality radius |
| `adj2/viol_evalgen_mp.py`, `adj2/viol_evalgen_nk.py` | construction of voltage-violation cases; matched N-0 / N-1 / N-2 variants |
| `adj2/run_nk_reactive.py` | driver of the matched-topology evaluation |
| `adj2/adj2_common.py` | shared constants and helpers (contains workspace paths to adapt) |
| `gridfm/reason/reactive_task.py`, `adjust_task.py` | task definitions, side features, action decoding |
| `scripts/` | label generation and action application used by the above |
| `deploy/run_adjust.py`, `deploy/scripts/adjust_ext.py` | deployment entry point; an overlay that expects this folder's `gridfm/` on `PYTHONPATH` |
| `tests/` | unit tests of the task modules |

A corrective action counts as successful only if the PSASP power flow converges and all valid bus voltages
lie inside the examination band of the grid.
