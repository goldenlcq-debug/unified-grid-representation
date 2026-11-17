# Power-flow calculation reasoner

Recovers the unknown AC operating state (bus voltage magnitude and angle, unknown generator P/Q) from an
unsolved case. The frozen encoder sees topology, parameters and branch status; specified quantities enter on
the reasoner side with known-value flags. The reasoner (2,456,846 trainable parameters; 16 query tokens,
2 cross-attention layers, 4 message-passing layers) predicts branch angle differences that are integrated to
bus angles by least-squares potential reconstruction, uses two zero-training hints (loss-aware DC angles and
a Q-V voltage estimate) and applies four embedded physical-correction rounds with learnable step sizes.

**Weights:** `power_flow_reasoner_weights.zip` on the Releases page - three independently trained
realizations (`seed0..2/best.pt`) with their training reports (`pf_report.json`: 2,500 steps, learning rate
3e-4, hint dropout 0.2, correction dropout 0.2, auxiliary pre-correction loss weight 1). They were trained on
encoder V1.

## Contents

| Path | Purpose |
|---|---|
| `gridfm/reason/pf_task.py` | reasoner, task-side inputs, loss |
| `gridfm/reason/pf_physics.py` | Kirchhoff mismatch, loss-aware DC hint, Q-V hint, correction steps |
| `gridfm/reason/pf_theta_edge.py` | branch-side features and angle reconstruction |
| `scripts/pf_reasoner.py` | training and evaluation entry point |
| `scripts/eval_pf_ckpt.py`, `eval_holdout_5060.py`, `eval_by_family_5060.py` | checkpoint evaluation on held-out and per-family sets |
| `scripts/diag_theta_nodes.py` | per-bus diagnostics on the large transfer grids |
| `scripts/probes_5060/make_master_table.py`, `make_two_reasoner_table.py` | accuracy tables per grid |
| `scripts/probes_5060/probe_refine_ablate.py`, `probe_posthoc_refine.py` | with / without correction rounds |
| `nk_eval/` | evaluation on matched N-0 / N-1 / N-2 cases; `hint_ref_eval.py` evaluates the zero-training hints alone |
| `tests/` | unit tests of the physics terms, hints, correction rounds and data plumbing |

Run names in the scripts follow the original experiment log; the released realizations correspond to
`pf_reasoner_5060_R1dcL_edge0_wrap_L_vh_rf4qv_aux1_hd0.2_rd0.2_s{0,1,2}`.
