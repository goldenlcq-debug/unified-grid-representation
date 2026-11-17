# Transient-stability assessment reasoner

Predicts the post-fault outcome from the solved prefault state and an explicitly specified disturbance.
The frozen encoder represents the prefault system. The event token combines the frozen token of the faulted
branch with four continuous descriptors (fault location, inception time, clearing time, duration). The
reasoner (16 query tokens, 2 cross-attention layers, 4 heads) outputs an instability score, five
instability-subtype scores, a critical-clearing-time estimate and generator-wise loss-of-synchronism scores.

**Weights** on the Releases page:

- `tsa_reasoner_weights.zip` - five realizations (`seed0..4/best.pt`) with training reports and the
  calibration file. Early stopping selected steps 17,200 / 21,000 / 15,400 / 15,000 / 12,200.
- `tsa_adaptation_ablation_weights.zip` - the five frozen-encoder and five fully adapted runs used for the
  matched-checkpoint comparison. The fully adapted checkpoints contain their adapted encoder copies.

## Contents

| Path | Purpose |
|---|---|
| `gridfm/reason/ts_task.py`, `ts_encoder_io.py` | reasoner, event representation, frozen-encoder loading |
| `gridfm/reason/fault_severity.py`, `dyn_side.py` | auxiliary fault and dynamic-parameter features |
| `scripts/ts_reasoner.py` | training and evaluation (`--unfreeze-encoder none` keeps the encoder frozen; `all` is the full-adaptation arm) |
| `scripts/ts_labels.py`, `ts_relabel.py` | PSASP time-domain simulation and stability labels |
| `scripts/ts_cct.py`, `cct_brackets.py` | critical clearing time by bisection |
| `scripts/ts_coherency.py` | loss-of-synchronism generator groups |
| `scripts/ts_seeds.py`, `ts_ensemble_eval.py`, `calibrate_ts.py`, `ts_eval_batch.py` | multi-seed runs, ensemble evaluation, calibration |
| `scripts/r4_seed_pairing.py`, `r5_gate.py`, `r2_assert_parity.py` | frozen vs. full adaptation at matched checkpoints, input-parity assertions |
| `gridfm/psasp_io/stana.py`, `stout.py` | PSASP transient result parsing |
| `tests/` | unit tests |
