# Operating-mode generation

Conditional generation of feasible operating modes. A latent diffusion denoiser (8 AdaLN-modulated
message-passing layers) operates on the frozen 192-dimensional grid representation and an eight-dimensional
request vector (target aggregate-load level, online non-slack generator count). A message-passing readout
decodes generated latent node states to bus loads and generator active outputs; a fixed convergence
prescreener orders the candidates before PSASP evaluation, and the first candidate that converges and meets
the voltage criterion is delivered.

**Weights:** `mode_generation_weights.zip` on the Releases page - `denoiser/denoiser.pt` (checkpoint at
27,000 steps), `readout/readout.pt`, `prescreener/deploy_head.json`, with training reports.

## Contents

| Path | Purpose |
|---|---|
| `gridfm/models/diffusion/` | denoiser and diffusion schedule |
| `gridfm/models/decoder/readout.py` | readout from latent node states to bus loads and generator outputs |
| `gridfm/data/cond_views.py`, `c_label.py` | encoder input view for generation, request-vector encoding |
| `scripts/g_diffusion_train.py`, `g_diffusion_eval.py` | diffusion training and sampling evaluation |
| `scripts/g15_train_readout.py` | readout training |
| `scripts/g5_prescreen_train.py`, `g5_prescreen_deploy.py`, `g5_prescreen_replay.py`, `g5_probe_*.py` | convergence prescreener |
| `scripts/g6_prescreen_delivery.py` | propose - screen - solve delivery evaluation |
| `scripts/g1_baseline.py` | write-back to PSASP cards and trial solution |
| `scripts/gen_families.py`, `gen_synth_large.py`, `gen_data.py` | synthetic network families and operating-case generation |
| `pkg_scripts/gen_mp_modes.py`, `eval_quality.py`, `audit_static_feasibility.py` | operating cases on MATPOWER-derived grids, delivery criteria, static feasibility audit |
| `nk_eval/`, `nk_eval_synth/` | matched N-0 / N-1 / N-2 request sets |
| `runs/` | small configuration and report files referenced by the scripts |
| `tests/` | unit tests |
