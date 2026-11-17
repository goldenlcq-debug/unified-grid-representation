# Unified power-grid representation: code and reasoner weights

Code and model weights accompanying the manuscript
**"A unified power-grid representation for reuse across network structures and computational tasks"**.

A self-supervised, task-agnostic graph encoder (encoder V1) is pretrained once and then kept frozen.
Five independently trained task-specific reasoners read its node, branch and global tokens:
power-flow calculation, reactive-power adjustment, operating-mode generation, power-flow convergence
correction and transient-stability assessment (TSA).

## Where things are

| Component | Folder | Weights |
|---|---|---|
| Grid encoder V1 (model definition, input schema, normalization, pretraining, representation probes) | [`encoder/`](encoder/) | Separate repository: [gridfm-encoder-v1](https://github.com/goldenlcq-debug/gridfm-encoder-v1), release `v1.0` |
| Power-flow calculation reasoner (physics hints + four embedded correction rounds) | [`power_flow/`](power_flow/) | `power_flow_reasoner_weights.zip` (3 realizations) |
| Reactive-power adjustment reasoner (unit selection, set-point classes, shunt selection, unit count) | [`reactive_power_adjustment/`](reactive_power_adjustment/) | `reactive_adjustment_reasoner_weights.zip` |
| Operating-mode generation (latent diffusion denoiser, readout, convergence prescreener) | [`operating_mode_generation/`](operating_mode_generation/) | `mode_generation_weights.zip` |
| Power-flow convergence correction (Double DQN on a candidate-graph Q network) | [`convergence_correction/`](convergence_correction/) | - |
| Transient-stability assessment reasoner | [`transient_stability/`](transient_stability/) | `tsa_reasoner_weights.zip` (5 realizations); `tsa_adaptation_ablation_weights.zip` (frozen vs. fully adapted encoder, 5 pairs) |

The reasoner weights are attached to the [Releases](../../releases) page of this repository.
File sizes and SHA-256 checksums are listed in [`WEIGHTS.md`](WEIGHTS.md).

## How the code is organized

Each task folder is a self-contained snapshot: the entry scripts used for that task together with the
subset of the shared `gridfm` library they import. The tasks were developed on separate branches of the
library, so the `gridfm` copies differ slightly between folders; they are published as they were run and
are intentionally not merged. Run the scripts of a task with that task's folder on `PYTHONPATH`.

Typical contents of a task folder:

- `gridfm/` - library subset (schema, PSASP card I/O, graph construction, encoder, task modules under `gridfm/reason/`)
- `scripts/` (and task-specific folders such as `adj2/`, `nk_eval/`, `probes/`) - training, evaluation and data-generation entry points
- `tests/` - unit tests that ship with the task
- `requirements*.txt` - Python dependencies used for that task

## What you need to know before running

- **Simulator.** Operating cases, labels and all physical verification are produced with the commercial
  Power System Analysis Software Package (PSASP). PSASP is not included and cannot be redistributed.
  Scripts that call the solver expect a local PSASP installation.
- **Data.** No grid datasets are included. Paths to data roots, run directories and the PSASP executable
  appear as absolute paths or constants in several scripts and must be adapted to your environment.
- **Frozen encoder files.** The reasoner scripts load the frozen encoder from a directory holding the
  original training checkpoint (`v1.pt`) and `normalizer.npz`, and some of them check a BLAKE2b fingerprint
  of that checkpoint. The public encoder release contains the identical encoder tensors with training-only
  state removed; [`encoder/examples/encode_with_public_weights.py`](encoder/examples/encode_with_public_weights.py)
  shows how to load it. Use the shipped normalization statistics as they are; do not re-fit them.
- **Language.** Code comments and docstrings are mostly in Chinese.
- **Tested environment.** The encoder example was run with Python 3.8, PyTorch 2.4.1, PyTorch Geometric 2.6.1.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

To be added when the accompanying paper is published.
