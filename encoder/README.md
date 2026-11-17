# Grid encoder V1

Model definition, input schema, normalization, self-supervised pretraining and representation probes of the
frozen grid encoder (GPS-style graph transformer: width 192, 8 layers, 4 heads, 16 learnable global tokens,
10,096,706 parameters).

**Weights** are distributed separately:
[gridfm-encoder-v1, release v1.0](https://github.com/goldenlcq-debug/gridfm-encoder-v1/releases/tag/v1.0)
(`encoder_v1_weights_20260907.zip`). The README of that repository documents the 25 node columns,
14 branch columns and their normalization.

## Quick check

```bash
python examples/encode_with_public_weights.py --weights-dir /path/to/encoder_v1
```

Builds the encoder from this folder, loads the released tensors with `strict=True` and encodes a generated
four-bus placeholder graph. Expected output shapes: node tokens `[4, 192]`, global tokens `[1, 16, 192]`,
directed-branch tokens `[6, 192]`.

## Contents

| Path | Purpose |
|---|---|
| `gridfm/models/encoder/gps.py` | encoder (edge-aware message passing + global-token attention) |
| `gridfm/schema/` | field tables, sample container, assembly of a sample from PSASP cards |
| `gridfm/data/normalize.py`, `posenc.py`, `graph.py` | normalization, Laplacian / random-walk encodings, graph construction with sample-local identity channels |
| `gridfm/train/pretrain.py`, `masking.py` | masked-reconstruction pretraining and masking rules |
| `gridfm/models/decoder/recon.py`, `gridfm/models/heads/tag_recon.py` | reconstruction heads, identity-channel auxiliary head |
| `gridfm/probes/`, `gridfm/eval/` | retrieval, pointer and linear probes; retention, twin-branch and transfer evaluation |
| `gridfm/data/synth.py`, `sampler.py`, `dataset.py` | synthetic network families, operating-condition sampling, dataset wrapper |
| `gridfm/psasp_io/`, `gridfm/solver/` | PSASP card I/O and solver wrapper |
| `scripts/p1_full_report.py` | representation report of the frozen encoder |
| `probes/encoder_probe_large.py`, `encoder_probe_nk.py` | probes on larger grids and on matched N-0 / N-1 / N-2 sets |
| `probes_tools/` | paired N-k case production with PSASP, assembly, disjointness checks, table builders |
| `examples/encode_with_public_weights.py` | loading example for the public weights |

The probe scripts contain absolute paths of the original workspace; adapt them before use.
