"""Load the publicly released encoder V1 weights and run a forward pass on a generated four-node graph.

Download and unpack the weights first (they are not stored in this repository):
    https://github.com/goldenlcq-debug/gridfm-encoder-v1/releases/download/v1.0/encoder_v1_weights_20260907.zip
The zip unpacks to a directory ``encoder_v1/`` containing ``encoder_v1_weights.pt`` and ``normalizer.npz``.

Usage (from the ``encoder/`` directory of this repository):
    python examples/encode_with_public_weights.py --weights-dir /path/to/encoder_v1 [--device cpu|cuda]

No power-grid dataset is needed: the script builds a four-bus, three-branch placeholder sample with all
feature groups marked missing, so it only checks that the model definition in this repository matches the
released tensors (strict ``load_state_dict``) and that the encoder returns node, global and directed-branch
tokens of width 192. To encode a real case, assemble a ``GridSample`` (see ``gridfm/schema``), apply
``norm.transform(sample)`` and call ``sample_to_data`` exactly as below.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]          # the ``encoder/`` directory (contains ``gridfm/``)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_public_encoder(weights_dir, device="cpu"):
    """Return (encoder, cfg, normalizer) built from the public release files. The encoder is frozen."""
    import torch
    from gridfm.data.normalize import Normalizer
    from gridfm.train.pretrain import TrainConfig, build_models

    weights_dir = Path(weights_dir)
    ck = torch.load(weights_dir / "encoder_v1_weights.pt", map_location="cpu", weights_only=True)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in allowed})
    encoder, _decoder = build_models(cfg)            # the reconstruction decoder is not part of the release
    encoder.load_state_dict(ck["model_state"], strict=True)
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.to(torch.device(device)).eval()
    norm = Normalizer.load(weights_dir / "normalizer.npz")   # load only; never re-fit
    return encoder, cfg, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", required=True, help="directory holding encoder_v1_weights.pt and normalizer.npz")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()

    import numpy as np
    import torch
    from torch_geometric.data import Batch
    from gridfm.data.graph import sample_to_data
    from gridfm.schema.fields import BRANCH_TABLE, NODE_TABLE
    from gridfm.schema.sample import GridSample

    encoder, cfg, norm = load_public_encoder(args.weights_dir, args.device)
    assert NODE_TABLE.total_dim == 25 and BRANCH_TABLE.total_dim == 14
    assert (cfg.d, cfg.n_layers, cfg.n_global, cfg.n_heads) == (192, 8, 16, 4)

    sample = GridSample(
        node_feat=np.zeros((4, 25), dtype=np.float32),
        branch_feat=np.zeros((3, 14), dtype=np.float32),
        edge_index=np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        node_missing=np.ones((4, 4), dtype=bool), node_masked=np.zeros((4, 4), dtype=bool),
        branch_missing=np.ones((3, 4), dtype=bool), branch_masked=np.zeros((3, 4), dtype=bool),
        sample_id="synthetic_smoke", family_id="synthetic_smoke")
    norm.transform(sample)
    data = sample_to_data(sample, lape_k=cfg.lape_k, rwse_steps=cfg.rwse_steps,
                          random_dim=cfg.random_dim, branch_random_dim=cfg.branch_random_dim)
    batch = Batch.from_data_list([data]).to(next(encoder.parameters()).device)
    with torch.no_grad():
        node_tokens, global_tokens, branch_tokens = encoder(batch)

    print(json.dumps({
        "ok": True,
        "parameters": sum(p.numel() for p in encoder.parameters()),
        "trainable_parameters": sum(p.numel() for p in encoder.parameters() if p.requires_grad),
        "node_input_shape": list(data.x.shape),
        "branch_input_shape": list(data.edge_attr.shape),
        "node_tokens": list(node_tokens.shape),
        "global_tokens": list(global_tokens.shape),
        "branch_tokens": list(branch_tokens.shape),
        "finite": bool(all(torch.isfinite(t).all().item() for t in (node_tokens, global_tokens, branch_tokens))),
        "torch": torch.__version__,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
