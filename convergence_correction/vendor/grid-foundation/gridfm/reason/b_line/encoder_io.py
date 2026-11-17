"""B1 冻结编码器装载与缺 STATE 输入的唯一入口。"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import fields
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Batch

from ...data.graph import sample_to_data
from ...data.normalize import Normalizer
from ...psasp_io.case import PsaspCase
from ...psasp_io.catalog import Catalog
from ...schema.assemble import build_sample
from ...schema.fields import BRANCH_TABLE, Group, NODE_TABLE
from ...schema.sample import GridSample, group_bit_index
from ...train.pretrain import TrainConfig, build_models
from ..ts_encoder_io import encoder_fingerprint


def _blake16(path: Path) -> str:
    return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()


def assemble_b_sample(case_dir: Path | str, *, sample_id: str, family_id: str) -> GridSample:
    """从卡片装配 B 线原始样本；不读取任何 LF 解算结果。"""
    case = PsaspCase(Path(case_dir))
    return build_sample(
        case, Catalog(case), sample_id=sample_id, family_id=family_id,
        schema="v0", allow_missing_state=True,
    )


def prepare_missing_state_sample(raw: GridSample, normalizer: Normalizer) -> GridSample:
    """施加 v1 normalizer 后将 missing 组重新归零，匹配预训练 group-dropout。

    Normalizer 对原始 0 会做 z-score，得到非零 ``-mean/std``；若不二次归零，冻结编码器
    实际看到的不是预训练整组遮蔽模式。只清被标 missing 的行，普通历史样本不走本入口。
    """
    sample = copy.deepcopy(raw)
    normalizer.transform(sample)
    state = group_bit_index(Group.STATE)
    ns = NODE_TABLE.group_slice(Group.STATE)
    bs = BRANCH_TABLE.group_slice(Group.STATE)
    nmask = sample.node_missing[:, state]
    bmask = sample.branch_missing[:, state]
    sample.node_feat[nmask, ns] = 0.0
    sample.branch_feat[bmask, bs] = 0.0
    if not np.isfinite(sample.node_feat).all() or not np.isfinite(sample.branch_feat).all():
        raise ValueError("B 线归一化输入含 NaN/Inf")
    return sample


class FrozenBEncoder:
    """冻结 v1 编码器 + normalizer；拒绝权重漂移和 schema 错配。"""

    def __init__(self, v1_dir: Path | str, *, device: str = "cpu") -> None:
        root = Path(v1_dir)
        self.checkpoint_path = root / "v1.pt"
        self.normalizer_path = root / "normalizer.npz"
        if not self.checkpoint_path.is_file() or not self.normalizer_path.is_file():
            raise FileNotFoundError(f"冻结编码器双件不完整:{root}")
        ck = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        allowed = {f.name for f in fields(TrainConfig)}
        cfg_dict = {k: v for k, v in ck["config"].items() if k in allowed}
        self.config = TrainConfig(**cfg_dict)
        if self.config.schema != "v0":
            raise ValueError(f"encoder_v1 必须消费 schema v0，实际 {self.config.schema}")
        self.encoder, _decoder = build_models(self.config)
        self.encoder.load_state_dict(ck["model_state"])
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.normalizer = Normalizer.load(self.normalizer_path)
        if len(self.normalizer.node_mean) != NODE_TABLE.total_dim:
            raise ValueError("v1 normalizer 与 schema v0 列宽不匹配")
        self.device = torch.device(device)
        self.encoder.to(self.device)
        self.file_fingerprints = {
            "encoder": _blake16(self.checkpoint_path),
            "normalizer": _blake16(self.normalizer_path),
        }
        self.state_fingerprint = encoder_fingerprint(self.encoder)

    @torch.no_grad()
    def encode(self, raw: GridSample, *, prepared: Optional[GridSample] = None):
        sample = prepared if prepared is not None else prepare_missing_state_sample(
            raw, self.normalizer)
        data = sample_to_data(
            sample, self.config.lape_k, self.config.rwse_steps,
            random_dim=self.config.random_dim,
            branch_random_dim=self.config.branch_random_dim,
        )
        batch = Batch.from_data_list([data]).to(self.device)
        h, g, e = self.encoder(batch)
        return h.cpu(), g.cpu(), e.cpu()

    def assert_still_frozen(self) -> None:
        if any(p.requires_grad for p in self.encoder.parameters()):
            raise RuntimeError("B 线冻结编码器出现可训练参数")
        if encoder_fingerprint(self.encoder) != self.state_fingerprint:
            raise RuntimeError("B 线冻结编码器权重已漂移")
