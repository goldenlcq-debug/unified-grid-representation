# -*- coding: utf-8 -*-
"""推理器电压残差头(--v-hint,预注册 §14.2)单测:接线、梯度、关闭时逐位兼容、D15。"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from gridfm.data.normalize import Normalizer
from gridfm.reason.pf_task import PF_FIELDS, PF_SIDE_DIM, PFReasoner
from gridfm.reason.pf_theta_edge import EDGE_SIDE_DIM
from gridfm.schema.fields import NODE_TABLE
from gridfm.schema.sample import GridSample
from gridfm.train.pretrain import TrainConfig, build_models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pf_reasoner as M  # noqa: E402

FX = Path(__file__).parent / "fixtures" / "mp_case118_mode"
VC = NODE_TABLE.offset_of("v")


@pytest.fixture(scope="module")
def pack():
    raw = GridSample.load_npz(FX / "mp_case118_00000.npz")
    n_hit, fams = M.attach_family_extra([raw], [FX.parent])      # FX.parent 下没有 families/ → 不命中
    assert n_hit == 0
    z = np.load(FX / "family_extra.npz")
    raw.branch_tap, raw.branch_shift_rad = z["branch_tap"], z["branch_shift_rad"]
    norm = Normalizer().fit([raw])
    s = copy.deepcopy(raw)
    norm.transform(s)
    cfg = TrainConfig()
    vstat = (float(norm.node_mean[VC]), float(norm.node_std[VC]))
    return raw, s, norm, cfg, vstat


def _batch(pack, vhint):
    raw, s, norm, cfg, vstat = pack
    d = M._pf_data(s, cfg, view="group", raw=raw, dc="lossy", phys=True, edge=True,
                   vhint=vhint, vstat=vstat if vhint else None)
    return Batch.from_data_list([d])


def test_pf_data_attaches_hint_and_side_cols(pack):
    b0, b1 = _batch(pack, False), _batch(pack, True)
    assert b1.side.shape[1] == b0.side.shape[1] + 3
    assert hasattr(b1, "v_hint") and hasattr(b1, "v_hint_ok")
    assert b1.v_hint_ok.sum() > 50
    # 残差基 = z 化的 v̂:无效处 0;有效处与侧通道首列相同
    assert torch.equal(b1.v_hint, b1.side[:, -3])
    assert float(b1.v_hint[~b1.v_hint_ok].abs().max()) == 0.0


def test_reasoner_v_hint_forward_grad_and_off_bitwise(pack):
    raw, s, norm, cfg, vstat = pack
    enc, _ = build_models(cfg)
    enc.eval()
    b0, b1 = _batch(pack, False), _batch(pack, True)
    from gridfm.reason.pf_physics import pf_phys_denorm
    dn = torch.from_numpy(pf_phys_denorm(norm))
    kw = dict(d=cfg.d, n_queries=4, n_layers=1, gnn_layers=1, theta_head=True,
              edge_side_dim=EDGE_SIDE_DIM, theta_edge=True, theta_wrap=True)
    torch.manual_seed(0)
    r0 = PFReasoner(side_dim=b0.side.shape[1], **kw)
    torch.manual_seed(0)
    r1 = PFReasoner(side_dim=b1.side.shape[1], v_hint=True, **kw)
    # 配对判错:开了 v_hint 却不给张量 / 没开却给
    with pytest.raises(ValueError):
        M._forward(enc, r1, b0, False, torch.float32, torch.device("cpu"), dn=dn)
    p1, loss1 = M._forward(enc, r1, b1, False, torch.float32, torch.device("cpu"), dn=dn)
    assert torch.isfinite(p1).all()
    loss1.backward()
    assert all(p.grad is not None for n, p in r1.named_parameters() if "head" in n)
    # 残差语义:把头的 v 输出强制为 0 → 预测 v(归一化空间)== v̂
    with torch.no_grad():
        r1.head[-1].bias[PF_FIELDS.index("v")].zero_()
        r1.head[-1].weight[PF_FIELDS.index("v")].zero_()
    p2, _ = M._forward(enc, r1, b1, False, torch.float32, torch.device("cpu"), dn=dn)
    kv = PF_FIELDS.index("v")
    assert torch.allclose(p2[b1.v_hint_ok, kv], b1.v_hint[b1.v_hint_ok], atol=1e-6)
    # 关闭 v_hint 的推理器与旧结构逐位一致(同 seed、同输入)
    torch.manual_seed(0)
    r_ref = PFReasoner(side_dim=b0.side.shape[1], **kw)
    pa, _ = M._forward(enc, r0, b0, False, torch.float32, torch.device("cpu"), dn=dn)
    pb, _ = M._forward(enc, r_ref, b0, False, torch.float32, torch.device("cpu"), dn=dn)
    assert torch.equal(pa, pb)


def test_v_hint_side_channel_leak_free(pack):
    """D15:扰动被监督量后,侧通道里的提示列与残差基逐位不变。"""
    raw, s, norm, cfg, vstat = pack
    from dataclasses import replace
    from gridfm.train.masking import PF_UNKNOWN_FIELDS
    b1 = _batch(pack, True)
    nf = raw.node_feat.copy()
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    for i in range(raw.n_nodes):
        for f in PF_UNKNOWN_FIELDS[int(types[i])]:
            nf[i, NODE_TABLE.offset_of(f)] += 3.3
    raw2 = replace(raw, node_feat=nf)
    raw2.branch_tap, raw2.branch_shift_rad = raw.branch_tap, raw.branch_shift_rad
    d2 = M._pf_data(s, cfg, view="group", raw=raw2, dc="lossy", phys=True, edge=True, vhint=True, vstat=vstat)
    b2 = Batch.from_data_list([d2])
    assert torch.equal(b1.v_hint, b2.v_hint) and torch.equal(b1.side[:, -3:], b2.side[:, -3:])
