# -*- coding: utf-8 -*-
"""E19(预注册 §22)单测:边头零初始化(§22.1 E19a)与边基线换源 `--edge-base bdelta`(E19b)。

夹具沿用 E18 的 mp_case118_mode(有 sidecar 变比);"基态"同样用**独立构造**的自洽方式
(注入 ×0.97 的带网损 DC),只验机制不验精度 —— 精度由离线闸对真基态验(§22.3 G1/G2)。
"""
from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gridfm.data.normalize import Normalizer
from gridfm.reason.pf_physics import dc_theta_lossy
from gridfm.reason.pf_task import PFReasoner
from gridfm.reason.pf_theta_edge import DTH_DC_COL, EDGE_SIDE_DIM, potential_reconstruct
from gridfm.schema.fields import NODE_TABLE
from gridfm.schema.sample import GridSample
from gridfm.train.pretrain import TrainConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pf_reasoner as M  # noqa: E402

FX = Path(__file__).parent / "fixtures" / "mp_case118_mode"
noff = NODE_TABLE.offset_of


@pytest.fixture(scope="module")
def raw():
    r = GridSample.load_npz(FX / "mp_case118_00000.npz")
    z = np.load(FX / "family_extra.npz")
    r.branch_tap, r.branch_shift_rad = z["branch_tap"], z["branch_shift_rad"]
    return r


@pytest.fixture(scope="module")
def base_npz(raw, tmp_path_factory):
    """写一份"基态缓存"npz(注入 ×0.97 的自洽 DC 解),供 load_base_hint 读。"""
    nf = raw.node_feat.copy()
    for f in ("pg", "pl"):
        nf[:, noff(f)] *= 0.97
    rb = replace(raw, node_feat=nf)
    d = dc_theta_lossy(rb)
    p = tmp_path_factory.mktemp("base_cache")
    np.savez(p / f"{raw.family_id}.npz", theta=d[:, 0].astype(np.float64),
             pinj=(nf[:, noff("pg")] - nf[:, noff("pl")]).astype(np.float64))
    return p


def _data_kw(raw):
    norm = Normalizer().fit([raw])
    s = copy.deepcopy(raw)
    norm.transform(s)
    _vc = noff("v")
    return s, TrainConfig(), dict(raw=raw, phys=True, edge=True, vhint=True,
                                  vstat=(float(norm.node_mean[_vc]), float(norm.node_std[_vc])), shz=True)


# ---------------------------------------------------------------- E19a 零初始化

def test_zero_init_zeros_last_layer_and_keeps_structure():
    """零初始化只动末层参数:形状/键与非零初始化完全一致,末层恒零、其余非零。"""
    kw = dict(d=32, n_queries=4, n_layers=1, n_heads=4, side_dim=8, edge_side_dim=EDGE_SIDE_DIM,
              theta_edge=True, theta_wrap=True)
    torch.manual_seed(0)
    a = PFReasoner(**kw)
    torch.manual_seed(0)
    b = PFReasoner(**kw, edge_head_zero_init=True)
    assert a.state_dict().keys() == b.state_dict().keys()
    assert torch.count_nonzero(b.edge_head[-1].weight) == 0 and torch.count_nonzero(b.edge_head[-1].bias) == 0
    assert torch.count_nonzero(a.edge_head[-1].weight) > 0          # 对照:默认初始化非零
    assert torch.equal(a.edge_head[0].weight, b.edge_head[0].weight)  # 其余层不受影响


def test_zero_init_guards():
    """零初始化需要 theta_edge 且需要残差基线(否则 δ≡0,θ 退化为常数场)。"""
    with pytest.raises(ValueError, match="theta_edge"):
        PFReasoner(d=32, n_queries=4, n_layers=1, edge_head_zero_init=True)
    with pytest.raises(ValueError, match="edge_dc_residual"):
        PFReasoner(d=32, n_queries=4, n_layers=1, side_dim=8, edge_side_dim=EDGE_SIDE_DIM,
                   theta_edge=True, edge_dc_residual=False, edge_head_zero_init=True)


def test_zero_init_forward_equals_edge_baseline_integral(raw):
    """§22.3 G1 的单测版:零初始化的未训练模型,θ 输出 == 边基线 Δθ 的势重建积分(≤1e-5 rad)。"""
    from torch_geometric.data import Batch
    s, cfg, kw = _data_kw(raw)
    d = M._pf_data(s, cfg, "group", dc="lossyv", **kw)
    batch = Batch.from_data_list([d])
    side_dim, ed = batch.side.shape[1], batch.edge_side.shape[1]
    torch.manual_seed(0)
    r = PFReasoner(d=32, n_queries=4, n_layers=1, n_heads=4, side_dim=side_dim, edge_side_dim=ed,
                   theta_edge=True, theta_wrap=True, v_hint=True, edge_head_zero_init=True).eval()
    dd, n_e = 32, batch.edge_index.shape[1]
    h = torch.zeros(batch.num_nodes, dd)
    g = torch.zeros(1, 4, dd)
    e = torch.zeros(n_e, dd)
    with torch.no_grad():
        pred, _delta = r(h, g, e, batch.batch, batch.batch[batch.edge_index[0]],
                         side=batch.side, edge_index=batch.edge_index, edge_side=batch.edge_side,
                         edge_use=batch.edge_use, theta_ref=batch.theta_ref, ref_mask=batch.ref_mask,
                         reach_mask=batch.reach_mask, return_edge=True,
                         v_hint=batch.v_hint, v_hint_ok=batch.v_hint_ok)
        ref = potential_reconstruct(batch.edge_side[:, DTH_DC_COL].float(), batch.edge_index,
                                    batch.edge_use.float(), batch.batch, batch.theta_ref, batch.ref_mask,
                                    num_graphs=1, ridge_mask=~batch.reach_mask)
    from gridfm.reason.pf_task import PF_FIELDS
    from gridfm.reason.pf_theta_edge import wrap_angle
    th = pred[:, PF_FIELDS.index("theta")]
    m = batch.reach_mask
    assert torch.abs(th[m] - wrap_angle(ref)[m]).max().item() < 1e-5


# ---------------------------------------------------------------- E19b 边基线换源

def test_edge_base_acv_is_default_and_bitwise(raw):
    """不传 edge_base == 传 "acv":Data 各张量逐位一致(旧检查点安全)。"""
    s, cfg, kw = _data_kw(raw)
    M._BASE_HINT_CACHE.clear()
    d0 = M._pf_data(s, cfg, "group", dc="lossyv", **kw)
    d1 = M._pf_data(s, cfg, "group", dc="lossyv", edge_base="acv", **kw)
    for k in ("edge_side", "edge_use", "theta_ref", "ref_mask", "v_hint", "phys_b", "side", "target"):
        np.testing.assert_array_equal(getattr(d0, k).numpy(), getattr(d1, k).numpy(), err_msg=k)


def test_edge_base_bdelta_without_cache_falls_back_bitwise(raw, monkeypatch):
    """无基态缓存(门控不通过):bdelta 与 acv 逐位一致 —— 退回路径不得引入任何差异。"""
    s, cfg, kw = _data_kw(raw)
    monkeypatch.setattr(M, "BASE_CACHE_DIR", Path("__no_such_dir__"))
    M._BASE_HINT_CACHE.clear()
    d0 = M._pf_data(s, cfg, "group", dc="lossyv", edge_base="acv", **kw)
    M._BASE_HINT_CACHE.clear()
    d1 = M._pf_data(s, cfg, "group", dc="lossyv", edge_base="bdelta", **kw)
    for k in ("edge_side", "theta_ref", "ref_mask", "v_hint", "side"):
        np.testing.assert_array_equal(getattr(d0, k).numpy(), getattr(d1, k).numpy(), err_msg=k)


def test_edge_base_bdelta_swaps_only_edge_columns(raw, base_npz, monkeypatch):
    """有基态缓存(门控通过):**只有边侧的 Δθ/p_dc 两列换源**,
    Q–V 提示 / 参考角 / 可达 / 侧通道 / 物理常量一列不动(§22.1 的核心约束)。"""
    s, cfg, kw = _data_kw(raw)
    monkeypatch.setattr(M, "BASE_CACHE_DIR", base_npz)
    M._BASE_HINT_CACHE.clear()
    d0 = M._pf_data(s, cfg, "group", dc="lossyv", edge_base="acv", **kw)
    M._BASE_HINT_CACHE.clear()
    d1 = M._pf_data(s, cfg, "group", dc="lossyv", edge_base="bdelta", **kw)
    assert M.base_delta_seed(raw) is not None, "夹具应当通过门控(有缓存 + 有 tap + rel 小)"
    e0, e1 = d0.edge_side.numpy(), d1.edge_side.numpy()
    p_col = M.EDGE_SIDE_FIELDS.index("p_dc") if hasattr(M, "EDGE_SIDE_FIELDS") else 6
    changed = {DTH_DC_COL, p_col}
    for c in range(e0.shape[1]):
        if c in changed:
            assert not np.array_equal(e0[:, c], e1[:, c]), f"第 {c} 列应换源却逐位相同"
        else:
            np.testing.assert_array_equal(e0[:, c], e1[:, c], err_msg=f"第 {c} 列不该动")
    for k in ("theta_ref", "ref_mask", "reach_mask", "v_hint", "v_hint_ok", "side", "phys_b", "edge_use"):
        np.testing.assert_array_equal(getattr(d0, k).numpy(), getattr(d1, k).numpy(), err_msg=k)


def test_edge_base_bdelta_reverse_edge_antisymmetry(raw, base_npz, monkeypatch):
    """换源后反向边的 Δθ 仍严格反号(势重建的前提;换错列会破坏这一性质)。"""
    s, cfg, kw = _data_kw(raw)
    monkeypatch.setattr(M, "BASE_CACHE_DIR", base_npz)
    M._BASE_HINT_CACHE.clear()
    d = M._pf_data(s, cfg, "group", dc="lossyv", edge_base="bdelta", **kw)
    e = d.edge_side.numpy()
    m = e.shape[0] // 2
    np.testing.assert_array_equal(e[:m, DTH_DC_COL], -e[m:, DTH_DC_COL])


def test_base_delta_seed_matches_lossybv_side_channel(raw, base_npz, monkeypatch):
    """重构核对:抽出的 `base_delta_seed` 与 E18 侧通道用的是同一个数组(逐位)。"""
    s, cfg, kw = _data_kw(raw)
    monkeypatch.setattr(M, "BASE_CACHE_DIR", base_npz)
    M._BASE_HINT_CACHE.clear()
    d = M._pf_data(s, cfg, "group", dc="lossybv", **kw)
    M._BASE_HINT_CACHE.clear()
    seed = M.base_delta_seed(raw)
    assert seed is not None
    side = d.side.numpy()
    dc_cols = side[:, side.shape[1] - 3 - 3: side.shape[1] - 3]      # v-hint 3 列之前的 dc 3 列
    np.testing.assert_allclose(dc_cols, seed, rtol=0, atol=0)
