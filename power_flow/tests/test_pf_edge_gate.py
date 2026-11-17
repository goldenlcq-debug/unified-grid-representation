# -*- coding: utf-8 -*-
"""E20(预注册-E20-规模门控边基线-20260825)单测:`edge_base="gate<N>"` 的解析与 `_pf_data` 门控分派。

要点:①阈值编码进现有 `edge_base` 字符串,不新增 `_pf_data` 形参(plumbing REQUIRED 不变);
②n_nodes < N → 与 "acv" 逐位一致;③n_nodes ≥ N → 与 "bdelta" 逐位一致(且确实换了基线);④非法串报错。
夹具与基态注入手法同 tests/test_pf_base_delta.py(118 = 有 sidecar;基态用注入 ×0.97 的带网损 DC 自构)。
"""
from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gridfm.data.normalize import Normalizer
from gridfm.reason.pf_physics import dc_theta_lossy
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
def harness(raw):
    cfg = TrainConfig()
    norm = Normalizer().fit([raw])
    s = copy.deepcopy(raw)
    norm.transform(s)
    _vc = noff("v")
    vstat = (float(norm.node_mean[_vc]), float(norm.node_std[_vc]))
    return cfg, s, dict(raw=raw, dc="lossyv", phys=True, edge=True, vhint=True, vstat=vstat, shz=True)


@pytest.fixture()
def with_base(raw):
    """把自构基态(注入 ×0.97 的带网损 DC)注入 _BASE_HINT_CACHE,退出时清空。"""
    nf = raw.node_feat.copy()
    for f in ("pg", "pl"):
        nf[:, noff(f)] *= 0.97
    rb = replace(raw, node_feat=nf)
    d = dc_theta_lossy(rb)
    M._BASE_HINT_CACHE.clear()
    M._BASE_HINT_CACHE[raw.family_id] = (d[:, 0].astype(np.float64),
                                         (nf[:, noff("pg")] - nf[:, noff("pl")]).astype(np.float64))
    yield
    M._BASE_HINT_CACHE.clear()


def test_resolve_edge_base_values():
    assert M.resolve_edge_base("acv", 70000) == "acv"
    assert M.resolve_edge_base("bdelta", 118) == "bdelta"          # 显式 bdelta 不看规模(样本级门控在下游)
    assert M.resolve_edge_base("gate15000", 70000) == "bdelta"
    assert M.resolve_edge_base("gate15000", 14999) == "acv"
    assert M.resolve_edge_base("gate15000", 15000) == "bdelta"     # 边界:≥ 阈值
    for bad in ("gate", "gate-5", "gate5k", "bdelt", "", "GATE100"):
        with pytest.raises(ValueError):
            M.resolve_edge_base(bad, 100)


def _tensors(d):
    return {k: getattr(d, k).numpy() for k in ("edge_side", "edge_use", "theta_ref", "ref_mask",
                                               "reach_mask", "side", "v_hint", "target")}


def test_gate_below_threshold_bitwise_acv(harness, with_base):
    """118 < 200:即使基态可用,gate200 也必须与 acv 逐位一致(门没开,一列不动)。"""
    cfg, s, kw = harness
    d_acv = M._pf_data(s, cfg, "group", edge_base="acv", **kw)
    d_gate = M._pf_data(s, cfg, "group", edge_base="gate200", **kw)
    for k, v in _tensors(d_acv).items():
        np.testing.assert_array_equal(v, _tensors(d_gate)[k], err_msg=k)


def test_gate_at_or_above_threshold_bitwise_bdelta(harness, with_base):
    """118 ≥ 100:gate100 与显式 bdelta 逐位一致,且边侧 Δθ 基线确实换了(与 acv 不同)。"""
    cfg, s, kw = harness
    d_bd = M._pf_data(s, cfg, "group", edge_base="bdelta", **kw)
    d_gate = M._pf_data(s, cfg, "group", edge_base="gate100", **kw)
    for k, v in _tensors(d_bd).items():
        np.testing.assert_array_equal(v, _tensors(d_gate)[k], err_msg=k)
    d_acv = M._pf_data(s, cfg, "group", edge_base="acv", **kw)
    assert np.abs(d_gate.edge_side.numpy() - d_acv.edge_side.numpy()).max() > 1e-6   # 基线真的换了


def test_gate_without_base_cache_falls_back_bitwise_acv(harness, monkeypatch):
    """门开但基态缺席(无缓存):样本级门控退回 → 与 acv 逐位一致(E19b 既有语义原样叠加)。"""
    cfg, s, kw = harness
    M._BASE_HINT_CACHE.clear()
    monkeypatch.setattr(M, "BASE_CACHE_DIR", Path("__no_such_dir__"))
    d_acv = M._pf_data(s, cfg, "group", edge_base="acv", **kw)
    d_gate = M._pf_data(s, cfg, "group", edge_base="gate100", **kw)
    for k, v in _tensors(d_acv).items():
        np.testing.assert_array_equal(v, _tensors(d_gate)[k], err_msg=k)
    M._BASE_HINT_CACHE.clear()


# ---------------------------------------------------------------------------
# E21 失配自门控(预注册-E21 §1/§2 G1c)
# ---------------------------------------------------------------------------

def test_auto_without_base_bitwise_acv(harness, monkeypatch):
    """auto 且基态缺席:选择器不动作 → 与 acv 逐位一致。"""
    cfg, s, kw = harness
    M._BASE_HINT_CACHE.clear()
    monkeypatch.setattr(M, "BASE_CACHE_DIR", Path("__no_such_dir__"))
    d_acv = M._pf_data(s, cfg, "group", edge_base="acv", **kw)
    d_auto = M._pf_data(s, cfg, "group", edge_base="auto", **kw)
    for k, v in _tensors(d_acv).items():
        np.testing.assert_array_equal(v, _tensors(d_auto)[k], err_msg=k)
    M._BASE_HINT_CACHE.clear()


def test_auto_dispatch_matches_scored_arm(harness, raw, with_base):
    """auto 的分派 = 选择器所选臂的显式路径,逐位一致;两次调用选择一致(确定性)。"""
    from gridfm.reason.pf_physics import dc_theta_lossy_v
    cfg, s, kw = harness
    dc_arr = dc_theta_lossy_v(raw, tap=raw.branch_tap, shift=raw.branch_shift_rad,
                              shunt_zero_harmless=True)
    sd = M.base_delta_seed(raw)
    assert sd is not None
    pick, scores = M.edge_base_auto_choice(raw, dc_arr, sd, None, raw.branch_tap,
                                           raw.branch_shift_rad, True, return_scores=True)
    pick2 = M.edge_base_auto_choice(raw, dc_arr, sd, None, raw.branch_tap,
                                    raw.branch_shift_rad, True)
    assert pick in ("acv", "bdelta") and pick2 == pick
    assert scores["acv"] > 0 and scores["bdelta"] > 0 and scores["acv"] != scores["bdelta"]
    d_auto = M._pf_data(s, cfg, "group", edge_base="auto", **kw)
    d_pick = M._pf_data(s, cfg, "group", edge_base=pick, **kw)
    for k, v in _tensors(d_pick).items():
        np.testing.assert_array_equal(v, _tensors(d_auto)[k], err_msg=k)


def test_auto_choice_leak_free(raw, with_base):
    """G1c(D15):扰动全部被监督量(PQ v/θ、PV qg/θ、平衡机 pg/qg)→ **选择器本体**的选择与分值逐位不变
    (候选 θ_bΔ 给定;打分只读给定量:eqp_clean 掩掉平衡机行、PQ 的 V 只用提示、θ 用计算型候选)。
    口径注:可用性门 base_delta_seed 的 rel 读平衡机 pg **卡值**(§21.1b 归档口径属给定量,E18 已判),
    不属本轮选择器,不在此测。"""
    from dataclasses import replace
    from gridfm.reason.pf_physics import dc_theta_lossy_v
    from gridfm.train.masking import PF_UNKNOWN_FIELDS
    sd0 = M.base_delta_seed(raw)
    assert sd0 is not None

    def _choice(r):
        dc_arr = dc_theta_lossy_v(r, tap=r.branch_tap, shift=r.branch_shift_rad,
                                  shunt_zero_harmless=True)
        return M.edge_base_auto_choice(r, dc_arr, sd0, None, r.branch_tap, r.branch_shift_rad,
                                       True, return_scores=True)
    p0, s0 = _choice(raw)
    nf = raw.node_feat.copy()
    tc = noff("node_type")
    types = np.argmax(raw.node_feat[:, tc:tc + 3], axis=1)
    for i in range(raw.n_nodes):
        for f in PF_UNKNOWN_FIELDS[int(types[i])]:
            nf[i, noff(f)] += 7.7
    r2 = replace(raw, node_feat=nf)
    r2.branch_tap, r2.branch_shift_rad = raw.branch_tap, raw.branch_shift_rad
    p1, s1 = _choice(r2)
    assert p0 == p1 and s0 == s1
