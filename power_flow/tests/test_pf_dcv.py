# -*- coding: utf-8 -*-
"""E16(预注册 §20 + §20.1b 修订,用户 2026-08-23 批准)单测:`dc_theta_lossy_v` 与 `--dc-encoding lossyv` 双轨。

要点:θ_acv 只作边侧 Δθ 基线与 Q–V 提示的 θ;侧通道节点级 θ 仍是带网损 DC;
Q–V 提示整体无效的样本(老库形态)逐位退回带网损 DC;D15 无泄漏。
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gridfm.data.normalize import Normalizer
from gridfm.reason.pf_physics import dc_theta_lossy, dc_theta_lossy_v, qv_hint
from gridfm.reason.pf_theta_edge import DTH_DC_COL, use_edges
from gridfm.schema.fields import BRANCH_TABLE, BRANCH_TYPES, NODE_TABLE
from gridfm.schema.sample import GridSample
from gridfm.train.masking import PF_UNKNOWN_FIELDS, _pf_typed_mask
from gridfm.train.pretrain import TrainConfig

FX118 = Path(__file__).parent / "fixtures" / "mp_case118_mode"
FX300 = Path(__file__).parent / "fixtures" / "mp_case300_mode"
boff, noff = BRANCH_TABLE.offset_of, NODE_TABLE.offset_of
TH, VC = noff("theta"), noff("v")


@pytest.fixture(scope="module")
def raw118():
    r = GridSample.load_npz(FX118 / "mp_case118_00000.npz")
    z = np.load(FX118 / "family_extra.npz")
    r.branch_tap, r.branch_shift_rad = z["branch_tap"], z["branch_shift_rad"]
    return r


@pytest.fixture(scope="module")
def raw300():
    r = GridSample.load_npz(FX300 / "mp_case300_00000.npz")
    z = np.load(FX300 / "family_extra.npz")
    r.branch_tap, r.branch_shift_rad = z["branch_tap"], z["branch_shift_rad"]
    return r


def _edge_mae(raw, arr):
    """θ 数组的逐边 Δθ 对存档真值的 MAE(use 支路)。"""
    i, j = raw.edge_index[0].astype(np.int64), raw.edge_index[1].astype(np.int64)
    ok = use_edges(raw) & (arr[i, 1] > 0.5) & (arr[j, 1] > 0.5)
    tt = raw.node_feat[:, TH].astype(np.float64)
    return float(np.abs((arr[j[ok], 0] - arr[i[ok], 0]).astype(np.float64) - (tt[j[ok]] - tt[i[ok]])).mean())


def test_dcv_edge_and_vhint_beat_dc(raw118, raw300):
    """判别力:118 / 300 夹具上,acv 的逐边 Δθ 与重算的 v 提示都明显优于带网损 DC(离线闸的单测版)。"""
    for raw in (raw118, raw300):
        d0 = dc_theta_lossy(raw)
        d1 = dc_theta_lossy_v(raw, tap=raw.branch_tap, shift=raw.branch_shift_rad)
        assert _edge_mae(raw, d1) < 0.7 * _edge_mae(raw, d0)
        vh0 = qv_hint(raw, tap=raw.branch_tap, shift=raw.branch_shift_rad, dc=d0)
        vh1 = qv_hint(raw, tap=raw.branch_tap, shift=raw.branch_shift_rad, dc=d1)
        _mf, _tg, lm = _pf_typed_mask(raw.node_feat, raw.node_missing)
        m = lm[:, VC] & (vh0[:, 1] > 0.5)
        vt = raw.node_feat[:, VC].astype(np.float64)
        assert np.abs(vh1[m, 0] - vt[m]).mean() < 0.75 * np.abs(vh0[m, 0] - vt[m]).mean()
        # valid / fallback 位继承;平衡机处 = 给定角
        np.testing.assert_array_equal(d1[:, 1], d0[:, 1])
        tc = noff("node_type")
        sl = np.argmax(raw.node_feat[:, tc:tc + 3], axis=1) == 2
        np.testing.assert_allclose(d1[sl, 0], raw.node_feat[sl, TH], atol=1e-6)


def test_dcv_leak_free(raw118):
    """D15:扰动全部被监督量(PQ 的 v/θ、PV 的 qg/θ、平衡机 pg/qg)不得改变 θ_acv。"""
    base = dc_theta_lossy_v(raw118, tap=raw118.branch_tap, shift=raw118.branch_shift_rad)
    nf = raw118.node_feat.copy()
    tc = noff("node_type")
    types = np.argmax(raw118.node_feat[:, tc:tc + 3], axis=1)
    n_pert = 0
    for i in range(raw118.n_nodes):
        for f in PF_UNKNOWN_FIELDS[int(types[i])]:
            nf[i, noff(f)] += 7.7
            n_pert += 1
    assert n_pert > 0
    pert = replace(raw118, node_feat=nf)
    pert.branch_tap, pert.branch_shift_rad = raw118.branch_tap, raw118.branch_shift_rad
    np.testing.assert_array_equal(base, dc_theta_lossy_v(pert, tap=pert.branch_tap, shift=pert.branch_shift_rad))


def test_dcv_hint_invalid_sample_bitwise_dc(raw118):
    """Q–V 提示整体无效的样本(老库形态:无 sidecar + 在运变压器,E10b 规则)→ 逐位退回带网损 DC。"""
    vh = qv_hint(raw118)                                  # 不给 tap → E10b:整样本提示作废
    assert int(vh[:, 1].sum()) == 0
    out = dc_theta_lossy_v(raw118)                        # 同样不给 tap
    np.testing.assert_array_equal(out, dc_theta_lossy(raw118))


def test_pf_data_lossyv_dual_track(raw118):
    """`--dc-encoding lossyv` 双轨:侧通道 dc 三列 = 带网损 DC(与 lossy 逐位相同);
    边侧 Δθ 基线与 v 提示 = θ_acv(与 lossy 不同且 v 提示更准);theta_ref / reach 不变。"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import pf_reasoner as M
    norm = Normalizer().fit([raw118])
    s = copy.deepcopy(raw118)
    norm.transform(s)
    cfg = TrainConfig()
    vstat = (float(norm.node_mean[VC]), float(norm.node_std[VC]))
    dl = M._pf_data(s, cfg, view="group", raw=raw118, dc="lossy", phys=True, edge=True, vhint=True, vstat=vstat)
    dv = M._pf_data(s, cfg, view="group", raw=raw118, dc="lossyv", phys=True, edge=True, vhint=True, vstat=vstat)
    # 侧通道:dc 的 3 列(位于 v-hint 3 列之前)逐位相同
    np.testing.assert_array_equal(dl.side[:, :-3].numpy(), dv.side[:, :-3].numpy())
    # v 提示列(末 3 列)不同,且 lossyv 的 v 提示更准
    assert not np.array_equal(dl.side[:, -3:].numpy(), dv.side[:, -3:].numpy())
    _mf, _tg, lm = _pf_typed_mask(raw118.node_feat, raw118.node_missing)
    m = lm[:, VC]
    vt = ((raw118.node_feat[:, VC] - vstat[0]) / vstat[1]).astype(np.float32)   # z 化真值
    e_l = np.abs(dl.v_hint.numpy()[m] - vt[m]).mean()
    e_v = np.abs(dv.v_hint.numpy()[m] - vt[m]).mean()
    assert e_v < 0.75 * e_l
    # 边侧 Δθ 基线不同(acv);use / ok / 静态列相同;theta_ref / ref_mask / reach 相同
    el, ev = dl.edge_side.numpy(), dv.edge_side.numpy()
    assert not np.array_equal(el[:, DTH_DC_COL], ev[:, DTH_DC_COL])
    keep = [k for k in range(el.shape[1]) if k != DTH_DC_COL and k != el.shape[1] - 1]   # 除 Δθ 与 p_dc(由 Δθ 派生)
    np.testing.assert_array_equal(el[:, keep], ev[:, keep])
    np.testing.assert_array_equal(dl.theta_ref.numpy(), dv.theta_ref.numpy())
    np.testing.assert_array_equal(dl.ref_mask.numpy(), dv.ref_mask.numpy())
    np.testing.assert_array_equal(dl.reach_mask.numpy(), dv.reach_mask.numpy())
    # 物理常量与 lossy 逐位相同(与 dc 编码无关)
    np.testing.assert_array_equal(dl.phys_b.numpy(), dv.phys_b.numpy())


def test_lossy_path_bitwise_unchanged(raw118):
    """旧 `lossy` 路径逐位不变(dc_theta_lossy_v 只在 lossyv 分支被调用)。"""
    d_now = dc_theta_lossy(raw118)
    ref = np.load(FX118 / "qv_hint_ref_tap.npz")["qv_hint"]   # 既有参照:qv_hint(tap) 逐位
    vh = qv_hint(raw118, tap=raw118.branch_tap, shift=raw118.branch_shift_rad)
    np.testing.assert_array_equal(vh, ref)
    assert d_now.shape == (raw118.n_nodes, 3)
