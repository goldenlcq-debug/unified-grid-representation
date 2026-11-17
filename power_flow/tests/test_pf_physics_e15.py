# -*- coding: utf-8 -*-
"""E15(预注册 §19.2,2026-08-23)两个开关的单测:

1. `shunt_zero_harmless`(修法 A,`_ybus_terms` 一处分支):在运并补记录一律不牵连相邻节点方程;r≠0 的并补按 y=1/(r+jx) 计入
   Ybus,r=0 并补仍走 −1/x(逐位不变)。口径修正依据见 `_ybus_terms` docstring:mp_case300 的 15 条 x=0 并补是纯电导并补(r=714~5000),
   按 §18.1 原文"判无害忽略"真值代入 |ΔP| 过不了 1e-4,故按记录算准。
   夹具:tests/fixtures/mp_case300_mode(真实记录:15 条 x=0 + 2 条 r≠0,x≠0)、mp_case118_mode / 118 golden(无 r≠0 并补 → 逐位不变守卫)、
   118 上人工插入的 r=x=0 空记录(§18.1 单测 1 的字面版本)。
2. `phys-rownorm`(E15b):物理残差损失逐行除以 max(|B_ii|, 1)(`pf_phys_row_scale` / `pf_phys_loss(row_scale=)`):
   真值代入仍 ≈0、各行量级同阶、关掉逐位不变、梯度只经未知量位置。
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gridfm.data.normalize import Normalizer
from gridfm.reason.pf_physics import (
    _ybus_terms, pf_phys_arrays, pf_phys_denorm, pf_phys_loss, pf_phys_residual, pf_phys_row_scale, qv_hint,
)
from gridfm.reason.pf_task import PF_FIELDS
from gridfm.schema.fields import BRANCH_TABLE, BRANCH_TYPES, NODE_TABLE
from gridfm.schema.sample import GridSample
from gridfm.train.masking import PF_UNKNOWN_FIELDS, _pf_typed_mask

FX300 = Path(__file__).parent / "fixtures" / "mp_case300_mode"
FX118 = Path(__file__).parent / "fixtures" / "mp_case118_mode"
TOL = 1e-4
boff, noff = BRANCH_TABLE.offset_of, NODE_TABLE.offset_of
ARR_KEYS = ("index", "g", "b", "eqp_clean", "eqq_clean", "eqp_def", "eqq_def", "eqp_cand", "eqq_cand")


# ---------------- 夹具与工具 ----------------

@pytest.fixture(scope="module")
def raw300():
    return GridSample.load_npz(FX300 / "mp_case300_00000.npz")


@pytest.fixture(scope="module")
def sidecar300():
    z = np.load(FX300 / "family_extra.npz")
    return z["branch_tap"].astype(np.float64), z["branch_shift_rad"].astype(np.float64)


@pytest.fixture(scope="module")
def raw118():
    return GridSample.load_npz(FX118 / "mp_case118_00000.npz")


@pytest.fixture(scope="module")
def sidecar118():
    z = np.load(FX118 / "family_extra.npz")
    return z["branch_tap"].astype(np.float64), z["branch_shift_rad"].astype(np.float64)


def _branch_types(raw):
    return np.asarray(BRANCH_TYPES)[np.argmax(raw.branch_feat[:, boff("branch_type"):boff("branch_type") + 6], axis=1)]


def _node_types(raw):
    tc = noff("node_type")
    return np.argmax(raw.node_feat[:, tc:tc + 3], axis=1)


def _pq(raw):
    return (_node_types(raw) == 0) & ~(raw.node_feat[:, noff("is_null")] > 0.5)


def _nonslack(raw):
    return (_node_types(raw) != 2) & ~(raw.node_feat[:, noff("is_null")] > 0.5)


def _truth_pack(raw):
    """真值代入(完美模型)所需的 target / loss_mask / pred / denorm。"""
    norm = Normalizer().fit([raw])
    s = copy.deepcopy(raw)
    norm.transform(s)
    _mf, tgt, lm = _pf_typed_mask(s.node_feat, s.node_missing)
    t = torch.from_numpy(tgt)
    pred = t[:, [noff(f) for f in PF_FIELDS]].clone()
    return t, torch.from_numpy(lm), pred, torch.from_numpy(pf_phys_denorm(norm))


def _truth_mismatch(raw, arr):
    t, lm, pred, dn = _truth_pack(raw)
    dp, dq = pf_phys_residual(pred, t, lm, torch.from_numpy(arr["index"]),
                              torch.from_numpy(arr["g"]), torch.from_numpy(arr["b"]), dn)
    return dp.numpy(), dq.numpy()


def _shunt_rows(raw):
    bt = _branch_types(raw)
    x = raw.branch_feat[:, boff("x")]
    r = raw.branch_feat[:, boff("r")]
    on = raw.branch_feat[:, boff("mark")] == 1
    sh = (bt == "shunt_comp") & on
    return sh & (np.abs(x) <= 1e-9), sh & (np.abs(x) > 1e-9) & (np.abs(r) > 1e-9), sh


def _assert_arrays_equal(a, b):
    for k in ARR_KEYS:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def _append_shunt(raw, tap, shift, node, r, x):
    """在 node 上追加一条在运并补记录(branch_feat / edge_index / missing / masked / random / sidecar 同步加一行)。"""
    bt_on = np.zeros(6, dtype=np.float32)
    bt_on[BRANCH_TYPES.index("shunt_comp")] = 1.0
    row = np.zeros(raw.branch_feat.shape[1], dtype=raw.branch_feat.dtype)
    row[boff("branch_type"):boff("branch_type") + 6] = bt_on
    row[boff("r")], row[boff("x")], row[boff("b_half")], row[boff("mark")] = r, x, 0.0, 1.0
    raw2 = replace(
        raw,
        branch_feat=np.vstack([raw.branch_feat, row[None]]),
        edge_index=np.concatenate([raw.edge_index, np.array([[node], [node]], dtype=raw.edge_index.dtype)], axis=1),
        branch_missing=np.vstack([raw.branch_missing, raw.branch_missing[:1] * False]),
        branch_masked=np.vstack([raw.branch_masked, raw.branch_masked[:1] * False]),
        branch_random=(None if raw.branch_random is None
                       else np.vstack([raw.branch_random, raw.branch_random[:1]])),
    )
    return raw2, np.append(tap, 1.0), np.append(shift, 0.0)


# ---------------- 修法 A:真实 mp_case300 记录 ----------------

def test_case300_fixture_has_conductance_shunts(raw300):
    """夹具本身要有判别力:15 条 x=0 且 r>0 的在运并补、2 条 r≠0 且 x≠0 的并补,全部挂在 PQ 节点上。"""
    zero_x, rc, _ = _shunt_rows(raw300)
    assert int(zero_x.sum()) == 15 and int(rc.sum()) == 2
    r = raw300.branch_feat[:, boff("r")]
    assert np.all(r[zero_x] >= 700.0)                 # r = 714~5000(纯电导并补 g = 1/r = 2e-4~1.4e-3)
    assert np.all(_pq(raw300)[raw300.edge_index[0][zero_x]])
    # 存档的支路有功 pi 与 V²/r 吻合(它们是真吃有功的电导并补,不是空记录)
    v = raw300.node_feat[raw300.edge_index[0][zero_x], noff("v")].astype(np.float64)
    pi = raw300.branch_feat[zero_x, boff("pi")].astype(np.float64)
    assert np.abs(pi - v * v / r[zero_x]).max() < 1e-5
    assert (v * v / r[zero_x]).min() > TOL          # 忽略它们留下的 ΔP 一定 > 1e-4 → "判无害"口径在这批数据上不成立


def test_case300_off_drops_15_rows_on_full_coverage_and_zero_mismatch(raw300, sidecar300):
    """开关关:15 个挂电导并补的 PQ 节点的 P/Q 方程被作废(现状复现);
    开关开:覆盖率 100%,且新纳入的 15 行 + 2 条 RC 并补行真值代入 |ΔP| / |ΔQ| < 1e-4。"""
    tap, shift = sidecar300
    zero_x, rc, _ = _shunt_rows(raw300)
    zn, rcn = raw300.edge_index[0][zero_x], raw300.edge_index[0][rc]
    off = pf_phys_arrays(raw300, tap=tap, shift=shift)
    on = pf_phys_arrays(raw300, tap=tap, shift=shift, shunt_zero_harmless=True)
    n_p, n_q = int(_nonslack(raw300).sum()), int(_pq(raw300).sum())
    assert int(off["eqp_clean"].sum()) == n_p - 15 and int(off["eqq_clean"].sum()) == n_q - 15
    assert not off["eqp_clean"][zn].any() and not off["eqq_clean"][zn].any()
    assert int(on["eqp_clean"].sum()) == n_p and int(on["eqq_clean"].sum()) == n_q
    dp, dq = _truth_mismatch(raw300, on)
    assert np.abs(dp[zn]).max() < TOL and np.abs(dq[zn]).max() < TOL
    assert np.abs(dp[rcn]).max() < TOL and np.abs(dq[rcn]).max() < TOL
    # 新纳入的行不拖高全图最坏值(其余行的残差来自存档本身的求解精度,两种开关下相同)
    dp0, dq0 = _truth_mismatch(raw300, off)
    assert np.abs(dp[on["eqp_clean"]]).max() <= np.abs(dp0[off["eqp_clean"]]).max()
    assert np.abs(dq[on["eqq_clean"]]).max() <= np.abs(dq0[off["eqq_clean"]]).max()


def test_case300_rows_untouched_by_switch_are_bitwise_identical(raw300, sidecar300):
    """开关只改动挂 r≠0 并补的 17 个节点:其余干净行的真值失配逐位相同;旧口径下 2 条 RC 并补行(忽略 r)失配 > 1e-3,开关开后修好。"""
    tap, shift = sidecar300
    zero_x, rc, _ = _shunt_rows(raw300)
    touched = np.zeros(raw300.n_nodes, dtype=bool)
    touched[raw300.edge_index[0][zero_x | rc]] = True
    off = pf_phys_arrays(raw300, tap=tap, shift=shift)
    on = pf_phys_arrays(raw300, tap=tap, shift=shift, shunt_zero_harmless=True)
    dp0, dq0 = _truth_mismatch(raw300, off)
    dp1, dq1 = _truth_mismatch(raw300, on)
    keep = off["eqp_clean"] & ~touched
    np.testing.assert_array_equal(dp0[keep], dp1[keep])
    np.testing.assert_array_equal(dq0[keep & off["eqq_clean"]], dq1[keep & off["eqq_clean"]])
    rcn = raw300.edge_index[0][rc]
    assert np.abs(dp0[rcn]).max() > 1e-3 and np.abs(dp1[rcn]).max() < TOL


def test_case300_ignoring_conductance_would_fail(raw300, sidecar300):
    """反向守卫(口径修正的依据):开关开但把 r≠0 并补的导纳项置 0(= §18.1 原文的"判无害忽略"),
    15 个节点的真值 |ΔP| 必须 > 1e-4 —— 证明测试有判别力,也证明必须按 1/z 计。"""
    tap, shift = sidecar300
    zero_x, rc, _ = _shunt_rows(raw300)
    on = pf_phys_arrays(raw300, tap=tap, shift=shift, shunt_zero_harmless=True)
    n_rz = int((zero_x | rc).sum())
    T = _ybus_terms(raw300, tap, shift, shunt_zero_harmless=True)
    assert T["n_shunt_rz"] == n_rz
    bad = dict(on)
    g, b = on["g"].copy(), on["b"].copy()
    g[-n_rz:], b[-n_rz:] = 0.0, 0.0                      # r≠0 并补项在 COO 末尾(与 _ybus_terms 拼接顺序一致)
    bad["g"], bad["b"] = g, b
    dp, _dq = _truth_mismatch(raw300, bad)
    zn = raw300.edge_index[0][zero_x]
    assert np.abs(dp[zn]).min() > TOL


def test_case300_qv_hint_on_covers_all_pq_rows_and_is_leak_free(raw300, sidecar300):
    """Q–V 提示与物理损失同源:开关开后 15 个节点也进提示方程(valid 从 n_pq−15 → n_pq),提示精度不劣,且仍无泄漏。"""
    tap, shift = sidecar300
    off = qv_hint(raw300, tap=tap, shift=shift)
    on = qv_hint(raw300, tap=tap, shift=shift, shunt_zero_harmless=True)
    n_q = int(_pq(raw300).sum())
    assert int(off[:, 1].sum()) == n_q - 15 and int(on[:, 1].sum()) == n_q
    assert not on[:, 2].any()
    vc = noff("v")
    m_on, m_off = on[:, 1] > 0.5, off[:, 1] > 0.5
    err_on = np.abs(on[m_on, 0] - raw300.node_feat[m_on, vc]).mean()
    err_off = np.abs(off[m_off, 0] - raw300.node_feat[m_off, vc]).mean()
    assert err_on <= err_off
    nf = raw300.node_feat.copy()
    types = _node_types(raw300)
    for i in range(raw300.n_nodes):
        for f in PF_UNKNOWN_FIELDS[int(types[i])]:
            nf[i, noff(f)] += 7.7
    pert = qv_hint(replace(raw300, node_feat=nf), tap=tap, shift=shift, shunt_zero_harmless=True)
    np.testing.assert_array_equal(on, pert)


# ---------------- 修法 A:其余家族逐位不变 ----------------

def test_switch_is_bitwise_noop_on_families_without_r_shunts(raw118, sidecar118, case_dir):
    """mp_case118(有 sidecar / 无 sidecar)与 118 golden 都没有 r≠0 的并补:开关开 / 关的物理常量与 Q–V 提示逐位相同。"""
    from gridfm.psasp_io.case import PsaspCase
    from gridfm.psasp_io.catalog import Catalog
    from gridfm.schema.assemble import build_sample
    tap, shift = sidecar118
    assert not _shunt_rows(raw118)[1].any() and not _shunt_rows(raw118)[0].any()
    for kw in (dict(tap=tap, shift=shift), dict()):
        _assert_arrays_equal(pf_phys_arrays(raw118, **kw), pf_phys_arrays(raw118, shunt_zero_harmless=True, **kw))
        np.testing.assert_array_equal(qv_hint(raw118, **kw), qv_hint(raw118, shunt_zero_harmless=True, **kw))
    c = PsaspCase(case_dir)
    g = build_sample(c, Catalog(c), sample_id="s0", family_id="t")
    _assert_arrays_equal(pf_phys_arrays(g), pf_phys_arrays(g, shunt_zero_harmless=True))
    np.testing.assert_array_equal(qv_hint(g), qv_hint(g, shunt_zero_harmless=True))


def test_synthetic_empty_record_118(raw118, sidecar118):
    """§18.1 单测 1 的字面版本:在 118 夹具一个 PQ 节点上插入 r=x=0 的在运并补空记录。
    开关关 → 该节点 P/Q 方程被作废(复现现状);开关开 → 与未插入时逐位相同(空记录被忽略,真值失配 < 1e-4 自然保持)。"""
    tap, shift = sidecar118
    base = pf_phys_arrays(raw118, tap=tap, shift=shift)
    node = int(np.where(_pq(raw118) & base["eqq_clean"])[0][0])
    raw2, tap2, shift2 = _append_shunt(raw118, tap, shift, node, r=0.0, x=0.0)
    off = pf_phys_arrays(raw2, tap=tap2, shift=shift2)
    assert not off["eqp_clean"][node] and not off["eqq_clean"][node]
    assert int(off["eqq_clean"].sum()) == int(base["eqq_clean"].sum()) - 1
    on = pf_phys_arrays(raw2, tap=tap2, shift=shift2, shunt_zero_harmless=True)
    _assert_arrays_equal(on, base)
    dp, dq = _truth_mismatch(raw2, on)
    assert abs(dp[node]) < TOL and abs(dq[node]) < TOL


def test_synthetic_conductance_record_118(raw118, sidecar118):
    """在 118 夹具的 PQ 节点上插入 r=1250、x=0 的电导并补,并把该节点有功负荷减去 V²/r(让存档解仍满足方程):
    开关开 → 该行纳入且真值失配 < 1e-4;若把它当空记录忽略(导纳项置 0)→ 该行失配 = V²/r ≈ 7e-4 > 1e-4。"""
    tap, shift = sidecar118
    base = pf_phys_arrays(raw118, tap=tap, shift=shift)
    node = int(np.where(_pq(raw118) & base["eqq_clean"])[0][0])
    r = 1250.0
    v = float(raw118.node_feat[node, noff("v")])
    raw2, tap2, shift2 = _append_shunt(raw118, tap, shift, node, r=r, x=0.0)
    nf = raw2.node_feat.copy()
    nf[node, noff("pl")] -= v * v / r
    raw2 = replace(raw2, node_feat=nf)
    on = pf_phys_arrays(raw2, tap=tap2, shift=shift2, shunt_zero_harmless=True)
    assert on["eqp_clean"][node] and on["eqq_clean"][node]
    assert on["index"][0, -1] == node and on["index"][1, -1] == node
    assert abs(on["g"][-1] - 1.0 / r) < 1e-9 and on["b"][-1] == 0.0      # g 以 float32 存(1/1250 的相对误差 ~3e-8)
    dp, dq = _truth_mismatch(raw2, on)
    assert abs(dp[node]) < TOL and abs(dq[node]) < TOL
    bad = dict(on)
    bad["g"] = np.concatenate([on["g"][:-1], [0.0]]).astype(np.float32)
    dp_bad, _ = _truth_mismatch(raw2, bad)
    assert abs(dp_bad[node]) > 5e-4


# ---------------- E15b:物理残差损失逐行归一化 ----------------

def test_row_scale_matches_dense_diagonal(raw118, sidecar118):
    tap, shift = sidecar118
    arr = pf_phys_arrays(raw118, tap=tap, shift=shift)
    n = raw118.n_nodes
    dense = np.zeros((n, n))
    np.add.at(dense, (arr["index"][0], arr["index"][1]), arr["b"].astype(np.float64))
    rs = pf_phys_row_scale(torch.from_numpy(arr["index"]), torch.from_numpy(arr["b"]), n).numpy()
    np.testing.assert_allclose(rs, np.maximum(np.abs(np.diag(dense)), 1.0), rtol=1e-5, atol=1e-6)
    assert rs.max() / rs.min() > 20         # 夹具上各行 |B_ii| 跨度要足够大,否则下面的"同阶"测试没有判别力


def test_rownorm_truth_still_zero_and_off_is_bitwise(raw118, sidecar118):
    """真值代入:归一化损失 ≤ 原损失(尺度 ≥ 1)且 < 1e-4;row_scale=None 与全 1 尺度都与旧调用逐位相同。"""
    tap, shift = sidecar118
    arr = pf_phys_arrays(raw118, tap=tap, shift=shift)
    t, lm, pred, dn = _truth_pack(raw118)
    idx, g, b = torch.from_numpy(arr["index"]), torch.from_numpy(arr["g"]), torch.from_numpy(arr["b"])
    masks = {k: torch.from_numpy(arr[k]) for k in ("eqp_clean", "eqq_clean", "eqp_def", "eqq_def")}
    dp, dq = pf_phys_residual(pred, t, lm, idx, g, b, dn)
    rs = pf_phys_row_scale(idx, b, raw118.n_nodes)
    l_old, m_old = pf_phys_loss(dp, dq, masks, w_clean=1.0)
    l_none, _ = pf_phys_loss(dp, dq, masks, w_clean=1.0, row_scale=None)
    l_ones, _ = pf_phys_loss(dp, dq, masks, w_clean=1.0, row_scale=torch.ones(raw118.n_nodes))
    l_rs, m_rs = pf_phys_loss(dp, dq, masks, w_clean=1.0, row_scale=rs)
    assert float(l_none) == float(l_old) and float(l_ones) == float(l_old)
    assert float(l_rs) <= float(l_old) and float(l_rs) < TOL
    assert m_rs["phys_clean"] <= m_old["phys_clean"]


def test_rownorm_equalizes_rows(raw118, sidecar118):
    """同样的 δθ=1e-3 扰动分别加在 |B_ii| 最小 / 中位 / 最大的干净 PQ 行上:
    原始 |ΔP_i| 随 |B_ii| 跨数十倍(高导纳行主导损失);逐行归一化后全部 ≈ δ·V²(同阶,彼此相差 < 1.5 倍)。"""
    tap, shift = sidecar118
    arr = pf_phys_arrays(raw118, tap=tap, shift=shift)
    t, lm, pred0, dn = _truth_pack(raw118)
    idx, g, b = torch.from_numpy(arr["index"]), torch.from_numpy(arr["g"]), torch.from_numpy(arr["b"])
    rs = pf_phys_row_scale(idx, b, raw118.n_nodes).numpy()
    rows = np.where(arr["eqq_clean"] & lm.numpy()[:, noff("theta")])[0]
    rows = rows[np.argsort(rs[rows])]
    picks = [rows[0], rows[len(rows) // 2], rows[-1]]
    delta, kth = 1e-3, PF_FIELDS.index("theta")
    raw_abs, norm_abs = [], []
    for node in picks:
        pred = pred0.clone()
        pred[node, kth] += delta
        dp, _ = pf_phys_residual(pred, t, lm, idx, g, b, dn)
        raw_abs.append(abs(float(dp[node])))
        norm_abs.append(abs(float(dp[node])) / rs[node])
    assert max(raw_abs) / min(raw_abs) > 10
    assert max(norm_abs) / min(norm_abs) < 1.5
    for x in norm_abs:
        assert 0.5 * delta < x < 1.5 * delta


def test_rownorm_grad_only_through_unknown_positions(raw118, sidecar118):
    tap, shift = sidecar118
    arr = pf_phys_arrays(raw118, tap=tap, shift=shift)
    t, lm, pred, dn = _truth_pack(raw118)
    pred = pred.requires_grad_(True)
    idx, g, b = torch.from_numpy(arr["index"]), torch.from_numpy(arr["g"]), torch.from_numpy(arr["b"])
    masks = {k: torch.from_numpy(arr[k]) for k in ("eqp_clean", "eqq_clean", "eqp_def", "eqq_def")}
    dp, dq = pf_phys_residual(pred, t, lm, idx, g, b, dn)
    loss, _ = pf_phys_loss(dp, dq, masks, w_clean=1.0, row_scale=pf_phys_row_scale(idx, b, raw118.n_nodes))
    loss.backward()
    grad = pred.grad
    known = torch.ones_like(grad, dtype=torch.bool)
    unknown_any = False
    for k, f in enumerate(PF_FIELDS):
        m = lm[:, noff(f)]
        known[m, k] = False
        unknown_any = unknown_any or bool(grad[m, k].abs().sum() > 0)
    assert unknown_any and grad[known].abs().max() == 0.0
