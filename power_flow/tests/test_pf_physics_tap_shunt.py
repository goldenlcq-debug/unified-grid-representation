# -*- coding: utf-8 -*-
"""物理残差损失的变比 / 并补口径(2026-08-22,预注册 §14.1)+ 电压物理提示 qv_hint(§14.2)单测。

夹具 tests/fixtures/mp_case118_mode:IEEE-118 衍生(MATPOWER 底图 → convert → PSASP 解)的一个方式
+ 家族 sidecar(9 台变比≠1、18 台在运并补)。118 golden 夹具的并补全部退运,抓不到并补口径,故另立夹具。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import copy
import numpy as np
import pytest
import torch

from gridfm.data.normalize import Normalizer
from gridfm.reason.pf_physics import has_in_service_xfmr, pf_phys_arrays, pf_phys_denorm, pf_phys_residual, qv_hint
from gridfm.schema.fields import BRANCH_TABLE, BRANCH_TYPES, NODE_TABLE
from gridfm.schema.sample import GridSample
from gridfm.train.masking import PF_UNKNOWN_FIELDS, _pf_typed_mask

FX = Path(__file__).parent / "fixtures" / "mp_case118_mode"
TOL = 1e-4
VC = NODE_TABLE.offset_of("v")


@pytest.fixture(scope="module")
def raw118():
    return GridSample.load_npz(FX / "mp_case118_00000.npz")


@pytest.fixture(scope="module")
def sidecar():
    z = np.load(FX / "family_extra.npz")
    return z["branch_tap"].astype(np.float64), z["branch_shift_rad"].astype(np.float64)


def _truth_mismatch(raw, arr):
    norm = Normalizer().fit([raw])
    s = copy.deepcopy(raw)
    norm.transform(s)
    _mf, tgt, lm = _pf_typed_mask(s.node_feat, s.node_missing)
    t = torch.from_numpy(tgt)
    pred = t[:, [NODE_TABLE.offset_of(f) for f in ("pg", "qg", "v", "theta")]].clone()
    dn = torch.from_numpy(pf_phys_denorm(norm))
    dp, dq = pf_phys_residual(pred, t, torch.from_numpy(lm), torch.from_numpy(arr["index"]),
                              torch.from_numpy(arr["g"]), torch.from_numpy(arr["b"]), dn)
    return dp.numpy(), dq.numpy()


def _pq_nodes(raw):
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    null = raw.node_feat[:, NODE_TABLE.offset_of("is_null")] > 0.5
    return (types == 0) & ~null


def _shunt_nodes(raw):
    boff = BRANCH_TABLE.offset_of
    bt = np.asarray(BRANCH_TYPES)[np.argmax(raw.branch_feat[:, boff("branch_type"):boff("branch_type") + 6], axis=1)]
    sh = (raw.branch_feat[:, boff("mark")] == 1) & (bt == "shunt_comp")
    has = np.zeros(raw.n_nodes, dtype=bool)
    has[raw.edge_index[0][sh]] = True
    return has, int(sh.sum())


def test_fixture_has_in_service_shunts_and_taps(raw118, sidecar):
    """夹具本身要有判别力:在运并补 > 0、变比≠1 > 0(否则本文件的测试全是空转)。"""
    has, n_sh = _shunt_nodes(raw118)
    tap, _ = sidecar
    assert n_sh >= 10 and has.sum() >= 10
    assert int((np.abs(tap - 1) > 1e-6).sum()) >= 5


def test_tap_gives_full_coverage_and_zero_truth_mismatch(raw118, sidecar):
    """带 sidecar 变比:全部 PQ 节点都有 Q 方程(覆盖 100%),真值代入失配 < 1e-4 —— 含变压器相邻与挂并补的节点。"""
    tap, shift = sidecar
    arr = pf_phys_arrays(raw118, tap=tap, shift=shift)
    pq = _pq_nodes(raw118)
    assert int(arr["eqq_clean"].sum()) == int(pq.sum()), "带变比后 Q 方程应全覆盖 PQ 节点"
    dp, dq = _truth_mismatch(raw118, arr)
    assert np.abs(dp[arr["eqp_clean"]]).max() < TOL
    assert np.abs(dq[arr["eqq_clean"]]).max() < TOL
    has, _ = _shunt_nodes(raw118)
    assert (arr["eqq_clean"] & has).sum() >= 5            # 挂并补的 PQ 行确实在方程里且失配≈0


def test_without_tap_keeps_conservative_rows_but_fixes_shunts(raw118):
    """无 sidecar:变压器相邻行仍作废(旧口径保持),但并补行已按 −1/x 建,失配 < 1e-4。"""
    arr = pf_phys_arrays(raw118)
    pq = _pq_nodes(raw118)
    assert int(arr["eqq_clean"].sum()) < int(pq.sum())      # 仍有行被保守作废
    dp, dq = _truth_mismatch(raw118, arr)
    assert np.abs(dp[arr["eqp_clean"]]).max() < TOL
    assert np.abs(dq[arr["eqq_clean"]]).max() < TOL


def test_old_shunt_convention_would_fail(raw118, sidecar):
    """反向守卫:若把并补当 0(旧 bug),挂并补节点的真值失配必须明显 > 1e-4 —— 证明测试有判别力。"""
    tap, shift = sidecar
    arr = pf_phys_arrays(raw118, tap=tap, shift=shift)
    boff = BRANCH_TABLE.offset_of
    bt = np.asarray(BRANCH_TYPES)[np.argmax(raw118.branch_feat[:, boff("branch_type"):boff("branch_type") + 6], axis=1)]
    sh_rows = np.where((raw118.branch_feat[:, boff("mark")] == 1) & (bt == "shunt_comp"))[0]
    # 把并补对应的 COO 项置 0(等价于旧实现的 2·b_half = 0)
    idx = arr["index"]
    is_self = (idx[0] == idx[1]) & np.isin(idx[0], raw118.edge_index[0][sh_rows])
    b = arr["b"].copy()
    # 只去掉并补贡献:同一自环位置还可能叠加了线路的 b_half,故用 −1/x 精确扣减
    x = raw118.branch_feat[sh_rows, boff("x")].astype(np.float64)
    bad = dict(arr)
    bb = b.astype(np.float64)
    # 并补项在 COO 末尾(与 _ybus_terms 的拼接顺序一致)
    bb[-len(sh_rows):] = 0.0
    bad["b"] = bb.astype(np.float32)
    _dp, dq = _truth_mismatch(raw118, bad)
    has, _ = _shunt_nodes(raw118)
    assert np.abs(dq[arr["eqq_clean"] & has]).max() > 0.01


def test_qv_hint_accuracy_and_flags(raw118, sidecar):
    tap, shift = sidecar
    vh = qv_hint(raw118, tap=tap, shift=shift)
    _mf, _tg, lm = _pf_typed_mask(raw118.node_feat, raw118.node_missing)
    m = lm[:, VC]
    assert vh.shape == (raw118.n_nodes, 3)
    assert vh[m, 1].all() and not vh[m, 2].any()
    err = np.abs(vh[m, 0] - raw118.node_feat[m, VC])
    flat = np.abs(1.0 - raw118.node_feat[m, VC])
    assert err.mean() < 0.002 and err.mean() < 0.2 * flat.mean(), (err.mean(), flat.mean())


def test_qv_hint_leak_free(raw118, sidecar):
    """D15:扰动全部被监督量(PQ 的 v/θ、PV 的 qg/θ、平衡机 pg/qg)不得改变提示。"""
    tap, shift = sidecar
    base = qv_hint(raw118, tap=tap, shift=shift)
    nf = raw118.node_feat.copy()
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(raw118.node_feat[:, tcol:tcol + 3], axis=1)
    n_pert = 0
    for i in range(raw118.n_nodes):
        for f in PF_UNKNOWN_FIELDS[int(types[i])]:
            nf[i, NODE_TABLE.offset_of(f)] += 7.7
            n_pert += 1
    assert n_pert > 0
    pert = qv_hint(replace(raw118, node_feat=nf), tap=tap, shift=shift)
    np.testing.assert_array_equal(base, pert)


def test_qv_hint_uses_given_reactive_load(raw118, sidecar):
    """扰动某 PQ 节点的给定无功负荷 ql 必须改变提示(证明它真在解方程)。"""
    tap, shift = sidecar
    base = qv_hint(raw118, tap=tap, shift=shift)
    pq = np.where(_pq_nodes(raw118))[0]
    nf = raw118.node_feat.copy()
    nf[pq[0], NODE_TABLE.offset_of("ql")] += 0.5
    pert = qv_hint(replace(raw118, node_feat=nf), tap=tap, shift=shift)
    assert np.abs(pert[:, 0] - base[:, 0]).max() > 1e-4


def test_qv_hint_no_tap_with_xfmr_is_fully_invalid_e10b(raw118):
    """E10b(预注册 §15.1):无 sidecar 且含在运变压器 → 整样本提示作废:valid 全 0、v̂ 全 1、fallback 全 1。
    (此前在残缺子图上解出 54/64 个"valid"行,IEEE-39 方式库上反而把 v 从 0.018 拉坏到 0.029。)"""
    assert has_in_service_xfmr(raw118)
    vh = qv_hint(raw118)
    assert vh.shape == (raw118.n_nodes, 3)
    assert int(vh[:, 1].sum()) == 0
    assert np.all(vh[:, 0] == 1.0) and np.all(vh[:, 2] == 1.0)


def test_qv_hint_no_tap_without_in_service_xfmr_still_solves(raw118):
    """规则只咬"在运变压器":把变压器全部置退运(mark=0)后,无 tap 也照常解出一个有效子集。"""
    boff = BRANCH_TABLE.offset_of
    bt = np.asarray(BRANCH_TYPES)[np.argmax(raw118.branch_feat[:, boff("branch_type"):boff("branch_type") + 6], axis=1)]
    bf = raw118.branch_feat.copy()
    bf[np.isin(bt, ["xfmr_2w", "xfmr_3w", "phase_shifter"]), boff("mark")] = 0.0
    raw2 = replace(raw118, branch_feat=bf)
    assert not has_in_service_xfmr(raw2)
    vh = qv_hint(raw2)
    assert int(vh[:, 1].sum()) > 0
    assert np.all(vh[vh[:, 1] < 0.5, 0] == 1.0)


def test_qv_hint_golden_no_tap_all_invalid(case_dir):
    """118 golden 夹具(9 台在运变压器、无 sidecar)→ E10b 规则生效,valid.sum()==0。"""
    from gridfm.psasp_io.case import PsaspCase
    from gridfm.psasp_io.catalog import Catalog
    from gridfm.schema.assemble import build_sample
    c = PsaspCase(case_dir)
    g = build_sample(c, Catalog(c), sample_id="s0", family_id="t")
    assert has_in_service_xfmr(g)
    vh = qv_hint(g)
    assert int(vh[:, 1].sum()) == 0 and np.all(vh[:, 2] == 1.0) and np.all(vh[:, 0] == 1.0)


def test_qv_hint_with_tap_bitwise_unchanged_by_e10b(raw118, sidecar):
    """有 sidecar 的样本不受 E10b 影响:与补丁前存下的参考输出逐位相同(tests/fixtures/mp_case118_mode/qv_hint_ref_tap.npz)。"""
    tap, shift = sidecar
    ref = np.load(FX / "qv_hint_ref_tap.npz")["qv_hint"]
    vh = qv_hint(raw118, tap=tap, shift=shift)
    np.testing.assert_array_equal(vh, ref)
    assert int(vh[:, 1].sum()) == 64 and int(vh[:, 2].sum()) == 0
