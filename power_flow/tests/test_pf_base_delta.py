# -*- coding: utf-8 -*-
"""E18(预注册 §21)单测:`dc_theta_base_delta` 与 `dc_theta_lossy_v(seed=)` 与 `_pf_data(dc="lossybv")`。

夹具:mp_case118_mode(有 sidecar)。"基态"用**独立构造**的方式:把注入整体缩放 0.97 后用带网损 DC 生成一套
自洽的 (θ_b, pinj_b)——单测只验机制(参考系、退回、无泄漏、逐位),不验提示精度(精度由离线闸对真基态验)。
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gridfm.reason.pf_physics import dc_theta_base_delta, dc_theta_lossy, dc_theta_lossy_v
from gridfm.schema.fields import NODE_TABLE
from gridfm.schema.sample import GridSample
from gridfm.train.masking import PF_UNKNOWN_FIELDS, _pf_typed_mask

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pf_reasoner as M  # noqa: E402

FX = Path(__file__).parent / "fixtures" / "mp_case118_mode"
noff = NODE_TABLE.offset_of
TH = noff("theta")


@pytest.fixture(scope="module")
def raw():
    r = GridSample.load_npz(FX / "mp_case118_00000.npz")
    z = np.load(FX / "family_extra.npz")
    r.branch_tap, r.branch_shift_rad = z["branch_tap"], z["branch_shift_rad"]
    return r


@pytest.fixture(scope="module")
def base(raw):
    """独立"基态":注入 ×0.97,θ_b = 该注入下的带网损 DC(平衡机给定角不动)。只用给定量构造。"""
    nf = raw.node_feat.copy()
    for f in ("pg", "pl"):
        nf[:, noff(f)] *= 0.97
    rb = replace(raw, node_feat=nf)
    d = dc_theta_lossy(rb)
    th_b = d[:, 0].astype(np.float64)
    pinj_b = (nf[:, noff("pg")] - nf[:, noff("pl")]).astype(np.float64)
    return th_b, pinj_b


def test_base_delta_reduces_to_dc_when_base_is_dc_of_same_injection(raw):
    """判别力 + 一致性:若基态 = 本方式自己的带网损 DC(ΔP=0),θ_bΔ 应与该 DC 逐位同值(Δθ=0、无损耗增量)。"""
    d = dc_theta_lossy(raw)
    th_b = d[:, 0].astype(np.float64)
    pinj = (raw.node_feat[:, noff("pg")] - raw.node_feat[:, noff("pl")]).astype(np.float64)
    out = dc_theta_base_delta(raw, th_b, pinj)
    m = (out[:, 1] > 0.5) & (d[:, 1] > 0.5)
    assert m.sum() > 100
    np.testing.assert_allclose(out[m, 0], d[m, 0], atol=2e-7)   # float32 存储差
    assert int(out[:, 1].sum()) == int(d[:, 1].sum())


def test_base_delta_beats_raw_base_and_reference_shift(raw, base):
    """Δ 修正确实在干活:θ_bΔ 对真 θ 优于"基态原样";平衡机参考系差被 Dirichlet 吸收(整体平移不敏感)。"""
    th_b, pinj_b = base
    _mf, _tg, lm = _pf_typed_mask(raw.node_feat, raw.node_missing)
    m = lm[:, TH]
    tt = raw.node_feat[:, TH].astype(np.float64)
    out = dc_theta_base_delta(raw, th_b, pinj_b)
    ok = m & (out[:, 1] > 0.5)
    e_hint = np.abs(out[ok, 0] - tt[ok]).mean()
    e_base = np.abs(th_b[ok] - tt[ok]).mean()
    assert e_hint < e_base * 0.9
    # 参考系差:基态整体平移 0.3 rad 后输出不变(Δθ_s = θ_s^方式 − θ_s^基态 把平移吃掉)
    out2 = dc_theta_base_delta(raw, th_b + 0.3, pinj_b)
    np.testing.assert_allclose(out2[ok, 0], out[ok, 0], atol=1e-5)


def test_base_delta_leak_free(raw, base):
    """D15:扰动本方式全部被监督量(PQ v/θ、PV qg/θ、平衡机 pg/qg)不得改变 θ_bΔ。"""
    th_b, pinj_b = base
    ref = dc_theta_base_delta(raw, th_b, pinj_b)
    nf = raw.node_feat.copy()
    tc = noff("node_type")
    types = np.argmax(raw.node_feat[:, tc:tc + 3], axis=1)
    for i in range(raw.n_nodes):
        for f in PF_UNKNOWN_FIELDS[int(types[i])]:
            nf[i, noff(f)] += 7.7
    pert = dc_theta_base_delta(replace(raw, node_feat=nf), th_b, pinj_b)
    np.testing.assert_array_equal(ref, pert)


def test_base_delta_shape_mismatch_returns_invalid(raw, base):
    th_b, pinj_b = base
    out = dc_theta_base_delta(raw, th_b[:-1], pinj_b[:-1])
    assert out.shape == (raw.n_nodes, 3) and not (out[:, 1] > 0.5).any()


def test_lossy_v_seed_none_bitwise_and_seeded_uses_seed(raw, base):
    """seed=None 与 E16 逐位一致;seed 给定时逐边修正的基点确实换了(输出不同且有限)。"""
    a0 = dc_theta_lossy_v(raw, tap=raw.branch_tap, shift=raw.branch_shift_rad)
    a1 = dc_theta_lossy_v(raw, tap=raw.branch_tap, shift=raw.branch_shift_rad, seed=None)
    np.testing.assert_array_equal(a0, a1)
    th_b, pinj_b = base
    seed = dc_theta_base_delta(raw, th_b, pinj_b)
    a2 = dc_theta_lossy_v(raw, tap=raw.branch_tap, shift=raw.branch_shift_rad, seed=seed)
    assert a2.shape == a0.shape and np.all(np.isfinite(a2))
    m = (a0[:, 1] > 0.5) & (seed[:, 1] > 0.5)
    # 人造基态由 DC 生成,基态+Δ 按线性近似回到 DC(差 ~3e-5):只验"种子确实换了基点"(非全同数组);
    # 精度差由离线闸对真基态验(runs/diag_5060/e18_offline.json)
    diff = np.abs(a2[m, 0] - a0[m, 0])
    assert diff.max() > 1e-6 and (diff > 0).mean() > 0.9


def test_pf_data_lossybv_falls_back_bitwise_without_cache(raw, monkeypatch):
    """无基态缓存的家族:lossybv 与 lossyv 的 Data 各张量逐位一致(退回路径);有缓存时侧通道换 θ_bΔ。"""
    import copy
    from gridfm.data.normalize import Normalizer
    from gridfm.train.pretrain import TrainConfig
    cfg = TrainConfig()
    norm = Normalizer().fit([raw])
    s = copy.deepcopy(raw)
    norm.transform(s)
    _vc = noff("v")
    vstat = (float(norm.node_mean[_vc]), float(norm.node_std[_vc]))
    M._BASE_HINT_CACHE.clear()
    monkeypatch.setattr(M, "BASE_CACHE_DIR", Path("__no_such_dir__"))
    kw = dict(raw=raw, phys=True, edge=True, vhint=True, vstat=vstat, shz=True)
    d_v = M._pf_data(s, cfg, "group", dc="lossyv", **kw)
    M._BASE_HINT_CACHE.clear()
    d_bv = M._pf_data(s, cfg, "group", dc="lossybv", **kw)
    for k in ("edge_side", "theta_ref", "v_hint", "phys_b", "target"):
        np.testing.assert_array_equal(getattr(d_v, k).numpy(), getattr(d_bv, k).numpy(), err_msg=k)
    # 侧通道:除 dc 的 fallback 列(§21.1b:基态缺席 → 恒 1,模型可见"这是 DC 替补")外逐位一致
    sv, sb = d_v.side.numpy(), d_bv.side.numpy()
    fb_col = sv.shape[1] - 3 - 1                                # v-hint 3 列在末尾,其前是 dc 的 fallback 列
    keep = [c for c in range(sv.shape[1]) if c != fb_col]
    np.testing.assert_array_equal(sv[:, keep], sb[:, keep], err_msg="side(除 fallback 列)")
    assert np.all(sb[:, fb_col] == 1.0) and not np.all(sv[:, fb_col] == 1.0)
    # 有缓存:侧通道最后 dc 3 列应换成 θ_bΔ
    nf = raw.node_feat.copy()
    for f in ("pg", "pl"):
        nf[:, noff(f)] *= 0.97
    rb = replace(raw, node_feat=nf)
    db = dc_theta_lossy(rb)
    th_b = db[:, 0].astype(np.float64)
    pinj_b = (nf[:, noff("pg")] - nf[:, noff("pl")]).astype(np.float64)
    M._BASE_HINT_CACHE.clear()
    M._BASE_HINT_CACHE[raw.family_id] = (th_b, pinj_b)
    d_bv2 = M._pf_data(s, cfg, "group", dc="lossybv", **kw)
    M._BASE_HINT_CACHE.clear()
    seed = dc_theta_base_delta(raw, th_b, pinj_b)
    side = d_bv2.side.numpy()
    v_cols = 3                                                  # v-hint 3 列在末尾,dc 3 列在其前
    np.testing.assert_allclose(side[:, -(v_cols + 3)], seed[:, 0], atol=1e-6)
    assert np.all(side[:, -(v_cols + 1)] == 0.0)                # 基态提示生效:fallback 位 0
    assert np.abs(side[:, -(v_cols + 3)] - d_v.side.numpy()[:, -(v_cols + 3)]).max() > 1e-6   # 同上:非全同即可
