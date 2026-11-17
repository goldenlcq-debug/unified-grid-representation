# -*- coding: utf-8 -*-
"""E12 推理器内物理修正步(预注册 §15.2;2026-08-22 第 3 轮)单测:
P–θ 半步与 numpy 直接解逐项对拍(含 2M 对称边的 0.5 因子、ΔP/V 约定、串补带符号 B′)、Q–V 半步对拍与降误差、
覆盖率闸 / 发散闸、D15 端到端泄漏锁定、β=0 与关闭 Q–V 等价、COO 稀疏解的隐式微分梯度检查、串补口径与自由节点集。
夹具:tests/fixtures/mp_case118_mode(有 sidecar 变比 → 方程覆盖 100%);118 golden(无 tap、9 台在运变压器 → 覆盖率闸生效)。
"""
from __future__ import annotations

import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import splu
from torch_geometric.data import Batch

from gridfm.data.normalize import Normalizer
from gridfm.reason import pf_theta_edge as PTE
from gridfm.reason.pf_physics import dc_theta_lossy, pf_phys_arrays, pf_phys_denorm
from gridfm.reason.pf_task import PF_FIELDS, PFReasoner
from gridfm.reason.pf_theta_edge import (
    EDGE_SIDE_DIM, coo_solve, pf_edge_susceptance, pf_refine_free, pf_theta_ref, use_edges,
)
from gridfm.schema.fields import BRANCH_TABLE, NODE_TABLE
from gridfm.schema.sample import GridSample
from gridfm.train.masking import PF_UNKNOWN_FIELDS, _pf_typed_mask
from gridfm.train.pretrain import TrainConfig, build_models

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import pf_reasoner as M  # noqa: E402

FX = Path(__file__).parent / "fixtures" / "mp_case118_mode"
noff, boff = NODE_TABLE.offset_of, BRANCH_TABLE.offset_of
TH, VC = noff("theta"), noff("v")
K_TH, K_V = PF_FIELDS.index("theta"), PF_FIELDS.index("v")
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def pack():
    raw = GridSample.load_npz(FX / "mp_case118_00000.npz")
    z = np.load(FX / "family_extra.npz")
    raw.branch_tap, raw.branch_shift_rad = z["branch_tap"], z["branch_shift_rad"]
    norm = Normalizer().fit([raw])
    s = copy.deepcopy(raw)
    norm.transform(s)
    cfg = TrainConfig()
    vstat = (float(norm.node_mean[VC]), float(norm.node_std[VC]))
    return raw, s, norm, cfg, vstat


def _batch(pack, raw=None, s=None):
    raw0, s0, norm, cfg, vstat = pack
    d = M._pf_data(s if s is not None else s0, cfg, view="group", raw=raw if raw is not None else raw0,
                   dc="lossy", phys=True, edge=True, vhint=True, vstat=vstat)
    return Batch.from_data_list([d])


def _phys(batch, dn):
    return dict(index=batch.phys_index, g=batch.phys_g, b=batch.phys_b, denorm=dn, target=batch.target,
                loss_mask=batch.loss_mask, eqp_clean=batch.eqp_clean, eqq_clean=batch.eqq_clean,
                eqq_cand=batch.eqq_cand, edge_b=batch.edge_b, refine_free=batch.refine_free)


def _truth_pred(batch):
    """z 空间的真值四元组作"预测"(pg/qg/v/θ 列取 target)。"""
    cols = [noff(f) for f in PF_FIELDS]
    return batch.target[:, cols].clone().float()


def _np_state(raw):
    tt = raw.node_feat[:, TH].astype(np.float64)
    tv = raw.node_feat[:, VC].astype(np.float64)
    p = (raw.node_feat[:, noff("pg")] - raw.node_feat[:, noff("pl")]).astype(np.float64)
    q = (raw.node_feat[:, noff("qg")] - raw.node_feat[:, noff("ql")]).astype(np.float64)
    return tt, tv, p, q


def _np_mismatch(arr, th, vv, p, q):
    n = th.size
    idx, gy, by = arr["index"], arr["g"].astype(np.float64), arr["b"].astype(np.float64)
    vre, vim = vv * np.cos(th), vv * np.sin(th)
    ire = np.zeros(n); iim = np.zeros(n)
    np.add.at(ire, idx[0], gy * vre[idx[1]] - by * vim[idx[1]])
    np.add.at(iim, idx[0], gy * vim[idx[1]] + by * vre[idx[1]])
    return p - (vre * ire + vim * iim), q - (vim * ire - vre * iim)


def _np_bprime(raw, negb="sign"):
    n = raw.n_nodes
    b = pf_edge_susceptance(raw, negb).astype(np.float64)[:raw.n_branches]
    i, j = raw.edge_index[0].astype(np.int64), raw.edge_index[1].astype(np.int64)
    k = b != 0
    i, j, b = i[k], j[k], b[k]
    return sp.coo_matrix((np.r_[b, b, -b, -b], (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n)).tocsr()


def _reasoner(batch, cfg, **kw):
    torch.manual_seed(0)
    return PFReasoner(d=cfg.d, n_queries=4, n_layers=1, side_dim=batch.side.shape[1], gnn_layers=1,
                      theta_head=True, edge_side_dim=EDGE_SIDE_DIM, theta_edge=True, theta_wrap=True, v_hint=True, **kw)


# ---------------------------------------------------------------- P–θ 半步对拍

@pytest.mark.parametrize("force_sparse", [False, True])
def test_ptheta_step_matches_numpy_direct_solve(pack, monkeypatch, force_sparse):
    """一步 α=1 的 P–θ 修正 == numpy 直接解 B′·Δθ = ΔP/V(B′ 每条支路只算一次、串补带符号;自由节点 = refine_free);
    从带网损 DC 起点、神谕 V,一步后 θ 误差至少压 2 倍。稠密 / 稀疏两条路径都对拍。"""
    raw, s, norm, cfg, vstat = pack
    if force_sparse:
        monkeypatch.setattr(PTE, "DENSE_MAX_NODES", 8)
    batch = _batch(pack)
    assert float(batch.refine_free.float().mean()) > 0.9
    dn = torch.from_numpy(pf_phys_denorm(norm))
    r = _reasoner(batch, cfg, refine_steps=1)
    pred0 = _truth_pred(batch)
    d = dc_theta_lossy(raw)
    th0 = torch.from_numpy(d[:, 0].astype(np.float32))
    th1, pred1 = r._refine(th0, pred0, K_TH, _phys(batch, dn), batch.edge_index, batch.batch,
                           batch.ref_mask, batch.reach_mask, 1)
    assert r.refine_stats["cov_skip_p"] == 0 and r.refine_stats["div_skip_p"] == 0
    assert torch.equal(pred1, pred0)                      # 没开 Q–V:pred 原样返回
    # numpy 参照
    arr = pf_phys_arrays(raw, tap=raw.branch_tap, shift=raw.branch_shift_rad)
    tt, tv, p, q = _np_state(raw)
    dp, _ = _np_mismatch(arr, d[:, 0].astype(np.float64), tv, p, q)
    free = batch.refine_free.numpy()
    fidx = np.where(free)[0]
    rhs = np.where(arr["eqp_clean"], dp / np.maximum(tv, 0.5), 0.0)
    dth = np.zeros(raw.n_nodes)
    dth[fidx] = splu(_np_bprime(raw)[fidx][:, fidx].tocsc()).solve(rhs[fidx])
    got = (th1 - th0).detach().numpy().astype(np.float64)
    assert np.abs(got[fidx] - dth[fidx]).max() < 2e-5, np.abs(got[fidx] - dth[fidx]).max()
    assert np.all(got[~free] == 0.0)
    _mf, _tg, lm = _pf_typed_mask(raw.node_feat, raw.node_missing)
    m = lm[:, TH]
    e0 = np.abs(d[m, 0] - tt[m]).mean()
    e1 = np.abs(th1.detach().numpy()[m] - tt[m]).mean()
    assert e1 < 0.5 * e0, (e0, e1)


def test_ptheta_step_from_truth_is_noop_and_differentiable(pack):
    """真值起点失配≈0 → Δθ≈0;对 θ⁰ 与 α 都有有限梯度(隐式微分)。"""
    raw, s, norm, cfg, vstat = pack
    batch = _batch(pack)
    dn = torch.from_numpy(pf_phys_denorm(norm))
    r = _reasoner(batch, cfg, refine_steps=1)
    pred0 = _truth_pred(batch)
    th0 = batch.target[:, TH].clone().float().requires_grad_(True)
    th1, _ = r._refine(th0, pred0, K_TH, _phys(batch, dn), batch.edge_index, batch.batch,
                       batch.ref_mask, batch.reach_mask, 1)
    assert float((th1 - th0).abs().max()) < 1e-3
    th1.sum().backward()
    assert torch.isfinite(th0.grad).all() and torch.isfinite(r.refine_alpha.grad).all()


# ---------------------------------------------------------------- 两道闸

def test_divergence_gate_skips_graph(pack):
    """把 B′ 权重缩小 1e-6 → Δθ 巨大 → 发散闸:该图本步不更新(θ 逐位不变),闸门计数 1。"""
    raw, s, norm, cfg, vstat = pack
    batch = _batch(pack)
    dn = torch.from_numpy(pf_phys_denorm(norm))
    r = _reasoner(batch, cfg, refine_steps=1)
    phys = _phys(batch, dn)
    phys["edge_b"] = phys["edge_b"] * 1e-6
    th0 = torch.from_numpy(dc_theta_lossy(raw)[:, 0].astype(np.float32))
    th1, _ = r._refine(th0, _truth_pred(batch), K_TH, phys, batch.edge_index, batch.batch,
                       batch.ref_mask, batch.reach_mask, 1)
    assert r.refine_stats["div_skip_p"] == 1 and r.refine_stats["max_dth"] > 1.0
    assert torch.equal(th1, th0)


def test_coverage_gate_skips_graph(pack, case_dir):
    """覆盖率闸:(a) 把一半干净 P 方程行作废 → 该图不修(θ 逐位不变);(b) 118 golden(无 tap、9 台在运变压器)覆盖率 < 99% → 整图不修。"""
    raw, s, norm, cfg, vstat = pack
    batch = _batch(pack)
    dn = torch.from_numpy(pf_phys_denorm(norm))
    r = _reasoner(batch, cfg, refine_steps=1)
    phys = _phys(batch, dn)
    eq = phys["eqp_clean"].clone()
    idx = torch.where(eq)[0]
    eq[idx[: idx.numel() // 2]] = False
    phys["eqp_clean"] = eq
    th0 = torch.from_numpy(dc_theta_lossy(raw)[:, 0].astype(np.float32))
    th1, _ = r._refine(th0, _truth_pred(batch), K_TH, phys, batch.edge_index, batch.batch,
                       batch.ref_mask, batch.reach_mask, 1)
    assert r.refine_stats["cov_skip_p"] == 1 and torch.equal(th1, th0)
    # (b) golden:无 sidecar → 变压器相邻行作废 → 覆盖率不足,修正步对它是空操作
    from gridfm.psasp_io.case import PsaspCase
    from gridfm.psasp_io.catalog import Catalog
    from gridfm.schema.assemble import build_sample
    c = PsaspCase(case_dir)
    g = build_sample(c, Catalog(c), sample_id="s0", family_id="t")
    arr = pf_phys_arrays(g)
    _tr, rm, reach, _av = pf_theta_ref(g)
    free = reach & ~rm
    cov = (arr["eqp_clean"] & free).sum() / free.sum()
    assert cov < 0.99, cov
    ng = Normalizer().fit([g])
    gs = copy.deepcopy(g); ng.transform(gs)
    dg = M._pf_data(gs, cfg, view="group", raw=g, dc="lossy", phys=True, edge=True, vhint=True,
                    vstat=(float(ng.node_mean[VC]), float(ng.node_std[VC])))
    bg = Batch.from_data_list([dg])
    rg = _reasoner(bg, cfg, refine_steps=2, refine_qv=True)
    dng = torch.from_numpy(pf_phys_denorm(ng))
    th0 = torch.from_numpy(dc_theta_lossy(g)[:, 0].astype(np.float32))
    p0 = _truth_pred(bg)
    th1, p1 = rg._refine(th0, p0, K_TH, _phys(bg, dng), bg.edge_index, bg.batch, bg.ref_mask, bg.reach_mask, 1)
    assert rg.refine_stats["cov_skip_p"] == 1 and rg.refine_stats["cov_skip_q"] == 1
    assert torch.equal(th1, th0) and torch.equal(p1, p0)


# ---------------------------------------------------------------- Q–V 半步

def test_qv_step_matches_numpy_and_reduces_v_error(pack):
    """θ = 真值、PQ 电压从平 1.0 起步,α=0、β=1 的一步 Q–V == numpy 解 B″·ΔV = ΔQ/V;v 误差至少压 3 倍;θ 不动。"""
    raw, s, norm, cfg, vstat = pack
    batch = _batch(pack)
    dn = torch.from_numpy(pf_phys_denorm(norm))
    r = _reasoner(batch, cfg, refine_steps=1, refine_qv=True)
    with torch.no_grad():
        r.refine_alpha.zero_()
    pred0 = _truth_pred(batch)
    lm_v = batch.loss_mask[:, VC]
    z1 = (1.0 - vstat[0]) / vstat[1]
    pred0[lm_v, K_V] = z1                                   # PQ 电压 = 平 1.0
    th0 = batch.target[:, TH].clone().float()
    th1, pred1 = r._refine(th0, pred0, K_TH, _phys(batch, dn), batch.edge_index, batch.batch,
                           batch.ref_mask, batch.reach_mask, 1)
    assert torch.equal(th1, th0)
    assert r.refine_stats["cov_skip_q"] == 0 and r.refine_stats["div_skip_q"] == 0
    # numpy 参照
    arr = pf_phys_arrays(raw, tap=raw.branch_tap, shift=raw.branch_shift_rad)
    tt, tv, p, q = _np_state(raw)
    vv = tv.copy(); vv[lm_v.numpy()] = 1.0
    _, dq = _np_mismatch(arr, tt, vv, p, q)
    rows = arr["eqq_clean"]
    qidx = np.where(rows)[0]
    n = raw.n_nodes
    Bpp = sp.coo_matrix((-arr["b"].astype(np.float64), (arr["index"][0], arr["index"][1])), shape=(n, n)).tocsr()
    dv = splu(Bpp[qidx][:, qidx].tocsc() + 1e-8 * sp.identity(qidx.size, format="csc")).solve(dq[qidx] / np.maximum(vv[qidx], 0.5))
    v1 = pred1[:, K_V].detach().numpy().astype(np.float64) * vstat[1] + vstat[0]
    assert np.abs(v1[qidx] - (vv[qidx] + dv)).max() < 1e-4
    # 非 PQ 行的 v 列原样
    keep = ~lm_v.numpy()
    assert np.array_equal(pred1[keep, K_V].detach().numpy(), pred0[keep, K_V].detach().numpy())
    e0 = np.abs(vv[qidx] - tv[qidx]).mean()
    e1 = np.abs(v1[qidx] - tv[qidx]).mean()
    assert e1 < e0 / 3, (e0, e1)


def test_qv_beta_zero_equals_qv_off(pack):
    """β=0 的 Q–V 与不开 Q–V 同输出(v 只差一次 z↔pu 往返的浮点误差);β 有梯度。"""
    raw, s, norm, cfg, vstat = pack
    batch = _batch(pack)
    dn = torch.from_numpy(pf_phys_denorm(norm))
    r0 = _reasoner(batch, cfg, refine_steps=2)
    r1 = _reasoner(batch, cfg, refine_steps=2, refine_qv=True)
    with torch.no_grad():
        r1.refine_beta.zero_()
    th0 = torch.from_numpy(dc_theta_lossy(raw)[:, 0].astype(np.float32))
    p0 = _truth_pred(batch)
    a_th, a_p = r0._refine(th0, p0, K_TH, _phys(batch, dn), batch.edge_index, batch.batch, batch.ref_mask, batch.reach_mask, 1)
    b_th, b_p = r1._refine(th0, p0.clone().requires_grad_(True), K_TH, _phys(batch, dn), batch.edge_index, batch.batch,
                           batch.ref_mask, batch.reach_mask, 1)
    assert torch.allclose(a_th, b_th, atol=1e-5) and torch.allclose(a_p, b_p, atol=1e-5)
    (b_th.sum() + b_p.sum()).backward()
    assert r1.refine_beta.grad is not None and torch.isfinite(r1.refine_beta.grad).all()


# ---------------------------------------------------------------- D15 端到端

def test_refine_d15_leak_free_end_to_end(pack):
    """扰动全部被监督量(PQ 的 v/θ、PV 的 qg/θ、平衡机 pg/qg)后,含 K=2 + Q–V 修正步的推理器输出**逐位不变**。"""
    raw, s, norm, cfg, vstat = pack
    nf = raw.node_feat.copy()
    tcol = noff("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    for i in range(raw.n_nodes):
        for f in PF_UNKNOWN_FIELDS[int(types[i])]:
            nf[i, noff(f)] += 3.3
    raw2 = replace(raw, node_feat=nf)
    raw2.branch_tap, raw2.branch_shift_rad = raw.branch_tap, raw.branch_shift_rad
    s2 = copy.deepcopy(raw2); norm.transform(s2)
    b1, b2 = _batch(pack), _batch(pack, raw=raw2, s=s2)
    assert not torch.equal(b1.target, b2.target)            # 监督量确实被扰动了
    enc, _ = build_models(cfg); enc.eval()
    dn = torch.from_numpy(pf_phys_denorm(norm))
    r = _reasoner(b1, cfg, refine_steps=2, refine_qv=True)
    p1, _ = M._forward(enc, r, b1, False, torch.float32, CPU, dn=dn)
    p2, _ = M._forward(enc, r, b2, False, torch.float32, CPU, dn=dn)
    assert torch.equal(p1, p2)
    assert r.refine_stats["cov_skip_p"] == 0 and r.refine_stats["cov_skip_q"] == 0


def test_reasoner_forward_backward_with_qv_and_config_pairing(pack):
    """端到端:K=2 + Q–V 前向有限、反向 α/β 拿到梯度;refine_qv 需要 refine_steps。"""
    raw, s, norm, cfg, vstat = pack
    batch = _batch(pack)
    enc, _ = build_models(cfg); enc.eval()
    dn = torch.from_numpy(pf_phys_denorm(norm))
    r = _reasoner(batch, cfg, refine_steps=2, refine_qv=True)
    pred, loss = M._forward(enc, r, batch, False, torch.float32, CPU, dn=dn)
    assert torch.isfinite(pred).all()
    loss.backward()
    assert float(r.refine_alpha.grad.abs().sum()) > 0 and float(r.refine_beta.grad.abs().sum()) > 0
    with pytest.raises(ValueError):
        _reasoner(batch, cfg, refine_steps=0, refine_qv=True)


# ---------------------------------------------------------------- 求解器与口径

def test_coo_solve_gradcheck():
    """一般稀疏矩阵(非对称)的直接解:前向 == scipy;对 rhs 的隐式微分过 gradcheck。"""
    rng = np.random.default_rng(3)
    n = 30
    A = sp.random(n, n, density=0.15, random_state=3).toarray() + 5 * np.eye(n)
    coo = sp.coo_matrix(A)
    rows, cols = torch.from_numpy(coo.row.astype(np.int64)), torch.from_numpy(coo.col.astype(np.int64))
    vals = torch.from_numpy(coo.data)
    free = torch.ones(n, dtype=torch.bool); free[::7] = False
    rhs = torch.from_numpy(rng.normal(size=n))
    x = coo_solve(rhs, rows, cols, vals, free).double()
    fidx = np.where(free.numpy())[0]
    ref = np.zeros(n); ref[fidx] = np.linalg.solve(A[np.ix_(fidx, fidx)], rhs.numpy()[fidx])
    assert np.abs(x.numpy() - ref).max() < 1e-8
    rhs_g = rhs.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda r_: coo_solve(r_, rows, cols, vals, free).double(), (rhs_g,), eps=1e-6, atol=1e-6)


def test_edge_susceptance_negb_and_refine_free(pack):
    """串补口径:zero 置 0、sign 带符号、abs 取绝对值;把一个叶节点的唯一支路改成 x<0 后,zero 口径下它成为孤块(不在 refine_free),
    sign 口径下仍可达。"""
    raw = pack[0]
    m = raw.n_branches
    use = use_edges(raw)
    i, j = raw.edge_index[0].astype(np.int64), raw.edge_index[1].astype(np.int64)
    deg = np.bincount(np.r_[i[use], j[use]], minlength=raw.n_nodes)
    leaf = int(np.where(deg == 1)[0][0])
    k = int(np.where(use & ((i == leaf) | (j == leaf)))[0][0])
    bf = raw.branch_feat.copy()
    bf[k, boff("x")] = -abs(bf[k, boff("x")])
    raw2 = replace(raw, branch_feat=bf)
    bz, bs_, ba = (pf_edge_susceptance(raw2, nb)[:m] for nb in ("zero", "sign", "abs"))
    assert bz[k] == 0.0 and bs_[k] < 0 and ba[k] == -bs_[k] > 0
    assert np.array_equal(bz[np.arange(m) != k], bs_[np.arange(m) != k])
    _tr, rm, reach, _av = pf_theta_ref(raw2)
    assert reach[leaf]
    fz = pf_refine_free(raw2, bz, rm, reach)
    fs = pf_refine_free(raw2, bs_, rm, reach)
    assert not fz[leaf] and fs[leaf]
    assert fs.sum() == (reach & ~rm).sum() and fz.sum() == fs.sum() - 1
    with pytest.raises(ValueError):
        pf_edge_susceptance(raw2, "bogus")


# ---------------------------------------------------------------- E13 开关(默认关 = E12 逐位一致)

def test_e13_aux_loss_and_pre_refine_pred(pack):
    """refine_aux_weight=0 与原损失逐位一致;>0 时损失变化且有限;pre_refine_pred 的 θ 列 = 修正前(折回)、与最终输出不同。"""
    raw, s, norm, cfg, vstat = pack
    batch = _batch(pack)
    enc, _ = build_models(cfg); enc.eval()
    dn = torch.from_numpy(pf_phys_denorm(norm))
    r = _reasoner(batch, cfg, refine_steps=2, refine_qv=True)
    p0, l0 = M._forward(enc, r, batch, False, torch.float32, CPU, dn=dn)
    p1, l1 = M._forward(enc, r, batch, False, torch.float32, CPU, dn=dn, refine_aux_weight=0.0)
    assert torch.equal(p0, p1) and torch.equal(l0, l1)
    p2, l2 = M._forward(enc, r, batch, False, torch.float32, CPU, dn=dn, refine_aux_weight=1.0)
    assert torch.equal(p0, p2) and torch.isfinite(l2) and not torch.equal(l0, l2)
    pre = r.pre_refine_pred
    assert pre.shape == p0.shape and float(pre[:, K_TH].abs().max()) <= np.pi + 1e-6
    assert not torch.equal(pre[:, K_TH], p0[:, K_TH]) and not torch.equal(pre[:, K_V], p0[:, K_V])
    l2.backward()
    assert all(torch.isfinite(p.grad).all() for p in r.parameters() if p.grad is not None)


def test_e13_train_dropout_hint_and_refine(pack):
    """hint_p=1:侧通道末三列 → (0,0,1)、残差基 0、v_hint_ok 全 False(与 E10b 同一种输入);refine_p=1:refine_free 置空 →
    覆盖率闸整图跳过,θ 与 α=0 的输出逐位相同;p=0 不动数据。"""
    raw, s, norm, cfg, vstat = pack
    rng = np.random.default_rng(0)
    d = M._pf_data(s, cfg, view="group", raw=raw, dc="lossy", phys=True, edge=True, vhint=True, vstat=vstat)
    side0, vh0, free0 = d.side.clone(), d.v_hint.clone(), d.refine_free.clone()
    assert M.apply_train_dropout([d], rng, 0.0, 0.0) == (0, 0)
    assert torch.equal(d.side, side0) and torch.equal(d.v_hint, vh0) and torch.equal(d.refine_free, free0)
    assert M.apply_train_dropout([d], rng, 1.0, 0.0) == (1, 0)
    assert torch.equal(d.side[:, :-3], side0[:, :-3])
    assert torch.equal(d.side[:, -3:], torch.tensor([0.0, 0.0, 1.0]).expand(d.side.shape[0], 3))
    assert float(d.v_hint.abs().max()) == 0.0 and not bool(d.v_hint_ok.any())
    assert M.apply_train_dropout([d], rng, 0.0, 1.0) == (0, 1)
    assert not bool(d.refine_free.any())
    enc, _ = build_models(cfg); enc.eval()
    dn = torch.from_numpy(pf_phys_denorm(norm))
    r = _reasoner(_batch(pack), cfg, refine_steps=2, refine_qv=True)
    b_drop = Batch.from_data_list([d])
    p_drop, _ = M._forward(enc, r, b_drop, False, torch.float32, CPU, dn=dn)
    assert r.refine_stats["cov_skip_p"] == 1
    with torch.no_grad():
        r.refine_alpha.zero_(); r.refine_beta.zero_()
    p_a0, _ = M._forward(enc, r, b_drop, False, torch.float32, CPU, dn=dn)
    assert torch.allclose(p_drop[:, K_TH], p_a0[:, K_TH], atol=1e-6)


def _mean_dp_rows(batch, dn, theta, pred):
    """干净 P 行的 mean|ΔP|(物理量纲),用于线搜索断言。"""
    from gridfm.reason.pf_physics import pf_phys_residual
    cols = [pred[:, c] for c in range(pred.shape[1])]
    cols[K_TH] = theta
    p = torch.stack(cols, -1)
    dp, _ = pf_phys_residual(p, batch.target.float(), batch.loss_mask, batch.phys_index, batch.phys_g, batch.phys_b, dn)
    rows = batch.eqp_clean & batch.refine_free
    return float(dp[rows].abs().mean())


def test_e13_line_search_never_increases_residual_and_off_is_bitwise(pack):
    """refine_ls=False 与 E12 逐位一致;refine_ls=True 时:(a) 正常步(α≈1)与不搜索的一步相同(尺度 1 被选中);
    (b) 把 α 放大到 6 造成过冲 → 线搜索缩到 ½/¼ 或跳过,修正后的 mean|ΔP| 不高于修正前;(c) 对 θ⁰ 可微。"""
    raw, s, norm, cfg, vstat = pack
    batch = _batch(pack)
    dn = torch.from_numpy(pf_phys_denorm(norm))
    th0 = torch.from_numpy(dc_theta_lossy(raw)[:, 0].astype(np.float32))
    pred0 = _truth_pred(batch)
    r0 = _reasoner(batch, cfg, refine_steps=1)
    r1 = _reasoner(batch, cfg, refine_steps=1, refine_ls=True)
    a, _ = r0._refine(th0, pred0, K_TH, _phys(batch, dn), batch.edge_index, batch.batch, batch.ref_mask, batch.reach_mask, 1)
    b, _ = r1._refine(th0, pred0, K_TH, _phys(batch, dn), batch.edge_index, batch.batch, batch.ref_mask, batch.reach_mask, 1)
    assert torch.equal(a, b) and r1.refine_stats["ls_scales"][0][0] == 1.0
    res0 = _mean_dp_rows(batch, dn, th0, pred0)
    assert _mean_dp_rows(batch, dn, b, pred0) < res0
    # 过冲:α=6 → 不搜索时残差变大;搜索后不增
    with torch.no_grad():
        r0.refine_alpha.fill_(6.0); r1.refine_alpha.fill_(6.0)
    c, _ = r0._refine(th0, pred0, K_TH, _phys(batch, dn), batch.edge_index, batch.batch, batch.ref_mask, batch.reach_mask, 1)
    d, _ = r1._refine(th0.clone().requires_grad_(True), pred0, K_TH, _phys(batch, dn), batch.edge_index, batch.batch,
                      batch.ref_mask, batch.reach_mask, 1)
    res_c, res_d = _mean_dp_rows(batch, dn, c, pred0), _mean_dp_rows(batch, dn, d.detach(), pred0)
    assert res_c > res0, "α=6 应造成过冲(测试要有判别力)"
    assert res_d <= res0 + 1e-9, (res0, res_d)
    assert r1.refine_stats["ls_scales"][0][0] in (0.0, 0.5, 0.25)
    d.sum().backward()
