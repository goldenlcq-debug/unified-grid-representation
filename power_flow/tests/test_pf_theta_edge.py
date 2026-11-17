# -*- coding: utf-8 -*-
"""θ 边势重建(pf_theta_edge,2026-08-21 5060 线)单测:D15 泄漏锁定 + 势重建精确性 + 结构/梯度。"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from gridfm.data.graph import sample_to_data
from gridfm.data.normalize import Normalizer
from gridfm.psasp_io.case import PsaspCase
from gridfm.psasp_io.catalog import Catalog
from gridfm.reason.pf_physics import dc_theta
from gridfm.reason.pf_task import (
    PF_FIELDS, PF_SIDE_DIM, PFReasoner, STATE_FIELDS, pf_input_view, pf_loss, pf_side_features,
)
from gridfm.reason.pf_theta_edge import (
    EDGE_SIDE_DIM, DTH_DC_COL, pf_edge_loss, pf_edge_side_features, pf_theta_ref,
    potential_reconstruct, use_edges, wrap_angle,
)
from gridfm.schema.assemble import build_sample
from gridfm.schema.fields import BRANCH_TABLE, NODE_TABLE
from gridfm.train.masking import PF_UNKNOWN_FIELDS, _pf_typed_mask
from gridfm.train.pretrain import TrainConfig, build_models

TH = NODE_TABLE.offset_of("theta")


def _raw(case_dir, sid="s0"):
    c = PsaspCase(case_dir)
    return build_sample(c, Catalog(c), sample_id=sid, family_id="t")


def _norm(case_dir, sid="s0"):
    s = _raw(case_dir, sid)
    Normalizer().fit([s]).transform(s)
    return s


# ---------------- D15 泄漏锁定(强制) ----------------

def test_edge_side_features_leak_free(case_dir):
    """扰动全部被监督量(PQ/PV 的 θ、V,PV/平衡机的 qg/pg)与支路潮流 → 边侧特征与节点侧常量逐位不变。"""
    raw = _raw(case_dir)
    es0, eu0 = pf_edge_side_features(raw)
    tr0, rm0, reach0, avail0 = pf_theta_ref(raw)
    assert es0.shape == (2 * raw.n_branches, EDGE_SIDE_DIM) and eu0.shape == (2 * raw.n_branches,)
    assert eu0.sum() > 0 and reach0.sum() > 0 and rm0.sum() >= 1
    _mf, _tg, lm = _pf_typed_mask(raw.node_feat, raw.node_missing)
    pert = raw.node_feat.copy()
    rng = np.random.default_rng(0)
    pert[lm] += rng.normal(0, 1.0, size=int(lm.sum()))
    bf = raw.branch_feat.copy()
    for f in ("pi", "qi", "pj", "qj"):
        bf[:, BRANCH_TABLE.offset_of(f)] += rng.normal(0, 1.0, size=bf.shape[0])
    from dataclasses import replace
    raw2 = replace(raw, node_feat=pert, branch_feat=bf)
    es1, eu1 = pf_edge_side_features(raw2)
    tr1, rm1, reach1, avail1 = pf_theta_ref(raw2)
    np.testing.assert_array_equal(es0, es1)
    np.testing.assert_array_equal(eu0, eu1)
    np.testing.assert_array_equal(tr0, tr1)
    np.testing.assert_array_equal(rm0, rm1)
    np.testing.assert_array_equal(reach0, reach1)
    np.testing.assert_array_equal(avail0, avail1)
    # 平衡机给定角是 theta_ref 的唯一来源;非平衡机行恒 0
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    assert (tr0[types != 2] == 0).all()
    assert np.allclose(tr0[rm0], raw.node_feat[rm0, TH])


def test_dc_theta_lossy_leak_free_and_guarded(case_dir):
    """带网损 DC:形状 [N,3];扰动全部被监督量与支路潮流不改特征(D15);118 上不回退且不劣于普通 DC;前两列语义同 dc_theta。"""
    from gridfm.reason.pf_physics import dc_theta_lossy
    raw = _raw(case_dir)
    dl = dc_theta_lossy(raw)
    d0 = dc_theta(raw)
    assert dl.shape == (raw.n_nodes, 3) and np.isfinite(dl).all()
    np.testing.assert_array_equal(dl[:, 1], d0[:, 1])           # valid 位同义
    assert dl[:, 2].max() == 0.0, "118 黄金算例不应触发回退"
    _mf, _tg, lm = _pf_typed_mask(raw.node_feat, raw.node_missing)
    m = lm[:, TH] & (d0[:, 1] > 0.5)
    err0 = np.abs(d0[m, 0] - raw.node_feat[m, TH]).mean()
    errL = np.abs(dl[m, 0] - raw.node_feat[m, TH]).mean()
    assert errL <= err0 * 1.05, (err0, errL)
    # 泄漏锁定:扰动被监督位与支路潮流
    rng = np.random.default_rng(1)
    pert = raw.node_feat.copy(); pert[lm] += rng.normal(0, 1.0, size=int(lm.sum()))
    bf = raw.branch_feat.copy()
    for f in ("pi", "qi", "pj", "qj"):
        bf[:, BRANCH_TABLE.offset_of(f)] += rng.normal(0, 1.0, size=bf.shape[0])
    from dataclasses import replace
    dl2 = dc_theta_lossy(replace(raw, node_feat=pert, branch_feat=bf))
    np.testing.assert_array_equal(dl, dl2)
    # 发散保护:把 loss_cap 压到 0 → 第一轮即触发回退,θ 退回普通 DC,fallback=1
    dl3 = dc_theta_lossy(raw, loss_cap=0.0)
    assert dl3[:, 2].max() == 1.0
    np.testing.assert_allclose(dl3[:, 0], d0[:, 0], atol=1e-6)


def test_edge_side_layout_and_dc_consistency(case_dir):
    """前 M 条 i→j、后 M 条 j→i:Δθ_dc 与 p_dc 反号,其余列相同;Δθ_dc 与 dc_theta 一致。"""
    raw = _raw(case_dir)
    d = dc_theta(raw)
    es, eu = pf_edge_side_features(raw, d)
    m = raw.n_branches
    fwd, rev = es[:m], es[m:]
    np.testing.assert_allclose(rev[:, DTH_DC_COL], -fwd[:, DTH_DC_COL])
    keep = [k for k in range(EDGE_SIDE_DIM) if k not in (DTH_DC_COL, EDGE_SIDE_DIM - 1)]
    np.testing.assert_array_equal(rev[:, keep], fwd[:, keep])
    use = use_edges(raw)
    i, j = raw.edge_index
    ok = use & (d[i, 1] > 0.5) & (d[j, 1] > 0.5)
    np.testing.assert_allclose(fwd[ok, DTH_DC_COL], (d[j, 0] - d[i, 0])[ok], atol=1e-5)
    assert (fwd[~ok, DTH_DC_COL] == 0).all()
    np.testing.assert_array_equal(eu[:m], use)


# ---------------- 势重建 ----------------

def _rand_graph(n, m_extra, rng):
    """随机连通图:生成树 + m_extra 条额外边,双向化。"""
    src, dst = [], []
    for v in range(1, n):
        u = int(rng.integers(0, v))
        src.append(u); dst.append(v)
    for _ in range(m_extra):
        u, v = rng.integers(0, n, size=2)
        if u != v:
            src.append(int(u)); dst.append(int(v))
    ei = np.array([src + dst, dst + src], dtype=np.int64)
    return torch.from_numpy(ei)


def test_potential_reconstruct_exact_and_batched():
    """δ 取自真势 → 精确回收(含网状图、两图拼批、不同大小、多平衡机)。"""
    rng = np.random.default_rng(0)
    graphs = [(37, 20), (61, 0), (12, 30)]
    eis, phis, refs, rmask, nb = [], [], [], [], []
    off = 0
    for g, (n, mx) in enumerate(graphs):
        ei = _rand_graph(n, mx, rng)
        phi = torch.from_numpy(rng.normal(0, 3.0, size=n))
        eis.append(ei + off)
        phis.append(phi)
        r = torch.zeros(n, dtype=torch.bool)
        r[0] = True
        if g == 2:
            r[5] = True                      # 多参考节点
        rmask.append(r)
        refs.append(torch.where(r, phi, torch.zeros_like(phi)))
        nb.append(torch.full((n,), g, dtype=torch.long))
        off += n
    ei = torch.cat(eis, 1)
    phi = torch.cat(phis)
    delta = (phi[ei[1]] - phi[ei[0]]).float().requires_grad_(True)
    theta = potential_reconstruct(delta, ei, torch.ones(ei.size(1)), torch.cat(nb),
                                  torch.cat(refs).float(), torch.cat(rmask), num_graphs=3)
    assert torch.allclose(theta.double(), phi, atol=1e-4), (theta - phi).abs().max()
    theta.sum().backward()
    assert delta.grad is not None and torch.isfinite(delta.grad).all() and delta.grad.abs().sum() > 0


def test_potential_reconstruct_least_squares_averages_inconsistency():
    """三角形回路上 δ 不自洽 → 最小二乘解:所有边残差同量,且不崩。"""
    ei = torch.tensor([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]])
    delta = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])      # 绕一圈 +3,无势能精确满足
    ref = torch.zeros(3); rm = torch.tensor([True, False, False])
    th = potential_reconstruct(delta, ei, torch.ones(6), torch.zeros(3, dtype=torch.long), ref, rm, 1)
    assert torch.isfinite(th).all() and abs(float(th[0])) < 1e-6
    res = th[ei[1]] - th[ei[0]] - delta
    assert torch.allclose(res[:3].abs(), res[:3].abs().mean().expand(3), atol=1e-5)


def test_potential_reconstruct_large_path_exact_no_ridge_bias():
    """N=1000 路径图、0.05 rad/跳(末端 ~50 rad):有参考分量须精确回收(2026-08-21 审查:全行加岭会偏 ~7 rad)。"""
    n = 1000
    ar = torch.arange(n - 1)
    ei = torch.stack([torch.cat([ar, ar + 1]), torch.cat([ar + 1, ar])])
    phi = 0.05 * torch.arange(n, dtype=torch.float64)
    delta = (phi[ei[1]] - phi[ei[0]]).float()
    rm = torch.zeros(n, dtype=torch.bool); rm[0] = True
    th = potential_reconstruct(delta, ei, torch.ones(ei.size(1)), torch.zeros(n, dtype=torch.long),
                               phi.float(), rm, 1, ridge_mask=torch.zeros(n, dtype=torch.bool))
    assert (th.double() - phi).abs().max() < 1e-4, (th.double() - phi).abs().max()
    # 混合:同批另一图无参考(整图 ridge_mask=True)→ 有界;有参考图仍精确
    n2 = 30
    ei2 = _rand_graph(n2, 10, np.random.default_rng(3)) + n
    ei_all = torch.cat([ei, ei2], 1)
    d_all = torch.cat([delta, torch.randn(ei2.size(1))])
    nb = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n2, dtype=torch.long)])
    rm_all = torch.cat([rm, torch.zeros(n2, dtype=torch.bool)])
    ridge_all = torch.cat([torch.zeros(n, dtype=torch.bool), torch.ones(n2, dtype=torch.bool)])
    th2 = potential_reconstruct(d_all, ei_all, torch.ones(ei_all.size(1)), nb,
                                torch.cat([phi.float(), torch.zeros(n2)]), rm_all, 2, ridge_mask=ridge_all)
    assert (th2[:n].double() - phi).abs().max() < 1e-4
    assert torch.isfinite(th2[n:]).all() and th2[n:].abs().max() < 1e3


def test_theta_ref_reach_requires_reference_in_component(case_dir):
    """平衡机行 STATE missing → 无有效参考角 → reach 全 False(前向回退直接头),而非零均值岭解。"""
    raw = _raw(case_dir)
    tr, rm, reach, avail = pf_theta_ref(raw)
    assert rm.sum() >= 1 and reach.sum() > 0
    from dataclasses import replace
    from gridfm.schema.fields import Group
    from gridfm.schema.sample import group_bit_index
    nm = raw.node_missing.copy()
    nm[rm, group_bit_index(Group.STATE)] = True
    raw2 = replace(raw, node_missing=nm)
    tr2, rm2, reach2, avail2 = pf_theta_ref(raw2)
    assert rm2.sum() == 0 and reach2.sum() == 0
    assert (avail2.sum() == avail.sum() - rm.sum())


def test_potential_reconstruct_sparse_matches_dense_and_scipy():
    """稀疏 PCG 路径:与稠密路径同解(≤4096 节点)、与 scipy 直接解同解(3 万节点网状图)、隐式梯度与稠密一致。"""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    rng = np.random.default_rng(7)
    # (1) 小图:dense vs sparse 同解、同梯度
    n = 300
    ei = _rand_graph(n, 200, rng)
    phi = torch.from_numpy(rng.normal(0, 2.0, size=n))
    delta_true = (phi[ei[1]] - phi[ei[0]]).float()
    noise = torch.from_numpy(rng.normal(0, 0.05, size=ei.size(1))).float()
    rm = torch.zeros(n, dtype=torch.bool); rm[0] = True; rm[17] = True
    ref = torch.where(rm, phi.float(), torch.zeros(n))
    nb = torch.zeros(n, dtype=torch.long)
    zr = torch.zeros(n, dtype=torch.bool)
    d1 = (delta_true + noise).clone().requires_grad_(True)
    d2 = (delta_true + noise).clone().requires_grad_(True)
    th_d = potential_reconstruct(d1, ei, torch.ones(ei.size(1)), nb, ref, rm, 1, ridge_mask=zr, mode="dense")
    th_s = potential_reconstruct(d2, ei, torch.ones(ei.size(1)), nb, ref, rm, 1, ridge_mask=zr, mode="sparse")
    assert torch.allclose(th_d, th_s, atol=1e-5), (th_d - th_s).abs().max()
    wgt = torch.randn(n)
    (th_d * wgt).sum().backward(); (th_s * wgt).sum().backward()
    assert torch.allclose(d1.grad, d2.grad, atol=1e-5), (d1.grad - d2.grad).abs().max()
    # (2) 1.2 万节点网状图(生成树 + 8 千随机边):sparse vs scipy spsolve;auto 模式自动走 sparse
    #     (3 万 / 7 万节点实测:CPU 0.09 s / 0.22 s,~100 次矩阵向量乘,误差 1e-6;此处缩小只为测试提速)
    n = 12_000
    ei = _rand_graph(n, 8_000, rng)
    phi = torch.from_numpy(np.cumsum(rng.normal(0, 0.05, size=n)))
    delta = (phi[ei[1]] - phi[ei[0]]).float() + torch.from_numpy(rng.normal(0, 0.02, size=ei.size(1))).float()
    rm = torch.zeros(n, dtype=torch.bool); rm[0] = True
    ref = torch.where(rm, phi.float(), torch.zeros(n))
    th = potential_reconstruct(delta, ei, torch.ones(ei.size(1)), torch.zeros(n, dtype=torch.long), ref, rm, 1,
                               ridge_mask=torch.zeros(n, dtype=torch.bool), mode="auto")
    s, t = ei[0].numpy(), ei[1].numpy()
    L = sp.coo_matrix((np.r_[np.ones(s.size), np.ones(s.size), -np.ones(s.size), -np.ones(s.size)],
                       (np.r_[s, t, s, t], np.r_[s, t, t, s])), shape=(n, n)).tocsr()
    b = np.zeros(n); np.add.at(b, t, delta.double().numpy()); np.add.at(b, s, -delta.double().numpy())
    free = np.arange(1, n)
    x = np.zeros(n); x[0] = float(phi[0])
    x[free] = spsolve(L[free][:, free].tocsc(), b[free] - L[free][:, [0]].toarray().ravel() * x[0])
    assert np.abs(th.double().numpy() - x).max() < 1e-4, np.abs(th.double().numpy() - x).max()


def test_potential_reconstruct_no_reference_is_bounded():
    """无参考角的图:岭解有界(零均值势),调用方按 reach 回退。"""
    ei = _rand_graph(20, 5, np.random.default_rng(1))
    delta = torch.randn(ei.size(1))
    th = potential_reconstruct(delta, ei, torch.ones(ei.size(1)), torch.zeros(20, dtype=torch.long),
                               torch.zeros(20), torch.zeros(20, dtype=torch.bool), 1)
    assert torch.isfinite(th).all() and th.abs().max() < 1e3


# ---------------- 折回 / 周期损失 / 边损失 ----------------

def test_wrap_angle_values_and_grad():
    x = torch.tensor([0.0, 3.0, 4.0, -4.0, 7.0, -7.0], requires_grad=True)
    y = wrap_angle(x)
    exp = torch.tensor([0.0, 3.0, 4.0 - 2 * np.pi, -4.0 + 2 * np.pi, 7.0 - 2 * np.pi, -7.0 + 2 * np.pi])
    assert torch.allclose(y, exp, atol=1e-6)
    assert (y.abs() <= np.pi + 1e-6).all()
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_pf_loss_periodic_vs_plain():
    k = PF_FIELDS.index("theta")
    n, F = 5, NODE_TABLE.offset_of("adjustable") + 1
    target = torch.zeros(n, F)
    lm = torch.zeros(n, F, dtype=torch.bool)
    lm[:, TH] = True
    pred = torch.zeros(n, len(PF_FIELDS))
    pred[:, k] = 2 * np.pi
    assert abs(float(pf_loss(pred, target, lm)) - 2 * np.pi) < 1e-5
    assert float(pf_loss(pred, target, lm, theta_periodic=True)) < 1e-5
    # 非 θ 字段不受周期开关影响
    lm2 = torch.zeros(n, F, dtype=torch.bool)
    lm2[:, NODE_TABLE.offset_of("v")] = True
    pred2 = torch.zeros(n, len(PF_FIELDS)); pred2[:, PF_FIELDS.index("v")] = 2 * np.pi
    assert abs(float(pf_loss(pred2, target, lm2, theta_periodic=True)) - 2 * np.pi) < 1e-5


def test_pf_edge_bias_loss_detects_shrinkage_and_ignores_zero_mean_noise():
    from gridfm.reason.pf_theta_edge import pf_edge_bias_loss
    rng = np.random.default_rng(0)
    n = 200
    ei = _rand_graph(n, 100, rng)
    phi = torch.from_numpy(rng.normal(0, 0.3, size=n)).float()     # |Δθ| < π,不触发折回
    dtrue = phi[ei[1]] - phi[ei[0]]
    assert dtrue.abs().max() < np.pi
    use = torch.ones(ei.size(1), dtype=torch.bool); avail = torch.ones(n, dtype=torch.bool)
    eb = torch.zeros(ei.size(1), dtype=torch.long)
    # 20% 均匀收缩 → 惩罚 ≈ 0.2·mean|Δθ|
    l_shrink = pf_edge_bias_loss(0.8 * dtrue, ei, use, phi, avail, eb, 1)
    assert abs(float(l_shrink) - 0.2 * float(dtrue.abs().mean())) < 1e-4
    # 零均值、与 Δθ 无关的噪声 → 惩罚 ≈ 0(按 1/√E 收敛)
    noise = torch.from_numpy(rng.normal(0, 0.05, size=ei.size(1))).float()
    l_noise = pf_edge_bias_loss(dtrue + noise, ei, use, phi, avail, eb, 1)
    assert float(l_noise) < 0.01
    # 两图各自平均,再对图取均值;掩码生效
    eb2 = (torch.arange(ei.size(1)) % 2).long()
    l2 = pf_edge_bias_loss(0.8 * dtrue, ei, use, phi, avail, eb2, 2)
    assert abs(float(l2) - float(l_shrink)) < 0.05
    use0 = torch.zeros_like(use)
    assert float(pf_edge_bias_loss(0.8 * dtrue, ei, use0, phi, avail, eb, 1)) == 0.0


def test_pf_edge_loss_wraps_target_and_masks():
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])
    theta_t = torch.tensor([0.0, 3.0, -3.0])            # 1→2 存档差 −6 ≈ 物理 +0.283
    avail = torch.tensor([True, True, True])
    use = torch.tensor([True, True, False])
    delta = torch.tensor([3.0, 2 * np.pi - 6.0, 100.0])  # 第三条边被 use 掩掉
    assert float(pf_edge_loss(delta, ei, use, theta_t, avail)) < 1e-5
    avail2 = torch.tensor([True, False, True])
    assert float(pf_edge_loss(torch.tensor([3.0, 0.0, 0.0]), ei, use, theta_t, avail2)) < 1e-5


# ---------------- 推理器结构 / 梯度 / 配对 ----------------

def _frozen_tiny():
    cfg = TrainConfig(d=32, n_layers=2, n_global=4, seed=0)
    enc, _dec = build_models(cfg)
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    return enc, cfg


def _batch_for(case_dir, cfg, sids=("s0", "s1")):
    datas, raws, targets, lms = [], [], [], []
    for sid in sids:
        raw = _raw(case_dir, sid)
        s = _norm(case_dir, sid)
        view, target, lm = pf_input_view(s, view="group")
        data = sample_to_data(view, cfg.lape_k, cfg.rwse_steps, random_dim=cfg.random_dim,
                              branch_random_dim=cfg.branch_random_dim)
        d = dc_theta(raw)
        side = np.concatenate([pf_side_features(s), d], axis=1)
        data.side = torch.from_numpy(side)
        es, eu = pf_edge_side_features(raw, d)
        tr, rm, reach, avail = pf_theta_ref(raw, d)
        data.edge_side, data.edge_use = torch.from_numpy(es), torch.from_numpy(eu)
        data.theta_ref, data.ref_mask = torch.from_numpy(tr), torch.from_numpy(rm)
        data.reach_mask, data.theta_avail = torch.from_numpy(reach), torch.from_numpy(avail)
        data.target, data.loss_mask = torch.from_numpy(target), torch.from_numpy(lm)
        datas.append(data)
    return Batch.from_data_list(datas)


def test_reasoner_theta_edge_forward_grad_and_wrap(case_dir):
    torch.manual_seed(0)
    enc, cfg = _frozen_tiny()
    batch = _batch_for(case_dir, cfg)
    h, g, e = enc(batch)
    eb = batch.batch[batch.edge_index[0]]
    r = PFReasoner(d=cfg.d, n_queries=4, n_layers=1, side_dim=PF_SIDE_DIM + 2, gnn_layers=1,
                   theta_head=True, edge_side_dim=EDGE_SIDE_DIM, theta_edge=True, theta_wrap=True)
    pred, delta = r(h, g, e, batch.batch, eb, side=batch.side, edge_index=batch.edge_index,
                    edge_side=batch.edge_side, edge_use=batch.edge_use, theta_ref=batch.theta_ref,
                    ref_mask=batch.ref_mask, reach_mask=batch.reach_mask, return_edge=True)
    k = PF_FIELDS.index("theta")
    assert pred.shape == (batch.num_nodes, len(PF_FIELDS)) and delta.shape == (batch.edge_index.size(1),)
    assert (pred[:, k].abs() <= np.pi + 1e-5).all(), "theta_wrap 输出须在 (−π, π]"
    # 平衡机(参考节点)θ 被势重建钉在给定角
    rm = batch.ref_mask
    assert torch.allclose(pred[rm, k], wrap_angle(batch.theta_ref[rm]), atol=1e-4)
    loss = pf_loss(pred, batch.target, batch.loss_mask, theta_periodic=True) \
        + pf_edge_loss(delta, batch.edge_index, batch.edge_use, batch.target[:, TH], batch.theta_avail)
    loss.backward()
    assert all(p.grad is None for p in enc.parameters()), "冻结编码器不得有梯度(D13)"
    for name in ("edge_head.0.weight", "edge_side_proj.weight", "head.0.weight"):
        p = dict(r.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name
    # 直接 θ 头只服务无参考角的孤岛节点;118 的 θ 监督节点全部可达(不可达的只有 NULL 占位行)
    # → 其梯度恒 0(势重建完全接管 θ)
    assert bool(batch.reach_mask[batch.loss_mask[:, TH]].all())
    g_th = dict(r.named_parameters())["th_head.0.weight"].grad
    assert g_th is None or g_th.abs().sum() == 0
    # 配对判错
    with pytest.raises(ValueError):
        r(h, g, e, batch.batch, eb, side=batch.side, edge_index=batch.edge_index)
    r0 = PFReasoner(d=cfg.d, n_queries=4, n_layers=1, side_dim=PF_SIDE_DIM + 2, theta_head=True)
    with pytest.raises(ValueError):
        r0(h, g, e, batch.batch, eb, side=batch.side, edge_index=batch.edge_index,
           edge_side=batch.edge_side)
    assert not any(n.startswith("edge_") or n.startswith("ln_e") for n, _ in r0.named_parameters())


def _legacy_forward(r, h, g, e, node_batch, edge_batch, side, edge_index):
    """2026-08-21 改动前 PFReasoner.forward 的逐行参考实现(同一组子模块),用于逐位回归。"""
    from torch_geometric.utils import to_dense_batch
    if r.side_proj is not None:
        h = h + r.side_proj(side)
    for mp in r.mp:
        h = mp(h, e, edge_index)
    hd, hmask = to_dense_batch(h, node_batch)
    ed, emask = to_dense_batch(e, edge_batch)
    ones = torch.ones(g.shape[:2], dtype=torch.bool, device=g.device)
    arch = torch.cat([hd, ed, g], dim=1)
    kpm = ~torch.cat([hmask, emask, ones], dim=1)
    q = r.queries.unsqueeze(0).expand(hd.size(0), -1, -1)
    for layer in r.layers:
        q = layer(q, arch, kpm)
    ctx = r.node_cross(r.ln_node(hd), q, q, need_weights=False)[0]
    feat = torch.cat([h, ctx[hmask]], dim=-1)
    pred = r.head(feat)
    if r.th_head is not None:
        k = PF_FIELDS.index("theta")
        pred = torch.cat([pred[:, :k], r.th_head(feat)], dim=-1)
    return pred


@pytest.mark.parametrize("theta_head,gnn_layers", [(True, 2), (False, 0), (True, 0)])
def test_reasoner_legacy_config_bitwise_unchanged(case_dir, theta_head, gnn_layers):
    """theta_edge/theta_wrap 关闭时,新 forward 与改动前的参考实现逐位同输出(旧检查点兼容)。"""
    torch.manual_seed(0)
    enc, cfg = _frozen_tiny()
    batch = _batch_for(case_dir, cfg, sids=("s0", "s1"))
    h, g, e = enc(batch)
    eb = batch.batch[batch.edge_index[0]]
    r = PFReasoner(d=cfg.d, n_queries=4, n_layers=1, side_dim=PF_SIDE_DIM + 2, gnn_layers=gnn_layers,
                   theta_head=theta_head)
    keys = set(r.state_dict().keys())
    assert not any(k.startswith(("edge_", "ln_e", "ln_he")) for k in keys)
    ei = batch.edge_index if gnn_layers else None
    with torch.no_grad():
        p1 = r(h, g, e, batch.batch, eb, side=batch.side, edge_index=ei)
        p_ref = _legacy_forward(r, h, g, e, batch.batch, eb, batch.side, ei)
    assert p1.shape == (batch.num_nodes, 4)
    assert torch.equal(p1, p_ref), (p1 - p_ref).abs().max()
    assert not r.theta_edge and not r.theta_wrap


def test_training_script_pipeline_edge_path(case_dir):
    """scripts/pf_reasoner.py 的 _pf_data(edge=) → Batch → _forward → eval_pu_mae 在 theta_edge 推理器上跑通,
    且 edge 属性在拼批后与 edge_index 对齐(审查:tests/ 原不覆盖 scripts/)。"""
    import importlib.util
    import sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "pf_reasoner_script", Path(__file__).resolve().parent.parent / "scripts" / "pf_reasoner.py")
    M = importlib.util.module_from_spec(spec)
    sys.modules["pf_reasoner_script"] = M
    spec.loader.exec_module(M)
    torch.manual_seed(0)
    enc, cfg = _frozen_tiny()
    raws = [_raw(case_dir, "s0"), _raw(case_dir, "s1")]
    norm = Normalizer().fit(raws)
    samples = []
    for r in raws:
        import copy
        s = copy.deepcopy(r); norm.transform(s); samples.append(s)
    datas = [M._pf_data(s, cfg, "group", raw=r, dc="raw", phys=True, edge=True) for s, r in zip(samples, raws)]
    batch = Batch.from_data_list(datas)
    E = batch.edge_index.size(1)
    assert batch.edge_side.shape == (E, EDGE_SIDE_DIM) and batch.edge_use.shape == (E,)
    assert batch.theta_ref.shape == (batch.num_nodes,) and batch.reach_mask.shape == (batch.num_nodes,)
    # Δθ_dc 列与按拼批后 edge_index 重算的 θ_dc 差一致(行对齐)
    dcs = torch.from_numpy(np.concatenate([dc_theta(r) for r in raws]))
    src, dst = batch.edge_index
    ok = batch.edge_use & (dcs[src, 1] > 0.5) & (dcs[dst, 1] > 0.5)
    assert torch.allclose(batch.edge_side[ok, DTH_DC_COL], (dcs[dst, 0] - dcs[src, 0])[ok], atol=1e-5)
    r = PFReasoner(d=cfg.d, n_queries=4, n_layers=1, side_dim=PF_SIDE_DIM + 2, gnn_layers=1,
                   theta_head=True, edge_side_dim=EDGE_SIDE_DIM, theta_edge=True, theta_wrap=True)
    dev = torch.device("cpu")
    pred, loss = M._forward(enc, r, batch, False, torch.float32, dev, field_weights={"theta": 4.0},
                            edge_weight=1.0)
    assert pred.shape == (batch.num_nodes, 4) and torch.isfinite(loss)
    loss.backward()
    assert dict(r.named_parameters())["edge_head.0.weight"].grad.abs().sum() > 0
    mae = M.eval_pu_mae(enc, r, samples, cfg, norm, dev, False, torch.float32, bs=2, view="group",
                        raws=raws, dn=None, dc="raw", phys=False, edge=True)
    assert all(np.isfinite(mae[f]) for f in PF_FIELDS) and mae["theta"] <= np.pi + 1e-6


def test_reasoner_edge_head_local_forward_grad_and_independence(case_dir):
    """E4 局部物理边头:前向/梯度通路;δ 不依赖节点 token(h 扰动不改 δ),依赖边侧特征与两端给定量。"""
    torch.manual_seed(0)
    enc, cfg = _frozen_tiny()
    batch = _batch_for(case_dir, cfg)
    h, g, e = enc(batch)
    eb = batch.batch[batch.edge_index[0]]
    r = PFReasoner(d=cfg.d, n_queries=4, n_layers=1, side_dim=PF_SIDE_DIM + 2, gnn_layers=1,
                   theta_head=True, edge_side_dim=EDGE_SIDE_DIM, theta_edge=True, theta_wrap=True,
                   edge_head_local=True)
    kw = dict(side=batch.side, edge_index=batch.edge_index, edge_side=batch.edge_side, edge_use=batch.edge_use,
              theta_ref=batch.theta_ref, ref_mask=batch.ref_mask, reach_mask=batch.reach_mask, return_edge=True)
    pred, delta = r(h, g, e, batch.batch, eb, **kw)
    assert pred.shape == (batch.num_nodes, 4) and delta.shape == (batch.edge_index.size(1),)
    loss = pf_loss(pred, batch.target, batch.loss_mask, theta_periodic=True) \
        + pf_edge_loss(delta, batch.edge_index, batch.edge_use, batch.target[:, TH], batch.theta_avail)
    loss.backward()
    P = dict(r.named_parameters())
    for name in ("edge_head.0.weight", "edge_side_proj.weight", "loc_node_proj.weight"):
        assert P[name].grad is not None and P[name].grad.abs().sum() > 0, name
    assert not any(k.startswith(("ln_e.", "ln_he.")) for k in P)      # 非局部头的模块不存在
    # δ 只通过 V̂(停梯度)间接依赖节点 token:扰动 h 改变 V̂ → 允许 δ 变;但把 V̂ 固定时 δ 不变。
    # 等价检验:同一 side/edge_side 下,两次前向 δ 一致(确定性),且 δ 对 h 的梯度为 0(V̂ 已 detach)
    h2 = h.detach().clone().requires_grad_(True)
    pred2, delta2 = r(h2, g, e, batch.batch, eb, **kw)
    delta2.sum().backward()
    assert h2.grad is None or h2.grad.abs().sum() == 0, "局部边头的 δ 不得对节点 token 有梯度"
    # 绝对角列被屏蔽:改动 side 的 theta 列/known 位不改 δ
    side_p = batch.side.clone()
    k_th = STATE_FIELDS.index("theta"); k = len(STATE_FIELDS)
    side_p[:, k_th] += 1.0; side_p[:, k + k_th] = 1.0
    with torch.no_grad():
        _, delta3 = r(h, g, e, batch.batch, eb, **{**kw, "side": side_p})
    # side 也进 side_proj → h → V̂ 会变,故只比较"V̂ 相同"的情形:用 no_grad 下把 th_head/head 固定不现实,
    # 改为直接检验 side_keep 掩码:屏蔽列权重不参与
    assert float(r.side_keep[k_th]) == 0.0 and float(r.side_keep[k + k_th]) == 0.0
    assert torch.isfinite(delta3).all()


def test_training_script_split_holdout(case_dir):
    """库内留出:按 sample_id 分池,顺序保持,未命中的全留训练池。"""
    import importlib.util, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "pf_reasoner_script2", Path(__file__).resolve().parent.parent / "scripts" / "pf_reasoner.py")
    M = importlib.util.module_from_spec(spec); sys.modules["pf_reasoner_script2"] = M; spec.loader.exec_module(M)
    samples = [_raw(case_dir, f"s{i}") for i in range(5)]
    keep, held = M.split_holdout(samples, {"s1", "s3", "s9"})
    assert [s.sample_id for s in keep] == ["s0", "s2", "s4"] and [s.sample_id for s in held] == ["s1", "s3"]
    keep2, held2 = M.split_holdout(samples, set())
    assert len(keep2) == 5 and held2 == []


def test_reasoner_theta_edge_overfits_single_sample(case_dir):
    """单样本 150 步:边损失 + 周期节点损失实质性下降(可训性冒烟)。"""
    torch.manual_seed(0)
    enc, cfg = _frozen_tiny()
    batch = _batch_for(case_dir, cfg, sids=("s0",))
    with torch.no_grad():
        h, g, e = enc(batch)
    eb = batch.batch[batch.edge_index[0]]
    r = PFReasoner(d=cfg.d, n_queries=4, n_layers=1, side_dim=PF_SIDE_DIM + 2,
                   theta_head=True, edge_side_dim=EDGE_SIDE_DIM, theta_edge=True, theta_wrap=True)
    opt = torch.optim.Adam(r.parameters(), lr=3e-3)
    first = last = None
    for _ in range(150):
        opt.zero_grad()
        pred, delta = r(h, g, e, batch.batch, eb, side=batch.side, edge_index=batch.edge_index,
                        edge_side=batch.edge_side, edge_use=batch.edge_use, theta_ref=batch.theta_ref,
                        ref_mask=batch.ref_mask, reach_mask=batch.reach_mask, return_edge=True)
        loss = pf_loss(pred, batch.target, batch.loss_mask, theta_periodic=True) \
            + pf_edge_loss(delta, batch.edge_index, batch.edge_use, batch.target[:, TH], batch.theta_avail)
        loss.backward(); opt.step()
        first = first if first is not None else float(loss)
        last = float(loss)
    assert last < 0.7 * first, f"未收敛:{first:.4f} -> {last:.4f}"


# ---------------------------------------------------------------------------
# 2026-08-21 第 2 轮修法(结果档 §3.11 第 3~4 项)的行为锁定
# ---------------------------------------------------------------------------

def test_cpu_lu_no_unconditional_ridge_on_referenced_component():
    """修法 3:有参考角的分量必须拿到**精确**解,不再被无条件的 1e-12·I 掺一点收缩偏差。

    构造长路径图(有效电阻 ~N²,微岭偏差最放大的形状),与 scipy 的无岭直接解逐位比。
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    from gridfm.reason import pf_theta_edge as PE

    n = 1500
    src = torch.arange(n - 1)
    dst = torch.arange(1, n)
    ei = torch.stack([src, dst])
    phi = torch.from_numpy(np.linspace(0.0, 50.0, n))          # 末端 50 rad
    delta = (phi[dst] - phi[src]).float()
    rm = torch.zeros(n, dtype=torch.bool); rm[0] = True
    ref = torch.where(rm, phi.float(), torch.zeros(n))
    zr = torch.zeros(n, dtype=torch.bool)
    th = potential_reconstruct(delta, ei, torch.ones(n - 1), torch.zeros(n, dtype=torch.long),
                               ref, rm, 1, ridge_mask=zr, mode="sparse")
    s, t = src.numpy(), dst.numpy()
    o = np.ones(s.size)
    L = sp.coo_matrix((np.r_[o, o, -o, -o], (np.r_[s, t, s, t], np.r_[s, t, t, s])), shape=(n, n)).tocsr()
    b = np.zeros(n)
    np.add.at(b, t, delta.double().numpy()); np.add.at(b, s, -delta.double().numpy())
    free = np.arange(1, n)
    x = np.zeros(n); x[0] = float(phi[0])
    x[free] = spsolve(L[free][:, free].tocsc(), b[free] - L[free][:, [0]].toarray().ravel() * x[0])
    err = float(np.abs(th.double().numpy() - x).max())
    # 参考:1e-6 的岭在同形状上偏 0.02~0.2 rad(§5b);1e-12 的岭偏 ~1e-7。去掉后只剩 fp32 输出分辨率。
    assert err < 1e-4, f"有参考分量应为精确解,实测最大偏差 {err:.2e}"
    assert PE.SPARSE_SOLVER == "splu", "本测试针对 CPU 直接 LU 路径"


def test_cpu_lu_still_solves_singular_component_without_reference():
    """修法 3 的兜底:无参考角**且**调用方没给岭 → 系统奇异,仍须给出有界解(不得抛异常)。"""
    n = 40
    ei = _rand_graph(n, 10, np.random.default_rng(3))
    delta = torch.randn(ei.size(1))
    th = potential_reconstruct(delta, ei, torch.ones(ei.size(1)), torch.zeros(n, dtype=torch.long),
                               torch.zeros(n), torch.zeros(n, dtype=torch.bool), 1,
                               ridge_mask=torch.zeros(n, dtype=torch.bool), mode="sparse")
    assert torch.isfinite(th).all() and th.abs().max() < 1e6


def test_auto_dispatch_uses_batch_elems_not_just_nmax():
    """修法 4:auto 分流按 **整批** B·nmax² 判,不只看单图 nmax;两条路径同解。

    单图 2000 节点(< DENSE_MAX_NODES 4096)但 batch 8 → 32e6 元素 > 上限,应转稀疏;
    同一批用 batch 2(8e6)则仍走稠密。两者与显式 dense/sparse 结果一致。
    """
    from gridfm.reason import pf_theta_edge as PE

    rng = np.random.default_rng(11)
    n = 2000
    assert n <= PE.DENSE_MAX_NODES and 8 * n * n > PE.DENSE_MAX_BATCH_ELEMS > 2 * n * n

    def _batch(B):
        eis, ds, nbs, refs, rms, zrs = [], [], [], [], [], []
        for g in range(B):
            ei = _rand_graph(n, 300, rng)
            phi = torch.from_numpy(np.cumsum(rng.normal(0, 0.02, size=n)))
            d = (phi[ei[1]] - phi[ei[0]]).float()
            rm = torch.zeros(n, dtype=torch.bool); rm[0] = True
            eis.append(ei + g * n); ds.append(d); nbs.append(torch.full((n,), g, dtype=torch.long))
            refs.append(torch.where(rm, phi.float(), torch.zeros(n))); rms.append(rm)
            zrs.append(torch.zeros(n, dtype=torch.bool))
        return (torch.cat(ds), torch.cat(eis, 1), torch.cat(nbs), torch.cat(refs),
                torch.cat(rms), torch.cat(zrs), B)

    d, ei, nb, ref, rm, zr, B = _batch(8)
    w = torch.ones(ei.size(1))
    auto = potential_reconstruct(d, ei, w, nb, ref, rm, B, ridge_mask=zr, mode="auto")
    spa = potential_reconstruct(d, ei, w, nb, ref, rm, B, ridge_mask=zr, mode="sparse")
    den = potential_reconstruct(d, ei, w, nb, ref, rm, B, ridge_mask=zr, mode="dense")
    assert torch.equal(auto, spa), "batch 8 × 2000 节点应走稀疏路径"
    assert torch.allclose(den, spa, atol=1e-5), (den - spa).abs().max()

    d, ei, nb, ref, rm, zr, B = _batch(2)
    w = torch.ones(ei.size(1))
    auto2 = potential_reconstruct(d, ei, w, nb, ref, rm, B, ridge_mask=zr, mode="auto")
    den2 = potential_reconstruct(d, ei, w, nb, ref, rm, B, ridge_mask=zr, mode="dense")
    assert torch.equal(auto2, den2), "batch 2 × 2000 节点(8e6 元素)应仍走稠密路径"


# ---------------------------------------------------------------------------
# E6 · 失配修正步(预注册 §13.1)
# ---------------------------------------------------------------------------

def test_mismatch_correction_recovers_dc_solution():
    """`mismatch_correction` 解的确实是 B′Δθ = ΔP(参考节点 Δθ=0):

    构造已知的 θ*、由它生成"注入" P = B′θ*,从 θ=0 出发一步修正应**精确**回到 θ*
    (DC 方程是线性的,一步 Newton 即精确解)。
    """
    from gridfm.reason.pf_theta_edge import mismatch_correction

    rng = np.random.default_rng(5)
    n = 400
    ei = _rand_graph(n, 250, rng)
    b = torch.from_numpy(rng.uniform(2.0, 40.0, size=ei.size(1)))     # 1/x 量级
    phi = torch.from_numpy(rng.normal(0, 0.3, size=n))
    phi = phi - phi[0]                                                # 参考节点角 = 0
    src, dst = ei[0], ei[1]
    diff = b * (phi[src] - phi[dst])
    p = torch.zeros(n, dtype=torch.float64).index_add_(0, src, diff).index_add_(0, dst, -diff)
    rm = torch.zeros(n, dtype=torch.bool); rm[0] = True
    p = torch.where(rm, torch.zeros_like(p), p)                       # 平衡机方程不参与
    for mode in ("dense", "sparse"):
        dth = mismatch_correction(p, ei, b, torch.zeros(n, dtype=torch.long), rm, 1,
                                  ridge_mask=torch.zeros(n, dtype=torch.bool), mode=mode)
        err = float((dth.double() - phi).abs().max())
        assert err < 1e-4, f"{mode} 路径应精确回收 θ*,实测偏差 {err:.2e}"


def test_mismatch_correction_grad_and_batched():
    """对 ΔP 可微(隐式微分),且多图拼批时逐图独立。"""
    from gridfm.reason.pf_theta_edge import mismatch_correction

    rng = np.random.default_rng(6)
    n = 120
    eis, ps, nbs, rms = [], [], [], []
    for g in range(3):
        ei = _rand_graph(n, 60, rng)
        eis.append(ei + g * n)
        ps.append(torch.from_numpy(rng.normal(0, 0.1, size=n)))
        nbs.append(torch.full((n,), g, dtype=torch.long))
        rm = torch.zeros(n, dtype=torch.bool); rm[0] = True
        rms.append(rm)
    ei = torch.cat(eis, 1); nb = torch.cat(nbs); rm = torch.cat(rms)
    w = torch.ones(ei.size(1), dtype=torch.float64)
    p = torch.cat(ps).requires_grad_(True)
    zr = torch.zeros(3 * n, dtype=torch.bool)
    dth = mismatch_correction(p, ei, w, nb, rm, 3, ridge_mask=zr, mode="dense")
    dth[:n].sum().backward()
    assert torch.isfinite(p.grad).all()
    assert float(p.grad[n:].abs().max()) == 0.0, "图 0 的输出不应对图 1/2 的 ΔP 有梯度"
    # dense / sparse 同解
    p2 = torch.cat(ps)
    a = mismatch_correction(p2, ei, w, nb, rm, 3, ridge_mask=zr, mode="dense")
    b2 = mismatch_correction(p2, ei, w, nb, rm, 3, ridge_mask=zr, mode="sparse")
    assert torch.allclose(a, b2, atol=1e-5), (a - b2).abs().max()


def test_edge_susceptance_layout_and_leak_free(case_dir):
    """b_k = 1/x 的布局与对称化边同序;扰动被监督量不改变它(D15)。"""
    from dataclasses import replace

    from gridfm.reason.pf_theta_edge import pf_edge_susceptance

    raw = _raw(case_dir)
    b = pf_edge_susceptance(raw)
    m = raw.n_branches
    assert b.shape == (2 * m,)
    np.testing.assert_array_equal(b[:m], b[m:])          # 正反向同值
    use = use_edges(raw)
    assert np.all(b[:m][~use] == 0.0) and np.all(b[:m][use] > 0.0)
    nf = raw.node_feat.copy()
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    for i in range(raw.n_nodes):
        for f in PF_UNKNOWN_FIELDS[int(types[i])]:
            nf[i, NODE_TABLE.offset_of(f)] += 7.7
    np.testing.assert_array_equal(b, pf_edge_susceptance(replace(raw, node_feat=nf)))


def test_reasoner_refine_steps_forward_grad_and_off_is_bitwise(case_dir):
    """推理器接线:refine_steps>0 时前向/反向可用、α 有梯度;refine_steps=0 与旧结构逐位一致。
    2026-08-22 第 3 轮起改用带 sidecar 变比的 mp_case118 夹具:golden 无 tap → 方程覆盖率 < 99% → E12 的覆盖率闸会让修正步成为空操作
    (那条行为在 tests/test_pf_refine_e12.py 单独锁定)。"""
    import copy as _copy
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "scripts"))
    import pf_reasoner as M
    from gridfm.reason.pf_physics import pf_phys_denorm
    from gridfm.schema.sample import GridSample as _GS

    _fx = _P(__file__).parent / "fixtures" / "mp_case118_mode"
    raw = _GS.load_npz(_fx / "mp_case118_00000.npz")
    _z = np.load(_fx / "family_extra.npz")
    raw.branch_tap, raw.branch_shift_rad = _z["branch_tap"], _z["branch_shift_rad"]
    norm = Normalizer().fit([raw])
    s = _copy.deepcopy(raw); norm.transform(s)
    cfg = TrainConfig()
    data = M._pf_data(s, cfg, view="group", raw=raw, dc="lossy", phys=True, edge=True)
    batch = Batch.from_data_list([data])
    enc, _ = build_models(cfg)
    enc.eval()
    dn = torch.from_numpy(pf_phys_denorm(norm))
    kw = dict(d=cfg.d, n_queries=4, n_layers=1, side_dim=batch.side.shape[1],
              gnn_layers=1, theta_head=True, edge_side_dim=EDGE_SIDE_DIM,
              theta_edge=True, theta_wrap=True, edge_head_local=True)
    torch.manual_seed(0)
    r0 = PFReasoner(**kw)
    torch.manual_seed(0)
    r1 = PFReasoner(**kw, refine_steps=1)
    r1.load_state_dict(r0.state_dict(), strict=False)

    p0, _ = M._forward(enc, r0, batch, False, torch.float32, torch.device("cpu"), dn=dn)
    with torch.no_grad():
        r1.refine_alpha.zero_()                       # α=0 → 修正步不动 θ,应与关闭逐位一致
    p1, _ = M._forward(enc, r1, batch, False, torch.float32, torch.device("cpu"), dn=dn)
    assert torch.equal(p0, p1), (p0 - p1).abs().max()

    with torch.no_grad():
        r1.refine_alpha.fill_(1.0)
    p2, loss2 = M._forward(enc, r1, batch, False, torch.float32, torch.device("cpu"), dn=dn)
    assert torch.isfinite(p2).all() and not torch.equal(p0, p2), "α=1 时修正步必须改变 θ"
    loss2.backward()
    assert r1.refine_alpha.grad is not None and torch.isfinite(r1.refine_alpha.grad).all()
    assert float(r1.refine_alpha.grad.abs().sum()) > 0, "步长 α 必须拿到梯度"
