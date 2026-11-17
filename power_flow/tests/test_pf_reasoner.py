"""潮流任务推理器:D15 输入协议泄漏单测(方案 5.2 强制)+ 结构/训练冒烟。

泄漏单测是路线红线:潮流任务编码器输入张量中,被监督字段必须全部为缺失标记(清零),
真值只出现在损失侧;支路潮流(解算产物)不得留在输入里。
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch

from gridfm.data.graph import feature_slices, sample_to_data
from gridfm.data.normalize import Normalizer
from gridfm.psasp_io.case import PsaspCase
from gridfm.psasp_io.catalog import Catalog
from gridfm.reason.pf_task import (
    BRANCH_SOLVED_FIELDS, PF_FIELDS, PF_SIDE_DIM, PFReasoner, STATE_FIELDS,
    pf_input_view, pf_loss, pf_side_features, pf_zero_learning_baseline,
)
from gridfm.schema.assemble import build_sample
from gridfm.schema.fields import BRANCH_TABLE, MASKABLE_GROUPS, NODE_TABLE, Group
from gridfm.schema.sample import group_bit_index
from gridfm.train.masking import COL_GROUP, PF_UNKNOWN_FIELDS
from gridfm.train.pretrain import TrainConfig, build_models


def _sample(case_dir, sid="s0"):
    c = PsaspCase(case_dir)
    s = build_sample(c, Catalog(c), sample_id=sid, family_id="t")
    Normalizer().fit([s]).transform(s)
    return s


# ---------------- D15 泄漏单测(强制) ----------------

def test_pf_view_masks_unknowns_by_type(case_dir):
    s = _sample(case_dir)
    view, target, lm = pf_input_view(s)
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(s.node_feat[:, tcol:tcol + 3], axis=1)
    sbit = group_bit_index(Group.STATE)
    null = s.node_feat[:, NODE_TABLE.offset_of("is_null")] > 0.5
    n_checked = 0
    for i in range(s.n_nodes):
        if null[i] or s.node_missing[i, sbit]:
            assert not lm[i].any(), "NULL/missing 节点不得进监督"
            continue
        unknowns = PF_UNKNOWN_FIELDS[int(types[i])]
        for f in ("pg", "qg", "pl", "ql", "v", "theta", "v_set"):
            c = NODE_TABLE.offset_of(f)
            if f in unknowns:
                assert view.node_feat[i, c] == 0.0, f"未知量 {f} 未清零(节点 {i})"
                assert lm[i, c], f"未知量 {f} 未进损失掩码"
                n_checked += 1
            else:
                assert view.node_feat[i, c] == s.node_feat[i, c], f"给定量 {f} 被误动"
                assert not lm[i, c]
    assert n_checked > 0
    np.testing.assert_array_equal(target, s.node_feat)


def test_pf_view_zeroes_branch_flows(case_dir):
    """支路潮流是解算产物,留在输入 = 从流量反推 V/θ 的后门。"""
    s = _sample(case_dir)
    view, _t, _lm = pf_input_view(s)
    for f in BRANCH_SOLVED_FIELDS:
        assert (view.branch_feat[:, BRANCH_TABLE.offset_of(f)] == 0.0).all(), f"{f} 未清零"
    c = BRANCH_TABLE.offset_of("mark")
    np.testing.assert_array_equal(view.branch_feat[:, c], s.branch_feat[:, c])


def test_pf_encoder_input_tensor_leak_free(case_dir):
    """方案 5.2 原文:潮流任务编码器输入张量中,被监督字段全部为缺失标记。"""
    s = _sample(case_dir)
    view, _t, lm = pf_input_view(s)
    x = sample_to_data(view, 16, 16).x.numpy()
    sl = feature_slices(NODE_TABLE.total_dim, 16, 16, 8)
    feat = x[:, sl["feat"]]
    assert (feat[lm] == 0.0).all(), "编码器输入张量在被监督位置必须为 0"
    # 随机特征同源(视图不得重抽身份标签)
    np.testing.assert_array_equal(view.random_features(), s.random_features())


def test_pf_view_target_only_in_loss_side(case_dir):
    """被监督位置的真值不得以任何列出现在视图特征里(逐列断言,防换列藏值)。"""
    s = _sample(case_dir)
    view, target, lm = pf_input_view(s)
    rows = np.where(lm.any(axis=1))[0]
    for i in rows:
        for c in np.where(lm[i])[0]:
            tv = target[i, c]
            if abs(tv) < 1e-9:
                continue                       # 真值本身为 0 无从判泄漏
            assert not np.isclose(view.node_feat[i], tv).any() or \
                np.isclose(s.node_feat[i], tv).sum() > 1, \
                f"节点 {i} 被监督值 {tv} 疑似泄漏进输入行"


# ---------------- A 口径(group 视图 + 推理器侧给定量;07-11 裁决点 3 对照) ----------------

def test_pf_group_view_matches_pretrain_group_dropout(case_dir):
    """A 口径节点侧花样 = 预训练组缺失课程(c_topo)同款——分布内主张的逐列锁定;
    且与 B 口径 target/loss_mask 逐位一致(两口径结果直接可比)。

    2026-07-11 评审加固:直调 masking.apply_node_mask 活代码(而非转写快照),
    使预训练组缺失分支演化时本锁定不会静默失效。"""
    from gridfm.train.masking import MaskConfig, apply_node_mask

    s = _sample(case_dir)
    vf, tf, lf = pf_input_view(s, view="field")
    vg, tg, lg = pf_input_view(s, view="group")
    np.testing.assert_array_equal(tf, tg)
    np.testing.assert_array_equal(lf, lg)
    # 直调活代码:p_group_dropout=1.0 → u∈[0,1)<1.0 恒真,确定性走整组缺失分支
    rng = np.random.default_rng(0)
    mf, _t, _lm = apply_node_mask(s.node_feat, s.node_missing,
                                  MaskConfig(p_group_dropout=1.0), rng)
    np.testing.assert_array_equal(vg.node_feat, mf)
    # 支路潮流仍清零;随机特征同源(不得重抽)
    for f in BRANCH_SOLVED_FIELDS:
        assert (vg.branch_feat[:, BRANCH_TABLE.offset_of(f)] == 0.0).all(), f"{f} 未清零"
    np.testing.assert_array_equal(vg.random_features(), s.random_features())


def test_pf_group_view_null_not_missing_corner(case_dir):
    """覆盖缺口(2026-07-11 评审):is_null=1 且 STATE 非 missing 的合成角。
    此角能区分'正确组缺失'与'误加 _pf_typed_mask 式 null 豁免'的错误实现——
    fixture 天然无此行(3 个 null 全同时 STATE-missing),故人造之。"""
    from dataclasses import replace

    from gridfm.train.masking import MaskConfig, apply_node_mask

    s = _sample(case_dir)
    sbit = group_bit_index(Group.STATE)
    nf = s.node_feat.copy()
    nm = s.node_missing.copy()
    i = int(np.argmax(~nm[:, sbit]))              # 取一个 STATE 非 missing 的节点
    assert not nm[i, sbit]
    nf[i, NODE_TABLE.offset_of("is_null")] = 1.0  # 强制其为 NULL 占位
    s2 = replace(s, node_feat=nf, node_missing=nm)
    vg, _t, _lm = pf_input_view(s2, view="group")
    rng = np.random.default_rng(0)
    mf, _t2, _lm2 = apply_node_mask(nf, nm, MaskConfig(p_group_dropout=1.0), rng)
    # 组缺失不看 is_null(仅豁免 missing)→ 该 NULL 行的 STATE 列仍被清零,与活代码一致
    np.testing.assert_array_equal(vg.node_feat, mf)
    for c in [NODE_TABLE.offset_of(f) for f in ("pg", "v", "theta")]:
        assert vg.node_feat[i, c] == 0.0, "NULL-非missing 行的 STATE 列应被组缺失清零"


def test_pf_group_view_hides_all_state_from_encoder(case_dir):
    """A 口径编码器输入张量:非 missing 节点的 STATE 组全列(含给定量)必须为 0。"""
    s = _sample(case_dir)
    vg, _t, _lm = pf_input_view(s, view="group")
    x = sample_to_data(vg, 16, 16).x.numpy()
    sl = feature_slices(NODE_TABLE.total_dim, 16, 16, 8)
    feat = x[:, sl["feat"]]
    sbit = group_bit_index(Group.STATE)
    rows = ~s.node_missing[:, sbit]
    cols = [NODE_TABLE.offset_of(f) for f in STATE_FIELDS]
    assert (feat[np.ix_(rows, cols)] == 0.0).all()


def test_pf_side_features_leak_free(case_dir):
    """侧通道泄漏单测(D15 延伸,强制):类型化未知量的值与 known 位恒 0;
    给定量原样保留、known 位 = 1;STATE missing / NULL 行全 0。"""
    s = _sample(case_dir)
    side = pf_side_features(s)
    k = len(STATE_FIELDS)
    assert side.shape == (s.n_nodes, PF_SIDE_DIM) and PF_SIDE_DIM == 2 * k
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(s.node_feat[:, tcol:tcol + 3], axis=1)
    sbit = group_bit_index(Group.STATE)
    null = s.node_feat[:, NODE_TABLE.offset_of("is_null")] > 0.5
    n_known = 0
    for i in range(s.n_nodes):
        if null[i] or s.node_missing[i, sbit]:
            assert (side[i] == 0.0).all(), f"missing/NULL 行 {i} 侧通道应全 0"
            continue
        unknowns = PF_UNKNOWN_FIELDS[int(types[i])]
        for j, f in enumerate(STATE_FIELDS):
            if f in unknowns:
                assert side[i, j] == 0.0 and side[i, k + j] == 0.0, \
                    f"未知量 {f} 泄漏进侧通道(节点 {i})"
            else:
                assert side[i, j] == s.node_feat[i, NODE_TABLE.offset_of(f)]
                assert side[i, k + j] == 1.0
                n_known += 1
    assert n_known > 0


def test_pf_reasoner_side_channel_paired(case_dir):
    """side_dim 配置与 side 输入必须成对(防 A/B 口径静默混用);梯度经 side_proj 流动。"""
    import pytest
    torch.manual_seed(0)
    s = _sample(case_dir)
    view, target, lm = pf_input_view(s, view="group")
    enc, cfg = _frozen_tiny(case_dir)
    h, g, e, nb, eb = _encode_view(enc, cfg, view)
    side = torch.from_numpy(pf_side_features(s))

    r = PFReasoner(d=cfg.d, n_queries=4, n_layers=1, side_dim=PF_SIDE_DIM)
    pred = r(h, g, e, nb, eb, side=side)
    assert pred.shape == (s.n_nodes, len(PF_FIELDS))
    pf_loss(pred, torch.from_numpy(target), torch.from_numpy(lm)).backward()
    assert r.side_proj.weight.grad is not None and r.side_proj.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        r(h, g, e, nb, eb)                               # group 口径漏供给定量
    r0 = PFReasoner(d=cfg.d, n_queries=4, n_layers=1)    # B 口径结构不变(旧检查点兼容)
    assert not any("side" in n for n, _ in r0.named_parameters())
    with pytest.raises(ValueError):
        r0(h, g, e, nb, eb, side=side)                   # field 口径重复供


# ---------------- 结构与训练冒烟 ----------------

def _frozen_tiny(case_dir):
    cfg = TrainConfig(d=32, n_layers=2, n_global=4, seed=0)
    enc, _dec = build_models(cfg)
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    return enc, cfg


def _encode_view(enc, cfg, view):
    data = sample_to_data(view, cfg.lape_k, cfg.rwse_steps,
                          random_dim=cfg.random_dim,
                          branch_random_dim=cfg.branch_random_dim)
    batch = Batch.from_data_list([data])
    h, g, e = enc(batch)
    return h, g, e, batch.batch, batch.batch[batch.edge_index[0]]


def test_pf_reasoner_forward_shapes_and_frozen_encoder(case_dir):
    s = _sample(case_dir)
    view, target, lm = pf_input_view(s)
    enc, cfg = _frozen_tiny(case_dir)
    reasoner = PFReasoner(d=cfg.d, n_queries=4, n_layers=2)
    h, g, e, nb, eb = _encode_view(enc, cfg, view)
    pred = reasoner(h, g, e, nb, eb)
    assert pred.shape == (s.n_nodes, len(PF_FIELDS))
    loss = pf_loss(pred, torch.from_numpy(target), torch.from_numpy(lm))
    loss.backward()
    assert all(p.grad is None for p in enc.parameters()), "冻结编码器不得有梯度(D13)"
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in reasoner.parameters()), "推理器必须有梯度"


def test_pf_reasoner_overfits_single_sample(case_dir):
    """单样本 200 步损失实质性下降(端到端可训性冒烟)。

    阈值 0.65×:随机 tiny 编码器的 token 质量限制单样本记忆上限(实测平台 ~0.5×,
    600 步不再降),此测只证梯度通路与可训性,真实拟合力由训练脚本在 v1 上度量。"""
    torch.manual_seed(0)
    s = _sample(case_dir)
    view, target, lm = pf_input_view(s)
    enc, cfg = _frozen_tiny(case_dir)
    reasoner = PFReasoner(d=cfg.d, n_queries=4, n_layers=1)
    h, g, e, nb, eb = _encode_view(enc, cfg, view)
    h, g, e = h.detach(), g.detach(), e.detach()
    tgt, lmt = torch.from_numpy(target), torch.from_numpy(lm)
    opt = torch.optim.Adam(reasoner.parameters(), lr=3e-3)
    first = last = None
    for _ in range(200):
        opt.zero_grad()
        loss = pf_loss(reasoner(h, g, e, nb, eb), tgt, lmt)
        loss.backward(); opt.step()
        first = first if first is not None else float(loss)
        last = float(loss)
    assert last < 0.65 * first, f"未收敛:{first:.4f} -> {last:.4f}"


def test_pf_e2e_joint_training_updates_encoder(case_dir):
    """组3(端到端联训,审查 A1)梯度通路:编码器可训时任务损失反传进编码器且一步
    优化实际改变编码器权重;与 test_pf_reasoner_forward_shapes_and_frozen_encoder
    (冻结臂零梯度)成对,防两臂配置静默混淆。"""
    torch.manual_seed(0)
    s = _sample(case_dir)
    view, target, lm = pf_input_view(s)
    cfg = TrainConfig(d=32, n_layers=2, n_global=4, seed=0)
    enc, _dec = build_models(cfg)          # 不加载 v1、不冻结 = e2e 臂
    reasoner = PFReasoner(d=cfg.d, n_queries=4, n_layers=1)
    opt = torch.optim.Adam(list(reasoner.parameters()) + list(enc.parameters()), lr=1e-3)
    before = [p.detach().clone() for p in enc.parameters()]
    h, g, e, nb, eb = _encode_view(enc, cfg, view)
    loss = pf_loss(reasoner(h, g, e, nb, eb), torch.from_numpy(target), torch.from_numpy(lm))
    opt.zero_grad()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in enc.parameters()), "e2e 臂编码器必须收到梯度"
    opt.step()
    changed = sum(int(not torch.equal(b, p.detach()))
                  for b, p in zip(before, enc.parameters()))
    assert changed > 0, "优化一步后编码器权重必须变化"


def test_pf_zero_learning_baseline_finite(case_dir):
    c = PsaspCase(case_dir)
    raw = build_sample(c, Catalog(c), sample_id="s0", family_id="t")
    rep = pf_zero_learning_baseline([raw])
    for f in PF_FIELDS:
        for lvl in ("L0", "L1", "L2"):
            assert np.isfinite(rep[f][lvl]), f"{f}/{lvl} 非有限"
    # θ 只有平衡机可见,L2 插值仍须给出有限数
    assert rep["theta"]["L2"] >= 0.0
