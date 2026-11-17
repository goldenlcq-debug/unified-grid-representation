"""潮流定向缺失课程档(方案 3.4.1"按节点类型定向缺失"的字段级实现;2026-07-11 用户批准实验)。

核心保证:①字段映射与 D15 任务输入协议(schema.assemble._PF_UNKNOWNS)一致;
②注入量在该模式下保持可见(θ 可学 DC 物理——实验的全部动机);
③p_pf_typed=0 时掩码流与旧实现**逐位不变**(验证/门禁/阶梯依赖此保证,8.9)。
"""
from __future__ import annotations

import numpy as np

from gridfm.psasp_io.case import PsaspCase
from gridfm.psasp_io.catalog import Catalog
from gridfm.schema.assemble import _PF_UNKNOWNS, PQ, PV, SLACK, build_sample
from gridfm.schema.fields import NODE_TABLE, Group
from gridfm.schema.sample import group_bit_index
from gridfm.train.masking import PF_UNKNOWN_FIELDS, MaskConfig, apply_node_mask
from gridfm.train.pretrain import TrainConfig, _masked_data


def _sample(case_dir):
    c = PsaspCase(case_dir)
    return build_sample(c, Catalog(c), sample_id="s", family_id="f")


def test_field_map_matches_d15_protocol():
    """掩码课程的字段映射必须与 schema.assemble 的 D15 任务输入协议逐项一致。"""
    assert (PQ, PV, SLACK) == (0, 1, 2)
    for t, fields in _PF_UNKNOWNS.items():
        assert tuple(fields) == PF_UNKNOWN_FIELDS[t]


def test_pf_typed_masks_unknowns_keeps_injections(case_dir):
    s = _sample(case_dir)
    cfg = MaskConfig(p_group_dropout=0.0, p_typed=0.0, p_high=0.0, p_pf_typed=1.0)
    rng = np.random.default_rng(0)
    mf, tgt, lm = apply_node_mask(s.node_feat, s.node_missing, cfg, rng)
    off = NODE_TABLE.offset_of
    sbit = group_bit_index(Group.STATE)
    tcol = off("node_type")
    types = np.argmax(s.node_feat[:, tcol:tcol + 3], axis=1)
    checked = 0
    for i in range(s.n_nodes):
        if s.node_missing[i, sbit] or s.node_feat[i, off("is_null")] > 0.5:
            assert not lm[i].any()               # missing/NULL 不遮
            continue
        want = {off(f) for f in PF_UNKNOWN_FIELDS[int(types[i])]}
        got = set(np.where(lm[i])[0].tolist())
        assert got == want                        # 恰好遮 PF 未知量
        for c in want:
            assert mf[i, c] == 0.0 and tgt[i, c] == s.node_feat[i, c]
        # 注入量可见:pl/ql 一定不在被遮集合(θ 可学 DC 物理的关键)
        assert off("pl") not in got and off("ql") not in got
        checked += 1
    assert checked > 50


def test_pf_typed_zero_is_bitwise_identical(case_dir):
    """p_pf_typed=0 时与旧默认配置同种子逐位一致——门禁/阶梯掩码流不变的硬保证。"""
    s = _sample(case_dir)
    for seed in range(8):                        # 覆盖各课程分支(u 抽样不同区间)
        a = apply_node_mask(s.node_feat, s.node_missing, MaskConfig(),
                            np.random.default_rng(seed))
        b = apply_node_mask(s.node_feat, s.node_missing, MaskConfig(p_pf_typed=0.0),
                            np.random.default_rng(seed))
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


def test_masked_data_eval_path_ignores_curriculum(case_dir):
    """augment=False(验证/门禁/阶梯)必须无视 mask_pf_typed——评估口径协议冻结。"""
    s = _sample(case_dir)
    base_cfg = TrainConfig(d=16, n_layers=1, n_global=2, seed=0)
    exp_cfg = TrainConfig(d=16, n_layers=1, n_global=2, seed=0, mask_pf_typed=0.99)
    da = _masked_data(s, base_cfg, np.random.default_rng(7), augment=False)
    db = _masked_data(s, exp_cfg, np.random.default_rng(7), augment=False)
    assert np.array_equal(da.loss_mask.numpy(), db.loss_mask.numpy())
    assert np.array_equal(da.x.numpy(), db.x.numpy())
    # 训练路径(augment=True)则应生效:同种子下高概率进入 pf 档,掩码不同
    ta = _masked_data(s, base_cfg, np.random.default_rng(7), augment=True)
    tb = _masked_data(s, exp_cfg, np.random.default_rng(7), augment=True)
    assert not np.array_equal(ta.loss_mask.numpy(), tb.loss_mask.numpy())


# ---------------- 连通片 v/theta 课程(v2 S2,2026-07-16) ----------------

def test_region_masks_only_vtheta_contiguous_keeps_injections(case_dir):
    """区域档:①只遮 v/theta 两列;②被遮节点集在有效诱导子图内连通(成片);
    ③注入量(pg/qg/pl/ql)与片外全部可见;④STATE missing/NULL 节点不遮。"""
    s = _sample(case_dir)
    cfg = MaskConfig(p_group_dropout=0.0, p_typed=0.0, p_high=0.0, p_region=1.0)
    rng = np.random.default_rng(3)
    mf, tgt, lm = apply_node_mask(s.node_feat, s.node_missing, cfg, rng,
                                  edge_index=s.edge_index)
    off = NODE_TABLE.offset_of
    vcols = {off("v"), off("theta")}
    inj_cols = [off(f) for f in ("pg", "qg", "pl", "ql")]
    masked_nodes = sorted(set(np.where(lm.any(axis=1))[0].tolist()))
    assert len(masked_nodes) >= 2, "应有成片节点被遮"
    for i in range(s.n_nodes):
        got = set(np.where(lm[i])[0].tolist())
        assert got <= vcols, f"节点 {i} 遮了 v/theta 之外的列 {got - vcols}"
        if got:
            assert got == vcols, "被遮节点须 v 与 theta 同遮"
    assert not lm[:, inj_cols].any(), "注入量必须保持可见"
    sbit = group_bit_index(Group.STATE)
    for i in masked_nodes:
        assert not s.node_missing[i, sbit] and s.node_feat[i, off("is_null")] <= 0.5
    # 连通性:掩码集内 BFS 应覆盖全部掩码节点
    adj = {i: set() for i in masked_nodes}
    for a, b in s.edge_index.T:
        a, b = int(a), int(b)
        if a in adj and b in adj:
            adj[a].add(b); adj[b].add(a)
    seen, q = {masked_nodes[0]}, [masked_nodes[0]]
    while q:
        cur = q.pop()
        for x in adj[cur]:
            if x not in seen:
                seen.add(x); q.append(x)
    assert seen == set(masked_nodes), "被遮节点集必须连通(成片),不是散点"
    # 值语义:遮蔽处置 0,target 保原值
    for i in masked_nodes:
        for c in vcols:
            assert mf[i, c] == 0.0 and tgt[i, c] == s.node_feat[i, c]


def test_region_zero_stream_invariant_to_edge_index(case_dir):
    """p_region=0(全部验证/门禁流):传不传 edge_index,掩码流逐位不变。"""
    s = _sample(case_dir)
    for pf in (0.0, 0.3):
        cfg = MaskConfig(p_pf_typed=pf)
        a = apply_node_mask(s.node_feat, s.node_missing, cfg, np.random.default_rng(7))
        b = apply_node_mask(s.node_feat, s.node_missing, cfg, np.random.default_rng(7),
                            edge_index=s.edge_index)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


def test_masked_data_eval_ignores_region_curriculum(case_dir):
    """验证/门禁路径(augment=False)无视 mask_region:与 mask_region=0 逐位一致。"""
    from gridfm.data.normalize import Normalizer
    s = _sample(case_dir)
    Normalizer().fit([s]).transform(s)
    d0 = _masked_data(s, TrainConfig(mask_region=0.0), np.random.default_rng(5), augment=False)
    d1 = _masked_data(s, TrainConfig(mask_region=0.9), np.random.default_rng(5), augment=False)
    np.testing.assert_array_equal(d0.x.numpy(), d1.x.numpy())
    np.testing.assert_array_equal(d0.loss_mask.numpy(), d1.loss_mask.numpy())
