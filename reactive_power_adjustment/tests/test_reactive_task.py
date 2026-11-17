"""无功调整推理器单测(方向二选项 B;双边深度 + v_set 归一 + 泄漏红线 + 检查点兼容)。

泄漏红线(同有功,含过压通道):输入两路(病态档案 + 双边越限深度)只消费档案给定量;
修复标签(机组集合 / Δv_set)只落 y_*。打乱标签行全部修复字段(含过压信息),两深度
通道逐位不变。
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from gridfm.data.graph import sample_to_data
from gridfm.data.normalize import Normalizer
from gridfm.psasp_io.case import PsaspCase
from gridfm.psasp_io.catalog import Catalog
from gridfm.reason.adjust_task import AdjustReasoner
from gridfm.reason.reactive_task import (
    DV_LEVELS, VIO_HIGH, VIO_LOW, VSET_SPAN, ReactiveShuntReasoner,
    build_reactive_example, build_reactive_reasoner, decode_action, react_side,
    reactive_loss, reactive_shunt_loss, shunt_cand_mask, unit_span_vset, viol_depth_2s,
)
from gridfm.reason.ts_task import gen_node_mask
from gridfm.schema.assemble import build_sample
from gridfm.schema.fields import BRANCH_TABLE, BRANCH_TYPES, NODE_TABLE
from gridfm.train.pretrain import TrainConfig, build_models

FIX = Path(__file__).parent / "fixtures" / "cepri36_golden"
V06_CKPT = Path(__file__).parent.parent / "runs" / "adjust_reasoner_v06_s0" / "ckpt" / "best.pt"
REACT_V0_CKPT = (Path(__file__).parent.parent / "runs" / "reactive_reasoner_s0"
                 / "ckpt" / "best.pt")


def _raw():
    c = PsaspCase(FIX)
    return build_sample(c, Catalog(c), sample_id="react_t", family_id="cepri36")


def _shunt_rows(raw):
    """档案里的并联补偿支路行(cepri36 夹具 6 个,全在运)。"""
    bo = BRANCH_TABLE.offset_of
    col = bo("branch_type") + BRANCH_TYPES.index("shunt_comp")
    return np.where(raw.branch_feat[:, col] > 0.5)[0]


def _raw_with_off_shunt():
    """把一个并补置为停运 → 该母线成为 shunt 候选(夹具本身无停运并补)。
    返回 (raw, 候选母线号)。"""
    raw = _raw()
    rows = _shunt_rows(raw)
    r = int(rows[0])
    raw.branch_feat[r, BRANCH_TABLE.offset_of("mark")] = 0.0
    return raw, int(raw.edge_index[0, r]) + 1


def _gen_buses(raw):
    return [int(i) + 1 for i in np.where(gen_node_mask(raw))[0]]


def _label_row(raw, feasible=True):
    buses = _gen_buses(raw)
    b1, b2 = str(buses[0]), str(buses[1])
    return {"sid": "cepri36_00000", "battery": "cepri36", "feasible": feasible,
            "violation": {"vmin": 0.91, "vmax": 1.05, "buses": {"7": "low"}},
            "n_fixes": 2,
            "fixes": [{"deltas": {b1: 0.03}, "l2": 0.03, "source": "reverse_perturb"},
                      {"deltas": {b2: 0.05}, "l2": 0.05, "source": "rule_vset"}],
            "canonical_fix": {"deltas": {b1: 0.03}, "l2": 0.03, "source": "rule_vset"}}


def test_viol_depth_2s_bilateral_reads_voltage_only():
    raw = _raw()
    d = viol_depth_2s(raw)
    assert d.shape == (raw.n_nodes, 2) and (d >= 0).all()
    vcol = NODE_TABLE.offset_of("v")
    live = raw.node_feat[:, vcol] > 0.5
    raw2 = copy.deepcopy(raw)
    i = int(np.argmax(live))
    raw2.node_feat[i, vcol] = 0.80                       # 压低 → 低压深度
    raw2.node_feat[int(np.where(live)[0][1]), vcol] = 1.12   # 抬高 → 过压深度
    d2 = viol_depth_2s(raw2)
    assert d2[i, 0] == pytest.approx(VIO_LOW - 0.80, abs=1e-6) and d2[i, 1] == 0.0
    j = int(np.where(live)[0][1])
    assert d2[j, 1] == pytest.approx(1.12 - VIO_HIGH, abs=1e-6) and d2[j, 0] == 0.0


def test_unit_span_vset_is_band_constant_on_gens():
    raw = _raw()
    s = unit_span_vset(raw)
    gm = gen_node_mask(raw)
    assert np.allclose(s[gm], VSET_SPAN) and np.allclose(s[~gm], 1.0)


def test_build_example_multihot_union_and_canonical_dvset():
    raw = _raw()
    ex = build_reactive_example(_label_row(raw), raw, "t0")
    buses = _gen_buses(raw)
    assert ex["y_units"][buses[0] - 1] == 1.0 and ex["y_units"][buses[1] - 1] == 1.0
    assert ex["y_units"].sum() == 2.0
    assert ex["y_dp"][buses[0] - 1] == pytest.approx(0.03 / VSET_SPAN, rel=1e-5), \
        "数值头监督规范解 Δv_set,按带宽常数归一(量级 O(0.1~1))"
    assert 0.1 <= abs(ex["y_dp"][buses[0] - 1]) <= 1.0, "归一后目标量级不塌到 1e-4"
    assert not ex["y_dp_mask"][buses[1] - 1]
    assert build_reactive_example(_label_row(raw, feasible=False), raw, "t1") is None


def test_build_example_rejects_misplaced_bus():
    raw = _raw()
    row = _label_row(raw)
    row["fixes"][0]["deltas"] = {"7": 0.03}              # 母线 7 非机组节点
    with pytest.raises(ValueError):
        build_reactive_example(row, raw, "t2")


def test_input_invariant_to_label_shuffle_incl_overvoltage():
    """泄漏红线:打乱标签行全部修复字段(含过压信息),两深度通道 + 归一逐位不变。"""
    raw = _raw()
    d0, s0 = viol_depth_2s(raw), unit_span_vset(raw)
    row = _label_row(raw)
    row["fixes"] = [{"deltas": {str(_gen_buses(raw)[2]): 0.09}, "l2": 0.09, "source": "x"}]
    row["canonical_fix"] = row["fixes"][0]
    row["violation"] = {"vmin": 0.1, "vmax": 9.9, "buses": {}}   # 含过压标签字段
    np.testing.assert_array_equal(viol_depth_2s(raw), d0)
    np.testing.assert_array_equal(unit_span_vset(raw), s0)


def test_reactive_reasoner_side_dim2_forward_frozen_encoder():
    torch.manual_seed(0)
    raw = _raw()
    norm = Normalizer().fit([raw])
    s = copy.deepcopy(raw); norm.transform(s)
    cfg = TrainConfig(d=32, n_layers=2, n_global=4, seed=0)
    enc, _dec = build_models(cfg)
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    data = sample_to_data(s, cfg.lape_k, cfg.rwse_steps, random_dim=cfg.random_dim,
                          branch_random_dim=cfg.branch_random_dim)
    ex = build_reactive_example(_label_row(raw), raw, "t3")
    data.depth = torch.from_numpy(viol_depth_2s(raw))     # [N,2]
    data.gen_mask = torch.from_numpy(gen_node_mask(raw))
    data.y_units = torch.from_numpy(ex["y_units"])
    data.y_dp = torch.from_numpy(ex["y_dp"])
    data.y_dp_mask = torch.from_numpy(ex["y_dp_mask"])
    batch = Batch.from_data_list([data])
    h, g, e = enc(batch)
    r = build_reactive_reasoner(d=cfg.d, n_queries=4, n_layers=1)
    assert tuple(r.side_proj.weight.shape) == (cfg.d, 2), "双边深度 side_dim=2"
    ptr, dp = r(h, g, e, batch.batch, batch.batch[batch.edge_index[0]], batch.depth)
    assert ptr.shape == (raw.n_nodes,) and dp.shape == (raw.n_nodes,)
    loss, diag = reactive_loss(ptr, dp, batch)
    loss.backward()
    assert all(p.grad is None for p in enc.parameters()), "冻结编码器不得有梯度(D13)"
    assert r.side_proj.weight.grad is not None and r.side_proj.weight.grad.shape[1] == 2
    assert diag["ptr"] > 0


# ---------------- 第二杠杆(shunt):候选掩码 / 侧通道 / 第三个头 ----------------

def test_shunt_cand_mask_only_counts_out_of_service_shunts():
    """候选 = **停运的并联补偿**;在运的不是候选(投不了)。夹具 6 个并补全在运 → 全 0。"""
    raw = _raw()
    assert len(_shunt_rows(raw)) > 0, "夹具应含并联补偿支路"
    assert shunt_cand_mask(raw).sum() == 0.0, "全在运 → 无可投候选"
    raw2, bus = _raw_with_off_shunt()
    m = shunt_cand_mask(raw2)
    assert m.shape == (raw2.n_nodes,) and m[bus - 1] == 1.0 and m.sum() == 1.0


def test_shunt_cand_mask_ignores_non_shunt_out_of_service_branches():
    """停运的**交流线**不是并补候选(只认 shunt_comp 类型位)。"""
    raw = _raw()
    bo = BRANCH_TABLE.offset_of
    ac = np.where(raw.branch_feat[:, bo("branch_type") +
                                  BRANCH_TYPES.index("acline")] > 0.5)[0]
    raw.branch_feat[int(ac[0]), bo("mark")] = 0.0
    assert shunt_cand_mask(raw).sum() == 0.0


def test_react_side_shapes_and_third_channel_is_the_mask():
    raw, _bus = _raw_with_off_shunt()
    s2 = react_side(raw, with_shunt=False)
    s3 = react_side(raw, with_shunt=True)
    assert s2.shape == (raw.n_nodes, 2) and s3.shape == (raw.n_nodes, 3)
    np.testing.assert_array_equal(s3[:, :2], viol_depth_2s(raw))
    np.testing.assert_array_equal(s3[:, 2], shunt_cand_mask(raw))


def test_side_channels_invariant_to_shunt_label_shuffle():
    """泄漏红线(第二杠杆):打乱含 shunt 在内的全部标签字段,三路侧通道逐位不变。"""
    raw, bus = _raw_with_off_shunt()
    s0 = react_side(raw, with_shunt=True)
    row = _label_row(raw)
    row["fixes"][0]["shunt"] = [bus]
    row["canonical_fix"]["shunt"] = [bus]
    build_reactive_example(row, raw, "t_leak")
    np.testing.assert_array_equal(react_side(raw, with_shunt=True), s0)


def test_build_example_y_shunt_multihot_union_and_misplacement_raises():
    raw, bus = _raw_with_off_shunt()
    row = _label_row(raw)
    row["fixes"][0]["shunt"] = [bus]                  # 并集口径:任一可行修法投的都算正
    ex = build_reactive_example(row, raw, "t_sh")
    assert ex["y_shunt"][bus - 1] == 1.0 and ex["y_shunt"].sum() == 1.0
    np.testing.assert_array_equal(ex["shunt_cand"], shunt_cand_mask(raw))
    # 无 shunt 字段的老标签 → 全 0,不炸(向后兼容 v0/v2 标签)
    assert build_reactive_example(_label_row(raw), raw, "t_old")["y_shunt"].sum() == 0.0
    bad = _label_row(raw)
    bad["fixes"][0]["shunt"] = [bus + 1 if bus + 1 <= raw.n_nodes else 1]
    with pytest.raises(ValueError):                   # 落在非候选母线 = 标签错位
        build_reactive_example(bad, raw, "t_bad")


def test_shunt_reasoner_three_heads_forward_and_masked_bce():
    torch.manual_seed(0)
    raw, bus = _raw_with_off_shunt()
    norm = Normalizer().fit([raw])
    s = copy.deepcopy(raw); norm.transform(s)
    cfg = TrainConfig(d=32, n_layers=2, n_global=4, seed=0)
    enc, _dec = build_models(cfg)
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    row = _label_row(raw); row["fixes"][0]["shunt"] = [bus]
    ex = build_reactive_example(row, raw, "t3s")
    data = sample_to_data(s, cfg.lape_k, cfg.rwse_steps, random_dim=cfg.random_dim,
                          branch_random_dim=cfg.branch_random_dim)
    data.depth = torch.from_numpy(react_side(raw, with_shunt=True))     # [N,3]
    data.gen_mask = torch.from_numpy(gen_node_mask(raw))
    data.y_units = torch.from_numpy(ex["y_units"])
    data.y_dp = torch.from_numpy(ex["y_dp"])
    data.y_dp_mask = torch.from_numpy(ex["y_dp_mask"])
    data.shunt_cand = torch.from_numpy(ex["shunt_cand"])
    data.y_shunt = torch.from_numpy(ex["y_shunt"])
    batch = Batch.from_data_list([data])
    h, g, e = enc(batch)
    r = build_reactive_reasoner(d=cfg.d, n_queries=4, n_layers=1, with_shunt=True)
    assert isinstance(r, ReactiveShuntReasoner)
    assert tuple(r.side_proj.weight.shape) == (cfg.d, 3), "侧通道 = 双边深度 + 候选掩码"
    out = r(h, g, e, batch.batch, batch.batch[batch.edge_index[0]], batch.depth)
    assert len(out) == 3
    ptr, dv, sh = out
    assert ptr.shape == dv.shape == sh.shape == (raw.n_nodes,)
    loss, diag = reactive_shunt_loss(ptr, dv, sh, batch)
    loss.backward()
    assert all(p.grad is None for p in enc.parameters()), "冻结编码器不得有梯度(D13)"
    assert r.head_shunt[0].weight.grad is not None and diag["shunt"] > 0
    assert diag["total"] == pytest.approx(diag["ptr"] + diag["dp"] + diag["shunt"], rel=1e-5)


def test_decode_action_gen_fallback_only_when_no_action_at_all():
    """兜底口径:shunt 头点火时**不得**再强塞一台机组。

    否则老师"只投并补、不动 v_set"的修法学生表达不出来,还会叠一个在该样本上无监督的
    寄生 Δv_set;lib118 判据只留 0.01pu 过压裕度,足以把本可通过的例子顶穿 1.08 判败。"""
    n = 5
    gm = np.array([True, False, True, False, True])
    span = np.full(n, VSET_SPAN, dtype=np.float32)
    dv = np.array([0.3, 0.0, 0.4, 0.0, 0.5], dtype=np.float32)
    lowp = np.array([0.1, 0.9, 0.2, 0.9, 0.05], dtype=np.float32)   # 机组位全部不过阈
    cand = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    # shunt 点火 → 纯投并补动作,dvset 必须为空
    act = decode_action(gm, lowp, dv, span, cand,
                        np.array([0.0, 0.9, 0.0, 0.1, 0.0], dtype=np.float32))
    assert act["shunt"] == [2] and act["dvset"] == {}, "纯投并补是可表达的合法动作"

    # shunt 不点火且机组也不过阈 → 才启用"至少一台机组"兜底(否则动作全空,必判败)
    act2 = decode_action(gm, lowp, dv, span, cand, np.zeros(n, dtype=np.float32))
    assert act2["shunt"] == [] and len(act2["dvset"]) == 1
    assert 3 in act2["dvset"], "兜底取机组位概率最大者(行 2 → 母线 3)"

    # 单杠杆(无 shunt 头)行为不变:兜底照常
    act3 = decode_action(gm, lowp, dv, span)
    assert act3["shunt"] == [] and len(act3["dvset"]) == 1


def test_decode_action_respects_candidate_mask_and_gen_mask():
    n = 4
    gm = np.array([True, False, True, False])
    span = np.full(n, VSET_SPAN, dtype=np.float32)
    dv = np.array([0.2, 9.9, 0.3, 9.9], dtype=np.float32)
    hi = np.full(n, 0.9, dtype=np.float32)
    cand = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    act = decode_action(gm, hi, dv, span, cand, hi)
    assert act["shunt"] == [2], "非候选位即使概率高也不投(投不了)"
    assert sorted(act["dvset"]) == [1, 3], "非机组位不选"
    assert act["dvset"][1] == pytest.approx(0.2 * VSET_SPAN, abs=1e-6), "反归一回 pu"


# ---------------- 数值头分类(治 L1 幅度塌缩) ----------------

def test_y_dv_cls_maps_canonical_delta_to_nearest_level():
    raw = _raw()
    buses = _gen_buses(raw)
    row = _label_row(raw)
    row["canonical_fix"] = {"deltas": {str(buses[0]): 0.05, str(buses[1]): 0.02},
                            "l2": 0.054, "source": "rule_vset"}
    row["fixes"] = [row["canonical_fix"]]
    ex = build_reactive_example(row, raw, "t_cls")
    assert ex["y_dv_cls"][buses[0] - 1] == DV_LEVELS.index(0.05)
    assert ex["y_dv_cls"][buses[1] - 1] == DV_LEVELS.index(0.02)
    assert ex["y_dv_cls"].dtype == np.int64
    # 负向(过压方向)也要能落到负档
    row2 = _label_row(raw)
    row2["canonical_fix"] = {"deltas": {str(buses[0]): -0.05}, "l2": 0.05, "source": "x"}
    row2["fixes"] = [row2["canonical_fix"]]
    assert build_reactive_example(row2, raw, "t_neg")["y_dv_cls"][buses[0] - 1] \
        == DV_LEVELS.index(-0.05)


def test_shunt_reasoner_dv_classification_head_shape_and_loss():
    torch.manual_seed(0)
    raw, bus = _raw_with_off_shunt()
    norm = Normalizer().fit([raw])
    s = copy.deepcopy(raw); norm.transform(s)
    cfg = TrainConfig(d=32, n_layers=2, n_global=4, seed=0)
    enc, _dec = build_models(cfg)
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.eval()
    row = _label_row(raw); row["fixes"][0]["shunt"] = [bus]
    ex = build_reactive_example(row, raw, "t_clsfwd")
    data = sample_to_data(s, cfg.lape_k, cfg.rwse_steps, random_dim=cfg.random_dim,
                          branch_random_dim=cfg.branch_random_dim)
    data.depth = torch.from_numpy(react_side(raw, with_shunt=True))
    data.gen_mask = torch.from_numpy(gen_node_mask(raw))
    data.y_units = torch.from_numpy(ex["y_units"])
    data.y_dp = torch.from_numpy(ex["y_dp"])
    data.y_dp_mask = torch.from_numpy(ex["y_dp_mask"])
    data.y_dv_cls = torch.from_numpy(ex["y_dv_cls"])
    data.shunt_cand = torch.from_numpy(ex["shunt_cand"])
    data.y_shunt = torch.from_numpy(ex["y_shunt"])
    batch = Batch.from_data_list([data])
    h, g, e = enc(batch)
    K = len(DV_LEVELS)
    r = build_reactive_reasoner(d=cfg.d, n_queries=4, n_layers=1, with_shunt=True,
                                n_dv_levels=K)
    ptr, dv, sh = r(h, g, e, batch.batch, batch.batch[batch.edge_index[0]], batch.depth)
    assert dv.shape == (raw.n_nodes, K), "分类版数值头输出 [N,K] logits"
    loss, diag = reactive_shunt_loss(ptr, dv, sh, batch)
    loss.backward()
    assert all(p.grad is None for p in enc.parameters()), "冻结编码器不得有梯度(D13)"
    assert diag["dp"] > 0 and diag["total"] == pytest.approx(
        diag["ptr"] + diag["dp"] + diag["shunt"], rel=1e-5)


def test_decode_action_classification_emits_discrete_levels():
    """分类版必须能给出**大动作** 0.05——这正是回归版被 L1 抹平、导致深病态修不动的那一档。"""
    n = 4
    gm = np.array([True, False, True, False])
    span = np.full(n, VSET_SPAN, dtype=np.float32)
    hi = np.array([0.9, 0.0, 0.9, 0.0], dtype=np.float32)
    K = len(DV_LEVELS)
    logits = np.zeros((n, K), dtype=np.float32)
    logits[0, DV_LEVELS.index(0.05)] = 9.0        # 行 0 → 大动作
    logits[2, DV_LEVELS.index(0.02)] = 9.0        # 行 2 → 小动作
    act = decode_action(gm, hi, logits, span, dv_levels=DV_LEVELS)
    assert act["dvset"] == {1: 0.05, 3: 0.02}, "档位值直接是 pu,不再乘 span"
    # 回归版口径不变(逐位兼容老检查点)
    act2 = decode_action(gm, hi, np.array([0.2, 0.0, 0.5, 0.0], dtype=np.float32), span)
    assert act2["dvset"] == {1: 0.02, 3: 0.05}


def test_loss_weights_actually_change_gradients():
    """三处加权(机组指针 / shunt 指针 / 数值头档位)都必须真的改变损失 —— 本项目已四次
    踩"稀有类不点火",而前两次的表征是**指标好看但头是死的**,故这里锁住加权确实生效。"""
    torch.manual_seed(0)
    n = 8
    ptr = torch.randn(n, requires_grad=True)
    dv = torch.randn(n, 4, requires_grad=True)
    sh = torch.randn(n, requires_grad=True)

    class _B:
        gen_mask = torch.ones(n, dtype=torch.bool)
        y_units = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
        # 档位监督必须**同时含两个类别**:F.cross_entropy(reduction='mean') 会除以权重之和,
        # 故单一类别的退化 batch 加权前后恒等(真实训练里 0.02/0.05 两档共存,加权才生效)
        y_dp_mask = torch.tensor([True, True] + [False] * (n - 2))
        y_dp = torch.zeros(n)
        y_dv_cls = torch.tensor([3, 2] + [0] * (n - 2))
        shunt_cand = torch.ones(n)
        y_shunt = torch.tensor([1.] + [0.] * (n - 1))

    b = _B()
    base, _ = reactive_shunt_loss(ptr, dv, sh, b)
    wp, _ = reactive_shunt_loss(ptr, dv, sh, b, ptr_pos_weight=4.1)
    ws, _ = reactive_shunt_loss(ptr, dv, sh, b, pos_weight=30.0)
    wd, _ = reactive_shunt_loss(ptr, dv, sh, b,
                                dv_class_weight=torch.tensor([0., 0., 0.5, 8.0]))
    assert not torch.isclose(base, wp), "机组指针正类权重必须生效(修法③)"
    assert not torch.isclose(base, ws), "shunt 正类权重必须生效"
    assert not torch.isclose(base, wd), "数值头档位类别权重必须生效"


def test_build_reactive_reasoner_rejects_dv_cls_without_shunt():
    with pytest.raises(ValueError):
        build_reactive_reasoner(d=16, n_queries=2, n_layers=1, n_dv_levels=4)


def test_shunt_loss_is_zero_when_no_candidate():
    """无停运并补的方式:shunt 项不计损(不可动的位置给不出监督信号)。"""
    torch.manual_seed(0)
    n = 6
    ptr = torch.zeros(n, requires_grad=True); dv = torch.zeros(n, requires_grad=True)
    sh = torch.zeros(n, requires_grad=True)

    class _B:
        gen_mask = torch.ones(n, dtype=torch.bool)
        y_units = torch.zeros(n)
        y_dp_mask = torch.zeros(n, dtype=torch.bool)
        y_dp = torch.zeros(n)
        shunt_cand = torch.zeros(n)
        y_shunt = torch.zeros(n)

    _loss, diag = reactive_shunt_loss(ptr, dv, sh, _B())
    assert diag["shunt"] == 0.0


def test_build_reactive_reasoner_default_stays_two_head_side_dim2():
    """缺省(单杠杆)必须逐位维持 v0/v2 结构,否则老检查点全部作废。"""
    r = build_reactive_reasoner(d=16, n_queries=2, n_layers=1)
    assert not isinstance(r, ReactiveShuntReasoner)
    assert not hasattr(r, "head_shunt")
    assert tuple(r.side_proj.weight.shape) == (16, 2)


@pytest.mark.skipif(not REACT_V0_CKPT.exists(), reason="无功 v0 检查点不在,跳过兼容回归")
def test_reactive_v0_checkpoint_still_loads_after_third_head():
    """加第三个头后,已定论的无功 v0 检查点(side_dim=2 双头)仍 strict load。"""
    b = torch.load(REACT_V0_CKPT, map_location="cpu", weights_only=False)
    rc = b["config"]
    r = build_reactive_reasoner(d=rc["d"], n_queries=rc["queries"], n_layers=rc["layers"],
                                with_shunt=bool(rc.get("with_shunt", False)))
    r.load_state_dict(b["reasoner_state"])            # strict=True
    assert tuple(r.side_proj.weight.shape) == (rc["d"], 2)


@pytest.mark.skipif(not V06_CKPT.exists(), reason="v0.6 检查点(runs/)不在,跳过兼容回归")
def test_v06_checkpoint_still_loads_after_side_dim_change():
    """审查 fix #7:AdjustReasoner 加 side_dim 参数后,v0.6 检查点(G2 交付物)仍 strict load。"""
    b = torch.load(V06_CKPT, map_location="cpu", weights_only=False)
    rc = b["config"]
    r = AdjustReasoner(d=rc["d"], n_queries=rc["queries"], n_layers=rc["layers"])  # 缺省 side_dim=1
    r.load_state_dict(b["reasoner_state"])                # strict=True
    assert tuple(r.side_proj.weight.shape) == (rc["d"], 1)
