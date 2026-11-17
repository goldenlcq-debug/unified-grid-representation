"""物理残差损失单测(方案执行清单 1a 强制配套:对 PSASP 已解真值失配应≈0)。

全部离线(118 golden fixture),不依赖 PSASP/数据根。容差 1e-4 依据:
118 夹具实测干净方程 max|ΔP|=1.0e-5 / max|ΔQ|=4.6e-6(LP1/LP2 打印精度决定的地板),
留一个数量级余量;val 合成库全量核对(404 样本)PQ 方程 p99=1.2e-5,同一地板。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from gridfm.data.normalize import Normalizer
from gridfm.psasp_io.case import PsaspCase
from gridfm.psasp_io.catalog import Catalog
from gridfm.reason.pf_physics import (
    PHYS_FIELDS, pf_phys_arrays, pf_phys_denorm, pf_phys_loss, pf_phys_residual,
)
from gridfm.reason.pf_task import PF_FIELDS, pf_input_view
from gridfm.schema.assemble import build_sample
from gridfm.schema.fields import BRANCH_TABLE, BRANCH_TYPES, NODE_TABLE, Group
from gridfm.schema.sample import group_bit_index

TOL = 1e-4


@pytest.fixture(scope="module")
def raw(case_dir):
    c = PsaspCase(case_dir)
    return build_sample(c, Catalog(c), sample_id="g", family_id="t")


@pytest.fixture(scope="module")
def norm(raw):
    return Normalizer().fit([raw])


def _norm_pack(raw, norm):
    """归一化副本 + target/loss_mask(与训练脚本同路径)。"""
    import copy
    s = copy.deepcopy(raw)
    norm.transform(s)
    _view, target, lm = pf_input_view(s, view="group")
    return torch.from_numpy(target).float(), torch.from_numpy(lm)


def _tensors(arr):
    return (torch.from_numpy(arr["index"]), torch.from_numpy(arr["g"]),
            torch.from_numpy(arr["b"]),
            {k: torch.from_numpy(arr[k]) for k in
             ("eqp_clean", "eqq_clean", "eqp_def", "eqq_def")})


def _pred_from_truth(target, lm):
    """预测 = 真值(完美模型):把 target 的 4 个监督字段抄进 pred 通道。"""
    pred = torch.zeros(target.shape[0], len(PF_FIELDS))
    for k, f in enumerate(PF_FIELDS):
        pred[:, k] = target[:, NODE_TABLE.offset_of(f)]
    return pred


# ---------------- 正确性:PSASP 已解真值失配 ≈ 0 ----------------

def test_truth_mismatch_near_zero_clean_eqs(raw, norm):
    """干净方程(PQ P/Q + PV P):真值代入,失配必须在打印精度地板内。
    这同时锁定:π 模型/并补口径、Ybus 装配、反归一化仿射、θ 弧度约定。"""
    arr = pf_phys_arrays(raw)
    target, lm = _norm_pack(raw, norm)
    idx, g, b, masks = _tensors(arr)
    dn = torch.from_numpy(pf_phys_denorm(norm))
    dp, dq = pf_phys_residual(_pred_from_truth(target, lm), target, lm, idx, g, b, dn)
    assert int(masks["eqp_clean"].sum()) >= 90     # 118 夹具实测 100(非平凡覆盖)
    assert int(masks["eqq_clean"].sum()) >= 80     # 实测 89
    assert dp[masks["eqp_clean"]].abs().max() < TOL
    assert dq[masks["eqq_clean"]].abs().max() < TOL


def test_def_eqs_match_lp5_solved_not_card(case_dir, raw, norm):
    """定义方程(PV Q):真值(assemble 已改读 LP5 解值,2026-07-12 裁决落地)代入 ≈0;
    L5 卡值代入显著非 0——账目方向反转后的更新版(原版在修复前以卡值为 target,
    docstring 预告的"卡值分支失效并更新"即此)。"""
    arr = pf_phys_arrays(raw)
    target, lm = _norm_pack(raw, norm)
    idx, g, b, masks = _tensors(arr)
    dn = torch.from_numpy(pf_phys_denorm(norm))
    qcol = NODE_TABLE.offset_of("qg")
    qk = PF_FIELDS.index("qg")
    m = masks["eqq_def"] & torch.from_numpy(lm.numpy()[:, qcol])
    assert int(m.sum()) > 0

    # 真值(= LP5 解值)代入 → 定义方程闭合
    dp, dq = pf_phys_residual(_pred_from_truth(target, lm), target, lm, idx, g, b, dn)
    assert dq[m].abs().max() < TOL, "解值真值代入后 PV Q 定义方程应≈0"

    # L5 卡值(解前值)代入 → 显著非 0(证明 assemble 用的确实是解值而非卡值)
    card = {gv.bs_no: gv.qg for gv in PsaspCase(case_dir).gens() if gv.mark == 1}
    pred = _pred_from_truth(target, lm)
    hit = 0
    for bus, qg_card in card.items():
        row = bus - 1
        if row < raw.n_nodes and lm[row, qcol]:
            pred[row, qk] = (qg_card - dn[qk, 0]) / dn[qk, 1]
            hit += 1
    assert hit > 0
    dp2, dq2 = pf_phys_residual(pred, target, lm, idx, g, b, dn)
    assert dq2[m].abs().max() > 0.1, "卡值代入应显著非 0(解值≠卡值的账目见证)"


# ---------------- 口径:Ybus 行完整性排除 ----------------

def test_xfmr_adjacent_nodes_excluded(raw):
    boff = BRANCH_TABLE.offset_of
    onehot = raw.branch_feat[:, boff("branch_type"):boff("branch_type") + 6]
    bt = np.asarray(BRANCH_TYPES)[np.argmax(onehot, axis=1)]
    on = raw.branch_feat[:, boff("mark")] == 1
    xf = np.isin(bt, ["xfmr_2w", "phase_shifter", "xfmr_3w"]) & on
    assert xf.any(), "118 夹具应含在运变压器(口径测试前提)"
    arr = pf_phys_arrays(raw)
    ei, ej = raw.edge_index
    adj = np.unique(np.concatenate([ei[xf], ej[xf]]))
    for k in ("eqp_clean", "eqq_clean", "eqp_def", "eqq_def"):
        assert not arr[k][adj].any(), f"{k}:与在运变压器相邻的节点必须排除"
    assert arr["eqp_clean"].any(), "其余节点应保留(不得整样本弃用)"


def test_null_and_missing_rows_excluded(raw):
    noff = NODE_TABLE.offset_of
    arr = pf_phys_arrays(raw)
    null = raw.node_feat[:, noff("is_null")] > 0.5
    smiss = raw.node_missing[:, group_bit_index(Group.STATE)]
    for k in ("eqp_clean", "eqq_clean", "eqp_def", "eqq_def"):
        assert not arr[k][null | smiss].any()


def test_rejects_normalized_sample(raw, norm):
    import copy
    s = copy.deepcopy(raw)
    norm.transform(s)
    with pytest.raises(ValueError):
        pf_phys_arrays(s)


# ---------------- 梯度与合成正确性 ----------------

def test_grad_flows_only_through_unknown_positions(raw, norm):
    """残差对未知量位置的 pred 有梯度;对给定量位置的 pred 无梯度(合成正确性:
    给定位置用真值、预测位置用 pred,泄漏与错用一测两防)。"""
    arr = pf_phys_arrays(raw)
    target, lm = _norm_pack(raw, norm)
    idx, g, b, masks = _tensors(arr)
    dn = torch.from_numpy(pf_phys_denorm(norm))
    pred = _pred_from_truth(target, lm).requires_grad_(True)
    dp, dq = pf_phys_residual(pred, target, lm, idx, g, b, dn)
    loss, met = pf_phys_loss(dp, dq, masks, w_clean=1.0, w_def=0.0)
    loss.backward()
    grad = pred.grad
    assert grad is not None
    known = torch.ones_like(grad, dtype=torch.bool)
    unknown_any = torch.zeros((), dtype=torch.bool)
    for k, f in enumerate(PF_FIELDS):
        m = lm[:, NODE_TABLE.offset_of(f)]
        known[m, k] = False
        unknown_any = unknown_any | (grad[m, k].abs().sum() > 0)
    assert bool(unknown_any), "残差必须对未知量预测有梯度(θ 杠杆的存在性)"
    assert grad[known].abs().max() == 0.0, "给定量位置的 pred 不得进入残差"
    assert np.isfinite(met["phys_clean"]) and np.isfinite(met["phys_def"])


def test_perturbed_theta_increases_clean_residual(raw, norm):
    """扰动一个未知 θ,干净方程失配单调上升(方程真的在约束 θ)。"""
    arr = pf_phys_arrays(raw)
    target, lm = _norm_pack(raw, norm)
    idx, g, b, masks = _tensors(arr)
    dn = torch.from_numpy(pf_phys_denorm(norm))
    pred = _pred_from_truth(target, lm)
    dp0, dq0 = pf_phys_residual(pred, target, lm, idx, g, b, dn)
    l0, _ = pf_phys_loss(dp0, dq0, masks, w_clean=1.0)
    tk = PF_FIELDS.index("theta")
    tcol = NODE_TABLE.offset_of("theta")
    rows = torch.where(torch.from_numpy(arr["eqp_clean"]) & lm[:, tcol])[0]
    assert len(rows) > 0
    pred[rows[0], tk] += 0.1
    dp1, dq1 = pf_phys_residual(pred, target, lm, idx, g, b, dn)
    l1, _ = pf_phys_loss(dp1, dq1, masks, w_clean=1.0)
    assert float(l1) > float(l0) + 1e-4


# ---------------- 批处理:跨图一致(PyG 偏移回归) ----------------

def test_batched_residual_matches_per_graph(raw, norm):
    """两图 batch 的残差与各自单图逐位一致(phys_index 偏移正确性;
    编码器 3.1 跨图泄漏单测的残差侧同款)。"""
    arr = pf_phys_arrays(raw)
    target, lm = _norm_pack(raw, norm)
    dn = torch.from_numpy(pf_phys_denorm(norm))

    def mk():
        d = Data(x=torch.zeros(raw.n_nodes, 1),
                 edge_index=torch.from_numpy(raw.edge_index.astype(np.int64)))
        d.num_nodes = raw.n_nodes
        d.phys_index = torch.from_numpy(arr["index"])
        d.phys_g = torch.from_numpy(arr["g"])
        d.phys_b = torch.from_numpy(arr["b"])
        d.target = target.clone()
        d.loss_mask = lm.clone()
        for k in ("eqp_clean", "eqq_clean", "eqp_def", "eqq_def"):
            setattr(d, k, torch.from_numpy(arr[k]))
        return d

    b2 = Batch.from_data_list([mk(), mk()])
    torch.manual_seed(0)
    pred1 = _pred_from_truth(target, lm) + 0.05 * torch.randn(raw.n_nodes, len(PF_FIELDS))
    pred2 = torch.cat([pred1, pred1])
    dp2, dq2 = pf_phys_residual(pred2, b2.target, b2.loss_mask,
                                b2.phys_index, b2.phys_g, b2.phys_b, dn)
    idx, g, b, _ = _tensors(arr)
    dp1, dq1 = pf_phys_residual(pred1, target, lm, idx, g, b, dn)
    torch.testing.assert_close(dp2[:raw.n_nodes], dp1, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(dp2[raw.n_nodes:], dp1, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(dq2[raw.n_nodes:], dq1, atol=1e-5, rtol=1e-4)


# ---------------- DC-θ 特征(杠杆 d,2026-07-12 用户批准 B 方案) ----------------

def test_dc_theta_no_leakage(raw):
    """扰动全部被监督位置(PQ/PV 的 θ、PQ 的 v、PV qg、平衡机 pg/qg)不得改变 DC-θ。"""
    from dataclasses import replace

    from gridfm.reason.pf_physics import dc_theta
    from gridfm.train.masking import PF_UNKNOWN_FIELDS

    base = dc_theta(raw)
    nf = raw.node_feat.copy()
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    n_pert = 0
    for i in range(raw.n_nodes):
        for f in PF_UNKNOWN_FIELDS[int(types[i])]:
            nf[i, NODE_TABLE.offset_of(f)] += 7.7    # 大幅扰动解算产物
            n_pert += 1
    assert n_pert > 0
    pert = dc_theta(replace(raw, node_feat=nf))
    np.testing.assert_array_equal(base, pert)


def test_dc_theta_uses_given_injections(raw):
    """扰动给定注入(负荷 pl)必须改变 DC-θ(证明它真在解方程,不是常数)。"""
    from dataclasses import replace

    from gridfm.reason.pf_physics import dc_theta

    base = dc_theta(raw)
    assert base[:, 1].sum() > 100                    # 118 夹具大部分节点 valid
    nf = raw.node_feat.copy()
    pl = NODE_TABLE.offset_of("pl")
    rows = np.where((raw.node_feat[:, pl] > 0.1) & (base[:, 1] > 0.5))[0]
    nf[rows[0], pl] += 1.0
    pert = dc_theta(replace(raw, node_feat=nf))
    assert np.abs(pert[:, 0] - base[:, 0]).max() > 1e-4


def test_dc_theta_beats_constant_on_fixture(raw):
    """DC-θ 对真值的 MAE 必须优于'恒答平衡机角'(存在性检验;精确度由训练报告的
    DC 裸值基线行持续追踪,门禁仍挂 L2,8.9)。"""
    from gridfm.reason.pf_physics import dc_theta
    from gridfm.schema.sample import group_bit_index

    dc = dc_theta(raw)
    th = NODE_TABLE.offset_of("theta")
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    smiss = raw.node_missing[:, group_bit_index(Group.STATE)]
    m = (dc[:, 1] > 0.5) & ~smiss & (types != 2)
    truth = raw.node_feat[m, th]
    slack_th = raw.node_feat[types == 2, th][0]
    mae_dc = np.abs(dc[m, 0] - truth).mean()
    mae_const = np.abs(slack_th - truth).mean()
    assert mae_dc < mae_const, f"DC {mae_dc:.4f} 应优于恒答平衡机角 {mae_const:.4f}"


# ---------------------------------------------------------------------------
# 2026-08-21 第 2 轮修法(结果档 §3.11 第 1~4 项)的行为锁定
# ---------------------------------------------------------------------------

def _two_slack_sample(raw):
    """把 118 夹具里另一台 PV 机改成平衡机,并给它一个**不同的**给定角 → 多平衡机算例。"""
    from dataclasses import replace

    tcol = NODE_TABLE.offset_of("node_type")
    th = NODE_TABLE.offset_of("theta")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    pv = np.where(types == 1)[0]
    assert pv.size, "夹具应有 PV 节点"
    nf = raw.node_feat.copy()
    k = int(pv[0])
    nf[k, tcol:tcol + 3] = 0.0
    nf[k, tcol + 2] = 1.0                       # → 平衡机
    nf[k, th] = 0.25                            # 与原平衡机不同的给定角
    return replace(raw, node_feat=nf), k


def test_dc_theta_multi_slack_dirichlet(raw):
    """修法 2:全部平衡机按给定角做 Dirichlet。

    验证 (i) 两台平衡机的输出角都等于各自给定角;(ii) 自由节点确实解的是
    B_ff θ_f = P_f − B_fr θ_r(残差 ≈ 0)。旧实现只锚首台、其余当普通注入,第二台的
    输出角会偏离给定值,本测试即为其守卫。
    """
    import scipy.sparse as sp

    from gridfm.reason.pf_physics import dc_theta
    from gridfm.schema.fields import BRANCH_TABLE as BT

    s2, k = _two_slack_sample(raw)
    th = NODE_TABLE.offset_of("theta")
    tcol = NODE_TABLE.offset_of("node_type")
    out = dc_theta(s2)
    types = np.argmax(s2.node_feat[:, tcol:tcol + 3], axis=1)
    slacks = np.where((types == 2) & (out[:, 1] > 0.5))[0]
    assert slacks.size >= 2, "构造后应有两台平衡机"
    np.testing.assert_allclose(out[slacks, 0], s2.node_feat[slacks, th], atol=1e-6)

    # 自由节点的法方程残差
    boff = BT.offset_of
    x = s2.branch_feat[:, boff("x")].astype(np.float64)
    mark = s2.branch_feat[:, boff("mark")]
    oh = s2.branch_feat[:, boff("branch_type"):boff("branch_type") + 6]
    bt = np.asarray(BRANCH_TYPES)[np.argmax(oh, axis=1)]
    use = (mark == 1) & (bt != "shunt_comp") & (np.abs(x) > 1e-9)
    i, j = s2.edge_index[0][use].astype(np.int64), s2.edge_index[1][use].astype(np.int64)
    b = 1.0 / x[use]
    n = s2.n_nodes
    B = sp.coo_matrix((np.r_[b, b, -b, -b], (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n)).tocsr()
    p_inj = (s2.node_feat[:, NODE_TABLE.offset_of("pg")]
             - s2.node_feat[:, NODE_TABLE.offset_of("pl")]).astype(np.float64)
    free = np.where((out[:, 1] > 0.5) & (types != 2))[0]
    resid = np.abs((B[free] @ out[:, 0].astype(np.float64)) - p_inj[free]).max()
    # 容差 1e-4:dc_theta 的输出按 float32 落盘(eps 1.2e-7),而 B 的行 1-范数在 118 夹具上
    # 达 ~100(1/x 量级 × 度数),故存储舍入本身就会放大到 1e-5 量级;旧实现的锚点错误是
    # ~0.25 rad × 行范数 = O(10),两者相差五个数量级,判别力充足。
    assert resid < 1e-4, f"多平衡机 Dirichlet 法方程残差 {resid:.2e} 应 ≈ 0"


def test_dc_theta_single_slack_unchanged(raw):
    """修法 2 在**单平衡机**(现语料全部如此)下必须与"解完再平移"逐位等价 —— 归档结果可复现的守卫。"""
    from gridfm.reason.pf_physics import dc_theta

    out = dc_theta(raw)
    tcol = NODE_TABLE.offset_of("node_type")
    th = NODE_TABLE.offset_of("theta")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    s0 = int(np.where(types == 2)[0][0])
    assert abs(float(out[s0, 0]) - float(raw.node_feat[s0, th])) < 1e-7
    assert out[:, 1].sum() > 100


def test_dc_theta_lossy_final_check_is_superset(raw):
    """修法 1:末轮复核只会让回退**更多**(fallback 是超集),且不改变未回退分量的 θ。"""
    from gridfm.reason.pf_physics import dc_theta_lossy

    a0 = dc_theta_lossy(raw, final_check=False)
    a1 = dc_theta_lossy(raw, final_check=True)
    ok = (a0[:, 1] > 0.5) & (a1[:, 1] > 0.5)
    assert np.all(a1[ok, 2] >= a0[ok, 2]), "开启末轮复核后不应有分量从回退变回不回退"
    same = ok & (a0[:, 2] == a1[:, 2])
    np.testing.assert_allclose(a0[same, 0], a1[same, 0], atol=0, rtol=0)


def test_dc_theta_lossy_final_check_no_leakage(raw):
    """D15:lossyg 与 lossy 一样,扰动被监督量不得改变特征。"""
    from dataclasses import replace

    from gridfm.reason.pf_physics import dc_theta_lossy
    from gridfm.train.masking import PF_UNKNOWN_FIELDS

    base = dc_theta_lossy(raw, final_check=True)
    nf = raw.node_feat.copy()
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    for i in range(raw.n_nodes):
        for f in PF_UNKNOWN_FIELDS[int(types[i])]:
            nf[i, NODE_TABLE.offset_of(f)] += 7.7
    np.testing.assert_array_equal(base, dc_theta_lossy(replace(raw, node_feat=nf), final_check=True))
