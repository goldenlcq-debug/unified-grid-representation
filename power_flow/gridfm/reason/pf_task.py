"""潮流计算任务推理器(方案 5.1/5.2/5.3;D13/D15;P2 载体任务之一)。

**输入协议(D15,防真值泄漏,强制)**:推理输入档案由"未解方式"编码得到——
1. 节点:按类型将 PF 未知量置缺失并清零(PQ:V/θ;PV:Q/θ;平衡机:P/Q),
   注入量保持可见。字段映射复用 masking.PF_UNKNOWN_FIELDS(pf_typed 课程同款,单测锁定);
2. 支路:pi/qi/pj/qj(潮流解算产物)一并清零——未解方式没有支路潮流,留着即真值泄漏
   (v0 决策:方案 5.2 文本只点名节点未知量,支路潮流按"未解方式"语义从严处理,
   mark 投退是给定量保留;见变更记录)。
真值只出现在损失监督侧。配强制泄漏单测(tests/test_pf_reasoner.py)。

**口径 A/B 对照(2026-07-11 用户裁决:先做对照再定,均不动编码器)**:
- view="field"(B,现行):上述字段级视图。该花样在 v1 预训练课程中结构性缺席
  (评审第 4 项,OOD 混杂);
- view="group"(A):节点侧整 STATE 组清零(**节点列输入花样**与预训练组缺失课程
  c_topo 逐位同款;逐列语义与 masking.apply_node_mask 组缺失分支单测锁定),给定量(注入/
  负荷/设定值)改由 pf_side_features 从推理器侧原始并读(D13:任务参数在推理器侧;
  known 位显式携带 = D15"missing 位"半边协议在推理器侧的落地)。
  支路潮流仍清零(泄漏红线;预训练从未遮支路,此半边两口径同为 OOD 残留,如实入档)。
**作用域限定(2026-07-11 评审)**:"逐位同款"仅在编码器输入张量的节点特征列层面成立;
编码器是消息传递 GNN + 全局 token,支路清零会经消息传递污染每个节点 token,故编码器输入
整体(含支路)仍含两口径共享的 OOD 残留,"节点 token 表征分布内"不因此成立。残留在 A/B
对照中相互抵消(两口径共享),不影响对照结论;但不得据此把 A 口径结果与预训练 c_topo
重构指标直接比较。两口径 target/loss_mask 逐位一致 → 结果直接可比。

**推理器基准结构(5.3)**:任务侧 Q 个可学习查询 token → n 层[交叉注意力读档案
(节点+支路+全局 token 全集)+ 自注意力 + FFN] → 逐节点共享回归头(节点 token
交叉读回任务 token 作上下文)。输出 (P, Q, V, θ) 四元组,损失按节点类型只对未知量计
(5.2 节点类型损失掩码)。参数 ~1e6(D9);全部参数在推理器侧,编码器冻结不动(D13)。
红线 4:交叉注意力的键 = 编码器 token,LaPE/RWSE 拓扑注入已含其中。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch_geometric.utils import to_dense_batch

from ..models.encoder.gps import dense_pad

from ..schema.fields import BRANCH_TABLE, NODE_TABLE, Group
from ..schema.sample import GridSample, group_bit_index
from ..train.masking import COL_GROUP, PF_UNKNOWN_FIELDS, _pf_typed_mask
from .pf_theta_edge import DTH_DC_COL, mismatch_correction, potential_reconstruct, wrap_angle

# 推理器监督/输出的 4 个状态字段(5.2 四元组;列序即头的输出通道序)
PF_FIELDS = ("pg", "qg", "v", "theta")
BRANCH_SOLVED_FIELDS = ("pi", "qi", "pj", "qj")
# STATE 组全列(A 口径整组遮蔽作用域;pl/ql/v_set 亦属 STATE,一并遮蔽后从侧通道供)
STATE_FIELDS = tuple(f.name for f in NODE_TABLE.fields if f.group == Group.STATE)
PF_VIEWS = ("field", "group")
PF_SIDE_DIM = 2 * len(STATE_FIELDS)   # 给定量值 + known 位,各 |STATE|


def pf_input_view(sample: GridSample, view: str = "field",
                  ) -> Tuple[GridSample, np.ndarray, np.ndarray]:
    """D15 未解方式编码。返回 (任务视图, target[N,F], loss_mask[N,F])。

    输入应为**已归一化**样本;清零即"标准化空间置 0"(与掩码课程同约定)。
    视图保持 sample_id/family_id → 持久化随机特征同源(7.4 逐位一致)。
    view 语义见模块 docstring(A/B 口径对照);两口径监督逐位一致。
    """
    if view not in PF_VIEWS:
        raise ValueError(f"未知口径 {view!r},应为 {PF_VIEWS}")
    masked_feat, target, loss_mask = _pf_typed_mask(sample.node_feat, sample.node_missing)
    if view == "group":
        sbit = group_bit_index(Group.STATE)
        rows = ~sample.node_missing[:, sbit]          # 组缺失分支同款:仅豁免本就 missing
        masked_feat[np.ix_(rows, np.where(COL_GROUP == sbit)[0])] = 0.0
    bf = sample.branch_feat.copy()
    for f in BRANCH_SOLVED_FIELDS:
        bf[:, BRANCH_TABLE.offset_of(f)] = 0.0
    view_s = replace(sample, node_feat=masked_feat, branch_feat=bf)
    return view_s, target, loss_mask


def pf_side_features(sample: GridSample) -> np.ndarray:
    """A 口径推理器侧给定量通道 [N, PF_SIDE_DIM](D13,不进编码器)。

    前 |STATE| 列 = STATE 组原始值(类型化未知量清零),后 |STATE| 列 = known 位。
    未知位置值与位恒 0(泄漏单测锁定);STATE missing / NULL 占位行全 0。
    known 位显式区分"给定 0"与"未知置 0"——D15 missing 位半边协议的推理器侧落地。
    """
    n = sample.n_nodes
    k = len(STATE_FIELDS)
    out = np.zeros((n, 2 * k), dtype=np.float32)
    sbit = group_bit_index(Group.STATE)
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(sample.node_feat[:, tcol:tcol + 3], axis=1)
    null = sample.node_feat[:, NODE_TABLE.offset_of("is_null")] > 0.5
    active = ~sample.node_missing[:, sbit] & ~null            # 携带给定量的行
    for j, f in enumerate(STATE_FIELDS):
        unknown = np.zeros(n, dtype=bool)                    # 该字段对本类型是否 PF 未知量
        for t, unk in PF_UNKNOWN_FIELDS.items():
            if f in unk:
                unknown |= types == t
        known = active & ~unknown
        out[known, j] = sample.node_feat[known, NODE_TABLE.offset_of(f)]
        out[known, k + j] = 1.0
    return out


class _MPLayer(nn.Module):
    """推理器侧薄消息传递(θ 杠杆 c;D13 注记:任务物理属推理器侧自由)。

    红线自查(2.4):①权重 d×d 级共享,无 N 依赖;②消息沿真实电网边(edge_index),
    边特征 = 编码器支路 token(含拓扑注入);③无按编号查表;④输出逐 token。
    动机(2026-07-11 诊断):A 口径注入量只经侧通道进推理器,原推理器无图局部传播
    算力,θ 误差随离平衡机图距离线性增长(+0.01~0.07 rad/跳)——补消息传递即补积分。
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.ln = nn.LayerNorm(d)

    def forward(self, h, e, edge_index):
        src, dst = edge_index[0], edge_index[1]
        x = self.ln(h)
        m = self.msg(torch.cat([x[src], x[dst], e], dim=-1))
        # index_add_ 不在 autocast 白名单:AMP 下 m 为 fp16 而累加器随 h 为 fp32,显式对齐
        agg = torch.zeros_like(h).index_add_(0, dst, m.to(h.dtype))
        return h + self.upd(torch.cat([x, agg], dim=-1))


class _ReasonerLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.ln_c, self.ln_s, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(),
                                 nn.Linear(ffn_mult * d, d))

    def forward(self, q, arch, kpm):
        x = self.ln_c(q)
        q = q + self.cross(x, arch, arch, key_padding_mask=kpm, need_weights=False)[0]
        x = self.ln_s(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        return q + self.ffn(self.ln_f(q))


class PFReasoner(nn.Module):
    def __init__(self, d: int, n_queries: int = 16, n_layers: int = 2,
                 n_heads: int = 4, ffn_mult: int = 2, side_dim: int = 0,
                 gnn_layers: int = 0, theta_head: bool = False,
                 edge_side_dim: int = 0, theta_edge: bool = False,
                 theta_wrap: bool = False, edge_dc_residual: bool = True,
                 edge_head_zero_init: bool = False,
                 edge_head_local: bool = False, refine_steps: int = 0,
                 v_hint: bool = False, refine_qv: bool = False, refine_cov: float = 0.99,
                 refine_dth_cap: float = 1.0, refine_dv_cap: float = 0.5, refine_ls: bool = False) -> None:
        super().__init__()
        # θ 边势重建(2026-08-21 5060 线,pf_theta_edge):逐边 Δθ 头 + 图上最小二乘积分;
        # theta_wrap:θ 输出折回 (−π, π](与存档真值同域)。两者皆 False 时结构与旧检查点逐位兼容。
        self.theta_edge = bool(theta_edge)
        self.theta_wrap = bool(theta_wrap)
        self.edge_dc_residual = bool(edge_dc_residual)
        self.edge_head_zero_init = bool(edge_head_zero_init)
        self.edge_head_local = bool(edge_head_local)
        # 失配修正步(预注册 §13.1,E6):势重建后展开 K 步 DC-Newton 修正,步长可学习(逐步一个标量,
        # 与图规模无关 → 不违反"输出禁固定 N 维"与"禁按编号查表"的红线)。0 = 关,结构与旧检查点逐位兼容。
        self.refine_steps = int(refine_steps)
        # E12(预注册 §15.2):Q–V 半步开关、覆盖率闸阈值、发散闸上限(逐图判)
        self.refine_qv = bool(refine_qv)
        self.refine_cov, self.refine_dth_cap, self.refine_dv_cap = float(refine_cov), float(refine_dth_cap), float(refine_dv_cap)
        # E13(2026-08-22 第 3 轮 70k 诊断):P–θ 步的残差回溯线搜索——候选 α_k×{1, ½, ¼},逐图取第一个让干净 P 行 mean|ΔP| 下降的尺度,
        # 都不降则该图本步跳过;发散闸作用在缩放后的步上。零参数、与规模无关;False = E12 行为逐位一致。
        self.refine_ls = bool(refine_ls)
        self.refine_ls_scales = (1.0, 0.5, 0.25)
        self.refine_stats = {}
        # 电压残差头(预注册 §14.2,E10):v = v̂(Q–V 物理提示,归一化空间)+ Δv(头输出);
        # 提示无效的节点退回直接头。False = 旧结构逐位兼容。
        self.v_hint = bool(v_hint)
        if self.refine_qv and not self.refine_steps:
            raise ValueError("refine_qv 需要 refine_steps > 0")
        if self.refine_steps:
            if not theta_edge:
                raise ValueError("refine_steps 需要 --theta-edge(修正步作用在势重建输出上)")
            self.refine_alpha = nn.Parameter(torch.ones(self.refine_steps))
            if self.refine_qv:
                self.refine_beta = nn.Parameter(torch.ones(self.refine_steps))
        if self.theta_edge:
            if edge_side_dim <= 0:
                raise ValueError("theta_edge 需要 edge_side_dim>0(边侧特征)")
            self.edge_side_proj = nn.Linear(edge_side_dim, d)
            if self.edge_head_local:
                # 局部物理边头(E4,2026-08-21 诊断:节点 token/上下文在大网分布外驱动错误修正):
                # 只看两端给定量(STATE 值 + known 位,排除绝对角列)、两端预测电压 V̂(停梯度)与边侧物理量;
                # 不看 h / ctx / e / 绝对 θ_dc → 与规模无关。需 side(A 口径)。
                if side_dim <= 0:
                    raise ValueError("edge_head_local 需要 A 口径侧通道(side_dim>0)")
                self.n_side_local = len(STATE_FIELDS) * 2            # 只取 STATE 值 + known 位
                keep = torch.ones(self.n_side_local)
                k_th = STATE_FIELDS.index("theta")
                keep[k_th] = 0.0
                keep[len(STATE_FIELDS) + k_th] = 0.0                  # 绝对角及其 known 位不进边头
                self.register_buffer("side_keep", keep)
                self.loc_node_proj = nn.Linear(self.n_side_local + 1, d)   # +1 = V̂(归一化空间)
                self.ln_loc = nn.LayerNorm(d)
                self.edge_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
            else:
                self.ln_e = nn.LayerNorm(d)
                self.ln_he = nn.LayerNorm(2 * d)
                # 逐边共享头:输入 [feat[src](2d) | feat[dst](2d) | LN(e)(d) | 边侧投影(d)]
                self.edge_head = nn.Sequential(nn.Linear(6 * d, d), nn.GELU(), nn.Linear(d, 1))
            if self.edge_head_zero_init:
                # E19a(预注册 §22.1):边头末层置零 → 训练步 0 的 δ 恒等于边基线 Δθ(edge_dc_residual),
                # θ 输出 = 零训练边基线积分(档案值 70k 0.163 / 25k 0.0069);训练从"不改基线"出发,
                # 只学基线之上的修正(E16 诊断:训练后 θ 反劣于自家基线 +25%(70k)/+64%(25k))。
                # False = 旧检查点逐位兼容(仅影响初始化,不改结构)。
                if not self.edge_dc_residual:
                    raise ValueError("edge_head_zero_init 需要 edge_dc_residual(否则 δ≡0,θ 退化为常数场)")
                nn.init.zeros_(self.edge_head[-1].weight)
                nn.init.zeros_(self.edge_head[-1].bias)
        else:
            self.edge_side_proj = None
            self.edge_head = None
            if edge_head_zero_init:
                raise ValueError("edge_head_zero_init 需要 theta_edge")
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.layers = nn.ModuleList(_ReasonerLayer(d, n_heads, ffn_mult)
                                    for _ in range(n_layers))
        self.ln_node = nn.LayerNorm(d)
        self.node_cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                  nn.Linear(d, len(PF_FIELDS)))
        # A 口径给定量侧通道(D13 推理器侧参数);0 = B 口径,结构与旧检查点逐位兼容
        self.side_proj = nn.Linear(side_dim, d) if side_dim else None
        # θ 杠杆 c:侧通道注入后的薄消息传递(0 = 关,旧检查点逐位兼容)
        self.mp = nn.ModuleList(_MPLayer(d) for _ in range(gnn_layers))
        # θ 杠杆 b:独立 θ 头(防 θ 梯度被 pg/qg/v 在共享头末层挤压)
        self.th_head = (nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
                        if theta_head else None)

    @staticmethod
    def _graph_any(flag: torch.Tensor, node_batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        """逐图"任一节点为真"[B] bool。"""
        return torch.bincount(node_batch[flag], minlength=n_graphs) > 0

    def _coverage_ok(self, have: torch.Tensor, cand: torch.Tensor, node_batch: torch.Tensor, n_graphs: int):
        """覆盖率闸 [B]:方程齐全(have)的节点数 ≥ refine_cov × 应有方程(cand)的节点数,且 cand 非空。"""
        n_c = torch.bincount(node_batch[cand], minlength=n_graphs).float()
        n_h = torch.bincount(node_batch[have & cand], minlength=n_graphs).float()
        return (n_c > 0) & (n_h >= self.refine_cov * n_c)

    def _refine(self, theta, pred, k_th, phys, edge_index, node_batch, ref_mask, reach_mask, n_graphs):
        """E12(预注册 §15.2,E6 复活;2026-08-22 第 3 轮):势重建输出上展开 refine_steps 步快速解耦潮流式修正。

        每步:
        ① P–θ 半步:用当前 θ、模型预测的 V(**停梯度**,第一版)与给定注入算节点有功失配 ΔP(全 Ybus:j 侧变比 + 并补),
           只保留"干净 P 方程"行,解 **B′·Δθ = ΔP / V**(FDPF 约定;B′ = 在运支路 1/x 的拉普拉斯,串补口径见
           `pf_edge_susceptance(negb)`;自由节点 = `refine_free`,其余 Dirichlet Δθ≡0;直接 LU + 隐式微分),θ ← θ + α_k·Δθ;
        ② 可选 Q–V 半步(refine_qv):用更新后的 θ(停梯度)与 V 算 ΔQ,解 B″·ΔV = ΔQ / V(B″ = −Im Y 在干净 Q 方程
           PQ 节点上的子块,其余 Dirichlet),V ← V + β_k·ΔV,写回 pred 的 v 列(归一化空间)。
        两道闸逐图判、互不影响:**覆盖率闸**——干净方程在自由节点(P)/ PQ 节点(Q)上的覆盖率 < refine_cov 的图本轮不修
        (§3.14 实测把无效行失配当 0 会凭空造出 +10~+424 pu 不平衡;与 E10b 同源,三个无 sidecar 的老方式库在此被跳过);
        **发散闸**——该图 max|Δθ| > refine_dth_cap / max|ΔV| > refine_dv_cap / 非有限 → 该步跳过。
        批里的边是对称化的 2M 条、edge_b 也是两份,`mismatch_correction` 组出的是 2·B′ → 权重乘 0.5(单测与 numpy 直接解对拍)。
        D15:ΔP/ΔQ 的输入只有预测量与给定量,不读被监督量的存档值(泄漏单测锁定)。返回 (θ, pred)。
        """
        from .pf_physics import pf_phys_residual, pf_phys_state        # 局部导入:pf_physics 依赖本模块的 PF_FIELDS
        from .pf_theta_edge import qv_correction

        theta = theta.float()
        pred = pred.float()
        kv = PF_FIELDS.index("v")
        B = int(n_graphs)
        eqp, free = phys["eqp_clean"], phys["refine_free"]
        target, lmask, dn = phys["target"].float(), phys["loss_mask"], phys["denorm"]
        ok_p = self._coverage_ok(eqp, free, node_batch, B)
        rows_p = eqp & free & ok_p[node_batch]
        no_ridge = torch.zeros_like(ref_mask)
        stats = dict(n_graphs=B, cov_skip_p=int((~ok_p).sum()), div_skip_p=0, max_dth=0.0)
        if self.refine_qv:
            eqq, cand_q = phys["eqq_clean"], phys["eqq_cand"]
            ok_q = self._coverage_ok(eqq, cand_q, node_batch, B)
            rows_q = eqq & ok_q[node_batch]
            stats.update(cov_skip_q=int((~ok_q).sum()), div_skip_q=0, max_dv=0.0)

        def _compose(th_col, v_col):
            cols = [pred[:, c].detach() for c in range(pred.shape[1])]
            cols[k_th] = th_col
            cols[kv] = v_col
            return torch.stack(cols, dim=-1)

        with torch.autocast(device_type=theta.device.type, enabled=False):
            for s in range(self.refine_steps):
                if bool(ok_p.any()):
                    pred_in = _compose(theta, pred[:, kv].detach())
                    dp, _dq = pf_phys_residual(pred_in, target, lmask, phys["index"], phys["g"], phys["b"], dn)
                    v_pu = pf_phys_state(pred_in, target, lmask, dn)[2].detach().clamp_min(0.5)
                    rhs = torch.where(rows_p, dp.float() / v_pu, torch.zeros_like(dp))
                    dth = mismatch_correction(rhs, edge_index, 0.5 * phys["edge_b"], node_batch, ~free,
                                              num_graphs=B, ridge_mask=no_ridge)
                    stats["max_dth"] = max(stats["max_dth"], float(torch.nan_to_num(dth.detach()[free]).abs().max()) if bool(free.any()) else 0.0)
                    dth_m = torch.where(free & ok_p[node_batch], torch.nan_to_num(dth), torch.zeros_like(dth))
                    if self.refine_ls:
                        n_rows = torch.bincount(node_batch[rows_p], minlength=B).float().clamp_min(1.0)
                        zero = torch.zeros_like(dp)

                        def _res(th_):
                            p_ = _compose(th_.detach(), pred[:, kv].detach())
                            dp_, _ = pf_phys_residual(p_, target, lmask, phys["index"], phys["g"], phys["b"], dn)
                            w_ = torch.where(rows_p, dp_.float().abs(), zero)
                            return torch.bincount(node_batch, weights=w_, minlength=B) / n_rows

                        r0 = torch.bincount(node_batch, weights=torch.where(rows_p, dp.detach().float().abs(), zero), minlength=B) / n_rows
                        scale_g = torch.zeros(B, dtype=theta.dtype, device=theta.device)
                        best = r0.clone()
                        a_s = self.refine_alpha[s].detach()
                        for sc in self.refine_ls_scales:
                            r_try = _res(theta.detach() + sc * a_s * dth_m.detach())
                            take = (scale_g == 0) & torch.isfinite(r_try) & (r_try < best)
                            scale_g = torch.where(take, torch.full_like(scale_g, sc), scale_g)
                            best = torch.where(take, r_try, best)
                        step = (scale_g[node_batch] * self.refine_alpha[s]) * dth_m
                        bad = ~torch.isfinite(step) | (step.abs() > self.refine_dth_cap)
                        bad_g = self._graph_any(bad & free, node_batch, B)
                        skip_g = bad_g | (scale_g == 0)
                        stats["div_skip_p"] += int(bad_g.sum())
                        stats["ls_skip_p"] = stats.get("ls_skip_p", 0) + int(((scale_g == 0) & ok_p).sum())
                        stats.setdefault("ls_scales", []).append([round(float(x), 2) for x in scale_g[:8]])
                        theta = theta + torch.where(skip_g[node_batch], torch.zeros_like(step), step)
                    else:
                        bad = ~torch.isfinite(dth) | (dth.abs() > self.refine_dth_cap)
                        bad_g = self._graph_any(bad & free, node_batch, B)
                        upd = free & ok_p[node_batch] & ~bad_g[node_batch]
                        stats["div_skip_p"] += int(bad_g.sum())
                        theta = theta + self.refine_alpha[s] * torch.where(upd, torch.nan_to_num(dth), torch.zeros_like(dth))
                if self.refine_qv and bool(ok_q.any()):
                    pred_in = _compose(theta.detach(), pred[:, kv])
                    _dp, dq = pf_phys_residual(pred_in, target, lmask, phys["index"], phys["g"], phys["b"], dn)
                    v_pu = pf_phys_state(pred_in, target, lmask, dn)[2]
                    rhs_q = torch.where(rows_q, dq.float() / v_pu.detach().clamp_min(0.5), torch.zeros_like(dq))
                    dv = qv_correction(rhs_q, phys["index"], phys["b"], rows_q)
                    bad = ~torch.isfinite(dv) | (dv.abs() > self.refine_dv_cap)
                    bad_g = self._graph_any(bad & rows_q, node_batch, B)
                    upd = rows_q & ~bad_g[node_batch]
                    stats["div_skip_q"] += int(bad_g.sum())
                    stats["max_dv"] = max(stats["max_dv"], float(torch.nan_to_num(dv.detach()[rows_q]).abs().max()) if bool(rows_q.any()) else 0.0)
                    v_new = v_pu + self.refine_beta[s] * torch.where(upd, torch.nan_to_num(dv), torch.zeros_like(dv))
                    v_z = (v_new - dn[kv, 0]) / dn[kv, 1]
                    pred = torch.cat([pred[:, :kv], torch.where(upd, v_z, pred[:, kv]).unsqueeze(-1), pred[:, kv + 1:]], dim=-1)
        self.refine_stats = stats
        return theta, pred

    def forward(self, h: torch.Tensor, g: torch.Tensor, e: torch.Tensor,
                node_batch: torch.Tensor, edge_batch: torch.Tensor,
                side: torch.Tensor = None, edge_index: torch.Tensor = None,
                edge_side: torch.Tensor = None, edge_use: torch.Tensor = None,
                theta_ref: torch.Tensor = None, ref_mask: torch.Tensor = None,
                reach_mask: torch.Tensor = None, return_edge: bool = False,
                phys: dict = None, v_hint: torch.Tensor = None, v_hint_ok: torch.Tensor = None):
        """h[N,d] 节点 token、g[B,K,d] 全局、e[E,d] 支路(对称化两份);返回 pred[N,4]。

        side[N,side_dim]:A 口径给定量(pf_side_features)。配置与输入必须成对——
        防止 group 视图漏供给定量 / field 视图重复供,两种静默口径混用都判错。
        edge_index[2,E]:gnn_layers>0 或 theta_edge 时必供(与 e 行对齐的对称化边);同样成对判错。
        theta_edge 时另需 edge_side[E,k]、edge_use[E]、theta_ref[N]、ref_mask[N]、reach_mask[N]
        (pf_theta_edge.pf_edge_side_features / pf_theta_ref);return_edge=True 时返回 (pred, delta[E])。
        """
        if (side is None) != (self.side_proj is None):
            raise ValueError("side 输入与 side_dim 配置不成对(A/B 口径混用)")
        need_ei = bool(len(self.mp)) or self.theta_edge
        if (edge_index is None) == need_ei:
            raise ValueError("edge_index 输入与 gnn_layers/theta_edge 配置不成对")
        if (edge_side is None) == self.theta_edge:
            raise ValueError("edge_side 输入与 theta_edge 配置不成对")
        if self.theta_edge and any(t is None for t in (edge_use, theta_ref, ref_mask, reach_mask)):
            raise ValueError("theta_edge 需要 edge_use/theta_ref/ref_mask/reach_mask")
        if (v_hint is None) == self.v_hint:
            raise ValueError("v_hint 输入与 v_hint 配置不成对")
        if self.side_proj is not None:
            h = h + self.side_proj(side)
        for mp in self.mp:
            h = mp(h, e, edge_index)
        # 工程项·同步税(2026-08-23 剖析:48 次 .item()/bool() 逐个同步占 70k 前向 57%):
        # B 直接取 g 的形状(纯 python int,零同步);nmax 与"有无参考角/无参考分量"合成一个张量、
        # **一次** .cpu() 取回,再显式传给 to_dense_batch 与势重建,消掉它们内部的逐个同步。数值逐位不变。
        B = g.shape[0]
        cn = torch.zeros(B, dtype=torch.long, device=node_batch.device).index_add_(
            0, node_batch, torch.ones_like(node_batch))     # bincount 在 CUDA 上隐式同步,index_add 不会
        ce = torch.zeros(B, dtype=torch.long, device=edge_batch.device).index_add_(
            0, edge_batch, torch.ones_like(edge_batch))
        scal = [cn.max(), ce.max()]
        if self.theta_edge:
            scal += [ref_mask.any().long(), (~reach_mask).any().long()]
        scal = torch.stack(scal).cpu()                       # 前向仅有的一次标量同步(合并取回)
        nmax_n, nmax_e = int(scal[0]), int(scal[1])
        has_ref = bool(scal[2]) if self.theta_edge else None
        has_ridge = bool(scal[3]) if self.theta_edge else None
        loc_n = torch.arange(node_batch.numel(), device=node_batch.device) - (torch.cumsum(cn, 0) - cn)[node_batch]
        loc_e = torch.arange(edge_batch.numel(), device=edge_batch.device) - (torch.cumsum(ce, 0) - ce)[edge_batch]
        hmask = torch.arange(nmax_n, device=node_batch.device)[None, :] < cn[:, None]
        emask = torch.arange(nmax_e, device=edge_batch.device)[None, :] < ce[:, None]
        hd = dense_pad(h, node_batch, B, nmax_n, loc_n, hmask)[0]
        ed = dense_pad(e, edge_batch, B, nmax_e, loc_e, emask)[0]
        ones = torch.ones(g.shape[:2], dtype=torch.bool, device=g.device)
        arch = torch.cat([hd, ed, g], dim=1)
        kpm = ~torch.cat([hmask, emask, ones], dim=1)
        q = self.queries.unsqueeze(0).expand(hd.size(0), -1, -1)
        for layer in self.layers:
            q = layer(q, arch, kpm)
        ctx = self.node_cross(self.ln_node(hd), q, q, need_weights=False)[0]
        feat = torch.cat([h, ctx[hmask]], dim=-1)
        pred = self.head(feat)
        if self.v_hint:
            # v 残差头:在 AMP 外 fp32 相加(v 目标千分之一级,fp16 分辨率不够);无效提示处 base=0
            kv = PF_FIELDS.index("v")
            base = torch.where(v_hint_ok, v_hint.float(), torch.zeros_like(v_hint, dtype=torch.float32))
            pred = torch.cat([pred[:, :kv].float(), (pred[:, kv].float() + base).unsqueeze(-1),
                              pred[:, kv + 1:].float()], dim=-1)
        k = PF_FIELDS.index("theta")
        theta = self.th_head(feat).squeeze(-1) if self.th_head is not None else pred[:, k]
        delta = None
        if self.theta_edge:
            src, dst = edge_index[0], edge_index[1]
            if self.edge_head_local:
                v_hat = pred[:, PF_FIELDS.index("v")].detach().unsqueeze(-1).to(side.dtype)
                loc_in = torch.cat([side[:, :self.n_side_local] * self.side_keep, v_hat], dim=-1)
                ln_loc = self.ln_loc(self.loc_node_proj(loc_in))
                ef = torch.cat([ln_loc[src], ln_loc[dst], self.edge_side_proj(edge_side)], dim=-1)
            else:
                fn = self.ln_he(feat)
                ef = torch.cat([fn[src], fn[dst], self.ln_e(e), self.edge_side_proj(edge_side)], dim=-1)
            # 边头与 δ 恒 fp32(AMP 外):δ ~0.05 rad 级,fp16 相对精度 1e-3 会经长路径积分放大
            with torch.autocast(device_type=ef.device.type, enabled=False):
                delta = self.edge_head(ef.float()).squeeze(-1)
                if self.edge_dc_residual:
                    delta = delta + edge_side[:, DTH_DC_COL].float()   # 以 Δθ_dc 为局部基线
                theta_rec = potential_reconstruct(delta, edge_index, edge_use.float(), node_batch,
                                                  theta_ref, ref_mask, num_graphs=B,
                                                  ridge_mask=~reach_mask, nmax=nmax_n,
                                                  has_ref=has_ref, has_ridge=has_ridge)
            theta = torch.where(reach_mask, theta_rec, theta.float())   # 无参考角分量回退到直接头
            if self.refine_steps:
                if phys is None:
                    raise ValueError("refine_steps>0 需要 phys(失配常量);训练/评估须开 --phys-weight 或显式供给")
                # E13(2026-08-22 第 3 轮消融发现:模型把活儿推给物理步,θ⁰/V⁰ 比 E10 退化 3.8×/9×):
                # 把修正前的四元组留给调用方做辅助监督(`--refine-aux-weight`),默认不用、对前向输出无影响。
                th_pre = wrap_angle(theta.float()) if self.theta_wrap else theta.float()
                self.pre_refine_pred = torch.cat([pred[:, :k].float(), th_pre.unsqueeze(-1), pred[:, k + 1:].float()], dim=-1)
                theta, pred = self._refine(theta, pred, k, phys, edge_index, node_batch,
                                           ref_mask, reach_mask, hd.size(0))
        theta = theta.float()                     # θ 与折回恒 fp32(AMP 下 fp16 在大角度分辨率不够)
        if self.theta_wrap:
            theta = wrap_angle(theta)
        pred = torch.cat([pred[:, :k].float(), theta.unsqueeze(-1), pred[:, k + 1:].float()], dim=-1)
        if return_edge:
            return pred, delta
        return pred

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def pf_supervised_cols() -> Dict[str, int]:
    return {f: NODE_TABLE.offset_of(f) for f in PF_FIELDS}


def pf_loss(pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor,
            field_weights: Dict[str, float] = None, theta_periodic: bool = False) -> torch.Tensor:
    """L1(归一化空间),只在类型化未知量位置计(5.2 节点类型损失掩码)。

    field_weights:字段级损失权重(θ 杠杆 b,防 θ 被 pg/qg/v 挤压;已实测损失竞争)。
    None = 全 1,与旧行为逐位一致;加权按 Σ w_f·|err| / Σ w_f·n_f(权重同乘分母,
    保持损失量纲,不等效于改学习率)。
    theta_periodic:θ 误差取 wrap(pred − target)(训练侧周期损失;存档 θ 为 ±π 截断值,
    2026-08-21 诊断)。False = 旧行为逐位一致。评估口径不经此函数。
    """
    cols = pf_supervised_cols()
    masks = [loss_mask[:, cols[f]] for f in PF_FIELDS]
    nsup = torch.stack([m.sum() for m in masks]).cpu()   # 同步税:4 字段的 any+sum 原先各同步一次,这里一次取回
    tot, cnt = pred.new_zeros(()), 0.0
    for k, f in enumerate(PF_FIELDS):
        m = masks[k]
        w = 1.0 if field_weights is None else float(field_weights.get(f, 1.0))
        n_f = int(nsup[k])
        if n_f and w > 0:
            err = pred[m, k] - target[m, cols[f]]
            if theta_periodic and f == "theta":
                err = wrap_angle(err)
            tot = tot + w * err.abs().sum()
            cnt += w * n_f
    return tot / max(cnt, 1e-9)


def pf_theta_hinge_loss(theta_pred: torch.Tensor, theta_true: torch.Tensor,
                        theta_hint: torch.Tensor, hint_ok: torch.Tensor,
                        sup_mask: torch.Tensor, node_batch: torch.Tensor, num_graphs: int):
    """E25(预注册 §28):图级非对称铰链 —— θ 通道"不得劣于自家零训练基线"。

    对每图 g:S_g = 被监督 ∧ 提示有效;E_model/E_hint = S_g 上 |wrap(θ−θ*)| 均值(rad);
    罚 max(0, E_model − E_hint),按 S_g 非空的图数平均。提示侧 detach(无梯度常数对手),
    梯度只经模型误差朝真值 —— 非对称:模型优于提示的图零惩罚、零拉扯(区别于把输出往
    提示拽的对称锚,§3.46(b) 改法 1 判负)。全程 fp32;CUDA index_add 容忍非确定序
    (与训练物理残差 _phys_terms 同待遇)。

    返回 (铰链标量, 激活图数张量, 参与图数):激活 = gap > 0;S_g 全空时 (0, 0, 0)。
    """
    m = sup_mask & hint_ok
    if not bool(m.any()):
        z = theta_pred.float().new_zeros(())
        return z, torch.zeros((), dtype=torch.long, device=theta_pred.device), 0
    tt = theta_true.float()
    e_m = wrap_angle(theta_pred.float() - tt).abs()
    e_h = wrap_angle(theta_hint.float().detach() - tt).abs()
    gi = node_batch[m]
    ones = e_h.new_ones(int(m.sum()))
    cnt = e_h.new_zeros(num_graphs).index_add_(0, gi, ones)
    s_m = e_m.new_zeros(num_graphs).index_add_(0, gi, e_m[m])
    s_h = e_h.new_zeros(num_graphs).index_add_(0, gi, e_h[m])
    ok = cnt > 0
    gap = (s_m[ok] - s_h[ok]) / cnt[ok]
    hinge = gap.clamp(min=0)
    return hinge.mean(), (gap.detach() > 0).sum(), int(ok.sum())


def pf_zero_learning_baseline(raw_samples: Sequence[GridSample]) -> Dict[str, Dict[str, float]]:
    """任务掩码下的零学习阶梯(D20 同款 L0/L1/L2,掩码 = D15 任务协议而非随机课程)。

    与 8.4 阶梯不可直接混排(掩码流不同);单独成表,用于"注入量可见后 θ 是否突破
    同信息零学习"的裁决。输入 = **原始(未归一化)**样本,MAE 即标幺。
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    sbit = group_bit_index(Group.STATE)
    fallback = {"pg": 0.0, "qg": 0.0, "v": 1.0, "theta": 0.0}
    errs = {f: {"L0": [], "L1": [], "L2": []} for f in PF_FIELDS}
    pooled = {f: [] for f in PF_FIELDS}
    per_sample = []
    for s in raw_samples:
        _mf, _tgt, lm = _pf_typed_mask(s.node_feat, s.node_missing)
        per_sample.append((s, lm))
        ok = ~s.node_missing[:, sbit]
        for f in PF_FIELDS:
            c = NODE_TABLE.offset_of(f)
            pooled[f].append(s.node_feat[ok, c].astype(np.float64))
    pool_mean = {f: (np.concatenate(v).mean() if v else 0.0) for f, v in pooled.items()}

    for s, lm in per_sample:
        n = s.n_nodes
        i, j = s.edge_index
        A = sp.coo_matrix((np.ones(i.size * 2), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
        A.data[:] = 1.0
        L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
        ok = ~s.node_missing[:, sbit]
        for f in PF_FIELDS:
            c = NODE_TABLE.offset_of(f)
            m = lm[:, c] & ok
            if not m.any():
                continue
            truth = s.node_feat[m, c].astype(np.float64)
            vis = ok & ~m
            vis_vals = s.node_feat[vis, c].astype(np.float64)
            errs[f]["L0"].append(np.abs(truth - pool_mean[f]))
            errs[f]["L1"].append(np.abs(truth - (vis_vals.mean() if vis_vals.size else fallback[f])))
            x = np.full(n, fallback[f], dtype=np.float64)
            x[vis] = s.node_feat[vis, c]
            if vis.any():
                I = np.where(m)[0]; B = np.where(vis)[0]
                LII = L[I][:, I].tocsc() + 1e-8 * sp.identity(I.size, format="csc")
                x[I] = spsolve(LII, -L[I][:, B] @ x[B])
            errs[f]["L2"].append(np.abs(x[m] - truth))
    return {f: {k: (float(np.concatenate(v).mean()) if v else float("nan"))
                for k, v in d.items()} for f, d in errs.items()}
