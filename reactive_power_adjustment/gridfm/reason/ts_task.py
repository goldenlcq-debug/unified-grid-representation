"""判稳任务推理器 v0(方案 5.1/5.2 稳定判别;D8/D13;交接 15 版本轮主线)。

**输入协议(与潮流任务相反,泄漏红线不同)**:判稳的运行方式是**已解方式**——解后
状态(V/θ/Pg/Qg、支路潮流)是故障前初值,属合法给定量,档案 = 标准档案(不掩码,
2.2 接口契约本义)。**泄漏红线 = 暂态产物**:输入两路(编码器路档案 + 推理器侧事件
特征)一律不得含任何暂态仿真产物——标签行的结果字段(stable/instability_types/
angle_peak/vmin/机群/CCT/…)只准出现在损失侧;事件侧特征只准由故障参数
(i, j, id, K%, Ts, Te, dur)构成。配强制泄漏单测(tests/test_ts_reasoner.py:
打乱标签行全部结果字段,输入张量必须逐位不变)。

**事件 token(方案 4.1/4.2,v0 单一"故障+切除"序列)**:线路故障 → 故障支路 token
副本(编码器支路 token,含拓扑注入,红线 4)拼连续参数 [K/100, Ts, Te, dur] 过
2 层 MLP → d 维,进档案 kv 集。支路行号经目录页 branch_row_by_key 翻译(4.4,
不进神经网络)。

**四头(交接 15 版裁决:类型按三分类;方案 5.2-b 的 v0 两类注记如实保留)**:
- 判稳二值:查询池化 → 1 logit(失稳=1,召回主指标 8.6);
- 失稳类型:3 路 multi-label(angle/voltage/frequency;仅失稳行计损——
  instability_types 可多触发,multi-label 同时覆盖"三分类"与共存);
- CCT 回归:1 标量(秒);非删失 L1,删失单侧铰链 max(0, t_max − pred)
  (7.3:禁把 t_max 当精确真值);
- 机群 multi-hot:发电机节点 token 交叉读回任务 token → **逐 token 共享打分头**
  (红线 3,指针多选 5.2-c);仅功角失稳行计损,其余类型掩码。

机群标签对齐:coherency.jsonl 的 (母线号, 机组序) → 节点 token 行 = 母线号 − 1
(assemble 惯例:token_row = 计算母线编号从 1 起);须命中 has_gen 行,不中即 raise
(标签错位比缺标签更危险)。breakup 行(窗口回退口径)旗标随行入例,如实计数。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch

from ..schema.fields import NODE_TABLE
from ..schema.sample import GridSample
from .pf_task import _ReasonerLayer

# 事件侧连续参数(唯一合法的故障侧输入;泄漏单测锁定维度与构成)
EVENT_PARAM_FIELDS = ("k_pct_over_100", "ts", "te", "dur")
EVENT_PARAM_DIM = len(EVENT_PARAM_FIELDS)
# 三大类 = 主报告口径(2026-07-14 裁决);细分五路 = 类型头输出通道
# (同日细分裁决:低压恢复/末态**合并为低电压**;过电压、高频、低频独立——
#  五子类样本量均过 7.3 升档门,账见变更记录 07-14 夜条)
TS_TYPES = ("angle", "voltage", "frequency")
TS_SUBTYPES = ("angle", "v_low", "v_over", "f_high", "f_low")
# 三大类 ← 子类聚合(voltage = v_low ∪ v_over;frequency = f_high ∪ f_low)
TS_TYPE_FROM_SUB = {"angle": (0,), "voltage": (1, 2), "frequency": (3, 4)}
FREQ_HI_PU, FREQ_LO_PU = 1.02, 0.98            # 51/49 Hz(D21 频带,50Hz 基准)
FREQ_HOLD_S = 0.5                              # D21④ 持续门(judge 同值)
CCT_T_MAX = 0.5                                # 7.3 CCT 协议上界(三相短路)

_DUR_SUFFIX = re.compile(r"_d\d+$")


def cct_join_tag(tag: str) -> str:
    """labels 行 tag 带 _dXXX 时长后缀,cct.jsonl 不带——join 前剥后缀。"""
    return _DUR_SUFFIX.sub("", tag)


def join_key(row: Dict) -> Tuple[str, str]:
    """**跨批 join 的唯一键 = (battery, 剥时长后的 tag)**(2026-07-15 对抗审查 critical)。

    tag 格式 = `{sid}_L{i}-{j}-{id}_d{dur}`,**不含 K%(故障位置)**——同一条线路的
    K=20% 与 K=0% 是**两个不同故障**却共用同一 tag:ieee39 v1 的 (34,28,23) 是 K=20,
    ieee39_v2 的同线是 K=0,实测 300 个 tag 撞车。故一切跨批字典(CCT join、去重、
    诊断索引)必须带 battery,**禁止裸 tag 全局 join**(旧注释"tag 天然不撞"是错的)。
    tag 格式本身不改(改了会让在产批的续跑 done 集失效、已产标签全部作废)。
    """
    return (str(row.get("battery")), cct_join_tag(str(row.get("tag"))))


def ts_event_params(label_row: Dict) -> np.ndarray:
    """事件侧连续参数 [EVENT_PARAM_DIM]。只消费故障参数键,结果字段一概不读。"""
    return np.array([float(label_row["k_pct"]) / 100.0, float(label_row["ts"]),
                     float(label_row["te"]), float(label_row["dur"])], dtype=np.float32)


def ts_subtype_labels(label_row: Dict) -> np.ndarray:
    """细分五路标签 [len(TS_SUBTYPES)](纯标签侧,损失监督用)。

    - v_low = voltage_modes 含 low_recover/low_steady(合并,2026-07-14 裁决);
      voltage 触发但 voltage_modes 缺席(旧批字段缺)保守记低电压(D21 低压是主导判据);
    - v_over = voltage_modes 含 overvoltage;
    - f_high/f_low:**主口径 = 触发方向的持续段时长**(freq_hold_high_s/low_s,
      judge 分方向累计;2026-07-15 补)——D21 判的是"持续 ≥0.5s",方向必须取自
      **触发失稳的那一段**,而非全窗极值:首摆瞬时冲高(不持续)叠加随后的持续跌低,
      极值代理会把两个方向都标上。字段缺席(旧批未重判)退**极值代理**并如实退化。
    """
    types = set(label_row.get("instability_types") or [])
    modes = set(label_row.get("voltage_modes") or [])
    y = np.zeros(len(TS_SUBTYPES), dtype=np.float32)
    y[0] = 1.0 if "angle" in types else 0.0
    if "voltage" in types:
        y[1] = 1.0 if ({"low_recover", "low_steady"} & modes or not modes) else 0.0
        y[2] = 1.0 if "overvoltage" in modes else 0.0
    if "frequency" in types:
        hi, lo = label_row.get("freq_hold_high_s"), label_row.get("freq_hold_low_s")
        if hi is not None or lo is not None:          # 严格口径:持续段方向
            y[3] = 1.0 if (hi or 0.0) >= FREQ_HOLD_S else 0.0
            y[4] = 1.0 if (lo or 0.0) >= FREQ_HOLD_S else 0.0
        else:                                          # 退化:全窗极值代理(旧批)
            fmax, fmin = label_row.get("fmax_pu"), label_row.get("fmin_pu")
            y[3] = 1.0 if (fmax is not None and fmax > FREQ_HI_PU) else 0.0
            y[4] = 1.0 if (fmin is not None and fmin < FREQ_LO_PU) else 0.0
    return y


def fault_branch_row(catalog, i_no: int, j_no: int, id_no: int) -> int:
    """故障支路 → token 行(目录页翻译;卡面 I/J 顺序与电池定义可能相反,双向查)。"""
    try:
        return catalog.branch_row_by_key(i_no, j_no, id_no)
    except KeyError:
        return catalog.branch_row_by_key(j_no, i_no, id_no)


def gen_node_mask(raw_sample: GridSample) -> np.ndarray:
    """发电机节点掩码 [N] bool(**原始**样本 has_gen 列;归一化后 0/1 语义被 z-score 破坏)。"""
    col = NODE_TABLE.offset_of("has_gen")
    null = raw_sample.node_feat[:, NODE_TABLE.offset_of("is_null")] > 0.5
    return (raw_sample.node_feat[:, col] > 0.5) & ~null


def group_multi_hot(coh_row: Dict, raw_sample: GridSample) -> Tuple[np.ndarray, bool]:
    """coherency 行 → 逐节点 multi-hot [N](发电机行上取 1)。

    返回 (y_group, breakup)。(bus, g) → 节点行 = bus − 1;未命中 has_gen 行即 raise
    (错位标签静默入训是比缺标签更坏的故障)。
    """
    n = raw_sample.n_nodes
    y = np.zeros(n, dtype=np.float32)
    gm = gen_node_mask(raw_sample)
    for bus, _g in coh_row.get("unstable_group") or []:
        row = int(bus) - 1
        if row < 0 or row >= n or not gm[row]:
            raise ValueError(f"机群标签错位:{coh_row.get('tag')} 机 ({bus},{_g}) → 行 {row} "
                             f"非发电机节点(has_gen=0 或越界)")
        y[row] = 1.0
    return y, bool(coh_row.get("breakup"))


# 标签行里除 join/故障键外的一切字段都视为"结果字段"(泄漏单测据此打乱)
TS_FAULT_KEYS = ("tag", "sample_id", "battery", "i", "j", "id", "k_pct", "ts", "te", "dur")


def build_ts_example(label_row: Dict, raw_sample: GridSample, catalog,
                     coh_row: Optional[Dict] = None,
                     cct_row: Optional[Dict] = None) -> Optional[Dict]:
    """一行判稳标签 → 训练例。输入侧只消费 TS_FAULT_KEYS;结果字段只落 y_*。

    infeasible/timeout 行返回 None(不入稳/失稳统计,7.3)。
    """
    if not label_row.get("feasible") or label_row.get("timeout"):
        return None
    ex: Dict = {
        "tag": label_row["tag"], "sample_id": label_row["sample_id"],
        "battery": label_row.get("battery"),
        "fault_branch_row": fault_branch_row(catalog, int(label_row["i"]),
                                             int(label_row["j"]), int(label_row["id"])),
        "event_params": ts_event_params(label_row),
        "y_unstable": 0.0 if label_row["stable"] else 1.0,
        "y_subtypes": ts_subtype_labels(label_row),
        "y_cct": float("nan"), "cct_valid": False, "cct_censored": False,
        "y_group": None, "group_valid": False, "breakup": False,
    }
    if cct_row is not None and not cct_row.get("infeasible") and cct_row.get("cct") is not None:
        ex["y_cct"] = float(cct_row["cct"])
        ex["cct_valid"] = True
        ex["cct_censored"] = bool(cct_row.get("is_censored"))
    if (coh_row is not None and ex["y_unstable"] > 0.5 and ex["y_subtypes"][0] > 0.5
            and not coh_row.get("needs_review")):
        y, breakup = group_multi_hot(coh_row, raw_sample)
        ex["breakup"] = breakup
        # **回退行不入机群头训练**(2026-07-15 对抗审查 major)。协议 v1.1 规定:
        # breakup / onset_at_edge 行的 unstable_group 语义**回退为"窗口起始失步提名"**
        # (双侧 90° 偏移峰值),而正常行是"末态真失步机集合"(±500° 偏离基线)——
        # 两种语义混进同一个共享打分头会互相教坏(前者是"谁先摆开",后者是"谁最终失步";
        # 实产 594/2269 = 26% 是回退行,占比不小)。方案 7.3 明写"下游须知悉其回退语义",
        # 此处即该知悉的落地:**回退行如实计数、不产训练标签**(不静默混用)。
        if y.any() and not breakup and not coh_row.get("onset_at_edge"):
            ex["y_group"], ex["group_valid"] = y, True
    return ex


class TSReasoner(nn.Module):
    """查询 token 交叉读档案(节点+支路+全局+事件 token)→ 四头(结构随 5.3 基准)。"""

    def __init__(self, d: int, n_queries: int = 16, n_layers: int = 2,
                 n_heads: int = 4, ffn_mult: int = 2,
                 event_dim: int = EVENT_PARAM_DIM) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)
        self.event_mlp = nn.Sequential(nn.Linear(d + event_dim, d), nn.GELU(),
                                       nn.Linear(d, d))
        self.layers = nn.ModuleList(_ReasonerLayer(d, n_heads, ffn_mult)
                                    for _ in range(n_layers))
        self.ln_pool = nn.LayerNorm(d)
        self.head_stable = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # 细分五路(07-14 裁决;三大类由子类聚合报告,不设独立头)
        self.head_type = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                       nn.Linear(d, len(TS_SUBTYPES)))
        self.head_cct = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # 机群:节点 token 交叉读回任务 token → 逐 token 共享打分头(红线 3)
        self.ln_node = nn.LayerNorm(d)
        self.node_cross = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.head_group = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, h: torch.Tensor, g: torch.Tensor, e: torch.Tensor,
                node_batch: torch.Tensor, edge_batch: torch.Tensor,
                fault_edge_ptr: torch.Tensor, event_params: torch.Tensor,
                ablate_graph: bool = False,
                ablate_event_edge: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """h[N,d]/g[B,K,d]/e[E,d] 冻结档案;fault_edge_ptr[B] = 故障支路在 e 的全局行
        (调用方按 edge_batch 段起点 + 图内行号折算);event_params[B,EVENT_PARAM_DIM]。

        返回 (stable_logit[B], type_logit[B,3], cct[B], group_logit[N])。
        """
        e_fault = e[fault_edge_ptr]
        if ablate_event_edge:
            # **诊断开关 2**(2026-07-14 对抗审查 critical 补):事件 token 是由故障支路的
            # **完整档案嵌入** e[fault_edge_ptr](含 r/x/b_half/潮流 + 8 层 GPS 全图条件化,
            # 红线 4 拓扑注入)拼 [K,Ts,Te,dur] 构成的 —— 它是一条**绕过档案交叉注意力**的
            # 图信息旁路。ablate_graph 只屏蔽了档案三段、**从没碰这条旁路**,所以"屏蔽图后
            # 仍有 1.36"不是"图无用"的反证。此开关把 e[fault_edge_ptr] 置零,只留事件参数
            # [K,Ts,Te,dur] —— 与 ablate_graph 合用即"零图信息",组内应塌回 ≈1.000。
            e_fault = torch.zeros_like(e_fault)
        ev = self.event_mlp(torch.cat([e_fault,
                                       event_params.to(e.dtype)], dim=-1))       # [B,d]
        hd, hmask = to_dense_batch(h, node_batch)
        ed, emask = to_dense_batch(e, edge_batch)
        ones_g = torch.ones(g.shape[:2], dtype=torch.bool, device=g.device)
        ones_e = torch.ones(ev.shape[0], 1, dtype=torch.bool, device=g.device)
        arch = torch.cat([hd, ed, g, ev.unsqueeze(1)], dim=1)
        kpm = ~torch.cat([hmask, emask, ones_g, ones_e], dim=1)
        if ablate_graph:
            # 屏蔽档案的节点/支路/全局三段(**不含**事件 token 里的故障支路旁路,见上)。
            # 机群头(node_cross)不受此开关影响(它本就必须读节点)。
            kpm = kpm.clone()
            kpm[:, :hd.size(1) + ed.size(1) + g.size(1)] = True
        q = self.queries.unsqueeze(0).expand(hd.size(0), -1, -1)
        for layer in self.layers:
            q = layer(q, arch, kpm)
        pooled = self.ln_pool(q).mean(dim=1)                              # [B,d]
        s_logit = self.head_stable(pooled).squeeze(-1)
        t_logit = self.head_type(pooled)
        cct = self.head_cct(pooled).squeeze(-1)
        ctx = self.node_cross(self.ln_node(hd), q, q, need_weights=False)[0]
        feat = torch.cat([h, ctx[hmask]], dim=-1)
        grp_logit = self.head_group(feat).squeeze(-1)                     # [N]
        return s_logit, t_logit, cct, grp_logit

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def ts_loss(s_logit: torch.Tensor, t_logit: torch.Tensor, cct: torch.Tensor,
            grp_logit: torch.Tensor, batch, weights: Dict[str, float] = None,
            ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """四头合成损失。掩码纪律:类型头仅失稳行;机群头仅功角失稳且标签有效行
    (且只在发电机节点位置);CCT 仅有标签行,删失走单侧铰链(7.3)。
    """
    w = {"stable": 1.0, "type": 0.5, "cct": 1.0, "group": 0.5, **(weights or {})}
    diag: Dict[str, float] = {}
    # AMP 下前向输出 fp16 而本函数可在 autocast 外调用:损失一律 fp32 计
    s_logit, t_logit = s_logit.float(), t_logit.float()
    cct, grp_logit = cct.float(), grp_logit.float()
    y_un = batch.y_unstable
    loss = w["stable"] * F.binary_cross_entropy_with_logits(s_logit, y_un)
    diag["stable"] = float(loss.detach())

    t_mask = y_un > 0.5
    if t_mask.any() and w["type"] > 0:
        # B1(2026-07-17 过账):稀有子类(f_low ~4% 失稳行且单源家族)在无加权 BCE 下
        # 会种子性塌缩;weights["type_pos_weight"] = 逐子类正类权重(仅乘正类项),
        # 缺省 None = 历史行为逐位不变
        pw = w.get("type_pos_weight")
        pw_t = (torch.as_tensor(pw, dtype=t_logit.dtype, device=t_logit.device)
                if pw else None)
        lt = F.binary_cross_entropy_with_logits(t_logit[t_mask], batch.y_subtypes[t_mask],
                                                pos_weight=pw_t)
        loss = loss + w["type"] * lt
        diag["type"] = float(lt.detach())

    c_mask = batch.cct_valid
    if c_mask.any() and w["cct"] > 0:
        cen = batch.cct_censored & c_mask
        non = c_mask & ~batch.cct_censored
        terms = []
        if non.any():
            terms.append((cct[non] - batch.y_cct[non]).abs().sum())
        if cen.any():   # 删失:真值 ≥ t_max,预测低于 t_max 才罚(单侧铰链)
            terms.append(F.relu(CCT_T_MAX - cct[cen]).sum())
        lc = sum(terms) / int(c_mask.sum())
        loss = loss + w["cct"] * lc
        diag["cct"] = float(lc.detach())

    g_rows = batch.group_valid[batch.batch] & batch.gen_mask   # [N]:有效行 × 发电机位
    if g_rows.any() and w["group"] > 0:
        # B6(2026-07-17 过账):机群行正类率 ~0.39~0.52,低正例率机器在无加权 BCE 下
        # 不点火(B5 诊断:118 五台中频机三种子零发);weights["group_pos_weight"] =
        # 全局标量正类权重(仅乘正类项),缺省 None = 历史行为逐位不变
        gpw = w.get("group_pos_weight")
        gpw_t = (torch.as_tensor(gpw, dtype=grp_logit.dtype, device=grp_logit.device)
                 if gpw else None)
        lg = F.binary_cross_entropy_with_logits(grp_logit[g_rows], batch.y_group[g_rows],
                                                pos_weight=gpw_t)
        loss = loss + w["group"] * lg
        diag["group"] = float(lg.detach())
    diag["total"] = float(loss.detach())
    return loss, diag


def fault_edge_pointer(edge_batch: torch.Tensor, fault_row: torch.Tensor) -> torch.Tensor:
    """图内支路行号 → 批内 e 全局行。e 每图先正向 M 行后反向 M 行(sample_to_data
    对称化),批内按图连续;段起点 = edge_batch 计数前缀和,正向副本行 = 段起点 + 行号。
    """
    counts = torch.bincount(edge_batch, minlength=int(fault_row.numel()))
    starts = torch.cumsum(counts, 0) - counts
    return starts[torch.arange(fault_row.numel(), device=fault_row.device)] + fault_row


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC(AP 口径,手写免依赖;8.6 辅指标)。**并列分数按整组结算**。

    **2026-07-14 对抗审查 major(已复现)**:旧写法 `argsort(kind="stable")` 对并列
    分数保留**原数组顺序**,而 out_exs 的排列 = 标签批的拼接序(ts_labels_ieee39 300 行
    在前、_v2 10,200 行在后),与真标签强相关 → **凭空刷出判别力**:实测 [外] 上一个
    **恒定预测器**(全输出 1.0,零信息)的 AP = 0.6781,而基率只有 0.6635 → 白得
    1.022× 的"PR-AUC/base 判别力"。模型也会吃到这份红利(fp32 sigmoid 在 |logit|≳17
    时精确饱和到 1.0,实测 [外] 召回 0.9951 的那些 run 里大量样本分数逐位相同)。

    修法 = 并列组内取**平均精确率**(与 sklearn.average_precision_score 的秩序无关口径
    同源):把同分样本视为一个整体,该组内不假设任何顺序。单测锁定:恒定分数的 AP
    必须**恰等于基率**;打乱输入顺序不得改变 AP。
    """
    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    denom = y_true.sum()
    if not denom:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    y = y_true[order]
    s = score[order]
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    prec = tp / k
    # 并列组:同分样本共享该组**末端**的精确率(组内顺序不可知,不得假设"正例排前")
    # np.unique 的 index 给出每个分数组的首元素位置;组末位置 = 下一组首 − 1
    _vals, first = np.unique(-s, return_index=True)
    last = np.append(first[1:], len(y)) - 1
    grp_prec = np.empty_like(prec)
    for f, l in zip(first, last):
        grp_prec[f:l + 1] = prec[l]
    return float((grp_prec * y).sum() / denom)
