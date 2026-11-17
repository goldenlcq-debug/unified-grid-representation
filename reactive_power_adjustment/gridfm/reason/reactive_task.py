"""无功调整推理器(方向二选项 B;输出机组机端电压设定调整 Δv_set 修电压越限)。

**第二杠杆(shunt)**:with_shunt=True 时任务变为**混合动作** —— 连续 Δv_set + 离散
投切并联补偿。侧通道加第三路 = shunt 候选掩码(该母线有停运并补,输入自身确定函数),
网络加第三个头 = shunt 指针(ReactiveShuntReasoner),损失加 shunt BCE(仅候选位)。
动机:v_set 单杠杆的规则老师只到 0.61 且约 68/120 个训练方式一个都修不动(打不出标签),
训练数据多样性被老师能力范围封顶;加 shunt 同时抬老师(实测 0.74)与解锁那批方式的打标。

与有功推理器(adjust_task)同构,三处差异(见立项草案第 2 版 + S0 探针结论):
1. **双边越限深度侧通道** viol_depth_2s → [N,2](低压深度 + 过压深度),推理器用
   AdjustReasoner(side_dim=2);有功是单边 [N]。
2. **作动器 = 机端电压设定 v_set**(GenView.v0),非机组有功 Pg;标签 = Δv_set。
3. **数值头归一 = 电压带宽常数 VSET_SPAN**(非有功的机组容量幅度;S0 审查:照搬功率量纲会
   使目标落 1e-4 被指针 BCE 淹没)。

判据双边(S0 定):健康 = 收敛且 vmin>=VIO_LOW 且 vmax<=VIO_HIGH(VIO_HIGH=1.08 由 S0 实测
lib118 基态 vmax<=1.0700 定,给修法 0.01pu 裕度且不纳入任何过压基态)。

**输入协议 / 泄漏红线(同 adjust,口径不变)**:病态方式是已解方式,V/θ/v_set 是合法给定量,
档案不掩码;唯一输入侧任务条件 = viol_depth_2s(输入自身确定函数,D13 计算型特征);Δv_set
集合与数值只准出现在损失侧(泄漏单测打乱含过压深度在内的全部标签字段,验证两深度通道逐位不变)。
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..schema.fields import BRANCH_TABLE, BRANCH_TYPES, NODE_TABLE
from ..schema.sample import GridSample
from .adjust_task import AdjustReasoner, adjust_loss  # 网络与损失结构复用(side_dim=2)
from .ts_task import gen_node_mask

VIO_LOW = 0.93
VIO_HIGH = 1.08          # S0 实测定(lib118 基态 vmax<=1.0700;取代草案 RD-2 的 1.07)
VSET_SPAN = 0.10         # Δv_set 归一带宽常数(pu);典型 Δv_set 0.01~0.05 → 目标 O(0.1~0.5)

# 数值头分类档位(pu,带符号)。规则打标器的步长梯度是 _STEP_LADDER=(0.02,0.05),
# 故标签的逐机 Δv_set **只有离散取值**:实测 675 条标签共 2529 个机组位,
# 全部落在 +0.02(2379)/ +0.05(150)两个值上(94:6;负档来自过压方向,当前数据未出现)。
# **为什么要分类**:回归头对这种双峰离散目标必然塌缩——L1 的最优解是中位数(0.020),
# 实测模型输出挤在 0.0201±0.0006、8 机大动作 0 例,而深病态恰恰要 0.05 的大步。
# 分类头让"大动作"成为一个可被选中的类,而不是被平均掉的尾巴。
DV_LEVELS = (-0.05, -0.02, 0.02, 0.05)

# 标签行里除输入定位键外一切修复字段视为结果字段(泄漏单测据此打乱)
REACT_INPUT_KEYS = ("sid", "battery", "tag")


def viol_depth_2s(raw_sample: GridSample, vio_low: float = VIO_LOW,
                  vio_high: float = VIO_HIGH) -> np.ndarray:
    """逐节点双边越限深度 [N,2] = [max(0, vio_low−V), max(0, V−vio_high)];
    失电(V<=0.01)/NULL 行两列均 0。输入自身确定函数(只读电压列,不读任何标签)。"""
    noff = NODE_TABLE.offset_of
    v = raw_sample.node_feat[:, noff("v")].astype(np.float64)
    null = raw_sample.node_feat[:, noff("is_null")] > 0.5
    low = np.maximum(0.0, vio_low - v)
    high = np.maximum(0.0, v - vio_high)
    dead = (v <= 0.01) | null
    low[dead] = 0.0
    high[dead] = 0.0
    return np.stack([low, high], axis=1).astype(np.float32)


def shunt_cand_mask(raw_sample: GridSample) -> np.ndarray:
    """逐节点 shunt 候选掩码 [N] float32:该母线挂有**停运的并联补偿**(可投切)则 1。

    第二杠杆(投并联补偿)的可动集合。**纯输入函数**:只读档案支路表的
    branch_type onehot(shunt_comp)+ mark(停运)+ edge_index(自环 i==j 定位母线),
    不读任何标签字段 —— 与 viol_depth_2s 同属 D13 计算型侧通道,不破泄漏红线。

    实测(lib118,archive 口径与 LF.L2 卡片口径逐位一致):每方式 14 个并联补偿全停运,
    其中 12 个容性(X<0,抬压)在母线 20/21/25/30/59/61/62/63/74/107/111/113,
    2 个感性(X>0)在母线 57/71。容/感不在掩码里区分——方向由双边越限深度通道给出,
    X 的符号本身也在冻结编码器可见的支路特征里。
    """
    n = raw_sample.n_nodes
    mask = np.zeros(n, dtype=np.float32)
    bf = raw_sample.branch_feat
    if bf.shape[0] == 0:
        return mask
    boff = BRANCH_TABLE.offset_of
    is_shunt = bf[:, boff("branch_type") + BRANCH_TYPES.index("shunt_comp")] > 0.5
    off_service = bf[:, boff("mark")] != 1.0
    rows = np.where(is_shunt & off_service)[0]
    for r in rows:
        bus_row = int(raw_sample.edge_index[0, r])
        if 0 <= bus_row < n:
            mask[bus_row] = 1.0
    return mask


def react_side(raw_sample: GridSample, vio_low: float = VIO_LOW,
               vio_high: float = VIO_HIGH, with_shunt: bool = False) -> np.ndarray:
    """推理器侧通道 [N,2](v_set 单杠杆)或 [N,3](加 shunt 候选掩码,第二杠杆)。
    全部为输入自身的确定函数;side_dim 随之为 2 / 3。"""
    d2 = viol_depth_2s(raw_sample, vio_low, vio_high)
    if not with_shunt:
        return d2
    return np.concatenate([d2, shunt_cand_mask(raw_sample)[:, None]], axis=1).astype(np.float32)


def unit_span_vset(raw_sample: GridSample) -> np.ndarray:
    """Δv_set 归一分母 [N] = 电压带宽常数(机组节点 VSET_SPAN,非机组 1 防除零)。
    量纲与数量级与 v_set 一致,避免照搬有功功率幅度导致目标塌到 1e-4。"""
    span = np.ones(raw_sample.n_nodes, dtype=np.float32)
    span[gen_node_mask(raw_sample)] = VSET_SPAN
    return span


def build_reactive_example(label_row: Dict, raw_sample: GridSample,
                           tag: str) -> Optional[Dict]:
    """一行无功调整标签 + 病态方式档案 → 训练例。输入侧只消费档案;标签只落 y_*。

    y_units [N] = 全部可行修法机组并集 multi-hot(节点行 = 母线号 − 1,须 has_gen,错位即
    raise);y_dv [N] = 规范解 Δv_set / VSET_SPAN(仅规范解机组位有效,y_dv_mask 标注)。
    与 build_adjust_example 同构,仅把 ΔP/容量幅度 换成 Δv_set/带宽常数。
    """
    if not label_row.get("feasible"):
        return None
    n = raw_sample.n_nodes
    gm = gen_node_mask(raw_sample)
    span = unit_span_vset(raw_sample)
    y_units = np.zeros(n, dtype=np.float32)
    for fx in label_row["fixes"]:
        for bus_s in fx["deltas"]:
            row = int(bus_s) - 1
            if row < 0 or row >= n or not gm[row]:
                raise ValueError(f"无功标签错位:{tag} 母线 {bus_s} → 行 {row} 非机组节点")
            y_units[row] = 1.0
    canon = label_row["canonical_fix"]
    y_dv = np.zeros(n, dtype=np.float32)
    y_dv_mask = np.zeros(n, dtype=bool)
    # **越界/机组位校验**(2026-07-25 审查 F4-④):与 y_units 循环同款纪律。
    # 缺校验时母线号 '0' 会以 row=-1 负索引静默写到最后一行、非机组母线会静默进入
    # y_dp_mask —— 而 D² 的排序正例与规模头目标都以 y_dp_mask∩gen 为准,
    # y_dp_mask⊆gen_mask 此前只靠数据巧合成立(错位标签比缺标签更坏)。
    for bus_s, dv in canon["deltas"].items():
        row = int(bus_s) - 1
        if row < 0 or row >= n or not gm[row]:
            raise ValueError(f"无功规范解标签错位:{tag} 母线 {bus_s} → 行 {row} 非机组节点")
        y_dv[row] = float(dv) / span[row]
        y_dv_mask[row] = True
    # 数值头分类目标:规范解位上取最近档位的下标;非规范解位填 0(由 y_dp_mask 掩掉不计损)
    y_dv_cls = np.zeros(n, dtype=np.int64)
    lv = np.asarray(DV_LEVELS, dtype=np.float64)
    for bus_s, dv in canon["deltas"].items():
        row = int(bus_s) - 1
        y_dv_cls[row] = int(np.argmin(np.abs(lv - float(dv))))
    ex = {"tag": tag, "sid": label_row["sid"], "battery": label_row["battery"],
          "y_units": y_units, "y_dp": y_dv, "y_dp_mask": y_dv_mask,
          "y_dv_cls": y_dv_cls,
          "n_fixes": int(label_row.get("n_fixes", 1))}
    # 第二杠杆:投切并联补偿的 multi-hot(全部可行修法的并集,与 y_units 同口径)。
    # 只落 y_*;候选掩码从档案算(shunt_cand_mask),不来自标签。
    cand = shunt_cand_mask(raw_sample)
    y_shunt = np.zeros(n, dtype=np.float32)
    for fx in label_row["fixes"]:
        for bus in fx.get("shunt", []):
            row = int(bus) - 1
            if row < 0 or row >= n or cand[row] < 0.5:
                raise ValueError(f"shunt 标签错位:{tag} 母线 {bus} → 行 {row} 非停运并补候选")
            y_shunt[row] = 1.0
    ex["y_shunt"] = y_shunt
    ex["shunt_cand"] = cand
    return ex


def decode_action(gen_mask: np.ndarray, ptr_prob: np.ndarray, dv_pred: np.ndarray,
                  span: np.ndarray, shunt_cand: Optional[np.ndarray] = None,
                  shunt_prob: Optional[np.ndarray] = None,
                  thr: float = 0.5, dv_levels: Optional[Sequence[float]] = None,
                  top_k: Optional[int] = None) -> Dict:
    """预测张量 → 混合动作 {"dvset": {母线: Δv_set}, "shunt": [母线]}(1 基母线号)。

    两条口径(训练导出与 PSASP 验收必须共用,否则两处会漂):
    1. **机组指针**过阈即选;shunt 指针过阈且在候选位才选(不可投的位置不选)。
    2. **机组兜底只在完全没有动作时才补**(既没选机组、也没选并补)——若 shunt 头已点火
       就不再强塞一台机组:否则"只投并补、不动 v_set"这一类动作(老师的纯并补修法,
       去风险 13 例救回中占多数)学生在动作空间上根本表达不出来,而且会叠上一个在该样本
       上无监督的寄生 Δv_set;lib118 基态 vmax 可达 1.07、判据 1.08 只剩 0.01pu 裕度,
       这个寄生量足以把本可通过的例子顶穿判败。
    """
    sel_shunt: list = []
    if shunt_cand is not None and shunt_prob is not None:
        sel_shunt = [int(i) + 1 for i in
                     np.where((shunt_cand > 0.5) & (shunt_prob > thr))[0]]
    if top_k is not None:
        # D② top-k 口径:机组数由规模头显式给出(0 = 纯并补),按指针分取前 k
        rows = np.where(gen_mask)[0]
        order = rows[np.argsort(-ptr_prob[rows], kind="stable")]
        sel_gen = [int(i) for i in order[:max(0, int(top_k))]]
    else:
        sel_gen = [int(i) for i in np.where(gen_mask & (ptr_prob > thr))[0]]
    if not sel_gen and not sel_shunt:
        sel_gen = [int(np.where(gen_mask)[0][np.argmax(ptr_prob[gen_mask])])]
    if dv_levels is not None:
        # 分类版:取 argmax 档位的**物理值**(pu),不再乘 span(档位本身就是 pu)
        lv = np.asarray(dv_levels, dtype=np.float64)
        dvset = {int(i) + 1: round(float(lv[int(np.argmax(dv_pred[i]))]), 4)
                 for i in sel_gen}
    else:
        dvset = {int(i) + 1: round(float(dv_pred[i]) * float(span[i]), 4)
                 for i in sel_gen}
    return {"dvset": dvset, "shunt": sel_shunt}


class ReactiveShuntReasoner(AdjustReasoner):
    """无功推理器(双杠杆版):在 AdjustReasoner 的两个头之外加**第三个头 = shunt 指针**。

    共用主干 node_feat(不改父类参数集合,故 v0.6 有功检查点仍 strict load);侧通道
    side_dim=3 = [低压深度, 过压深度, shunt 候选掩码]。forward 返回三元组
    (ptr_logit[N], dv_pred[N], shunt_logit[N]),与父类两元组显式区分。
    """

    def __init__(self, d: int, n_queries: int = 16, n_layers: int = 2,
                 n_heads: int = 4, ffn_mult: int = 2, side_dim: int = 3,
                 n_dv_levels: int = 0) -> None:
        super().__init__(d=d, n_queries=n_queries, n_layers=n_layers, n_heads=n_heads,
                         ffn_mult=ffn_mult, side_dim=side_dim)
        self.head_shunt = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        # n_dv_levels=0 → 数值头保持回归([N] 标量,与既有检查点逐位兼容);
        # >0 → 换成 K 路分类头([N,K] logits),治双峰离散目标下的 L1 幅度塌缩
        self.n_dv_levels = int(n_dv_levels)
        if self.n_dv_levels > 0:
            self.head_dp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                         nn.Linear(d, self.n_dv_levels))

    def forward(self, h, g, e, node_batch, edge_batch, depth):
        """返回 (ptr_logit[N], dv[N] 或 dv_logits[N,K], shunt_logit[N])。"""
        feat = self.node_feat(h, g, e, node_batch, edge_batch, depth)
        dv = self.head_dp(feat)
        dv = dv if self.n_dv_levels > 0 else dv.squeeze(-1)
        return (self.head_ptr(feat).squeeze(-1), dv, self.head_shunt(feat).squeeze(-1))


# D② 集合规模头上限。**支撑实测(2026-07-25 深度审查)**:675 条标签的规范解机组数
# 分布 = {1:263, 2:157, 4:22, 8:233},**k=0(纯并补)零支撑**(rulelabel 无 shunt-only
# 修法),3/5/6/7/9~12 亦零支撑。故"k=0 = 纯并补"只是解码链保留的能力,**在当前数据
# 分布下不可达**,不得作为 D² 的卖点宣称(审查发现 A-8,宣称已收回)。
# 留 12 是给未来扩标签的裕量;零支撑类由 balanced 类别权重处理(见 rank_loss)。
K_MAX_UNITS = 12
K_SUPPORTED = (1, 2, 4, 8)   # 当前标签分布的有支撑档(逐类点火率须实测,防死头)


class ReactiveRankReasoner(ReactiveShuntReasoner):
    """D②(2026-07-24 用户批路线一):指针头集合化,治"选错机组集合 + 选太多"。

    诊断依据(runs/diag_action_gap.json,2026-07-21):老师机组+模型幅度救 6/9、模型机组+
    最大幅度救 0/9(病在指针头);模型点 6~8 台 vs 老师 4 台(BCE 逐点独立打分 + 正类
    加权过冲,表达不了"集合该多大");调权重三组实测无效,须动结构。

    两处结构改动(输入协议/编码器不动):
    1. 指针监督由逐点 BCE 换**图内成对排序**(规范解位 vs 非解位,logistic rank)——
       只学"谁该排前面",不受正负比/阈值刻度绑架;
    2. 新增**集合规模头 head_k**(全图节点特征均值池化 → 0..K_MAX 分类,目标 = 规范解
       机组数;0 = 纯并补)——解码改 top-k,机组数显式受监督,不再由阈值副作用决定。
    """

    def __init__(self, d: int, n_queries: int = 16, n_layers: int = 2,
                 n_heads: int = 4, ffn_mult: int = 2, side_dim: int = 3,
                 n_dv_levels: int = 0, n_k_max: int = K_MAX_UNITS) -> None:
        super().__init__(d=d, n_queries=n_queries, n_layers=n_layers, n_heads=n_heads,
                         ffn_mult=ffn_mult, side_dim=side_dim, n_dv_levels=n_dv_levels)
        self.n_k_max = int(n_k_max)
        self.head_k = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(),
                                    nn.Linear(d, self.n_k_max + 1))

    def forward(self, h, g, e, node_batch, edge_batch, depth, n_graphs=None):
        """返回 (ptr_logit[N], dv[N] 或 [N,K], shunt_logit[N], k_logits[B,n_k_max+1])。

        **B 必须来自批的真实图数**(n_graphs,调用方传 batch.num_graphs):用
        node_batch.max()+1 推断,在"批尾有空图(零节点)"时会缩水,末尾图静默失去 k
        监督(2026-07-25 审查 F4-③)。缺省 None 时退回 max()+1 并保持旧行为。
        **池化在 fp32 累加**:autocast 下 feat 是 fp16,index_add_ 逐节点累加 O(1e2~1e3)
        项会吃掉千分位精度(同上审查 F4-③);算完再按 feat.dtype 交回下游头。
        """
        feat = self.node_feat(h, g, e, node_batch, edge_batch, depth)
        dv = self.head_dp(feat)
        dv = dv if self.n_dv_levels > 0 else dv.squeeze(-1)
        nb = node_batch
        B = int(n_graphs) if n_graphs is not None else (int(nb.max()) + 1 if nb.numel() else 0)
        f32 = feat.float()
        pooled = torch.zeros(B, f32.shape[-1], dtype=torch.float32, device=feat.device)
        pooled.index_add_(0, nb, f32)
        cnt = torch.zeros(B, dtype=torch.float32, device=feat.device)
        cnt.index_add_(0, nb, torch.ones_like(nb, dtype=torch.float32))
        pooled = (pooled / cnt.clamp_min(1.0).unsqueeze(-1)).to(feat.dtype)
        return (self.head_ptr(feat).squeeze(-1), dv,
                self.head_shunt(feat).squeeze(-1), self.head_k(pooled))


def build_reactive_reasoner(d: int, n_queries: int = 16, n_layers: int = 2,
                            with_shunt: bool = False, n_dv_levels: int = 0,
                            ptr_rank: bool = False, **kw) -> AdjustReasoner:
    """无功推理器。缺省 = v_set 单杠杆(AdjustReasoner,side_dim=2,与 v0/v2 检查点兼容);
    with_shunt=True = 双杠杆(ReactiveShuntReasoner,side_dim=3,三头);
    n_dv_levels>0 = 数值头改 K 路分类(仅双杠杆版支持);
    ptr_rank=True = D② 集合化(ReactiveRankReasoner,四头,要求 with_shunt+n_dv_levels)。"""
    if ptr_rank:
        if not (with_shunt and n_dv_levels):
            raise ValueError("D② ptr_rank 建立在 v5 配方之上(须 with_shunt + dv_cls)")
        return ReactiveRankReasoner(d=d, n_queries=n_queries, n_layers=n_layers,
                                    side_dim=3, n_dv_levels=n_dv_levels, **kw)
    if with_shunt:
        return ReactiveShuntReasoner(d=d, n_queries=n_queries, n_layers=n_layers,
                                     side_dim=3, n_dv_levels=n_dv_levels, **kw)
    if n_dv_levels:
        raise ValueError("数值头分类目前只在 with_shunt 版实现(单杠杆版保持回归以兼容 v0/v2)")
    return AdjustReasoner(d=d, n_queries=n_queries, n_layers=n_layers, side_dim=2, **kw)


# 损失结构与有功一致(指针 BCE 多正标签 + 数值头 L1 仅规范解位);复用 adjust_loss。
reactive_loss = adjust_loss


def reactive_shunt_loss(ptr_logit: torch.Tensor, dv_pred: torch.Tensor,
                        shunt_logit: torch.Tensor, batch, w_ptr: float = 1.0,
                        w_dp: float = 1.0, w_shunt: float = 1.0,
                        pos_weight: Optional[float] = None,
                        dv_class_weight: Optional[torch.Tensor] = None,
                        ptr_pos_weight: Optional[float] = None):
    """双杠杆损失 = reactive_loss(机组指针 BCE + Δv_set L1)+ **shunt BCE 仅候选位**。

    候选位以外(该母线没有停运并补)不计损:不可动的位置给不出监督信号,计了只会
    让头学"到处都不投"的先验。无候选的样本 shunt 项为 0。

    **pos_weight**:shunt 正例极稀(实测合并集 100/9450 = 1.06%),裸 BCE 下"永不投切"
    就能拿到 98.9% 正确率,头会塌成恒不点火(判稳线 B1 同款坑,当年用子类加权修好)。
    None = 不加权(缺省,逐位可复现);给正数则作为正类权重传给 BCE。
    """
    if dv_pred.dim() > 1:
        # 数值头分类版:指针 BCE 照旧,Δv_set 换成**仅规范解位的交叉熵**
        gm = batch.gen_mask
        # 机组指针正例率实测 19.7%(2529/12825);标签里 233 例要同时点 8 台机,
        # 而 BCE 在 4:1 负正比下倾向少点火(实测模型最多只点 3 台,老师 35% 的标签点 8 台)
        # —— 深病态要"8 台合力"才修得动,故给正类加权(缺省 None = 不加权,逐位可复现)。
        pw_ptr = (torch.as_tensor(float(ptr_pos_weight), device=ptr_logit.device,
                                  dtype=ptr_logit.dtype)
                  if ptr_pos_weight is not None else None)
        ptr = F.binary_cross_entropy_with_logits(ptr_logit[gm], batch.y_units[gm],
                                                 pos_weight=pw_ptr)
        m = batch.y_dp_mask
        # **类别加权必需**:档位分布 94:6(0.02:0.05),裸交叉熵把所有位预测成多数类
        # 就能拿 0.95 准确率 —— 与 L1 收敛到中位数是同一个病的两种表现(实测 s0 大动作 0 例)。
        cw = (dv_class_weight.to(dv_pred.dtype).to(dv_pred.device)
              if dv_class_weight is not None else None)
        dvl = (F.cross_entropy(dv_pred[m], batch.y_dv_cls[m], weight=cw) if bool(m.any())
               else torch.zeros((), device=ptr_logit.device))
        total = w_ptr * ptr + w_dp * dvl
        diag = {"total": float(total.detach()), "ptr": float(ptr.detach()),
                "dp": float(dvl.detach()) if bool(m.any()) else 0.0}
    else:
        total, diag = adjust_loss(ptr_logit, dv_pred, batch, w_ptr, w_dp)
    cm = batch.shunt_cand > 0.5
    if bool(cm.any()):
        pw = (torch.as_tensor(float(pos_weight), device=shunt_logit.device,
                              dtype=shunt_logit.dtype)
              if pos_weight is not None else None)
        sh = F.binary_cross_entropy_with_logits(shunt_logit[cm], batch.y_shunt[cm],
                                                pos_weight=pw)
    else:
        sh = torch.zeros((), device=ptr_logit.device)
    total = total + w_shunt * sh
    diag = dict(diag)
    diag["shunt"] = float(sh.detach())
    diag["total"] = float(total.detach())
    return total, diag


def k_class_weight(train_examples, n_k_max: int = K_MAX_UNITS,
                   device=None) -> torch.Tensor:
    """规模头 balanced 类别权重(逆频率;零支撑档权重 0)。

    **必需(2026-07-25 审查 F4-①)**:规范解机组数分布 {1:263, 2:157, 4:22, 8:233},
    k=4 仅占 3.3%、k=0/3/5~12 零支撑——裸 CE 只预测多数档就能拿高 k_acc,正是本仓
    "稀有类不加权 → 头是死的"踩过四次的场景(判稳子类 / shunt / dv 档位 / 机组指针)。
    零支撑档给 0 权重(不学也不惩罚),有支撑档按逆频率归一。
    """
    import numpy as np
    cnt = np.zeros(n_k_max + 1, dtype=np.float64)
    for x in train_examples:
        k = int(min(int(np.asarray(x["y_dp_mask"]).sum()), n_k_max))
        cnt[k] += 1
    present = cnt > 0
    w = np.zeros_like(cnt)
    w[present] = cnt[present].sum() / (present.sum() * cnt[present])
    return torch.tensor(w, dtype=torch.float32, device=device)


def reactive_rank_loss(ptr_logit: torch.Tensor, dv_pred: torch.Tensor,
                       shunt_logit: torch.Tensor, k_logits: torch.Tensor, batch,
                       w_ptr: float = 1.0, w_dp: float = 1.0, w_shunt: float = 1.0,
                       w_k: float = 1.0, pos_weight: Optional[float] = None,
                       dv_class_weight: Optional[torch.Tensor] = None,
                       k_weight: Optional[torch.Tensor] = None):
    """D② 损失 = 图内成对排序(指针)+ 规模头 CE + Δv_set 档位 CE + shunt BCE。

    - 排序正例 = **规范解机组位**(y_dp_mask∩gen;老师最小 Σ 解,与 Jaccard 诊断同口径);
      负例 = gen 内**任何可行修法都不用**的机组(y_units=0)。可行但非规范的机组位(y_units=1
      且非规范)语义含糊,**不入对**(既不当正也不当负);
    - 规模头目标 = 规范解机组数(0 = 纯并补修法),clip 到 n_k_max;
    - 无正例或无负例的图不产排序项(纯并补例只教规模头与 shunt 头);
    - 指针不再用正类加权(排序天然不受基率绑架;v5 的 auto=4.08 正是过冲源之一)。
    """
    gm = batch.gen_mask
    nb = batch.batch
    n_k_max = k_logits.shape[-1] - 1
    B = k_logits.shape[0]
    rank_terms = []
    k_tgt = torch.zeros(B, dtype=torch.long, device=ptr_logit.device)
    canon = batch.y_dp_mask & gm
    for b in range(B):
        rows = (nb == b) & gm
        pos = rows & canon
        neg = rows & (batch.y_units < 0.5)
        k_tgt[b] = min(int(pos.sum()), n_k_max)
        if bool(pos.any()) and bool(neg.any()):
            diff = ptr_logit[pos].unsqueeze(1) - ptr_logit[neg].unsqueeze(0)
            rank_terms.append(F.softplus(-diff).mean())
    ptr = (torch.stack(rank_terms).mean() if rank_terms
           else torch.zeros((), device=ptr_logit.device))
    kw = (k_weight.to(k_logits.device).float() if k_weight is not None else None)
    kl = F.cross_entropy(k_logits.float(), k_tgt, weight=kw)
    m = batch.y_dp_mask
    cw = (dv_class_weight.to(dv_pred.dtype).to(dv_pred.device)
          if dv_class_weight is not None else None)
    dvl = (F.cross_entropy(dv_pred[m], batch.y_dv_cls[m], weight=cw) if bool(m.any())
           else torch.zeros((), device=ptr_logit.device))
    cm = batch.shunt_cand > 0.5
    if bool(cm.any()):
        pw = (torch.as_tensor(float(pos_weight), device=shunt_logit.device,
                              dtype=shunt_logit.dtype)
              if pos_weight is not None else None)
        sh = F.binary_cross_entropy_with_logits(shunt_logit[cm], batch.y_shunt[cm],
                                                pos_weight=pw)
    else:
        sh = torch.zeros((), device=ptr_logit.device)
    total = w_ptr * ptr + w_k * kl + w_dp * dvl + w_shunt * sh
    return total, {"total": float(total.detach()), "ptr": float(ptr.detach()),
                   "k": float(kl.detach()), "dp": float(dvl.detach()),
                   "shunt": float(sh.detach())}
