"""跨规模迁移评估(方案 v2.1 §8.4 增量阶梯,D20;2026-07-10 整合)。

度量 = 掩码重构中运行状态通道 (V, θ, Pg, Qg) 的**标幺 MAE**(同一随机种子掩码协议)。
注意是**标幺**(物理量)而非归一化空间——归一化空间的 MAE 会被各字段的 std 缩放掩盖,
跨规模比较时不同规模档的 std 不同,退化倍数会失真。故预测值与目标值都经 Normalizer 逆变换。

  分子 = 合成大网(300~1000 节点)冻结测试集   ← 留出规模档
  分母 = 同分布留出集(训练规模档、留出家族)   ← 同规模、留出家族

**D20 口径(门禁)**:增量阶梯 L0 全集均值 / L1 样本内均值 / L2 图调和插值(预注册零学习
基线)/ L4 完整模型;门禁挂技巧分 skill = MAE_L2 ÷ MAE_L4:
  a 增量存在:skill_内 ≥ 1.2(初值,待校准);b 增量存活:skill_外/skill_内 ≥ 0.7(初值)。
裸 MAE 退化倍数**降为诊断项**继续报告(v2.0 判据把"模型变笨"与"题目变难"混为一谈:
θ 靶子离散度随规模天然近 2×)。L3(随机初始化编码器)= 5.3 组2 消融,另报不进本门禁。
L2 参考实现自 scripts/increment_ladder.py 固化迁入(数字已与门禁核对逐位一致),
该脚本改为引用此处的唯一实现。

评估纪律(8.9):固定掩码种子(与 pretrain.eval_loss 同源),冻结评估集内容哈希入报告,
阈值只允许依据验证集/冒烟结果修改且必须走 docs/变更记录.md;禁止事后更换基线族让模型通过。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.linalg import spsolve
from torch_geometric.data import Batch

from ..data.normalize import Normalizer
from ..schema.fields import NODE_TABLE, Group
from ..schema.sample import GridSample, group_bit_index
from ..train.pretrain import _VAL_MASK_SEED_OFFSET, TrainConfig, _masked_data

# 8.4 度量的状态字段。pl/ql 是给定量(潮流输入),不列入迁移门禁;v_set 同理。
TRANSFER_FIELDS = ("v", "theta", "pg", "qg")
# 门禁字段 = V、θ(D20);裸退化倍数上限保留为诊断参考(不再裁决)
GATE_FIELDS = ("v", "theta")
GATE_MAX_RATIO = 2.0
# D20 增量阶梯(§8.4):零知识常数(平衡机参考 / 标幺额定)与技巧分门禁阈值。
# 门禁 a 逐字段(2026-07-11 校准,用户预批回退路线,8.9 依据全部为 val 侧):
# θ 1.2→1.0——门禁掩码把注入量与 θ 同组遮蔽,被遮节点 θ 的信息论最优≈零注入调和插值
# (=L2 基线),技巧分上限即 1.0 附近;五个杠杆(权重×7.5/步数×2/容量×2.2/数据+2 家族/
# 掩码细分课程)实测全部平台 0.94~1.01。θ≥1.0 含义 = "达到同信息零学习最优"。v 维持 1.2。
LADDER_FALLBACK = {"theta": 0.0, "v": 1.0}
GATE_SKILL_MIN = {"v": 1.2, "theta": 1.0}   # 门禁 a:skill_内 = MAE_L2(内)/MAE_L4(内)
GATE_SURVIVAL_MIN = 0.7                      # 门禁 b:skill_外/skill_内


def _field_cols() -> Dict[str, int]:
    return {f: NODE_TABLE.offset_of(f) for f in TRANSFER_FIELDS}


@torch.no_grad()
def state_field_mae(enc, dec, samples: Sequence[GridSample], cfg: TrainConfig,
                    norm: Normalizer, dev: Optional[torch.device] = None) -> Dict[str, float]:
    """逐字段标幺 MAE(只在被遮蔽且非 missing 的位置上统计)。

    掩码协议与 pretrain.eval_loss 同源(固定种子 + 关增广)→ 内外集用同一批掩码规则,
    退化倍数才是"同一把尺子"量出来的。
    """
    if not samples:
        return {f: float("nan") for f in TRANSFER_FIELDS}
    dev = dev or next(enc.parameters()).device
    dec_dev = next(dec.parameters()).device
    if torch.device(dec_dev).type != torch.device(dev).type:
        raise RuntimeError(
            f"编码器/解码器不在同一设备(enc→{dev}, dec→{dec_dev})。"
            f"探针段把 enc 搬到 CPU 时必须一并搬 dec。")
    use_amp = cfg.amp_dtype in ("float16", "bfloat16") and dev.type == "cuda"
    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16
    total_cols = NODE_TABLE.total_dim
    cols = _field_cols()

    was_training = enc.training
    enc.eval(); dec.eval()
    rng_v = np.random.default_rng(cfg.seed + _VAL_MASK_SEED_OFFSET)
    bs = max(1, min(cfg.batch_graphs, len(samples)))
    abs_err = {f: 0.0 for f in TRANSFER_FIELDS}
    counts = {f: 0 for f in TRANSFER_FIELDS}

    for i in range(0, len(samples), bs):
        chunk = samples[i:i + bs]
        datas = [_masked_data(s, cfg, rng_v, augment=False) for s in chunk]
        batch = Batch.from_data_list(datas).to(dev)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
            h, _g, _e = enc(batch)
            pred = dec(h, total_cols)
        pred = pred.float().cpu().numpy()
        target = batch.target.float().cpu().numpy()
        lm = batch.loss_mask.cpu().numpy()
        # 逆归一化回标幺(逐行独立,列语义一致 → 整块变换)
        pred_pu = norm.inverse_transform_node(pred)
        target_pu = norm.inverse_transform_node(target)
        for f, c in cols.items():
            m = lm[:, c]
            if not m.any():
                continue
            abs_err[f] += float(np.abs(pred_pu[m, c] - target_pu[m, c]).sum())
            counts[f] += int(m.sum())

    if was_training:
        enc.train(); dec.train()
    return {f: (abs_err[f] / counts[f]) if counts[f] else float("nan") for f in TRANSFER_FIELDS}


def ladder_baselines(samples: Sequence[GridSample], seed: int = 0) -> Dict[str, Dict[str, float]]:
    """增量阶梯零学习基线(D20 / §8.4)。返回 {field: {L0, L1, L2}} 的标幺 MAE。

    输入 = **原始(未归一化)样本**,状态值即标幺;掩码流与 state_field_mae 逐位同源
    (同 seed+_VAL_MASK_SEED_OFFSET、同样本顺序、每样本一次 apply_node_mask)。
    L2 = 图调和插值:被遮蔽条目由可见条目沿图 Laplacian 插值(解 L_II x_I = −L_IB x_B),
    全遮蔽模式回退零知识常数(LADDER_FALLBACK)。预注册基线,禁止事后更换(8.9)。
    实现自 scripts/increment_ladder.py 逐位迁入(2026-07-10,数字已与门禁核对一致)。
    """
    from ..train.masking import MaskConfig, apply_node_mask

    sbit = group_bit_index(Group.STATE)
    rng = np.random.default_rng(seed + _VAL_MASK_SEED_OFFSET)
    errs = {f: {"L0": [], "L1": [], "L2": []} for f in LADDER_FALLBACK}
    pooled = {f: [] for f in LADDER_FALLBACK}

    per_sample = []
    for s in samples:
        _mf, _tgt, lm = apply_node_mask(s.node_feat, s.node_missing, MaskConfig(), rng)
        per_sample.append((s, lm))
        ok = ~s.node_missing[:, sbit]
        for f in LADDER_FALLBACK:
            c = NODE_TABLE.offset_of(f)
            pooled[f].append(s.node_feat[ok, c].astype(np.float64))

    pool_mean = {f: np.concatenate(v).mean() if v else 0.0 for f, v in pooled.items()}

    for s, lm in per_sample:
        n = s.n_nodes
        i, j = s.edge_index
        A = sp.coo_matrix((np.ones(i.size * 2), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
        A.data[:] = 1.0
        L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
        ok = ~s.node_missing[:, sbit]
        for f, fb in LADDER_FALLBACK.items():
            c = NODE_TABLE.offset_of(f)
            m = lm[:, c] & ok
            if not m.any():
                continue
            truth = s.node_feat[m, c].astype(np.float64)
            vis = ok & ~m
            vis_vals = s.node_feat[vis, c].astype(np.float64)
            errs[f]["L0"].append(np.abs(truth - pool_mean[f]))
            errs[f]["L1"].append(np.abs(truth - (vis_vals.mean() if vis_vals.size else fb)))
            x = np.full(n, fb, dtype=np.float64)
            x[vis] = s.node_feat[vis, c]
            if vis.any():
                I = np.where(m)[0]; B = np.where(vis)[0]
                LII = L[I][:, I].tocsc() + 1e-8 * sp.identity(I.size, format="csc")
                x[I] = spsolve(LII, -L[I][:, B] @ x[B])
            errs[f]["L2"].append(np.abs(x[m] - truth))
    return {f: {k: float(np.concatenate(v).mean()) if v else float("nan")
                for k, v in d.items()} for f, d in errs.items()}


def content_hash(paths: Sequence[Path]) -> str:
    """冻结评估集内容哈希(8.9 ①)。路径排序后累积字节。"""
    h = hashlib.blake2b(digest_size=16)
    for p in sorted(paths, key=lambda q: str(q).replace("\\", "/")):
        h.update(Path(p).name.encode("utf-8"))
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def transfer_report(enc, dec, in_samples: Sequence[GridSample], out_samples: Sequence[GridSample],
                    cfg: TrainConfig, norm: Normalizer,
                    dev: Optional[torch.device] = None,
                    gate_max_ratio: float = GATE_MAX_RATIO,
                    in_raw: Optional[Sequence[GridSample]] = None,
                    out_raw: Optional[Sequence[GridSample]] = None,
                    gate_skill_min=None,
                    gate_survival_min: float = GATE_SURVIVAL_MIN) -> dict:
    """8.4 报告。in = 同分布留出集(分母);out = 留出规模档冻结测试集(分子)。

    D20 口径:传入 in_raw/out_raw(**未归一化**样本,与 in/out 同序)时计算增量阶梯与
    技巧分门禁(a skill_内 ≥ gate_skill_min;b 存活 ≥ gate_survival_min),passed 由技巧分
    裁决,裸退化倍数降为诊断项。不传 raw 时退回旧裸比值门禁(仅遗留测试/诊断用)。
    某字段掩码位置数为 0 时该字段指标 = nan 且**不计入门禁通过**(如实报告,不静默跳过)。
    """
    mae_in = state_field_mae(enc, dec, in_samples, cfg, norm, dev)
    mae_out = state_field_mae(enc, dec, out_samples, cfg, norm, dev)

    ratio: Dict[str, float] = {}
    for f in TRANSFER_FIELDS:
        a, b = mae_out[f], mae_in[f]
        ratio[f] = float(a / b) if (np.isfinite(a) and np.isfinite(b) and b > 0) else float("nan")

    finite = [ratio[f] for f in TRANSFER_FIELDS if np.isfinite(ratio[f])]
    rep = {
        "mae_in_pu": mae_in,
        "mae_out_pu": mae_out,
        "ratio": ratio,                     # 裸退化倍数(D20 起为诊断项)
        "gate_fields": list(GATE_FIELDS),
        "gate_max_ratio": gate_max_ratio,
        "mean_ratio": float(np.mean(finite)) if finite else float("nan"),
        "n_in": len(in_samples),
        "n_out": len(out_samples),
    }

    if in_raw is None or out_raw is None:
        # 遗留口径:裸比值裁决(D20 前;保留给不带 raw 集的调用方)
        gate = {f: bool(np.isfinite(ratio[f]) and ratio[f] <= gate_max_ratio) for f in GATE_FIELDS}
        rep.update({"verdict_mode": "legacy_ratio", "gate": gate, "passed": all(gate.values())})
        return rep

    # ---- D20 增量阶梯(掩码流与 state_field_mae 逐位同源:同 seed、同样本顺序)----
    # 门禁 a 阈值逐字段(dict);兼容标量(测试/诊断用,施加到全部字段)
    if gate_skill_min is None:
        gate_skill_min = GATE_SKILL_MIN
    if not isinstance(gate_skill_min, dict):
        gate_skill_min = {f: float(gate_skill_min) for f in LADDER_FALLBACK}
    # 防线:raw 集与归一化集必须同长同序(掩码逐位同源依赖位置对应;乱序/子集会静默
    # 得到错误技巧分,2026-07-10 评审)
    for name, aa, bb in (("in", in_samples, in_raw), ("out", out_samples, out_raw)):
        if len(aa) != len(bb):
            raise ValueError(f"{name}_raw 与 {name}_samples 长度不一致:{len(bb)} vs {len(aa)}")
        for k, (x, y) in enumerate(zip(aa, bb)):
            if x.sample_id != y.sample_id:
                raise ValueError(f"{name}_raw[{k}] 样本乱序:{y.sample_id} vs {x.sample_id}"
                                 "(掩码逐位同源要求同序)")
    ladder_in = ladder_baselines(in_raw, cfg.seed)
    ladder_out = ladder_baselines(out_raw, cfg.seed)
    skill_in, skill_out, survival = {}, {}, {}
    gate_a, gate_b = {}, {}
    for f in LADDER_FALLBACK:
        l2i, l2o = ladder_in[f]["L2"], ladder_out[f]["L2"]
        m_i, m_o = mae_in[f], mae_out[f]
        skill_in[f] = float(l2i / m_i) if (np.isfinite(l2i) and np.isfinite(m_i) and m_i > 0) else float("nan")
        skill_out[f] = float(l2o / m_o) if (np.isfinite(l2o) and np.isfinite(m_o) and m_o > 0) else float("nan")
        survival[f] = (float(skill_out[f] / skill_in[f])
                       if (np.isfinite(skill_out[f]) and np.isfinite(skill_in[f]) and skill_in[f] > 0)
                       else float("nan"))
        gate_a[f] = bool(np.isfinite(skill_in[f]) and skill_in[f] >= gate_skill_min[f])
        gate_b[f] = bool(np.isfinite(survival[f]) and survival[f] >= gate_survival_min)
    rep.update({
        "verdict_mode": "d20_skill",
        "ladder_in": ladder_in, "ladder_out": ladder_out,
        "skill_in": skill_in, "skill_out": skill_out, "survival": survival,
        "gate_skill_min": gate_skill_min, "gate_survival_min": gate_survival_min,
        "gate_a": gate_a, "gate_b": gate_b,
        "gate": {f: gate_a[f] and gate_b[f] for f in LADDER_FALLBACK},
        "passed": all(gate_a.values()) and all(gate_b.values()),
    })
    return rep


def format_report(rep: dict) -> str:
    lines = [
        f"[8.4 跨规模迁移] 分母(同分布留出) n={rep['n_in']}  分子(留出规模档) n={rep['n_out']}",
        f"{'字段':<8}{'内 MAE(pu)':>14}{'外 MAE(pu)':>14}{'退化倍数':>10}",
    ]
    for f in TRANSFER_FIELDS:
        lines.append(f"{f:<8}{rep['mae_in_pu'][f]:>14.5f}{rep['mae_out_pu'][f]:>14.5f}"
                     f"{rep['ratio'][f]:>10.2f}")
    lines.append(f"(裸退化倍数为诊断项,D20;汇总均值 {rep['mean_ratio']:.2f}×)")

    if rep.get("verdict_mode") != "d20_skill":
        verdict = "通过" if rep["passed"] else "未过"
        lines.append(f"门禁[遗留裸比值]({'/'.join(rep['gate_fields'])} 各 ≤{rep['gate_max_ratio']}×):{verdict}")
        return "\n".join(lines)

    li, lo = rep["ladder_in"], rep["ladder_out"]
    lines.append("")
    lines.append(f"[增量阶梯 D20]{'':<4}{'θ内':>9}{'θ外':>9}{'v内':>10}{'v外':>10}")
    for lvl in ("L0", "L1", "L2"):
        lines.append(f"{lvl:<16}{li['theta'][lvl]:>9.4f}{lo['theta'][lvl]:>9.4f}"
                     f"{li['v'][lvl]:>10.5f}{lo['v'][lvl]:>10.5f}")
    lines.append(f"{'L4 模型':<15}{rep['mae_in_pu']['theta']:>9.4f}{rep['mae_out_pu']['theta']:>9.4f}"
                 f"{rep['mae_in_pu']['v']:>10.5f}{rep['mae_out_pu']['v']:>10.5f}")
    for f in LADDER_FALLBACK:
        gmin = rep["gate_skill_min"][f] if isinstance(rep["gate_skill_min"], dict) else rep["gate_skill_min"]
        ga = "✅" if rep["gate_a"][f] else f"❌ <{gmin}"
        gb = "✅" if rep["gate_b"][f] else f"❌ <{rep['gate_survival_min']}"
        lines.append(f"{f:>6}: skill_内 {rep['skill_in'][f]:.2f} ({ga})  "
                     f"skill_外 {rep['skill_out'][f]:.2f}  存活 {rep['survival'][f]:.2f} ({gb})")
    lines.append(f"门禁③(D20 技巧分,V/θ 各须 a、b 双过):{'通过' if rep['passed'] else '未过'}")
    return "\n".join(lines)
