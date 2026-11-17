"""T4.3 失稳机群提取(方案 7.3;**v1.1 末态失步口径,2026-07-13 用户裁决"按推荐优化"**)。

**v1.1 标签规则(成员资格 = 末态真失步,替代 v0 的"窗口先摆开")**:
- **仅功角失稳样本**(t* = 功角差首次触达 500°,与 D21① 同源);
- **末态聚类基线**:取末 1s 逐机角度中位(抗滑极振铃),按排序邻差 > **500°** 切簇,
  **最大簇 = 同步群**,其中位 = 基线。纯基于机间角差、与参考系无关——等权 COI 参考会被
  大幅超前机拉偏、把其余机整体压负(均摊伪影,实测 20/185 例"9 台全负"),且正侧同样
  被污染(中等幅度迟到机的相对 COI 角被压回阈内,合成用例实证),聚类基线法两侧同时
  免疫;**最大簇并列 = 全网连锁解列**(首台失步后系统崩解、末态全散,实测 80/420)——
  末态"谁失步"不适定,但切机要害是**起始失步机** → 回退 v0 窗口先摆开提名并打
  breakup=True(提名为空才 needs_review);
- **失步机** = 末态角偏离基线 > **+500°**(超前)或 < **−500°**(落后)。500° 与失稳判据
  同源,远离 90° 贴边;天然涵盖窗口外才走失的迟到机、天然剔除首摆越限后回摆的机——
  正是 v0 两大误差源(实测 cepri36 回摆混入 186/420、迟到漏纳 141/420);
- **标签(multi-hot)= 超前 ∪ 落后**,方向逐机携带(direction:advance/lag)——下游
  (D8 信号加富、稳控切机对象)需要的是"失步机集合",不只送端;
- **v0 窗口法降级为诊断保留**:[t*±0.5s] 窗 + 相对角 +90° 的"先摆开提名"与逐机窗口
  峰值(摇摆序)照算照存——稳控候选排序(7.3 措施协议"按摇摆序")仍消费它;
- 功角差曾越 500° 但末态无失步机(滑极后全部追回/临界多摆)→ needs_review 如实标记;
- 惯量权 Tj 不可得时等权(交付卡 Tj=0,与 stout.coi_frequency 同一退化口径);
- 标签 = 发电机 token 的 multi-hot((母线号, 机组序) 列表,token 映射在数据侧按目录页做)。

v0(窗口 90° 口径)在两库的实证账(needs_review 0/605,但语义偏差三条)见变更记录
2026-07-13;v1.1 阈值若再调整仍走 8.9(只看 val/冒烟)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .stout import ANGLE_SENTINEL_DEG

ANGLE_UNSTABLE_DEG = 500.0     # 与 stana.judge_stability_v1 同源(D21 ①)
REL_THRESHOLD_DEG = 90.0       # 方案 7.3 初值(待校准)
WINDOW_S = 0.5
# 封顶哨兵(定义见 stout.ANGLE_SENTINEL_DEG)。对机群标签的具体危害:封顶机的 δ 真值
# **不可得**(只知 |δ| ≥ 99998),而 v1.1 末态聚类假定角度是真值 → 多台机同时封顶时
# 它们的角差恒为 0,会被聚成一个"同步群";若该群过半即被选作基线,**基线锚到封顶机
# 上、正常机反被判失步(标签整体反转)**。历史四批(cepri36/ieee39)角峰最大 72k/44k,
# 从未触顶,故此前无此雷;118 实测 19 机中 10 机封顶,恰好过半。
SENTINEL_DEG = ANGLE_SENTINEL_DEG


@dataclass
class CoherencyResult:
    t_star: Optional[float]                       # 功角差首次 ≥500° 的时刻
    unstable_group: List[Tuple[int, int]] = field(default_factory=list)  # v1.1 标签:失步机集合
    advance_group: List[Tuple[int, int]] = field(default_factory=list)   # 超前失步(末态 >+500°)
    lag_group: List[Tuple[int, int]] = field(default_factory=list)       # 落后失步(基线 −500°)
    rest_group: List[Tuple[int, int]] = field(default_factory=list)      # 保持同步的机
    direction: Dict[str, str] = field(default_factory=dict)              # "母线-机组序" → advance/lag
    v0_window_group: List[Tuple[int, int]] = field(default_factory=list)  # 诊断:v0 先摆开提名
    swing_order: List[Tuple[int, int]] = field(default_factory=list)     # 诊断:摇摆序(窗口峰值降序)
    rel_peak_deg: Dict[str, float] = field(default_factory=dict)         # 逐机窗口内相对角峰值
    rel_tail_deg: Dict[str, float] = field(default_factory=dict)         # 逐机末 1s 相对角中位
    tail_baseline_deg: float = 0.0                                       # 同步群基线(非超前机中位)
    breakup: bool = False                                                # 全网连锁解列(回退窗口提名)
    onset_at_edge: bool = False                                          # 窗末边缘失稳(回退窗口提名)
    n_saturated: int = 0                                                 # 功角输出封顶机数(真值不可得)
    needs_review: bool = False
    note: str = ""
    n_gen: int = 0
    window: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict:
        return {
            "t_star": self.t_star,
            "window": list(self.window),
            "unstable_group": [list(p) for p in self.unstable_group],
            "advance_group": [list(p) for p in self.advance_group],
            "lag_group": [list(p) for p in self.lag_group],
            "rest_group": [list(p) for p in self.rest_group],
            "direction": self.direction,
            "v0_window_group": [list(p) for p in self.v0_window_group],
            "swing_order": [list(p) for p in self.swing_order],
            "rel_peak_deg": self.rel_peak_deg,
            "rel_tail_deg": self.rel_tail_deg,
            "tail_baseline_deg": self.tail_baseline_deg,
            "breakup": self.breakup,
            "onset_at_edge": self.onset_at_edge,
            "n_saturated": self.n_saturated,
            "needs_review": self.needs_review,
            "note": self.note,
            "n_gen": self.n_gen,
            "protocol": "7.3_v1.1_endstate_500deg_20260713",
        }


def coi_reference(angles: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """COI 参考角序列(°):逐时刻惯量加权平均;weights 缺省等权。"""
    a = np.asarray(angles, dtype=float)
    w = np.ones(a.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    return (a * w).sum(axis=1) / w.sum()


def first_angle_violation(t: np.ndarray, angles: np.ndarray,
                          threshold_deg: float = ANGLE_UNSTABLE_DEG) -> Optional[float]:
    """功角差(max−min)首次 ≥ 阈值的时刻;从未触达 → None(该样本非功角失稳)。"""
    spread = angles.max(axis=1) - angles.min(axis=1)
    hit = np.nonzero(spread >= threshold_deg)[0]
    return float(t[hit[0]]) if hit.size else None


def _fallback_membership(res: CoherencyResult, pairs, nominated, end_dev) -> None:
    """回退分支(解列/窗末)共用:机群 = 起始失步提名(双侧、逐时刻中位参考——
    v0 的正向单侧 COI 提名在负向先摆开时会整体反转标签:起始机永不入提名,同时把
    COI 拉偏让全部同步机越 +90° 集体入选,2026-07-13 审查双怀疑者独立复现);
    方向按末态偏差符号。"""
    res.unstable_group = list(nominated)
    res.advance_group = [p for p in res.unstable_group if end_dev[p] >= 0]
    res.lag_group = [p for p in res.unstable_group if end_dev[p] < 0]
    res.rest_group = [p for p in pairs if p not in res.unstable_group]
    res.direction = {f"{b}-{g}": ("advance" if end_dev[(b, g)] >= 0 else "lag")
                     for (b, g) in res.unstable_group}


def extract_coherency(t: np.ndarray, angles: np.ndarray,
                      pairs: List[Tuple[int, int]],
                      weights: np.ndarray = None,
                      threshold_deg: float = ANGLE_UNSTABLE_DEG,
                      rel_deg: float = REL_THRESHOLD_DEG,
                      window_s: float = WINDOW_S,
                      sustain_deg: float = 500.0,
                      n_gen_case: int = None) -> CoherencyResult:
    """v1.1 末态失步口径(模块 docstring;2026-07-13 用户裁决"按推荐优化")。

    非功角失稳样本返回 t_star=None;rel_deg/window_s 只用于 v0 先摆开提名与摇摆序诊断,
    不再决定标签成员资格;sustain_deg = 末态失步阈(与 threshold_deg 同源 500°)。
    """
    t = np.asarray(t, dtype=float)
    a = np.asarray(angles, dtype=float)
    if a.ndim != 2 or a.shape[1] != len(pairs):
        raise ValueError(f"角度矩阵 {a.shape} 与机组列表 {len(pairs)} 不匹配")
    res = CoherencyResult(t_star=None, n_gen=len(pairs))
    # **覆盖门**(2026-07-13 审查):SME 功角记录来自算例 GUI 模板,可能缺机且无人校验。
    # 缺机的后果全是静默的:COI 在子集上算(参考系偏移)、缺席机永不进机群(标签错到
    # 别人头上)、spread=max−min 被低估(t* 推迟或直接判"非功角失稳",而 STANA col6 的
    # 全机功角差照样越限)。基准 = 算例真实机数(LF.LP5 行数),不足即拒产标签。
    if n_gen_case and len(pairs) < n_gen_case:
        res.needs_review = True
        res.note = (f"sme_angle_coverage_partial: 功角曲线仅 {len(pairs)}/{n_gen_case} 机"
                    "——COI 参考系与机群分割均不可信,拒产标签")
        return res
    sat_mask = np.abs(a).max(axis=0) >= SENTINEL_DEG      # 全程任一时刻触顶即封顶机
    sat_pairs = {p for p, s in zip(pairs, sat_mask) if s}
    res.n_saturated = len(sat_pairs)
    t_star = first_angle_violation(t, a, threshold_deg)
    if t_star is None:
        res.note = "not_angle_unstable"
        return res
    res.t_star = t_star

    # ---- 窗口先摆开(v0 诊断,COI 口径照存)+ 起始失步提名(双侧、逐时刻中位参考) ----
    lo, hi = t_star - window_s, t_star + window_s
    res.window = (float(lo), float(hi))
    sel = (t >= lo) & (t <= hi)
    if not sel.any():                              # 理论不可达(t_star 取自 t)
        sel = t == t_star
    coi_win = coi_reference(a[sel], weights)
    peak = (a[sel] - coi_win[:, None]).max(axis=0)
    res.rel_peak_deg = {f"{b}-{g}": round(float(v), 3) for (b, g), v in zip(pairs, peak)}
    res.v0_window_group = [p for p, v in zip(pairs, peak) if v > rel_deg]
    # 起始失步提名:逐时刻**全机中位**为参考(对 <50% 失步鲁棒)、**双侧**峰值 >rel_deg
    # ——v0 的正向单侧 COI 提名在负向先摆开时整体反转(起始机永不入提名、同步机集体
    # 越阈,审查双怀疑者独立复现);摇摆序也改按该口径的偏移幅值降序(稳控排序消费)
    med_win = np.median(a[sel], axis=1)
    exc_win = np.abs(a[sel] - med_win[:, None]).max(axis=0)
    onset_nom = [p for p, v in zip(pairs, exc_win) if v > rel_deg]
    res.swing_order = [p for _, p in sorted(zip(exc_win.tolist(), pairs),
                                            key=lambda kv: -kv[0])]

    # ---- v1.1 末态失步判定(标签成员资格;聚类只负责定位基线与解列检测) ----
    tail = t >= t[-1] - 1.0                        # 末 1s 中位:抗滑极振铃与单点噪声
    coi_tail = coi_reference(a[tail], weights)
    rel_tail = np.median(a[tail] - coi_tail[:, None], axis=0)   # 报告基准(机间差不变)
    res.rel_tail_deg = {f"{b}-{g}": round(float(v), 3)
                        for (b, g), v in zip(pairs, rel_tail)}
    end_med = float(np.median(a[-1]))              # 末刻中位(末刻副判/回退方向用)
    end_dev = {p: float(a[-1, i] - end_med) for i, p in enumerate(pairs)}
    order = np.argsort(rel_tail)
    vals = rel_tail[order]
    clusters, start = [], 0
    for i in range(1, len(vals)):
        if vals[i] - vals[i - 1] > sustain_deg:    # 邻差 >500° = 不同步簇
            clusters.append(list(range(start, i)))
            start = i
    clusters.append(list(range(start, len(vals))))
    max_size = max(len(c) for c in clusters)
    biggest = [c for c in clusters if len(c) == max_size]
    # **多数门**(2026-07-13 审查):"最大簇 = 同步群"只在同步核足够大时成立。
    # 严格并列门会被"恰好两台连锁机末态相距 <500°"击穿——size-2 失步对压过其余单机簇,
    # 基线锚到失步机上、标签整体反转(怀疑者构造复现)。门 = 并列或**不足半数**
    # (恰好半数仍可作基线:如 4 机中 2 机同步核对 2 台各自散开的机,是合法参考)。
    if len(biggest) > 1 or max_size * 2 < len(pairs):
        res.breakup = True
        if onset_nom:
            _fallback_membership(res, pairs, onset_nom, end_dev)
            res.note = (f"全网连锁解列(末态 {len(clusters)} 簇,最大簇 {max_size}/"
                        f"{len(pairs)} 机不过半):机群 = 起始失步机(窗口双侧提名)")
        else:
            res.needs_review = True
            res.note = (f"全网连锁解列且窗口提名为空(末态 {len(clusters)} 簇):须人工复核")
        return res
    sync_idx = set(biggest[0])
    sync_pairs_pre = {pairs[order[i]] for i in sync_idx}
    # **封顶守卫**(2026-07-16):同步群里出现输出封顶机 → 基线不可信,拒产标签。
    # 封顶机的 δ 真值不可得(只知 ≥99998),多台同时封顶时彼此角差恒 0 会伪装成
    # "同步"(118 实测 19 机封顶 10 台、恰好过半 → 基线锚到最发散的那群机上,
    # 其余正常机反被判失步)。**封顶机不在同步群时不拦**:此时它们必然 |偏离基线|
    # 远超 500°,被正确判为失步机,真值不可得也不影响成员资格(同覆盖门的哲学:
    # 只拒"基准本身不可信"的行,不拒"结论不受基准误差影响"的行)。
    if sync_pairs_pre & sat_pairs:
        res.needs_review = True
        res.note = (f"angle_output_saturated_in_sync_core: {len(sat_pairs)}/{len(pairs)} "
                    f"机功角输出封顶(≥{SENTINEL_DEG:.0f}°),且落在末态最大簇内——"
                    "封顶机彼此角差恒 0 会伪装成同步群、基线锚到失步机上(标签反转),"
                    "拒产标签")
        return res
    baseline = float(np.median(vals[list(sync_idx)]))
    res.tail_baseline_deg = round(baseline, 3)
    # 末刻副判基线:同步簇成员的末刻中位(末段加速失步机的末 1s 中位≈末刻−r/2,
    # 单用 tail 中位会把 r≈2000°/s 的迟起滑机漏在同步簇里——审查独立复现)
    sync_pairs = {pairs[order[i]] for i in sync_idx}
    end_base = float(np.median([a[-1, i] for i, p in enumerate(pairs) if p in sync_pairs]))
    adv, lag = [], []
    # **成员资格对全部机判**(含最大簇成员):聚类只定位基线——链式衔接(0/400/800)
    # 会把偏离基线 >500° 的机并进最大簇,若跳过簇内成员即静默漏检(审查 critical,
    # cepri36 实产 12/346 行最大簇内跨度 >500°,00109 例 −938° 机曾漏在 rest)
    for i, p in enumerate(pairs):
        v_tail = float(rel_tail[i]) - baseline
        v_end = float(a[-1, i]) - end_base
        v_eff = v_tail if abs(v_tail) >= abs(v_end) else v_end   # 末刻副判:取偏离更大者
        if v_eff > sustain_deg:
            adv.append(p)
        elif v_eff < -sustain_deg:
            lag.append(p)
    res.advance_group = adv
    res.lag_group = lag
    res.unstable_group = adv + lag
    res.rest_group = [p for p in pairs if p not in res.unstable_group]
    res.direction = {f"{b}-{g}": ("advance" if (b, g) in adv else "lag")
                     for (b, g) in res.unstable_group}
    if not res.unstable_group:
        if float(t[-1]) - t_star < 2.0 and onset_nom:
            # **窗末边缘失稳**(实测 6/420,t* 均 >11s):分离在仿真窗结束时才刚展开,
            # 末态尚未拉够 500°——回退窗口起始提名(此时窗口就在轨迹末段,提名即
            # "正在拉开的机"),onset_at_edge 如实标注
            res.onset_at_edge = True
            _fallback_membership(res, pairs, onset_nom, end_dev)
            res.note = (f"窗末边缘失稳(t*={t_star:.2f}s,距窗末 "
                        f"{float(t[-1]) - t_star:.2f}s):机群 = 窗口双侧提名")
        else:
            res.needs_review = True
            res.note = (f"功角差曾越 {threshold_deg}° 但末态无失步机(滑极后全部追回/"
                        f"临界多摆);末态相对角带 [{float(rel_tail.min()):.0f}, "
                        f"{float(rel_tail.max()):.0f}]°")
    return res


def extract_from_files(ddd_path: Path, sme_path: Path,
                       weights: np.ndarray = None, **kw) -> CoherencyResult:
    """文件级入口:STOUT.DDD + ST.SME → 机群结果(多类型列块由 stout 统一处理)。"""
    from .stout import extract_angles, parse_stout_blocks
    t, blocks = parse_stout_blocks(Path(ddd_path), Path(sme_path))
    angles, pairs = extract_angles(blocks)
    return extract_coherency(t, angles, pairs, weights=weights, **kw)
