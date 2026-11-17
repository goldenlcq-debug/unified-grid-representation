"""STANA.DAT 极值摘要解析 + 判稳(T4.2;v1 判据 = 2026-07-12 用户裁决)。

STANA.DAT 无官方格式条目(数据文件格式 V3.8 与 51 格式 Word 版均无,检索记录见
docs/任务笔记-M4-手册出处.md),仅配 ST.SME 的工程产出(118 fixture 无 ST.SME 即不产)。
**列布局随工程 L5 版式二分**(2026-07-16 118 实算破译;与 SME 寻址语义同源):
- 带机组序版(cepri36/39,LF.L5 21 格,实测行宽 28):极值定位对 = (母线,机组序),
  且带 Vbase 列 → col6=最大功角差(°)、col8=vmin、col12=fmin(pu/50Hz)、
  col14=vmax、col18=fmax。列语义由 CEPRI-36 实算反推(2026-07-11/12,稳 vs 失稳
  全列域对照验证,变更记录)。
- 基准版(118,LF.L5 20 格无机组序列,实测行宽恒 18):定位对只有母线号、无 Vbase
  列 → **col4=最大功角差、col6=vmin、col8=fmin、col10=vmax、col12=fmax**。
  锚:t=0 行 vmin/vmax = 潮流解 0.8991/1.0500 逐位;故障中 vmin=6e-05(金属短路);
  失稳例 col4 末行 99999(引擎功角哨兵封顶)。
  历史排雷:四批留档 + 种子 + 夹具全量抽样行宽恒 28,无 18 格旧行,按行宽分支安全
  (2026-07-16 扫描)。旧解析在 118 上曾把 vmin 读成"角差 0.8991°"、fmin 读成
  "vmin 0.876"——118 判稳标签的真正拦路虎(交接坑 3 的"列数差 1"是误诊)。

**v1 判据(2026-07-12 用户裁决,替代 v0 手册缺省;阶段 2 精确口径 2026-07-13)**:
①功角:任意两机相对功角差绝对值 >500°,任意时刻触达即失稳;
②电压:故障发生 10s 以后机端电压不得 <0.8,且最终稳定电压不得 <0.9
  ——机端电压主口径 = ST.SME Coord_Type=2 机组端母线电压曲线(阶段 2 精确口径,
  经 judge_files_v1/vt_t+vterm 注入);曲线缺失时回退全网最低母线电压(col8)
  **保守代理**(网最低 ≤ 机端最低,偏严),v_source 如实标注;"10s 以后"分句
  仅当仿真窗越过 ts+10 时可评(evaluable 标志如实携带);过电压三级同理:主口径
  = 逐母线持续越限(Type=2 全母线曲线),回退 vmax 包络("存在母线"松弛);
③频率:系统频率 >51 Hz 或 <49 Hz,COI 持续 ≥0.5s 主口径;无轨迹退化机组频率
  极值瞬时口径(col12/col18,49/50=0.98,51/50=1.02),freq_source 如实标注。
v0 判据(手册 3.3.8:功角 500° + 低压 0.7pu 持续 1s)保留作对照;其低压 hold 在
摇摆谷间复位的盲区正是 v1 替代它的动因(变更记录 07-12)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# 判据版本(随**判据行为或落盘 schema 变更**升版;收口门按此拒纳未重判的旧批)。
# v1_20260715:频率触发方向拆分(freq_hold_high_s/low_s 新字段;跨向不再累加)。
# 判稳二值语义与 v1_20260712 逐位等价(三库离线重判零翻案),但 schema 变了 →
# 旧批缺字段会让下游静默退化到极值代理,故必须升版并强制重判(对抗审查 07-15)。
CRITERIA_VERSION = "v1_20260715"

ANGLE_UNSTABLE_DEG = 500.0     # 手册 3.3.8 缺省 = 用户 v1 判据同值
VLOW_PU = 0.7
VLOW_HOLD_S = 1.0
# v1(2026-07-12 用户裁决;过电压三级于同日追加裁决纳入,移植自 openclaw 默认参数)
V1_V_RECOVER = 0.8             # 故障后 10s 起,机端电压下限
V1_V_STEADY = 0.9              # 最终稳定电压下限
V1_F_LO_PU = 49.0 / 50.0       # 系统频率下限(pu/50Hz)
V1_F_HI_PU = 51.0 / 50.0
# 过电压三级(阈值 pu, 持续 s):任一级越限持续达标即失稳。注:此默认参数下第一级
# (最浅阈值+最短持续)在逻辑上支配后两级——分级结构保留,供改成"越深越快"的
# 反时限参数时使用(变更记录 07-12 收轮条)。绝对值模式(openclaw 相对值模式不采)。
V1_OVERVOLT_TIERS = ((1.2, 0.02), (1.3, 0.1), (1.4, 0.5))
_COL_T, _COL_ANGLE, _COL_VMIN = 0, 6, 8
_COL_FMIN, _COL_VMAX, _COL_FMAX = 12, 14, 18
# 基准版(118 类,行宽恒 18)的列映射(语义顺序与带机组序版一致:角差/vmin/fmin/vmax/fmax)
_BUSONLY_NCELLS = 18
_BUSONLY_ANGLE, _BUSONLY_VMIN = 4, 6
_BUSONLY_FMIN, _BUSONLY_VMAX, _BUSONLY_FMAX = 8, 10, 12


@dataclass
class StanaSeries:
    t: List[float] = field(default_factory=list)
    angle_max: List[float] = field(default_factory=list)
    vmin: List[float] = field(default_factory=list)
    fmin: List[float] = field(default_factory=list)   # 最低机组频率(pu);旧短行文件为空
    fmax: List[float] = field(default_factory=list)
    vmax: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.t)

    @property
    def has_freq(self) -> bool:
        return len(self.fmin) == len(self.t) and len(self.t) > 0


def parse_stana(path: Path) -> StanaSeries:
    """版式按首个有效行的行宽定死(整文件一致;混宽行按所定版式跳过):
    行宽 == 18 → 基准版(118 类);其余(实测 28)→ 带机组序版(cepri36/39 类)。"""
    out = StanaSeries()
    busonly = None
    for line in Path(path).read_text(encoding="gb18030", errors="replace").splitlines():
        cells = [c.strip() for c in line.split(",")]
        n = len(cells)
        if n <= _COL_VMIN:
            continue
        if busonly is None:
            busonly = (n == _BUSONLY_NCELLS)
        if busonly:
            if n != _BUSONLY_NCELLS:
                continue
            ci_a, ci_v = _BUSONLY_ANGLE, _BUSONLY_VMIN
            ci_fl, ci_vx, ci_fh = _BUSONLY_FMIN, _BUSONLY_VMAX, _BUSONLY_FMAX
            has_ext = True
        else:
            ci_a, ci_v = _COL_ANGLE, _COL_VMIN
            ci_fl, ci_vx, ci_fh = _COL_FMIN, _COL_VMAX, _COL_FMAX
            has_ext = n > _COL_FMAX
        try:
            t = float(cells[_COL_T])
            a = float(cells[ci_a])
            v = float(cells[ci_v])
        except ValueError:
            continue
        out.t.append(t)
        out.angle_max.append(a)
        out.vmin.append(v)
        if has_ext:
            try:
                fl = float(cells[ci_fl])
                vx = float(cells[ci_vx])
                fh = float(cells[ci_fh])
            except ValueError:
                continue
            out.fmin.append(fl)
            out.vmax.append(vx)
            out.fmax.append(fh)
    return out


def judge_stability(series: StanaSeries,
                    angle_deg: float = ANGLE_UNSTABLE_DEG,
                    vlow_pu: float = VLOW_PU,
                    vlow_hold_s: float = VLOW_HOLD_S,
                    vlow_from_s: float = 0.0) -> Dict:
    """v0 双判据(功角 + 低压持续);返回 stable 与逐判据明细。空序列判 infeasible。

    vlow_from_s(2026-07-11 评审修订):低压持续判据只统计 t ≥ 该时刻的行——金属性
    短路期间 vmin 必然 <0.7,把故障窗计入等效把"切除后低压 ≥1s"收紧为
    "≥(1s − 故障时长)",判稳与 Te 耦合且已翻转过一条种子标签。调用方传故障切除
    时刻(_verdict 从 ST.S11 取);功角判据保持全程(故障中失步是真物理)。
    """
    if not series.t:
        return {"stable": False, "reason": "empty_stana", "n_points": 0}
    peak = max(series.angle_max)
    angle_ok = peak < angle_deg
    hold, worst_hold = 0.0, 0.0
    for i in range(1, len(series.t)):
        if series.t[i] < vlow_from_s:
            continue
        dt = series.t[i] - series.t[i - 1]
        if series.vmin[i] < vlow_pu:
            hold += max(dt, 0.0)
            worst_hold = max(worst_hold, hold)
        else:
            hold = 0.0
    volt_ok = worst_hold < vlow_hold_s
    return {
        "stable": bool(angle_ok and volt_ok),
        "angle_ok": bool(angle_ok), "angle_peak_deg": float(peak),
        "volt_ok": bool(volt_ok), "vlow_worst_hold_s": float(worst_hold),
        "vmin_pu": float(min(series.vmin)),
        "n_points": len(series.t), "t_end": float(series.t[-1]),
        "criteria": {"angle_deg": angle_deg, "vlow_pu": vlow_pu,
                     "vlow_hold_s": vlow_hold_s, "vlow_from_s": vlow_from_s},
    }


def _tier_hold(t, v, th_v: float) -> float:
    """序列 v 高于阈值 th_v 的最长持续时长(s);投切重复时刻 dt=0 天然无贡献。"""
    run = worst = 0.0
    for i in range(1, len(t)):
        if v[i] > th_v:
            run += max(t[i] - t[i - 1], 0.0)
            worst = max(worst, run)
        else:
            run = 0.0
    return worst


def judge_stability_v1(series: StanaSeries, fault_ts: float = 0.0,
                       angle_deg: float = ANGLE_UNSTABLE_DEG,
                       v_recover: float = V1_V_RECOVER,
                       v_steady: float = V1_V_STEADY,
                       f_lo_pu: float = V1_F_LO_PU,
                       f_hi_pu: float = V1_F_HI_PU,
                       fsys_t=None, fsys_hz=None,
                       freq_hold_s: float = 0.5,
                       vt_t=None, vterm=None, vterm_n_gen: int = None,
                       vbus_t=None, vbus=None, bus_nos=None) -> Dict:
    """v1 判据(2026-07-12 用户裁决;阶段 2 精确口径 2026-07-13)。
    返回 stable + 逐判据明细 + 失稳类型列表。

    - angle:任意时刻 ≥angle_deg 即失稳;
    - voltage ①recover(D21 显式"机端电压"):t ≥ fault_ts+10 任一机端 < v_recover
      失稳。机端轨迹(vt_t[N] + vterm[N,G])**必须覆盖全部机组**(vterm_n_gen =
      功角列机数)才够格作精确口径——部分覆盖的逐步最小值 ≥ 全机集,单调朝"稳"
      放松,缺席机跌破即漏判(ieee39 GUI 模板实测只配 9/10 机);覆盖不全或无轨迹
      一律退全网 vmin 保守代理(网最低 ≤ 机端最低,偏严),v_recover_source 如实标注。
      窗不够长则 recover_evaluable=False 且本分句不参与判定;
    - voltage ②steady(D21 第二分句字面 = "末 1s **最低电压**",未限定机端):一律以
      全网最低母线电压判(STANA col8;Type=2 全母线曲线在位时两者实证等价 5e-6),
      不收窄到机端(收窄 = 放松,超出裁决文本);机端末 1s 另报 vterm_last1s 供审计;
    - voltage ③过电压三级:给了逐母线电压轨迹(vbus_t[N] + vbus[N,B])则逐母线判
      持续越限(D21 精确逐母线口径,"同一母线持续"语义);未给则全网 vmax 包络代理
      ("存在母线"松弛,偏严);
    - frequency:COI 系统频率(fsys)越 [f_lo_pu, f_hi_pu] 持续 ≥freq_hold_s;无 COI
      输入退化单机极值瞬时口径(freq_source 如实标注)。
    空序列判 infeasible 语义(stable=False, reason=empty_stana)。
    """
    if not series.t:
        return {"stable": False, "reason": "empty_stana", "n_points": 0,
                "criteria_version": CRITERIA_VERSION}
    t_end = series.t[-1]
    peak = max(series.angle_max)
    angle_ok = peak < angle_deg

    recover_deadline = fault_ts + 10.0
    n_vterm = len(vterm[0]) if (vterm is not None and len(vterm) > 0) else 0
    vterm_ok_axis = (vt_t is not None and vterm is not None
                     and len(vt_t) > 0 and len(vt_t) == len(vterm))
    vterm_full = bool(vterm_ok_axis and vterm_n_gen is not None
                      and n_vterm >= vterm_n_gen > 0)
    if vterm_full:
        v_recover_source = "sme_terminal_exact"
        v_t_axis = list(vt_t)
        v_series = [min(row) for row in vterm]         # 逐步机端最低(全机集)
    else:
        v_recover_source = "network_vmin_proxy(机端保守代理)"
        if vterm_ok_axis:                              # 有机端曲线但不全:如实说明为何不用
            v_recover_source = (f"network_vmin_proxy(机端曲线仅 {n_vterm}/"
                                f"{vterm_n_gen} 机,不足以作精确口径)")
        v_t_axis, v_series = series.t, series.vmin
    recover_evaluable = v_t_axis[-1] > recover_deadline
    recover_ok = True
    recover_vmin = None
    if recover_evaluable:
        post = [v for tt, v in zip(v_t_axis, v_series) if tt >= recover_deadline]
        recover_vmin = min(post) if post else None
        recover_ok = (recover_vmin is None) or (recover_vmin >= v_recover)
    # steady 分句:D21 字面"末 1s 最低电压",全网口径(不收窄机端)
    last1 = [v for tt, v in zip(series.t, series.vmin) if tt >= t_end - 1.0]
    steady_ok = (min(last1) >= v_steady) if last1 else True
    vterm_last1s = None
    if vterm_ok_axis:
        tail = [min(row) for tt, row in zip(vt_t, vterm) if tt >= vt_t[-1] - 1.0]
        vterm_last1s = float(min(tail)) if tail else None

    # 过电压三级:精确口径 = 逐母线持续越限(阶段 2);代理 = vmax 包络("存在母线"
    # 松弛,偏严)。openclaw 逐曲线判的移植,绝对值模式。
    has_vbus = (vbus_t is not None and vbus is not None
                and len(vbus_t) > 0 and len(vbus_t) == len(vbus))
    overvolt_tier = None
    overvolt_ok = True
    if has_vbus:
        overvolt_evaluable = True
        ov_source = "sme_per_bus_exact"
        ncols = len(vbus[0])
        for th_v, dur in V1_OVERVOLT_TIERS:
            worst_ov, worst_bus = 0.0, None
            for b in range(ncols):
                h = _tier_hold(vbus_t, [row[b] for row in vbus], th_v)
                if h > worst_ov:
                    worst_ov, worst_bus = h, (bus_nos[b] if bus_nos else b)
            if worst_ov >= dur:
                overvolt_ok = False
                overvolt_tier = {"threshold_pu": th_v, "duration_s": dur,
                                 "worst_hold_s": round(worst_ov, 4),
                                 "bus_no": worst_bus}
                break
    else:
        overvolt_evaluable = len(series.vmax) == len(series.t) and len(series.t) > 0
        ov_source = "network_vmax_envelope_proxy(存在母线松弛,偏严)"
        if overvolt_evaluable:
            for th_v, dur in V1_OVERVOLT_TIERS:
                worst_ov = _tier_hold(series.t, series.vmax, th_v)
                if worst_ov >= dur:
                    overvolt_ok = False
                    overvolt_tier = {"threshold_pu": th_v, "duration_s": dur,
                                     "worst_hold_s": round(worst_ov, 4)}
                    break
    volt_ok = recover_ok and steady_ok and overvolt_ok

    # 频率:主口径 = COI 系统频率(fsys_hz,由 STOUT 逐机功角求导加权,见 stout.coi_frequency)
    # 越 [49,51] 持续 ≥freq_hold_s 判失稳——单机瞬时极值在小网重故障首摆期普遍越线后回摆
    # (实测 318 例仅此翻案),不构成"系统频率"越限;敏感性四口径对照见变更记录 07-12。
    # 无 COI 输入时退化为单机极值瞬时口径(STANA col12/18)并标注 freq_source。
    # **方向拆分(2026-07-15)**:持续段按高/低分别累计——细分类型头(5.2-b:f_high/f_low)
    # 需要的是"**触发失稳的那段持续越限**的方向",而非全窗极值方向:首摆瞬时冲高(不持续)
    # 叠加随后的持续跌低,极值代理会把两个方向都标上。freq_ok 语义不变(任一方向持续
    # ≥freq_hold_s 即失稳),分方向 max 与原单计数器在物理轨迹上等价(采样步 0.02s,
    # 频率不可能不经过带内直接从 >f_hi 跳到 <f_lo),但更严格:病态跨向连续越界不再累加。
    f_min = f_max = None
    hold_hi = hold_lo = None
    if fsys_hz is not None and fsys_t is not None and len(fsys_hz) > 2:
        freq_evaluable = True
        freq_source = "coi_sustained"
        f_lo_hz, f_hi_hz = f_lo_pu * 50.0, f_hi_pu * 50.0
        run_hi = run_lo = worst_hi = worst_lo = 0.0
        for i in range(1, len(fsys_t)):
            dt = max(fsys_t[i] - fsys_t[i - 1], 0.0)
            if fsys_hz[i] > f_hi_hz:
                run_hi += dt
                worst_hi = max(worst_hi, run_hi)
                run_lo = 0.0
            elif fsys_hz[i] < f_lo_hz:
                run_lo += dt
                worst_lo = max(worst_lo, run_lo)
                run_hi = 0.0
            else:
                run_hi = run_lo = 0.0
        hold_hi, hold_lo = round(worst_hi, 4), round(worst_lo, 4)
        freq_ok = max(worst_hi, worst_lo) < freq_hold_s
        f_min, f_max = float(min(fsys_hz)) / 50.0, float(max(fsys_hz)) / 50.0
    else:
        freq_evaluable = series.has_freq
        freq_source = "machine_extreme_fallback"
        freq_ok = True
        if freq_evaluable:
            f_min, f_max = min(series.fmin), max(series.fmax)
            freq_ok = (f_min >= f_lo_pu) and (f_max <= f_hi_pu)

    types = [name for name, ok in
             (("angle", angle_ok), ("voltage", volt_ok), ("frequency", freq_ok)) if not ok]
    return {
        "stable": not types,
        "instability_types": types,                     # 失稳类型标签源(5.2 信号加富)
        "angle_ok": bool(angle_ok), "angle_peak_deg": float(peak),
        "volt_ok": bool(volt_ok), "recover_ok": bool(recover_ok),
        "recover_evaluable": bool(recover_evaluable),
        "recover_vmin": (float(recover_vmin) if recover_vmin is not None else None),
        "steady_ok": bool(steady_ok),
        "overvolt_ok": bool(overvolt_ok), "overvolt_evaluable": bool(overvolt_evaluable),
        "overvolt_tier": overvolt_tier,
        "voltage_modes": [m for m, bad in (("low_recover", not recover_ok),
                                           ("low_steady", not steady_ok),
                                           ("overvoltage", not overvolt_ok)) if bad],
        "vmax_pu": float(max(series.vmax)) if series.vmax else None,
        "last1s_vmin": float(min(last1)) if last1 else None,
        "vmin_pu": float(min(series.vmin)),
        "vterm_min_pu": (float(min(min(r) for r in vterm)) if vterm_ok_axis else None),
        "vterm_last1s": vterm_last1s,
        "vterm_full": bool(vterm_full),
        "n_vterm_curves": n_vterm, "n_gen": vterm_n_gen,
        "n_vbus_curves": (len(vbus[0]) if has_vbus else 0),
        "freq_ok": bool(freq_ok), "freq_evaluable": bool(freq_evaluable),
        "freq_source": freq_source,
        "fmin_pu": f_min, "fmax_pu": f_max,
        # 触发方向的严格口径(持续段时长;COI 口径才有,退化口径为 None)——
        # 细分类型 f_high/f_low 的标签源(5.2-b);极值 fmin/fmax 保留作诊断
        "freq_hold_high_s": hold_hi, "freq_hold_low_s": hold_lo,
        "n_points": len(series.t), "t_end": float(t_end),
        "criteria_version": CRITERIA_VERSION,
        "criteria": {"angle_deg": angle_deg, "v_recover": v_recover,
                     "v_steady": v_steady, "f_lo_pu": f_lo_pu, "f_hi_pu": f_hi_pu,
                     "fault_ts": fault_ts,
                     "freq_hold_s": freq_hold_s,
                     "overvolt_tiers": list(V1_OVERVOLT_TIERS),
                     "v_recover_source": v_recover_source,
                     "v_steady_source": "network_vmin(D21 字面:末 1s 最低电压)",
                     "overvolt_source": ov_source},
    }


def judge_files_v1(stana_path: Path, ddd_path: Path = None, sme_path: Path = None,
                   fault_ts: float = 0.0, n_gen_case: int = None) -> Dict:
    """文件级 v1 判稳统一入口(psasp._verdict 与 ts_relabel 共用,免两处胶水漂移)。

    STANA 必需;STOUT/SME 可选——在位则叠加 COI 系统频率(D21 ④主口径)与机端/
    逐母线精确电压口径(阶段 2),解析失败/与 STANA 不同步按既有纪律退化为代理
    口径并在 freq_source/v_recover_source/traj_note 中如实标注(不静默失败)。

    **SME 必须与该 .ddd 同批**:列布局随 Type=2 注入而变(43→82 列),跨批混用会
    解出错列或尺寸不自洽——ts_labels 逐 tag 留档 SME 即为此(2026-07-13 审查)。
    行数与 STANA 不一致 = 轨迹被截断(引擎后台续写未收全),整份轨迹弃用退代理:
    截断轨迹的"末 1s / 故障后 10s"落在错误时刻,比没有更危险。

    **n_gen_case(算例真实机组数,LF.LP5 行数)**:机端覆盖门的基准。不给则退而用 SME
    功角列机数作基准——但**功角列本身可能缺机**(GUI 模板自带,无人校验),那样基准
    就被击穿(9/9 会被当成"全覆盖")。调用方能拿到算例时务必传(2026-07-13 审查)。
    """
    series = parse_stana(Path(stana_path))
    fsys_t = fsys_hz = vt_t = vterm = vbus_t = vbus = bus_nos = None
    n_gen = None
    traj_note = None
    if ddd_path and sme_path and Path(ddd_path).exists() and Path(sme_path).exists():
        try:
            import numpy as np

            from .stout import (coi_frequency, extract_angles, extract_bus_voltage,
                                extract_terminal_voltage, parse_stout_blocks)
            tt, blocks = parse_stout_blocks(Path(ddd_path), Path(sme_path))
            if series.t and len(tt) != len(series.t):
                raise ValueError(f"STOUT 行数 {len(tt)} ≠ STANA 行数 {len(series.t)}:"
                                 "轨迹截断或跨批 SME,整份弃用")
            angles, _pairs = extract_angles(blocks)
            keep = np.concatenate([[True], np.diff(tt) > 1e-12])
            fsys_t = tt[keep].tolist()
            fsys_hz = coi_frequency(tt, angles).tolist()
            vt_arr, term_pairs, n_gen_angle = extract_terminal_voltage(blocks)
            # 机端覆盖门的基准 = 算例真实机数(给了就用它;功角列可能自己就缺机)
            n_gen = max(n_gen_angle, n_gen_case) if n_gen_case else n_gen_angle
            if n_gen_case and n_gen_angle < n_gen_case:
                traj_note = (f"sme_angle_coverage_partial: 功角列仅 {n_gen_angle}/"
                             f"{n_gen_case} 机(COI 与机端覆盖基准受影响)")
            if len(term_pairs) > 0:
                vt_t = tt.tolist()
                vterm = vt_arr.tolist()
            vb_arr, buses = extract_bus_voltage(blocks)
            if buses:
                vbus_t = tt.tolist()
                vbus = vb_arr.tolist()
                bus_nos = buses
        except Exception as e:  # noqa: BLE001  轨迹不可解 → 代理口径,来源字段如实标注
            traj_note = f"stout_parse_fail: {e}"
            fsys_t = fsys_hz = vt_t = vterm = vbus_t = vbus = bus_nos = n_gen = None
    elif n_gen_case:
        n_gen = n_gen_case
    det = judge_stability_v1(series, fault_ts=fault_ts,
                             fsys_t=fsys_t, fsys_hz=fsys_hz,
                             vt_t=vt_t, vterm=vterm, vterm_n_gen=n_gen,
                             vbus_t=vbus_t, vbus=vbus, bus_nos=bus_nos)
    if traj_note:
        det["traj_note"] = traj_note
    return det
