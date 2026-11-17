"""T4.2 DoD:与 openclaw 判稳逻辑交叉核对(方案 10.3;阶段 2 起为**逐母线曲线**对齐)。

openclaw 判据函数化(出处 E:/AI-2/openclaw/判稳/相对值版-基准值更新版-初始值版-SME版/
st_curve_judgestablity.py,2026-07-13 逐行复核后忠实转写;其数据源为 FN*.DAT 曲线,
与我们的 STOUT.DDD 同为逐曲线数值,阶段 2 注入 Type=2 全母线电压后**两侧口径可逐母线对齐**):

- **功角**(judgeGenAngleStablility:205):任意时刻最大功角差 ≥500° → 失稳。与 D21 ①
  同源(手册 3.3.8),预期恒一致;我们取 STANA col6(= 逐机 max−min,单测锚定)。
- **低压 recover**(judgeBusVolStablility:246-263):**逐母线曲线独立判**,条件三合一 =
  ①t > fault_start+10 ②该母线在 [fault_start, fault_start+10] 内最低电压曾 <0.8
  ③当前点 <0.8。单点触发,无持续时间门槛。
  **与 D21 ② 的差异(如实标注,不改我们的判据)**:D21 字面"故障发生 10s 以后机端电压
  不得 <0.8",**无②号前置条件**——即 10s 后新出现的跌破(前 10s 一直健康)openclaw 不判、
  D21 判。我方以 D21 为准(用户裁决优先于外部脚本)。
- **低压 steady**(:265-279):末 1s 内该母线最低 <0.9 → 失稳。逐母线"任一"= 全网 vmin
  末 1s <0.9(等价),与 D21 第二分句口径一致。
- **过电压三级**(:281-344):逐母线,连续持续,绝对值模式默认 1.2/0.02s、1.3/0.1s、
  1.4/0.5s(与 D21 ③ 同参数,本项目已移植)。
- **频率**:openclaw 的 judgeFrequencyStablility 对"发电机频率差 Δf(Hz)"曲线直接用
  **绝对** Hz 阈值(49/48.5/48)比较(:446),实测其自身报告里 Δf≈0 的曲线被判"低于
  48.5Hz 持续"而恒失稳(判稳.txt:2119)——**口径不可比,不入核对**(我方 D21 ④ 为 COI
  系统频率持续 0.5s,用户裁决)。
- 故障起始时刻:openclaw 逐曲线自动检测(电压跳变最大点,:142-178);我方取 ST.S11 的 Ts。
  本脚本对 openclaw 侧忠实复现其自动检测,差异如实计入报告。

抽样:--n 0 = 全量核对(1800 例分钟级,阶段 2 默认);否则全部失稳例 + 功角临界带优先。
输出 <out>/crosscheck.json:逐例双方判定 + 分歧明细 + 一致率 + 分歧归因分句。

  python scripts/ts_crosscheck.py --labels runs/ts_labels_cepri36 [--n 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.psasp_io.stana import parse_stana  # noqa: E402
from gridfm.psasp_io.stout import (extract_bus_voltage,  # noqa: E402
                                   parse_stout_blocks)

ANGLE_DEG = 500.0
V_RECOVER = 0.8
V_STEADY = 0.9
OV_TIERS = ((1.2, 0.02), (1.3, 0.1), (1.4, 0.5))
VOL_JUMP_THRESHOLD = 0.1          # openclaw detect_fault_start_time 默认


def openclaw_angle(series) -> bool:
    """True = 稳(max 功角差 <500°)。judgeGenAngleStablility:205。"""
    return (max(series.angle_max) < ANGLE_DEG) if series.t else True


def openclaw_detect_fault_start(t, v) -> float:
    """openclaw detect_fault_start_time(:142-178)忠实转写:相邻点跳变绝对值最大
    且 >0.1pu 的时刻;否则首个 <0.8 的时刻;否则 0.0。**逐曲线各算各的**。"""
    max_change, fault_time = 0.0, 0.0
    for i in range(1, len(t)):
        ch = abs(v[i] - v[i - 1])
        if ch > VOL_JUMP_THRESHOLD and ch > max_change:
            max_change, fault_time = ch, t[i]
    if max_change > VOL_JUMP_THRESHOLD:
        return float(fault_time)
    for i in range(len(t)):
        if v[i] < V_RECOVER:
            return float(t[i])
    return 0.0


def openclaw_bus_voltage(t, v) -> dict:
    """openclaw judgeBusVolStablility(:213-344)对**一条母线电压曲线**的忠实转写。"""
    if len(t) == 0:
        return {"stable": True, "branch": "empty"}
    fs = openclaw_detect_fault_start(t, v)
    fe = fs + 10.0
    t_last = float(t[-1])
    min_post10 = float("inf")
    recover_bad = False
    for i in range(len(t)):
        tp, vv = float(t[i]), float(v[i])
        if tp < fs:
            continue
        if tp <= fe and vv < min_post10:
            min_post10 = vv
        if tp > fe and min_post10 < V_RECOVER and vv < V_RECOVER:
            recover_bad = True
    last1 = [float(v[i]) for i in range(len(t)) if t[i] >= t_last - 1.0]
    steady_bad = bool(last1) and min(last1) < V_STEADY
    ov_bad, ov_tier = False, None
    for th, dur in OV_TIERS:
        active, start = False, 0.0
        for i in range(len(t)):
            tp, vv = float(t[i]), float(v[i])
            if vv > th:
                if not active:
                    active, start = True, tp
                elif tp - start >= dur:
                    ov_bad, ov_tier = True, {"threshold_pu": th, "duration_s": dur}
                    break
            else:
                active = False
        if ov_bad:
            break
    branches = [b for b, bad in (("recover", recover_bad), ("steady", steady_bad),
                                 ("overvoltage", ov_bad)) if bad]
    return {"stable": not branches, "branches": branches, "fault_start": fs,
            "min_post10": (None if min_post10 == float("inf") else round(min_post10, 5)),
            "last1s_min": round(min(last1), 5) if last1 else None,
            "overvolt_tier": ov_tier,
            "recover_branch_active": fe < t_last}


def openclaw_voltage_all_buses(t, vbus, bus_nos) -> dict:
    """全母线扫描:任一母线失稳 → 电压失稳(openclaw 逐曲线独立判 + 报告汇总口径)。"""
    hits, branches = [], Counter()
    for k, b in enumerate(bus_nos):
        d = openclaw_bus_voltage(t, vbus[:, k])
        if not d["stable"]:
            hits.append({"bus": b, **{kk: d[kk] for kk in
                                      ("branches", "fault_start", "min_post10",
                                       "last1s_min", "overvolt_tier")}})
            for br in d["branches"]:
                branches[br] += 1
    return {"stable": not hits, "n_bad_buses": len(hits),
            "branch_counts": dict(branches), "worst": hits[:3]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="ts_labels 输出目录")
    ap.add_argument("--n", type=int, default=0, help="0 = 全量核对(阶段 2 默认)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.labels)
    src = root / "labels_v1.jsonl"
    if not src.exists():
        print(f"[核对] 缺 {src}(先跑 ts_relabel)")
        return 1
    rows = [json.loads(ln) for ln in src.read_text(encoding="utf-8").splitlines()]
    feas = [r for r in rows if r["feasible"]
            and (root / "stana" / f"{r['tag']}.dat").exists()
            and (root / "stout" / f"{r['tag']}.ddd").exists()]
    if a.n:
        uns = [r for r in feas if not r["stable"]]
        marg = [r for r in feas if r["stable"] and (r.get("angle_peak_deg") or 0) >= 300]
        rest = [r for r in feas if r["stable"] and (r.get("angle_peak_deg") or 0) < 300]
        rng = np.random.default_rng(a.seed)
        pick = uns[:max(a.n // 2, 1)] + marg[:max((a.n - min(len(uns), a.n // 2)) // 2, 0)]
        need = a.n - len(pick)
        if need > 0 and rest:
            pick += [rest[i] for i in rng.choice(len(rest), min(need, len(rest)),
                                                 replace=False)]
        feas = pick[:a.n]

    out_rows, agree = [], 0
    disagree_by = Counter()
    n_novbus = 0
    for r in feas:
        series = parse_stana(root / "stana" / f"{r['tag']}.dat")
        sme = root / "sme" / f"{r['tag']}.sme"
        if not sme.exists():
            sme = root / "ST.SME"
        try:
            t, blocks = parse_stout_blocks(root / "stout" / f"{r['tag']}.ddd", sme)
            vbus, bus_nos = extract_bus_voltage(blocks)
        except Exception as e:  # noqa: BLE001
            out_rows.append({"tag": r["tag"], "error": str(e)})
            continue
        if not bus_nos:                       # 无 Type=2 曲线(旧批)→ 不做逐母线核对
            n_novbus += 1
            continue
        oc_a = openclaw_angle(series)
        oc_v = openclaw_voltage_all_buses(t, vbus, bus_nos)
        oc_stable = oc_a and oc_v["stable"]
        # 我方口径(D21 v1);频率分句在 openclaw 侧不可比 → 另报"去频率"对照
        ours = r["stable"]
        ours_nofreq = not [ty for ty in (r.get("instability_types") or [])
                           if ty != "frequency"]
        same = (oc_stable == ours_nofreq)
        agree += int(same)
        if not same:
            key = ("我稳它失稳" if oc_stable is False else "我失稳它稳")
            disagree_by[key] += 1
        out_rows.append({
            "tag": r["tag"], "ours_stable": ours, "ours_stable_nofreq": ours_nofreq,
            "ours_types": r.get("instability_types"),
            "ours_voltage_modes": r.get("voltage_modes"),
            "openclaw_stable": oc_stable, "agree": same,
            "openclaw_angle_ok": oc_a, "openclaw_voltage": oc_v,
            "angle_peak_deg": r.get("angle_peak_deg"),
            "vmin_pu": r.get("vmin_pu"), "last1s_vmin": r.get("last1s_vmin"),
            "n_buses": len(bus_nos),
        })
    cmp_rows = [r for r in out_rows if "agree" in r]
    rep = {"n_compared": len(cmp_rows), "agree": agree,
           "agree_rate": round(agree / max(len(cmp_rows), 1), 4),
           "disagree_by": dict(disagree_by),
           "n_skipped_no_vbus": n_novbus,
           "compare_basis": "openclaw(功角 + 逐母线电压三分句)vs 我方 D21 v1 去频率分句",
           "notes": [
               "阶段 2 起为**逐母线曲线**对齐(SME Type=2 全母线电压);此前只能在 STANA "
               "vmin/vmax 包络上松弛为'存在母线'",
               "频率不入核对:openclaw 对 Δf(Hz) 曲线用绝对 Hz 阈值比较(其自身报告里"
               "Δf≈0 曲线恒判失稳,判稳.txt:2119),口径不可比;我方 D21 ④ = COI 持续 0.5s",
               "recover 分句差异(已知,不改我方):openclaw 多一个前置条件'故障后 10s 窗内"
               "最低电压曾 <0.8';D21 字面无此前置——10s 后新出现的跌破,openclaw 不判、D21 判",
               "故障起始时刻:openclaw 逐曲线自动检测(电压跳变最大点);我方取 ST.S11 Ts",
           ],
           "rows": out_rows}
    (root / "crosscheck.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    print(f"[核对] 逐母线口径 {len(cmp_rows)} 例:一致 {agree}"
          f"(率 {rep['agree_rate']:.4f});分歧 {dict(disagree_by)}"
          f"{f';跳过无电压曲线 {n_novbus}' if n_novbus else ''}")
    for r in cmp_rows:
        if not r["agree"]:
            print(f"  分歧 {r['tag']}: 我方(去频率)={r['ours_stable_nofreq']} "
                  f"类型{r['ours_types']} 电压模式{r['ours_voltage_modes']} | "
                  f"openclaw={r['openclaw_stable']} 角{r['openclaw_angle_ok']} "
                  f"坏母线{r['openclaw_voltage']['n_bad_buses']} "
                  f"{r['openclaw_voltage']['branch_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
