# -*- coding: utf-8 -*-
r"""从阅卷工作副本算"实际落地"的动作幅度(2026-08-21):模型臂名义动作会被 [Pmin,Pmax]/v_set 界截断,
必须用 _original_backup/LF.L5(病态起点)与 LF.L5(修后)逐台对比。
  有功:Σ|ΔP| 实际、改动机数、推到上/下限的机数、非平衡机总出力变化(≈平衡机反向吸收量);
  无功:Σ|Δv_set| 实际、改动机数、并补投切数(LF.L2 mark 变化)。
  python realized_actions.py --task active --tier 118     # 打印并写 runs/<task>_<tier>/realized.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(r"E:\统一计算大模型-迁移包\增量-20260902\N-k其他推理器实验-20260905\grid-foundation")
LINE = Path(r"E:\统一计算大模型-迁移包\增量-20260902\大算例样本线-20260820")
RUNS = LINE / "调整器跨规模验证-20260821" / "runs"
sys.path.insert(0, str(REPO))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
from gridfm.psasp_io.cards import AclineView, GenView  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402
sys.path.insert(0, str(LINE / "scripts"))
from neg_card_diagnosis import parse_lfreport  # noqa: E402


def _slack_dist(p, pm):
    """平衡机 P 到 [Pmin,Pmax] 带的距离(带内为 0)。"""
    if p is None or pm is None:
        return None
    return max(0.0, pm[0] - p, p - pm[1])


def slack_after(work: Path) -> dict:
    """修前/修后平衡机注入(PSASP 残留报告 slack 行;* = 越出限值)。
    合理性用**相对判据**(2026-08-21 14:30 改):修后到限值带的距离 ≤ 修前 + 0.1 pu(回到带内或在往回走;0.1 pu 为网损噪声容差)。
    理由:负荷增长病种本身就把平衡机顶出限外(118 有功初始 28 pu vs Pmax 8;3120 底图平衡机本就超 Pmax),绝对判据谁都过不了。"""
    rep1 = parse_lfreport(work).get("slack_report") or {}
    rep0 = parse_lfreport(work / "_original_backup").get("slack_report") or {}
    l5 = Card.load(work / "LF.L5")
    pm = None
    for i, r in enumerate(l5.records):
        g = GenView(r, i)
        if g.mark == 1 and g.type_code == 0:
            pm = (g.pmin, g.pmax)
            break
    d0, d1 = _slack_dist(rep0.get("p"), pm), _slack_dist(rep1.get("p"), pm)
    # 容差 0.1 pu(10 MW):无功动作只改网损,平衡机 P 抖动 ±0.03 pu 是噪声,不应判"变远"
    plausible = (d1 is not None and d0 is not None and d1 <= d0 + 0.1)
    return {"slack_p_before": rep0.get("p"), "slack_p": rep1.get("p"),
            "slack_p_over_before": rep0.get("p_over"), "slack_p_over": rep1.get("p_over"),
            "slack_q_over": rep1.get("q_over"), "slack_pmin_pmax": pm,
            "slack_dist_before": (round(d0, 3) if d0 is not None else None),
            "slack_dist_after": (round(d1, 3) if d1 is not None else None),
            "plausible": plausible}


def gens_of(l5: Card):
    out = {}
    for i, r in enumerate(l5.records):
        g = GenView(r, i)
        if g.mark == 1 and g.type_code != 0:
            out[i] = (g.bs_no, g.pg, g.pmin, g.pmax, g.v0)
    return out


def shunt_marks(l2: Card):
    out = {}
    for i, r in enumerate(l2.records):
        a = AclineView(r, i)
        if a.i_no == a.j_no:
            out[i] = a.mark
    return out


def realized(work: Path, task: str) -> dict:
    bak = work / "_original_backup"
    g0, g1 = gens_of(Card.load(bak / "LF.L5")), gens_of(Card.load(work / "LF.L5"))
    if task == "active":
        dps = {gi: g1[gi][1] - g0[gi][1] for gi in g0 if gi in g1 and abs(g1[gi][1] - g0[gi][1]) > 1e-6}
        at_hi = sum(1 for gi in dps if g0[gi][3] > g0[gi][2] + 1e-9 and abs(g1[gi][1] - g0[gi][3]) < 1e-6)
        at_lo = sum(1 for gi in dps if abs(g1[gi][1] - g0[gi][2]) < 1e-6)
        out = {"n_changed": len(dps), "sum_abs_dp": round(sum(abs(v) for v in dps.values()), 3),
               "net_dp": round(sum(dps.values()), 3), "n_pushed_to_pmax": at_hi, "n_pushed_to_pmin": at_lo,
               "pg_total_before": round(sum(g[1] for g in g0.values()), 2),
               "pg_total_after": round(sum(g[1] for g in g1.values()), 2)}
        out.update(slack_after(work))
        return out
    dvs = {gi: g1[gi][4] - g0[gi][4] for gi in g0 if gi in g1 and abs(g1[gi][4] - g0[gi][4]) > 1e-6}
    s0, s1 = shunt_marks(Card.load(bak / "LF.L2")), shunt_marks(Card.load(work / "LF.L2"))
    n_sw = sum(1 for i in s0 if i in s1 and s0[i] != s1[i])
    out = {"n_changed": len(dvs), "sum_abs_dv": round(sum(abs(v) for v in dvs.values()), 4),
           "max_abs_dv": round(max([abs(v) for v in dvs.values()] or [0.0]), 4), "n_shunt_switched": n_sw}
    out.update(slack_after(work))
    out["plausible"] = None          # 无功任务不适用平衡机 P 判据(只改 v_set,P 变化是网损副作用;成功判据已含电压带)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["active", "reactive"])
    ap.add_argument("--tier", required=True)
    a = ap.parse_args()
    rd = RUNS / ("%s_%s" % (a.task, a.tier))
    res = json.loads((rd / "arms_results.json").read_text(encoding="utf-8"))
    out = {}
    for cid, v in res.items():
        rec = {}
        for arm in ("model", "rule"):
            w = rd / cid / arm
            if (w / "LF.L5").is_file() and (w / "_original_backup" / "LF.L5").is_file():
                rec[arm] = realized(w, a.task)
                rec[arm]["ok"] = bool(v[arm]["ok"])
        out[cid] = rec
    (rd / "realized.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    key = "sum_abs_dp" if a.task == "active" else "sum_abs_dv"
    for arm in ("model", "rule"):
        okv = [r[arm] for r in out.values() if arm in r and r[arm]["ok"]]
        allv = [r[arm] for r in out.values() if arm in r]
        if not allv:
            continue
        def med(lst, k):
            xs = [x[k] for x in lst]
            return (round(st.median(xs), 3), round(min(xs), 3), round(max(xs), 3)) if xs else None
        print("[%s/%s] %s 实际: 全部 n=%d 改机数 med/min/max %s | %s %s" % (
            a.task, a.tier, arm, len(allv), med(allv, "n_changed"), key, med(allv, key)))
        if okv:
            extra = ""
            n_plaus = sum(1 for x in okv if x.get("plausible")) if a.task == "active" else None
            n_in = sum(1 for x in okv if not x.get("slack_p_over"))
            extra0 = (" | 合理(平衡机距限不变远) %d/%d [修后在限内 %d; 修前距限 med %s → 修后 med %s]" % (
                n_plaus, len(okv), n_in, med(okv, "slack_dist_before"), med(okv, "slack_dist_after"))
                      if a.task == "active" else " | 合理性闸不适用(无功)")
            if a.task == "active":
                extra = extra0 + " | 推到Pmax %s 推到Pmin %s | 非平衡机总出力变化 net_dp %s" % (
                    med(okv, "n_pushed_to_pmax"), med(okv, "n_pushed_to_pmin"), med(okv, "net_dp"))
            else:
                extra = extra0 + " | 投切并补 %s | 单机最大|Δv| %s" % (med(okv, "n_shunt_switched"), med(okv, "max_abs_dv"))
            print("            成功例 n=%d 改机数 %s | %s %s%s" % (len(okv), med(okv, "n_changed"), key, med(okv, key), extra))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
