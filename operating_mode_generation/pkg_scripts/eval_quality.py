# -*- coding: utf-8 -*-
r"""方式库质量验收器（外协包 E 唯一计分脚本；甲方盲验收用同一份代码、只换种子）。

用法：
  python scripts/eval_quality.py --lib libs/<你的库名> --tier 3120
输出：
  屏幕摘要 + <库目录>/quality_report.json

指标（口径与主线合格闸逐字一致）：
  ① 合格率（主指标）：逐方式读 LF.LP1 电压、对照 MATPOWER 自带逐母线带（.m 的 VMIN/VMAX 列）——
     3120（放宽档）：bvr_own（越限母线占比）≤ 0.1% 判合格；118（严格档）：越限母线数 = 0 判合格。
  ② 收得率：gen_report.json 的 n_converged / n_attempts（生成脚本自动写）。
  ③ 多样性护栏（防"缩到一个安全点"式刷分，须报告）：
     - 方式间总有功负荷 ΣPL 的变异系数（CV）；
     - vmin / vmax 的方式间极差。
     与 baseline/audit_baseline.json 的现库对照并排打印；护栏显著塌缩（CV < 现库 60%）会被标注。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG / "grid-foundation"))
sys.path.insert(0, str(_PKG / "scripts"))

from gridfm.psasp_io import cards            # noqa: E402
from gridfm.psasp_io.lowlevel import Card    # noqa: E402
from audit_static_feasibility import read_m  # noqa: E402（包内版，路径自足）

# [外协包 F,2026-08-25] 补 2000/300 两档:任务书 §6 自检第二步写的
# `--tier 2000` 在随包版 TIER_M 里不存在,原样跑会 argparse 报错。
# 2000/300 沿用"严格档"(越限母线数 = 0 判合格),与 118 同口径;3120 仍为放宽档。
TIER_M = {"118": ("case118.m", 118, False), "300": ("case300.m", 300, False),
          "2000": ("case_ACTIVSg2000.m", 2000, False),
          "3120": ("case3120sp.m", 3120, True)}
BVR_RELAXED = 0.001
EPS = 5e-4


def bus_voltage(case_dir: Path, n_bus: int) -> np.ndarray:
    v = np.zeros(n_bus)
    p = case_dir / "LF.LP1"
    if not p.is_file():
        return v
    for r in cards.parse_lp1(Card.load(p)):
        if 1 <= r.bus_no <= n_bus and np.isfinite(r.v_pu) and r.v_pu < 100:
            v[r.bus_no - 1] = r.v_pu
    return v


def sum_pl(case_dir: Path) -> float:
    p = case_dir / "LF.L6"
    if not p.is_file():
        return float("nan")
    tot = 0.0
    for i, rec in enumerate(Card.load(p).records):
        lv = cards.LoadView(rec, i)
        if lv.mark == 1:
            tot += float(lv.pl)
    return tot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True, help="方式库目录（含 cases/<sid>/，每方式一个 PSASP 目录且已有 LF.LP1）")
    ap.add_argument("--tier", required=True, choices=list(TIER_M))
    args = ap.parse_args()

    mfile, n_bus, relaxed = TIER_M[args.tier]
    _base, bus, _br = read_m(_PKG / "matpower_cases" / mfile)
    vmax_own, vmin_own = bus[:, 11].copy(), bus[:, 12].copy()

    lib = Path(args.lib)
    if not lib.is_absolute():
        lib = _PKG / lib
    case_root = lib / "cases"
    dirs = sorted(d for d in case_root.iterdir() if d.is_dir()) if case_root.is_dir() else []
    if not dirs:
        print("!! %s 下没有 cases/<sid>/ 方式目录" % lib)
        return 2

    rows, n_ok = [], 0
    for d in dirs:
        v = bus_voltage(d, n_bus)
        live = v > 0.01
        n_live = int(live.sum())
        if n_live == 0:
            rows.append({"sid": d.name, "ok": False, "reason": "no_lp1"})
            continue
        viol = live & ((v < vmin_own - EPS) | (v > vmax_own + EPS))
        nvo = int(viol.sum())
        bvr = nvo / max(n_live, 1)
        ok = (bvr <= BVR_RELAXED + 1e-9) if relaxed else (nvo == 0)
        n_ok += int(ok)
        rows.append({"sid": d.name, "ok": bool(ok), "n_viol_own": nvo, "bvr_own": round(bvr, 5),
                     "vmin": round(float(v[live].min()), 4), "vmax": round(float(v[live].max()), 4),
                     "sum_pl": round(sum_pl(d), 3)})

    n = len(rows)
    pls = np.array([r["sum_pl"] for r in rows if r.get("sum_pl") == r.get("sum_pl")])
    vmins = np.array([r["vmin"] for r in rows if "vmin" in r])
    vmaxs = np.array([r["vmax"] for r in rows if "vmax" in r])
    cv = float(pls.std() / pls.mean()) if pls.size and pls.mean() > 0 else None
    gr = lib / "gen_report.json"
    yld = None
    if gr.is_file():
        g = json.loads(gr.read_text(encoding="utf-8"))
        att = g.get("n_attempts") or 0
        yld = (g.get("n_converged") or 0) / att if att else None

    # 现库对照
    base_line = {}
    bl = _PKG / "baseline" / "audit_baseline.json"
    if bl.is_file():
        b = json.loads(bl.read_text(encoding="utf-8"))
        brs = (b["tiers"].get(args.tier) or {}).get("rows") or []
        if brs:
            if relaxed:
                bok = sum(1 for r in brs if r["bvr_own"] <= BVR_RELAXED + 1e-9)
            else:
                bok = sum(1 for r in brs if r["n_viol_own"] == 0)
            base_line = {"n": len(brs), "n_ok": bok,
                         "vmin_range": [min(r["vmin"] for r in brs), max(r["vmin"] for r in brs)],
                         "vmax_range": [min(r["vmax"] for r in brs), max(r["vmax"] for r in brs)]}

    rep = {"lib": str(lib), "tier": args.tier, "relaxed": relaxed, "n_modes": n,
           "n_qualified": n_ok, "qualified_rate": round(n_ok / n, 4) if n else None,
           "yield_converged_per_attempt": (round(yld, 4) if yld is not None else None),
           "diversity": {"sum_pl_cv": (round(cv, 4) if cv is not None else None),
                         "vmin_range": ([round(float(vmins.min()), 4), round(float(vmins.max()), 4)] if vmins.size else None),
                         "vmax_range": ([round(float(vmaxs.min()), 4), round(float(vmaxs.max()), 4)] if vmaxs.size else None)},
           "baseline_ref": base_line, "rows": rows}
    (lib / "quality_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[eval] %s tier=%s 方式 %d | 合格 %d (%.0f%%)%s | 收得率 %s | ΣPL CV %s | vmin %s vmax %s" % (
        lib.name, args.tier, n, n_ok, 100.0 * n_ok / n if n else 0,
        (" [现库参照 %d/%d]" % (base_line.get("n_ok", 0), base_line.get("n", 0))) if base_line else "",
        ("%.2f" % yld) if yld is not None else "—",
        ("%.4f" % cv) if cv is not None else "—",
        rep["diversity"]["vmin_range"], rep["diversity"]["vmax_range"]))
    print("[eval] 报告已写 %s" % (lib / "quality_report.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
