# -*- coding: utf-8 -*-
r"""局部性预筛半径普查（零 PSASP，§15.8 纪律"半径不能拍"的通用版）。
逐病例数"距越限母线 ≤H 跳内的可调机组台数"，并按既定标准推荐 H：
  **最小的 H，满足 (a) 无一病例 0 台（否则代码回退成不限制=白跑对照）且 (b) 台数中位落在 [8, 80]**；
  若没有 H 同时满足，退而取"无一例 0 台"里中位最接近 [8,80] 的那个，并在输出里标 fallback。
  python hop_census.py --lib viol_q_9241_local --n-bus 9241 --low 0.82 --high 1.18
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adj2_common as C  # noqa: E402
from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402

HOPS = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20]
# 定档标准（2026-08-27 09:30 第二次修正，三网全半径扫描实证）：
#   ① 最初"无 0 台 且 台数中位 ∈ [8,80]"是从 25k 抄的 → 04:55 证伪（成功率对更紧单调上升）；
#   ② 改成"无一例 0 台的最小 H" → 09:30 再修：它在 2869 上正好命中峰值，但在 1354/25k 上偏保守 3~6 个点。
#   全半径扫描（首批同卷，候选中位 / 成功率 / 0台回退例数）：
#     1354：188台 37% → 61台 57%(0退) → 22台 70%(0退) → **11台 73%(1退)** → 6台 63%(5退) → 3台 47%(7退)
#     2869：48台 57% → 23台 70%(0退) → **12台 70%(0退,潮流最省)** → 4台 63%(3退) → 3台 53%(6退) → 2台 40%(9退)
#     25k ：185台 44% → 45台 50%(0退) → **14台 56%(1退)** → 4台 28%(3退)
#   **峰值一律落在"候选中位 11~14 台"，与网规模无关；再紧就掉，且掉的那档正是 0 台回退从 ≤1 例跳到 3~9 例的那档。**
#   ③ 只用"0 台回退 ≤5%"在 70k 上会选 H=7（候选中位 **4.5**），远低于峰值带 —— 70k 实测 H=6(中位 2 台) 只有 30%、
#      H=10(中位 30 台) 55%，说明**回退率不是可移植的量，候选台数才是**。
#   故定档 = **最小的 H，同时满足 (a) 0 台回退 ≤ 5%(至少容 1 例) 且 (b) 候选台数中位 ≥ 8**。
#   已测三网仍全部命中峰值（1354 H5 中位 11 / 2869 H6 中位 12 / 25k H8 中位 14），
#   并对 70k 给出**可证伪的预测：H=8（中位 8）应优于 H=10 的 55%**（已排链实测）。
#   （0 台候选 = 代码回退成"不限制"，那一例等于没预筛。）
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--n-bus", type=int, required=True)
    ap.add_argument("--low", type=float, default=None, help="缺省从考卷 eval_summary.json 的 vio_band 取")
    ap.add_argument("--high", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lib = Path(a.lib) if Path(a.lib).is_absolute() else C.LINE / "libs" / a.lib
    low, high = a.low, a.high
    if low is None or high is None:
        band = json.loads((lib / "eval_summary.json").read_text(encoding="utf-8"))["vio_band"]
        low = low if low is not None else band[0]
        high = high if high is not None else band[1]
    cases = sorted(p for p in (lib / "cases").iterdir() if p.is_dir())
    tab = {h: [] for h in HOPS}
    n_tot = []
    for cd in cases:
        v = C.bus_voltage(cd, a.n_bus)
        live = v > 0.01
        bad = [int(i) + 1 for i in np.where(live & ((v < low - 1e-4) | (v > high + 1e-4)))[0]]
        _slack, gens = C.read_gens(Card.load(cd / "LF.L5"))
        gbus = {g["bus"] for g in gens.values()}
        n_tot.append(len(gbus))
        if not bad:
            continue
        d = C.hops_from(C.bus_adjacency(PsaspCase(cd)), bad)
        for h in HOPS:
            tab[h].append(sum(1 for b in gbus if d.get(b, 10 ** 9) <= h))
    print("[census] %s（%d 例，可诊 %d 例；可调机组母线数中位 %d；带 [%.2f, %.2f]）"
          % (a.lib, len(cases), len(tab[HOPS[0]]), int(np.median(n_tot)), low, high), flush=True)
    rows = {}
    for h in HOPS:
        xs = tab[h]
        if not xs:
            continue
        z = sum(1 for x in xs if x == 0)
        med = float(np.median(xs))
        rows[h] = {"med": med, "min": int(min(xs)), "max": int(max(xs)), "n_zero": z, "n": len(xs)}
        print("   ≤%2d 跳：中位 %6.0f  min %4d  max %5d  |  0 台病例 %d/%d"
              % (h, med, min(xs), max(xs), z, len(xs)), flush=True)
    n_cases = rows[HOPS[0]]["n"] if rows.get(HOPS[0]) else max((r["n"] for r in rows.values()), default=0)
    zmax = max(1, int(math.ceil(0.05 * n_cases)))
    MED_MIN = 8
    rec, fb = None, False
    for h in HOPS:                       # 最小的"回退 ≤5% 且 候选中位 ≥8"半径
        r = rows.get(h)
        if r and r["n_zero"] <= zmax and r["med"] >= MED_MIN:
            rec = h
            break
    if rec is None:                      # 放宽一条：只要中位 ≥8，取回退最少者（并明示回退例数）
        cands = [h for h in HOPS if rows.get(h) and rows[h]["med"] >= MED_MIN]
        if cands:
            rec = min(cands, key=lambda h: (rows[h]["n_zero"], h)); fb = True
        else:
            cands = [h for h in HOPS if rows.get(h)]
            if cands:
                rec = max(cands, key=lambda h: (rows[h]["med"], -h)); fb = True
    print("[census] 推荐 H = %s（标准：**回退 ≤ %d 例(5%%) 且 候选中位 ≥ %d 的最小 H**；台数中位 %s，回退 %s 例）%s"
          % (rec, zmax, MED_MIN, (rows[rec]["med"] if rec else None), (rows[rec]["n_zero"] if rec else None),
             "  ⚠fallback：无 H 同时满足两条,已放宽" if fb else ""), flush=True)
    out = Path(a.out) if a.out else (Path(__file__).resolve().parent / ("hop_census_%s.json" % a.lib))
    out.write_text(json.dumps({"lib": a.lib, "band": [low, high], "rows": {str(k): v for k, v in rows.items()},
                               "recommended_h": rec, "fallback": fb}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[census] 报告 %s" % out, flush=True)
    return 0 if rec else 5


if __name__ == "__main__":
    sys.exit(main())
