# -*- coding: utf-8 -*-
r"""负类卡片级诊断(2026-08-21;静态首考配套,**不跑 PSASP、不作裁决**):
只读 libs/neg_* 的输入卡与发散残留报告,回答"降载规则为何全败"的机理问题。

每个负类算:负荷因子、非平衡机总出力 PG(已被 Pmax 截断)、总负荷 PL、缺口 PL-PG、
平衡机 Pmax、发散残留 lfreport.lis 的平衡机注入(*=越界)与失配轨迹;再按 load_only
规则(全网负荷 ×0.85^k,机组不动)**算术推演** k=1..3 的缺口——若一步就反号成巨额倒送,
即说明该规则对"发电缺口型"发散结构性失配(PSASP 复核待狗恢复后由 static_exam_a 做)。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(r"E:\统一计算大模型-迁移包\增量-20260902\N-k其他推理器实验-20260905\grid-foundation")
LINE = Path(r"E:\统一计算大模型-迁移包\增量-20260902\大算例样本线-20260820")
sys.path.insert(0, str(REPO))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.psasp_io.cards import GenView, LoadView  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402

TIERS = {"2000": ["neg_activsg2000", "neg_activsg2000_b2"],
         "70k": ["neg_activsg70k", "neg_activsg70k_b2"]}
POS = {"2000": ["mp_activsg2000", "mp_activsg2000_b2"],
       "70k": ["mp_activsg70k", "mp_activsg70k_b2"]}
LOAD_SCALE = 0.85


def card_stats(cdir: Path) -> dict:
    l5 = Card.load(cdir / "LF.L5")
    l6 = Card.load(cdir / "LF.L6")
    pg = 0.0
    n_at_pmax = n_pv = 0
    slack = None
    for i, r in enumerate(l5.records):
        g = GenView(r, i)
        if g.mark != 1:
            continue
        if g.type_code == 0:
            slack = {"bus": g.bs_no, "pg_card": g.pg, "pmin": g.pmin, "pmax": g.pmax}
            continue
        n_pv += 1
        pg += g.pg
        if g.pmax > 0 and g.pg >= g.pmax - 1e-6:
            n_at_pmax += 1
    pl = ql = 0.0
    for i, r in enumerate(l6.records):
        lv = LoadView(r, i)
        if lv.mark == 1:
            pl += lv.pl
            ql += lv.ql
    return {"pg_nonslack": pg, "pl": pl, "ql": ql, "gap_pl_minus_pg": pl - pg,
            "n_pv": n_pv, "n_at_pmax": n_at_pmax, "slack": slack}


def parse_lfreport(cdir: Path) -> dict:
    """发散残留:失配轨迹(迭代号,母线,失配值)与平衡机注入行(数值后 * = 越界)。"""
    p = cdir / "lfreport.lis"
    if not p.exists():
        return {}
    txt = p.read_bytes().decode("gb18030", errors="replace")
    it = re.findall(r"^\s+(\d+)\s+(\S+)\s+([-\d.]+)\s*$", txt, flags=re.M)
    traj = [(int(a), b, float(c)) for a, b, c in it]
    # 两段重复打印(PSASP 报告格式),只取第一段:迭代号重新从 0 开始处截断
    first = []
    for k, row in enumerate(traj):
        if k > 0 and row[0] <= first[-1][0]:
            break
        first.append(row)
    m = re.search(r"^\s+(B\S+)\s+([-\d.]+)(\*?)\s+([-\d.]+)(\*?)\s*$", txt, flags=re.M)
    slack_line = None
    if m:
        slack_line = {"bus": m.group(1), "p": float(m.group(2)), "p_over": m.group(3) == "*",
                      "q": float(m.group(4)), "q_over": m.group(5) == "*"}
    return {"n_iter": (first[-1][0] + 1) if first else 0,
            "mismatch_first": first[0][2] if first else None,
            "mismatch_last": first[-1][2] if first else None,
            "mismatch_min": min(r[2] for r in first) if first else None,
            "worst_bus_last": first[-1][1] if first else None,
            "slack_report": slack_line}


def main() -> int:
    out = {"note": "卡片级诊断,不跑 PSASP、无裁决权;PSASP 复核见 static_exam_a 结果",
           "load_scale_rule": LOAD_SCALE, "tiers": {}}
    for tier, libs in TIERS.items():
        rows = []
        for lib in libs:
            base = card_stats(LINE / "libs" / lib / "_base_case")
            for d in sorted((LINE / "libs" / lib / "cases").iterdir()):
                if not d.is_dir():
                    continue
                st = card_stats(d)
                rep = parse_lfreport(d)
                gap = st["gap_pl_minus_pg"]
                pmax = st["slack"]["pmax"] if st["slack"] else None
                # load_only 算术推演:负荷 ×0.85^k,机组不动 → 缺口 = PL*0.85^k - PG
                proj = {f"k{k}": st["pl"] * LOAD_SCALE ** k - st["pg_nonslack"] for k in (1, 2, 3)}
                rows.append({
                    "case": d.name, "lib": lib,
                    "pl": round(st["pl"], 2), "pg_nonslack": round(st["pg_nonslack"], 2),
                    "gap_pl_minus_pg": round(gap, 2),
                    "base_gap": round(base["gap_pl_minus_pg"], 2),
                    "pl_over_base": round(st["pl"] / base["pl"], 4),
                    "pg_over_base": round(st["pg_nonslack"] / base["pg_nonslack"], 4),
                    "n_pv": st["n_pv"], "n_at_pmax": st["n_at_pmax"],
                    "n_at_pmax_base": base["n_at_pmax"],
                    "slack_bus": st["slack"]["bus"] if st["slack"] else None,
                    "slack_pmin": st["slack"]["pmin"] if st["slack"] else None,
                    "slack_pmax": pmax,
                    "gap_over_slack_pmax": round(gap / pmax, 2) if pmax else None,
                    "load_only_gap_k1_k2_k3": [round(v, 1) for v in proj.values()],
                    "report": rep,
                })
        n = len(rows)
        gaps = [r["gap_pl_minus_pg"] for r in rows]
        over = [r for r in rows if r["slack_pmax"] and r["gap_pl_minus_pg"] > r["slack_pmax"]]
        flip = [r for r in rows if r["load_only_gap_k1_k2_k3"][0] < -abs(r["gap_pl_minus_pg"])]
        rep_over = [r for r in rows if (r["report"].get("slack_report") or {}).get("p_over")]
        out["tiers"][tier] = {
            "n_neg": n,
            "gap_min_max": [round(min(gaps), 2), round(max(gaps), 2)] if gaps else None,
            "n_gap_exceeds_slack_pmax": len(over),
            "n_report_slack_p_over_limit": len(rep_over),
            "n_load_only_k1_overshoots_to_surplus": len(flip),
            "lf_min_max": [min(r["pl_over_base"] for r in rows), max(r["pl_over_base"] for r in rows)] if rows else None,
            "pg_ratio_min_max": [min(r["pg_over_base"] for r in rows), max(r["pg_over_base"] for r in rows)] if rows else None,
            "rows": rows,
        }
        # 正库对照:同档正类缺口分布(只取前 30 个,卡片级)
        pos = []
        for lib in POS[tier]:
            for d in sorted((LINE / "libs" / lib / "cases").iterdir())[:15]:
                st = card_stats(d)
                pos.append(round(st["gap_pl_minus_pg"], 2))
        out["tiers"][tier]["pos_gap_sample"] = {"n": len(pos), "min": min(pos), "max": max(pos)}
        print(f"[{tier}] 负类 {n}: 缺口 PL-PG ∈ [{min(gaps):.1f}, {max(gaps):.1f}] pu; "
              f"缺口>平衡机Pmax {len(over)}/{n}; 残留报告平衡机P越界 {len(rep_over)}/{n}; "
              f"负荷×0.85 一步即反号成更大倒送 {len(flip)}/{n}; "
              f"正类缺口样本 ∈ [{min(pos):.1f}, {max(pos):.1f}] (n={len(pos)})", flush=True)
    o = LINE / "静态首考-20260821" / "neg_card_diagnosis.json"
    o.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"落盘 {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
