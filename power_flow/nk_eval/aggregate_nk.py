#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总潮流 N-0/N-1/N-2 配对评估（三座均值 / 三座折回最大 / 零训练提示），输出 nk_results/aggregate_pf_nk.json 与 markdown 表。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = 57.29577951308232
HERE = Path(__file__).resolve().parent / "nk_results"
E13 = "R1dcL_edge0_wrap_L_vh_rf4qv_aux1_hd0.2_rd0.2"
E10 = "R1dcL_edge0_wrap_L_vh"


def fmt_v(x):
    return f"{x:.1e}".replace("e-0", "×10⁻").replace("e-", "×10⁻") if x < 1e-4 else f"{x:.5f}".rstrip("0")


def main():
    agg = {}
    for k in (0, 1, 2):
        files = sorted(HERE.glob(f"byfam_pf_*_N{k}.json"))
        if not files:
            continue
        groups = {}
        for f in files:
            r = json.load(open(f, encoding="utf-8"))
            for g in r["groups"]:
                groups.setdefault((g["model"].rsplit("_s", 1)[0], g["family"]), []).append(g)
        h = {}
        for hf in sorted(HERE.glob(f"hint_pf_*_N{k}*.json")):
            for x in json.load(open(hf, encoding="utf-8")):
                h[x["family"]] = x
        for (m, fam), gs in groups.items():
            agg.setdefault(fam, {}).setdefault(f"N{k}", {})[("E13" if m == E13 else "E10" if m == E10 else m)] = dict(
                n=gs[0]["n_samples"], seeds=len(gs),
                th_mean_deg=float(np.mean([x["theta"]["mean"] for x in gs]) * R),
                th_max_deg=float(max(x["theta_wrap"]["max"] for x in gs) * R),
                v_mean=float(np.mean([x["v"]["mean"] for x in gs])), v_max=float(max(x["v"]["max"] for x in gs)),
                per_seed_th=[x["theta"]["mean"] * R for x in gs])
            if fam in h:
                agg[fam][f"N{k}"]["hints"] = dict(dc_theta_deg=h[fam]["dc_lossy_theta_deg"]["mean"], qv_v=h[fam]["qv_hint_v_pu"]["mean"])
    (HERE / "aggregate_pf_nk.json").write_text(json.dumps(agg, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["| 网 | 变体 | n | E13 θ mean / max (°) | E13 V mean / max | E10 θ mean / max (°) | E10 V mean | DC 提示 θ (°) | Q–V 提示 V |", "|---|---|---|---|---|---|---|---|---|"]
    for fam in agg:
        for k in ("N0", "N1", "N2"):
            d = agg[fam].get(k)
            if not d or "E13" not in d:
                continue
            e13, e10, hh = d["E13"], d.get("E10", {}), d.get("hints", {})
            lines.append(f"| {fam} | {k} | {e13['n']} | {e13['th_mean_deg']:.5f} / {e13['th_max_deg']:.3f} | {fmt_v(e13['v_mean'])} / {fmt_v(e13['v_max'])} | "
                         f"{e10.get('th_mean_deg', float('nan')):.4f} / {e10.get('th_max_deg', float('nan')):.3f} | {fmt_v(e10.get('v_mean', float('nan'))) if e10 else '—'} | "
                         f"{hh.get('dc_theta_deg', float('nan')):.4f} | {hh.get('qv_v', float('nan')):.5f} |")
    (HERE / "aggregate_pf_nk.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
