"""[G] R5 · **按预注册判门**(A 的推广裁决)—— 机械执行,不许人眼裁。

判据全文见 `docs/R5-补种子到5座-预注册-20260726.md` §3,本脚本逐条兑现:

**推广门(两条都过才建议推广成共享编码器)**
- **G1** 主指标②(低基率 90% 召回复核比例,**预注册三层基**,窗口口径):
  5 座 **≥4/5 同号(更省)** 且 `|Δ| > 2σ`;
  且**剔无分辨力层的两层基不得与三层基反号**(不许挑对自己有利的基)。
- **G2** ieee39 [外] pr_auc(窗口口径,对级共有网格):
  5 座 **≥4/5 同号(正)** 且 `|Δ| > 2σ` 且**中位 ≥5pp**(该指标自身三种子分辨阈)。

**硬门(不满足则该座不得进主读数)**:H1 收敛(触发早停)/ H2 口径 parity /
H3 共享基座指纹 / H4 解冻真发生。

118 [内] acc **只并报、不作推广理由**(锚口径已证不可分辨)。

用法:python scripts/r5_gate.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
FP_V1 = "ccf4260f2a8f5db5b6650a33b17d7a0f"
FP_NM = "12281dd483206efe09fdf19475472ebc"

A1 = ["ts_reasoner_r4_none_s0", "ts_reasoner_r4_none_s1", "ts_reasoner_r4_none_s2",
      "ts_reasoner_r5_none_s3", "ts_reasoner_r5_none_s4"]
A3 = ["ts_reasoner_r4_all_s0", "ts_reasoner_r2c_all_s1_long", "ts_reasoner_r4_all_s2",
      "ts_reasoner_r5_all_s3", "ts_reasoner_r5_all_s4"]


def hard_gates():
    rows, ok = [], True
    for arm, runs in (("臂① none", A1), ("臂③ all", A3)):
        for r in runs:
            f = RUNS / r / "ts_report.json"
            if not f.is_file():
                rows.append({"run": r, "H1": "缺 ts_report.json"}); ok = False; continue
            d = json.loads(f.read_text(encoding="utf-8"))
            last = d["eval_history"][-1]["step"] if d.get("eval_history") else None
            h1 = bool(d.get("early_stopped")) or (last is not None and last < d["steps"])
            h3 = (d.get("v1_fingerprint") == FP_V1 and d.get("normalizer_fingerprint") == FP_NM)
            h4 = True
            if "all" in arm:
                h4 = (bool(d.get("encoder_grad_reached"))
                      and float(d["encoder_drift_from_v1"]["rel_l2_overall"]) > 0
                      and d.get("unfreeze_encoder") == "all")
            rows.append({"run": r, "arm": arm, "best_step": d["best_step"],
                         "last_eval_step": last, "steps": d["steps"],
                         "H1_收敛": h1, "H3_基座指纹": h3, "H4_解冻真发生": h4,
                         "n_resumes": d.get("n_resumes"),
                         "budget_note": (d.get("budget_note") or "")[:60],
                         "drift": (d["encoder_drift_from_v1"]["rel_l2_overall"]
                                   if "all" in arm else None)})
            ok = ok and h1 and h3 and h4
    return ok, rows


def parity_gate(path="runs/r5_parity.json"):
    p = REPO / path
    if not p.is_file():
        return None, {"error": f"缺 {path}"}
    d = json.loads(p.read_text(encoding="utf-8"))
    # 真凭据是每臂的 parity_ok / diffs / arg_diffs,不是 verdict 那句中文
    arms = d.get("arms", {})
    detail = {k: {"parity_ok": v.get("parity_ok"),
                  "diffs": v.get("diffs"), "arg_diffs": v.get("arg_diffs")}
              for k, v in arms.items()}
    allgreen = bool(arms) and all(
        v.get("parity_ok") and not v.get("diffs") and not v.get("arg_diffs")
        for v in arms.values())
    return allgreen, {"verdict_text": d.get("verdict", {}), "per_arm": detail}


def _pair(x, y, better="lower"):
    """y − x;better='lower' 时负号为好(更省),'higher' 时正号为好。"""
    d = [(q - p) * 100 for p, q in zip(x, y)]
    sem = st.stdev(d) / len(d) ** 0.5
    good = sum(1 for v in d if (v < 0 if better == "lower" else v > 0))
    return {"d_pp_by_seed": [round(v, 2) for v in d], "mean_pp": round(st.mean(d), 2),
            "median_pp": round(st.median(d), 2), "sem_pp": round(sem, 2),
            "two_sigma_pp": round(2 * sem, 2), "n_good": good, "n": len(d),
            "resolvable": bool(abs(st.mean(d)) > 2 * sem)}


def g1():
    p = REPO / "runs/r5_screen_matched.json"
    if not p.is_file():
        return {"error": "缺 runs/r5_screen_matched.json(先跑 r4_screen_matched.py --r5)"}
    d = json.loads(p.read_text(encoding="utf-8"))
    w = d["window_6200_8200"]
    out = {}
    for basis in ("prereg3", "resolvable2"):
        out[basis] = _pair(w["arms"]["A1_none"][basis], w["arms"]["A3_all"][basis], "lower")
    n, tot = out["prereg3"]["n_good"], out["prereg3"]["n"]
    same_sign = (out["prereg3"]["mean_pp"] < 0) == (out["resolvable2"]["mean_pp"] < 0)
    out["verdict"] = {
        "cond_sign": f"{n}/{tot} 同号(更省),门 >=4/5" if tot == 5 else f"{n}/{tot}(座数不足 5)",
        "cond_sign_pass": bool(n >= 4 and tot == 5),
        "cond_resolvable": out["prereg3"]["resolvable"],
        "cond_two_basis_same_sign": bool(same_sign),
        "PASS": bool(n >= 4 and tot == 5 and out["prereg3"]["resolvable"] and same_sign)}
    return out


def g2():
    p = REPO / "runs/multi_anchor_report.json"
    if not p.is_file():
        return {"error": "缺 runs/multi_anchor_report.json(先跑 multi_anchor_report.py --r5)"}
    d = json.loads(p.read_text(encoding="utf-8"))
    if not d.get("include_r5"):
        return {"error": "multi_anchor_report.json 不是 --r5 口径(座数不是 5)"}
    pair = next(x for x in d["paired"] if x["pair"].startswith("臂③all@4万"))
    row = next((r for r in pair["rows"] if str(r["anchor"]).startswith("win")), None)
    if not row or not row.get("available"):
        return {"error": f"窗口锚不可用:{row.get('reason') if row else '缺行'}"}
    m = row["by_metric"]["ieee39[外]pr_auc"]
    med = st.median(m["d_pp_by_seed"])
    out = {"steps_used": row["steps_used"], "detail": m, "median_pp": round(med, 2),
           "also_118acc": row["by_metric"]["118[内]acc"],
           "verdict": {
               "cond_sign": f"{m['n_pos']}/{m['n']} 为正,门 >=4/5",
               "cond_sign_pass": bool(m["n_pos"] >= 4 and m["n"] == 5),
               "cond_resolvable": m["resolvable_at_2sigma"],
               "cond_median_ge5pp": bool(med >= 5.0),
               "PASS": bool(m["n_pos"] >= 4 and m["n"] == 5
                            and m["resolvable_at_2sigma"] and med >= 5.0)}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/r5_gate.json")
    a = ap.parse_args()

    hok, hrows = hard_gates()
    pok, pv = parity_gate()
    r1, r2 = g1(), g2()
    promote = bool(hok and pok and r1.get("verdict", {}).get("PASS")
                   and r2.get("verdict", {}).get("PASS"))
    rep = {"note": "R5 按预注册判门;两门都过才建议推广共享编码器,且即便过门本轮也不动下游。",
           "hard_gates": {"pass": hok, "rows": hrows},
           "parity": {"pass": pok, "verdict": pv},
           "G1_主指标②": r1, "G2_ieee39": r2,
           "PROMOTE_RECOMMENDED": promote}
    (REPO / a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=== 硬门")
    for r in hrows:
        print("  " + json.dumps(r, ensure_ascii=False))
    print(f"  硬门合计:{'过' if hok else '**不过**'};parity:{'过' if pok else '**不过/缺**'}")
    for nm, r in (("G1 主指标②(越省越好)", r1), ("G2 ieee39 [外]", r2)):
        print(f"=== {nm}")
        if "error" in r:
            print("  " + r["error"]); continue
        if nm.startswith("G1"):
            for b in ("prereg3", "resolvable2"):
                x = r[b]
                print(f"  {b:12s} Δ {x['mean_pp']:+6.2f}pp 逐座 {x['d_pp_by_seed']} "
                      f"2σ {x['two_sigma_pp']:.2f} 更省 {x['n_good']}/{x['n']} "
                      f"→ {'可分辨' if x['resolvable'] else '不可分辨'}")
        else:
            x = r["detail"]
            print(f"  共有网格 {r['steps_used']};Δ {x['mean_pp']:+.2f}pp 中位 {r['median_pp']:+.2f}pp "
                  f"逐座 {x['d_pp_by_seed']} 2σ {x['two_sigma_pp']:.2f} 正 {x['n_pos']}/{x['n']}")
            print(f"  并报 118 acc:{r['also_118acc']['mean_pp']:+.2f}pp "
                  f"({'可分辨' if r['also_118acc']['resolvable_at_2sigma'] else '不可分辨'})")
        for k, v in r["verdict"].items():
            print(f"    {k}: {v}")
    print(f"\n=== 裁决:{'**建议推广**(仍须另行过账下游 PSASP)' if promote else '**不推广**,分叉维持在册'}")
    print(f"[out] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
