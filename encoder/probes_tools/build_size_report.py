# -*- coding: utf-8 -*-
"""汇总"编码器探针按规模分组 + 真实网配对 N-1/N-2"（任务书 2026-09-06）的全部 JSON
→ size_results/size_group_summary.json + size_results/进稿表-编码器探针按规模分组.md。
行序：源网 150–270 → 400 → 700 → 1000 → PEGASE 1354 → ACTIVSg2000 → PEGASE 2869 → case3120sp。
列：①寻址 top-1（随机基线）②并联同型支路 sep_min（通道置零 max）③V MAE（技巧分）④θ MAE（技巧分）
    ⑤相对源网倍数 V/θ ⑥N-1 变化量 V/θ ⑦N-2 变化量 V/θ。
分母 = probe_legacy_by_size_src.json 的 gate3_transfer.mae_in_pu（同一次运行的源网值）。缺文件的格写"待"。"""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR, LR = HERE / "size_results", HERE / "large_results"
GROUPS = [("src", "源网 150–270"), ("400", "合成 400"), ("700", "合成 700"), ("1000", "合成 1,000")]
REAL = [("PEGASE1354", "PEGASE 1354"), ("ACTIVSg2000", "ACTIVSg2000"), ("PEGASE2869", "PEGASE 2869"), ("case3120sp", "case3120sp")]
SYN_GROUPS = ("400", "700", "1000")


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x, nd, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "待"
    if sign and round(x, nd) == 0:
        return f"{0:.{nd}f}"  # 四舍五入后为零的变化量不带正负号，避免出现 "-0.0000"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def sci(x):
    return "待" if x is None else f"{x:.2e}"


def add_deltas(p):
    if "N0" not in p:
        return
    for k in ("N1", "N2"):
        if k in p:
            p[k]["delta_v"] = p[k]["mae_v"] - p["N0"]["mae_v"]
            p[k]["delta_theta"] = p[k]["mae_theta"] - p["N0"]["mae_theta"]
            p[k]["ratio_v_vs_N0"] = p[k]["mae_v"] / p["N0"]["mae_v"]
            p[k]["ratio_theta_vs_N0"] = p[k]["mae_theta"] / p["N0"]["mae_theta"]


def syn_row(g, src_mae):
    """合成组：原始集 JSON（列①–⑤）+ 配对集三变体（列⑥⑦）。"""
    j = load(SR / f"probe_legacy_by_size_{g}.json")
    row = {"group": g, "legacy": {}, "paired": {}}
    is_src = g == "src"
    if j:
        t = j["gate3_transfer"]
        mae = t["mae_in_pu"] if is_src else t["mae_out_pu"]
        lad = t["ladder_in"] if is_src else t["ladder_out"]
        sk = t["skill_in"] if is_src else t["skill_out"]
        s2 = j["gate2_twin"]["summary"]
        row["legacy"] = {
            "json": f"size_results/probe_legacy_by_size_{g}.json", "n_in": t["n_in"], "n_out": t["n_out"],
            "families_out": j["families_out"], "probes_on": j.get("probes_on"),
            "retrieval_top1": j["gate1_retrieval"]["mean_top1"], "retrieval_min": j["gate1_retrieval"]["min"],
            "retrieval_n": j["gate1_retrieval"]["n_samples"], "random_baseline": j["gate1_retrieval"]["random_baseline"],
            "branch_sep_min": s2["branch_sep_min"], "branch_ablated_max": s2["branch_ablated_max"], "branch_seps": s2["branch_seps"],
            "twin_families": [b["family_id"] for b in j["gate2_twin"]["branch"]],
            "mae_v": mae["v"], "mae_theta": mae["theta"], "L2_v": lad["v"]["L2"], "L2_theta": lad["theta"]["L2"],
            "skill_v": sk["v"], "skill_theta": sk["theta"],
            "src_mae_v_same_run": t["mae_in_pu"]["v"], "src_mae_theta_same_run": t["mae_in_pu"]["theta"],
            "mae_field": "gate3_transfer.mae_in_pu" if is_src else "gate3_transfer.mae_out_pu",
        }
        if src_mae:
            row["legacy"]["ratio_v"] = mae["v"] / src_mae["v"]
            row["legacy"]["ratio_theta"] = mae["theta"] / src_mae["theta"]
    for k in (0, 1, 2):
        jk = load(SR / f"probe_nk_by_size_N{k}_{g}.json")
        if not jk:
            continue
        t = jk["gate3_transfer"]
        mae = t["mae_in_pu"] if is_src else t["mae_out_pu"]
        sk = t["skill_in"] if is_src else t["skill_out"]
        s2 = jk["gate2_twin"]["summary"]
        row["paired"][f"N{k}"] = {
            "json": f"size_results/probe_nk_by_size_N{k}_{g}.json", "n_in": t["n_in"], "n_out": t["n_out"],
            "n": t["n_in"] if is_src else t["n_out"],
            "mae_v": mae["v"], "mae_theta": mae["theta"], "skill_v": sk["v"], "skill_theta": sk["theta"],
            "retrieval_top1": jk["gate1_retrieval"]["mean_top1"], "branch_sep_min": s2["branch_sep_min"],
            "branch_ablated_max": s2["branch_ablated_max"],
            "mae_field": "gate3_transfer.mae_in_pu" if is_src else "gate3_transfer.mae_out_pu",
        }
    add_deltas(row["paired"])
    return row


def real_row(g, src_mae):
    """真实网：N-0 沿用 large_results/probe_large_<g>.json（倍数按本次源网分母重算）；配对三变体 size_results/probe_real_<g>_N{k}.json。"""
    j = load(LR / f"probe_large_{g}.json")
    row = {"group": g, "legacy": {}, "paired": {}}
    if j:
        rc = j["recovery"]
        row["legacy"] = {
            "json": f"large_results/probe_large_{g}.json", "n": j["n_samples"], "root": j["root"],
            "retrieval_top1": j["retrieval"]["mean_top1"], "retrieval_min": j["retrieval"]["min"],
            "retrieval_n": j["retrieval"]["n_samples"], "random_baseline": j["retrieval"]["random_baseline"],
            "n_pairs": j["parallel_pairs"]["n_pairs"], "branch_sep_min": j["parallel_pairs"]["sep_min"],
            "branch_ablated_max": j["parallel_pairs"]["ablated_max"],
            "mae_v": rc["mae_pu"]["v"], "mae_theta": rc["mae_pu"]["theta"],
            "L2_v": rc["ladder"]["v"]["L2"], "L2_theta": rc["ladder"]["theta"]["L2"],
            "skill_v": rc["skill"]["v"], "skill_theta": rc["skill"]["theta"], "mae_field": "recovery.mae_pu",
            "old_ratio_v": j["ratio_vs_source"]["v"], "old_ratio_theta": j["ratio_vs_source"]["theta"],
            "old_source_mae": j["source_mae_pu"], "sanitized_nonfinite_n": j["sanitized_nonfinite"]["n_samples"],
        }
        if src_mae:
            row["legacy"]["ratio_v"] = rc["mae_pu"]["v"] / src_mae["v"]
            row["legacy"]["ratio_theta"] = rc["mae_pu"]["theta"] / src_mae["theta"]
    for k in (0, 1, 2):
        jk = load(SR / f"probe_real_{g}_N{k}.json")
        if not jk:
            continue
        rc = jk["recovery"]
        row["paired"][f"N{k}"] = {
            "json": f"size_results/probe_real_{g}_N{k}.json", "n": jk["n_samples"], "root": jk["root"],
            "mae_v": rc["mae_pu"]["v"], "mae_theta": rc["mae_pu"]["theta"], "skill_v": rc["skill"]["v"], "skill_theta": rc["skill"]["theta"],
            "retrieval_top1": jk["retrieval"]["mean_top1"], "n_pairs": jk["parallel_pairs"]["n_pairs"],
            "branch_sep_min": jk["parallel_pairs"]["sep_min"], "branch_ablated_max": jk["parallel_pairs"]["ablated_max"],
            "sanitized_nonfinite_n": jk["sanitized_nonfinite"]["n_samples"], "mae_field": "recovery.mae_pu",
        }
    add_deltas(row["paired"])
    return row


def main():
    src = load(SR / "probe_legacy_by_size_src.json")
    src_mae = src["gate3_transfer"]["mae_in_pu"] if src else None
    rows = [syn_row(g, src_mae) for g, _ in GROUPS]
    real = {g: real_row(g, src_mae) for g, _ in REAL}
    order = [(GROUPS[i][1], rows[i]) for i in range(4)] + [(lab, real[g]) for g, lab in REAL]
    out = {"denominator_source_mae_pu": src_mae,
           "denominator_json": "size_results/probe_legacy_by_size_src.json:gate3_transfer.mae_in_pu",
           "rows": {lab: r for lab, r in order}}
    (SR / "size_group_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    # 进稿表
    L = ["| 电网组 | ① 母线寻址 top-1（随机基线） | ② 并联同型支路最小可分距离（置零后最大） | ③ V 恢复 MAE p.u.（技巧分） | ④ θ 恢复 MAE rad（技巧分） | ⑤ 相对源网倍数 V / θ | ⑥ N-1 下 V / θ MAE 变化量 | ⑦ N-2 下 V / θ MAE 变化量 |",
         "|---|---|---|---|---|---|---|---|"]
    for lab, r in order:
        lg, p = r["legacy"], r["paired"]
        if not lg:
            L.append(f"| {lab} | 待 | 待 | 待 | 待 | 待 | 待 | 待 |")
            continue
        c1 = f"{fmt(lg.get('retrieval_top1'), 4)}（{sci(lg.get('random_baseline'))}）"
        if lg.get("n_pairs", 1) == 0:
            c2 = "无并联同型对"
        else:
            c2 = f"{fmt(lg.get('branch_sep_min'), 4)}（{sci(lg.get('branch_ablated_max'))}）"
        c3 = f"{fmt(lg.get('mae_v'), 5)}（{fmt(lg.get('skill_v'), 2)}）"
        c4 = f"{fmt(lg.get('mae_theta'), 4)}（{fmt(lg.get('skill_theta'), 2)}）"
        c5 = f"{fmt(lg.get('ratio_v'), 2)} / {fmt(lg.get('ratio_theta'), 2)}"
        c6 = f"{fmt(p['N1'].get('delta_v'), 5, True)} / {fmt(p['N1'].get('delta_theta'), 4, True)}" if "N1" in p and "delta_v" in p["N1"] else "待"
        c7 = f"{fmt(p['N2'].get('delta_v'), 5, True)} / {fmt(p['N2'].get('delta_theta'), 4, True)}" if "N2" in p and "delta_v" in p["N2"] else "待"
        L.append(f"| {lab} | {c1} | {c2} | {c3} | {c4} | {c5} | {c6} | {c7} |")

    # 配对集绝对值与比值（正文/内部用）
    P = ["| 电网组 | 配对 N-0 V / θ MAE | N-1 V / θ MAE（相对 N-0 比值） | N-2 V / θ MAE（相对 N-0 比值） | 配对例数（每变体） | 寻址 N-0 / N-1 / N-2 | 并联可分 N-0 / N-1 / N-2 |",
         "|---|---|---|---|---|---|---|"]
    for lab, r in order:
        p = r["paired"]
        if "N0" not in p:
            P.append(f"| {lab} | 待 | 待 | 待 | 待 | 待 | 待 |")
            continue

        def kk(k):
            if k not in p:
                return "待"
            return (f"{fmt(p[k]['mae_v'], 5)} / {fmt(p[k]['mae_theta'], 4)}"
                    f"（{fmt(p[k].get('ratio_v_vs_N0'), 3)} / {fmt(p[k].get('ratio_theta_vs_N0'), 3)}）")

        ret = " / ".join(fmt(p[k]["retrieval_top1"], 4) if k in p else "待" for k in ("N0", "N1", "N2"))
        sep = " / ".join((fmt(p[k]["branch_sep_min"], 4) if p[k].get("branch_sep_min") is not None else "无对") if k in p else "待"
                         for k in ("N0", "N1", "N2"))
        P.append(f"| {lab} | {fmt(p['N0']['mae_v'], 5)} / {fmt(p['N0']['mae_theta'], 4)} | {kk('N1')} | {kk('N2')} | {p['N0']['n']} | {ret} | {sep} |")

    # 例数表
    C = ["| 电网组 | 原始探针集例数（列①–⑤） | 族数 | 配对集例数（每变体，N-0 = N-1 = N-2） |", "|---|---|---|---|"]
    for lab, r in order:
        lg, p = r["legacy"], r["paired"]
        if r["group"] in SYN_GROUPS:
            n0, nf = lg.get("n_out"), len(lg.get("families_out", [])) if lg else None
        elif r["group"] == "src":
            n0, nf = lg.get("n_in"), 8
        else:
            n0, nf = lg.get("n"), 1
        np_ = p["N0"]["n"] if "N0" in p else "待"
        C.append(f"| {lab} | {n0 if n0 is not None else '待'} | {nf if nf is not None else '待'} | {np_} |")

    md = ("## 进稿表（Table R1-3 用；表内不列例数）\n\n" + "\n".join(L)
          + "\n\n## 配对集绝对值与比值（正文与内部记录用）\n\n" + "\n".join(P)
          + "\n\n## 例数表（正文用）\n\n" + "\n".join(C) + "\n")
    if src_mae:
        md += (f"\n分母（源网同一次运行值）：V {src_mae['v']:.7f} p.u. / θ {src_mae['theta']:.6f} rad"
               f"（`{out['denominator_json']}`）。\n")
    else:
        md += "\n分母：待。\n"
    (SR / "进稿表-编码器探针按规模分组.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
