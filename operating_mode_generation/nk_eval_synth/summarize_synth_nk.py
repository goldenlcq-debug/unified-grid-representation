# -*- coding: utf-8 -*-
"""汇总合成集 N-1/N-2 复测（任务书 D 项）：按集合（val8 同尺寸 160 请求 / transfer6 跨尺寸 90 请求）
报成功交付数 / 请求数、PSASP 调用数与每请求调用数；并与论文原跑（f_g6_guard_s5001_blend05 = N-0）逐族核对请求 sid 是否同一批。
输出 results/synth_nk_summary.json + results/生成器-合成集N-k-汇总表.md
"""
import json, io, sys
from pathlib import Path
W = Path(__file__).resolve().parent.parent
F = Path(r"G:/统一计算大模型-迁移包/增量-20260902/外协包-F-大网方式生成-20260825")
VAL8 = ["synth210_s2007", "synth150_s2000", "synth210_s2002", "synth240_s2003", "synth270_s2004", "synth150_s2005", "synth180_s2001", "synth180_s2006"]
TR6 = ["synth400_s3000", "synth400_s3001", "synth400_s3002", "synth400_s3003", "synth700_s3004", "synth1000_s3008"]
paper = json.load(io.open(F / "runs/f_g6_guard_s5001_blend05/report.json", encoding="utf-8"))["families"]
out = {"protocol": "座1 f_diff_s5001 + f_ro_v3 + calib_neutral + deploy_head; M=8 budget=8 seed_base=1000; --balance-fix --balance-fix-blend 0.5（= 论文 f_g6_guard_s5001_blend05 同协议）; 判据 band_2s",
       "N0_source": "论文原跑 runs/f_g6_guard_s5001_blend05/report.json（N-0 沿用旧数据，不重跑）", "sets": {}, "families": {}}
for k in (1, 2):
    rp = W / f"gen_runs/nk_synth_N{k}/report.json"
    if not rp.exists():
        print(f"N{k}: 无 report"); continue
    d = json.load(io.open(rp, encoding="utf-8"))
    fams = d["families"]
    for setname, lst in (("val8_same_size_150-270", VAL8), ("transfer6_cross_size_400-1000", TR6)):
        s = out["sets"].setdefault(setname, {"n_req": 0, "N0_delivered": 0, "N0_calls": 0})
        s.setdefault(f"N{k}_delivered", 0); s.setdefault(f"N{k}_calls", 0); s.setdefault(f"N{k}_partial", False)
        for f in lst:
            key = f"{f}_N{k}"
            fr = fams.get(key, {})
            solved = fr.get("solved", {}) or {}
            trials = fr.get("trials", {}) or {}
            p0 = paper[f]
            same_req = sorted(trials.keys()) == sorted((p0.get("trials") or {}).keys()) if trials else None
            rec = out["families"].setdefault(f, {"n_req": p0["n_req"], "N0_delivered": len(p0["solved"]), "N0_calls": p0["psasp_calls"]})
            rec[f"N{k}_delivered"] = len(solved); rec[f"N{k}_calls"] = fr.get("psasp_calls"); rec[f"N{k}_n_req"] = fr.get("n_req"); rec[f"N{k}_same_requests_as_paper"] = same_req
            if k == 1:
                s["n_req"] += p0["n_req"]; s["N0_delivered"] += len(p0["solved"]); s["N0_calls"] += p0["psasp_calls"]
            s[f"N{k}_delivered"] += len(solved); s[f"N{k}_calls"] += (fr.get("psasp_calls") or 0)
            if fr.get("n_req") != p0["n_req"]: s[f"N{k}_partial"] = True
    out[f"N{k}_report_partial_flag"] = d.get("partial")
lines = ["# 生成器 · 合成集 N-0 / N-1 / N-2 汇总（2026-09-06）", "", "| 集合 | 请求数 | N-0（论文原跑） | N-1 | N-2 | PSASP 次/请求 N-0 / N-1 / N-2 |", "|---|---|---|---|---|---|"]
for name, s in out["sets"].items():
    n = s["n_req"]
    def cell(k):
        dlv = s.get(f"N{k}_delivered"); c = s.get(f"N{k}_calls")
        return (f"{dlv}/{n} = {dlv/n:.3f}" if dlv is not None and n else "—"), (f"{c/n:.2f}" if c and n else "—")
    d0, c0 = f"{s['N0_delivered']}/{n} = {s['N0_delivered']/n:.3f}", f"{s['N0_calls']/n:.2f}"
    d1, c1 = cell(1); d2, c2 = cell(2)
    lines.append(f"| {name} | {n} | {d0} | {d1} | {d2} | {c0} / {c1} / {c2} |")
lines += ["", "逐族（交付/请求；PSASP 调用）：", "", "| 族 | 请求 | N-0 | N-1 | N-2 | 调用 N-0/N-1/N-2 | 请求同批 |", "|---|---|---|---|---|---|---|"]
for f, r in out["families"].items():
    lines.append(f"| {f} | {r['n_req']} | {r['N0_delivered']} | {r.get('N1_delivered','—')} | {r.get('N2_delivered','—')} | {r['N0_calls']}/{r.get('N1_calls','—')}/{r.get('N2_calls','—')} | {r.get('N1_same_requests_as_paper')}/{r.get('N2_same_requests_as_paper')} |")
(W / "results").mkdir(exist_ok=True)
io.open(W / "results/synth_nk_summary.json", "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
io.open(W / "results/生成器-合成集N-k-汇总表.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
