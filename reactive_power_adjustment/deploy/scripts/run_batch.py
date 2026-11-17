#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""批量调整 / 验收驱动(2026-08-23):对一个目录下的全部算例子目录,用 run_adjust.adjust() 同一代码路径逐例调整(模型只载一次),
汇总成功率(Wilson95)、每例 PSASP 次数、实际落地干预。PSASP 串行;逐例落盘可断点续跑。

  python3 scripts/run_batch.py --task reactive --cases-root /data/cases --out /data/out --psasp-bin /opt/psasp [run_adjust 的其他参数]
  --work-name model   : 每个病例的工作目录名固定为 model(复现 2026-08-21 包 + B 段"目录名决定随机列"的行为,对拍用;配 --sample-id legacy)
  --check-meta        : 病例目录里有 meta.json(含 vmin)时,核对初判 vmin 与之吻合(|Δ| ≤ 5e-4),不吻合计入 mismatch
产物:<out>/batch_results.json(逐例报告)、<out>/batch_summary.json、<out>/<cid>/<work-name>/(调整后的卡片 + _original_backup/adjust_report.json)。
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
import run_adjust as RA  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def _med(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    xs = sorted(xs)
    return [round(xs[len(xs) // 2], 4), round(xs[0], 4), round(xs[-1], 4)]


def summarize(results: dict, a) -> dict:
    recs = [r for r in results.values() if r.get("complete")]
    n = len(recs)
    out = {"n": n, "task": a.task, "ckpt": None, "variant": None, "eps": a.eps, "fixed_units": a.fixed_units, "sample_id": a.sample_id,
           "ladder": not a.no_ladder, "work_name": a.work_name, "n_healthy": 0, "n_mismatch_vmin": 0, "arms": {}}
    if not recs:
        return out
    rp0 = next((r["report"] for r in recs if r["report"].get("ckpt")), None)
    if rp0:
        out["ckpt"], out["variant"] = rp0.get("ckpt"), rp0.get("variant")
    out["n_healthy"] = sum(1 for r in recs if r["report"]["verdict"] == "healthy_no_action")
    out["n_mismatch_vmin"] = sum(1 for r in recs if r.get("vmin_mismatch"))

    def arm(name, okf, callsf):
        oks = [bool(okf(r["report"])) for r in recs]
        calls = [callsf(r["report"]) for r in recs]
        k = sum(oks)
        return {"n": n, "n_ok": k, "rate": round(k / n, 3), "wilson95": wilson(k, n),
                "psasp_calls_mean": round(sum(calls) / n, 2), "psasp_calls_max": max(calls)}
    healthy = lambda rp: rp["verdict"] == "healthy_no_action"  # noqa: E731
    first_ok = lambda rp: healthy(rp) or bool(rp.get("model_ok"))  # noqa: E731
    lad_ok = lambda rp: healthy(rp) or rp["verdict"] in ("model_fixed", "ladder_fixed")  # noqa: E731
    casc_ok = lambda rp: healthy(rp) or rp["verdict"] in ("model_fixed", "ladder_fixed", "rule_fixed")  # noqa: E731

    def first_calls(rp):
        return 1 if healthy(rp) else 1 + (rp.get("model_psasp_calls") or 0)

    def model_calls(rp):
        return 1 if healthy(rp) else 1 + (rp.get("model_psasp_calls") or 0) + ((rp.get("ladder") or {}).get("psasp_calls") or 0)
    out["arms"]["model_first"] = arm("model_first", first_ok, first_calls)
    out["arms"]["model"] = arm("model", lad_ok, model_calls)
    if any(r["report"].get("rule_ok") is not None for r in recs):
        out["arms"]["rule_on_fallthrough"] = {"n": sum(1 for r in recs if r["report"].get("rule_ok") is not None),
                                              "n_ok": sum(1 for r in recs if r["report"].get("rule_ok")),
                                              "psasp_calls_mean": round(sum(r["report"].get("rule_psasp_calls") or 0 for r in recs
                                                                            if r["report"].get("rule_ok") is not None)
                                                                        / max(1, sum(1 for r in recs if r["report"].get("rule_ok") is not None)), 2)}
    out["arms"]["cascade"] = arm("cascade", casc_ok, lambda rp: rp.get("total_psasp_calls") or 0)
    # 阶梯
    lads = [r["report"]["ladder"] for r in recs if r["report"].get("ladder")]
    ok_at = []
    for r in recs:
        rp = r["report"]
        if healthy(rp):
            ok_at.append(1)
        elif rp.get("model_ok"):
            ok_at.append(first_calls(rp))
        elif rp.get("ladder") and rp["ladder"].get("fixed_at_pf"):
            ok_at.append(rp["ladder"]["fixed_at_pf"])
        else:
            ok_at.append(None)
    hist = {}
    for l_ in lads:
        if l_.get("fixed_at_rung") is not None:
            k_ = "%d×%g" % tuple(l_["rungs"][l_["fixed_at_rung"]])
            hist[k_] = hist.get(k_, 0) + 1
    max_pf = max([a.ladder_max_pf] + [x for x in ok_at if x])
    out["ladder_stats"] = {"n_ladder_used": len(lads), "n_ladder_fixed": sum(1 for l_ in lads if l_.get("fixed_at_rung") is not None),
                           "fixed_at_rung_hist": hist,
                           "ok_by_pf_curve": {str(p): sum(1 for x in ok_at if x is not None and x <= p) for p in range(1, max_pf + 1)}}
    # 实际干预(成功例 / 首步成功例 / 阶梯成功例 / 规则成功例)
    key = "sum_abs_dp" if a.task == "active" else "sum_abs_dv"

    def inter(sel):
        rz = [r["report"]["realized"] for r in recs if sel(r["report"]) and r["report"].get("realized")]
        d = {"n": len(rz), "n_changed_med_min_max": _med([x.get("n_changed") for x in rz]), "sum_abs_med_min_max": _med([x.get(key) for x in rz])}
        if a.task == "active":
            d.update({"n_plausible": sum(1 for x in rz if x.get("plausible")), "n_pushed_to_pmax_med": _med([x.get("n_pushed_to_pmax") for x in rz]),
                      "slack_dist_before_med": _med([x.get("slack_dist_before") for x in rz]), "slack_dist_after_med": _med([x.get("slack_dist_after") for x in rz]),
                      "frac_dp_on_fixed_max": max([x.get("frac_dp_on_fixed") or 0.0 for x in rz] or [0.0]),
                      "n_with_fixed_units_changed": sum(1 for x in rz if (x.get("n_fixed_units_changed") or 0) > 0)})
        else:
            d.update({"n_shunt_switched_med": _med([x.get("n_shunt_switched") for x in rz])})
        return d
    out["intervention"] = {"first_ok": inter(lambda rp: rp["verdict"] == "model_fixed"),
                           "ladder_ok": inter(lambda rp: rp["verdict"] == "ladder_fixed"),
                           "rule_ok": inter(lambda rp: rp["verdict"] == "rule_fixed"),
                           "any_ok": inter(lambda rp: rp["verdict"] in ("model_fixed", "ladder_fixed", "rule_fixed"))}
    out["verdicts"] = {v: sum(1 for r in recs if r["report"]["verdict"] == v) for v in sorted({r["report"]["verdict"] for r in recs})}
    return out


def main() -> int:
    ap = RA.build_argparser()
    # --case 在批量模式下不用:改成可选
    for act in ap._actions:
        if act.dest == "case":
            act.required = False
    ap.add_argument("--cases-root", required=True, help="病例根目录(每个子目录一个 PSASP 算例)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-name", default=None, help="工作目录名(缺省 = 病例目录名;'model' = 复现 2026-08-21 包/B 段行为)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check-meta", action="store_true", help="核对病例 meta.json 的 vmin 与初判 vmin(落料对账)")
    a = ap.parse_args()
    exe_name = RA.setup_env(a.card_encoding, a.psasp_exe)
    if not (Path(a.psasp_bin) / exe_name).is_file():
        print("[error] 未找到潮流可执行文件: %s" % (Path(a.psasp_bin) / exe_name), flush=True)
        return 3
    from gridfm.solver.psasp import LicenseError, PsaspSolver
    from adjust_apply_eval import CountingSolver

    root = Path(a.cases_root)
    cases = sorted(p for p in root.iterdir() if p.is_dir() and (p / "LF.L0").is_file())
    if a.limit:
        cases = cases[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "batch_results.json"
    results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.is_file() else {}
    models = RA.load_models(a)
    solver = CountingSolver(PsaspSolver(psasp_bin=str(a.psasp_bin)))
    print("[batch] %s 病例 %d 检查点 %s(%s) eps=%g fixed=%s sample_id=%s 阶梯=%s 已有 %d" % (
        a.task, len(cases), models["ckpt_dir"], models["variant"], a.eps, a.fixed_units, a.sample_id,
        "关" if a.no_ladder else "开", len(results)), flush=True)
    t_all = time.time()
    fused = False
    try:
        for i, src in enumerate(cases):
            cid = src.name
            if cid in results and results[cid].get("complete"):
                continue
            w = out / cid / (a.work_name or cid)
            if w.exists():
                shutil.rmtree(w, ignore_errors=True)
            shutil.copytree(src, w, ignore=shutil.ignore_patterns("_original_backup"))
            t0 = time.time()
            rep = RA.adjust(w, a, models=models, solver=solver)
            rec = {"case_id": cid, "report": rep, "wall": round(time.time() - t0, 1), "complete": True}
            if a.check_meta and (src / "meta.json").is_file():
                meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
                exp = meta.get("vmin")
                if exp is not None and rep.get("init_vmin") is not None:
                    rec["vmin_expected"] = exp
                    rec["vmin_mismatch"] = abs(float(exp) - float(rep["init_vmin"])) > 5e-4
            results[cid] = rec
            res_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            lad = rep.get("ladder")
            print("[batch] %d/%d %s %s vmin %s→%s 首步%s%s 规则%s (%d 次, %.0fs)%s" % (
                i + 1, len(cases), cid, rep["verdict"], ("%.4f" % rep["init_vmin"]) if rep.get("init_vmin") is not None else "—",
                ("%.4f" % rep["final_vmin"]) if rep.get("final_vmin") is not None else "—",
                "成" if rep.get("model_ok") else "败", (" 阶梯%s" % ("第%d阶" % (lad["fixed_at_rung"] + 1) if lad.get("fixed_at_rung") is not None else "败")) if lad else "",
                {None: "—", True: "成", False: "败"}[rep.get("rule_ok")], rep["total_psasp_calls"], time.time() - t_all,
                " [vmin 对账不符]" if rec.get("vmin_mismatch") else ""), flush=True)
    except LicenseError as e:
        fused = True
        print("[batch] 加密狗失效熔断: %s" % e, flush=True)
        res_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    summ = summarize(results, a)
    summ.update({"cases_root": str(root), "wall_secs": round(time.time() - t_all, 1), "license_fused": fused})
    (out / "batch_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[batch] 汇总 n=%d 首步 %s 模型(含阶梯) %s 级联 %s | PSASP 级联均 %s | 干预 %s" % (
        summ["n"], summ["arms"].get("model_first", {}).get("rate"), summ["arms"].get("model", {}).get("rate"),
        summ["arms"].get("cascade", {}).get("rate"), summ["arms"].get("cascade", {}).get("psasp_calls_mean"),
        (summ.get("intervention") or {}).get("any_ok")), flush=True)
    return 9 if fused else 0


if __name__ == "__main__":
    raise SystemExit(main())
