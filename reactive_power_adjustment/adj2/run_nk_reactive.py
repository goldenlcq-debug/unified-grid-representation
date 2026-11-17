#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""无功调整推理器（v6d，冻结）· 同网架 N-0 / N-1 / N-2 配对考卷流水线驱动（2026-09-05，长卿）。

每张网四步（全部幂等，产物在就跳过；PSASP 串行）：
  ① produce：produce_modes_nk.py 同一抽样产 N-0/N-1/N-2 配对源方式（新种子 9840 起，id 从 3000 起），--save-all-cards 落盘 PSASP 卡；
  ② exam   ：viol_evalgen_nk.py 三变体同扰动造病（继承该网现有考卷的电压带与造病配方），三变体都判病才入卷；
  ③ diag   ：adj3_scale_diag.py 算指针偏离量 Δ（模型前 1 台跳数中位 − 规则前 1 台跳数中位）；Δ ≥ 8 才开预筛，
              半径由 hop_census.py 普查定档（0 台回退 ≤5% 且候选中位 ≥8 的最小 H）——每个变体各自重算；
  ④ eval   ：adj2_eval.py 部署配置（v6d，扩档阶梯 1,2,4,8,16,32 × 0.02,0.05、≤16 次；预筛按 ③），PSASP 唯一裁判；
  ⑤ summary：results/<tier>_nk.json + results/无功推理器-N-k-汇总表.md（一网一行，N-0/N-1/N-2 三列并排）。

  python run_nk_reactive.py --tiers 118,300 [--steps produce,exam,diag,eval,summary] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INC = ROOT.parent
LINE = INC / "大算例样本线-20260820"
LIBS = LINE / "libs"
MPC = INC / "MATPOWER转换-20260818" / "cases"
PKG = INC / "内网部署包-Linux-20260827"
CKPT = PKG / "runs" / "reactive_reasoner_v6d_s0"
PSASP = r"D:\simulation\DISK1中国电力科学研究院有限公司7.95.07.10758冬滚正式版\PSASP"
PY = sys.executable
ENV = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", GRIDFM_PSASP_BIN=PSASP, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"), MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS", "4"),
           GRIDFM_POSENC_DISK_CACHE=str(ROOT / "_posenc_cache"))   # 位置编码磁盘缓存：诊断与跑分共享，大网省时
LADDER = ["--ladder", "--ladder-ks", "1,2,4,8,16,32", "--ladder-mags", "0.02,0.05", "--ladder-max-pf", "16"]
DELTA_OPEN = 8          # 指针偏离量 Δ ≥ 8 才开局部性预筛（2026-08-27 十网 316 例普查定的判据）
DEVICE = os.environ.get("NK_DEVICE", "cpu")   # 2026-09-05：无功线诊断/跑分改 CPU（LaPE 与 PSASP 才是瓶颈），GPU 留给生成器采样

# 造病配方与电压带 = 该网现有考卷（论文 Table 2.3 所用）的 eval_summary.json；种子 9840 起（用户裁决）
TIERS = {
    "118":  dict(fid="mp_case118", base=LIBS / "mp_case118" / "_base_case", n_bus=118, ref="viol_q_118", name="viol_q_118",
                 q_mode="global", f=(2.0, 3.0), hops=2, seed_modes=9840, seed_exam=9850, draws=60, keep_p1d=False, timeout=300, max_draws=400),
    "300":  dict(fid="mp_case300", base=LIBS / "mp_case300" / "_base_case", n_bus=300, ref="viol_q_300_local", name="viol_q_300_local",
                 q_mode="local", f=(3.0, 6.0), hops=2, seed_modes=9841, seed_exam=9851, draws=60, keep_p1d=False, timeout=300, max_draws=400),
    "1354": dict(fid="mp_case1354pegase", base=MPC / "case1354pegase_v02", n_bus=1354, ref="viol_q_1354_local", name="viol_q_1354_local",
                 q_mode="local", f=(3.0, 6.0), hops=2, seed_modes=9842, seed_exam=9852, draws=60, keep_p1d=False, timeout=300, max_draws=600),
    "2869": dict(fid="mp_case2869pegase", base=MPC / "case2869pegase_v02", n_bus=2869, ref="viol_q_2869_local", name="viol_q_2869_local",
                 q_mode="local", f=(3.0, 6.0), hops=2, seed_modes=9843, seed_exam=9853, draws=60, keep_p1d=False, timeout=300, max_draws=600),
    "3120": dict(fid="mp_case3120sp", base=LIBS / "mp_case3120sp_fix" / "_base_case", n_bus=3120, ref="viol_q_3120", name="viol_q_3120",
                 q_mode="global", f=(2.0, 3.0), hops=2, seed_modes=9844, seed_exam=9854, draws=60, keep_p1d=False, timeout=300, max_draws=400),
    "2000": dict(fid="mp_activsg2000", base=LIBS / "mp_activsg2000" / "_base_case", n_bus=2000, ref="viol_q_2000_local", name="viol_q_2000_local",
                 q_mode="local", f=(3.0, 6.0), hops=2, seed_modes=9845, seed_exam=9855, draws=60, keep_p1d=False, timeout=300, max_draws=400),
    "25k":  dict(fid="mp_activsg25k", base=LIBS / "mp_activsg25k" / "_base_case", n_bus=25000, ref="viol_q_25k_local_deep_b2", name="viol_q_25k_local_deep",
                 q_mode="local", f=(5.0, 10.0), hops=2, seed_modes=9846, seed_exam=9856, draws=40, keep_p1d=True, timeout=600, max_draws=600, tries=20),
    "70k":  dict(fid="mp_activsg70k", base=LIBS / "mp_activsg70k" / "_base_case", n_bus=70000, ref="viol_q_70k_local_deep", name="viol_q_70k_local_deep",
                 q_mode="local", f=(5.0, 10.0), hops=2, seed_modes=9847, seed_exam=9857, draws=40, keep_p1d=True, timeout=900, max_draws=400, tries=20),
}
KS = [0, 1, 2]


class Paused(Exception):
    pass


CURRENT_TIER = [""]


def run(cmd, log: Path, dry: bool) -> int:
    if (ROOT / "PAUSE").exists() or (ROOT / f"PAUSE_{CURRENT_TIER[0]}").exists():   # 全局 PAUSE 或本网 PAUSE_<tier>：步骤边界停
        print("[nk-run] 发现 PAUSE 文件，在步骤边界停止（已完成的步骤不受影响）", flush=True)
        raise Paused()
    print("[nk-run] $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        rc = subprocess.call([str(c) for c in cmd], cwd=str(HERE), env=ENV, stdout=f, stderr=subprocess.STDOUT)
    print("[nk-run]   exit %d, %.0fs, log %s" % (rc, time.time() - t, log.name), flush=True)
    return rc


def band_of(ref_name: str):
    return [float(x) for x in json.loads((LIBS / ref_name / "eval_summary.json").read_text(encoding="utf-8"))["vio_band"]]


def delta_of(diag_json: Path):
    d = json.loads(diag_json.read_text(encoding="utf-8"))
    cs = d.get("cases") or []
    m = [c["model_top1_hop_median"] for c in cs if c.get("model_top1_hop_median") is not None]
    r = [c["rule_top1_hop_median"] for c in cs if c.get("rule_top1_hop_median") is not None]
    if not m or not r:
        return None, None, None, len(cs)
    return int(st.median(m)) - int(st.median(r)), int(st.median(m)), int(st.median(r)), len(cs)


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    dnm = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dnm
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dnm
    return [round(c - h, 3), round(c + h, 3)]


def do_tier(tier: str, steps, dry: bool, ks=None) -> bool:
    ks = list(KS) if ks is None else ks
    CURRENT_TIER[0] = tier
    cfg = TIERS[tier]
    raw = ROOT / "nk_raw" / cfg["fid"]
    pack = ROOT / "nk_pack" / cfg["fid"]
    logs = ROOT / "logs"
    low, high = band_of(cfg["ref"])
    print(f"[nk-run] ===== {tier} ({cfg['fid']}) 带=[{low},{high}] 配方={cfg['q_mode']}×{cfg['f']} =====", flush=True)
    if "produce" in steps and not (raw / "batch_report.json").exists():
        cmd = [PY, HERE / "produce_modes_nk.py", "--fid", cfg["fid"], "--base", cfg["base"], "--out", raw, "--work", ROOT / "_work" / cfg["fid"],
               "--seed", cfg["seed_modes"], "--id-start", 3000, "--n", cfg["draws"], "--amp", 0.5, "--timeout", cfg["timeout"], "--save-all-cards", pack]
        if cfg["keep_p1d"]:
            cmd.append("--keep-p1d")
        if run(cmd, logs / f"{tier}_produce.log", dry):
            return False
    exam = {k: ROOT / "libs_nk" / f"{cfg['name']}_N{k}" for k in KS}
    if "exam" in steps and not (exam[0] / "eval_summary.json").exists():
        cmd = [PY, HERE / "viol_evalgen_nk.py", "--pack", pack, "--name", cfg["name"], "--out-root", ROOT / "libs_nk", "--tier", tier,
               "--band-from", LIBS / cfg["ref"] / "eval_summary.json", "--q-mode", cfg["q_mode"], "--f-lo", cfg["f"][0], "--f-hi", cfg["f"][1],
               "--local-hops", cfg["hops"], "--seed", cfg["seed_exam"], "--n-target", 30, "--tries-per-mode", cfg.get("tries", 6), "--max-draws", cfg["max_draws"],
               "--max-depth", 0.08, "--keep-unpaired"]
        if run(cmd, logs / f"{tier}_exam.log", dry):
            return False
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    decision = {}
    if True:                                   # 判据读取对所有步骤都做（只重算汇总时也要读回 H）；诊断子进程只在 diag 步且缺 JSON 时启动
        for k in KS:
            dj = res / f"scale_diag_{tier}_N{k}.json"
            if not dj.exists() and "diag" not in steps:
                decision[k] = dict(delta=None, H=0, note="no diag json")
                continue
            if "diag" in steps and not dj.exists() and k not in ks:
                decision[k] = dict(delta=None, H=0, note="variant not in scope; diag json absent")
                continue
            if "diag" in steps and not dj.exists():
                cmd = [PY, HERE / "adj3_scale_diag.py", "--lib", exam[k], "--ckpt", CKPT, "--n-bus", cfg["n_bus"], "--low", low, "--high", high,
                       "--device", DEVICE, "--out", dj]
                if run(cmd, logs / f"{tier}_diag_N{k}.log", dry):
                    return False
            if dry and not dj.exists():
                decision[k] = dict(delta=None, H=0, note="dry-run")
                continue
            delta, mh, rh, n = delta_of(dj)
            H, census = 0, None
            if delta is not None and delta >= DELTA_OPEN:
                cj = res / f"hop_census_{tier}_N{k}.json"
                if not cj.exists():
                    cmd = [PY, HERE / "hop_census.py", "--lib", exam[k], "--n-bus", cfg["n_bus"], "--low", low, "--high", high, "--out", cj]
                    if run(cmd, logs / f"{tier}_census_N{k}.log", dry):
                        return False
                if cj.exists():
                    census = json.loads(cj.read_text(encoding="utf-8"))
                    H = int(census.get("recommended_h") or 0)
            decision[k] = dict(delta=delta, model_top1_hop=mh, rule_top1_hop=rh, n_diag_cases=n, H=H,
                               census_fallback=(census or {}).get("fallback"), rule=f"Δ≥{DELTA_OPEN} 开预筛；H=普查推荐")
            print(f"[nk-run]   N{k}: Δ={delta} (模型 {mh} / 规则 {rh} 跳, {n} 例) → 预筛 {'开 H=%d' % H if H else '关'}", flush=True)
    if "eval" in steps:
        # 主口径：网级部署配置 = N-0 考卷上的判据结果（与论文 Table 2.3 各行"按网定档"同一逻辑），三个变体同一配置，只比拓扑；
        # 敏感性：若某变体按自身考卷重算的判据给出不同 H（3120 实测 N-1/N-2 的"规则最近机跳数"从 12 掉到 3 → Δ 从 0 跳到 9），
        #         另跑一组该变体自身判据的配置留档，表格只报主口径。
        H_main = decision[0]["H"]
        if "H" not in decision.get(0, {}) or decision[0].get("note"):
            print(f"[nk-run]   警告：N0 判据缺失（{decision.get(0)}），H_main 按 0 处理", flush=True)
        for k in ks:
            for H, tag in ((H_main, "main"), (decision[k]["H"], "variant_rule")):
                if tag == "variant_rule" and H == H_main:
                    continue
                out_name = f"nk_{tier}_N{k}_h{H}"
                sj = ROOT / "runs" / f"eval_{out_name}" / "summary.json"
                n_cases = len(list((exam[k] / "cases").iterdir())) if (exam[k] / "cases").exists() else 0
                if sj.exists() and json.loads(sj.read_text(encoding="utf-8")).get("n_cases") == n_cases:
                    print(f"[nk-run]   N{k}[{tag}]: 已有 {sj.parent.name}（{n_cases} 例），跳过", flush=True)
                    continue
                cmd = [PY, HERE / "adj2_eval.py", "--task", "reactive", "--tier", tier, "--lib-name", exam[k], "--ckpt", CKPT, "--out-name", out_name,
                       "--ptr-local-hops", H, "--device", DEVICE, "--no-reuse-rule"] + LADDER
                if run(cmd, logs / f"{tier}_eval_N{k}_h{H}.log", dry):
                    return False
    if "summary" in steps and not dry:
        summarize_tier(tier, cfg, exam, decision, low, high)
    return True


def summarize_tier(tier, cfg, exam, decision, low, high):
    res = ROOT / "results"
    out = {"tier": tier, "fid": cfg["fid"], "vio_band": [low, high], "recipe": {"q_mode": cfg["q_mode"], "f": cfg["f"], "hops": cfg["hops"]},
           "seeds": {"modes": cfg["seed_modes"], "exam": cfg["seed_exam"]}, "ckpt": str(CKPT), "config": "v6d 扩档阶梯 1,2,4,8,16,32×0.02,0.05 ≤16 次；预筛按 Δ 判据",
           "variants": {}}
    pr = ROOT / "libs_nk" / f"{cfg['name']}_pairs.json"
    if pr.exists():
        out["pairs"] = json.loads(pr.read_text(encoding="utf-8"))
    def read_eval(sj: Path):
        if not sj.exists():
            return None
        s = json.loads(sj.read_text(encoding="utf-8"))
        m = s["arms"].get("model", {})
        f = s["arms"].get("model_first", {})
        return {"n": s["n_cases"], "n_ok": m.get("n_ok"), "rate": m.get("rate"), "wilson95": m.get("wilson95"),
                "first_rate": f.get("rate"), "first_n_ok": f.get("n_ok"), "psasp_calls_mean": m.get("psasp_calls_mean"),
                "ok_n_changed_med_min_max": s["intervention_model"].get("ok_n_changed_med_min_max"),
                "ok_sum_abs_dv_med_min_max": s["intervention_model"].get("ok_sum_abs_med_min_max"),
                "healthy_no_action": s.get("healthy_no_action"), "run": sj.parent.name}
    H_main = decision.get(0, {}).get("H", 0)
    out["H_main"] = H_main
    out["H_rule"] = "主口径 = N-0 考卷判据定的网级配置（三变体同配置）；H_variant_rule = 各变体自身考卷重算判据（不同则另跑一组留档）"
    for k in KS:
        H_var = decision.get(k, {}).get("H", 0)
        es = exam[k] / "eval_summary.json"
        v = {"H": H_main, "H_variant_rule": H_var, "decision": decision.get(k)}
        if es.exists():
            e = json.loads(es.read_text(encoding="utf-8"))
            v["exam"] = {kk: e.get(kk) for kk in ("n_sick", "n_draws", "n_paired_sick", "outcomes_this_variant", "sick_rate_this_variant", "seed")}
        v["eval"] = read_eval(ROOT / "runs" / f"eval_nk_{tier}_N{k}_h{H_main}" / "summary.json")
        if H_var != H_main:
            v["eval_variant_rule"] = read_eval(ROOT / "runs" / f"eval_nk_{tier}_N{k}_h{H_var}" / "summary.json")
        out["variants"][f"N{k}"] = v
    (res / f"{tier}_nk.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    write_table()


def write_table():
    res = ROOT / "results"
    rows = []
    order = ["118", "300", "1354", "2869", "3120", "2000", "25k", "70k"]
    names = {"118": "IEEE118（转换版）", "300": "IEEE300", "1354": "PEGASE1354", "2869": "PEGASE2869", "3120": "case3120sp", "2000": "ACTIVSg2000",
             "25k": "ACTIVSg25k（深局部）", "70k": "ACTIVSg70k（深局部）"}
    for t in order:
        p = res / f"{t}_nk.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        cells = []
        for k in KS:
            v = d["variants"].get(f"N{k}", {})
            e = v.get("eval")
            if not e:
                cells.append("（未跑）")
                continue
            w = e.get("wilson95") or [None, None]
            cell = "%d/%d = %.2f (%.2f–%.2f)；首步 %.2f；%.1f 次/例；H=%s" % (
                e["n_ok"], e["n"], e["rate"], w[0], w[1], e["first_rate"] or 0, e["psasp_calls_mean"], v.get("H") or "关")
            ev = v.get("eval_variant_rule")
            if ev:
                cell += "（按本变体判据 H=%s：%d/%d = %.2f）" % (v.get("H_variant_rule") or "关", ev["n_ok"], ev["n"], ev["rate"])
            cells.append(cell)
        n_pair = d.get("pairs", {}).get("n_paired_sick")
        rows.append("| %s | [%.2f, %.2f] | %s | %s | %s | %s |" % (names.get(t, t), d["vio_band"][0], d["vio_band"][1],
                                                            ("%d" % n_pair) if n_pair is not None else "—", cells[0], cells[1], cells[2]))
    md = ["# 无功调整推理器（v6d 冻结）· 同网架 N-0 / N-1 / N-2 配对考卷 · 成功率（部署配置，PSASP 裁判）",
          "", "> 自动汇总（run_nk_reactive.py）；每格 = 修好数/例数 = 成功率 (Wilson 95%)；首步 = 模型第一个动作单独成功率；次/例 = 每例 PSASP 次数（含初判）；H = 局部性预筛半径（关 = Δ<8 不开）。",
          "> 三列为同一批抽样、同一扰动、只改拓扑的配对考卷；三列只比变化量，N-0 绝对值与论文 Table 2.3 不同源（新种子、新考卷）。",
          "> 预筛配置按网定：以 N-0 考卷上的判据（Δ≥8 才开、普查定 H）作为该网三列的统一配置；若某变体按自身考卷重算得到不同 H，括号内给出该配置的成绩作敏感性参考。", "",
          "| 网 | 考卷带 | 配对例数 | N-0 | N-1 | N-2 |", "|---|---|---|---|---|---|"] + rows
    (res / "无功推理器-N-k-汇总表.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="118,300")
    ap.add_argument("--steps", default="produce,exam,diag,eval,summary")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--variants", default="0,1,2", help="只对这些变体做 diag/eval（并行分工用；summary 始终看全部三个）")
    a = ap.parse_args()
    steps = set(a.steps.split(","))
    ks = [int(x) for x in a.variants.split(",")]
    t0 = time.time()
    for tier in a.tiers.split(","):
        tier = tier.strip()
        if tier not in TIERS:
            print(f"[nk-run] 未知档 {tier}", flush=True)
            return 2
        try:
            ok = do_tier(tier, steps, a.dry_run, ks)
        except Paused:
            print("[nk-run] 已暂停；删除 PAUSE 文件后重跑同一命令即可续跑", flush=True)
            return 7
        print(f"[nk-run] ===== {tier} {'完成' if ok else '失败，停止'} ({(time.time() - t0) / 60:.1f} min) =====", flush=True)
        if not ok:
            return 1
    print("[nk-run] all done %.1f min" % ((time.time() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
