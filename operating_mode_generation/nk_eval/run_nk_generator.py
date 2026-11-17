#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""运行方式生成器（F 包，座 1）· 同网架 N-0 / N-1 / N-2 底图交付率流水线（2026-09-05，长卿）。

口径 = 论文 Table 2.5 的交付协议（_f_work/run_accept.sh 与 chain_20260828.sh 原命令）：
  去噪器 runs/f_diff_s5001/denoiser.pt + 读出 runs/f_ro_v3/readout.pt（GRIDFM_READOUT_CKPT）+ 中性校准 + 部署头；
  M=8 候选、预算 8、候选种子 1000–1007；--balance-fix（总量校正）；2000 档另加 --balance-fix-blend 0.5；
  3120 主口径 band_2s 不可达（论文报逐母线自带限值补充口径，= _f_diag/supp_caliber_3120.py 的判定式）。
三个变体：N-0 = 论文原跑（用户 09-05 裁决：所有 N-0 结果延用之前的数据；本机同种子重跑只记 N0_rerun）；N-1 / N-2 = build_nk_holdout_families.py 建的族（每个请求各配一次随机开断）。
产物：gen_runs/nk_g6_<grid>_N{k}/（g6 原生 report.json + delivered/ + violations/）、results/gen_<grid>_nk.json、results/生成器-N-k-汇总表.md。

  python run_nk_generator.py --grids 3120,2000 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PKG = Path(r"E:\统一计算大模型-迁移包\增量-20260902\外协包-F-大网方式生成-20260825")
REPO = PKG / "grid-foundation"
DATA = PKG / "gridfm-data" / "families" / "holdout_f"
PSASP = r"D:\simulation\DISK1中国电力科学研究院有限公司7.95.07.10758冬滚正式版\PSASP"
PY = sys.executable
ENV = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", PYTHONUTF8="1", GRIDFM_PSASP_BIN=PSASP, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"), MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS", "4"),
           GRIDFM_READOUT_CKPT=str(PKG / "runs" / "f_ro_v3" / "readout.pt"), GRIDFM_DATA_ROOT=str(PKG / "gridfm-data"))
GRIDS = {
    "3120": dict(fam="f3120_ho", n=50, blend=None, seed=9848, criterion="own-limit (eval_quality tier3120, bvr_own<=0.1%)",
                 paper_run=PKG / "runs" / "f_g6_3120_s5001_bal", paper_supp=PKG / "_f_diag" / "supp_3120_f_g6_3120_s5001_bal.json", skip_n0=True),
    "2000": dict(fam="f2000_ho", n=60, blend=0.5, seed=9849, criterion="band_2s 0.85-1.10 (g6 主口径)",
                 paper_run=PKG / "runs" / "f_g6_2000_s5001_blend05", paper_supp=None, skip_n0=True),
}
# 用户 09-05 裁决："所有关于 N-0 的结果延用之前的数据" → N-0 列 = 论文原跑（paper_run），不再同机重跑；
# 3120 的 N-0 在裁决前已重跑完（gen_runs/nk_g6_3120_N0），只作同机复现记录（summary 的 N0_rerun），不进表。
KS = [0, 1, 2]


class Paused(Exception):
    pass


def run(cmd, log: Path, dry: bool, cwd: Path) -> int:
    if (ROOT / "PAUSE").exists():
        print("[nk-gen] 发现 PAUSE 文件，在步骤边界停止", flush=True)
        raise Paused()
    print("[nk-gen] $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(str(c) for c in cmd) + "\n"); f.flush()
        rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd), env=ENV, stdout=f, stderr=subprocess.STDOUT)
    print("[nk-gen]   exit %d, %.0fs, log %s" % (rc, time.time() - t, log.name), flush=True)
    return rc


def report_done(out: Path) -> bool:
    rp = out / "report.json"
    if not rp.exists():
        return False
    d = json.loads(rp.read_text(encoding="utf-8"))
    return not d.get("partial", True)


def do_grid(grid: str, dry: bool) -> bool:
    cfg = GRIDS[grid]
    logs = ROOT / "logs"
    fams = {0: cfg["fam"], 1: f"{cfg['fam']}_N1", 2: f"{cfg['fam']}_N2"}
    if not all((DATA / fams[k] / "manifest.csv").exists() for k in (1, 2)):
        cmd = [PY, HERE / "build_nk_holdout_families.py", "--fam", cfg["fam"], "--seed", cfg["seed"]]
        if run(cmd, logs / f"gen_{grid}_build_families.log", dry, HERE):
            return False
    for k in KS:
        out = ROOT / "gen_runs" / f"nk_g6_{grid}_N{k}"
        if k == 0 and cfg.get("skip_n0") and not report_done(out):
            print(f"[nk-gen]   {grid} N0: 沿用论文原跑 {cfg['paper_run'].name}（用户裁决），不重跑", flush=True)
            continue
        if report_done(out):
            print(f"[nk-gen]   {grid} N{k}: 已完成，跳过", flush=True)
            continue
        if (out / "report.json").exists():
            # g6 的续跑只按"族是否已在 report 里"跳过，半途崩溃的族会被当成完成 → 半成品整段作废重跑（改名留档）
            stale = out.with_name(out.name + f"_partial_{int(time.time())}")
            out.rename(stale)
            print(f"[nk-gen]   {grid} N{k}: 发现半途产物，已改名 {stale.name}，整段重跑", flush=True)
        cmd = [PY, "-u", "-X", "utf8", REPO / "scripts" / "g6_prescreen_delivery.py", "--only-extra", "--balance-fix",
               "--extra-plan", f"holdout_f:{fams[k]}:{cfg['n']}", "--denoiser", PKG / "runs" / "f_diff_s5001" / "denoiser.pt",
               "--calib-json", "runs/f_calib_neutral/calib_neutral.json", "--out", out]
        if cfg["blend"] is not None:
            cmd += ["--balance-fix-blend", str(cfg["blend"])]
        if run(cmd, logs / f"gen_{grid}_N{k}.log", dry, REPO):
            return False
    if not dry:
        summarize(grid, cfg, fams)
    return True


def own_limit_recount(run_dir: Path, fam: str):
    """3120 补充口径：与 F 包 _f_diag/supp_caliber_3120.py 同判定式（直接 import 随包 eval_quality），只改输出位置。"""
    sys.path.insert(0, str(PKG / "scripts")); sys.path.insert(0, str(REPO))
    import eval_quality as EQ  # noqa: E402
    from collections import defaultdict
    mfile, n_bus, relaxed = EQ.TIER_M["3120"]
    _b, bus, _br = EQ.read_m(PKG / "matpower_cases" / mfile)
    vmax_own, vmin_own = bus[:, 11].copy(), bus[:, 12].copy()
    rep = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    frec = rep.get("families", {}).get(fam, {}) or {}
    solved = frec.get("solved", {}) or {}
    n_req = frec.get("n_req")
    vio_root = run_dir / "violations" / fam
    cand_dirs = sorted(d for d in vio_root.iterdir() if d.is_dir()) if vio_root.is_dir() else []
    per_sid, rows = defaultdict(list), []
    for d in cand_dirs:
        sid = d.name.rsplit("_z", 1)[0]
        v = EQ.bus_voltage(d, n_bus)
        live = v > 0.01
        if not live.any():
            rows.append({"cand": d.name, "sid": sid, "converged": False}); per_sid[sid].append(False); continue
        viol = live & ((v < vmin_own - EQ.EPS) | (v > vmax_own + EQ.EPS))
        nvo = int(viol.sum()); bvr = nvo / max(int(live.sum()), 1)
        ok = bool(bvr <= EQ.BVR_RELAXED + 1e-9)
        rows.append({"cand": d.name, "sid": sid, "converged": True, "n_viol_own": nvo, "bvr_own": round(bvr, 6), "ok": ok,
                     "vmin": round(float(v[live].min()), 4), "vmax": round(float(v[live].max()), 4)})
        per_sid[sid].append(ok)
    for sid in solved:
        per_sid[sid].append(True)
    # 不收敛候选不进 violations/（g6 删其 work 目录）：候选总数按 trials 记账补齐
    n_cand_total = sum(len(t) for t in (frec.get("trials") or {}).values())
    n_nonconv = sum(1 for t in (frec.get("trials") or {}).values() for x in t if not x.get("conv"))
    conv = [r for r in rows if r.get("converged")]
    okc = [r for r in conv if r["ok"]]
    nvos = sorted(r["n_viol_own"] for r in conv)
    sids_pass = sum(1 for s in per_sid if any(per_sid[s]))
    return {"criteria": "eval_quality tier3120: own-limit ±EPS, bvr_own<=%.4f" % EQ.BVR_RELAXED, "n_req": n_req,
            "n_candidates_tried": n_cand_total, "n_nonconverged": n_nonconv, "n_candidates_in_violations": len(rows),
            "n_converged": len(conv), "cand_ok": len(okc), "cand_rate": round(len(okc) / len(conv), 4) if conv else None,
            "median_n_viol_own": (nvos[len(nvos) // 2] if nvos else None), "sids_pass": sids_pass,
            "solved_mainband": len(solved), "psasp_calls": frec.get("psasp_calls"), "rows": rows}


def summarize(grid, cfg, fams):
    res = ROOT / "results"; res.mkdir(exist_ok=True)
    out = {"grid": grid, "fam": cfg["fam"], "protocol": "座1 f_diff_s5001 + f_ro_v3 + calib_neutral + deploy_head; M=8 budget=8 seed_base=1000; --balance-fix"
           + (f" --balance-fix-blend {cfg['blend']}" if cfg["blend"] else ""), "criterion": cfg["criterion"], "seed_outage": cfg["seed"], "variants": {}}
    def variant_from_run(run_dir: Path, fam: str, grid: str, supp_json: Path = None):
        rp = run_dir / "report.json"
        v = {"family": fam, "run": str(run_dir)}
        if not rp.exists():
            return v
        d = json.loads(rp.read_text(encoding="utf-8"))
        f = d.get("families", {}).get(fam, {}) or {}
        v.update({"n_req": f.get("n_req"), "delivered_mainband": len(f.get("solved", {}) or {}), "psasp_calls": f.get("psasp_calls"), "partial": d.get("partial")})
        if grid == "3120":
            if supp_json is not None and supp_json.exists():
                s = json.loads(supp_json.read_text(encoding="utf-8"))
                v["own_limit"] = {k2: s.get(k2) for k2 in ("criteria", "n_req", "n_candidates", "n_converged", "cand_ok", "cand_rate", "median_n_viol_own", "sids_pass", "solved_mainband")}
                v["own_limit"]["source"] = str(supp_json)
            else:
                v["own_limit"] = own_limit_recount(run_dir, fam)
            v["delivered"] = v["own_limit"]["sids_pass"]
        else:
            v["delivered"] = v["delivered_mainband"]
        n_req = v.get("n_req") or 0
        v["rate"] = round(v["delivered"] / n_req, 4) if n_req else None
        v["calls_per_delivered"] = round(v["psasp_calls"] / v["delivered"], 2) if v.get("delivered") else None
        return v
    out["N0_policy"] = "N-0 = 论文原跑（用户 09-05 裁决：所有 N-0 结果延用之前的数据）；N0_rerun = 本机同种子重跑（仅复现记录，不进表）"
    for k in KS:
        run_dir = ROOT / "gen_runs" / f"nk_g6_{grid}_N{k}"
        if k == 0:
            v = variant_from_run(cfg["paper_run"], cfg["fam"], grid, cfg.get("paper_supp"))
            v["source"] = "paper_run"
            if (run_dir / "report.json").exists():
                out["N0_rerun"] = variant_from_run(run_dir, cfg["fam"], grid)
            out["variants"]["N0"] = v
            continue
        rp = run_dir / "report.json"
        v = {"family": fams[k]}
        if rp.exists():
            d = json.loads(rp.read_text(encoding="utf-8"))
            f = d.get("families", {}).get(fams[k], {}) or {}
            v.update({"n_req": f.get("n_req"), "delivered_mainband": len(f.get("solved", {}) or {}), "psasp_calls": f.get("psasp_calls"),
                      "partial": d.get("partial")})
            if grid == "3120":
                v["own_limit"] = own_limit_recount(run_dir, fams[k])
                v["delivered"] = v["own_limit"]["sids_pass"]
            else:
                v["delivered"] = v["delivered_mainband"]
            n_req = v.get("n_req") or 0
            v["rate"] = round(v["delivered"] / n_req, 4) if n_req else None
            v["calls_per_delivered"] = round(v["psasp_calls"] / v["delivered"], 2) if v.get("delivered") else None
        out["variants"][f"N{k}"] = v
    (res / f"gen_{grid}_nk.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    write_table()


def write_table():
    res = ROOT / "results"
    names = {"3120": "case3120sp（逐母线自带限值判据）", "2000": "ACTIVSg2000（0.85–1.10 带）"}
    rows = []
    for g in ("3120", "2000"):
        p = res / f"gen_{g}_nk.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        cells = []
        for k in KS:
            v = d["variants"].get(f"N{k}", {})
            if v.get("rate") is None:
                cells.append("（未跑）"); continue
            cells.append("%d/%d = %.3f；%.2f 次/交付" % (v["delivered"], v["n_req"], v["rate"], v["calls_per_delivered"] or 0))
        rows.append("| %s | %s | %s | %s |" % (names[g], *cells))
    md = ["# 运行方式生成器（F 包座 1，冻结）· 同网架 N-0 / N-1 / N-2 底图 · PSASP 校核可行交付率", "",
          "> 自动汇总（run_nk_generator.py）；N-0 = 论文原跑（用户 09-05 裁决，不重跑；3120 的本机同种子复现记在 gen_3120_nk.json 的 N0_rerun）；N-1/N-2 = 同一批请求、每个请求各配一次随机开断的底图；请求集、候选种子、协议与论文 Table 2.4（v0.7.13 前编号 2.5）相同。",
          "", "| 网 | N-0 | N-1 | N-2 |", "|---|---|---|---|"] + rows
    (res / "生成器-N-k-汇总表.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="3120,2000")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cuda-blocking", action="store_true", help="CUDA_LAUNCH_BLOCKING=1 同步启动内核（本机异步模式下 g6 采样偶发 CUDA unknown error）")
    a = ap.parse_args()
    if a.cuda_blocking:
        ENV["CUDA_LAUNCH_BLOCKING"] = "1"
    t0 = time.time()
    for g in a.grids.split(","):
        g = g.strip()
        try:
            ok = do_grid(g, a.dry_run)
        except Paused:
            print("[nk-gen] 已暂停", flush=True); return 7
        print(f"[nk-gen] ===== {g} {'完成' if ok else '失败，停止'} ({(time.time() - t0) / 60:.1f} min) =====", flush=True)
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
