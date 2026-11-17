#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""配对式无功考卷生成：N-0 / N-1 / N-2 三个变体**同一扰动、只改拓扑**（2026-09-05，N-k 其他推理器线，长卿）。

输入 = produce_modes_nk.py --save-all-cards 落盘的"源方式包"：<pack>/N{0,1,2}/<sid>/（三个变体都收敛的抽样，PSASP 已解）。
做法（沿 viol_evalgen_mp.py 的无功造病机制，逐字段同口径）：
  1. 只取三个变体都在考卷带内（双边健康）的源方式；N-0 健康而 N-1/N-2 已越限的抽样单列为"拓扑致病"，不入本卷、只记账；
  2. 对每个源方式每次抽样：**一次抽出全部无功负荷放大系数**（global：逐负荷 U(f_lo,f_hi)；local：锚点 ≤hops 跳内负荷 U(f_lo,f_hi)，
     锚点与区域按 N-0 拓扑取），把**完全相同的系数**写进三个变体的 LF.L6（三变体的负荷卡逐条相同，只有 LF.L2/L3 的 mark 不同）；
  3. 三个变体各送 PSASP 一次；判"病"= 收敛 且 越限（vmin<low−eps 或 vmax>high+eps）且 不过深（≤ max_depth）；
  4. **三个变体都判病的抽样才入卷**（配对；case_id 三卷相同 = <sid>__qNN），其余记 attempts.csv；--keep-unpaired 时把"部分变体判病"的也留档到 _unpaired；
  5. 考卷带不重算，直接继承该网现有考卷 eval_summary.json 的 vio_band（保证三列与论文同带）。
产物：<out-root>/<name>_N{k}/{cases/, eval_cases.jsonl, eval_summary.json}（与 viol_evalgen_mp 同布局，adj2_eval / hop_census / adj3_scale_diag 直接消费）
      + <out-root>/<name>_pairs.json（配对记账）+ attempts.csv。

  python viol_evalgen_nk.py --pack ../nk_pack/mp_case118 --name viol_q_118 --out-root ../libs_nk --tier 118 \
      --band-from <libs>/viol_q_118/eval_summary.json --q-mode global --f-lo 2.0 --f-hi 3.0 --seed 9850 --n-target 30
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("GRIDFM_PSASP_BIN",
                      r"D:\simulation\DISK1中国电力科学研究院有限公司7.95.07.10758冬滚正式版\PSASP")
import viol_evalgen_mp as VG  # noqa: E402  打过路径补丁的副本：EPS / INPUT_CARDS / sha256_files / read_loads / bus_adjacency / hops_from

from gridfm.data.repair_perturb import _clear_stale_pf_outputs  # noqa: E402
from gridfm.data.synth import psasp_referee  # noqa: E402
from gridfm.fsutil import robust_copytree  # noqa: E402
from gridfm.psasp_io import cards  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402
from gridfm.solver.psasp import LicenseError, PsaspSolver  # noqa: E402
from psasp_guard import guarded  # noqa: E402  2026-09-05：加密狗不响应守卫

EPS = VG.EPS


def read_v(d: Path):
    """从已解算目录的 LF.LP1 读母线电压（0 基数组，母线号-1；失电/溢出为 0）。"""
    p = d / "LF.LP1"
    if not p.is_file():
        return None
    rows = list(cards.parse_lp1(Card.load(p)))
    if not rows:
        return None
    n = max(r.bus_no for r in rows)
    v = np.zeros(n)
    for r in rows:
        if np.isfinite(r.v_pu) and r.v_pu < 100:
            v[r.bus_no - 1] = r.v_pu
    return v


def vminmax(v):
    vv = v[v > 0.01]
    return (float(vv.min()), float(vv.max())) if vv.size else (float("nan"), float("nan"))


def classify(rf, low, high, max_depth):
    if not rf.get("converged"):
        return "diverged", None, None
    vmin, vmax = float(rf["vmin"]), float(rf["vmax"])
    if not (vmin < low - EPS or vmax > high + EPS):
        return "healthy", vmin, vmax
    if vmin < low - max_depth or vmax > high + max_depth:
        return "too_deep", vmin, vmax
    return "sick", vmin, vmax


def draw_factors(loads, rng, mode, f_lo, f_hi, adj=None, hops=2):
    """一次抽出全部放大系数（全精度），返回 (factors{记录号:系数}, 摘要)。与 VG.q_global / q_local 同分布同候选规则。"""
    if mode == "global":
        fac = {i: float(rng.uniform(f_lo, f_hi)) for i in loads}
        return fac, {"mode": "global", "n_loads_scaled": len(fac),
                     "mean_factor": round(float(np.mean(list(fac.values()))), 4)}
    cand = sorted({ld["bus"] for ld in loads.values() if abs(ld["ql"]) > 1e-6 or abs(ld["pl"]) > 1e-6})
    anchor = int(cand[int(rng.integers(0, len(cand)))])
    region = set(VG.hops_from(adj, [anchor], hops).keys())
    fac = {}
    for i, ld in loads.items():
        if ld["bus"] in region:
            fac[i] = float(rng.uniform(f_lo, f_hi))
    return fac, {"mode": "local", "anchor_bus": anchor, "hops": hops, "region_n_bus": len(region),
                 "n_loads_scaled": len(fac),
                 "mean_factor": (round(float(np.mean(list(fac.values()))), 4) if fac else None)}


def apply_factors(work: Path, fac: dict):
    l6 = Card.load(work / "LF.L6")
    loads = VG.read_loads(l6)
    for i, f in fac.items():
        loads[i]["view"].ql = loads[i]["ql"] * f
    l6.save(work / "LF.L6")
    return loads


def load_signature(loads):
    return tuple((i, ld["bus"], round(ld["pl"], 6), round(ld["ql"], 6)) for i, ld in sorted(loads.items()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="produce_modes_nk --save-all-cards 的目录（含 N0/N1/N2）")
    ap.add_argument("--name", required=True, help="考卷名前缀，如 viol_q_118 → viol_q_118_N0/N1/N2")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--band-from", required=True, help="现有考卷的 eval_summary.json（继承 vio_band）")
    ap.add_argument("--variants", default="0,1,2")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-target", type=int, default=30, help="配对病例目标数")
    ap.add_argument("--tries-per-mode", type=int, default=6)
    ap.add_argument("--max-draws", type=int, default=400, help="抽样次数上限（每次抽样 = 三个变体各 1 次 PSASP）")
    ap.add_argument("--q-mode", default="global", choices=["global", "local"])
    ap.add_argument("--f-lo", type=float, required=True)
    ap.add_argument("--f-hi", type=float, required=True)
    ap.add_argument("--local-hops", type=int, default=2)
    ap.add_argument("--max-depth", type=float, default=0.08)
    ap.add_argument("--keep-unpaired", action="store_true")
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    ks = [int(x) for x in a.variants.split(",")]
    pack, out_root = Path(a.pack), Path(a.out_root)
    ref = json.loads(Path(a.band_from).read_text(encoding="utf-8"))
    low, high = [float(x) for x in ref["vio_band"]]
    outs = {k: out_root / f"{a.name}_N{k}" for k in ks}
    for k, o in outs.items():
        if (o / "eval_summary.json").exists():
            print(f"[viol-nk] {o} 已有 eval_summary.json，拒绝覆盖（新产物新目录纪律）", flush=True)
            return 3
        (o / "cases").mkdir(parents=True, exist_ok=True)
    unp = out_root / f"{a.name}_unpaired"
    work_root = Path(a.work) if a.work else (out_root.parent / "_work" / a.name)
    work_root.mkdir(parents=True, exist_ok=True)

    # ---- 源方式：三变体都在 & 三变体都健康 ----
    sids = None
    for k in ks:
        s = {p.name for p in (pack / f"N{k}").iterdir() if p.is_dir()}
        sids = s if sids is None else sids & s
    sids = sorted(sids or [])
    src_info, healthy, topo_induced, other = {}, [], [], []
    for sid in sids:
        info, ok = {}, True
        for k in ks:
            d = pack / f"N{k}" / sid
            v = read_v(d)
            if v is None:
                info[k] = {"vmin": None, "vmax": None, "healthy": False}
                ok = False
                continue
            vmin, vmax = vminmax(v)
            h = bool(vmin >= low - EPS and vmax <= high + EPS)
            meta = {}
            if (d / "nk_meta.json").is_file():
                meta = json.loads((d / "nk_meta.json").read_text(encoding="utf-8"))
            info[k] = {"vmin": round(vmin, 4), "vmax": round(vmax, 4), "healthy": h,
                       "n1_branch": meta.get("n1_branch", ""), "load_factor": meta.get("load_factor")}
            ok = ok and h
        src_info[sid] = info
        if ok:
            healthy.append(sid)
        elif info[ks[0]]["healthy"]:
            topo_induced.append(sid)
        else:
            other.append(sid)
    rng = np.random.default_rng(a.seed)
    pool = [healthy[int(i)] for i in rng.permutation(len(healthy))]
    print(f"[viol-nk] {a.name} tier={a.tier} band=[{low:.2f},{high:.2f}] 源方式 三变体齐全 {len(sids)} / 三变体都健康 {len(healthy)} / "
          f"拓扑致病(N-0 健康、开断后越限) {len(topo_induced)} / 其他 {len(other)}；seed={a.seed} mode={a.q_mode} f=[{a.f_lo},{a.f_hi}]", flush=True)

    solver = PsaspSolver()
    rows = {k: [] for k in ks}
    attempts = []
    counts = {k: {"sick": 0, "healthy": 0, "diverged": 0, "too_deep": 0} for k in ks}
    n_pair = n_draw = n_partial = 0
    n_src_used = 0
    fused = False
    t0 = time.time()

    def flush():
        for k in ks:
            (outs[k] / "eval_cases.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows[k]) + ("\n" if rows[k] else ""), encoding="utf-8")
        with open(out_root / f"{a.name}_attempts.csv", "w", newline="", encoding="utf-8") as f:
            cols = ["draw", "sid", "try", "variant", "outcome", "vmin", "vmax", "paired"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(attempts)

    try:
        for sid in pool:
            if n_pair >= a.n_target or n_draw >= a.max_draws:
                break
            n_src_used += 1
            works, hashes, base_loads = {}, {}, {}
            for k in ks:
                wk = work_root / f"N{k}"
                shutil.rmtree(wk, ignore_errors=True)
                robust_copytree(pack / f"N{k}" / sid, wk)
                works[k] = wk
                hashes[k] = VG.sha256_files(pack / f"N{k}" / sid)
            adj0 = VG.bus_adjacency(works[ks[0]]) if a.q_mode == "local" else None
            for t in range(a.tries_per_mode):
                if n_pair >= a.n_target or n_draw >= a.max_draws:
                    break
                fac, pert = None, None
                verdict = {}
                sig0 = None
                for k in ks:
                    wk = works[k]
                    _clear_stale_pf_outputs(wk)
                    for name in ("LF.L5", "LF.L6", "LF.P1D"):
                        src = pack / f"N{k}" / sid / name
                        if src.is_file():
                            shutil.copy2(src, wk / name)
                    if fac is None:
                        l6 = Card.load(wk / "LF.L6")
                        loads0 = VG.read_loads(l6)
                        sig0 = load_signature(loads0)
                        fac, pert = draw_factors(loads0, rng, a.q_mode, a.f_lo, a.f_hi, adj0, a.local_hops)
                        for i, f in fac.items():
                            loads0[i]["view"].ql = loads0[i]["ql"] * f
                        l6.save(wk / "LF.L6")
                    else:
                        l6 = Card.load(wk / "LF.L6")
                        loads_k = VG.read_loads(l6)
                        assert load_signature(loads_k) == sig0, f"{sid} N{k} 负荷卡与 N{ks[0]} 不一致，无法配对扰动"
                        for i, f in fac.items():
                            loads_k[i]["view"].ql = loads_k[i]["ql"] * f
                        l6.save(wk / "LF.L6")
                    rf = guarded(lambda: psasp_referee(wk, solver), wk, lambda r: bool(r.get("converged")), tag=f"{sid}/N{k}")
                    verdict[k] = classify(rf, low, high, a.max_depth) + (rf,)
                n_draw += 1
                paired = all(verdict[k][0] == "sick" for k in ks)
                partial = (not paired) and any(verdict[k][0] == "sick" for k in ks)
                for k in ks:
                    counts[k][verdict[k][0]] += 1
                    attempts.append({"draw": n_draw, "sid": sid, "try": t + 1, "variant": f"N{k}", "outcome": verdict[k][0],
                                     "vmin": (round(verdict[k][1], 4) if verdict[k][1] is not None else ""),
                                     "vmax": (round(verdict[k][2], 4) if verdict[k][2] is not None else ""),
                                     "paired": int(paired)})
                cid = "%s__q%02d" % (sid, t + 1)
                if paired:
                    n_pair += 1
                if partial:
                    n_partial += 1
                if paired or (partial and a.keep_unpaired):
                    for k in ks:
                        if verdict[k][0] != "sick":
                            continue
                        dst = (outs[k] / "cases" / cid) if paired else (unp / f"N{k}" / "cases" / cid)
                        robust_copytree(works[k], dst)
                        vmin, vmax, rf = verdict[k][1], verdict[k][2], verdict[k][3]
                        v = np.asarray(rf["v"])
                        live = v > 0.01
                        buses_low = [int(i) + 1 for i in np.where(live & (v < low - EPS))[0]]
                        buses_high = [int(i) + 1 for i in np.where(live & (v > high + EPS))[0]]
                        (dst / "perturb.json").write_text(json.dumps({
                            "case_id": cid, "source_sid": sid, "source_pack": str(pack), "variant": f"N{k}",
                            "n1_branch": src_info[sid][k].get("n1_branch", ""),
                            "source_hash_sha256": hashes[k], "disease": "q", "try": t + 1, "seed": a.seed,
                            "paired": paired,
                            "perturb": {**pert, "factors": {str(i): round(f, 6) for i, f in fac.items()}}},
                            ensure_ascii=False, indent=1), encoding="utf-8")
                        row = {"case_id": cid, "sid": sid, "source_lib": pack.name, "tier": a.tier, "disease": "q", "try": t + 1,
                               "variant": f"N{k}", "n1_branch": src_info[sid][k].get("n1_branch", ""), "paired": paired,
                               "source_hash_sha256": hashes[k],
                               "source_vmin": src_info[sid][k]["vmin"], "source_vmax": src_info[sid][k]["vmax"],
                               "vmin": round(vmin, 4), "vmax": round(vmax, 4),
                               "n_bus_low": len(buses_low), "n_bus_high": len(buses_high),
                               "buses_low": buses_low[:50], "buses_high": buses_high[:50],
                               "perturb_summary": {**pert, "stage": 0}}
                        if paired:
                            rows[k].append(row)
                        else:
                            (unp / f"N{k}").mkdir(parents=True, exist_ok=True)
                            with open(unp / f"N{k}" / "eval_cases.jsonl", "a", encoding="utf-8") as f:
                                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print("[viol-nk] %s 第%d抽: %s %s(%d/%d 配对, 抽样 %d, %.0fs)" % (
                    sid, t + 1, " ".join("N%d=%s%s" % (k, verdict[k][0], ("(%.4f/%.4f)" % (verdict[k][1], verdict[k][2])) if verdict[k][1] is not None else "") for k in ks),
                    "★配对入卷" if paired else ("部分" if partial else ""), n_pair, a.n_target, n_draw, time.time() - t0), flush=True)
                flush()
    except LicenseError as e:
        fused = True
        print("[viol-nk] 加密狗失效熔断，立即停止不重试: %s" % e, flush=True)
    finally:
        flush()
        shutil.rmtree(work_root, ignore_errors=True)

    secs = round(time.time() - t0, 1)
    params = ({"f_lo": a.f_lo, "f_hi": a.f_hi} if a.q_mode == "global"
              else {"hops": a.local_hops, "f_lo": a.f_lo, "f_hi": a.f_hi})
    for k in ks:
        sz = sum(p.stat().st_size for p in outs[k].rglob("*") if p.is_file()) / 1e6
        summary = {"tier": a.tier, "disease": "q", "family": "A_qload", "mechanism": a.q_mode, "params": params,
                   "variant": f"N{k}", "paired_design": True, "pack": str(pack), "source_pack_variant_dir": str(pack / f"N{k}"),
                   "n_source_pool": len(healthy), "n_source_used": n_src_used, "n_base_modes": len({r["sid"] for r in rows[k]}),
                   "n_draws": n_draw, "n_sick": len(rows[k]), "n_paired_sick": n_pair, "n_partial_sick_draws": n_partial,
                   "outcomes_this_variant": counts[k],
                   "sick_rate_this_variant": round(counts[k]["sick"] / max(n_draw, 1), 3),
                   "paired_rate": round(n_pair / max(n_draw, 1), 3),
                   "criterion": "bilateral", "vio_band": [low, high],
                   "band_rule": {"inherited_from": str(a.band_from), "rule": ref.get("band_rule", {}).get("rule") if isinstance(ref.get("band_rule"), dict) else None},
                   "eps": EPS, "max_depth": a.max_depth, "seed": a.seed, "tries_per_mode": a.tries_per_mode, "max_draws": a.max_draws,
                   "hit_target": n_pair >= a.n_target, "license_fused": fused, "wall_secs": secs,
                   "secs_per_draw_3variants": round(secs / max(n_draw, 1), 2), "lib_mb": round(sz, 1),
                   "note": "配对考卷：三变体同一扰动系数、只改拓扑；三变体都判病才入卷；无修复标签；阅卷走功能性判据（PSASP 裁判）"}
        (outs[k] / "eval_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    pair_rep = {"name": a.name, "tier": a.tier, "pack": str(pack), "vio_band": [low, high], "seed": a.seed,
                "n_sids_all_variants_converged": len(sids), "n_sids_all_healthy": len(healthy),
                "topology_induced_violation_sids": {s: src_info[s] for s in topo_induced},
                "n_topology_induced": len(topo_induced), "n_other_unhealthy": len(other),
                "n_draws": n_draw, "n_paired_sick": n_pair, "n_partial_sick_draws": n_partial,
                "outcomes_by_variant": counts, "wall_secs": secs, "license_fused": fused}
    (out_root / f"{a.name}_pairs.json").write_text(json.dumps(pair_rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[viol-nk] 收尾：配对病例 %d/%d 抽（源方式 %d；各变体 %s）%.0fs 熔断=%s" % (
        n_pair, n_draw, n_src_used, json.dumps(counts), secs, fused), flush=True)
    return 1 if fused else 0


if __name__ == "__main__":
    raise SystemExit(main())
