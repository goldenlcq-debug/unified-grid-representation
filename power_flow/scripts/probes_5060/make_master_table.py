# -*- coding: utf-8 -*-
"""生产候选(E13)的总表:网架 × 训练状态 × (v, θ) 的平均 / p99 / 最大偏差,三种子聚合。

口径:
- 误差只在 D15 类型化未知量位置统计(θ:PQ+PV;v:PQ),与正式 MAE 同源;
- θ 用**折回**误差 |wrap(pred − 存档)|(存档是 PSASP ±180° 截断值,70k 有 3 个方式被缠绕,原样最大会被 2π 伪误差顶满);
- 平均 = 三种子的平均;p99 / 最大 = **三种子里最坏的那个**(上界口径,标注);
- 同一家族在两个语料根下都出现时(mp_activsg2000),按样本数加权合并。
用法:python scripts/probes_5060/make_master_table.py [--run-tpl ...] [--seeds 0,1,2]
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT = "R1dcL_edge0_wrap_L_vh_rf4qv_aux1_hd0.2_rd0.2_s{}"
NAME = {"mp_case118": "IEEE-118(MATPOWER 底图)", "mp_case300": "IEEE-300", "mp_activsg2000": "ACTIVSg-2000",
        "mp_activsg10k": "ACTIVSg-10k", "mp_activsg25k": "ACTIVSg-25k", "mp_activsg70k": "ACTIVSg-70k",
        "lib118": "IEEE-118(老方式库)", "lib_ieee39": "IEEE-39(老方式库)", "lib_cepri36": "CEPRI-36(老方式库)"}
STATUS = {"真实大网·训练见过": "训练见过", "同拓扑·留出未见": "同拓扑·留出未见", "冻结外集·跨拓扑": "跨拓扑·冻结外集",
          "冻结外集": "跨拓扑·冻结外集", "方式库": "训练见过(无变比)", "方式库·训练见过": "训练见过(无变比)"}
ORDER = ["mp_case118", "mp_case300", "mp_activsg2000", "mp_activsg10k", "mp_activsg25k", "mp_activsg70k",
         "lib118", "lib_ieee39", "lib_cepri36"]


def load(run, run70k=None):
    """返回 {(family, status): [每种子一条 rec]};run70k 给定时 byfam70k 单独取该 run
    (2026-08-30 追加:生产件 fkE_gE30k 门下(<30000)逐位 = fkE,byfam/byfam25k 只在 fkE 名下有产物,
    byfam70k 在 gE30k 名下 —— 见 fk_make_verify-20260828.json V1/V2 逐位验证)。"""
    out = {}
    for stem in ("byfam", "byfam25k", "byfam70k"):
        r_ = run70k if (stem == "byfam70k" and run70k) else run
        f = ROOT / "runs" / "diag_5060" / f"{stem}_{r_}.json"
        if not f.exists():
            continue
        for g in json.loads(f.read_text(encoding="utf-8"))["groups"]:
            key = (g["family"], STATUS.get(g["status"], g["status"]))
            out.setdefault(key, []).append(g)
    # 同一 (family,status) 在一个种子里出现多次(两个语料根)→ 按样本数加权合并
    merged = {}
    for key, gs in out.items():
        n = sum(g["n_samples"] for g in gs)
        w = lambda k1, k2: sum(g[k1][k2] * g["n_samples"] for g in gs) / n            # noqa: E731
        merged[key] = dict(n_samples=n, n_nodes=gs[0]["n_nodes"],
                           v=dict(mean=w("v", "mean"), p99=max(g["v"]["p99"] for g in gs), max=max(g["v"]["max"] for g in gs)),
                           th=dict(mean=w("theta_wrap", "mean"), p99=max(g["theta_wrap"]["p99"] for g in gs),
                                   max=max(g["theta_wrap"]["max"] for g in gs)))
    return merged


def load_compare(model, files):
    """对照模型(旧文件,种子 0):{(family,status): rec}"""
    out = {}
    for f in files:
        fp = ROOT / "runs" / "diag_5060" / f
        if not fp.exists():
            continue
        for g in json.loads(fp.read_text(encoding="utf-8"))["groups"]:
            if g["model"] != model:
                continue
            key = (g["family"], STATUS.get(g["status"], g["status"]))
            if key in out:      # 同族多来源:取样本多的
                if g["n_samples"] <= out[key]["n_samples"]:
                    continue
            out[key] = dict(n_samples=g["n_samples"], v=g["v"]["mean"], v_max=g["v"]["max"],
                            th=g["theta_wrap"]["mean"], th_max=g["theta_wrap"]["max"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-tpl", default=DEFAULT)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--compare", default="R1dcL_edge0_wrap_L_vh_s0", help="对照模型(E10 种子 0)")
    ap.add_argument("--run-tpl-70k", default=None, help="byfam70k 单独用的 run 模板(默认同 --run-tpl)")
    ap.add_argument("--tag", default=None, help="输出文件名 master_table_<tag>.json(默认按旧规则推断)")
    a = ap.parse_args()
    cmp_ = load_compare(a.compare, ["byfam_r5_fast.json", "byfam_r5_big.json", "byfam_e10b_libs.json"])
    seeds = [int(x) for x in a.seeds.split(",")]
    per = [load(a.run_tpl.format(s), a.run_tpl_70k.format(s) if a.run_tpl_70k else None) for s in seeds]
    keys = sorted({k for p in per for k in p}, key=lambda k: (ORDER.index(k[0]) if k[0] in ORDER else 99, k[1]))
    print(f"{'网架':<22}{'节点':>7}{'训练状态':<16}{'方式':>5} | "
          f"{'v 平均':>11}{'v p99':>9}{'v 最大':>9} | {'θ 平均°':>9}{'θ p99°':>8}{'θ 最大°':>8} | {'E10 v 平均':>11}{'E10 θ 平均°':>11}")
    rows = []
    for k in keys:
        gs = [p[k] for p in per if k in p]
        vm = float(np.mean([g["v"]["mean"] for g in gs]))
        tm = float(np.mean([g["th"]["mean"] for g in gs]))
        vp, vx = max(g["v"]["p99"] for g in gs), max(g["v"]["max"] for g in gs)
        tp, tx = max(g["th"]["p99"] for g in gs), max(g["th"]["max"] for g in gs)
        rows.append(dict(fam=k[0], status=k[1], n=gs[0]["n_samples"], nodes=gs[0]["n_nodes"], seeds=len(gs),
                         v_mean=vm, v_p99=vp, v_max=vx, th_mean=tm, th_p99=tp, th_max=tx))
        c = cmp_.get(k)
        rows[-1]["e10_v"] = c["v"] if c else None
        rows[-1]["e10_th"] = c["th"] if c else None
        rows[-1]["e10_v_max"] = c["v_max"] if c else None
        rows[-1]["e10_th_max"] = c["th_max"] if c else None
        print(f"{NAME.get(k[0], k[0]):<22}{gs[0]['n_nodes']:>7}{k[1]:<16}{gs[0]['n_samples']:>5} | "
              f"{vm:>11.6f}{vp:>9.5f}{vx:>9.5f} | {math.degrees(tm):>9.4f}{math.degrees(tp):>8.3f}{math.degrees(tx):>8.3f} | "
              f"{(f'{c[chr(118)]:.5f}' if c else '—'):>11}{(f'{math.degrees(c[chr(116)+chr(104)]):.3f}' if c else '—'):>11}")
    tag = a.tag if a.tag else ("e13" if "rf4qv" in a.run_tpl else "e10" if a.run_tpl.endswith("_vh_s{}")
           else "e16" if "R1dcLv" in a.run_tpl
           else "e15b" if "shz_hd0.2_prn" in a.run_tpl else "e15a" if "shz_hd0.2" in a.run_tpl else "x")
    (ROOT / "runs" / "diag_5060" / f"master_table_{tag}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写 runs/diag_5060/master_table_{tag}.json")

if __name__ == "__main__":
    main()
