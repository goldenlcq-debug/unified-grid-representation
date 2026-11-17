#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐样本误差诊断：对某个家族的 N0/N1/N2 配对集，逐样本报 E13 与 E10（s0）的 θ 平均/最大误差、PSASP 解的 θ 极值、开断支路。
用法：python diag_per_sample.py --fam mp_case300 [--seeds 0]"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(r"E:/潮流线优化包-20260821/grid-foundation")
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
import pf_reasoner as M  # noqa: E402
import eval_by_family_5060 as E  # noqa: E402
from gridfm.schema.sample import GridSample  # noqa: E402
from gridfm.schema.fields import NODE_TABLE  # noqa: E402

HERE = Path(__file__).resolve().parent
R = 57.29577951308232
E13 = "R1dcL_edge0_wrap_L_vh_rf4qv_aux1_hd0.2_rd0.2"
E10 = "R1dcL_edge0_wrap_L_vh"
TH = NODE_TABLE.offset_of("theta")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--variants", default="0,1,2")
    ap.add_argument("--models", default="E13,E10")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for tag, name in (("E13", E13), ("E10", E10)):
        if tag not in a.models.split(","):
            continue
        ck = PKG / "runs" / f"pf_reasoner_5060_{name}_s{a.seed}" / "ckpt" / "best.pt"
        models[tag] = E.build_from_ckpt(str(ck), str(PKG / "runs" / "encoder_v1"), dev, 0)
    out = {}
    ks = [int(x) for x in a.variants.split(",")]
    for k in ks:
        root = HERE / "gridfm-data-nk" / f"N{k}"
        fam_dir = root / "families" / "val" / a.fam
        man = {r["sample_id"]: r for r in csv.DictReader(open(fam_dir / "manifest.csv", encoding="utf-8"))}
        for sid in sorted(man):
            raw = GridSample.load_npz(fam_dir / f"{sid}.npz")
            M.attach_family_extra([raw], [root])
            rec = out.setdefault(sid, {})
            th = raw.node_feat[:, TH]
            rec[f"N{k}"] = {"branch": man[sid]["n1_branch"], "theta_absmax_deg": float(np.degrees(np.abs(th).max())),
                            "theta_min_deg": float(np.degrees(th.min())), "theta_max_deg": float(np.degrees(th.max()))}
            for tag, (enc, reasoner, cfg, norm, meta) in models.items():
                dn = torch.from_numpy(E.pf_phys_denorm(norm)).to(dev) if meta["use_phys"] else None
                et, ev, etw = E.errors(enc, reasoner, [raw], norm, cfg, dev, meta, dn, 1)
                rec[f"N{k}"][tag] = {"th_mean_deg": float(et.mean() * R), "th_max_deg": float(et.max() * R), "thw_max_deg": float(etw.max() * R),
                                     "v_mean": float(ev.mean()), "v_max": float(ev.max()),
                                     "n_nodes_theta_gt1deg": int((etw * R > 1).sum()), "n_nodes_theta_gt5deg": int((etw * R > 5).sum()),
                                     "n_nodes_v_gt0.01": int((ev > 0.01).sum()), "n_theta_nodes": int(etw.size)}
    (HERE / "nk_results" / f"diag_{a.fam}_s{a.seed}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    mtag = a.models.split(",")[0]
    for k in [x for x in ks if x > 0]:
        worst = sorted(((rec[f"N{k}"][mtag]["thw_max_deg"], sid) for sid, rec in out.items() if f"N{k}" in rec), reverse=True)[:a.top]
        print(f"== N{k} worst by {mtag} wrapped max θ error:")
        for m, sid in worst:
            r = out[sid]; d = r[f"N{k}"][mtag]
            print(f"  {sid} branch {r[f'N{k}']['branch']:<36} {mtag} mean {d['th_mean_deg']:.4f}° max {m:.1f}° | nodes θ>1° {d['n_nodes_theta_gt1deg']} / >5° {d['n_nodes_theta_gt5deg']} of {d['n_theta_nodes']}; V>0.01 {d['n_nodes_v_gt0.01']} "
                  f"| PSASP θ range [{r[f'N{k}']['theta_min_deg']:.1f}, {r[f'N{k}']['theta_max_deg']:.1f}]°")
    for k in ks:
        ms = [rec[f"N{k}"][mtag]["thw_max_deg"] for rec in out.values() if f"N{k}" in rec]
        print(f"N{k}: 样本数 {len(ms)}, {mtag} 逐样本最大误差中位 {np.median(ms):.3f}°, >1° 的样本 {sum(1 for x in ms if x > 1)}, >10° 的样本 {sum(1 for x in ms if x > 10)}")


if __name__ == "__main__":
    main()
