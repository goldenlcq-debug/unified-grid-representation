"""[G] ① 预筛头 · 候选特征重放重建(2026-08-03;GPU 确定性重放,零 PSASP)。

预注册 = docs/预注册-预筛头回放-20260803.md(v2:全池悲观界口径)。本脚本按在案
z 种子精确重生成**全池候选**:三臂 × 全 14 族 × 8 种子 × 族内全需求(6,000 行,
其中 2,544 行带真实三态标签、其余 label=unknown)+ exit_vsetbase 单抽 750 行;
提取**仿真前可得**特征落表;并做写死的保真校验:ok 样本 recompute
(gen_load 4位/gen_online)vs 当年 report 逐条对齐,≥99% 一致才允许进训练。

采样循环与 scripts/g_bestofk_pilot.py 逐行同构(全族批 z_T + 切片、DDIM250/clip4、
CFG1.5、uncal、vset_base 一刀);候选顺位/z_seed 不入特征(预注册禁入面)。

  python scripts/g5_prescreen_replay.py [--out runs/g5_prescreen]
  python scripts/g5_prescreen_replay.py --only-model s1   # 分段跑
"""
import argparse
import copy
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import gridfm  # noqa: E402
assert gridfm.__file__.startswith(str(REPO)), f"窜仓:{gridfm.__file__}"

from g_clabel_bandin import (MODE_FIELDS, build_labels, load_stack,  # noqa: E402
                             prep_family, resolve_cp)
from gridfm.data import c_label as CL  # noqa: E402
from gridfm.schema.fields import NODE_TABLE, Group  # noqa: E402
from gridfm.schema.sample import group_bit_index  # noqa: E402
from gridfm.train.masking import COL_GROUP  # noqa: E402

CFG_SCALE = 1.5
VAL8 = ["synth210_s2007", "synth150_s2000", "synth210_s2002", "synth240_s2003",
        "synth270_s2004", "synth150_s2005", "synth180_s2001", "synth180_s2006"]
TRANSFER6 = ["synth400_s3000", "synth400_s3001", "synth400_s3002", "synth400_s3003",
             "synth700_s3004", "synth1000_s3008"]
ARMS = {
    "s1": dict(denoiser="runs/diffusion_g1_cgen_g4_s1/denoiser.pt",
               calib="runs/g_clabel_calib/calib_s1.json", seeds=list(range(1000, 1008))),
    "s2": dict(denoiser="runs/diffusion_g1_cgen_g4_s2/denoiser.pt",
               calib="runs/g_clabel_calib/calib_s2.json", seeds=list(range(2000, 2008))),
    "lg": dict(denoiser="runs/diffusion_g1_cgen_g4_large_s1/denoiser.pt",
               calib="runs/g_clabel_calib/calib_large_s1.json", seeds=list(range(3000, 3008))),
}
EXIT_OF = {("s1", "val"): "exit_val_s1_vsetbase.json",
           ("s1", "transfer"): "exit_transfer_s1_vsetbase.json",
           ("s2", "val"): "exit_val_s2_vsetbase.json",
           ("s2", "transfer"): "exit_transfer_s2_vsetbase.json",
           ("lg", "val"): "exit_val_large_s1_vsetbase.json",
           ("lg", "transfer"): "exit_transfer_large_s1_vsetbase.json"}


def wanted_labels():
    """(model, z_seed, sid) -> label(三态);解析逻辑与盘点脚本同构(已互验)。"""
    R = REPO / "runs"
    out = {}

    def bok(path, model, base, k):
        r = json.load(open(path, encoding="utf-8"))
        assert r["partial"] is False
        for fam, F in r["families"].items():
            viol = {(v["sid"], v["draw"]) for v in F.get("violations", [])}
            for sid, v in F["solved"].items():
                for dd in range(1, v["draw"]):
                    out.setdefault((model, base + dd - 1, sid),
                                   "viol" if (sid, dd) in viol else "nonconv")
                out.setdefault((model, base + v["draw"] - 1, sid), "ok")
            for sid in F["unsolved_sids"]:
                for dd in range(1, k + 1):
                    out.setdefault((model, base + dd - 1, sid),
                                   "viol" if (sid, dd) in viol else "nonconv")
        return r

    dv = bok(R / "g2_delivery_v1/report.json", "s1", 1000, 5)
    bok(R / "g2_k8_s2/report.json", "s2", 2000, 8)
    bok(R / "g2_k8_lg/report.json", "lg", 3000, 8)
    ux = json.load(open(R / "g2_union_ext/report.json", encoding="utf-8"))
    model_of = {"s1_ext": "s1", "s2_alt": "s2", "lg_alt": "lg"}
    for arm_name, arm in ux["arms"].items():
        model = model_of[arm_name]
        for fam, F in arm["families"].items():
            viol = {(v["sid"], v.get("draw_in_arm")) for v in F.get("violations", [])}
            zs = [m["z_seed"] for m in F["draw_marginal"]]
            solved_at = {sid: v["draw_in_arm"] for sid, v in F["solved"].items()}
            pend = list(dv["families"][fam]["unsolved_sids"])
            for i, z in enumerate(zs, start=1):
                for sid in pend:
                    if solved_at.get(sid, 99) >= i:
                        lab = ("ok" if solved_at.get(sid) == i
                               else "viol" if (sid, i) in viol else "nonconv")
                        out.setdefault((model, z, sid), lab)
    # exit 副数据(seed 0,单抽,uncal_1.5 cell;显式 conv+vmin/vmax → 三态)
    for (model, split), fn in EXIT_OF.items():
        r = json.load(open(R / "g_clabel_bandin" / fn, encoding="utf-8"))
        for fam, F in r["results"].items():
            for row in F["cells"]["uncal_1.5"]["rows"]:
                if not row["conv"]:
                    lab = "nonconv"
                elif row["vmin"] >= 0.85 and row["vmax"] <= 1.10:
                    lab = "ok"
                else:
                    lab = "viol"
                out.setdefault((model, 0, row["sid"]), lab)
    return out


def ok_truth():
    """保真靶:(model, z_seed, sid) -> (gen_load, gen_online) 来自 report solved。"""
    R = REPO / "runs"
    t = {}
    dv = json.load(open(R / "g2_delivery_v1/report.json", encoding="utf-8"))
    for src, model, base in ((R / "g2_delivery_v1/report.json", "s1", 1000),
                             (R / "g2_k8_s2/report.json", "s2", 2000),
                             (R / "g2_k8_lg/report.json", "lg", 3000)):
        r = json.load(open(src, encoding="utf-8"))
        for fam, F in r["families"].items():
            for sid, v in F["solved"].items():
                t[(model, base + v["draw"] - 1, sid)] = (v["gen_load"], v["gen_online"])
    ux = json.load(open(R / "g2_union_ext/report.json", encoding="utf-8"))
    model_of = {"s1_ext": "s1", "s2_alt": "s2", "lg_alt": "lg"}
    for arm_name, arm in ux["arms"].items():
        for fam, F in arm["families"].items():
            for sid, v in F["solved"].items():
                key = (model_of[arm_name], v["z_seed"], sid)
                t.setdefault(key, (None, None))
    # exit 批 rows 全员有 gen_load/gen_online(含失败行)→ 全员进校验
    for (model, _split), fn in EXIT_OF.items():
        r = json.load(open(R / "g_clabel_bandin" / fn, encoding="utf-8"))
        for fam, F in r["results"].items():
            for row in F["cells"]["uncal_1.5"]["rows"]:
                t[(model, 0, row["sid"])] = (row["gen_load"], row["gen_online"])
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/g5_prescreen")
    ap.add_argument("--only-model", default=None, choices=(None, "s1", "s2", "lg"))
    ap.add_argument("--only-fam", default=None, help="小样本保真验证用(如 synth150_s2000)")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    want = wanted_labels()
    truth = ok_truth()
    # 全池:主臂 = 固定 8 种子 × 全族全需求(标签缺失 → unknown);exit = seed 0
    by_arm_fam = {}
    for model, spec in ARMS.items():
        for fam in VAL8 + TRANSFER6:
            by_arm_fam[(model, fam)] = set(spec["seeds"]) | {0}
    print(f"[replay] 标签样本 {len(want)};族级采样任务 "
          f"{sum(len(v) for v in by_arm_fam.values())} 次(含 unknown 全池)", flush=True)

    o = NODE_TABLE.offset_of
    col = {f: o(f) for f in MODE_FIELDS}
    sbit = group_bit_index(Group.STATE)
    scols = np.where(COL_GROUP == sbit)[0]
    rows_out, fid_ok, fid_bad, fid_tot = [], 0, [], 0
    t00 = time.time()
    for model, spec in ARMS.items():
        if a.only_model and model != a.only_model:
            continue
        S = load_stack(dev, spec["denoiser"])
        calib = json.loads((REPO / spec["calib"]).read_text(encoding="utf-8"))
        supp = calib["support"]
        for split, fams in (("val", VAL8), ("transfer", TRANSFER6)):
            for fam in fams:
                if a.only_fam and fam != a.only_fam:
                    continue
                zs = sorted(by_arm_fam.get((model, fam), ()))
                if not zs:
                    continue
                N = 20 if split == "val" else 15
                F = prep_family(fam, N, S, dev, split)
                cp = resolve_cp(calib, CFG_SCALE, fam, "pooled", supp)
                lab_c = build_labels(F, "uncal", cp, S["lnorm"], dev)
                lab_u = torch.stack(
                    [torch.from_numpy(CL.encode(None, S["lnorm"])).to(dev)] * F["N"])
                n, Nf = F["n"], F["N"]
                net, dec, diff = S["net"], S["dec"], S["diff"]
                z_mean, z_std = S["z_mean"], S["z_std"]
                sid_idx = {F["test_s"][j].sample_id: j for j in range(Nf)}

                @torch.no_grad()
                def eps_fn(z_t, t):
                    tt = torch.full((Nf,), t, dtype=torch.long, device=dev)
                    e_c = net(z_t, F["hc"], F["ec"], F["ei"], tt, lab_c, F["npg"])
                    e_u = net(z_t, F["hc"], F["ec"], F["ei"], tt, lab_u, F["npg"])
                    return e_u + CFG_SCALE * (e_c - e_u)

                base_vset = F["base_feat"][:, col["v_set"]]
                gen_ns = F["gen_ns"]
                for z_seed in zs:
                    torch.manual_seed(z_seed)
                    z_T = torch.randn(Nf * n, S["d"], device=dev)
                    with torch.no_grad():
                        z0 = diff.ddim_sample(eps_fn, z_T, steps=250, clip_x0=4.0)
                        fields = dec(z0 * z_std + z_mean, F["ec"], F["ei"]).cpu().numpy()
                    z0c = z0.cpu().numpy()
                    for sid in sorted(sid_idx):
                        j = sid_idx[sid]
                        full = F["normed"][j].node_feat.copy()
                        full[:, scols] = fields[j * n:(j + 1) * n][:, scols]
                        pu = S["norm"].inverse_transform_node(full)
                        vset_gen = pu[:, col["v_set"]].copy()   # 模型"想设"的电压(特征)
                        pu[:, col["v_set"]] = base_vset          # vset_base 一刀(交付写回)
                        g = copy.deepcopy(F["test_s"][j]); g.node_feat = pu
                        gl = CL.recompute(g, F["base_pl"])
                        tl = F["true_labels"][j]
                        zj = z0c[j * n:(j + 1) * n]
                        pg = pu[:, col["pg"]][gen_ns]
                        ql = pu[:, col["ql"]]
                        pl = pu[:, col["pl"]]
                        dv_ = vset_gen[gen_ns] - base_vset[gen_ns]
                        lab = want.get((model, z_seed, sid), "unknown")
                        # 保真校验(report 有 recompute 记录者全员比对)
                        tr = truth.get((model, z_seed, sid))
                        if tr and tr[0] is not None and tr[1] is not None:
                            fid_tot += 1
                            same = (round(float(gl["load_level"]), 4) == tr[0]
                                    and int(gl["online_count"]) == tr[1])
                            if same:
                                fid_ok += 1
                            else:
                                fid_bad.append(dict(
                                    model=model, z=z_seed, sid=sid,
                                    got=[round(float(gl["load_level"]), 4),
                                         int(gl["online_count"])], want=list(tr)))
                        ld = abs(gl["load_level"] - tl["load_level"]) / max(
                            abs(tl["load_level"]), 1e-9)
                        rows_out.append({
                            "model": model, "z_seed": z_seed, "sid": sid, "fam": fam,
                            "split": split, "n_nodes": n, "label": lab,
                            "y": -1 if lab == "unknown" else (1 if lab == "ok" else 0),
                            # 需求级(部署可得)
                            "req_load": round(float(tl["load_level"]), 4),
                            "req_online": int(tl["online_count"]),
                            "req_ngen": int(tl["n_gen"]),
                            # 候选级 recompute
                            "gen_load": round(float(gl["load_level"]), 4),
                            "gen_online": int(gl["online_count"]),
                            "load_rel_dev": round(float(ld), 4),
                            "online_dev": int(abs(gl["online_count"] - tl["online_count"])),
                            # pg/ql/pl 分布
                            "pg_mean": round(float(pg.mean()), 4),
                            "pg_std": round(float(pg.std()), 4),
                            "pg_max": round(float(pg.max()), 4),
                            "pg_neg": int((pg < -1e-6).sum()),
                            "ql_std": round(float(ql.std()), 4),
                            "pl_std": round(float(pl.std()), 4),
                            # 模型想设电压 vs base(历史机理:推高电压破坏收敛)
                            "vset_dev_mean": round(float(dv_.mean()), 4),
                            "vset_dev_std": round(float(dv_.std()), 4),
                            "vset_dev_maxabs": round(float(np.abs(dv_).max()), 4),
                            "vset_hi": int((vset_gen[gen_ns] > 1.06).sum()),
                            # 潜码统计(候选切片)
                            "z0_std": round(float(zj.std()), 4),
                            "z0_maxabs": round(float(np.abs(zj).max()), 4),
                            "z0_clip_frac": round(float((np.abs(zj) > 3.98).mean()), 5),
                        })
                    print(f"[replay] {model} {fam} z{z_seed}: {Nf} 候选"
                          f" 累计 {len(rows_out)} ({time.time()-t00:.0f}s)", flush=True)
                del F
                torch.cuda.empty_cache()
    # 落盘
    suffix = (f"_{a.only_model}" if a.only_model else "") + \
             (f"_{a.only_fam}" if a.only_fam else "")
    fcsv = out / f"features{suffix}.csv"
    with open(fcsv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    fid = {"n_checked": fid_tot, "n_match": fid_ok,
           "rate": None if not fid_tot else round(fid_ok / fid_tot, 4),
           "mismatches": fid_bad[:50]}
    (out / f"fidelity{suffix}.json").write_text(
        json.dumps(fid, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[replay] 完成:{len(rows_out)} 行 → {fcsv}")
    print(f"[replay] 保真:{fid_ok}/{fid_tot}"
          f"({0 if not fid_tot else fid_ok/fid_tot:.1%});门=99%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
