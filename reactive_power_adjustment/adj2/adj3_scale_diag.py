# -*- coding: utf-8 -*-
r"""超大网泛化诊断：模型指针选的机组，到底离病灶多远？（2026-08-26，零 PSASP）

背景：v6d 在 25k 深卷 18 例上 **首步 0/18、含阶梯 2/18**，而规则臂（就近选台）**8/18**，
级联 = 规则单干（模型贡献为零）。当年归因是"25k 指针超阈候选一两千台、**排序前 32 台
远离病灶**"（结果档 §12.2）——但那是定性推断，从未量化。

本脚本量化它（只前向 + 图上跳数，不解潮流）：
  对每个病例：编码 → 指针前向 → 取模型排序前 k 台 → 算它们到**最近低压母线**的跳数；
  对照：全网可调机组中离病灶最近的那些台（= 规则臂会选的）。
  若"模型前 k 台的跳数"远大于"最近可用机组的跳数"，则归因坐实，
  且**解法是解码期加局部性预筛（零重训）**，而非重训。

  python adj3_scale_diag.py --lib viol_q_25k_local_deep --ckpt <v6d目录> --n-bus 25000
"""
from __future__ import annotations

import argparse
import copy as _copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adj2_common as C  # noqa: E402
from adj2_eval import load_reasoner  # noqa: E402
from adj2_train import side_feats  # noqa: E402
from gridfm.data.graph import sample_to_data  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.psasp_io.catalog import Catalog  # noqa: E402
from gridfm.reason.ts_task import gen_node_mask  # noqa: E402
from gridfm.schema.assemble import build_sample  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-bus", type=int, required=True)
    ap.add_argument("--task", default="reactive", choices=("reactive", "active"))
    ap.add_argument("--low", type=float, default=0.93)
    ap.add_argument("--high", type=float, default=1.08)
    ap.add_argument("--topk", default="1,4,8,16,32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    topk = [int(x) for x in a.topk.split(",")]

    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    v1 = C.REPO / "runs" / "encoder_v1"
    ck = torch.load(v1 / "v1.pt", map_location="cpu", weights_only=False)
    from dataclasses import fields as dc_fields
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items()
                         if k in {f.name for f in dc_fields(TrainConfig)}})
    enc, _ = build_models(cfg)
    enc.load_state_dict(ck["model_state"])
    enc.to(dev).eval()
    norm = Normalizer.load(v1 / "normalizer.npz")
    reasoner, variant, _rcfg, ckpt_path = load_reasoner(a.task, a.ckpt, dev, cfg.d)
    print("[diag] %s 变体=%s ckpt=%s" % (a.task, variant, ckpt_path), flush=True)

    lib = Path(a.lib) if Path(a.lib).is_absolute() else C.LINE / "libs" / a.lib   # 2026-09-05: 允许绝对路径
    rows = [json.loads(ln) for ln in (lib / "eval_cases.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    if a.limit:
        rows = rows[: a.limit]
    print("[diag] 考卷 %s %d 例 低压线 %.2f 前 k 档 %s" % (a.lib, len(rows), a.low, topk), flush=True)

    out = []
    for ci, r in enumerate(rows):
        cdir = lib / "cases" / r["case_id"]
        pc = PsaspCase(cdir)
        raw = build_sample(pc, Catalog(pc), sample_id=r["case_id"],
                           family_id=r.get("source_lib", a.lib), schema="v0")
        # 病灶 = 低压母线（1 基）
        v = C.bus_voltage(cdir, a.n_bus)
        live = v > 0.01
        low_buses = [int(i) + 1 for i in np.where(live & (v < a.low - 1e-4))[0]]
        if not low_buses:
            print("[diag] %s 无低压母线,跳过" % r["case_id"], flush=True)
            continue
        # 图上跳数（从全部低压母线出发）
        adj = C.bus_adjacency(pc)
        hops = C.hops_from(adj, low_buses)
        # 可调机组（与跑分器同口径）
        gm = gen_node_mask(raw).copy()
        from gridfm.schema.fields import NODE_TABLE as NT
        if a.task == "active":
            pmax = raw.node_feat[:, NT.offset_of("pmax")]
            pmin = raw.node_feat[:, NT.offset_of("pmin")]
            gm = gm & (pmax > pmin + 1e-9)
        gen_rows = np.where(gm)[0]
        gen_hops = np.array([hops.get(int(i) + 1, 9999) for i in gen_rows])
        reach = gen_hops < 9999
        # 模型前向
        s = _copy.deepcopy(raw)
        norm.transform(s)
        C.apply_dyn_fill(s, raw, "zero")
        data = sample_to_data(s, cfg.lape_k, cfg.rwse_steps, random_dim=cfg.random_dim,
                              branch_random_dim=cfg.branch_random_dim)
        data.depth = torch.from_numpy(side_feats(a.task, variant, raw, a.low, a.high))
        batch = Batch.from_data_list([data]).to(dev)
        with torch.no_grad():
            h, g, e = enc(batch)
            o = reasoner(h, g, e, batch.batch, batch.batch[batch.edge_index[0]], batch.depth)
        ptr = o[0] if isinstance(o, (tuple, list)) else o
        prob = torch.sigmoid(ptr.float()).cpu().numpy()
        order = gen_rows[np.argsort(-prob[gen_rows], kind="stable")]
        model_hops = np.array([hops.get(int(i) + 1, 9999) for i in order])
        # 规则臂会选的（最近的）
        rule_hops = np.sort(gen_hops[reach]) if reach.any() else np.array([9999])

        rec = {"case_id": r["case_id"], "n_low_bus": len(low_buses),
               "n_gen_adjustable": int(gm.sum()),
               "n_gen_reachable": int(reach.sum()),
               "nearest_gen_hop": int(rule_hops[0]) if len(rule_hops) else None,
               "n_over_thr": int((prob[gen_rows] > 0.5).sum())}
        for k in topk:
            mh = model_hops[:k]
            rh = rule_hops[:k] if len(rule_hops) >= k else rule_hops
            rec["model_top%d_hop_median" % k] = (int(np.median(mh[mh < 9999])) if (mh < 9999).any() else None)
            rec["model_top%d_unreachable" % k] = int((mh >= 9999).sum())
            rec["rule_top%d_hop_median" % k] = (int(np.median(rh[rh < 9999])) if (rh < 9999).any() else None)
        out.append(rec)
        print("[diag] %2d/%d %s 低压 %d 母线 | 可调机 %d (可达 %d) | 最近机 %s 跳 | 模型前8 中位 %s 跳(不可达 %d) | 规则前8 中位 %s 跳 | 超阈 %d"
              % (ci + 1, len(rows), r["case_id"], len(low_buses), rec["n_gen_adjustable"],
                 rec["n_gen_reachable"], rec["nearest_gen_hop"],
                 rec["model_top8_hop_median"], rec["model_top8_unreachable"],
                 rec["rule_top8_hop_median"], rec["n_over_thr"]), flush=True)

    if out:
        print("\n[diag] ===== 汇总（%d 例）=====" % len(out))
        for k in topk:
            mm = [r["model_top%d_hop_median" % k] for r in out if r["model_top%d_hop_median" % k] is not None]
            rr = [r["rule_top%d_hop_median" % k] for r in out if r["rule_top%d_hop_median" % k] is not None]
            un = [r["model_top%d_unreachable" % k] for r in out]
            print("  前 %2d 台：模型跳数中位 %s | 规则跳数中位 %s | 模型不可达台数均 %.1f"
                  % (k, (int(np.median(mm)) if mm else None), (int(np.median(rr)) if rr else None),
                     float(np.mean(un))))
        ng = [r["n_gen_adjustable"] for r in out]
        nt = [r["n_over_thr"] for r in out]
        print("  可调机组数中位 %d | 指针超阈(>0.5)台数中位 %d" % (int(np.median(ng)), int(np.median(nt))))
        p = Path(a.out) if a.out else (C.LINE / "调整器效果提升-20260822" / "pilots" /
                                       ("scale_diag_%s.json" % a.lib))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"lib": a.lib, "ckpt": str(ckpt_path), "cases": out},
                                ensure_ascii=False, indent=1), encoding="utf-8")
        print("  报告 %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
