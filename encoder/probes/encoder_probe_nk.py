#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结编码器 v1（seed0）在任意方式集上的 §1.1 / §1.2 探针复跑（2026-09-04，长卿）。
与 scripts/p1_full_report.py 的门禁段逐位同源（同函数、同抽样规则），只是把数据根改成可指定：
  ① 寻址（8.2 重采样检索）：in 集按家族轮转取 20 个方式，余弦 top-1；
  ② 并联支路 / 节点孪生：in 集前 4 个家族各构造一对；
  ③ V/θ 恢复与图调和 L2 参照、增量阶梯、技巧分：in = 源网集，out = 更大网集（transfer_report，D20 口径）；
  ④ 固定掩码重构损失 eval_loss（in / out 各一个数）。
用法：
  python encoder_probe_nk.py --in-root <root> --in-split val --out-root <root> --out-split transfer --label N0 --out-json x.json
数据根布局：root/families/<split>/<family>/*.npz（manifest 驱动）。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1] / "grid-foundation"   # 包内副本（= 原 D 盘仓 grid-foundation 的 gridfm，逐文件 md5 见 04_核验）
PKG = Path(__file__).resolve().parents[2]   # 包根；编码器在 PKG/02_编码器（--v1-dir 缺省已改为该目录）
sys.path.insert(0, str(REPO))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import torch  # noqa: E402
from gridfm.data.dataset import load_families  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.eval.transfer import transfer_report, format_report  # noqa: E402
from gridfm.eval.twins import add_branch_twin, add_node_twin, branch_twin_separability, twin_probe  # noqa: E402
from gridfm.probes.retrieval import _valid_node_mask, retrieval_probe  # noqa: E402
from gridfm.schema.fields import NODE_TABLE  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models, eval_loss  # noqa: E402


def _gen_bus(sample):
    ei = sample.edge_index
    off = NODE_TABLE.offset_of
    for a in range(sample.n_nodes):
        if sample.node_feat[a, off("is_null")] > 0.5:
            continue
        if int((ei[0] == a).sum() + (ei[1] == a).sum()) >= 2:
            return a
    return 0


def load_encoder(v1_dir: Path):
    ck = torch.load(v1_dir / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in allowed})
    enc, dec = build_models(cfg)
    enc.load_state_dict(ck["model_state"]); dec.load_state_dict(ck["decoder_state"])
    enc.eval(); dec.eval()
    norm = Normalizer.load(v1_dir / "normalizer.npz")
    return enc, dec, cfg, norm, ck.get("global_step")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", default=str(Path(__file__).resolve().parents[2] / "02_编码器"))
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--in-split", default="val")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--out-split", default="transfer")
    ap.add_argument("--label", default="")
    ap.add_argument("--fam-limit", type=int, default=0, help="每家族最多取几个方式（0 = 全部；按 manifest 顺序）")
    ap.add_argument("--in-families", default=None, help="逗号分隔的 family_id 或前缀，过滤 in 集")
    ap.add_argument("--out-families", default=None, help="逗号分隔的 family_id 或前缀，过滤 out 集")
    ap.add_argument("--probes-on", default="in", choices=["in", "out"], help="①寻址②孪生在哪个集上做（默认 in；分组任务用 out）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    enc, dec, cfg, norm, step = load_encoder(Path(a.v1_dir))
    in_raw = load_families(Path(a.in_root), a.in_split)
    out_raw = load_families(Path(a.out_root), a.out_split) if a.out_root else []

    def famfilter(ss, spec):
        if not spec:
            return ss
        pats = [x.strip() for x in spec.split(",") if x.strip()]
        return [s for s in ss if any(s.family_id == q or s.family_id.startswith(q) for q in pats)]
    in_raw, out_raw = famfilter(in_raw, a.in_families), famfilter(out_raw, a.out_families)

    def limit(ss):
        if not a.fam_limit:
            return ss
        by = {}
        for s in ss:
            by.setdefault(s.family_id, []).append(s)
        return [s for f in sorted(by) for s in by[f][:a.fam_limit]]
    in_raw, out_raw = limit(in_raw), limit(out_raw)

    def prep(ss):
        o = [copy.deepcopy(s) for s in ss]
        for s in o:
            norm.transform(s)
        return o
    val, xfer = prep(in_raw), prep(out_raw)
    fams = lambda ss: sorted({s.family_id for s in ss})  # noqa: E731
    rep = {"label": a.label, "encoder_step": step, "in_root": a.in_root, "in_split": a.in_split,
           "out_root": a.out_root, "out_split": a.out_split,
           "n_in": len(val), "n_out": len(xfer), "families_in": fams(val), "families_out": fams(xfer)}
    print(f"[probe:{a.label}] in {len(val)} 例 / {len(fams(val))} 族; out {len(xfer)} 例 / {len(fams(xfer))} 族; encoder step {step}", flush=True)

    enc.to("cpu"); dec.to("cpu")
    probe_set = xfer if (a.probes_on == "out" and xfer) else val
    rep["probes_on"] = a.probes_on
    # ① 寻址：家族轮转取 20 个（在 probe_set 上）
    by_fam = {}
    for s in probe_set:
        by_fam.setdefault(s.family_id, []).append(s)
    probes, k = [], 0
    while len(probes) < 20 and any(len(v) > k for v in by_fam.values()):
        for fid in sorted(by_fam):
            if len(by_fam[fid]) > k and len(probes) < 20:
                probes.append(by_fam[fid][k])
        k += 1
    accs = [retrieval_probe(enc, s, np.random.default_rng(i), lape_k=cfg.lape_k, rwse_steps=cfg.rwse_steps,
                            random_dim=cfg.random_dim, branch_random_dim=cfg.branch_random_dim) for i, s in enumerate(probes)]
    accs = [x for x in accs if np.isfinite(x)]
    nvalid = float(np.mean([_valid_node_mask(s).sum() for s in probes])) if probes else 1.0
    rep["gate1_retrieval"] = {"mean_top1": float(np.mean(accs)) if accs else float("nan"),
                              "min": float(np.min(accs)) if accs else None, "n_samples": len(accs),
                              "n_families": len({s.family_id for s in probes}), "random_baseline": 1.0 / max(nvalid, 1.0)}
    print(f"  ①寻址 top-1 {rep['gate1_retrieval']['mean_top1']:.6f} (n={len(accs)}, 基线 {1.0/max(nvalid,1.0):.6f})", flush=True)

    # ② 孪生：前 4 个家族
    node_reps, branch_reps = [], []
    for fid in sorted(by_fam)[:4]:
        base = by_fam[fid][0]
        ntw, npair = add_node_twin(base, _gen_bus(base))
        tp = twin_probe(enc, ntw, [npair], random_dim=cfg.random_dim, branch_random_dim=cfg.branch_random_dim,
                        lape_k=cfg.lape_k, rwse_steps=cfg.rwse_steps)
        tp["family_id"] = fid; node_reps.append(tp)
        btw, bpair = add_branch_twin(base, 0)
        bt = branch_twin_separability(enc, btw, [bpair], random_dim=cfg.random_dim, branch_random_dim=cfg.branch_random_dim,
                                      lape_k=cfg.lape_k, rwse_steps=cfg.rwse_steps)
        bt["family_id"] = fid; branch_reps.append(bt)
    rep["gate2_twin"] = {"node": node_reps, "branch": branch_reps,
                         "summary": {"decode_twin_rate": float(np.mean([t["decode_twin_rate"] for t in node_reps])),
                                     "decode_xview_rate": float(np.mean([t["decode_xview_rate"] for t in node_reps])),
                                     "node_ablated_max": max(t["ablated_sep_max"] for t in node_reps),
                                     "branch_sep_min": min(b["sep_min"] for b in branch_reps),
                                     "branch_ablated_max": max(b["ablated_sep_max"] for b in branch_reps),
                                     "branch_seps": [b["sep_min"] for b in branch_reps]}}
    s2 = rep["gate2_twin"]["summary"]
    print(f"  ②并联支路 sep_min {s2['branch_sep_min']:.4f} (各族 {[round(x,4) for x in s2['branch_seps']]}); 通道置零 max {s2['branch_ablated_max']:.2e}; "
          f"节点可解码 {s2['decode_twin_rate']:.3f} / 跨视图 {s2['decode_xview_rate']:.3f}", flush=True)

    # ③ 迁移：V/θ 恢复、L2 参照、技巧分
    gdev = torch.device("cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if xfer:
        enc.to(gdev); dec.to(gdev)
        tr = transfer_report(enc, dec, val, xfer, cfg, norm, dev=gdev, in_raw=in_raw, out_raw=out_raw)
        rep["gate3_transfer"] = tr
        print("  ③迁移\n" + format_report(tr), flush=True)
    # ④ 固定掩码重构损失
    enc.to(gdev); dec.to(gdev)
    rep["eval_loss"] = {"in": eval_loss(enc, dec, val, cfg, dev=gdev), "out": eval_loss(enc, dec, xfer, cfg, dev=gdev) if xfer else None}
    print(f"  ④固定掩码重构损失 in {rep['eval_loss']['in']:.4f} out {rep['eval_loss']['out']}", flush=True)
    Path(a.out_json).write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"[probe:{a.label}] 报告 {a.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
