"""判稳收尾评测器(B2/B5,2026-07-17 过账;纯评估零烧机,GPU 仅推理)。

两用:
1. 批评测(B2 定向补测集挂门):--batches 指定标签批,eval_ts 全量口径出子类/类型/判稳
   指标。--with-held-family 可把该 run 自己的 [内] 留出例并入(= "留出方式自然+定向合并"
   口径,给 500 例报告门用;定向口径单独挂门,永不与自然分布口径混算——报告里两者分列)。
2. 机群头逐机诊断(B5):--group-diag 在该 run 自己的 [内] 留出集上逐发电机统计 FP/FN,
   并按真实机群规模分桶报微平均 F1(118 低分的错误集中度定位)。

用法:
  python scripts/ts_eval_batch.py --root ..\\gridfm-data \\
      --runs runs/ts_reasoner_v5w_s0 ... --batches ts_labels_cepri36_sub25 ... \\
      --with-held-family cepri36 --out runs/b2_sub_eval.json
  python scripts/ts_eval_batch.py --root ..\\gridfm-data \\
      --runs runs/ts_reasoner_v5_base_s0 ... --group-diag --out runs/b5_group_diag.json
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.reason.ts_task import (  # noqa: E402
    EVENT_PARAM_DIM, TSReasoner, fault_edge_pointer,
)
from gridfm.reason.ts_encoder_io import build_encoder_for_run  # noqa: E402
from gridfm.train.pretrain import TrainConfig  # noqa: E402

_spec = importlib.util.spec_from_file_location("tsr", Path(__file__).resolve().parent / "ts_reasoner.py")
tsr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tsr)


def _build(run: Path, ck, cfg, dev):
    """与 calibrate_ts 同款:编码器走公共装载口(R2 第 7 条)+ run 的 best.pt 推理器。"""
    enc, rep, best, _src = build_encoder_for_run(run, ck, cfg, dev=dev)
    cf = best.get("config", {})
    reasoner = TSReasoner(d=cf.get("d", cfg.d), n_queries=cf.get("queries", 16),
                          n_layers=cf.get("layers", 2), event_dim=EVENT_PARAM_DIM)
    reasoner.load_state_dict(best["reasoner_state"])
    reasoner.eval().to(dev)
    return rep, enc, reasoner


def _load(root, labels_root, batches, norm, cache):
    key = tuple(sorted(batches))
    if key not in cache:
        exps, raw_by_sid, _cov = tsr.load_examples(Path(root), Path(labels_root), [],
                                                   only_batches=list(key))
        normed = {}
        for sid, smp in raw_by_sid.items():
            c = copy.deepcopy(smp)
            norm.transform(c)
            normed[sid] = c
        cache[key] = (exps, raw_by_sid, normed)
    return cache[key]


@torch.no_grad()
def group_diag(enc, reasoner, exs, normed, raw, cfg, dev, bs=32):
    """逐发电机 FP/FN + 按真实机群规模分桶(仅 group_valid 图)。"""
    per_bus, per_size = {}, {}
    for i in range(0, len(exs), bs):
        chunk = [x for x in exs[i:i + bs]]
        datas = [tsr._ts_data(x, normed[x["sample_id"]], raw[x["sample_id"]], cfg)
                 for x in chunk]
        batch = Batch.from_data_list(datas).to(dev)
        h, g, e = enc(batch)
        eb = batch.batch[batch.edge_index[0]]
        ptr = fault_edge_pointer(eb, batch.fault_row)
        _s, _t, _c, gr = reasoner(h, g, e, batch.batch, eb, ptr, batch.event_params)
        pred = (torch.sigmoid(gr.float()) > 0.5).cpu().numpy()
        true = (batch.y_group > 0.5).cpu().numpy()
        gv = batch.group_valid.cpu().numpy()
        gm = batch.gen_mask.cpu().numpy()
        bidx = batch.batch.cpu().numpy()
        starts = batch.ptr.cpu().numpy()
        for gi, ex in enumerate(chunk):
            if not gv[gi]:
                continue
            rows = np.where((bidx == gi) & gm)[0]
            fam = ex["family"]
            tsize = int(true[rows].sum())
            for r in rows:
                bus = int(r - starts[gi]) + 1          # 节点行 → 母线号(group_multi_hot 逆映射)
                st = per_bus.setdefault((fam, bus), {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
                p, t = bool(pred[r]), bool(true[r])
                st["tp" if (p and t) else "fp" if p else "fn" if t else "tn"] += 1
            key = (fam, min(tsize, 12))
            ss = per_size.setdefault(key, {"tp": 0, "fp": 0, "fn": 0, "n_graphs": 0})
            ss["n_graphs"] += 1
            for r in rows:
                p, t = bool(pred[r]), bool(true[r])
                if p and t:
                    ss["tp"] += 1
                elif p:
                    ss["fp"] += 1
                elif t:
                    ss["fn"] += 1
    def f1(d):
        return round(2 * d["tp"] / max(2 * d["tp"] + d["fp"] + d["fn"], 1), 4)
    return ({f"{f}|bus{b}": {**v, "f1": f1(v)} for (f, b), v in sorted(per_bus.items())},
            {f"{f}|size{s}": {**v, "f1": f1(v)} for (f, s), v in sorted(per_size.items())})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels-root", default="runs")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--batches", nargs="*", default=[],
                    help="要评测的标签批(留空 + --group-diag = 用 run 自身粮仓)")
    ap.add_argument("--sids-json", default="", help="方式名单 JSON(限定评测方式)")
    ap.add_argument("--with-held-family", default="",
                    help="并入该 run 自己 [内] 留出集的指定家族例(合并口径给报告门)")
    ap.add_argument("--group-diag", action="store_true")
    ap.add_argument("--v1-dir", default="runs/encoder_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ck = torch.load(Path(a.v1_dir) / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in allowed})
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    norm = Normalizer.load(Path(a.v1_dir) / "normalizer.npz")
    use_amp = dev.type == "cuda"
    sids = None
    if a.sids_json:
        d = json.loads(Path(a.sids_json).read_text(encoding="utf-8"))
        sids = set(d if isinstance(d, list) else d.get("sample_ids", []))

    report, cache = {"runs": {}}, {}
    for run in a.runs:
        run = Path(run)
        rep, enc, reasoner = _build(run, ck, cfg, dev)
        own_batches = sorted(b["batch"] for b in rep["coverage"]["batches"])
        rr = {}
        if a.group_diag:
            exps, raw, normed = _load(a.root, a.labels_root, own_batches, norm, cache)
            held = rep["split"]["held_ids"]
            held_set = {(f, s) for f, ids in held.items() for s in ids}
            in_held = [x for x in exps if (x["family"], x["sample_id"]) in held_set]
            pb, ps = group_diag(enc, reasoner, in_held, normed, raw, cfg, dev)
            rr["group_per_bus"] = pb
            rr["group_per_size"] = ps
            worst = sorted((v["f1"], k) for k, v in pb.items()
                           if k.startswith("ieee118") and (v["tp"] + v["fn"]) >= 20)[:8]
            print(f"[{run.name}] 118 逐机 F1 最差 8 台:" +
                  " ".join(f"{k}={f:.2f}" for f, k in worst))
        if a.batches:
            exps, raw, normed = _load(a.root, a.labels_root, a.batches, norm, cache)
            exs = [x for x in exps if sids is None or x["sample_id"] in sids]
            tag = "+".join(a.batches)
            m = tsr.eval_ts(enc, reasoner, exs, normed, raw, cfg, dev, use_amp,
                            torch.float16, rep["weights"])
            rr[f"batch_eval[{tag}]"] = m
            if a.with_held_family:
                exps0, raw0, normed0 = _load(a.root, a.labels_root, own_batches, norm, cache)
                held = rep["split"]["held_ids"]
                held_set = {(f, s) for f, ids in held.items() for s in ids}
                fam_held = [x for x in exps0 if x["family"] == a.with_held_family
                            and (x["family"], x["sample_id"]) in held_set]
                merged = fam_held + exs
                raw_m = {**raw0, **raw}
                normed_m = {**normed0, **normed}
                mm = tsr.eval_ts(enc, reasoner, merged, normed_m, raw_m, cfg, dev, use_amp,
                                 torch.float16, rep["weights"])
                rr[f"merged_eval[{a.with_held_family}_held+{tag}]"] = mm
                print(f"[{run.name}] 定向批 n={len(exs)} | 合并口径 n={len(merged)}"
                      f" | f_low n={mm.get('sub_f_low_n')} recall={mm.get('sub_f_low_recall')}"
                      f" | v_over n={mm.get('sub_v_over_n')} recall={mm.get('sub_v_over_recall')}")
        report["runs"][run.name] = rr
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[eval] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
