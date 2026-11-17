"""[G] 判稳三种子 ensemble 评估(R1-①;纯评估零烧机,GPU 仅推理)。

**前提 = 共享固定留出**(ts_reasoner --holdout-seed):此前各种子留出互异,跨种子概率
平均没有无泄漏验证口径(甲的留出是乙的训练集),ensemble 这条便宜增益一直用不了。

口径:逐族(禁混池)在共享 [内] held 上报——单种子 vs 三种子概率均值 ensemble 的
accuracy / 失稳召回 / 精度 / PR-AUC / 判别力(PR-AUC÷基率),温度各自在 held 上拟合
(与 calibrate_ts 同款;温度单调不改排序,只改刻度)。

  python scripts/ts_ensemble_eval.py --root <归档gridfm-data> --labels-root <归档runs> \
      --runs runs/ts_reasoner_r1_s0 runs/ts_reasoner_r1_s1 runs/ts_reasoner_r1_s2 \
      --cct <归档>/ts_cct_cepri36 <归档>/ts_cct_ieee118 --out runs/ts_ensemble_r1.json
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.reason.ts_task import average_precision  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402

_s = importlib.util.spec_from_file_location("tsr", Path(__file__).resolve().parent / "ts_reasoner.py")
tsr = importlib.util.module_from_spec(_s)
_s.loader.exec_module(tsr)
_c = importlib.util.spec_from_file_location("cts", Path(__file__).resolve().parent / "calibrate_ts.py")
cts = importlib.util.module_from_spec(_c)
_c.loader.exec_module(cts)

FAMS = ("cepri36", "ieee118")


def _metrics(p, y):
    pred = p > 0.5
    tp = int((pred & (y > 0.5)).sum())
    fp = int((pred & (y < 0.5)).sum())
    fn = int((~pred & (y > 0.5)).sum())
    base = float(y.mean())
    ap = average_precision(y, p)
    return {"n": int(len(y)), "base_rate": round(base, 4),
            "accuracy": round(float((pred == (y > 0.5)).mean()), 4),
            "unstable_recall": round(tp / max(tp + fn, 1), 4),
            "unstable_precision": round(tp / max(tp + fp, 1), 4),
            "pr_auc": round(float(ap), 4),
            "pr_auc_over_base": round(float(ap / base), 4) if base else None,
            "pred_pos_rate": round(float(pred.mean()), 4)}


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--root", required=True)
    ap_.add_argument("--labels-root", required=True)
    ap_.add_argument("--runs", nargs="+", required=True)
    ap_.add_argument("--cct", nargs="*", default=[])
    ap_.add_argument("--v1-dir", default="runs/encoder_v1")
    ap_.add_argument("--device", default="cuda")
    ap_.add_argument("--out", required=True)
    a = ap_.parse_args()

    ck = torch.load(Path(a.v1_dir) / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in allowed})
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    norm = Normalizer.load(Path(a.v1_dir) / "normalizer.npz")

    report = {"runs": [Path(r).name for r in a.runs], "by_family": {}, "protocol":
              "共享固定留出(--holdout-seed)下的三种子概率均值 ensemble;逐族禁混池;"
              "温度各自在 held 拟合(单调,不改排序)"}
    held_ref, exs_ref, probs = None, None, []
    cache = {}
    for run in a.runs:
        run = Path(run)
        rep, enc, reasoner = cts._build(run, ck, cfg, dev)
        batches = tuple(sorted(b["batch"] for b in rep["coverage"]["batches"]))
        if batches not in cache:
            e, raw, _cov = tsr.load_examples(Path(a.root), Path(a.labels_root),
                                             list(a.cct), only_batches=list(batches))
            normed = {}
            for sid, smp in raw.items():
                c = copy.deepcopy(smp)
                norm.transform(c)
                normed[sid] = c
            cache[batches] = (e, raw, normed)
        exps, raw_by_sid, normed = cache[batches]
        held = rep["split"]["held_ids"]
        if held_ref is None:
            held_ref = held
        elif held != held_ref:
            raise SystemExit("[ens] ✋ 各 run 的留出集不一致 → 无干净 ensemble 口径"
                             "(须用 --holdout-seed 训练同一留出)")
        held_set = {(f, s) for f, ids in held.items() for s in ids}
        in_held = [x for x in exps if x["family"] in rep["split"]["train_families"]
                   and (x["family"], x["sample_id"]) in held_set]
        if exs_ref is None:
            exs_ref = [x["tag"] for x in in_held]
        elif [x["tag"] for x in in_held] != exs_ref:
            raise SystemExit("[ens] ✋ 留出例集顺序/内容不一致")
        lg = cts._logits(enc, reasoner, in_held, normed, raw_by_sid, cfg, dev)
        y = np.array([x["y_unstable"] for x in in_held], dtype=np.float64)
        T = cts.fit_temperature(lg, y)
        probs.append(cts._sig(lg / T))
        print(f"[ens] {run.name} T={T:.3f}", flush=True)
        del enc, reasoner
        torch.cuda.empty_cache()

    fam_of = {x["tag"]: x["family"] for x in in_held}
    y_all = np.array([x["y_unstable"] for x in in_held], dtype=np.float64)
    pm = np.mean(probs, axis=0)
    for fam in FAMS:
        idx = [i for i, t in enumerate(exs_ref) if fam_of[t] == fam]
        if not idx:
            continue
        y = y_all[idx]
        singles = [_metrics(p[idx], y) for p in probs]
        ens = _metrics(pm[idx], y)
        report["by_family"][fam] = {
            "singles": singles, "ensemble": ens,
            "delta_acc_vs_mean_single": round(
                ens["accuracy"] - float(np.mean([m["accuracy"] for m in singles])), 4),
            "delta_prauc_vs_mean_single": round(
                ens["pr_auc"] - float(np.mean([m["pr_auc"] for m in singles])), 4)}
        ms = [f"{m['accuracy']:.4f}" for m in singles]
        print(f"[ens] {fam}: 单种子 acc {ms} 均 "
              f"{np.mean([m['accuracy'] for m in singles]):.4f} → ensemble "
              f"{ens['accuracy']:.4f}(Δ{report['by_family'][fam]['delta_acc_vs_mean_single']:+.4f});"
              f" PR-AUC Δ{report['by_family'][fam]['delta_prauc_vs_mean_single']:+.4f}", flush=True)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[ens] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
