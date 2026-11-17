"""P1 校准修复(计划 v3 P1;纯评估,GPU 仅推理,零烧机)。

问题:判稳推理器在 [外](未见拓扑)上预测正类率 0.97~1.00(饱和),0.5 阈值下游
(8.7 批扫粗筛/稳控触发)全废。判别力(排序)尚存,坏的是概率**刻度**。

做法(温度缩放,单调 → 不改排序/判别力,只改校准):p = sigmoid(logit / T)。
- **T 只在 [内] 留出集上拟合**(最小化 NLL),**严禁碰 [外]**(D11/铁律);拟合后迁到 [外]。
- 报:ECE(校准误差)、预测正类率、Brier、0.5 阈值准确率——[外] 标定前(T=1)vs 后(T*)。
- 另报 [外] 的**oracle T**(在 [外] 上直接拟合的 T,不可用于生产,仅诊断"[内]标定能迁移多少")。
- [内] 留出方式 = ts_report.json 的 split.held_ids(逐族);[外] = ieee39 整族。

分位阈值(备口径):在 [内] 上取"预测正类率 = 基率"的分位阈,迁到 [外] 报阈值化判别力。

用法:python scripts/calibrate_ts.py --root ..\\gridfm-data \\
        --runs runs/ts_reasoner_v3_C_full_s0 runs/ts_reasoner_v3_C_full_s1 ... \\
        --out runs/calib_ts.json
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch_geometric.data import Batch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.reason.ts_task import (  # noqa: E402
    EVENT_PARAM_DIM, TSReasoner, average_precision, fault_edge_pointer,
)
from gridfm.reason.ts_encoder_io import build_encoder_for_run  # noqa: E402
from gridfm.train.pretrain import TrainConfig  # noqa: E402

_spec = importlib.util.spec_from_file_location("tsr", Path(__file__).resolve().parent / "ts_reasoner.py")
tsr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tsr)


@torch.no_grad()
def _logits(enc, reasoner, exs, normed_by_sid, raw_by_sid, cfg, dev) -> np.ndarray:
    out = []
    for i in range(0, len(exs), 32):
        chunk = exs[i:i + 32]
        datas = [tsr._ts_data(x, normed_by_sid[x["sample_id"]], raw_by_sid[x["sample_id"]], cfg)
                 for x in chunk]
        batch = Batch.from_data_list(datas).to(dev)
        h, g, e = enc(batch)
        eb = batch.batch[batch.edge_index[0]]
        ptr = fault_edge_pointer(eb, batch.fault_row)
        sl, _t, _c, _gr = reasoner(h, g, e, batch.batch, eb, ptr, batch.event_params,
                                   ablate_graph=False)
        out += sl.float().cpu().tolist()
    return np.array(out, dtype=np.float64)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """最小化二元 NLL 拟合温度 T(T>0);单调,不改排序。"""
    def nll(logT):
        p = _sig(logits / np.exp(logT))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(50.0)), method="bounded")
    return float(np.exp(r.x))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """期望校准误差(等宽分箱,|bin 内平均预测 − 平均真实| 按样本加权)。"""
    edges = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(p)
    for b in range(n_bins):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < n_bins - 1 else p <= edges[b + 1])
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def metrics(logits: np.ndarray, y: np.ndarray, T: float) -> dict:
    p = _sig(logits / T)
    base = float(y.mean())
    pred = p > 0.5
    tp = int((pred & (y > 0.5)).sum()); fp = int((pred & (y < 0.5)).sum())
    ap = average_precision(y, p)
    return {"T": round(T, 3), "ece": round(ece(p, y), 4),
            "brier": round(float(((p - y) ** 2).mean()), 4),
            "pred_pos_rate": round(float(pred.mean()), 4),
            "acc": round(float((pred == (y > 0.5)).mean()), 4),
            "unstable_precision": round(tp / max(tp + fp, 1), 4),
            "pr_auc_over_base": round(float(ap / base), 4) if base else None,
            "base_rate": round(base, 4), "n": int(len(y))}


def quantile_threshold(logits_in: np.ndarray, y_in: np.ndarray) -> float:
    """[内] 上取'预测正类率 = 基率'的概率分位阈(备口径,分位阈化)。"""
    p = _sig(logits_in)   # 未温度化的原始概率
    base = float(y_in.mean())
    return float(np.quantile(p, 1.0 - base))


def recall_cost(scores: np.ndarray, y: np.ndarray, recalls=(0.90, 0.95, 0.99)) -> dict:
    """8.7 批扫粗筛口径(P1.2):按分数降序复核 top-K,达目标失稳召回所需复核比例。

    复核比例越低越省(1−复核比例 = 可安全跳过 PSASP 的比例)。温度/阈值不改此曲线
    (纯排序口径)。"""
    order = np.argsort(-scores, kind="stable")
    ys = y[order].astype(np.float64)
    tot = ys.sum()
    if tot <= 0:
        return {R: None for R in recalls}
    cum = np.cumsum(ys)
    out = {}
    for R in recalls:
        k = int(np.searchsorted(cum, R * tot) + 1)
        out[R] = round(k / len(y), 4)
    return out


def _build(run: Path, ck, cfg, dev, at_step=None):
    # R2 第 7 条:编码器出身一律走公共装载口(有 encoder_state 优先加载并打印来源;
    # 报告标解冻臂却无 encoder_state 则拒绝运行)。**本函数被 ts_screen / ts_ensemble_eval
    # 复用**,是 7 个消费方里覆盖面最大的一个。
    # at_step:改装该步的周期检查点(同步数配对出数);None = 历史行为逐位不变。
    enc, rep, best, _src = build_encoder_for_run(run, ck, cfg, dev=dev, at_step=at_step)
    cf = best.get("config", {})
    reasoner = TSReasoner(d=cf.get("d", cfg.d), n_queries=cf.get("queries", 16),
                          n_layers=cf.get("layers", 2), event_dim=EVENT_PARAM_DIM)
    reasoner.load_state_dict(best["reasoner_state"])
    reasoner.eval().to(dev)
    return rep, enc, reasoner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels-root", default="runs")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--v1-dir", default="runs/encoder_v1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/calib_ts.json")
    a = ap.parse_args()

    ck = torch.load(Path(a.v1_dir) / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in allowed})
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    norm = Normalizer.load(Path(a.v1_dir) / "normalizer.npz")

    report = {"runs": {}}
    cache = {}
    for run in a.runs:
        run = Path(run)
        rep, enc, reasoner = _build(run, ck, cfg, dev)
        batches = tuple(sorted(b["batch"] for b in rep["coverage"]["batches"]))
        held = rep["split"]["held_ids"]                      # 逐族 [内] 留出方式
        train_fams = rep["split"]["train_families"]
        if batches not in cache:
            exps, raw_by_sid, _cov = tsr.load_examples(
                Path(a.root), Path(a.labels_root), [], only_batches=list(batches))
            normed = {}
            for sid, smp in raw_by_sid.items():
                c = copy.deepcopy(smp); norm.transform(c); normed[sid] = c
            cache[batches] = (exps, raw_by_sid, normed)
        exps, raw_by_sid, normed = cache[batches]
        held_set = {(f, sid) for f, ids in held.items() for sid in ids}
        in_held = [x for x in exps if x["family"] in train_fams
                   and (x["family"], x["sample_id"]) in held_set]
        out_exs = [x for x in exps if x["family"] in tsr.OUT_FAMILIES]
        assert len(out_exs) == rep["split"]["n_out"], (len(out_exs), rep["split"]["n_out"])
        assert len(in_held) == rep["split"]["n_in"], (len(in_held), rep["split"]["n_in"])

        li = _logits(enc, reasoner, in_held, normed, raw_by_sid, cfg, dev)
        yi = np.array([x["y_unstable"] for x in in_held], dtype=np.float64)
        lo = _logits(enc, reasoner, out_exs, normed, raw_by_sid, cfg, dev)
        yo = np.array([x["y_unstable"] for x in out_exs], dtype=np.float64)

        T = fit_temperature(li, yi)                          # [内] 拟合温度
        T_oracle = fit_temperature(lo, yo)                   # [外] oracle(仅诊断)
        rr = {
            "T_fit_on_in": round(T, 3), "T_oracle_out_diagnostic": round(T_oracle, 3),
            "in_held_before": metrics(li, yi, 1.0), "in_held_after": metrics(li, yi, T),
            "out_before": metrics(lo, yo, 1.0),
            "out_after_in_T": metrics(lo, yo, T),
            "out_after_oracle_T": metrics(lo, yo, T_oracle),
            "recall_cost_out": recall_cost(_sig(lo), yo),    # P1.2 批扫粗筛(排序口径)
            "recall_cost_in_held": recall_cost(_sig(li), yi),
        }
        report["runs"][run.name] = rr
        b, af, orc = rr["out_before"], rr["out_after_in_T"], rr["out_after_oracle_T"]
        print(f"\n=== {run.name}  T([内]拟合)={T:.2f}  T(oracle[外])={T_oracle:.2f} ===")
        print(f"  [外] 标定前 : ECE {b['ece']:.3f} | 预测正类率 {b['pred_pos_rate']:.3f} "
              f"| 0.5准确率 {b['acc']:.3f} | 判别力 {b['pr_auc_over_base']}")
        print(f"  [外] [内]T后: ECE {af['ece']:.3f} | 预测正类率 {af['pred_pos_rate']:.3f} "
              f"| 0.5准确率 {af['acc']:.3f} | 判别力 {af['pr_auc_over_base']}(排序不变)")
        print(f"  [外] oracleT: ECE {orc['ece']:.3f} | 预测正类率 {orc['pred_pos_rate']:.3f} "
              f"| 0.5准确率 {orc['acc']:.3f}(上界:[外]自拟合能到多好)")
        rc, rci = rr["recall_cost_out"], rr["recall_cost_in_held"]
        print(f"  批扫粗筛 达 99%/95%/90% 失稳召回所需复核比例: "
              f"[外] {rc[0.99]}/{rc[0.95]}/{rc[0.90]}  vs  [内] {rci[0.99]}/{rci[0.95]}/{rci[0.90]}")

    # 3 种子聚合
    if len(a.runs) > 1:
        def col(path):
            vals = [report["runs"][Path(r).name] for r in a.runs]
            for k in path:
                vals = [v[k] for v in vals]
            return vals
        print("\n=== 3 种子聚合([外]) ===")
        for lab, key in (("标定前 ECE", ["out_before", "ece"]),
                         ("[内]T后 ECE", ["out_after_in_T", "ece"]),
                         ("标定前 预测正类率", ["out_before", "pred_pos_rate"]),
                         ("[内]T后 预测正类率", ["out_after_in_T", "pred_pos_rate"]),
                         ("[内]拟合 T", ["T_fit_on_in"])):
            v = np.array(col(key), dtype=float)
            print(f"  {lab:18s}: {v.round(3).tolist()}  均值 {v.mean():.3f}")
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[calib] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
