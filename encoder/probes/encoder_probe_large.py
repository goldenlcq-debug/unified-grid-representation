#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结编码器 v1（seed0）在大网（2,000–70,000 母线）上的表征探针（2026-09-05，长卿；任务书-编码器大网表征探针补测）。
与论文 R1-1 / R1-2 / R1-3 同口径，函数与 scripts/p1_full_report.py 同源，只把数据换成指定网的已解算方式：
  ① 母线寻址：同一方式前向两次、第二次重抽节点与支路随机通道，余弦 top-1 命中率（大网分块算相似度）；随机匹配参考 = 1/有效母线数；
  ② 并联同型支路：在该网**真实存在**的同端点、同类型、同 r/x/b、同投退状态的并联支路对上量支路 token 余弦距离（最小值）；随机通道置零后的距离作机制消融；没有此类支路对则报"无"；
  ③ V/θ 恢复：state_field_mae（固定掩码协议）+ 图调和插值 L2 基线（ladder_baselines）+ 技巧分 = L2 / 模型；
  ④ 相对源网倍数：以 probe_legacy_reproduction.json（同环境下旧冻结验证集 404 例）的 V/θ MAE 为分母。
用法：
  python encoder_probe_large.py --grid ACTIVSg25k --root <dir> [--ids a,b,...|--limit N] [--retrieval-n 20] --label x --out-json y.json
--root 可以是含 manifest.csv 的方式库目录，或 families/<split>/<fam> 目录；npz 按 manifest 顺序读取。
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from dataclasses import fields as dc_fields, replace as dc_replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1] / "grid-foundation"   # 包内副本（= 原 D 盘仓 grid-foundation 的 gridfm，逐文件 md5 见 04_核验）
PKG = Path(__file__).resolve().parents[2]   # 包根；编码器在 PKG/02_编码器（--v1-dir 缺省已改为该目录）
sys.path.insert(0, str(REPO))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402
from gridfm.data.graph import sample_to_data  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.eval.transfer import state_field_mae, ladder_baselines  # noqa: E402
from gridfm.probes.retrieval import _valid_node_mask  # noqa: E402
from gridfm.schema.fields import BRANCH_TABLE, NODE_TABLE, BRANCH_TYPES  # noqa: E402
from gridfm.schema.sample import GridSample  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402

HERE = Path(__file__).resolve().parent


def load_encoder(v1_dir: Path, dev):
    ck = torch.load(v1_dir / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in allowed})
    enc, dec = build_models(cfg)
    enc.load_state_dict(ck["model_state"]); dec.load_state_dict(ck["decoder_state"])
    enc.to(dev).eval(); dec.to(dev).eval()
    return enc, dec, cfg, Normalizer.load(v1_dir / "normalizer.npz"), ck.get("global_step")


def load_samples(root: Path, ids=None, limit=0):
    man = root / "manifest.csv"
    if man.exists():
        rows = list(csv.DictReader(open(man, encoding="utf-8")))
        names = [r["sample_id"] for r in rows if r.get("solvable", "1").strip() != "0"]
    else:
        names = [p.stem for p in sorted(root.glob("*.npz")) if p.name != "family_extra.npz"]
    if ids:
        keep = set(ids); names = [n for n in names if n in keep]
    if limit:
        names = names[:limit]
    out = []
    for n in names:
        g = GridSample.load_npz(root / f"{n}.npz")
        # PEGASE 池的 2–4 个机组节点在动态字段 tj / xdp 上是 NaN（转换器无动态数据；该字段已标 missing）。
        # NaN 会让整张图的前向变成 NaN；缺失字段归一化后本就取 0，故非有限值置 0，并记账。
        bad_n = int((~np.isfinite(g.node_feat)).sum()); bad_b = int((~np.isfinite(g.branch_feat)).sum())
        if bad_n or bad_b:
            g.node_feat = np.nan_to_num(g.node_feat, nan=0.0, posinf=0.0, neginf=0.0)
            g.branch_feat = np.nan_to_num(g.branch_feat, nan=0.0, posinf=0.0, neginf=0.0)
            SANITIZED.append((g.sample_id, bad_n, bad_b))
        out.append(g)
    return out


SANITIZED = []


@torch.no_grad()
def encode_tokens(enc, sample, cfg, dev, rf=None, brf=None, ablate=False):
    """返回 (节点 token [N,d], 支路 token [2M,d]) —— 前 M 行为正向边（与 twins._encode 同约定）。"""
    kw = dict(random_dim=cfg.random_dim, branch_random_dim=cfg.branch_random_dim)
    if rf is not None:
        kw["random_feat"] = rf
    if brf is not None:
        kw["branch_random_feat"] = brf
    data = sample_to_data(sample, cfg.lape_k, cfg.rwse_steps, **kw)
    if ablate:  # 随机通道置零（与 twins._encode(ablate=True) 同义）：节点 / 支路特征末 random_dim 列
        data.x[:, -cfg.random_dim:] = 0
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr[:, -cfg.branch_random_dim:] = 0
    batch = Batch.from_data_list([data]).to(dev)
    h, _g, e = enc(batch)
    return h.float(), (e.float() if e is not None else None)


@torch.no_grad()
def retrieval_top1(enc, sample, cfg, dev, rng, chunk=4096):
    n, m = sample.n_nodes, sample.n_branches
    rf1 = rng.standard_normal((n, cfg.random_dim)).astype(np.float32)
    rf2 = rng.standard_normal((n, cfg.random_dim)).astype(np.float32)
    brf1 = rng.standard_normal((m, cfg.branch_random_dim)).astype(np.float32)
    brf2 = rng.standard_normal((m, cfg.branch_random_dim)).astype(np.float32)
    t1, _ = encode_tokens(enc, sample, cfg, dev, rf1, brf1)
    t2, _ = encode_tokens(enc, sample, cfg, dev, rf2, brf2)
    t1 = torch.nn.functional.normalize(t1, dim=-1); t2 = torch.nn.functional.normalize(t2, dim=-1)
    top1 = torch.empty(n, dtype=torch.long, device=dev)
    for s in range(0, n, chunk):
        top1[s:s + chunk] = (t1[s:s + chunk] @ t2.T).argmax(dim=1)
    top1 = top1.cpu().numpy()
    valid = _valid_node_mask(sample); idx = np.where(valid)[0]
    return float((top1[idx] == idx).mean()), int(idx.size)


def find_parallel_identical_pairs(sample, tol=1e-6):
    """同端点（无向）、同 branch_type、同 r/x/b_half、同 mark 的支路对（自环除外）。返回 [(a, b), ...] 索引对（正向边行号）。"""
    boff = BRANCH_TABLE.offset_of
    bf = sample.branch_feat; ei = sample.edge_index; m = sample.n_branches
    r, x, b, mk = bf[:m, boff("r")], bf[:m, boff("x")], bf[:m, boff("b_half")], bf[:m, boff("mark")]
    bt = np.argmax(bf[:m, boff("branch_type"):boff("branch_type") + len(BRANCH_TYPES)], axis=1)
    groups = {}
    for e in range(m):
        i, j = int(ei[0, e]), int(ei[1, e])
        if i == j:
            continue
        key = (min(i, j), max(i, j), int(bt[e]), int(mk[e] > 0.5))
        groups.setdefault(key, []).append(e)
    pairs = []
    for key, es in groups.items():
        if len(es) < 2 or key[3] != 1:
            continue
        for a_i in range(len(es)):
            for b_i in range(a_i + 1, len(es)):
                a, b2 = es[a_i], es[b_i]
                if np.allclose(bf[a], bf[b2], atol=tol, rtol=0):      # 全部支路字段逐位相同（含 r/x/b、投退、潮流），才是"电气上完全相同"的并联对
                    pairs.append((a, b2))
    return pairs


@torch.no_grad()
def parallel_separation(enc, sample, cfg, dev, pairs):
    _h, e = encode_tokens(enc, sample, cfg, dev)
    e = torch.nn.functional.normalize(e, dim=-1)
    seps = [float(1.0 - (e[a] * e[b]).sum()) for a, b in pairs]
    _h0, e0 = encode_tokens(enc, sample, cfg, dev, ablate=True)
    e0 = torch.nn.functional.normalize(e0, dim=-1)
    abl = [float(1.0 - (e0[a] * e0[b]).sum()) for a, b in pairs]
    return seps, abl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retrieval-n", type=int, default=20)
    ap.add_argument("--pairs-samples", type=int, default=3, help="并联支路可分在前几个方式上量（取最小）")
    ap.add_argument("--v1-dir", default=str(Path(__file__).resolve().parents[2] / "02_编码器"))
    ap.add_argument("--source-json", default=str(Path(__file__).resolve().parents[2] / "00_结果" / "probe_legacy_reproduction.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label", default="")
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu")
    enc, dec, cfg, norm, step = load_encoder(Path(a.v1_dir), dev)
    ids = a.ids.split(",") if a.ids else None
    raws = load_samples(Path(a.root), ids, a.limit)
    if not raws:
        raise SystemExit("no samples")
    n_nodes = int(np.mean([s.n_nodes for s in raws]))
    print(f"[large:{a.grid}] {len(raws)} 例, 节点≈{n_nodes}, encoder step {step}, dev {dev}", flush=True)
    rep = {"grid": a.grid, "label": a.label, "root": a.root, "n_samples": len(raws), "n_nodes_mean": n_nodes, "encoder_step": step,
           "sanitized_nonfinite": {"n_samples": len(SANITIZED), "entries": SANITIZED[:5]}}
    if SANITIZED:
        print(f"  [注] {len(SANITIZED)} 个样本含非有限值（动态字段 tj/xdp，已标 missing），已置 0：例 {SANITIZED[0]}", flush=True)
    t0 = time.time()
    # 与 p1_full_report 同口径：①②③ 的模型输入都是**归一化后**的样本（gate1/gate2 用 val = prep(val_raw)）；阶梯基线用原始样本
    normed = []
    for s in raws:
        c = copy.deepcopy(s); norm.transform(c); normed.append(c)
    # ① 寻址
    accs, nvalid = [], []
    for i, s in enumerate(normed[:a.retrieval_n]):
        acc, nv = retrieval_top1(enc, s, cfg, dev, np.random.default_rng(i))
        accs.append(acc); nvalid.append(nv)
    rep["retrieval"] = {"mean_top1": float(np.mean(accs)), "min": float(np.min(accs)), "n_samples": len(accs),
                        "random_baseline": 1.0 / max(float(np.mean(nvalid)), 1.0), "per_sample": accs}
    print(f"  ①寻址 top-1 {rep['retrieval']['mean_top1']:.4f} (min {rep['retrieval']['min']:.4f}, n={len(accs)}, 基线 {rep['retrieval']['random_baseline']:.2e}) {time.time()-t0:.0f}s", flush=True)
    # ② 并联同型支路
    pair_stats = []
    for s_raw, s in zip(raws[:a.pairs_samples], normed[:a.pairs_samples]):
        pairs = find_parallel_identical_pairs(s_raw)
        if not pairs:
            pair_stats.append({"sample_id": s.sample_id, "n_pairs": 0}); continue
        seps, abl = parallel_separation(enc, s, cfg, dev, pairs)
        pair_stats.append({"sample_id": s.sample_id, "n_pairs": len(pairs), "sep_min": float(min(seps)), "sep_median": float(np.median(seps)),
                           "ablated_max": float(max(abl))})
    have = [p for p in pair_stats if p["n_pairs"]]
    rep["parallel_pairs"] = {"per_sample": pair_stats,
                             "n_pairs": (have[0]["n_pairs"] if have else 0),
                             "sep_min": (min(p["sep_min"] for p in have) if have else None),
                             "ablated_max": (max(p["ablated_max"] for p in have) if have else None)}
    print(f"  ②并联同型支路对 {rep['parallel_pairs']['n_pairs']} 对; sep_min {rep['parallel_pairs']['sep_min']}; 置零后 max {rep['parallel_pairs']['ablated_max']}", flush=True)
    # ③ V/θ 恢复（逐样本 batch=1，防大网 OOM）
    cfg1 = dc_replace(cfg, batch_graphs=1)
    mae = state_field_mae(enc, dec, normed, cfg1, norm, dev)
    lad = ladder_baselines(raws, seed=cfg.seed)
    rep["recovery"] = {"mae_pu": mae, "ladder": lad,
                       "skill": {f: (lad[f]["L2"] / mae[f] if mae.get(f) and np.isfinite(mae[f]) and mae[f] > 0 else None) for f in ("v", "theta")}}
    def _f(x, nd=5):
        return "nan" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"
    sk = rep["recovery"]["skill"]
    print(f"  ③恢复 V {_f(mae['v'])} (L2 {_f(lad['v']['L2'])}, 技巧 {_f(sk['v'], 2)}) | θ {_f(mae['theta'], 4)} (L2 {_f(lad['theta']['L2'], 4)}, 技巧 {_f(sk['theta'], 3)}) | pg {_f(mae['pg'], 4)} qg {_f(mae['qg'], 4)} {time.time()-t0:.0f}s", flush=True)
    # ④ 相对源网倍数
    src = json.loads(Path(a.source_json).read_text(encoding="utf-8"))["gate3_transfer"]["mae_in_pu"]
    rep["ratio_vs_source"] = {f: (mae[f] / src[f]) for f in ("v", "theta", "pg", "qg")}
    rep["source_mae_pu"] = src
    print(f"  ④相对源网倍数 V {_f(rep['ratio_vs_source']['v'], 2)}× θ {_f(rep['ratio_vs_source']['theta'], 2)}×", flush=True)
    rep["secs"] = round(time.time() - t0, 1)
    Path(a.out_json).write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"[large:{a.grid}] 报告 {a.out_json}", flush=True)


if __name__ == "__main__":
    main()
