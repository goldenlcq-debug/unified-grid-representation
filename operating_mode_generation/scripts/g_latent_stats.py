"""[G] 潜码标准化统计(方案 6.3② 强制步骤;7.4 同等版本管理)。

对训练 240 方式(留出 60 不入,与 g15/诊断 A 同款切法)的**完整编码节点 token**
(扩散状态 z 场;用户裁决:z 收缩为节点 token 场,支路 token 只作条件)统计逐通道
mean/std,固化 npz;打印标准化后方差直方图作 sanity(应≈1;跳过此步 DDPM 噪声调度
会静默失配且与"扩散没训好"不可区分——6.3② 原文)。

产物:runs/latent_stats/latent_stats.npz(mean/std [d] + 版本元数据)+ 方差直方图
(控制台文本必打;matplotlib 可用时另存 png)。

  python scripts/g_latent_stats.py --root ..\\gridfm-data
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import gridfm  # noqa: E402
assert gridfm.__file__.startswith(str(REPO)), f"窜仓:{gridfm.__file__}"

from gridfm.data.dataset import load_library  # noqa: E402
from gridfm.data.graph import sample_to_data  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402

# 6.3② / 7.4:随检查点固化,升版须重训扩散。版本随语料口径走:
# single = lib118_diverse 单库(v0.x,latent_stats_v1);g4 = 多家族训练语料(G4,v2)。
# [外协包 F,2026-08-25] 新增 g4f 口径 = g4 语料 + 2000/3120 大网族。
# 为什么要单列一版:实测 3120 的节点潜码用 latent_stats_v2 标准化后逐通道方差只有 0.50
# (2000 为 0.93、训练语料 0.97),扩散假设状态为单位方差 → 3120 上信噪比错配一倍,
# 且该症状与"没训好"在损失曲线上不可区分。重算必须与重训扩散器配套(指纹三验会拦)。
STATS_VERSIONS = {"single": "latent_stats_v1", "g4": "latent_stats_v2", "g4f": "latent_stats_v3f"}


def _fp(path: Path) -> str:
    """就位三查同款指纹(BLAKE2b-128)。"""
    return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()


def _text_hist(values: np.ndarray, title: str, bins: int = 12, width: int = 48) -> None:
    lo, hi = float(values.min()), float(values.max())
    edges = np.linspace(lo, hi, bins + 1)
    cnt, _ = np.histogram(values, bins=edges)
    print(f"  {title}(n={values.size},min {lo:.4g} / max {hi:.4g}):")
    for k in range(bins):
        bar = "#" * int(round(width * cnt[k] / max(1, cnt.max())))
        print(f"    [{edges[k]:8.4f},{edges[k+1]:8.4f}) {cnt[k]:4d} {bar}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lib-name", default="lib118_diverse")
    ap.add_argument("--v1-dir", default="runs/encoder_v1")
    ap.add_argument("--out", default=None,
                    help="默认按语料分目录:single→runs/latent_stats,g4→runs/latent_stats_v2"
                         "(审查修复:两版共享编码器/normalizer 指纹,同目录同名会静默互覆)")
    ap.add_argument("--test-every", type=int, default=5, help="留出切法同 g1/g15,留出不入")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--corpus", choices=("single", "g4", "g4f"), default="single",
                    help="single=lib118_diverse 单库(v0.x 口径);"
                         "g4=多家族训练语料(corpus.py 单一出处,latent_stats_v2)")
    ap.add_argument("--device", default="cpu", help="编码前向设备(G4 语料 28× 建议 cuda)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(
        "runs/latent_stats_v2" if a.corpus == "g4" else "runs/latent_stats")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available())
                       else "cpu")

    lib = Path(a.root) / a.lib_name
    corpus_prov = None
    if a.corpus in ("g4", "g4f"):
        from gridfm.data.corpus import g4_train_corpus
        pool, corpus_prov = g4_train_corpus(Path(a.root), a.lib_name, a.test_every)
        print(f"[lstat] G4 语料:{corpus_prov['n_samples']} 样本 = "
              f"{corpus_prov['n_synth_families']} 合成家族 + {a.lib_name} 训练份 "
              f"{corpus_prov['lib_pool']['n_train']}(留出 "
              f"{corpus_prov['lib_pool']['n_holdout_excluded']} 不入);"
              f"指纹 {corpus_prov['fingerprint']}")
    else:
        lf_by_sid, solvable = {}, {}
        with open(lib / "manifest.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                lf_by_sid[r["sample_id"]] = float(r["load_factor"])
                solvable[r["sample_id"]] = r["solvable"] == "1"
        samples = [s for s in load_library(lib) if solvable.get(s.sample_id)]
        samples.sort(key=lambda s: lf_by_sid[s.sample_id])
        test_idx = set(range(0, len(samples), a.test_every))
        pool = [s for i, s in enumerate(samples) if i not in test_idx]
        print(f"[lstat] 库 {lib.name}:可解 {len(samples)},训练 {len(pool)}"
              f"(留出 {len(test_idx)} 不入,切法同 g1/g15)")

    ck = torch.load(Path(a.v1_dir) / "v1.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items()
                         if k in {f.name for f in dc_fields(TrainConfig)}})
    enc, _ = build_models(cfg)
    enc.load_state_dict(ck["model_state"])
    enc.eval()
    enc.to(dev)
    norm = Normalizer.load(Path(a.v1_dir) / "normalizer.npz")
    fp_v1 = _fp(Path(a.v1_dir) / "v1.pt")
    fp_nm = _fp(Path(a.v1_dir) / "normalizer.npz")
    print(f"[lstat] 冻结 v1 已载(device={dev.type});"
          f"指纹 v1.pt {fp_v1} / normalizer {fp_nm}")

    t0 = time.time()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(pool), a.batch):
            normed = []
            for raw in pool[i:i + a.batch]:
                s = copy.deepcopy(raw)
                norm.transform(s)
                normed.append(s)
            b = Batch.from_data_list(
                [sample_to_data(v, cfg.lape_k, cfg.rwse_steps,
                                random_dim=cfg.random_dim,
                                branch_random_dim=cfg.branch_random_dim)
                 for v in normed]).to(dev)
            h, _g, _e = enc(b)
            chunks.append(h.cpu().numpy())
    z = np.concatenate(chunks, axis=0)            # [Σ 节点, d] 完整编码节点 token 场
    sizes = {s.n_nodes for s in pool}
    n_nodes = z.shape[0] // len(pool) if len(sizes) == 1 else -1   # 混规模语料无单一 N
    # [外协包 F,2026-08-25] 用 float64 累加器统计。原实现走 numpy 对 float32 数组的默认
    # float32 累加:语料从约 190 万行(g4)涨到约 330 万行(g4+大网)后,累加误差使下面的
    # sanity 断言(标准化后方差≈1,atol=1e-3)失败——是数值精度问题,不是统计错误。
    mean = z.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = z.std(axis=0, dtype=np.float64).astype(np.float32)
    print(f"[lstat] z 场 [{z.shape[0]} 行 x {z.shape[1]} 通道]"
          f"({len(pool)} 方式,规模档 {sorted(sizes)};{time.time() - t0:.0f}s)")
    print(f"[lstat] 逐通道原始 std:min {std.min():.4g} / median {np.median(std):.4g}"
          f" / max {std.max():.4g}(min 过小则该通道近退化,标准化会放大噪声)")

    zs = (z - mean) / std
    # [外协包 F] 校验式同样需要 float64 累加器:319 万行下 float32 的 var 累加误差
    # 本身就有 5e-3 量级,会把"精度噪声"误报成"统计有错"(实测)。
    var_after = zs.var(axis=0, dtype=np.float64)
    print("[lstat] sanity:标准化后逐通道方差(应≈1):")
    _text_hist(var_after, "标准化后方差")
    _text_hist(std, "原始逐通道 std")
    _dev = float(np.abs(var_after - 1.0).max())
    print(f"[lstat] 标准化后方差与 1 的最大偏差 = {_dev:.2e}(float64 累加器)")
    assert np.allclose(var_after, 1.0, atol=1e-3),         f"标准化后方差偏离 1(最大 {_dev:.2e}),统计有错"

    version = STATS_VERSIONS[a.corpus]
    np.savez(out / "latent_stats.npz",
             mean=mean.astype(np.float32), std=std.astype(np.float32),
             meta=json.dumps({
                 "version": version,
                 "z_def": "完整编码节点 token 场(用户裁决:z 收缩为节点场,"
                          "支路 token 只作条件;全局 token 不参与扩散)",
                 "lib": (f"g4_train({corpus_prov['fingerprint']})"
                         if corpus_prov else lib.name),
                 "corpus_provenance": corpus_prov,
                 "n_samples": len(pool), "n_nodes": int(n_nodes),
                 "d": int(z.shape[1]),
                 "split": f"load_factor 排序 test_every={a.test_every} 留出不入"
                          "(g1_baseline 同款)",
                 "v1_fingerprint": fp_v1, "normalizer_fingerprint": fp_nm,
                 "std_min": float(std.min()), "std_median": float(np.median(std)),
                 "std_max": float(std.max()),
             }, ensure_ascii=False))
    print(f"[lstat] 固化 → {out / 'latent_stats.npz'}(version={version};"
          "扩散检查点须记录本文件指纹,解码前逆变换)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        axes[0].hist(std, bins=40)
        axes[0].set_title("raw per-channel std")
        axes[1].hist(var_after, bins=40)
        axes[1].set_title("variance after standardization (expect ~1)")
        fig.tight_layout()
        fig.savefig(out / "variance_hist.png", dpi=120)
        print(f"[lstat] 直方图 → {out / 'variance_hist.png'}")
    except ImportError:
        print("[lstat] matplotlib 不可用,png 略(文本直方图已打)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
