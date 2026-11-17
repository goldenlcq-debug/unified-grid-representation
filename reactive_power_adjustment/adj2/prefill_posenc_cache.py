#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""并行预填位置编码磁盘缓存（2026-09-05；70k 的 N-1/N-2 每例拓扑不同，LaPE 每例约 6 分钟，是跑分的瓶颈）。

对考卷目录 cases/* 的每个病例：读卡片 → build_sample → 调用 gridfm.data.graph._posenc（与 sample_to_data 同一函数、同参数），
在 GRIDFM_POSENC_DISK_CACHE 目录落盘；之后 adj3_scale_diag / adj2_eval 命中缓存不再重算。数值与原实现逐位相同（缓存的就是首次计算值）。
分片：--shard i/n 只处理排序后第 i 个分片，多个进程并行。

  GRIDFM_POSENC_DISK_CACHE=<dir> python prefill_posenc_cache.py --lib <exam dir> --shard 0/3
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import fields as dc_fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adj2_common as C  # noqa: E402  (把 REPO 挂进 sys.path)
import torch  # noqa: E402

import gridfm.data.graph as G  # noqa: E402
from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.psasp_io.catalog import Catalog  # noqa: E402
from gridfm.schema.assemble import build_sample  # noqa: E402
from gridfm.train.pretrain import TrainConfig  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if not os.environ.get("GRIDFM_POSENC_DISK_CACHE"):
        print("[prefill] 未设 GRIDFM_POSENC_DISK_CACHE，无意义，退出", flush=True)
        return 2
    i, n = [int(x) for x in a.shard.split("/")]
    ck = torch.load(C.REPO / "runs" / "encoder_v1" / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in allowed})
    cases = sorted(p for p in (Path(a.lib) / "cases").iterdir() if p.is_dir())
    mine = cases[i::n]
    print(f"[prefill] {Path(a.lib).name} 分片 {i}/{n}：{len(mine)}/{len(cases)} 例；lape_k={cfg.lape_k} rwse={cfg.rwse_steps} cache={os.environ['GRIDFM_POSENC_DISK_CACHE']}", flush=True)
    t0 = time.time()
    for j, cd in enumerate(mine):
        t = time.time()
        pc = PsaspCase(cd)
        gs = build_sample(pc, Catalog(pc), sample_id=cd.name, family_id="deploy", schema="v0")
        key = G._disk_key(gs.n_nodes, cfg.lape_k, cfg.rwse_steps, gs.edge_index)
        hit = (Path(os.environ["GRIDFM_POSENC_DISK_CACHE"]) / (key + ".npz")).is_file()
        G._posenc(gs.edge_index, gs.n_nodes, cfg.lape_k, cfg.rwse_steps)
        G._POSENC_CACHE.clear()          # 只要落盘，不占内存
        print(f"[prefill] {j + 1}/{len(mine)} {cd.name} {'命中' if hit else '新算'} {time.time() - t:.0f}s（累计 {time.time() - t0:.0f}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
