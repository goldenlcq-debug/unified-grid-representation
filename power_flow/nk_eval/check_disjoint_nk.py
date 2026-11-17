#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练集 / 测试集严格分离核验（2026-09-04，用户要求"训练集和测试集要严格分开"）。
对给定的训练语料根（全部 npz）算物理内容哈希（与 gridfm/data/corpus.py::_physics_hash 同算法），
再对每个新评估根逐文件比对：哈希重复数、样本号重复数、家族交集。结果写 JSON。
用法：
  python check_disjoint_nk.py --train <root>[|<root>...] --new <label>=<root>[|...] --out x.json
  root 可以是 families 树（自动递归 *.npz）或方式库目录。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r"E:/潮流线优化包-20260821/grid-foundation")
from gridfm.schema.sample import GridSample  # noqa: E402


def physics_hash(gs) -> str:
    hh = hashlib.blake2b(digest_size=16)
    for arr in (gs.node_feat, gs.branch_feat, gs.edge_index, gs.node_missing, gs.node_masked, gs.branch_missing, gs.branch_masked):
        a = np.ascontiguousarray(arr); hh.update(str(a.dtype).encode()); hh.update(str(a.shape).encode()); hh.update(a.tobytes())
    return hh.hexdigest()


def scan(root: Path):
    out = {}
    for p in sorted(root.rglob("*.npz")):
        if p.name == "family_extra.npz" or "_lowv_quarantine" in p.parts or "_quarantine" in str(p):
            continue
        gs = GridSample.load_npz(p)
        out[str(p)] = (physics_hash(gs), gs.sample_id, gs.family_id)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--new", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    train = {}
    for r in a.train.split("|"):
        train.update(scan(Path(r)))
    th = {}
    for p, (h, sid, fam) in train.items():
        th.setdefault(h, []).append(p)
    tsid = {sid for _, (_, sid, _) in train.items()}
    tfam = {fam for _, (_, _, fam) in train.items()}
    rep = {"train_roots": a.train.split("|"), "n_train_files": len(train), "n_train_hashes": len(th), "secs_scan_train": round(time.time() - t0, 1), "new": {}}
    print(f"[disjoint] 训练语料 {len(train)} 文件 / {len(th)} 个不同哈希 / {len(tfam)} 族 ({time.time()-t0:.0f}s)", flush=True)
    for spec in a.new:
        label, roots = spec.split("=", 1)
        new = {}
        for r in roots.split("|"):
            new.update(scan(Path(r)))
        dup = [p for p, (h, _, _) in new.items() if h in th]
        sid_dup = [p for p, (_, sid, _) in new.items() if sid in tsid]
        fams = {fam for _, (_, _, fam) in new.items()}
        rep["new"][label] = {"roots": roots.split("|"), "n_files": len(new), "n_hash_duplicates_vs_train": len(dup), "hash_duplicates": dup[:20],
                             "n_sample_id_collisions_vs_train": len(sid_dup), "families": sorted(fams), "families_also_in_train": sorted(fams & tfam)}
        print(f"[disjoint] {label}: {len(new)} 文件; 哈希重复 {len(dup)}; 样本号重复 {len(sid_dup)}; 族 {sorted(fams)[:6]}…; 族同在训练集 {sorted(fams & tfam)}", flush=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[disjoint] 报告 {a.out}")


if __name__ == "__main__":
    main()
