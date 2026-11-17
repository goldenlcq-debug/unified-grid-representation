#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""给运行方式生成器（F 包）的**合成评测集**建 N-1 / N-2 底图族（2026-09-06，长卿；任务书 D 项）。

= N-k 线 `build_nk_holdout_families.py`（2026-09-05）的同款做法，只改两点：
  ① 源族在 families/val（同尺寸 8 族）与 families/transfer（跨尺寸 6 族），不在 holdout_f；
  ② 产物写到 families/holdout_f/<fam>_N{k}（g6 的 --extra-plan 认 "holdout_f:<fam>_N1:<n>"），只增不改既有文件。
开断协议逐字同真实网：每个请求各配一次随机开断——N-1 开一条在运非自环支路，N-2 在 N-1 上再开一条；
底图拓扑上做连通性预检（排除桥支路）；cases/<sid> 卡片 LF.L2/L3 对应支路 mark=0；
<sid>.npz 从改后卡片重建并逐列核对（节点特征全同、支路只在开断行的 mark 列不同）；_base_case、manifest 原样拷贝。

只对每族**前 N 个请求**（val 20 / transfer 15，= g6 --set full 的取法 load_lib_lf_sorted()[:N]）建卡，其余方式不动。

  python build_nk_synth_families.py --split val --fam synth150_s2000 --n 20 --seed 9922
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PKG = Path(r"G:\统一计算大模型-迁移包\增量-20260902\外协包-F-大网方式生成-20260825")
REPO = PKG / "grid-foundation"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.fsutil import robust_copytree  # noqa: E402
from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.psasp_io.catalog import Catalog  # noqa: E402
from gridfm.schema.assemble import build_sample  # noqa: E402
from gridfm.schema.fields import BRANCH_TABLE  # noqa: E402
from gridfm.schema.sample import GridSample  # noqa: E402
from gridfm.data.corpus import load_lib_lf_sorted  # noqa: E402  (g6/prep_family 用的同一取请求函数)

FAM_ROOT = PKG / "gridfm-data" / "families"
DST_ROOT = FAM_ROOT / "holdout_f"
OFF_MK = BRANCH_TABLE.offset_of("mark")


def _in_service_branches(case):
    ac, xf = case.aclines(), case.xfmrs()
    return ([("L", l) for l in ac if l.mark == 1 and l.i_no != l.j_no]
            + [("T", t) for t in xf if t.mark == 1])


def _connected_without(case, excluded) -> bool:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ac, xf, n = case.aclines(), case.xfmrs(), int(case.n_buses)
    e = [(a.i_no - 1, a.j_no - 1) for a in ac if a.mark == 1 and a.i_no != a.j_no and a not in excluded]
    e += [(abs(t.i_no) - 1, abs(t.j_no) - 1) for t in xf if t.mark == 1 and t not in excluded]
    if not e:
        return False
    ii, jj = zip(*e)
    A = coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n))
    return connected_components(A, directed=False)[0] == 1


def pick_branch(case, frng, excluded, tries: int = 30):
    cand = [(k, b) for k, b in _in_service_branches(case) if b not in excluded]
    if not cand:
        return None
    for _ in range(tries):
        kind, b = cand[int(frng.integers(0, len(cand)))]
        if _connected_without(case, list(excluded) + [b]):
            return kind, b
    return None


def branch_key(kind, b) -> str:
    return f"{kind}{int(b.i_no)}-{int(b.j_no)}-{int(b.id_no)}"


def open_branches(case, keys) -> None:
    ac, xf = case.aclines(), case.xfmrs()
    for key in keys:
        kind = key[0]
        i, j, idn = key[1:].split("-")
        views = ac if kind == "L" else xf
        hit = [b for b in views if int(b.i_no) == int(i) and int(b.j_no) == int(j) and int(b.id_no) == int(idn)]
        if not hit:
            raise KeyError(f"支路 {key} 在副本中未找到")
        hit[0].mark = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("val", "transfer"))
    ap.add_argument("--fam", required=True)
    ap.add_argument("--n", type=int, required=True, help="请求数（val 20 / transfer 15，与 g6 --set full 同）")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variants", default="1,2")
    ap.add_argument("--out-log", default=None)
    a = ap.parse_args()
    ks = [int(x) for x in a.variants.split(",")]
    src = FAM_ROOT / a.split / a.fam
    assert (src / "manifest.csv").is_file() and (src / "_base_case").is_dir() and (src / "cases").is_dir(), f"评测族不完整：{src}"
    # 请求 = g6 同款取法：load_lib_lf_sorted(famdir)[:N]
    req_samples, _ = load_lib_lf_sorted(src)
    sids = [s.sample_id for s in req_samples[:a.n]]
    assert len(sids) == a.n, f"{a.fam}: 只取到 {len(sids)} 个请求"
    base = PsaspCase(src / "_base_case")
    frng = np.random.default_rng(a.seed + 7_000_000)
    outages = {}
    for sid in sids:
        keys, excluded = [], []
        for _ in range(max(ks)):
            r = pick_branch(base, frng, excluded)
            if r is None:
                break
            kind, b = r
            keys.append(branch_key(kind, b)); excluded.append(b)
        outages[sid] = keys
    report = {"split": a.split, "fam": a.fam, "seed": a.seed, "n_requests": len(sids), "request_sids": sids, "variants": {}}
    for k in ks:
        dst = DST_ROOT / f"{a.fam}_N{k}"
        if (dst / "manifest.csv").exists():
            print(f"[build-nk-synth] {dst} 已存在，跳过（新产物新目录纪律）", flush=True)
            continue
        dst.mkdir(parents=True, exist_ok=True)
        robust_copytree(src / "_base_case", dst / "_base_case")
        # manifest 只保留请求行（其余方式不拷；g6 只取前 N，行序与 lf 序一致性由 sids 顺序保证）
        rows = list(csv.DictReader(open(src / "manifest.csv", encoding="utf-8")))
        keep = {sid: r for r in rows for sid in [r["sample_id"]] if sid in set(sids)}
        with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
            for sid in sids:
                w.writerow(keep[sid])
        n_ok, checks = 0, []
        for sid in sids:
            keys = outages[sid][:k]
            if len(keys) < k:
                raise RuntimeError(f"{sid}: 选不出 {k} 条可开断支路")
            cdst = dst / "cases" / sid
            robust_copytree(src / "cases" / sid, cdst)
            ck = PsaspCase(cdst)
            open_branches(ck, keys)
            ck.save_inplace()
            pc = PsaspCase(cdst)
            gs = build_sample(pc, Catalog(pc), sample_id=sid, family_id=a.fam)
            o = GridSample.load_npz(src / f"{sid}.npz")
            assert gs.node_feat.shape == o.node_feat.shape and gs.branch_feat.shape == o.branch_feat.shape, f"{sid}: 形状不一致"
            node_same = bool(np.allclose(gs.node_feat, o.node_feat, equal_nan=True))
            diff_rows = np.where(~np.all(np.isclose(gs.branch_feat, o.branch_feat, equal_nan=True), axis=1))[0]
            mark_changed = [int(i) for i in diff_rows if not np.isclose(gs.branch_feat[i, OFF_MK], o.branch_feat[i, OFF_MK])]
            other_cols = sorted({int(c) for i in diff_rows for c in np.where(~np.isclose(gs.branch_feat[i], o.branch_feat[i], equal_nan=True))[0] if c != OFF_MK})
            ok = node_same and len(diff_rows) == k and len(mark_changed) == k and not other_cols
            checks.append({"sid": sid, "keys": keys, "node_feat_same": node_same, "n_branch_rows_changed": int(len(diff_rows)),
                           "mark_rows": mark_changed, "other_cols_changed": other_cols, "ok": ok})
            if not ok:
                raise RuntimeError(f"{sid} N{k}: 重建 npz 与原 npz 的差异不是'只有 {k} 行 mark'：{checks[-1]}")
            gs.save_npz(dst / f"{sid}.npz")
            n_ok += 1
        with open(dst / "nk_outages.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["sample_id", "variant", "branches"])
            for sid in sids:
                w.writerow([sid, f"N{k}", ";".join(outages[sid][:k])])
        (dst / "nk_build_report.json").write_text(json.dumps({"source_family": str(src), "variant": f"N{k}", "seed": a.seed,
                                                              "n_requests": len(sids), "n_built": n_ok, "checks": checks},
                                                             ensure_ascii=False, indent=1), encoding="utf-8")
        report["variants"][f"N{k}"] = {"dir": str(dst), "n_built": n_ok}
        print(f"[build-nk-synth] {a.fam}_N{k}: {n_ok}/{len(sids)} 请求已建", flush=True)
    log = Path(a.out_log) if a.out_log else (Path(__file__).resolve().parent.parent / "results" / f"families_{a.fam}.json")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({**report, "outages": outages}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[build-nk-synth] 记账 {log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
