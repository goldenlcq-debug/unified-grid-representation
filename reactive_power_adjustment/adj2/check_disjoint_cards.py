#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""训练集 / 测试集严格分离核验 · PSASP 卡片层（2026-09-05，N-k 其他推理器线）。

对参照集（本线全部方式库 libs/mp_*、全部标签库 libs/labels_*、全部既有考卷 libs/viol_*、accept_*）的每个卡片目录算两种哈希：
  full  = LF.L0/L1/L2/L3/L5/L6/L7 七张输入卡（与 viol_evalgen_mp.sha256_files 同算法）；
  state = 只 LF.L5 + LF.L6（机组/负荷卡）——同一运行点只改拓扑也能被抓到（N-1 变体的 L2/L3 不同、L5/L6 相同）。
再对新集合（nk_pack/<fid>/N{k}/<sid>、libs_nk/<考卷>/cases/<cid>）逐目录比对，重复数必须为 0。
  python check_disjoint_cards.py --new <label>=<dir-with-case-dirs>[|...] --out x.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LIBS = ROOT.parent / "大算例样本线-20260820" / "libs"
FULL = ("LF.L0", "LF.L1", "LF.L2", "LF.L3", "LF.L5", "LF.L6", "LF.L7")
STATE = ("LF.L5", "LF.L6")


def h(cdir: Path, names) -> str:
    hh = hashlib.sha256()
    for n in names:
        p = cdir / n
        if p.is_file():
            hh.update(n.encode("utf-8")); hh.update(b"\x00"); hh.update(p.read_bytes()); hh.update(b"\n")
    return hh.hexdigest()


def case_dirs(root: Path):
    """root 下的病例/方式目录：root/cases/* 或 root/* （含 LF.L5 的目录）。"""
    out = []
    if not root.exists():
        return out
    cand = list((root / "cases").iterdir()) if (root / "cases").is_dir() else list(root.iterdir())
    for d in cand:
        if d.is_dir() and (d / "LF.L5").is_file():
            out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    ref_full, ref_state = {}, {}
    n_ref = 0
    ref_roots = sorted(p for p in LIBS.iterdir() if p.is_dir() and (p.name.startswith(("mp_", "labels_", "viol_", "accept_", "neg_"))))
    for r in ref_roots:
        for d in case_dirs(r):
            n_ref += 1
            ref_full.setdefault(h(d, FULL), []).append(str(d))
            ref_state.setdefault(h(d, STATE), []).append(str(d))
    rep = {"reference_roots": [str(r) for r in ref_roots], "n_reference_dirs": n_ref, "n_full_hashes": len(ref_full),
           "n_state_hashes": len(ref_state), "secs_scan_reference": round(time.time() - t0, 1), "new": {}}
    print(f"[disjoint-cards] 参照 {n_ref} 个卡片目录（{len(ref_roots)} 库）{time.time() - t0:.0f}s", flush=True)
    for spec in a.new:
        label, roots = spec.split("=", 1)
        dirs = []
        for r in roots.split("|"):
            rp = Path(r)
            if (rp / "cases").is_dir() or any((c / "LF.L5").is_file() for c in rp.iterdir() if c.is_dir()):
                dirs += case_dirs(rp)
            else:                      # 两层：<root>/N0|N1|N2/<sid>
                for sub in sorted(x for x in rp.iterdir() if x.is_dir()):
                    dirs += case_dirs(sub)
        dup_full = [(str(d), ref_full[h(d, FULL)][:3]) for d in dirs if h(d, FULL) in ref_full]
        dup_state = [(str(d), ref_state[h(d, STATE)][:3]) for d in dirs if h(d, STATE) in ref_state]
        # 新集合内部：N0/N1/N2 同 sid 的 state 哈希应相同（配对），不同 sid 应不同
        rep["new"][label] = {"roots": roots.split("|"), "n_dirs": len(dirs), "n_full_dup_vs_reference": len(dup_full),
                             "n_state_dup_vs_reference": len(dup_state), "full_dups": dup_full[:20], "state_dups": dup_state[:20]}
        print(f"[disjoint-cards] {label}: {len(dirs)} 目录；七卡重复 {len(dup_full)}；机组/负荷卡重复 {len(dup_state)}", flush=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[disjoint-cards] 报告 {a.out}（{time.time() - t0:.0f}s）")
    return 0 if all(v["n_full_dup_vs_reference"] == 0 and v["n_state_dup_vs_reference"] == 0 for v in rep["new"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
