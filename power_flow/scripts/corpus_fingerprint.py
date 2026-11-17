#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""潮流线语料指纹 + 防泄漏核验(拷贝到新机后第一件事跑它)。

两件事:
1. **指纹**:对 families/{train,val,transfer} 与三个单拓扑库,按加载器的同一顺序
   (gridfm.data.dataset.family_files / library_files,manifest 驱动)把"族名 + manifest 字节
   + 每个将被加载的 npz 字节"喂进 BLAKE2b-128,得到每个集合一个指纹。与本包 README 记录的
   指纹逐位相同 = 语料与发包时一致;任何 npz 缺失/损坏/多出都会改变指纹。
2. **泄漏核验**(--leak-check):调用主仓库 gridfm.data.corpus._train_families 的四道守卫——
   train 家族与 val/transfer 家族零交集(manifest 列 + npz 字段双检)、留出集非空、
   train 样本与留出样本**内容级**零重合。任何一条不过会直接 AssertionError。

用法(在 grid-foundation/ 目录下):
  python scripts/corpus_fingerprint.py --root ../gridfm-data [--leak-check]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.data.dataset import family_files, library_files  # noqa: E402

LIBS = ("lib118", "lib_ieee39", "lib_cepri36")


def fp_families(root: Path, split: str):
    h = hashlib.blake2b(digest_size=16)
    sp = root / "families" / split
    fams = sorted(d for d in sp.iterdir() if d.is_dir()) if sp.is_dir() else []
    n = 0
    for fam in fams:
        h.update(fam.name.encode("utf-8"))
        mf = fam / "manifest.csv"
        if not mf.is_file():
            return None, len(fams), n, f"{fam.name} 缺 manifest.csv"
        h.update(mf.read_bytes())
    files = family_files(root, split)
    for p in files:
        h.update(p.read_bytes()); n += 1
    return h.hexdigest(), len(fams), n, ""


def fp_lib(root: Path, lib: str):
    h = hashlib.blake2b(digest_size=16)
    d = root / lib
    if not d.is_dir():
        return None, 0, f"目录不存在 {d}"
    h.update(lib.encode("utf-8"))
    mf = d / "manifest.csv"
    if not mf.is_file():
        return None, 0, "缺 manifest.csv"
    h.update(mf.read_bytes())
    files = library_files(d)
    for p in files:
        h.update(p.read_bytes())
    return h.hexdigest(), len(files), ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--leak-check", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    bad = 0
    print(f"语料根: {root.resolve()}")
    print(f"{'集合':<22}{'族':>5}{'加载文件':>9}  指纹(BLAKE2b-128)")
    for sp in ("train", "val", "transfer"):
        fp, nf, n, err = fp_families(root, sp)
        if fp is None:
            bad += 1
            print(f"{'families/'+sp:<22}{nf:>5}{n:>9}  [FAIL] {err}")
        else:
            print(f"{'families/'+sp:<22}{nf:>5}{n:>9}  {fp}")
    for lib in LIBS:
        fp, n, err = fp_lib(root, lib)
        if fp is None:
            bad += 1
            print(f"{lib:<22}{'-':>5}{n:>9}  [FAIL] {err}")
        else:
            print(f"{lib:<22}{'1':>5}{n:>9}  {fp}")

    # 主仓库官方口径的冻结评估集内容哈希(gridfm.eval.transfer.content_hash 的逐位复刻:
    # 路径按字符串排序,累积 "文件名 + 字节";families/build_report.json 的
    # transfer_content_hash 就是这样算的)。与 README 记录值一致 = 与主仓库冻结集同一份。
    h = hashlib.blake2b(digest_size=16)
    paths = family_files(root, "transfer")
    for p in sorted(paths, key=lambda q: str(q).replace("\\", "/")):
        h.update(Path(p).name.encode("utf-8"))
        h.update(Path(p).read_bytes())
    print(f"\n主仓库口径 transfer 内容哈希(gen_families/transfer.content_hash 同算法):"
          f"{h.hexdigest()}  ({len(paths)} npz)")

    if a.leak_check:
        print("\n泄漏核验(gridfm.data.corpus._train_families 四道守卫)...")
        from gridfm.data.corpus import _train_families
        h = hashlib.blake2b(digest_size=16)
        samples, per_family, val_fams, xfer_fams = _train_families(
            root, h, require_holdout=True, check_content_leak=True)
        print(f"  PASS:train 可解样本 {len(samples)} / {len(per_family)} 族;"
              f"留出家族 val {len(val_fams)} / transfer {len(xfer_fams)};"
              f"家族零交集 + 内容级零重合")
    print("\n总判定:", "FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
