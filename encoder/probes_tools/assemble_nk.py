#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 produce_modes_nk.py 的产出装配成评估用的数据根（2026-09-04，长卿）。
- 只保留三个变体都收敛的样本号（严格配对）；
- 潮流网族拷交付家族的 family_extra.npz（变比 sidecar；支路行序不变，开断只是 mark=0）；
- 输出 gridfm-data-nk/N{k}/families/<split>/<fam>/{*.npz, manifest.csv[, family_extra.npz]}，并写 assemble_report.json。
用法：
  python assemble_nk.py --raw nk_raw/pf --dst gridfm-data-nk --split val --sidecar-root <gridfm-data-large/families>
  python assemble_nk.py --raw nk_raw/encoder_val --dst gridfm-data-nk --split val
  python assemble_nk.py --raw nk_raw/encoder_transfer --dst gridfm-data-nk --split transfer
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def read_manifest(p: Path):
    if not p.exists():
        return []
    return list(csv.DictReader(open(p, encoding="utf-8")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--variants", default="0,1,2")
    ap.add_argument("--sidecar-root", default=None, help="含 <split>/<fam>/family_extra.npz 的交付 families 根（潮流网族用）")
    ap.add_argument("--only", default=None, help="只装配这些家族（逗号分隔）")
    a = ap.parse_args()
    ks = [int(x) for x in a.variants.split(",")]
    raw, dst = Path(a.raw), Path(a.dst)
    report = {}
    only = set(a.only.split(",")) if a.only else None
    for famdir in sorted(p for p in raw.iterdir() if p.is_dir() and not p.name.startswith("_")):
        fam = famdir.name
        if only and fam not in only:
            continue
        mans = {k: read_manifest(famdir / f"N{k}" / "manifest.csv") for k in ks}
        ids = None
        for k in ks:
            s = {r["sample_id"] for r in mans[k]}
            ids = s if ids is None else ids & s
        ids = sorted(ids or [])
        for k in ks:
            out = dst / f"N{k}" / "families" / a.split / fam
            out.mkdir(parents=True, exist_ok=True)
            keep = [r for r in mans[k] if r["sample_id"] in ids]
            for r in keep:
                shutil.copyfile(famdir / f"N{k}" / f"{r['sample_id']}.npz", out / f"{r['sample_id']}.npz")
            with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(keep[0].keys()) if keep else ["sample_id"])
                w.writeheader(); w.writerows(keep)
            if a.sidecar_root:
                for sp in ("train", "transfer", "val"):
                    sc = Path(a.sidecar_root) / sp / fam / "family_extra.npz"
                    if sc.exists():
                        shutil.copyfile(sc, out / "family_extra.npz"); break
        report[fam] = {f"N{k}_converged": len(mans[k]) for k in ks} | {"paired": len(ids)}
        print(f"[assemble] {fam}: " + " ".join(f"N{k}={len(mans[k])}" for k in ks) + f" → 配对 {len(ids)}", flush=True)
    (dst / f"assemble_report_{raw.name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
