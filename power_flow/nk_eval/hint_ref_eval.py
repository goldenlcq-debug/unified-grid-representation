#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零训练提示（带网损 DC 角度提示 / Q–V 电压提示）的单独误差，口径与 eval_by_family_5060.py 逐位同源：
- 只在 D15 类型化未知量位置统计（θ:PQ+PV；v:PQ），逐节点 |误差| 跨方式拼接后取 mean/p99/max；
- θ 用 dc_theta_lossy（E13 检查点 dc_encoding = "lossy"），v 用 qv_hint(dc=dc_theta_lossy, shz=False)
  （E13 检查点 shunt_zero_harmless=False，见 pf_report.json）；变比/移相角来自家族 sidecar family_extra.npz。

用法：
  python hint_ref_eval.py --set "标签:根目录:split[:家族]" [--val-holdout 清单.json --holdout-only] --out x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(r"E:/潮流线优化包-20260821/grid-foundation")
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import pf_reasoner as M  # noqa: E402
from gridfm.data.dataset import family_files  # noqa: E402
from gridfm.reason.pf_physics import dc_theta_lossy, qv_hint  # noqa: E402
from gridfm.schema.fields import NODE_TABLE  # noqa: E402
from gridfm.schema.sample import GridSample  # noqa: E402
from gridfm.train.masking import _pf_typed_mask  # noqa: E402

TH, VC = NODE_TABLE.offset_of("theta"), NODE_TABLE.offset_of("v")


def stat(e):
    if e.size == 0:
        return dict(n=0)
    return dict(n=int(e.size), mean=float(e.mean()), p99=float(np.percentile(e, 99)), max=float(e.max()))


def hint_errors(raws, shz=False):
    et, ev, n_fb, n_inv = [], [], 0, 0
    for s in raws:
        _mf, _tg, lm = _pf_typed_mask(s.node_feat, s.node_missing)
        d = dc_theta_lossy(s)
        m = lm[:, TH] & (d[:, 1] > 0.5)
        if m.any():
            et.append(np.abs(d[m, 0] - s.node_feat[m, TH]).astype(np.float64))
        tap = getattr(s, "branch_tap", None)
        shift = getattr(s, "branch_shift_rad", None)
        vh = qv_hint(s, tap=tap, shift=shift, dc=d, shunt_zero_harmless=shz)
        mv = lm[:, VC] & (vh[:, 1] > 0.5)
        n_fb += int(vh[0, 2] > 0.5) if vh.shape[0] else 0
        n_inv += int(not (vh[:, 1] > 0.5).any())
        if mv.any():
            ev.append(np.abs(vh[mv, 0] - s.node_feat[mv, VC]).astype(np.float64))
    return (np.concatenate(et) if et else np.zeros(0), np.concatenate(ev) if ev else np.zeros(0), n_fb, n_inv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help='"标签|根目录|split[|家族]"')
    ap.add_argument("--val-holdout", action="append", default=[])
    ap.add_argument("--holdout-only", action="store_true", help="只评估留出清单命中的方式")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    hold = set()
    for hp in a.val_holdout:
        hold |= set(json.loads(Path(hp).read_text(encoding="utf-8"))["val_holdout_sample_ids"])
    rep = []
    for spec in a.set:
        parts = spec.split("|")          # 用 | 分隔（Windows 路径含盘符冒号）
        tag, root, split = parts[0], parts[1], parts[2]
        fam_keep = parts[3] if len(parts) > 3 else None
        rp = Path(root)
        byfam = {}
        for p in family_files(rp, split or None):
            byfam.setdefault(Path(p).parent.name, []).append(p)
        for fam, ps in sorted(byfam.items()):
            if fam_keep and fam != fam_keep:
                continue
            if a.holdout_only:
                ps = [q for q in ps if Path(q).stem in hold]
            elif hold:
                ps = [q for q in ps if Path(q).stem not in hold]
            if a.limit:
                ps = ps[:a.limit]
            if not ps:
                continue
            raws = [GridSample.load_npz(q) for q in ps]
            n_hit, fams = M.attach_family_extra(raws, [rp])
            et, ev, n_fb, n_inv = hint_errors(raws)
            rec = dict(set=tag, family=fam, n_samples=len(raws), sidecar_hit=n_hit,
                       dc_lossy_theta_rad=stat(et), dc_lossy_theta_deg=stat(np.degrees(et)),
                       qv_hint_v_pu=stat(ev), qv_fallback_cases=n_fb, qv_invalid_cases=n_inv)
            rep.append(rec)
            print(f"{tag:<28}{fam:<16}n={len(raws):<4}DC θ mean {np.degrees(et.mean()) if et.size else float('nan'):.4f}° "
                  f"| Q–V v mean {ev.mean() if ev.size else float('nan'):.6f} pu (fallback {n_fb}, invalid {n_inv})", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[hint] 报告 {a.out}")


if __name__ == "__main__":
    main()
