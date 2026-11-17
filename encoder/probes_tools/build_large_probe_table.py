#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 large_results/probe_large_*.json → 编码器大网表征效果表（一网一行）+ 25k/70k 的 N-0/N-1/N-2 行；写 md 与 json。"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent / "large_results"
GRIDS = ["ACTIVSg2000", "case3120sp", "PEGASE1354", "PEGASE2869", "ACTIVSg25k", "ACTIVSg70k"]
SEEN = {"ACTIVSg2000": "未见", "case3120sp": "未见", "PEGASE1354": "未见", "PEGASE2869": "未见", "ACTIVSg25k": "未见", "ACTIVSg70k": "未见"}


def f(x, nd=4):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def row(r, name):
    rt, pp, rc, ra = r["retrieval"], r["parallel_pairs"], r["recovery"], r["ratio_vs_source"]
    pairs = (f"{pp['sep_min']:.4f}（{pp['n_pairs']} 对；置零后 {pp['ablated_max']:.1e}）" if pp.get("n_pairs") else "无此类支路对")
    return (f"| {name} | {rt['mean_top1']:.4f}（基线 {rt['random_baseline']:.1e}） | {pairs} | "
            f"{rc['mae_pu']['v']:.5f} / {rc['ladder']['v']['L2']:.5f} / {f(rc['skill']['v'], 2)} | "
            f"{rc['mae_pu']['theta']:.4f} / {rc['ladder']['theta']['L2']:.4f} / {f(rc['skill']['theta'], 2)} | "
            f"{ra['v']:.2f}× / {ra['theta']:.2f}× |")


def main():
    out = {}
    lines = ["| 网（编码器训练均未见） | 母线寻址准确率（20 例） | 并联同型支路最小可分距离 | V 恢复 MAE / 图调和参照 / 技巧分（p.u.） | θ 恢复 MAE / 图调和参照 / 技巧分（rad） | 相对源网倍数 V / θ |",
             "|---|---|---|---|---|---|"]
    notes = []
    for g in GRIDS:
        p = HERE / f"probe_large_{g}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8")); out[g] = r
        lines.append(row(r, g))
        notes.append(f"{g}：{r['n_samples']} 例（寻址 {r['retrieval']['n_samples']} 例）" + (f"；{r['sanitized_nonfinite']['n_samples']} 例动态字段非有限值置 0" if r.get("sanitized_nonfinite", {}).get("n_samples") else ""))
    lines2 = ["| 网 / 变体 | 母线寻址准确率 | 并联同型支路最小可分距离 | V 恢复 MAE / 参照 / 技巧分 | θ 恢复 MAE / 参照 / 技巧分 | 相对源网倍数 V / θ |", "|---|---|---|---|---|---|"]
    for g in ("ACTIVSg25k", "ACTIVSg70k"):
        for k in (0, 1, 2):
            p = HERE / f"probe_large_{g}_N{k}.json"
            if p.exists():
                r = json.loads(p.read_text(encoding="utf-8")); out[f"{g}_N{k}"] = r
                lines2.append(row(r, f"{g} N-{k}（{r['n_samples']} 例）"))
    src = None
    for r in out.values():
        src = r.get("source_mae_pu"); break
    md = ["# 编码器大网表征效果表（冻结 v1 seed0；2026-09-05）", "",
          f"分母（相对源网倍数）= 同环境复现的旧冻结验证集 404 例：V {src['v']:.5f}、θ {src['theta']:.4f}（`probe_legacy_reproduction.json`）。" if src else "",
          "例数（按用户裁决不进表）：" + "；".join(notes), "",
          "## P1 六张网（N-0，现有已解算方式）", *lines, "",
          "## P2 25k / 70k 配对 N-0 / N-1 / N-2（`gridfm-data-nk`）", *lines2, ""]
    (HERE / "编码器大网表征效果表-20260905.md").write_text("\n".join(md), encoding="utf-8")
    (HERE / "large_probe_summary.json").write_text(json.dumps({k: {kk: v[kk] for kk in ("grid", "n_samples", "retrieval", "parallel_pairs", "recovery", "ratio_vs_source", "sanitized_nonfinite") if kk in v} for k, v in out.items()}, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
