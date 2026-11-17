"""T4.3 失稳机群标签批量提取(方案 7.3;**v1.1 末态失步口径,2026-07-13 用户裁决**;
离线,读留档轨迹,不跑 PSASP)。

输入 = ts_labels 输出目录(labels_v1.jsonl + stout/<tag>.ddd + sme/<tag>.sme);
对**功角失稳**样本提取失步机群:超前群(末 1s 相对 COI 中位 >+500°,涵盖迟到机、剔除
回摆机)∪ 落后群(低于同步群中位基线 −500°,基线法消 COI 均摊伪影);v0 窗口先摆开
提名与摇摆序作诊断保留(gridfm/psasp_io/coherency.py)。产出:
- coherency.jsonl:逐样本机群(multi-hot 源)+ 方向 + 摇摆序 + needs_review;
- coherency_summary.json:覆盖率/机群规模分布/方向分布/v0↔v1.1 差异计数;
- --plot N:抽 N 个失稳例画逐机相对 COI 曲线,超前红/落后紫着色 → plots/<tag>.png
  (DoD"与人工看曲线一致"的证据留档;抽样按机群规模分层,不只挑好看的)。

  python scripts/ts_coherency.py --labels runs/ts_labels_ieee39 [--lib ...] [--plot 12]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402

from gridfm.psasp_io.coherency import extract_coherency  # noqa: E402
from gridfm.psasp_io.coherency import coi_reference  # noqa: E402
from gridfm.psasp_io.stout import extract_angles, parse_stout_blocks  # noqa: E402


def _sme_for(root: Path, tag: str) -> Path:
    p = root / "sme" / f"{tag}.sme"
    return p if p.exists() else root / "ST.SME"


def _plot(root: Path, tag: str, t, ang, pairs, res) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    coi = coi_reference(ang)
    rel = ang - coi[:, None]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    adv, lag = set(res.advance_group), set(res.lag_group)
    for k, (b, g) in enumerate(pairs):
        p = (b, g)
        color, mark = ("#d62728", " [超前]") if p in adv else (
            ("#7b2d8b", " [落后]") if p in lag else ("#7f9fbf", ""))
        ax.plot(t, rel[:, k], lw=2.0 if mark else 1.0, color=color,
                label=f"bus{b}-g{g}{mark}", zorder=3 if mark else 2)
    ax.axhline(500.0, color="#444", ls="--", lw=0.9)
    ax.axhline(-500.0, color="#444", ls=":", lw=0.7)
    if res.t_star is not None:
        ax.axvline(res.t_star, color="k", lw=1.0)
        ax.axvspan(res.window[0], res.window[1], color="#ffd", zorder=1)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("相对 COI 功角 (°)")
    ax.set_title(f"{tag}  t*={res.t_star}  失步 {len(res.unstable_group)}/{res.n_gen}"
                 f"(超前 {len(adv)} 落后 {len(lag)})"
                 + ("  [全网解列→起始机]" if res.breakup else "")
                 + ("  [needs_review]" if res.needs_review else ""))
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    out = root / "plots" / f"{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _n_gen_lookup(lib):
    """算例真实机组数(LF.LP5 行数)= 功角列覆盖门基准;按 sample_id 缓存。"""
    from gridfm.psasp_io import cards
    from gridfm.psasp_io.lowlevel import Card
    cache = {}

    def get(sid, row_hint=None):
        if row_hint:
            return row_hint
        if lib is None:
            return None
        if sid not in cache:
            p = Path(lib) / "cases" / sid / "LF.LP5"
            cache[sid] = len(cards.parse_lp5(Card.load(p))) if p.exists() else None
        return cache[sid]
    return get


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="ts_labels 输出目录")
    ap.add_argument("--lib", default=None,
                    help="方式库目录(取 LF.LP5 机组数作功角列覆盖门基准;SME 功角记录"
                         "来自算例 GUI 模板,缺机会让 COI 参考系与机群分割全部静默失真)")
    ap.add_argument("--plot", type=int, default=0, help="抽 N 个失稳例画曲线(DoD 证据)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.labels)
    n_gen_of = _n_gen_lookup(a.lib)
    src = root / "labels_v1.jsonl"
    if not src.exists():
        print(f"[t4.3] 缺 {src}(先跑 ts_relabel)")
        return 1
    rows = [json.loads(ln) for ln in src.read_text(encoding="utf-8").splitlines()]

    out_rows, sizes, notes = [], Counter(), Counter()
    n_angle, n_grouped, n_review = 0, 0, 0
    n_adv, n_lag_cases, n_breakup = 0, 0, 0
    n_v0_diff = 0                    # v1.1 标签 ≠ v0 窗口提名的样本数(口径差异账)
    for r in rows:
        if not r.get("feasible"):
            continue
        ddd = root / "stout" / f"{r['tag']}.ddd"
        if not ddd.exists():
            continue
        try:
            t, blocks = parse_stout_blocks(ddd, _sme_for(root, r["tag"]))
            ang, pairs = extract_angles(blocks)
        except Exception as e:  # noqa: BLE001  轨迹不可解:如实记账,不产标签
            notes[f"traj_fail"] += 1
            out_rows.append({"tag": r["tag"], "error": str(e)})
            continue
        res = extract_coherency(t, ang, pairs,
                                n_gen_case=n_gen_of(r["sample_id"], r.get("n_gen_case")))
        row = {"tag": r["tag"], "sample_id": r["sample_id"],
               "stable": r["stable"],
               "instability_types": r.get("instability_types"),
               **res.to_dict()}
        out_rows.append(row)
        if res.t_star is None:
            notes[res.note or "not_angle_unstable"] += 1
            if res.needs_review:            # 覆盖门拦下的样本(拒产标签,须人工/配置修复)
                n_review += 1
            continue
        n_angle += 1
        n_adv += int(bool(res.advance_group))
        n_lag_cases += int(bool(res.lag_group))
        n_breakup += int(res.breakup)
        n_v0_diff += int(set(res.unstable_group) != set(res.v0_window_group))
        if res.needs_review:
            n_review += 1
            notes["needs_review"] += 1
        else:
            n_grouped += 1
            sizes[len(res.unstable_group)] += 1

    (root / "coherency.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n",
        encoding="utf-8")
    summary = {
        "protocol": "7.3_v1.1_endstate_500deg_20260713",
        "n_rows": len(out_rows), "n_angle_unstable": n_angle,
        "n_grouped": n_grouped, "n_needs_review": n_review,
        "group_size_hist": {str(k): v for k, v in sorted(sizes.items())},
        "n_with_advance": n_adv, "n_with_lag": n_lag_cases,
        "n_breakup_fallback": n_breakup,
        "n_label_differs_from_v0_window": n_v0_diff,
        "notes": dict(notes),
        "coverage_of_angle_unstable": round(n_grouped / n_angle, 4) if n_angle else None,
        "rule": ["超前失步 = 末 1s 相对 COI 中位 > +500°(涵盖迟到机,剔除回摆机)",
                 "落后失步 = 非超前机中位基线 − 500° 以下(基线法消 COI 均摊伪影)",
                 "标签 = 超前 ∪ 落后(multi-hot,方向随行);v0 窗口先摆开提名 + 摇摆序"
                 "作诊断保留(稳控排序消费)"],
    }
    (root / "coherency_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[t4.3] 功角失稳 {n_angle} 例:分群 {n_grouped} / 待复核 {n_review};"
          f"机群规模分布 {dict(sizes)}")

    if a.plot:
        # 分层抽样:按机群规模分层各取若干(不只挑好看的),needs_review 例必抽
        cand = [r for r in out_rows if r.get("t_star") is not None]
        by_size = {}
        for r in cand:
            by_size.setdefault(len(r["unstable_group"]), []).append(r)
        rng = np.random.default_rng(a.seed)
        pick = [r for r in cand if r["needs_review"]][:max(a.plot // 4, 1)]
        keys = sorted(by_size)
        while len(pick) < a.plot and keys:
            for k in list(keys):
                pool = [r for r in by_size[k] if r not in pick]
                if not pool:
                    keys.remove(k)
                    continue
                pick.append(pool[int(rng.integers(len(pool)))])
                if len(pick) >= a.plot:
                    break
        for r in pick:
            t, blocks = parse_stout_blocks(root / "stout" / f"{r['tag']}.ddd",
                                           _sme_for(root, r["tag"]))
            ang, pairs = extract_angles(blocks)
            res = extract_coherency(t, ang, pairs)
            p = _plot(root, r["tag"], t, ang, pairs, res)
            print(f"  图 {p}  机群 {res.unstable_group}"
                  + ("  [needs_review]" if res.needs_review else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
