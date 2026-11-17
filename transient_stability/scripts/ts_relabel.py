"""T4.2 标签离线重判(2026-07-12 用户裁决 v1 判据;免重跑 PSASP)。

从 ts_labels 留档的 STANA+STOUT(out/stana/<tag>.dat、out/stout/<tag>.ddd)按 v1
判据重算全部判稳结果:功角 500°(任意时刻)+ 电压(故障后 10s 机端 ≥0.8 且末 1s
机端 ≥0.9,SME Type=2 曲线精确口径 + 过电压逐母线三级;轨迹缺失退化包络代理,
v_source 如实标注)+ 频率(COI 越 [49,51] Hz 持续 ≥0.5s;无轨迹退化单机极值)。

输出 labels_v1.jsonl(每行含逐判据明细与失稳类型列表)+ labels_v1_summary.json
(含 v0→v1 翻案统计)。原 labels.jsonl 不动(留档)。

  python scripts/ts_relabel.py --labels runs/ts_labels_cepri36
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

from gridfm.psasp_io.stana import judge_files_v1  # noqa: E402


def _n_gen_lookup(lib: Path):
    """算例真实机组数(LF.LP5 行数)= 机端/功角覆盖门基准;按 sample_id 缓存。"""
    from gridfm.psasp_io import cards
    from gridfm.psasp_io.lowlevel import Card
    cache = {}

    def get(sid, row_hint=None):
        if row_hint:                       # ts_labels 新批已把基准写进标签行
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
                    help="方式库目录(取 LF.LP5 机组数作覆盖门基准;旧批标签行无 "
                         "n_gen_case 字段时必须给,否则功角列缺机会被静默当成全覆盖)")
    a = ap.parse_args()
    root = Path(a.labels)
    n_gen_of = _n_gen_lookup(Path(a.lib) if a.lib else None)
    rows = [json.loads(line) for line in
            (root / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    out_rows, flips, types = [], Counter(), Counter()
    n_recover_na, n_freq_na = 0, 0
    src = Counter()                 # 口径来源分布(退化必须可见于汇总,不只逐行)
    n_passthrough_feasible = 0
    for r in rows:
        row = dict(r)
        p = root / "stana" / f"{r['tag']}.dat"
        if not r["feasible"] or not p.exists():
            # **不得盖 v1 假戳**(2026-07-15 对抗审查):两种命运不同的行此前共用一条
            # 直通路径并统一写 criteria_version=v1 —— infeasible 行下游会被丢弃(无害),
            # 但 **feasible 行缺 STANA 留档** 会把 v0 的 stable/types 原样冒充成 v1 判据
            # 产物入训(其 stable 实为 ST.CAL legacy 占位口径,判据自身已注明不可靠)。
            # 改:直通行标 v0_passthrough,收口门按版本拒纳整批 → 逼出重跑/补档。
            row["criteria_version"] = "v0_passthrough"
            row["relabel_note"] = ("infeasible" if not r["feasible"]
                                   else "feasible_but_no_stana_archive")
            if r["feasible"]:
                n_passthrough_feasible += 1
            out_rows.append(row)
            continue
        # SME = .ddd 的解码键,逐 tag 留档优先(列布局随 Type=2 注入而变;目录级
        # 单份 SME 在混批目录上会解出错列 → 静默退化,2026-07-13 审查)
        sme = root / "sme" / f"{r['tag']}.sme"
        if not sme.exists():
            sme = root / "ST.SME"
        det = judge_files_v1(p, root / "stout" / f"{r['tag']}.ddd", sme,
                             fault_ts=r["ts"],
                             n_gen_case=n_gen_of(r["sample_id"], r.get("n_gen_case")))
        flips[(r["stable"], det["stable"])] += 1
        for ty in det["instability_types"]:
            types[ty] += 1
        n_recover_na += int(not det["recover_evaluable"])
        n_freq_na += int(not det["freq_evaluable"])
        src[f"freq:{det.get('freq_source')}"] += 1
        src["v_recover:" + ("exact" if det.get("vterm_full") else "proxy")] += 1
        src["overvolt:" + ("exact" if det.get("n_vbus_curves") else "proxy")] += 1
        if det.get("traj_note"):
            src["traj_fail"] += 1
        row.update({"stable": det["stable"], "stable_v0": r["stable"],
                    "instability_types": det["instability_types"],
                    "voltage_modes": det.get("voltage_modes"),
                    "angle_peak_deg": det["angle_peak_deg"],
                    "vmin_pu": det["vmin_pu"], "last1s_vmin": det["last1s_vmin"],
                    "vterm_min_pu": det.get("vterm_min_pu"),
                    "vterm_last1s": det.get("vterm_last1s"),
                    "vterm_full": det.get("vterm_full"),
                    "n_vterm_curves": det.get("n_vterm_curves"),
                    "n_gen": det.get("n_gen"),
                    "n_vbus_curves": det.get("n_vbus_curves"),
                    "recover_vmin": det.get("recover_vmin"),
                    "overvolt_tier": det.get("overvolt_tier"),
                    "fmin_pu": det["fmin_pu"], "fmax_pu": det["fmax_pu"],
                    # 频率触发方向严格口径(持续段时长;细分 f_high/f_low 标签源)
                    "freq_hold_high_s": det.get("freq_hold_high_s"),
                    "freq_hold_low_s": det.get("freq_hold_low_s"),
                    "recover_evaluable": det["recover_evaluable"],
                    "freq_evaluable": det["freq_evaluable"],
                    "freq_source": det.get("freq_source"),
                    "v_recover_source": det.get("criteria", {}).get("v_recover_source"),
                    "traj_note": det.get("traj_note"),
                    "criteria_version": det["criteria_version"]})
        out_rows.append(row)
    (root / "labels_v1.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n",
        encoding="utf-8")
    feas = [r for r in out_rows if r["feasible"]]
    n_st = sum(1 for r in feas if r["stable"])
    summary = {
        "criteria_version": "v1_20260712",
        "n": len(out_rows), "feasible": len(feas),
        "stable": n_st, "unstable": len(feas) - n_st,
        "instability_type_counts": dict(types),
        "v0_to_v1_flips": {"stable->unstable": flips.get((True, False), 0),
                           "unstable->stable": flips.get((False, True), 0),
                           "unchanged": flips.get((True, True), 0) + flips.get((False, False), 0)},
        "recover_clause_not_evaluable": n_recover_na,
        "freq_not_evaluable": n_freq_na,
        "criteria_sources": dict(src),
        "notes": ["recover 分句(故障后 10s ≥0.8)主口径 = SME Type=2 机端曲线(须全机"
                  "覆盖);steady 分句(末 1s ≥0.9)按 D21 字面用全网最低电压;过电压"
                  "主口径 = 逐母线持续;频率主口径 = COI 持续 ≥0.5s",
                  "轨迹缺失/不可解/机端覆盖不全 → 该分句退化为保守代理(偏严),逐行"
                  "v_recover_source/freq_source/traj_note + 汇总 criteria_sources 如实计数",
                  "recover 分句在 t_end ≤ ts+10 时不可评,如实计数"],
    }
    (root / "labels_v1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[v1] {root.name}: 稳 {n_st} / 失稳 {len(feas)-n_st};翻案 "
          f"稳→失稳 {summary['v0_to_v1_flips']['stable->unstable']} / "
          f"失稳→稳 {summary['v0_to_v1_flips']['unstable->stable']};"
          f"类型 {dict(types)};recover 不可评 {n_recover_na}")
    print(f"[v1] 口径来源 {dict(src)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
