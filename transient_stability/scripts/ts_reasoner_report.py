"""判稳推理器多种子聚合报告(8.6 双报 + 8.9 显著性 + 5.3 组1/组2 消融)。

读 runs/ts_reasoner_v2_{v1,random}_s{0,1,2}/ts_report.json,产出:
- 逐指标 均值±标准差(3 种子;8.9"显著"要求 ≥3 种子重复);
- **5.3 判据**:组1(冻结预训练 v1)vs 组2(冻结随机初始化)——"组间均值差超合并标准差"
  才算显著(8.9 纪律);
- 8.6 双报格式:[内](cepri36 留出方式,同拓扑豁免标注)/ [外](ieee39 整族);
- 细分子类附样本量与**升档门标注**(7.3:>500 才够格挂门禁;不足门只报数)。

  python scripts/ts_reasoner_report.py [--runs runs] [--tag v2]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.reason.ts_task import TS_SUBTYPES, TS_TYPE_FROM_SUB  # noqa: E402

SUBTYPE_GATE = 500          # 7.3 升档门(样本 >500 才够格挂门禁)

# **判别力(skill_over_const = 精确率 ÷ 基率)排第一**:它是本阶段的验收指标。
# 教训(2026-07-15 对抗审查 critical):只报召回会被基率骗——[外] ieee39 的
# angle/v_low 基率 = 1.0,恒定预测正类即得召回 1.000,曾据此误报"跨族能力"。
# 判别力 ≤1 即无优势(模型不如"见谁都判失稳")。基率与预测正类率一并报,缺一不可。
MAIN = [("pr_auc_over_base", "**PR-AUC/base(阈值无关,主判)**"),
        ("skill_over_const", "**判别力(精确率÷基率;≤1 即无优势)**"),
        ("base_rate_unstable", "基率(恒定预测的精确率)"),
        ("pred_pos_rate", "预测正类率"),
        ("unstable_recall", "失稳召回(8.6 主指标)"),
        ("unstable_precision", "失稳精确率"),
        ("pr_auc", "PR-AUC"),
        ("accuracy", "准确率(参考项)"),
        ("cct_mae_noncensored", "CCT MAE(非删失)"),
        ("group_micro_f1", "机群 multi-hot microF1")]


def _mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0, len(vals)
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var), len(vals)


def _fmt(m, s, n):
    if m is None:
        return "  —  "
    return f"{m:.4f}±{s:.4f}" if n > 1 else f"{m:.4f}(n=1)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="v2", help="运行目录前缀 ts_reasoner_<tag>_<臂>_s<seed>")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    # **臂**(2026-07-14 起可多于两个):v2 轮的臂名恰为 v1/random(向后兼容);
    # v3 轮四臂 = A_dim1(dim-1 对照)/ B_dim2(+118)/ C_full(+k50)/ D_rand(随机编码器)。
    ap.add_argument("--arms", nargs="*", default=["v1", "random"])
    # 对照对 "实验臂:对照臂";B:A = 拓扑维度净效应;C:D = 5.3 消融
    ap.add_argument("--compare", nargs="*", default=["v1:random"])
    ap.add_argument("--main-arm", default=None, help="验收判定看哪一臂(缺省 = 第一个臂)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = Path(a.runs)
    reports = {}
    for arm in a.arms:
        for sd in a.seeds:
            p = runs / f"ts_reasoner_{a.tag}_{arm}_s{sd}" / "ts_report.json"
            if p.exists():
                reports.setdefault(arm, {})[sd] = json.loads(p.read_text(encoding="utf-8"))
    if not reports:
        raise SystemExit(f"[report] 没找到任何 ts_reasoner_{a.tag}_*_s*/ts_report.json")
    main_arm = a.main_arm or next(a_ for a_ in a.arms if a_ in reports)
    pairs = [tuple(c.split(":", 1)) for c in a.compare]
    pairs = [(x, y) for x, y in pairs if x in reports and y in reports]

    first = next(iter(reports[main_arm].values()))
    cov = first.get("coverage", {})
    granary = ", ".join(f"{b['batch']}({b['rows']})" for b in cov.get("batches", []))
    print(f"[report] 主臂 {main_arm} 粮仓:{granary}")
    sp = first["split"]
    print(f"[report] 划分:train {sp['n_train']} / [内] {sp['n_in']} / [外] {sp['n_out']};"
          f"**拓扑维度 {sp.get('topology_dim', '?')}**(训练家族 {sp.get('train_families')});"
          f"参数 {first['reasoner_params']/1e6:.2f}M")
    for arm in a.arms:
        if arm in reports:
            sp_ = next(iter(reports[arm].values()))["split"]
            print(f"[report]   臂 {arm}:种子 {sorted(reports[arm])};拓扑维度 "
                  f"{sp_.get('topology_dim', '?')};train {sp_['n_train']}")
    print(f"[report] 方式门踢出:{cov.get('skipped_low_base_vmin', 0)} 行"
          f"(基态 vmin<0.905;标签与故障弱相关)")
    print("[report] 口径:自然分布(平衡集未冻结,8.6 注记);[内] = 同拓扑豁免(D14)\n")

    agg = {}

    def _arm_stat(arm, key, mk, fam=None):
        rs = reports.get(arm, {}).values()
        if fam:
            vals = [(r.get("metrics_in_by_family") or {}).get(fam, {}).get(mk) for r in rs]
        else:
            vals = [r[key].get(mk) for r in rs]
        return _mean_std(vals)

    def _table(title, key, fam=None):
        print(f"{'='*78}\n[{title}]")
        hdr = f"{'指标':<26}" + "".join(f"{arm:>17}" for arm in a.arms if arm in reports)
        print(hdr + "".join(f"{f'  {x}−{y}':>16}" for x, y in pairs))
        for mk, label in MAIN:
            stats = {arm: _arm_stat(arm, key, mk, fam) for arm in a.arms if arm in reports}
            line = f"{label:<26}" + "".join(_fmt(*stats[arm]).rjust(17)
                                            for arm in stats)
            for x, y in pairs:
                mx, sx, _ = stats.get(x, (None, None, 0))
                my, sy, _ = stats.get(y, (None, None, 0))
                if mx is None or my is None:
                    line += " " * 16
                    continue
                pooled = math.sqrt(((sx or 0) ** 2 + (sy or 0) ** 2) / 2)
                diff = mx - my
                lower_better = "mae" in mk
                better = (diff < 0) if lower_better else (diff > 0)
                sig = abs(diff) > pooled if pooled > 0 else abs(diff) > 1e-9
                v = ("显著优" if (better and sig) else
                     "显著劣" if ((not better) and sig) else "无显著差")
                agg[f"{title}/{mk}/{x}-{y}"] = {
                    "exp": mx, "exp_std": sx, "ctl": my, "ctl_std": sy,
                    "pooled_std": pooled, "diff": diff, "verdict": v}
                line += f"{diff:+.3f} {v}".rjust(16)
            for arm, st in stats.items():
                agg[f"{title}/{mk}/{arm}"] = {"mean": st[0], "std": st[1], "n_seeds": st[2]}
            print(line)
        print()

    for split, key in (("内·合并(全训练家族留出)", "metrics_in"),
                       ("外(ieee39 整族,未见拓扑)", "metrics_out")):
        _table(split, key)
        print(f"  失稳类型(三大类,主臂 {main_arm}):")
        for t in TS_TYPE_FROM_SUB:
            r_ = _mean_std([r[key].get(f"type_{t}_recall") for r in reports[main_arm].values()])
            p_ = _mean_std([r[key].get(f"type_{t}_precision")
                            for r in reports[main_arm].values()])
            print(f"    {t:<12} recall {_fmt(*r_)}  prec {_fmt(*p_)}")
        print(f"  细分五路(主臂;升档门 = 7.3 样本 >{SUBTYPE_GATE}):")
        for s_ in TS_SUBTYPES:
            n_ = _mean_std([r[key].get(f"sub_{s_}_n") for r in reports[main_arm].values()])[0]
            r_ = _mean_std([r[key].get(f"sub_{s_}_recall") for r in reports[main_arm].values()])
            p_ = _mean_std([r[key].get(f"sub_{s_}_precision")
                            for r in reports[main_arm].values()])
            n_i = int(n_) if n_ else 0
            note = "  [评估集无正例]" if n_i == 0 else ""
            print(f"    {s_:<12} (评估集 n={n_i:>4}) recall {_fmt(*r_)}  prec {_fmt(*p_)}{note}")
        print()

    # ---- [内] 逐训练家族分报(2026-07-14 拓扑维度 2 起)----
    # 合并 [内] 会把两族的判别力平均掉;"预训练档案在同拓扑内值钱"必须**在每个训练
    # 家族上各自成立**才算数(118 与 cepri36 的基率/物理天差地别:实测 77% vs ~48%)。
    fams = sorted({f for arm in reports for r in reports[arm].values()
                   for f in (r.get("metrics_in_by_family") or {})})
    for fam in fams:
        _table(f"内·{fam}(该训练家族的留出方式,同拓扑)", "metrics_in", fam=fam)

    # ---- 验收(本阶段唯一):[外] 必须打赢 **event-only 空基线**,不是打赢恒定预测 ----
    # **2026-07-14 对抗审查 critical(实测复现)**:原判据"判别力 > 1.05"太松 ——
    # 一个**完全看不见电网**、只读 (K%, dur) 的查表模型,靠"故障越长越易失稳"这条
    # **不需要任何拓扑理解**的规律,在 [外] ieee39 上就能拿到 判别力 1.289 /
    # PR-AUC/base 1.370。门槛 1.05 会把"学会看切除时长"误判成"跨拓扑能力"。
    # 跨拓扑能力的**真正零点 = 该空基线**(逐轮由 ts_reasoner 落进 null_event_only)。
    print(f"{'='*78}\n[验收] 判据 = **[外] 打赢 event-only 空基线**(不是打赢恒定预测)")
    print("  背景:上一轮(拓扑维度 1)[外] 判别力 1.010±0.015 = 恒定预测,零跨拓扑能力。")
    OUT_KEY = "外(ieee39 整族,未见拓扑)"
    nulls = [r.get("null_event_only", {}).get("外") for r in reports[main_arm].values()]
    nulls = [x for x in nulls if x]
    null_skill = _mean_std([x.get("skill_over_const") for x in nulls])
    null_prauc = _mean_std([x.get("pr_auc_over_base") for x in nulls])
    if nulls:
        print(f"  **空基线(零图信息,只读 K%/dur 的查表模型)**:判别力 {_fmt(*null_skill)}"
              f" / PR-AUC/base {_fmt(*null_prauc)}")
        agg["空基线/外"] = {"skill_over_const": null_skill[0],
                          "pr_auc_over_base": null_prauc[0]}
    else:
        print("  ⚠ 无 null_event_only(旧版 ts_reasoner 产的报告)——验收无零点,"
              "**不得据此宣布跨拓扑能力**")
    for mk, label in (("pr_auc_over_base", "PR-AUC/base(阈值无关,**主判**)"),
                      ("skill_over_const", "判别力(0.5 阈)")):
        nm = (null_prauc if mk == "pr_auc_over_base" else null_skill)[0]
        for arm in a.arms:
            if arm not in reports:
                continue
            st = agg.get(f"{OUT_KEY}/{mk}/{arm}")
            if not st or st["mean"] is None:
                continue
            lo = st["mean"] - (st["std"] or 0.0)
            if nm is None:
                v = "无零点,不可判"
            elif lo > nm:
                v = f"**打赢空基线(均值−1σ {lo:.3f} > {nm:.3f})→ 跨拓扑能力**"
            elif st["mean"] > nm:
                v = f"边缘(均值 {st['mean']:.3f} > {nm:.3f} 但 −1σ 未过)→ 证据不足"
            else:
                v = f"**未打赢空基线({st['mean']:.3f} ≤ {nm:.3f})→ 无跨拓扑能力**"
            print(f"  {label} · 臂 {arm}:{_fmt(st['mean'], st['std'], st['n_seeds'])}"
                  f"  → {v}")
            agg[f"验收/{mk}/{arm}"] = {**st, "null": nm, "verdict": v}
        print()
    for arm in a.arms:
        st = agg.get(f"{OUT_KEY}/pred_pos_rate/{arm}")
        if st and st.get("mean") is not None:
            print(f"  预测正类率 · 臂 {arm} = {st['mean']:.3f}"
                  f"(≈1.0 见谁都判失稳 / ≈0 见谁都判稳,两头都是零判别力的指纹)")
    print("\n  [归因] 本轮同时变了 4 件事(加 118 家族 / 加 k50 / 方式门踢 cepri36 900 行 /"
          "118 用 5 档时长)。**B_dim2 − A_dim1 = 拓扑维度 2 的净效应**(其余三项在两臂间"
          "同款);C_full − B_dim2 = k50 的净效应;C_full vs D_rand = 5.3 消融。"
          "时长档的贡献由 event-only 空基线吸收(它恰好只读 K%/dur)。")

    print(f"{'='*78}\n[对照小结] 判据 8.9:≥3 种子;均值差 > 合并标准差才算显著")
    print("  另报**逐种子配对**(同 seed 下 实验臂 − 对照臂):3 种子的 σ 只含训练随机性,")
    print("  单看均值差功效弱;配对同号(3/3)是更强的证据(2026-07-14 对抗审查 major)。")
    for x, y in pairs:
        print(f"  ── {x} − {y} ──")
        for mk, label in MAIN[:4]:
            for split, key in ((OUT_KEY, "metrics_out"),
                               ("内·合并(全训练家族留出)", "metrics_in")):
                d = agg.get(f"{split}/{mk}/{x}-{y}")
                if not d:
                    continue
                # 逐种子配对
                paired = []
                for sd in a.seeds:
                    rx = reports.get(x, {}).get(sd)
                    ry = reports.get(y, {}).get(sd)
                    if rx and ry:
                        vx, vy = rx[key].get(mk), ry[key].get(mk)
                        if vx is not None and vy is not None:
                            paired.append(vx - vy)
                lower_better = "mae" in mk
                wins = sum(1 for v in paired if ((v < 0) if lower_better else (v > 0)))
                pstr = (f";配对 {wins}/{len(paired)} 胜"
                        f"({', '.join(f'{v:+.3f}' for v in paired)})" if paired else "")
                flag = " **配对全胜**" if paired and wins == len(paired) else ""
                print(f"    [{split[:2]}] {label}:{d['exp']:.4f} vs {d['ctl']:.4f}"
                      f"(差 {d['diff']:+.4f},合并σ {d['pooled_std']:.4f})"
                      f" → **{d['verdict']}**{pstr}{flag}")
                agg[f"{split}/{mk}/{x}-{y}"]["paired_diffs"] = paired
                agg[f"{split}/{mk}/{x}-{y}"]["paired_wins"] = wins

    print("\n[注] 判稳不设死门禁(阈值校准依 8.9 走 val);[内] 与上一轮 1.441 **不可直接比**:"
          "方式门踢掉 cepri36 900 行(12 个基态低压方式,失稳率 100%)→ 分母(基率)从"
          "0.5468 变到 0.4850,同一模型的判别力会机械抬高约 1.13×。")

    out = Path(a.out or runs / f"ts_reasoner_{a.tag}_summary.json")
    out.write_text(json.dumps({
        "tag": a.tag, "seeds": a.seeds, "arms": a.arms, "compare": a.compare,
        "main_arm": main_arm,
        "coverage": cov, "split": first["split"],
        "aggregate": agg,
        "note": "验收 = [外] 打赢 event-only 空基线(零图信息);B−A = 拓扑维度净效应;"
                "C vs D = 5.3 消融;8.9 显著性 = 均值差 > 合并标准差;"
                "[内] 与 v2 轮不可比(方式门改了基率)",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[report] 汇总 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
