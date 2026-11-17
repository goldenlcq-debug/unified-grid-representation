"""R4 呈裁前的**零机时**噪声审计:把 R4 的两个头条数字降到种子级配对上重读。

背景:`runs/r4_matched_step.json`(R4 §2)是**先按臂求均值再作差**,所以只能给
"中位 +4.01pp / 正号 6/8"这种臂级读数,看不出这 +4pp 是三座都有、还是一两座扛出来的。
本脚本用已有产物做两件事,**不训练、不评估、不碰 GPU**:

① **种子级同步数配对**(读 `runs/traj_r4_*.json`):臂① seed n 与臂③ seed n 在**同一 step**
   相减 → 每个指标得 3 座 × 8 步点 = 24 个配对差,报逐座中位/正号与整体离散度。
   判读用意:R4 的"解冻买到跨族泛化"若只由 1~2 座支撑,按本仓"单种子不上报"的纪律
   就不足以支撑"推广成共享编码器"这种不可逆动作。

② **主指标② 的两项敏感性**(读 `runs/ts_screen/{r3,r4}/*screen_eval*.json`):
   (a) 剔掉预注册自己标出的**无分辨力层**(`low_resolution_layers`,正例数少到
       90% 档实为 100% 召回)后,−7.16pp 还剩多少;
   (b) 用 **R4 自己两臂**的种子离散度重算配对 SEM,核对预注册那条"分辨力下限 3pp"
       (它标定在 R2c 的臂上,SEM 1.09~1.41pp)在 R4 的臂上是否仍然成立。

产物 `runs/r4_seed_pairing.json`。所有数字可由本脚本一键重算,文档不手抄。
"""
import argparse
import json
import statistics as st
from pathlib import Path

# 臂① seed n ↔ 臂③ seed n(臂③ s1 = r2c_all_s1_long,args 与本轮逐字同,仅 keep_last 不同)
ARM1_TRAJ = ["traj_r4_ts_reasoner_r4_none_s0.json",
             "traj_r4_ts_reasoner_r4_none_s1.json",
             "traj_r4_ts_reasoner_r4_none_s2.json"]
ARM3_TRAJ = ["traj_r4_ts_reasoner_r4_all_s0.json",
             "traj_r4_ts_reasoner_r2c_all_s1_long.json",
             "traj_r4_ts_reasoner_r4_all_s2.json"]
METRICS = [("out_ieee39_prauc", "ieee39[外]pr_auc"),
           ("in_ieee118_acc", "118[内]acc"),
           ("in_cepri36_acc", "cepri36[内]acc")]
GE_STEP = 6000  # R4 §2 的"仅 >=6000 步"切片,沿用同一阈


def _traj(runs_dir, name):
    d = json.loads((Path(runs_dir) / name).read_text(encoding="utf-8"))
    return d["best_step"], {r["step"]: r for r in d["rows"]}


def seed_pairing(runs_dir):
    a1 = [_traj(runs_dir, n) for n in ARM1_TRAJ]
    a3 = [_traj(runs_dir, n) for n in ARM3_TRAJ]
    # 六座共有的步点(臂① s0 最短,到 8200);排除 200(R4 §2 同样从 1200 起)
    common = sorted(set.intersection(*[set(t[1]) for t in a1 + a3]) - {200})
    out = {"common_steps": common,
           "arm1_best_steps": [t[0] for t in a1],
           "arm3_best_steps": [t[0] for t in a3],
           "metrics": {}}
    for key, lbl in METRICS:
        per_step, flat = [], []
        for s in common:
            ds = [(a3[n][1][s][key] - a1[n][1][s][key]) * 100 for n in range(3)]
            flat += ds
            per_step.append({"step": s, "d_pp_by_seed": [round(x, 2) for x in ds],
                             "mean_pp": round(st.mean(ds), 2),
                             "n_pos": sum(1 for x in ds if x > 0)})
        by_seed = []
        for n in range(3):
            ds = [(a3[n][1][s][key] - a1[n][1][s][key]) * 100 for s in common]
            by_seed.append({"seed": n, "median_pp": round(st.median(ds), 2),
                            "n_pos": sum(1 for x in ds if x > 0), "n": len(ds)})
        ge = [x for s in common if s >= GE_STEP
              for x in [(a3[n][1][s][key] - a1[n][1][s][key]) * 100 for n in range(3)]]
        out["metrics"][lbl] = {
            "per_step": per_step, "by_seed": by_seed,
            "all_pairs": {"n": len(flat), "median_pp": round(st.median(flat), 2),
                          "mean_pp": round(st.mean(flat), 2),
                          "sd_pp": round(st.stdev(flat), 2),
                          "n_pos": sum(1 for x in flat if x > 0)},
            f"ge{GE_STEP}": {"n": len(ge), "median_pp": round(st.median(ge), 2),
                             "mean_pp": round(st.mean(ge), 2),
                             "n_pos": sum(1 for x in ge if x > 0)},
        }
    return out


def _arm_primary(arm, layers):
    """按给定层集重算每座的 90% 召回复核比例均值(层间等权,与 r2_screen_eval 同法)。"""
    vals = []
    for s in arm["per_seed"]:
        bl = s["by_layer"]
        vals.append(sum(bl[k]["0.9"] for k in layers) / len(layers))
    return vals


def screen_sensitivity(paths):
    """paths = {tag: screen_eval.json};各档共用同一层集与同一 logit 口径。"""
    ds = {t: json.loads(Path(p).read_text(encoding="utf-8")) for t, p in paths.items()}
    ref = next(iter(ds.values()))
    low = list(ref["arms"][next(iter(ref["arms"]))]["low_base_layers"])
    lowres = list(ref["low_resolution_layers"])
    res = [k for k in low if k not in lowres]           # 有分辨力的低基率层
    arms = {}
    for tag, d in ds.items():
        assert list(d["arms"][next(iter(d["arms"]))]["low_base_layers"]) == low, \
            f"{tag} 层集与参照不同,禁止并比"
        assert d["score_field"] == ref["score_field"], f"{tag} 排序口径不同,禁止并比"
        for arm, v in d["arms"].items():
            arms[arm] = {
                "tag": tag, "n_seeds": v["n_seeds"],
                "prereg3": [round(x, 4) for x in _arm_primary(v, low)],
                "resolvable2": [round(x, 4) for x in _arm_primary(v, res)],
            }
    for a in arms.values():
        for k in ("prereg3", "resolvable2"):
            a[k + "_mean"] = round(st.mean(a[k]), 4)
            a[k + "_sd"] = round(st.stdev(a[k]), 4)
            a[k + "_sem"] = round(st.stdev(a[k]) / len(a[k]) ** 0.5, 4)
    return {"low_base_layers": low, "no_resolution_layers": lowres,
            "resolvable_layers": res, "score_field": ref["score_field"],
            "base_rate_by_layer": ref["base_rate_by_layer"],
            "n_unstable_by_layer": ref["n_unstable_by_layer"], "arms": arms}


def paired(arms, a, b, key):
    """按种子序配对(两臂同用 seed 0/1/2,配对合法且降方差);越小越省,故 b−a 负号 = b 更省。"""
    A, B = arms[a][key], arms[b][key]
    d = [(y - x) * 100 for x, y in zip(A, B)]
    sem = st.stdev(d) / len(d) ** 0.5
    return {"pair": f"{b} - {a}", "layers": key,
            "d_pp_by_seed": [round(x, 2) for x in d],
            "mean_pp": round(st.mean(d), 2), "sd_pp": round(st.stdev(d), 2),
            "sem_pp": round(sem, 2), "two_sigma_pp": round(2 * sem, 2),
            "n_neg": sum(1 for x in d if x < 0), "n": len(d),
            "resolvable_at_2sigma": bool(abs(st.mean(d)) > 2 * sem)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--r4-screen", default="runs/ts_screen/r4/r4_screen_eval.json")
    ap.add_argument("--r3-screen", default="runs/ts_screen/r3/r3_screen_eval_logit.json")
    ap.add_argument("--out", default="runs/r4_seed_pairing.json")
    a = ap.parse_args()

    rep = {"note": "R4 呈裁前零机时噪声审计:种子级同步数配对 + 主指标② 敏感性。"
                   "不改任何预注册判据,只并报成分。",
           "seed_pairing": seed_pairing(a.runs),
           "screen": screen_sensitivity({"R4@40000": a.r4_screen,
                                         "R3(4000 vs 20000)": a.r3_screen})}
    arms = rep["screen"]["arms"]
    rep["screen"]["paired"] = [
        paired(arms, x, y, k)
        for x, y in [("A1_none", "A3_all"), ("r1b_4000", "r3_20000"),
                     ("r1b_4000", "A3_all"), ("A1_none", "r3_20000")]
        for k in ("prereg3", "resolvable2")]

    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=== ① 种子级同步数配对(臂③ − 臂①;步点 %s)" % rep["seed_pairing"]["common_steps"])
    for lbl, m in rep["seed_pairing"]["metrics"].items():
        ap_, ge = m["all_pairs"], m[f"ge{GE_STEP}"]
        print(f"  {lbl:18s} 全配对 中位{ap_['median_pp']:+6.2f}pp 均值{ap_['mean_pp']:+6.2f} "
              f"sd{ap_['sd_pp']:5.2f} 正号{ap_['n_pos']}/{ap_['n']} | "
              f">={GE_STEP}步 中位{ge['median_pp']:+6.2f} 正号{ge['n_pos']}/{ge['n']} | 逐座中位 "
              + " ".join(f"s{b['seed']}{b['median_pp']:+.2f}({b['n_pos']}/{b['n']})"
                         for b in m["by_seed"]))
    print("=== ② 主指标② 敏感性(90% 召回复核比例,越小越省)")
    print(f"  无分辨力层(已剔) {rep['screen']['no_resolution_layers']} "
          f"正例数 {[rep['screen']['n_unstable_by_layer'][k] for k in rep['screen']['no_resolution_layers']]}")
    for name, v in arms.items():
        print(f"  {name:10s}({v['tag']:18s}) 预注册3层 {v['prereg3_mean']:.4f} "
              f"(sd {v['prereg3_sd']:.4f} sem {v['prereg3_sem']:.4f}) | "
              f"有分辨力2层 {v['resolvable2_mean']:.4f} "
              f"(sd {v['resolvable2_sd']:.4f} sem {v['resolvable2_sem']:.4f})")
    for p in rep["screen"]["paired"]:
        print(f"  {p['pair']:22s} {p['layers']:11s} {p['mean_pp']:+6.2f}pp "
              f"逐种子 {p['d_pp_by_seed']} sem {p['sem_pp']:.2f} "
              f"2σ {p['two_sigma_pp']:.2f} → {'可分辨' if p['resolvable_at_2sigma'] else '不可分辨'}")
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
