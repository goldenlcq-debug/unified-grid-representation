"""[G] ① 预筛头 · 训练 + 离线回放评估(2026-08-03;纯 CPU,零仿真)。

预注册 = docs/预注册-预筛头回放-20260803.md **v2 全池悲观界口径**(判据/分层/
泄漏面先写死;v2 修订于出数前,修订原因 = 前缀内重排对预筛头系统性乐观)。
输入 = g5_prescreen_replay.py 产出的全池特征表(每需求 8 种子候选全量,
2,544 行带三态标签、其余 unknown;GPU 重放重建,保真校验已过门)。

主判据(悲观下界):全 8 池按折外分数降序重排,顺序仿真,unknown 一律按
"失败、计一次仿真";到首个真 ok 交付。每交付仿真次数 = (Σ位次+Σ未解×8)/交付数;
基线 = 原 z_seed 序(ok 前全为已知失败 → 自动精确等于当年实测 2.97)。
成功门:s1 val8 层悲观界 ≤2.52(降 ≥15%)。随机零点在悲观口径下应劣于基线。

  python scripts/g5_prescreen_train.py [--dir runs/g5_prescreen] [--out eval.json]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

NUM_FEATS = ["n_nodes", "req_load", "req_online", "req_ngen",
             "gen_load", "gen_online", "load_rel_dev", "online_dev",
             "pg_mean", "pg_std", "pg_max", "pg_neg", "ql_std", "pl_std",
             "vset_dev_mean", "vset_dev_std", "vset_dev_maxabs", "vset_hi",
             "z0_std", "z0_maxabs", "z0_clip_frac"]
BANNED = {"label", "y", "z_seed"}          # 泄漏面(顺位/种子/仿真输出禁入)
SEEDS = (0, 1, 2)


def load_rows(d):
    rows = []
    for p in sorted(d.glob("features_*.csv")):
        # 只收臂级全量段(features_s1.csv 等),跳过小样本验证段
        if p.stem.count("_") > 1:
            continue
        with open(p, encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    seen = set()
    out = []
    for r in rows:
        key = (r["model"], r["z_seed"], r["sid"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def featurize(rows, with_fam):
    fams = sorted({r["fam"] for r in rows})
    models = sorted({r["model"] for r in rows})
    X = []
    for r in rows:
        v = [float(r[c]) for c in NUM_FEATS]
        v += [1.0 if r["model"] == m else 0.0 for m in models]
        if with_fam:
            v += [1.0 if r["fam"] == f else 0.0 for f in fams]
        X.append(v)
    return np.asarray(X, dtype=np.float64)


def _fit_score(X, y, tr, te, kind, seed):
    """tr 内只用带标签行训练;te 全打分(unknown 行也要折外分数供回放)。"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    trl = tr[y[tr] >= 0]
    if kind == "lr":
        sc = StandardScaler().fit(X[trl])
        m = LogisticRegression(max_iter=2000, class_weight="balanced")
        m.fit(sc.transform(X[trl]), y[trl])
        return m.predict_proba(sc.transform(X[te]))[:, 1]
    m = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced",
                                       max_iter=300, early_stopping=False)
    m.fit(X[trl], y[trl])
    return m.predict_proba(X[te])[:, 1]


def oof_scores(rows, y, groups, with_fam, kind, seed=0):
    from sklearn.model_selection import GroupKFold
    X = featurize(rows, with_fam)
    oof = np.full(len(rows), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        oof[te] = _fit_score(X, y, tr, te, kind, seed)
    assert not np.isnan(oof).any()
    return oof


def fam_oof_scores(rows, y, kind, seed=0):
    """严格口径:逐族轮出(族 one-hot 禁用)。"""
    X = featurize(rows, with_fam=False)
    fams = np.asarray([r["fam"] for r in rows])
    oof = np.full(len(rows), np.nan)
    for f in np.unique(fams):
        tr, te = np.where(fams != f)[0], np.where(fams == f)[0]
        oof[te] = _fit_score(X, y, tr, te, kind, seed)
    assert not np.isnan(oof).any()
    return oof


def replay_account(rows, scores):
    """全池悲观界回放:按分数降序顺序仿真,unknown 一律按失败计一次仿真,
    到首个真 ok 交付;无 ok 者试满全池。返回悲观 cpd 与 top-m。
    rows/scores 已按 (model, layer, z_seed!=0) 过滤;每需求应为全 8 池。"""
    by_sid = defaultdict(list)
    for r, s in zip(rows, scores):
        by_sid[r["sid"]].append((r, s))
    total_calls, delivered, n_req = 0, 0, len(by_sid)
    topm_hit = {2: 0, 4: 0, 8: 0}
    for sid, lst in by_sid.items():
        lst_sorted = sorted(lst, key=lambda t: int(t[0]["z_seed"]))  # 原序 = z 升序
        order = sorted(range(len(lst_sorted)),
                       key=lambda i: (-lst_sorted[i][1], i))         # 平局保原序
        ok_pos = None
        for newpos, i in enumerate(order, start=1):
            if lst_sorted[i][0]["label"] == "ok":
                ok_pos = newpos                                      # unknown=失败已计位
                break
        if ok_pos is not None:
            delivered += 1
            total_calls += ok_pos
            for m in topm_hit:
                if ok_pos <= m:
                    topm_hit[m] += 1
        else:
            total_calls += len(lst_sorted)                           # 全池试满
    cpd = None if not delivered else round(total_calls / delivered, 3)
    return dict(calls_per_delivered=cpd, delivered=delivered, n_req=n_req,
                total_calls=total_calls,
                top_m_rate={m: round(v / n_req, 4) for m, v in topm_hit.items()})


def metrics(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(set(y)) < 2:
        return {"auroc": None, "pr_auc": None, "base": round(float(np.mean(y)), 4)}
    return {"auroc": round(float(roc_auc_score(y, s)), 4),
            "pr_auc": round(float(average_precision_score(y, s)), 4),
            "base": round(float(np.mean(y)), 4),
            "pred_pos_rate@0.5": round(float(np.mean(np.asarray(s) > 0.5)), 4)}


VAL8 = {"synth210_s2007", "synth150_s2000", "synth210_s2002", "synth240_s2003",
        "synth270_s2004", "synth150_s2005", "synth180_s2001", "synth180_s2006"}
TRANSFER6 = {"synth400_s3000", "synth400_s3001", "synth400_s3002", "synth400_s3003",
             "synth700_s3004", "synth1000_s3008"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/g5_prescreen")
    ap.add_argument("--out", default="eval.json")
    a = ap.parse_args()
    d = REPO / a.dir
    rows = load_rows(d)
    assert set(NUM_FEATS).isdisjoint(BANNED)
    y = np.asarray([int(r["y"]) for r in rows])
    sids = np.asarray([r["sid"] for r in rows])
    lab_mask = y >= 0
    print(f"[train] 全池 {len(rows)} 行(带标签 {int(lab_mask.sum())},"
          f"ok {int((y == 1).sum())},标签基率 {y[lab_mask].mean():.3f});"
          f"需求 {len(set(sids))}", flush=True)

    results = {"n_rows": len(rows), "n_labeled": int(lab_mask.sum()),
               "base_rate_labeled": round(float(y[lab_mask].mean()), 4),
               "arms": {}, "discrimination": {}, "hardcore_sep": {},
               "notes": "预注册 v2 全池悲观界;分层禁并池;exit(z0)行只训练不入回放"}

    # ---- 折外分数:主口径(sid 分组)×{lr, gbt×3seed} + 严格口径(族留出) ----
    scores = {}
    scores[("main", "lr", 0)] = oof_scores(rows, y, sids, True, "lr")
    for sd in SEEDS:
        scores[("main", "gbt", sd)] = oof_scores(rows, y, sids, True, "gbt", sd)
    scores[("strict", "lr", 0)] = fam_oof_scores(rows, y, "lr")
    for sd in SEEDS:
        scores[("strict", "gbt", sd)] = fam_oof_scores(rows, y, "gbt", sd)

    # ---- 判别力(分层;仅带标签行;并池数字仅并列展示) ----
    layers = {"val8": VAL8, "transfer6": TRANSFER6}
    for (scope, kind, sd), s in scores.items():
        key = f"{scope}/{kind}" + (f"/s{sd}" if kind == "gbt" else "")
        ent = {}
        for lname, fams in layers.items():
            idx = [i for i, r in enumerate(rows) if r["fam"] in fams and y[i] >= 0]
            ent[lname] = metrics(y[idx], s[idx])
        li = np.where(lab_mask)[0]
        ent["pooled_display_only"] = metrics(y[li], s[li])
        results["discrimination"][key] = ent

    # ---- 回放账(每臂 × 层 × 打分器;基线 = 原顺序) ----
    def layer_rows(arm, fams):
        idx = [i for i, r in enumerate(rows)
               if r["model"] == arm and r["fam"] in fams and r["z_seed"] != "0"]
        return idx

    for arm in ("s1", "s2", "lg"):
        results["arms"][arm] = {}
        for lname, fams in layers.items():
            idx = layer_rows(arm, fams)
            if not idx:
                continue
            sub = [rows[i] for i in idx]
            base = replay_account(sub, [0.0] * len(sub))   # 原 z 序 = 当年实测
            rnd_cpds = []
            rng = np.random.default_rng(7)
            for _ in range(100):
                rnd_cpds.append(replay_account(
                    sub, rng.random(len(sub)))["calls_per_delivered"])
            ent = {"baseline_zorder": base,
                   "random_reorder_mean_sd_pessimistic": [
                       round(float(np.mean(rnd_cpds)), 3),
                       round(float(np.std(rnd_cpds)), 3)]}
            for (scope, kind, sd), s in scores.items():
                key = f"{scope}/{kind}" + (f"/s{sd}" if kind == "gbt" else "")
                ent[key] = replay_account(sub, [s[i] for i in idx])
            results["arms"][arm][lname] = ent

    # ---- 硬核识别分离度:需求级 max 折外分数,solved vs unsolved ----
    for (scope, kind, sd), s in scores.items():
        if kind == "gbt" and sd != SEEDS[0]:
            continue
        key = f"{scope}/{kind}"
        ent = {}
        for lname, fams in layers.items():
            for arm in ("s1", "s2", "lg"):
                idx = layer_rows(arm, fams)
                if not idx:
                    continue
                mx, solved = defaultdict(float), {}
                for i in idx:
                    sid = rows[i]["sid"]
                    mx[sid] = max(mx[sid], s[i])
                    solved[sid] = solved.get(sid, 0) or (rows[i]["label"] == "ok")
                ys = np.asarray([0 if solved[k2] else 1 for k2 in mx])   # 1=unsolved
                ss = np.asarray([-mx[k2] for k2 in mx])                  # 低分→硬
                ent[f"{arm}/{lname}"] = metrics(ys, ss)
        results["hardcore_sep"][key] = ent

    outp = d / a.out
    outp.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez(d / "oof_scores.npz",
             row_key=np.asarray([f"{r['model']}|{r['z_seed']}|{r['sid']}" for r in rows]),
             **{f"{sc}_{kd}_{sd}": s for (sc, kd, sd), s in scores.items()})

    # ---- stdout 摘要(主读数) ----
    print("\n== 主判据:s1 交付口径 每交付仿真次数(val8;悲观界)==")
    v = results["arms"]["s1"].get("val8", {})
    if v:
        print(f"  基线(原顺序=实测): {v['baseline_zorder']['calls_per_delivered']}"
              f"(随机重排悲观 {v['random_reorder_mean_sd_pessimistic']})")
        for kk, vv in v.items():
            if "/" in kk:
                print(f"  {kk:18s}: {vv['calls_per_delivered']}"
                      f"  top2/4率 {vv['top_m_rate'][2]}/{vv['top_m_rate'][4]}")
    print(f"[train] 完整结果 → {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
