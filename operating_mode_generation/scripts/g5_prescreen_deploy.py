"""[G] ① 预筛头 · 部署版训练 + 新标签消融(2026-08-03;纯 CPU 零仿真)。

c 项:把补标 210 条(labels_stage1.json)并入 3,294 条旧标签,训部署版 LR 头
(交付批产 g6_prescreen_delivery.py 消费),并做消融回答"新标签买到了什么"。

消融口径(防选择偏差):新 210 条全部来自 main_lr top-2 高分区(选择性标注),
故**公平测试集 = 旧标签行**(原始分布);训练臂 A = 仅旧标签 / 臂 B = 旧+新,
同折外(GroupKFold sid 5 折)对比 val8 AUROC/PR;含新行的测试数字并报但标注偏差。

部署头落盘 = runs/g5_prescreen/deploy_head.json(LR 系数 + scaler + 特征次序 +
训练数据指纹,可审计免 pickle)。

  python scripts/g5_prescreen_deploy.py
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "runs/g5_prescreen"
P = REPO / "runs/g5_probe"

NUM_FEATS = ["n_nodes", "req_load", "req_online", "req_ngen",
             "gen_load", "gen_online", "load_rel_dev", "online_dev",
             "pg_mean", "pg_std", "pg_max", "pg_neg", "ql_std", "pl_std",
             "vset_dev_mean", "vset_dev_std", "vset_dev_maxabs", "vset_hi",
             "z0_std", "z0_maxabs", "z0_clip_frac"]
VAL8 = {"synth210_s2007", "synth150_s2000", "synth210_s2002", "synth240_s2003",
        "synth270_s2004", "synth150_s2005", "synth180_s2001", "synth180_s2006"}
TRANSFER6 = {"synth400_s3000", "synth400_s3001", "synth400_s3002", "synth400_s3003",
             "synth700_s3004", "synth1000_s3008"}


def load_all():
    rows = []
    for p in sorted(D.glob("features_*.csv")):
        if p.stem.count("_") > 1:
            continue
        with open(p, encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    seen, uniq = set(), []
    for r in rows:
        k = (r["model"], r["z_seed"], r["sid"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    probe = json.loads((P / "labels_stage1.json").read_text(encoding="utf-8"))
    assert not probe["partial"]
    patch = {k: v["label"] for k, v in probe["rows"].items()}
    n_hit = 0
    for r in uniq:
        key = f"{r['fam']}|{r['z_seed']}|{r['sid']}"
        if key in patch and r["model"] == "s1":
            assert r["label"] == "unknown"
            r["label"] = patch[key]
            r["_new"] = True
            n_hit += 1
        else:
            r["_new"] = False
    assert n_hit == len(patch), (n_hit, len(patch))
    return [r for r in uniq if r["label"] != "unknown"]


def featurize(rows, fams, models, with_fam=True):
    X = []
    for r in rows:
        v = [float(r[c]) for c in NUM_FEATS]
        v += [1.0 if r["model"] == m else 0.0 for m in models]
        if with_fam:
            v += [1.0 if r["fam"] == f else 0.0 for f in fams]
        X.append(v)
    return np.asarray(X, dtype=np.float64)


def oof_auc(rows, X, y, train_mask_extra, label_note):
    """GroupKFold(sid);训练折附加过滤 train_mask_extra;测试 = 旧标签行(公平)。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    sids = np.asarray([r["sid"] for r in rows])
    oof = np.full(len(rows), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, sids):
        trl = tr[train_mask_extra[tr]]
        sc = StandardScaler().fit(X[trl])
        m = LogisticRegression(max_iter=2000, class_weight="balanced")
        m.fit(sc.transform(X[trl]), y[trl])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    out = {}
    for lname, fams_ in (("val8", VAL8), ("transfer6", TRANSFER6)):
        for test_scope, mask in (("旧行(公平)", [not r["_new"] for r in rows]),
                                 ("全行(含偏差)", [True] * len(rows))):
            idx = [i for i, r in enumerate(rows)
                   if r["fam"] in fams_ and mask[i]]
            yy, ss = y[idx], oof[idx]
            if len(set(yy)) < 2:
                continue
            out[f"{lname}/{test_scope}"] = {
                "auroc": round(float(roc_auc_score(yy, ss)), 4),
                "pr_auc": round(float(average_precision_score(yy, ss)), 4),
                "n": len(idx), "base": round(float(np.mean(yy)), 4)}
    print(f"[deploy] {label_note}:")
    for k, v in out.items():
        print(f"   {k:22s} auroc {v['auroc']} pr {v['pr_auc']} (n={v['n']} base {v['base']})")
    return out


def main():
    rows = load_all()
    fams = sorted({r["fam"] for r in rows})
    models = sorted({r["model"] for r in rows})
    y = np.asarray([1 if r["label"] == "ok" else 0 for r in rows])
    is_old = np.asarray([not r["_new"] for r in rows])
    X = featurize(rows, fams, models)
    print(f"[deploy] 标签行 {len(rows)}(旧 {int(is_old.sum())} + 新 {int((~is_old).sum())}),"
          f"ok {int(y.sum())}", flush=True)

    # ---- 消融:臂 A(仅旧)vs 臂 B(旧+新),测试均以旧行为公平集 ----
    resA = oof_auc(rows, X, y, is_old, "臂A 仅旧标签训练")
    resB = oof_auc(rows, X, y, np.ones(len(rows), bool), "臂B 旧+新标签训练")

    # ---- 部署头:全量标签训练(单 LR,确定性) ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=2000, class_weight="balanced")
    m.fit(sc.transform(X), y)
    feat_names = (NUM_FEATS + [f"model={m_}" for m_ in models]
                  + [f"fam={f}" for f in fams])
    data_fp = hashlib.blake2b(
        ("|".join(sorted(f"{r['model']}|{r['z_seed']}|{r['sid']}|{r['label']}"
                         for r in rows))).encode(), digest_size=16).hexdigest()
    head = {
        "kind": "logreg+standardscaler", "trained": "2026-08-03",
        "n_train": len(rows), "n_new_probe": int((~is_old).sum()),
        "train_data_fp": data_fp,
        "feat_names": feat_names,
        "scaler_mean": [round(float(v), 8) for v in sc.mean_],
        "scaler_scale": [round(float(v), 8) for v in sc.scale_],
        "coef": [round(float(v), 8) for v in m.coef_[0]],
        "intercept": round(float(m.intercept_[0]), 8),
        "ablation": {"armA_old_only": resA, "armB_old_plus_new": resB},
        "note": "score = sigmoid(coef·(x-mean)/scale + intercept);新族 fam one-hot 全零",
    }
    out = D / "deploy_head.json"
    out.write_text(json.dumps(head, ensure_ascii=False, indent=1), encoding="utf-8")
    # 自检:json 手算与 sklearn 输出逐位一致(前 5 行)
    z = (X[:5] - sc.mean_) / sc.scale_
    manual = 1 / (1 + np.exp(-(z @ np.asarray(head["coef"]) + head["intercept"])))
    skl = m.predict_proba(sc.transform(X[:5]))[:, 1]
    assert np.allclose(manual, skl, atol=1e-6), "json 头与 sklearn 不一致"
    print(f"[deploy] 部署头 → {out}(json 手算自检过;数据指纹 {data_fp[:8]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
