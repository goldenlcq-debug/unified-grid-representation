"""[G] ① 补标后双界与主判据重算(2026-08-03;纯 CPU;预注册 = 补标预注册 §2)。

判定一律用整数 total_calls(m1):过门 ⇔ 悲观 ≤372;判负 ⇔ 乐观 ≥373;否则夹门。
分数 = 冻结 main_lr_0(不重训);分母 delivered 恒 148(断言)。
并报:cpd 展示值、top-m 真值命中、其他打分器双界、阶段 2 徒劳性下限(m3)。

  python scripts/g5_probe_eval.py --after 1     # 阶段 1 后
  python scripts/g5_probe_eval.py --after 2     # 阶段 2 后
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
D = REPO / "runs/g5_prescreen"
P = REPO / "runs/g5_probe"
GATE_CALLS = 372          # ⌊2.52×148⌋
VAL8 = {"synth210_s2007", "synth150_s2000", "synth210_s2002", "synth240_s2003",
        "synth270_s2004", "synth150_s2005", "synth180_s2001", "synth180_s2006"}


def load_pool():
    rows = []
    with open(D / "features_s1.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["fam"] in VAL8 and r["z_seed"] != "0":
                rows.append(r)
    return rows


def apply_probe(rows, stages):
    patch = {}
    for st in stages:
        p = P / f"labels_stage{st}.json"
        rep = json.loads(p.read_text(encoding="utf-8"))
        assert not rep["partial"], f"labels_stage{st} 仍 partial,不得重算"
        for key, v in rep["rows"].items():
            patch[key] = v["label"]
    n_hit = 0
    for r in rows:
        key = f"{r['fam']}|{r['z_seed']}|{r['sid']}"
        if key in patch:
            assert r["label"] == "unknown", f"补标键原标签非 unknown:{key}"
            r["label"] = patch[key]
            n_hit += 1
    assert n_hit == len(patch), f"补标键未全部命中池:{n_hit}/{len(patch)}"
    return rows


def bounds(rows, score_name, extra_ok_keys=frozenset()):
    z = np.load(D / "oof_scores.npz", allow_pickle=False)
    key2i = {k: i for i, k in enumerate(z["row_key"])}
    s = z[score_name]
    by_sid = defaultdict(list)
    for r in rows:
        sc = float(s[key2i[f"s1|{r['z_seed']}|{r['sid']}"]])
        lab = r["label"]
        if f"{r['fam']}|{r['z_seed']}|{r['sid']}" in extra_ok_keys:
            lab = "ok"                       # 徒劳性下限:假设名单全 ok
        by_sid[r["sid"]].append((int(r["z_seed"]), sc, lab))
    pess = opt = 0
    delivered = 0
    topm = {2: 0, 4: 0, 8: 0}
    for sid, lst in by_sid.items():
        lst.sort(key=lambda t: t[0])
        order = sorted(range(len(lst)), key=lambda i: (-lst[i][1], i))
        labs = [lst[i][2] for i in order]
        if "ok" in labs:
            delivered += 1
            pos = labs.index("ok") + 1
            pess += pos
            for m in topm:
                if pos <= m:
                    topm[m] += 1
            first_o = pos
            if "unknown" in labs:
                first_o = min(pos, labs.index("unknown") + 1)
            opt += first_o
        else:
            pess += len(labs)
            opt += (labs.index("unknown") + 1) if "unknown" in labs else len(labs)
    return dict(pess_calls=pess, opt_calls=opt, delivered=delivered,
                n_req=len(by_sid), pess_cpd=round(pess / delivered, 3),
                opt_cpd=round(opt / delivered, 3),
                topm={m: round(v / len(by_sid), 4) for m, v in topm.items()})


def verdict(b):
    if b["pess_calls"] <= GATE_CALLS:
        return "过门(悲观界整数口径)"
    if b["opt_calls"] >= GATE_CALLS + 1:
        return "判负(乐观界整数口径)"
    return "仍夹门"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", type=int, required=True, choices=(1, 2))
    a = ap.parse_args()
    rows = apply_probe(load_pool(), range(1, a.after + 1))
    out = {"gate_calls": GATE_CALLS, "after_stage": a.after, "scorers": {}}
    main_b = bounds(rows, "main_lr_0")
    assert main_b["delivered"] == 148, f"分母漂移:{main_b['delivered']}"
    out["scorers"]["main_lr_0(主报)"] = {**main_b, "verdict": verdict(main_b)}
    for sn in ("main_gbt_0", "strict_lr_0", "strict_gbt_0"):
        b = bounds(rows, sn)
        out["scorers"][sn] = {**b, "verdict": verdict(b)}
    if a.after == 1:                        # m3 徒劳性下限
        pl = json.load(open(D / "probe_list.json", encoding="utf-8"))
        s2keys = {f"{x['fam']}|{x['z_seed']}|{x['sid']}" for x in pl["stage2_top3to4"]}
        fut = bounds(rows, "main_lr_0", extra_ok_keys=s2keys)
        out["stage2_futility"] = {
            "pess_calls_if_stage2_all_ok": fut["pess_calls"],
            "gate_reachable": fut["pess_calls"] <= GATE_CALLS}
    p = P / f"eval_after_stage{a.after}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    mb = out["scorers"]["main_lr_0(主报)"]
    print(f"[probe-eval] 主报 main_lr_0:悲观 {mb['pess_calls']} calls"
          f"(cpd {mb['pess_cpd']})/ 乐观 {mb['opt_calls']}({mb['opt_cpd']})"
          f" vs 门 {GATE_CALLS} → **{mb['verdict']}**")
    print(f"[probe-eval] top2/4/8 = {mb['topm']};delivered {mb['delivered']}/160")
    if "stage2_futility" in out:
        f2 = out["stage2_futility"]
        print(f"[probe-eval] 阶段2 徒劳性:全 ok 假设悲观 {f2['pess_calls_if_stage2_all_ok']}"
              f" → 过门{'可达' if f2['gate_reachable'] else '不可达'}")
    for sn, b in out["scorers"].items():
        if "主报" in sn:
            continue
        print(f"[probe-eval] {sn}: [{b['opt_calls']}, {b['pess_calls']}] → {b['verdict']}")
    print(f"[probe-eval] → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
