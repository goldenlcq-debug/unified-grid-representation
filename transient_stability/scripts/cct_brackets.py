"""[G] CCT **括号字段**派生 + 规则-标签自洽重分类(判稳收尾,2026-08-01;零 PSASP)。

## 为什么

二分只给下界:`search_cct` 存的 `cct` = 最后一个稳的 Te(last_stable),真值落在
`[cct, cct + 0.5/2^k)`,k = 二分次数(n_sims = 1 次探上界 + k 次二分)。
实测(定向CCT小批验证-结果-20260727 §4.1/§4.2):已有键上 40 条"规则-标签矛盾"里
**34 条(85%)的 |dur − cct| ≤ 括号宽 = 精度残余而非标签打架**;盲键上 9/9 全是。
把括号落档后,消费方对"dur 落在括号内"的行按**不可判**处理(不入规则判据、不入
L1 回归的强断言),3.35% 的"矛盾监督"即刻降为干净的"不可判"。

## 产出(全部派生件,**不改任何源 cct.jsonl** —— 归档只读铁律)

- `<out>/brackets_<源名>.jsonl`:每行 = 源行 + `cct_lo`(=cct)/ `cct_hi`
  (=cct + 0.5/2^k;删失行 = None)/ `bracket_w`;
- `<out>/bracket_audit.json`:对给定标签批逐行重分类
  {consistent / in_bracket(不可判)/ conflict(真分歧)} 的计数与真分歧清单。

用法:
  python scripts/cct_brackets.py --cct <目录...> --labels <ts_labels 目录...> \
      --out runs/cct_brackets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

T_PROBE = 0.5           # 上界探测点(search_cct 的 hi 起点;与 scripts/ts_cct.py 同源)


def bracket_width(n_sims: int) -> float:
    """残余括号宽 = 0.5 / 2^(二分次数);n_sims = 1(探上界)+ k(二分)。"""
    k = max(0, int(n_sims) - 1)
    return T_PROBE / (2 ** k)


def derive(rows):
    out = []
    for r in rows:
        r = dict(r)
        if r.get("cct") is None or r.get("infeasible"):
            r["cct_lo"] = r["cct_hi"] = r["bracket_w"] = None
        elif r.get("is_censored"):                      # 上侧删失:真值 ≥ T_PROBE
            r["cct_lo"], r["cct_hi"], r["bracket_w"] = float(r["cct"]), None, None
        elif r.get("is_censored_low"):                  # 下侧删失:真值 < 最小步
            r["cct_lo"], r["cct_hi"], r["bracket_w"] = None, float(r["cct"]), None
        else:
            w = bracket_width(r.get("n_sims", 7))
            r["cct_lo"] = float(r["cct"])
            r["cct_hi"] = float(r["cct"]) + w
            r["bracket_w"] = w
        out.append(r)
    return out


def classify(dur: float, lo, hi) -> str:
    """规则判据视角:dur < 真值 → 稳。真值只知道在 [lo, hi) 里。"""
    if lo is not None and dur < lo:
        return "rule_stable"          # dur < lo ≤ 真值 → 可判稳
    if hi is not None and dur >= hi:
        return "rule_unstable"        # dur ≥ hi > 真值 → 可判不稳
    return "in_bracket"               # lo ≤ dur < hi → **不可判**


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cct", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=[])
    ap.add_argument("--out", default="runs/cct_brackets")
    a = ap.parse_args()
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)

    by_key = {}
    for d in a.cct:
        d = Path(d)
        rows = [json.loads(l) for l in (d / "cct.jsonl").open(encoding="utf-8") if l.strip()]
        der = derive(rows)
        f = outd / f"brackets_{d.name}.jsonl"
        with f.open("w", encoding="utf-8") as fh:
            for r in der:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_br = sum(1 for r in der if r.get("bracket_w"))
        print(f"[br] {d.name}: {len(der)} 行 -> {f}(含括号 {n_br})")
        for r in der:
            if r.get("verify", {}).get("ok") is False:
                continue                                 # 双验不过的键不供判据
            by_key[(r["battery"], r["tag"])] = r

    audit = {"n_keys": len(by_key), "by_label_dir": {}, "conflicts": []}
    for d in a.labels:
        d = Path(d)
        lf = d / "labels_v1.jsonl"
        if not lf.is_file():
            lf = d / "labels.jsonl"
        # 与 loader 同口径:superseded 方式的标签描述的是已作废的运行点(W1 抬压重做),
        # 不剔会把"旧运行点 vs 新 CCT"的错配当成真分歧报出来。
        sup = set()
        sp = d / "superseded_sids.json"
        if sp.is_file():
            j = json.loads(sp.read_text(encoding="utf-8"))
            sup = set(j if isinstance(j, list) else j.get("sids", []))
        cnt = {"consistent": 0, "in_bracket": 0, "conflict": 0, "no_key": 0, "superseded": 0}
        for l in lf.open(encoding="utf-8"):
            if not l.strip():
                continue
            row = json.loads(l)
            if not row.get("feasible"):
                continue
            if row.get("sample_id") in sup:
                cnt["superseded"] += 1
                continue
            tag = row["tag"]
            base_tag = tag.rsplit("_d", 1)[0] if "_d" in tag else tag
            r = by_key.get((row["battery"], base_tag))
            if r is None or r.get("cct") is None:
                cnt["no_key"] += 1
                continue
            verdict = classify(round(float(row["dur"]), 4), r.get("cct_lo"), r.get("cct_hi"))
            if verdict == "in_bracket":
                cnt["in_bracket"] += 1
            elif (verdict == "rule_stable") == bool(row["stable"]):
                cnt["consistent"] += 1
            else:
                cnt["conflict"] += 1
                audit["conflicts"].append({"battery": row["battery"], "tag": tag,
                                           "dur": row["dur"], "stable": row["stable"],
                                           "cct_lo": r.get("cct_lo"), "cct_hi": r.get("cct_hi")})
        audit["by_label_dir"][d.name] = cnt
        tot = cnt["consistent"] + cnt["in_bracket"] + cnt["conflict"]
        print(f"[br] {d.name}: 覆盖 {tot} 行 = 自洽 {cnt['consistent']} + "
              f"不可判 {cnt['in_bracket']} + **真分歧 {cnt['conflict']}**"
              f"(无键 {cnt['no_key']} / superseded 剔 {cnt['superseded']})")
    (outd / "bracket_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[br] -> {outd / 'bracket_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
