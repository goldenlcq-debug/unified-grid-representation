"""调整方案功能性验收(方向二 R3 DoD;7.3 评估侧免疫:不比对标签,只认施加后有效)。

对留出集病态方式(adjust_materialize 目录)逐例:施加候选 ΔP 方案 → PSASP 复核
**越限消除(vmin≥线)且收敛** → 一次通过率;并报 ‖ΔP‖₁ 相对规范解之比(代价口径)。

四个臂(D-4 用户批定;--arms 选跑):
- model:adjust_reasoner 导出的 eval_predictions.jsonl;
- rule:规则基线 = 越限深度加权的余量定向增出力(share_by_headroom 先例:低压主因
  输送重载,就地/全网按上行余量比例加出力、平衡机降,幅度 = 阶梯 [0.3,0.6,1.0]×需求
  逐级试,首个通过即停,记实际用挡)——**零学习,ML 打不赢它不上**;
- trivial:平凡基线 = 单机 top-1(越限最深母线电气最近的机组,加 20% 容量幅度);
- oracle:规范解照抄(按构造应全过;锚定通过率上限与 ‖ΔP‖ 分母)。

  python scripts/adjust_apply_eval.py --cases runs/adjust_cases_cepri36 \
      --labels runs/adjust_labels_cepri36_r1c ... --preds runs/adjust_reasoner_s0 \
      --arms model rule trivial oracle --out runs/adjust_eval_s0
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.psasp_io.lowlevel import Card  # noqa: E402
from scripts.adjust_labels import (  # noqa: E402
    VIO_LOW_DEFAULT, _is_healthy, _referee, delta_l1, eff_bounds, read_gens,
    share_by_headroom,
)
from scripts.adjust_materialize import apply_fix_by_bus  # noqa: E402


class CountingSolver:
    """透明包裹真 solver,只计 run_powerflow 调用次数(选项 A① 求解器调用计量)。

    不改任何求解行为(逐位透传),故计量为被动观测——挂它前后通过率必逐位一致。
    每次 run_powerflow = 一次 PSASP 潮流调用(psasp_referee 恰调 1 次,已核)。"""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.n_pf = 0

    def run_powerflow(self, *args, **kwargs):
        self.n_pf += 1
        return self._inner.run_powerflow(*args, **kwargs)

    def __getattr__(self, name):
        # _inner 在 __init__ 首行入 __dict__,不会递归;其余属性(run_transient/bin 等)透传
        return getattr(object.__getattribute__(self, "_inner"), name)


def _load_rows(labels_dirs) -> Dict[str, Dict]:
    """(batch, row_index) → 标签行(规范解/oracle 与 ‖ΔP‖ 分母来源)。"""
    out = {}
    for lab in labels_dirs:
        lab = Path(lab)
        rows = [json.loads(ln) for ln in
                (lab / "adjust_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        for i, r in enumerate(rows):
            out[(lab.name, i)] = r
    return out


def rule_arm(work: Path, solver, vio_low: float) -> Optional[Dict[str, float]]:
    """规则基线:全网按上行余量比例加出力(平衡机吸收),需求 = 越限深度总量的电纳级
    代理(经验换算:sum_depth × 全网机组容量幅度和 × 0.5),阶梯 [0.3,0.6,1.0] 逐级试,
    首个通过即返回该方案;全试不过返回 None(如实记失败)。每级 1 次 PSASP。"""
    l5 = Card.load(work / "LF.L5")
    _s, gens = read_gens(l5)
    p0 = {gi: g["view"].pg for gi, g in gens.items()}
    rf = _referee(work, solver)
    if not rf.get("converged"):
        return None
    depth = float(np.maximum(0.0, vio_low - rf["v"][rf["v"] > 0.01]).sum())
    span_sum = sum(max(g["pmax"] - g["pmin"], abs(g["p"]), 1e-3) for g in gens.values())
    need0 = depth * span_sum * 0.5
    states = {gi: {"p": g["view"].pg, "pmin": g["pmin"], "pmax": g["pmax"]}
              for gi, g in gens.items()}
    for lvl in (0.3, 0.6, 1.0):
        cand = share_by_headroom(states, list(gens), lvl * need0)
        if not cand:
            break
        for gi, p in p0.items():
            gens[gi]["view"].pg = p
        for gi, dp in cand.items():
            g = gens[gi]
            lo, hi = eff_bounds(g["pmin"], g["pmax"], p0[gi])
            g["view"].pg = float(np.clip(p0[gi] + dp, lo, hi))
        l5.save(work / "LF.L5")
        rf2 = _referee(work, solver)
        if _is_healthy(rf2, vio_low, 9.9, vmax_gate=False):
            return {str(gens[gi]["bus"]): round(dp, 4) for gi, dp in cand.items()}
    for gi, p in p0.items():                     # 还原
        gens[gi]["view"].pg = p
    l5.save(work / "LF.L5")
    return None


def trivial_arm(work: Path, solver, vio_low: float) -> Optional[Dict[str, float]]:
    """平凡基线:越限最深母线 → 图上最近机组(BFS 跳数,同深度取小母线号)加 20% 容量
    幅度。1 次 PSASP。"""
    from gridfm.psasp_io.case import PsaspCase
    l5 = Card.load(work / "LF.L5")
    _s, gens = read_gens(l5)
    rf = _referee(work, solver)
    if not rf.get("converged"):
        return None
    v = rf["v"]
    live = v > 0.01
    worst = int(np.argmin(np.where(live, v, 9.9)))          # 0 基
    case = PsaspCase(work)
    n = case.n_buses
    adj: Dict[int, List[int]] = {}
    for ln in case.aclines():
        if ln.mark != 1:
            continue
        adj.setdefault(ln.i_no - 1, []).append(ln.j_no - 1)
        adj.setdefault(ln.j_no - 1, []).append(ln.i_no - 1)
    gen_rows = {}
    for gi, g in gens.items():
        gen_rows.setdefault(g["bus"] - 1, gi)
    # BFS 找最近机组
    seen = {worst}
    frontier = [worst]
    target_gi = gen_rows.get(worst)
    while frontier and target_gi is None:
        nxt = []
        for b in frontier:
            for nb in sorted(adj.get(b, [])):
                if nb in seen:
                    continue
                seen.add(nb)
                if nb in gen_rows:
                    target_gi = gen_rows[nb]
                    break
                nxt.append(nb)
            if target_gi is not None:
                break
        frontier = nxt
    if target_gi is None:
        return None
    g = gens[target_gi]
    span = max(g["pmax"] - g["pmin"], abs(g["p"]), 1e-3)
    _lo, hi = eff_bounds(g["pmin"], g["pmax"], g["view"].pg)
    dp = min(0.2 * span, hi - g["view"].pg)     # 未设限 hi=inf → dp=0.2×span
    if dp <= 1e-6:
        return None
    g["view"].pg = g["view"].pg + dp
    l5.save(work / "LF.L5")
    rf2 = _referee(work, solver)
    ok = _is_healthy(rf2, vio_low, 9.9, vmax_gate=False)
    g["view"].pg = g["view"].pg - dp
    l5.save(work / "LF.L5")
    return {str(g["bus"]): round(dp, 4)} if ok else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--preds", default=None,
                    help="adjust_reasoner 输出目录(model 臂;含 eval_predictions.jsonl)")
    ap.add_argument("--arms", nargs="+", default=["model", "rule", "trivial", "oracle"])
    ap.add_argument("--vio-low", type=float, default=VIO_LOW_DEFAULT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from gridfm.solver.psasp import PsaspSolver
    solver = PsaspSolver()
    rows_by_key = _load_rows(a.labels)
    manifest = []
    for cd in a.cases:
        for ln in (Path(cd) / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                m = json.loads(ln)
                m["cases_dir"] = cd
                manifest.append(m)
    by_tag = {m["tag"]: m for m in manifest}

    preds_by_tag = {}
    if "model" in a.arms:
        if not a.preds:
            raise SystemExit("[eval] model 臂需要 --preds")
        for ln in (Path(a.preds) / "eval_predictions.jsonl").read_text(
                encoding="utf-8").splitlines():
            if ln.strip():
                p = json.loads(ln)
                preds_by_tag[p["tag"]] = p
        tags = [t for t in preds_by_tag if t in by_tag]     # 评估域 = 留出集预测
    else:
        tags = sorted(by_tag)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    results = {arm: {"pass": 0, "fail": 0, "l1_ratio": []} for arm in a.arms}
    per_case = []
    for tag in tags:
        m = by_tag[tag]
        row = rows_by_key[(m["batch"], m["row_index"])]
        canon_l1 = delta_l1({b: d for b, d in row["canonical_fix"]["deltas"].items()})
        src = Path(m["cases_dir"]) / "cases" / tag
        rec = {"tag": tag}
        for arm in a.arms:
            if work.exists():
                shutil.rmtree(work, ignore_errors=True)
            shutil.copytree(src, work)
            deltas = None
            if arm == "oracle":
                deltas = row["canonical_fix"]["deltas"]
            elif arm == "model":
                deltas = preds_by_tag[tag]["deltas"]
            elif arm == "rule":
                deltas = rule_arm(work, solver, a.vio_low)
            elif arm == "trivial":
                deltas = trivial_arm(work, solver, a.vio_low)
            if arm in ("oracle", "model") and deltas is not None:
                l5 = Card.load(work / "LF.L5")
                apply_fix_by_bus(l5, deltas)
                l5.save(work / "LF.L5")
                rf = _referee(work, solver)
                ok = _is_healthy(rf, a.vio_low, 9.9, vmax_gate=False)
            else:
                ok = deltas is not None       # rule/trivial 已在臂内验收
            results[arm]["pass" if ok else "fail"] += 1
            rec[arm] = bool(ok)
            if ok and deltas:
                results[arm]["l1_ratio"].append(
                    round(delta_l1({b: float(d) for b, d in deltas.items()})
                          / max(canon_l1, 1e-9), 3))
        per_case.append(rec)
        print(f"[eval] {tag}: " + " ".join(f"{k}={rec.get(k)}" for k in a.arms), flush=True)

    summary = {"n": len(tags), "vio_low": a.vio_low, "arms": {}}
    for arm in a.arms:
        r = results[arm]
        n = r["pass"] + r["fail"]
        summary["arms"][arm] = {
            "pass_rate": round(r["pass"] / max(n, 1), 3), "n": n,
            "l1_over_canonical_median": (round(float(np.median(r["l1_ratio"])), 3)
                                         if r["l1_ratio"] else None)}
    (out / "apply_eval.json").write_text(json.dumps(
        {"summary": summary, "per_case": per_case}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[eval] 收尾:{json.dumps(summary['arms'], ensure_ascii=False)}")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
