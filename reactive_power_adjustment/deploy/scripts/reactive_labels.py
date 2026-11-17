"""无功调整推理器标签生成共享层(方向二选项 B;作动器 = 机端电压设定 v_set)。

与有功 adjust_labels 同构,核心差异(立项草案第 2 版 + S0 探针结论):
- 作动器 = 机端电压设定 v_set(GenView.v0),标签 = Δv_set(非机组有功 ΔP);
- 判据**双边**:健康 = 收敛且 vmin>=VIO_LOW 且 vmax<=VIO_HIGH(VIO_HIGH=1.08,S0 定);
- 病态两族(RD-4):**A 族 = 加无功负荷 ql 造病态(机制独立,训练+评估主体,规则打标)**;
  **B 族 = v_set 逆扰动病态(辅批,反向即修法,如实标注同义反复)**。

本模块 = 共享纯函数(读 v_set/界、写 Δv_set、双边判据、造病态)+ B 族逆扰动生成器。
A 族规则打标在 reactive_apply_eval.reactive_rule_arm(核心规则修法:targeted v_set 最小贪心
+ vmax 守门 + shunt 兜底);编排在 reactive_rulelabel。纯函数有单测,PSASP 驱动试产点火。

裁判永远 PSASP(D16 唯一平台)。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.psasp_io.cards import LoadView  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402
from gridfm.reason.reactive_task import VIO_HIGH, VIO_LOW  # noqa: E402
from scripts.adjust_labels import _referee, read_gens, violations  # noqa: E402

# v_set 可行域(PV 母线目标电压;越界抬到撞 qmax 转 PQ 即失效,PSASP 功能性验收兜底)
VSET_LO, VSET_HI = 0.95, 1.10


def read_ctrl(l5: Card) -> Tuple[Optional[int], Dict[int, dict]]:
    """LF.L5 → (平衡机母线号, {记录序: {bus, v0, qmax, qmin, view}});复用 read_gens 的
    版式自适应与平衡机排除,补 v_set/无功界读数(无功作动器口径)。"""
    slack, gens = read_gens(l5)
    for gi, g in gens.items():
        v = g["view"]
        g["v0"] = v.v0
        g["qmax"] = v.qmax
        g["qmin"] = v.qmin
    return slack, gens


def vset_eff_bounds(cur: float, vset_hi: float = None) -> Tuple[float, float]:
    """v_set 有效界(施加 Δv_set 裁剪用)。固定物理带 [VSET_LO, VSET_HI];cur 已在带外时
    放宽到含 cur(不倒逼回带内,保留原方式设定)。2026-08-23:vset_hi 给定时上界取 min(VSET_HI, vset_hi)
    ——PV 母线电压 = v_set,当前 PSASP 潮流不执行机组无功限制(Q 超 Qmax 仍保持 PV),v_set 超过电压上限必判 vmax 越限。"""
    hi = VSET_HI if vset_hi is None else min(VSET_HI, float(vset_hi))
    return min(VSET_LO, cur), max(hi, cur)


def apply_dvset(gens: Dict[int, dict], deltas: Dict[int, float], vset_hi: float = None) -> Dict[int, float]:
    """把 Δv_set 写入对应 LF.L5 记录的 v_set 字段(经 GenView.v0 setter),裁剪到有效界;
    返回**实际生效**的 Δv_set(裁剪后差值,防"名义扰动被界吃掉"标签失真)。"""
    applied: Dict[int, float] = {}
    for gi, dv in deltas.items():
        g = gens[gi]
        v0 = g["view"].v0
        lo, hi = vset_eff_bounds(v0, vset_hi)
        v1 = float(np.clip(v0 + dv, lo, hi))
        g["view"].v0 = v1
        applied[gi] = v1 - v0
    return applied


def sample_vset_perturbation(gens: Dict[int, dict], rng: np.random.Generator,
                             k: int, d_lo: float = 0.03, d_hi: float = 0.08,
                             sign: int = -1) -> Dict[int, float]:
    """B 族逆扰动:随机选 k 台机,各施 Δv_set = sign×U(d_lo,d_hi)(缺省降 v_set 造低压)。
    返回 {gens 索引: Δv_set}。k 截断到可用机组数。"""
    keys = list(gens)
    k = min(k, len(keys))
    if k == 0:
        return {}
    chosen = rng.choice(len(keys), size=k, replace=False)
    return {keys[int(c)]: sign * float(rng.uniform(d_lo, d_hi)) for c in chosen}


def scale_qload(work: Path, f: float) -> None:
    """A 族病态构造:LF.L6 全负荷无功 ql × f(mark==1;只动 ql = 纯无功驱动压降,
    无功杠杆可修——S0 探针据此造出可修低压病态)。"""
    l6 = Card.load(work / "LF.L6")
    for i, r in enumerate(l6.records):
        lv = LoadView(r, i)
        if lv.mark != 1:
            continue
        lv.ql = lv.ql * f
    l6.save(work / "LF.L6")


def _is_healthy_2s(rf: dict, low: float = VIO_LOW, high: float = VIO_HIGH,
                   eps: float = 5e-4) -> bool:
    """双边健康 = 收敛且 vmin>=low 且 vmax<=high(无功任务缺省双边,与有功单边线相反)。"""
    if not rf.get("converged"):
        return False
    return bool(rf["vmin"] >= low - eps and rf["vmax"] <= high + eps)


def _is_violated_2s(rf: dict, low: float = VIO_LOW, high: float = VIO_HIGH,
                    eps: float = 5e-4) -> bool:
    if not rf.get("converged"):
        return False
    return bool(rf["vmin"] < low - eps or rf["vmax"] > high + eps)


def delta_l2(deltas: Dict[int, float]) -> float:
    return float(np.sqrt(sum(d * d for d in deltas.values()))) if deltas else 0.0


# ----------------- B 族逆扰动生成(辅批;PSASP 驱动,试产点火) -----------------

def generate_bfam(case_dir: Path, work: Path, solver, rng: np.random.Generator,
                  low: float = VIO_LOW, high: float = VIO_HIGH, budget: int = 12,
                  k_lo: int = 3, k_hi: int = 8,
                  d_lo: float = 0.03, d_hi: float = 0.08) -> Optional[dict]:
    """B 族:健康方式 → 降 v_set 造收敛越限病态 → 反向(恢复 v_set)= 可行修法#1。

    **如实标注**:B 族修法 ≈ "把 v_set 复位基态"(同义反复),仅作最小改动量辅批,
    非评估集(S0 实证:B 族复位修复率 1.0,评估会被穿透——RD-4)。source 标 bfam_reverse。
    """
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(case_dir, work)
    rf0 = _referee(work, solver)
    if not _is_healthy_2s(rf0, low, high):
        return {"sid": case_dir.name, "feasible": False, "reason": "起点非健康(双边)",
                "vmin0": round(rf0["vmin"], 4) if rf0.get("converged") else None,
                "vmax0": round(rf0["vmax"], 4) if rf0.get("converged") else None}
    l5 = Card.load(work / "LF.L5")
    _slack, gens = read_ctrl(l5)
    v_orig = {gi: g["view"].v0 for gi, g in gens.items()}

    def _restore(vmap):
        for gi, v in vmap.items():
            gens[gi]["view"].v0 = v
        l5.save(work / "LF.L5")

    found = None
    for _ in range(budget):
        _restore(v_orig)
        k = int(rng.integers(k_lo, k_hi + 1))
        applied = apply_dvset(gens, sample_vset_perturbation(gens, rng, k, d_lo, d_hi, sign=-1))
        l5.save(work / "LF.L5")
        rf = _referee(work, solver)
        if _is_violated_2s(rf, low, high):
            found = {"applied": applied, "referee": rf}
            break
    _restore(v_orig)
    if found is None:
        return {"sid": case_dir.name, "feasible": False,
                "reason": "预算内未造出收敛越限病态(B 族降 v_set)"}
    fix1 = {gi: -dv for gi, dv in found["applied"].items()}
    fixes = [{"deltas": {gens[gi]["bus"]: round(dv, 4) for gi, dv in fix1.items()},
              "l2": round(delta_l2(fix1), 4), "source": "bfam_reverse"}]
    return {
        "sid": case_dir.name, "feasible": True, "family": "B_vset_perturb",
        "violation": {"vmin": round(found["referee"]["vmin"], 4),
                      "vmax": round(found["referee"]["vmax"], 4),
                      "buses": violations(found["referee"]["v"], low, high)},
        "n_fixes": 1, "fixes": fixes, "canonical_fix": fixes[0],
        "antifake_ge2": False, "note": "B族辅批:反向≈复位v_set(同义反复),不作评估集",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="B 族 v_set 逆扰动标签(辅批;A 族主体见 reactive_rulelabel)")
    ap.add_argument("--lib", required=True)
    ap.add_argument("--battery", required=True)
    ap.add_argument("--sids", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-target", type=int, default=30)
    ap.add_argument("--vio-low", type=float, default=VIO_LOW)
    ap.add_argument("--vio-high", type=float, default=VIO_HIGH)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from gridfm.solver.psasp import PsaspSolver
    solver = PsaspSolver()
    cases = Path(a.lib) / "cases"
    sids = a.sids or sorted(p.name for p in cases.iterdir() if p.is_dir())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    rng = np.random.default_rng(a.seed)

    rows: List[dict] = []
    n_feasible = 0
    for sid in sids:
        if n_feasible >= a.n_target:
            break
        cdir = cases / sid
        if not cdir.exists():
            continue
        row = generate_bfam(cdir, work, solver, rng, a.vio_low, a.vio_high, a.budget)
        if row is None:
            continue
        row["battery"] = a.battery
        rows.append(row)
        if row["feasible"]:
            n_feasible += 1
            print(f"[react-B] {sid}: 可行 越限 vmin={row['violation']['vmin']} "
                  f"vmax={row['violation']['vmax']}", flush=True)
        else:
            print(f"[react-B] {sid}: 未产({row['reason']})", flush=True)

    (out / "adjust_labels.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    summary = {"battery": a.battery, "family": "B_vset_perturb", "n_tried": len(rows),
               "n_feasible": n_feasible,
               "hit_rate": round(n_feasible / max(len(rows), 1), 3),
               "vio_band": [a.vio_low, a.vio_high], "criterion": "bilateral",
               "seed": a.seed}
    (out / "adjust_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[react-B] 收尾:可行 {n_feasible}/{len(rows)}(命中率 {summary['hit_rate']})")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
