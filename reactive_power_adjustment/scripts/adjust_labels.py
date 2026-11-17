"""有功调整推理器标签生成(方向二 R1,逆扰动 v0;方案 7.3 调整协议 + 防伪三件套)。

任务(与 A 线潮流调收敛区分):A 线 = 不收敛→收敛(收敛性);本脚本 = **已收敛但电压
越限 → 合格**(安全性)。做法:从健康方式(收敛且电压在带内)出发,对若干非平衡机组
施加 ΔP(平衡机吸收),用 PSASP 验证直到造出**收敛但越限**的病态方式;则该扰动的**反向**
(恢复原出力)就是一个已知可行修法。防伪:对同一病态搜 ≥2 个不同的可行修法全部入库;
规范解 = 最小 ‖ΔP‖ 解。评估一律用功能性判据(施加后 PSASP 验证越限消除且收敛),不比对
构造标签(7.3 评估侧免疫)。

**裁判永远 PSASP(D16 唯一平台)。** 越限判据(D-1 甲,用户 2026-07-18 采纳):
vmin<0.93 或 vmax>1.05 为越限;健康起点 = 收敛且 vmin≥0.93 且 vmax≤1.05。

  python scripts/adjust_labels.py --lib <数据根>/lib_cepri36 --battery cepri36 \
      --sids <方式号...> --out runs/adjust_labels_cepri36 --n-target 30

纯函数(read_gens/apply_deltas/sample_perturbation/violations/delta_l2)有单测
(tests/test_adjust_labels.py),无需 PSASP;PSASP 驱动在 main(),试产时点火。
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

from gridfm.psasp_io.lowlevel import Card  # noqa: E402

# LF.L5 解析一律走 psasp_io.GenView(版式自动探测 + 后验校验):存在两种版式——
# 基准版 20 单元格(ieee118/合成器)与带机组序号列版 21 单元格(ieee39/cepri36)。
# **首版试产事故如实记(2026-07-18)**:曾手工硬编码 21 格版索引,在 lib118(20 格版)
# 上类型列读到浮点直接崩——正是 GenView 注释里 2026-07-11 事故的重演;修法 = 回解析层,
# 禁止再手工索引。
from gridfm.psasp_io.cards import GenView  # noqa: E402

VIO_LOW_DEFAULT, VIO_HIGH_DEFAULT = 0.93, 1.05    # D-1 甲(用户采纳)


def read_gens(l5: Card) -> Tuple[Optional[int], Dict[int, dict]]:
    """LF.L5 → (平衡机母线号, {记录序: {bus, p, pmin, pmax, view}})。

    经 GenView 版式自适应读取;mark≠1(停运)跳过;平衡机(type_code=0)单列
    (不扰动,PSASP 吸收不平衡量);其余机组记录入候选(调整对象 = 机组记录)。
    """
    slack = None
    gens: Dict[int, dict] = {}
    for idx, r in enumerate(l5.records):
        g = GenView(r, idx)
        if g.mark != 1:
            continue
        if g.type_code == 0:
            slack = g.bs_no
            continue
        gens[idx] = {"bus": g.bs_no, "p": g.pg,
                     "pmin": g.pmin, "pmax": g.pmax, "view": g}
    return slack, gens


def sample_perturbation(gens: Dict[int, dict], rng: np.random.Generator,
                        k: int, frac_lo: float = 0.10,
                        frac_hi: float = 0.40) -> Dict[int, float]:
    """随机选 k 台机组,各施加 ΔP = ±(frac × 容量幅度),方向随机(7.3:10%~40%)。

    容量幅度 = max(pmax−pmin, |p|, 1e-3)(缺容量界时退回当前出力幅度,防零幅)。
    返回 {gens 索引: ΔP}。k 截断到可用机组数。
    """
    keys = list(gens)
    k = min(k, len(keys))
    if k == 0:
        return {}
    chosen = rng.choice(len(keys), size=k, replace=False)
    out: Dict[int, float] = {}
    for c in chosen:
        gi = keys[int(c)]
        g = gens[gi]
        span = max(g["pmax"] - g["pmin"], abs(g["p"]), 1e-3)
        frac = rng.uniform(frac_lo, frac_hi)
        sign = 1.0 if rng.random() < 0.5 else -1.0
        out[gi] = sign * frac * span
    return out


def apply_deltas(gens: Dict[int, dict], deltas: Dict[int, float]) -> Dict[int, float]:
    """把 ΔP 写入对应 LF.L5 记录的出力字段(经 GenView.pg setter,版式自适应),裁剪到
    [pmin, pmax];返回**实际生效**的 ΔP(裁剪后差值,防"名义扰动被界吃掉"时标签失真)。"""
    applied: Dict[int, float] = {}
    for gi, dp in deltas.items():
        g = gens[gi]
        p0 = g["view"].pg
        lo, hi = eff_bounds(g["pmin"], g["pmax"], p0)
        p1 = float(np.clip(p0 + dp, lo, hi))
        g["view"].pg = p1
        applied[gi] = p1 - p0
    return applied


def violations(v: np.ndarray, low: float = VIO_LOW_DEFAULT,
               high: float = VIO_HIGH_DEFAULT) -> Dict[int, str]:
    """电压向量(0 基索引)→ {母线号(1 基): 'low'|'high'}。仅计 v>0.01 的带电母线。"""
    out: Dict[int, str] = {}
    for b in range(len(v)):
        if v[b] <= 0.01:
            continue
        if v[b] < low:
            out[b + 1] = "low"
        elif v[b] > high:
            out[b + 1] = "high"
    return out


def delta_l2(deltas: Dict[int, float]) -> float:
    """‖ΔP‖₂(规范解选择用:同一病态多可行解取最小者)。"""
    return float(np.sqrt(sum(d * d for d in deltas.values()))) if deltas else 0.0


def delta_l1(deltas: Dict[int, float]) -> float:
    return float(sum(abs(d) for d in deltas.values()))


def eff_bounds(pmin: float, pmax: float, cur: float) -> tuple:
    """有效出力界(施加 ΔP 裁剪用)。**cepri36/弱网库多为 pmax=pmin=0 未设限**——此时
    按"未设限"处理:不上卡(+inf)、下 floor 至 min(0, 当前)。否则用声明的 [pmin,pmax]。

    **事故根因(2026-07-18)**:旧实现一律 clip 到 [pmin,pmax]=[0,0],使"基态 2.28→裁到 0"
    的病态可造(向下裁成立),但"从 0 加回 2.28"的恢复类修法被压回 0,规范解/模型/规则
    全部施加失败(oracle 恒 False)。sample_perturbation 用 span=max(pmax-pmin,|pg|,1e-3)
    本已回避退化,施加路径必须口径一致。**v0 局限**:未设限时不上卡 = 允许非物理大出力,
    PSASP 功能性验收兜底(不比标签),真上限待 v0.5 从 ST/DATALIB 卡取。
    """
    if pmax > pmin + 1e-9:
        return pmin, pmax
    return min(0.0, cur), float("inf")


def share_by_headroom(states: Dict[int, dict], pool: List[int],
                      need: float) -> Dict[int, float]:
    """把 need(需注入的净有功)按各机组在 need 方向上的**可用余量**比例分配(防伪第 2 解
    的定向候选;平摊会把量分给无余量机、被裁剪后凑不满 net)。

    states[gi]={'p','pmin','pmax'};need>0→加出力(用上行余量 pmax−p),need<0→减出力
    (用下行余量 p−pmin);只选正确方向有余量的机,按余量比例分,余量不足则按比例吃满
    (尽力,不超各自余量)。返回 {gi: dp}。
    """
    if abs(need) < 1e-9:
        return {}
    def headroom(gi: int) -> float:
        s = states[gi]
        lo, hi = eff_bounds(s["pmin"], s["pmax"], s["p"])
        if need > 0:                            # 上行余量;未设限(hi=inf)兜有限参考
            h = hi - s["p"]
            return h if h != float("inf") else max(abs(s["p"]), 1.0)
        return s["p"] - lo
    avail = {gi: headroom(gi) for gi in pool if headroom(gi) > 1e-6}
    total = sum(avail.values())
    if total < 1e-9:
        return {}
    frac = min(1.0, abs(need) / total)          # 余量不足→按比例吃满(尽力)
    sign = 1.0 if need > 0 else -1.0
    return {gi: sign * hr * frac for gi, hr in avail.items()}


# ----------------- PSASP 驱动(试产时点火;单测不触及) -----------------

def _referee(work: Path, solver) -> dict:
    from gridfm.data.synth import psasp_referee
    return psasp_referee(work, solver)


def _is_healthy(rf: dict, low: float, high: float, vmax_gate: bool = True) -> bool:
    """健康 = 收敛且 vmin≥low(vmax_gate 开时并要求 vmax≤high)。

    **D-1 细调(2026-07-18 R1 实测)**:两库建库决策为"vmin 硬卡、vmax 不硬卡"
    (lib118 抽检 vmax 全库 1.053~1.070、带 [0.93,1.05] 命中 0/25;cepri36 中位 1.067)
    ——带口径把几乎整库判成病态起点。v0 默认改 vmin 单边线(vmax 如实记录不判),
    与建库决策/M-W1 验收线/有功杠杆物理一致;--vmax-gate 保留带口径可切回。
    """
    if not rf.get("converged"):
        return False
    ok = rf["vmin"] >= low
    if vmax_gate:
        ok = ok and rf["vmax"] <= high
    return bool(ok)


def _is_violated(rf: dict, low: float, high: float, vmax_gate: bool = True) -> bool:
    if not rf.get("converged"):
        return False
    return bool(rf["vmin"] < low or (vmax_gate and rf["vmax"] > high))


def generate_one(case_dir: Path, work: Path, solver, rng: np.random.Generator,
                 low: float, high: float, budget: int = 12,
                 vmax_gate: bool = False,
                 k_lo: int = 1, k_hi: int = 3,
                 frac_lo: float = 0.10, frac_hi: float = 0.40) -> Optional[dict]:
    """一个健康方式 → 一条调整标签(或 None + 原因)。

    流程:①校验健康起点;②预算内逆扰动搜"收敛且越限"的病态;③反向扰动 = 可行修法#1
    (按构造恢复原健康态,PSASP 复验);④防伪:再搜 ≥1 个不同机组组合的可行修法。
    返回 dict(含 feasible/reason/病态越限描述/多可行解/规范解),或健康起点无效时 None。
    """
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(case_dir, work)
    rf0 = _referee(work, solver)
    if not _is_healthy(rf0, low, high, vmax_gate):
        return {"sid": case_dir.name, "feasible": False,
                "reason": "起点非健康", "referee0": rf0.get("converged"),
                "vmin0": (round(rf0["vmin"], 4) if rf0.get("converged") else None),
                "vmax0": (round(rf0["vmax"], 4) if rf0.get("converged") else None)}

    l5 = Card.load(work / "LF.L5")
    _slack, gens = read_gens(l5)
    p_orig = {gi: g["view"].pg for gi, g in gens.items()}

    def _restore(pmap):
        for gi, p in pmap.items():
            gens[gi]["view"].pg = p
        l5.save(work / "LF.L5")

    found = None
    for _ in range(budget):
        _restore(p_orig)                                # 每次从原健康态出发扰动(不累加)
        k = int(rng.integers(k_lo, k_hi + 1))           # 缺省 k∈{1,2,3}(7.3 原文)
        deltas = sample_perturbation(gens, rng, k, frac_lo, frac_hi)
        applied = apply_deltas(gens, deltas)
        l5.save(work / "LF.L5")
        rf = _referee(work, solver)
        if _is_violated(rf, low, high, vmax_gate):
            found = {"applied": applied, "referee": rf}
            break
    _restore(p_orig)                                    # 恢复原出力

    if found is None:
        return {"sid": case_dir.name, "feasible": False,
                "reason": "预算内未造出收敛越限病态(多为发散或仍健康)"}

    # 可行修法#1 = 反向扰动(恢复原健康态,已由 rf0 验证);ΔP = −applied
    fix1 = {gi: -dp for gi, dp in found["applied"].items()}
    fixes = [{"deltas": {gens[gi]["bus"]: round(dp, 4) for gi, dp in fix1.items()},
              "l2": round(delta_l2(fix1), 4), "source": "reverse_perturb"}]

    # 防伪:搜第 2 个**不同机组组合**的可行修法(病态态上试,PSASP 验证)。
    # 首版试产实测:纯无向随机 6 试防伪率 0 → 定向候选优先,随机兜底:
    #   (a) 子集反向:只反转 |ΔP| 最大的那台(原扰动 ≥2 台时才是真子集,组合不同);
    #   (b) 余量定向转嫁:净扰动量 −Σapplied 按未触及机在正确方向的余量比例分配
    #      (share_by_headroom;平摊会分给无余量机被裁剪→凑不满,是首版防伪率低主因);
    #   (c) 随机兜底 ≤4 试(未触及机组上采样)。
    p_sick = {gi: p_orig[gi] + found["applied"].get(gi, 0.0) for gi in gens}
    _restore(p_sick)                                     # 重建病态态
    touched = set(found["applied"])
    untouched = [gi for gi in gens if gi not in touched]
    candidates: List[Tuple[str, Dict[int, float]]] = []
    if len(found["applied"]) >= 2:
        gi_max = max(found["applied"], key=lambda g: abs(found["applied"][g]))
        candidates.append(("subset_reverse", {gi_max: -found["applied"][gi_max]}))
    if untouched:
        net = sum(found["applied"].values())
        states = {gi: {"p": gens[gi]["view"].pg, "pmin": gens[gi]["pmin"],
                       "pmax": gens[gi]["pmax"]} for gi in untouched}
        cand = share_by_headroom(states, untouched, -net)
        if cand:
            candidates.append(("share_transfer_hr", cand))
    for _ in range(4):
        if untouched:
            pool = {gi: gens[gi] for gi in untouched}
            k = int(rng.integers(1, min(4, len(pool) + 1)))
            candidates.append(("alt_random", sample_perturbation(pool, rng, k)))
    for source, cand in candidates:
        base = {gi: gens[gi]["view"].pg for gi in gens}
        applied2 = apply_deltas(gens, cand)
        l5.save(work / "LF.L5")
        rf2 = _referee(work, solver)
        if _is_healthy(rf2, low, high, vmax_gate):
            fixes.append({"deltas": {gens[gi]["bus"]: round(dp, 4)
                                     for gi, dp in applied2.items()},
                          "l2": round(delta_l2(applied2), 4), "source": source})
            break
        _restore(base)                                   # 撤销这次尝试

    canon = min(fixes, key=lambda f: f["l2"])              # 规范解 = 最小 ‖ΔP‖
    return {
        "sid": case_dir.name, "feasible": True,
        "violation": {"vmin": round(found["referee"]["vmin"], 4),
                      "vmax": round(found["referee"]["vmax"], 4),
                      "buses": violations(found["referee"]["v"], low,
                                          high if vmax_gate else float("inf"))},
        "n_fixes": len(fixes), "fixes": fixes,
        "canonical_fix": canon,
        "antifake_ge2": len(fixes) >= 2,                   # 防伪(a):≥2 可行解
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True, help="方式库目录(含 cases/<sid>/)")
    ap.add_argument("--battery", required=True)
    ap.add_argument("--sids", nargs="*", default=None, help="方式号(缺省=全库前 --n-target 个)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-target", type=int, default=30, help="目标病态方式数(R1 试产)")
    ap.add_argument("--vio-low", type=float, default=VIO_LOW_DEFAULT)
    ap.add_argument("--vio-high", type=float, default=VIO_HIGH_DEFAULT)
    ap.add_argument("--k-lo", type=int, default=1, help="逆扰动机组数下限(7.3 缺省 1)")
    ap.add_argument("--k-hi", type=int, default=3, help="逆扰动机组数上限(7.3 缺省 3;"
                    "118 有功-电压杠杆弱需加深配方,探针实据见变更记录 2026-07-18)")
    ap.add_argument("--frac-lo", type=float, default=0.10, help="扰动幅度下限(容量比)")
    ap.add_argument("--frac-hi", type=float, default=0.40, help="扰动幅度上限(容量比)")
    ap.add_argument("--vmax-gate", action="store_true",
                    help="启用 vmax 带上限判(带口径)。缺省关 = vmin 单边线口径"
                         "(D-1 细调,2026-07-18 R1 实测:两库建库为 vmin 硬卡/vmax "
                         "不硬卡,带口径把几乎整库判成病态起点,扫描数据见变更记录)")
    ap.add_argument("--budget", type=int, default=12, help="每方式逆扰动尝试上限")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from gridfm.solver.psasp import PsaspSolver
    solver = PsaspSolver()
    lib = Path(a.lib)
    cases = lib / "cases"
    sids = a.sids or sorted(p.name for p in cases.iterdir() if p.is_dir())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    rng = np.random.default_rng(a.seed)

    rows: List[dict] = []
    n_feasible = n_infeasible = n_ge2 = 0
    for sid in sids:
        if n_feasible >= a.n_target:
            break
        cdir = cases / sid
        if not cdir.exists():
            print(f"[adj] 跳过 {sid}:无卡片目录", flush=True)
            continue
        row = generate_one(cdir, work, solver, rng, a.vio_low, a.vio_high, a.budget,
                           vmax_gate=a.vmax_gate, k_lo=a.k_lo, k_hi=a.k_hi,
                           frac_lo=a.frac_lo, frac_hi=a.frac_hi)
        if row is None:
            continue
        row["battery"] = a.battery
        rows.append(row)
        if row["feasible"]:
            n_feasible += 1
            n_ge2 += int(row.get("antifake_ge2", False))
            print(f"[adj] {sid}: 可行,{row['n_fixes']} 解,越限 "
                  f"vmin={row['violation']['vmin']} vmax={row['violation']['vmax']}", flush=True)
        else:
            n_infeasible += 1
            print(f"[adj] {sid}: 未产({row['reason']})", flush=True)

    (out / "adjust_labels.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    n_unhealthy = sum(1 for r in rows
                      if not r["feasible"] and str(r.get("reason", "")).startswith("起点"))
    n_nosick = sum(1 for r in rows
                   if not r["feasible"] and str(r.get("reason", "")).startswith("预算"))
    n_healthy_start = len(rows) - n_unhealthy
    summary = {"battery": a.battery, "n_tried": len(rows),
               "n_unhealthy_start": n_unhealthy,
               "n_healthy_start": n_healthy_start,
               "n_no_sick_found": n_nosick,
               "n_feasible": n_feasible, "n_infeasible": n_infeasible,
               "n_antifake_ge2": n_ge2,
               "hit_rate": round(n_feasible / max(len(rows), 1), 3),
               "hit_rate_healthy": round(n_feasible / max(n_healthy_start, 1), 3),
               "ge2_rate": round(n_ge2 / max(n_feasible, 1), 3),
               "vio_band": [a.vio_low, a.vio_high], "vmax_gate": a.vmax_gate,
               "criterion": ("band" if a.vmax_gate else "vmin_line"),
               "recipe": {"k": [a.k_lo, a.k_hi], "frac": [a.frac_lo, a.frac_hi],
                          "budget": a.budget},
               "seed": a.seed}
    (out / "adjust_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[adj] 试产收尾:可行 {n_feasible}/{len(rows)}(命中率 {summary['hit_rate']}),"
          f"≥2 可行解 {n_ge2}(防伪率 {summary['ge2_rate']});{out/'adjust_summary.json'}")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
