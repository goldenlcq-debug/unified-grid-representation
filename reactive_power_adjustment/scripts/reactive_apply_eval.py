"""无功调整功能性验收 + 规则修法核心(方向二选项 B;7.3 评估侧免疫)。

`reactive_rule_arm` = 规则基线修法(老师),**两级杠杆**:
- 第一级 = targeted v_set 最小贪心 + vmax 守门(S0 探针实证:最小干预从最近 1 机小步起
  逐步扩,首个双边健康即停;uniform 抬全体顶穿过压故非平凡)——单独用时通过率 **0.61**;
- 第二级 = 就近投并联补偿(use_shunt=True 才开)——去风险实测把老师抬到 **0.74**
  (runs/derisk_shunt_rule.json,救回 13 例,其中 9 例只需投 1 台容性)。
  更重要的作用:解锁 v_set 单杠杆修不动的"规则敌对"方式的打标(约 68/120 个训练方式)。
既作 rule 臂(验收),又作打标器(reactive_rulelabel import 它给 A 族病态打标)。

**动作契约**:修法 = 混合结构 {"dvset": {母线: Δv_set}, "shunt": [投运的并补母线]};
None = 修不动;两项皆空 = 方式本就健康。施加顺序恒为**先投并补(LF.L2 的 mark)、
再写 Δv_set(LF.L5)**,见 apply_mixed。

四臂双边验收(model/rule/trivial/oracle):施加混合动作 → PSASP 复核 vmin>=low 且
vmax<=high 且收敛 → 一次通过率;不比标签(7.3)。第五臂"v_set 复位基态"(捷径探测)在
reactive_noninverse_eval(那里有基态 v_set)。
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

from gridfm.psasp_io.cards import AclineView  # noqa: E402
from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402
from gridfm.reason.reactive_task import VIO_HIGH, VIO_LOW  # noqa: E402
from scripts.adjust_apply_eval import CountingSolver  # noqa: E402
from scripts.adjust_labels import _referee  # noqa: E402
from scripts.reactive_labels import (  # noqa: E402
    _is_healthy_2s, _is_violated_2s, apply_dvset, delta_l2, read_ctrl,
)

# 最小贪心搜索梯度(S0 探针验证:多数低压病态 1 机最小步即修好;梯度收窄至 4×2=8 解上限,
# 兼顾修法最小性与打标速度——118 单解 ~1.3s,梯度过宽使不可修病态白烧 20 解)
_NGEN_LADDER = (1, 2, 4, 8)
_STEP_LADDER = (0.02, 0.05)
# 第二杠杆:投并联补偿的台数梯度(去风险实测 13 例被救回,其中 9 例只需 1 台)
_SHUNT_LADDER = (1, 2, 3)


# ---------------- 混合动作契约(v_set 连续 + shunt 离散) ----------------
# 修法统一表示为 {"dvset": {母线号: Δv_set}, "shunt": [投运的并补母线号, ...]};
# None 仍表示"修不动"。空修法(已健康)= 两项都空。

def mixed_fix(dvset: Optional[Dict[int, float]] = None,
              shunt: Optional[List[int]] = None) -> Dict:
    return {"dvset": {int(b): float(d) for b, d in (dvset or {}).items()},
            "shunt": [int(b) for b in (shunt or [])]}


def fix_is_empty(fix: Optional[Dict]) -> bool:
    """空修法(方式本就健康)= 既没调 v_set 也没投并补。None 不算空(那是修不动)。"""
    return fix is not None and not fix.get("dvset") and not fix.get("shunt")


def as_mixed(fix) -> Optional[Dict]:
    """兼容旧契约:老标签/老预测的 {母线: Δ} 纯 v_set 结构 → 混合结构。"""
    if fix is None:
        return None
    if isinstance(fix, dict) and ("dvset" in fix or "shunt" in fix):
        return mixed_fix(fix.get("dvset"), fix.get("shunt"))
    return mixed_fix(fix, [])


# ---------------- 并联补偿(第二杠杆)读写 ----------------

def shunt_records(work: Path, capacitive: bool = True) -> List:
    """停运的并联补偿记录 [(记录序, 母线号)]。capacitive=True 取容性(X<0,抬压),
    False 取感性(X>0,降压)。lib118:每方式 14 个全停运(12 容 / 2 感)。"""
    l2 = Card.load(work / "LF.L2")
    out = []
    for i, r in enumerate(l2.records):
        av = AclineView(r, i)
        if av.branch_type != "shunt_comp" or av.mark == 1:
            continue
        x = r.get_float(AclineView.X)
        if (x < 0) if capacitive else (x > 0):
            out.append((i, av.i_no))
    return out


def read_shunt_marks(work: Path, idxs) -> Dict[int, int]:
    l2 = Card.load(work / "LF.L2")
    return {int(i): int(AclineView(l2.records[i], i).mark) for i in idxs}


def set_shunt_marks(work: Path, marks: Dict[int, int]) -> None:
    """按 {记录序: mark} 写回 LF.L2(投运 mark=1 / 复位原值)。"""
    if not marks:
        return
    l2 = Card.load(work / "LF.L2")
    for i, mk in marks.items():
        AclineView(l2.records[i], i).mark = mk
    l2.save(work / "LF.L2")


def hops_from(adj: Dict[int, set], seed: int) -> Dict[int, int]:
    """BFS 跳数表(母线号 → 到 seed 的跳数);就近投切用。"""
    d = {seed: 0}
    frontier = [seed]
    while frontier:
        nxt = []
        for b in frontier:
            for nb in adj.get(b, []):
                if nb not in d:
                    d[nb] = d[b] + 1
                    nxt.append(nb)
        frontier = nxt
    return d


def restore_vset(work: Path, v_by_gi: Dict[int, float]) -> None:
    """从磁盘重载 LF.L5 并把 v_set 复位到给定值(记录序稳定,故 gi 键可跨重载复用)。"""
    l5 = Card.load(work / "LF.L5")
    _s, gens = read_ctrl(l5)
    for gi, v in v_by_gi.items():
        if gi in gens:
            gens[gi]["view"].v0 = v
    l5.save(work / "LF.L5")


def apply_mixed(work: Path, fix: Dict) -> Dict:
    """把混合动作施加到算例:**先投并补(改 LF.L2 的 mark),再写 Δv_set(改 LF.L5)**。
    返回实际生效的 {"dvset": {母线: 实际Δ}, "shunt": [实际投运的母线]}(裁剪/无候选后)。"""
    fix = as_mixed(fix) or mixed_fix()
    done_shunt: List[int] = []
    want = set(fix["shunt"])
    if want:
        # 母线 → 停运并补记录(容感都收,方向由规则/模型在选母线时已定;同母线多台取第一台)
        by_bus: Dict[int, int] = {}
        for cap in (True, False):
            for idx, bus in shunt_records(work, capacitive=cap):
                by_bus.setdefault(int(bus), idx)
        pick = {by_bus[b]: 1 for b in sorted(want) if b in by_bus}
        set_shunt_marks(work, pick)
        done_shunt = sorted(b for b in want if b in by_bus)
    applied_dv: Dict[int, float] = {}
    if fix["dvset"]:
        l5 = Card.load(work / "LF.L5")
        _s, gens = read_ctrl(l5)
        by_bus_g = {g["bus"]: gi for gi, g in gens.items()}
        got = apply_dvset(gens, {by_bus_g[b]: d for b, d in fix["dvset"].items()
                                 if b in by_bus_g})
        l5.save(work / "LF.L5")
        applied_dv = {gens[gi]["bus"]: dv for gi, dv in got.items()}
    return mixed_fix(applied_dv, done_shunt)


def _bus_adjacency(work: Path) -> Dict[int, set]:
    case = PsaspCase(work)
    adj: Dict[int, set] = {}
    for ln in case.aclines():
        if ln.mark != 1:
            continue
        adj.setdefault(ln.i_no, set()).add(ln.j_no)
        adj.setdefault(ln.j_no, set()).add(ln.i_no)
    return adj


def _nearest_gens(adj: Dict[int, set], seed_bus: int, gens: Dict[int, dict],
                  k: int) -> List[int]:
    """图上按跳数从 seed_bus 找最近的 k 个机组(记录序);同深度取小母线号。"""
    gen_by_bus = {g["bus"]: gi for gi, g in gens.items()}
    seen, frontier, out = {seed_bus}, [seed_bus], []
    if seed_bus in gen_by_bus:
        out.append(gen_by_bus[seed_bus])
    while frontier and len(out) < k:
        nxt = []
        for b in frontier:
            for nb in sorted(adj.get(b, [])):
                if nb in seen:
                    continue
                seen.add(nb)
                if nb in gen_by_bus and gen_by_bus[nb] not in out:
                    out.append(gen_by_bus[nb])
                    if len(out) >= k:
                        break
                nxt.append(nb)
            if len(out) >= k:
                break
        frontier = nxt
    return out


def _extreme_bus(rf: dict, low: bool) -> Optional[int]:
    """越限最深母线号(1 基)。low=True 取最低压,False 取最高压。rf['v'] 0 基。"""
    v = rf.get("v")
    if v is None:
        return None
    live = np.where(v > 0.01)[0]
    if live.size == 0:
        return None
    idx = live[np.argmin(v[live])] if low else live[np.argmax(v[live])]
    return int(idx) + 1


def _vset_greedy(work: Path, solver, vio_low: float,
                 vio_high: float) -> Optional[Dict[int, float]]:
    """v_set 单杠杆最小贪心(第一杠杆本体)。返回 {母线号: Δv_set} / {} 已健康 / None 修不动。

    低压(vmin<low)→ 抬最近若干机 v_set;过压(vmax>high)→ 降最近若干机 v_set。
    从最近 1 机最小步起逐步扩机/加步,首个双边健康(vmin>=low 且 vmax<=high 且收敛)即返回
    该(最小)干预的实际 Δv_set;全梯度不过则把 v_set 复位后返回 None。"""
    rf0 = _referee(work, solver)
    if not rf0.get("converged"):
        return None
    is_low = rf0["vmin"] < vio_low
    is_over = rf0["vmax"] > vio_high
    if not (is_low or is_over):
        return {}                                     # 已健康(空修法)
    seed = _extreme_bus(rf0, low=is_low)              # 低压优先(方案甲主体)
    if seed is None:
        return None
    sign = 1.0 if is_low else -1.0
    adj = _bus_adjacency(work)
    l5 = Card.load(work / "LF.L5")
    _s, gens = read_ctrl(l5)
    v_orig = {gi: g["view"].v0 for gi, g in gens.items()}

    def _restore():
        for gi, v in v_orig.items():
            gens[gi]["view"].v0 = v
        l5.save(work / "LF.L5")

    for ngen in _NGEN_LADDER:
        chosen = _nearest_gens(adj, seed, gens, ngen)
        if not chosen:
            break
        for step in _STEP_LADDER:
            _restore()
            applied = apply_dvset(gens, {gi: sign * step for gi in chosen})
            l5.save(work / "LF.L5")
            rf = _referee(work, solver)
            if _is_healthy_2s(rf, vio_low, vio_high):
                return {gens[gi]["bus"]: round(dv, 4) for gi, dv in applied.items()
                        if abs(dv) > 1e-6}
    _restore()
    return None


def reactive_rule_arm(work: Path, solver, vio_low: float = VIO_LOW,
                      vio_high: float = VIO_HIGH,
                      use_shunt: bool = False) -> Optional[Dict]:
    """规则修法(老师)。返回混合动作 {"dvset": {母线: Δv_set}, "shunt": [母线]} 或 None。

    两级杠杆(use_shunt=True 才开第二级;缺省关 = 与 v0/v2 定论的老师逐位同构):
      第一级 = v_set 最小贪心 + vmax 守门(_vset_greedy);
      第二级 = 就近投并联补偿 —— v_set 复位后,按到最深越限母线的图跳数就近投 1/2/3 台
               (低压投容性 X<0 抬压,过压投感性 X>0 降压),每档先看是否已健康,
               未健康但收敛则在该 shunt 状态上再跑一次 v_set 最小贪心。
    全档不过则把 v_set 与并补 mark 全部复位,返回 None(如实:双杠杆也修不动)。
    去风险实测(runs/derisk_shunt_rule.json):第二级把老师 0.61 抬到 0.74。"""
    dv = _vset_greedy(work, solver, vio_low, vio_high)
    if dv is not None:
        return mixed_fix(dv, [])                      # 含"已健康"的空修法 {}
    if not use_shunt:
        return None

    # ---- 第二杠杆:_vset_greedy 已把 v_set 复位到基态,从同一起点做 shunt 搜索 ----
    rf0 = _referee(work, solver)
    if not rf0.get("converged"):
        return None
    is_low = rf0["vmin"] < vio_low
    seed = _extreme_bus(rf0, low=is_low)
    if seed is None:
        return None
    cands = shunt_records(work, capacitive=is_low)     # 低压→容性抬压;过压→感性降压
    if not cands:
        return None
    hops = hops_from(_bus_adjacency(work), seed)
    # 只按跳数排,**不加母线号次序键**:Python sort 稳定,同跳数时保持 LF.L2 记录序,
    # 与去风险脚本 derisk_shunt_rule.py:96 逐位一致 —— 0.74 是那个实现测出来的。
    # (实测:加了母线号次序键后,118 个可能 seed 里有 40 个在 k=1 档会选中不同的并补。)
    cands.sort(key=lambda t: hops.get(t[1], 9999))
    orig_marks = read_shunt_marks(work, [i for i, _b in cands])
    l5b = Card.load(work / "LF.L5")
    _sb, gensb = read_ctrl(l5b)
    v_base = {gi: g["view"].v0 for gi, g in gensb.items()}

    for k in _SHUNT_LADDER:
        if k > len(cands):
            break
        set_shunt_marks(work, orig_marks)             # 并补复位
        restore_vset(work, v_base)                    # v_set 复位(上一档贪心可能改过)
        picked = cands[:k]
        set_shunt_marks(work, {i: 1 for i, _b in picked})
        buses = sorted(int(b) for _i, b in picked)
        rf1 = _referee(work, solver)
        if _is_healthy_2s(rf1, vio_low, vio_high):
            return mixed_fix({}, buses)               # 只投并补即合格(9/13 例是这种)
        if rf1.get("converged"):
            # 严格非空才算修好:dv2 == {} 只可能出现在 eps 边界上(贪心的无 eps 判据认为
            # 已健康,而裁判口径 _is_healthy_2s 刚判不健康)——保守不计通过,继续下一档
            dv2 = _vset_greedy(work, solver, vio_low, vio_high)
            if dv2:
                return mixed_fix(dv2, buses)
    set_shunt_marks(work, orig_marks)
    restore_vset(work, v_base)
    return None


def trivial_arm(work: Path, solver, vio_low: float = VIO_LOW,
                vio_high: float = VIO_HIGH) -> Optional[Dict]:
    """平凡基线(方向化):越限最深母线 → 最近 1 机,按越限方向 ±固定一步 v_set(0.03)。
    一次 PSASP。返回混合动作(shunt 恒空,平凡基线不用第二杠杆)或 None。"""
    rf0 = _referee(work, solver)
    if not rf0.get("converged"):
        return None
    is_low = rf0["vmin"] < vio_low
    is_over = rf0["vmax"] > vio_high
    if not (is_low or is_over):
        return mixed_fix()
    seed = _extreme_bus(rf0, low=is_low)
    adj = _bus_adjacency(work)
    l5 = Card.load(work / "LF.L5")
    _s, gens = read_ctrl(l5)
    chosen = _nearest_gens(adj, seed, gens, 1)
    if not chosen:
        return None
    step = 0.03 if is_low else -0.03
    applied = apply_dvset(gens, {chosen[0]: step})
    l5.save(work / "LF.L5")
    rf = _referee(work, solver)
    ok = _is_healthy_2s(rf, vio_low, vio_high)
    return (mixed_fix({gens[chosen[0]]["bus"]: round(applied[chosen[0]], 4)}, [])
            if ok else None)


def _load_rows(labels_dirs) -> Dict:
    out = {}
    for lab in labels_dirs:
        lab = Path(lab)
        rows = [json.loads(ln) for ln in
                (lab / "adjust_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        for i, r in enumerate(rows):
            out[(lab.name, i)] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--preds", default=None, help="model 臂:含 eval_predictions.jsonl 的目录")
    ap.add_argument("--arms", nargs="+", default=["model", "rule", "trivial", "oracle"])
    ap.add_argument("--vio-low", type=float, default=VIO_LOW)
    ap.add_argument("--vio-high", type=float, default=VIO_HIGH)
    ap.add_argument("--rule-shunt", action="store_true",
                    help="rule 臂开第二杠杆(投并联补偿);缺省关 = v_set 单杠杆老师")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from gridfm.solver.psasp import PsaspSolver
    solver = CountingSolver(PsaspSolver())
    rows_by_key = _load_rows(a.labels)
    manifest = []
    for cd in a.cases:
        for ln in (Path(cd) / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                m = json.loads(ln); m["cases_dir"] = cd; manifest.append(m)
    by_tag = {m["tag"]: m for m in manifest}

    preds_by_tag = {}
    if "model" in a.arms:
        if not a.preds:
            raise SystemExit("[react-eval] model 臂需要 --preds")
        for ln in (Path(a.preds) / "eval_predictions.jsonl").read_text(
                encoding="utf-8").splitlines():
            if ln.strip():
                p = json.loads(ln); preds_by_tag[p["tag"]] = p
        tags = [t for t in preds_by_tag if t in by_tag]
    else:
        tags = sorted(by_tag)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    work = out / "_work"
    results = {arm: {"pass": 0, "fail": 0, "l2": [], "n_shunt": []} for arm in a.arms}
    per_case = []
    for tag in tags:
        m = by_tag[tag]
        row = rows_by_key[(m["batch"], m["row_index"])]
        canon_l2 = row["canonical_fix"]["l2"]
        src = Path(m["cases_dir"]) / "cases" / tag
        rec = {"tag": tag}
        for arm in a.arms:
            if work.exists():
                shutil.rmtree(work, ignore_errors=True)
            shutil.copytree(src, work)
            fix = None
            if arm == "oracle":
                cf = row["canonical_fix"]
                fix = mixed_fix({int(b): float(d) for b, d in cf["deltas"].items()},
                                cf.get("shunt", []))
            elif arm == "model":
                p = preds_by_tag[tag]
                fix = mixed_fix({int(b): float(d) for b, d in p["deltas"].items()},
                                p.get("shunt", []))
            elif arm == "rule":
                fix = reactive_rule_arm(work, solver, a.vio_low, a.vio_high, a.rule_shunt)
            elif arm == "trivial":
                fix = trivial_arm(work, solver, a.vio_low, a.vio_high)
            if arm in ("oracle", "model") and fix is not None:
                apply_mixed(work, fix)                # 先投并补,再写 Δv_set
                ok = _is_healthy_2s(_referee(work, solver), a.vio_low, a.vio_high)
            else:
                ok = fix is not None
            results[arm]["pass" if ok else "fail"] += 1
            rec[arm] = bool(ok)
            if ok and fix and not fix_is_empty(fix):
                results[arm]["l2"].append(round(delta_l2(fix["dvset"]) /
                                                max(canon_l2, 1e-9), 3))
                results[arm]["n_shunt"].append(len(fix["shunt"]))
        per_case.append(rec)
        print(f"[react-eval] {tag}: " + " ".join(f"{k}={rec.get(k)}" for k in a.arms), flush=True)

    summary = {"n": len(tags), "vio_low": a.vio_low, "vio_high": a.vio_high, "arms": {}}
    for arm in a.arms:
        r = results[arm]; n = r["pass"] + r["fail"]
        summary["arms"][arm] = {"pass_rate": round(r["pass"] / max(n, 1), 3), "n": n,
                                "l2_over_canonical_median": (round(float(np.median(r["l2"])), 3)
                                                             if r["l2"] else None),
                                "shunt_mean_per_pass": (round(float(np.mean(r["n_shunt"])), 3)
                                                        if r["n_shunt"] else None)}
    (out / "apply_eval.json").write_text(json.dumps(
        {"summary": summary, "per_case": per_case}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[react-eval] 收尾:{json.dumps(summary['arms'], ensure_ascii=False)}")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
