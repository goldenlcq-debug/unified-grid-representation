"""T4.4 CCT 批量标签(方案 7.3 CCT 协议;**单路串行铁律** D16/7.3)。

CCT 是数值敏感标签,**不走 7.3 并行豁免**:全程 workers=1,且启动前不得有其他长跑
PSASP 任务(CPU 争用扰动暂态数值)。

协议(7.3):二分区间 [0, t_max](三相短路 t_max=0.5s),精度 0.01s,单样本仿真次数
上限 8;上界仍稳 → (cct=t_max, is_censored=True);infeasible 单独打标不入统计。
判稳判据 = D21 v1(阶段 2 精确口径,t_end=12s——与判稳批同窗:10s 窗下"故障后 10s"
分句不可评,CCT 会按更松的判据搜出偏大的值)。

**临界故障挑选**(--from-labels):从判稳批标签里取"同一 (方式,故障支路) 在种子切除
时长档上出现稳/失稳翻转"的组合——CCT 落在扫过的档位之间,是信息量最高的样本;
其余组合的 CCT 要么 <最小档要么 >最大档(退化为删失/近零),先不烧机时。

**DoD 验证**(--verify):对每个非删失 CCT 做临界性双验(2 次额外仿真):
Te=cct 必稳、Te=cct+tol 必失稳。这是 CCT 定义的直接验证,比"与另一份二分实现比对"
更强(同一实现比对只能验一致性,验不了正确性)。

  python scripts/ts_cct.py --lib .../lib_cepri36 --battery cepri36 \
      --from-labels runs/ts_labels_cepri36 --limit 30 --verify

**定向模式**(2026-07-27;按模型不确定性选样,清单由 `scripts/cct_pilot_jobs.py` 产):

  python scripts/ts_cct.py --lib .../lib118 --battery ieee118_v1 \
      --from-jobs runs/cct_pilot_jobs.jsonl --verify --out runs/ts_cct_pilot_ieee118_v1

- **一次只吃一个 battery**:清单跨电池时须按电池跑多次,**每次一个独立 `--out`**
  (裸 tag 跨电池会撞,done 集与 work 目录都已按 battery 隔离,但产物混在一个
  `cct.jsonl` 里会让下游的分层报数与 `receive_audit` 的重复检查打架);
- **不得给 `--limit`**:清单本身就是预注册抽样结果,截断会破坏分层与族配额(代码已拒);
- 产物 `cct_summary.json` 带 `by_stratum` / `by_battery` 拆分,**判据一律读 `n_usable`
  (排除上/下侧删失)而非 `n_valid`**。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from ts_seeds import BATTERY, BATTERY_V2, BATTERY_V3, TS  # noqa: E402

from gridfm.fsutil import robust_copytree, robust_rmtree  # noqa: E402
from gridfm.psasp_io import cards  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402
from gridfm.psasp_io.modegate import TS_BASE_VMIN_FLOOR, ts_mode_ok  # noqa: E402
from gridfm.psasp_io.stout import (  # noqa: E402
    add_bus_voltage_records, build_sme_from_scratch,
)
from gridfm.solver.base import FaultSpec  # noqa: E402
from gridfm.solver.psasp import LicenseError, PsaspSolver, set_s0_duration  # noqa: E402

# v2 电池纳入(2026-07-14,B2 前置补丁;B2 发包 pin 必须 ≥ 本补丁提交)
ALL_BATTERY = {**BATTERY, **BATTERY_V2, **BATTERY_V3}

T_MAX = 0.5
TOL = 0.01
MAX_SIMS = 8


def prep_fault_card(work: Path, battery_def: dict) -> None:
    """blank_s11 电池(ieee39_v2)在写卡前清空 ST.S11——与 ts_labels 同款:交付卡为
    K≠0 GUI 模板(故障点母线只对原线路有效),换线路不清空 = 残行污染仿真。
    search_cct/verify 的每次迭代都经 write_fault_card 在同一 work 目录改写,
    拷贝后清空一次即可。"""
    if battery_def.get("blank_s11"):
        (work / "ST.S11").write_bytes(b"")


def _critical_pairs(labels_dir: Path):
    """从判稳批挑临界 (sample_id, 故障) 组合:切除时长档上出现稳/失稳翻转者。"""
    src = labels_dir / "labels_v1.jsonl"
    if not src.exists():
        src = labels_dir / "labels.jsonl"
    rows = [json.loads(ln) for ln in src.read_text(encoding="utf-8").splitlines()]
    by_key = defaultdict(list)
    for r in rows:
        if not r.get("feasible"):
            continue
        by_key[(r["sample_id"], r["i"], r["j"], r["id"], r["k_pct"])].append(r)
    out = []
    for key, rs in by_key.items():
        st = {r["dur"]: r["stable"] for r in rs}
        if len(set(st.values())) > 1:            # 档位间翻转 = CCT 落在扫过的区间内
            durs = sorted(st)
            flip_lo = max((d for d in durs if st[d]), default=None)
            flip_hi = min((d for d in durs if not st[d]), default=None)
            out.append({"sample_id": key[0], "i": key[1], "j": key[2], "id": key[3],
                        "k_pct": key[4], "dur_stable_max": flip_lo,
                        "dur_unstable_min": flip_hi})
    out.sort(key=lambda d: (d["sample_id"], d["i"], d["j"], d["id"]))
    return out


def _jobs_for_battery(jobs_path: Path, battery: str):
    """`--from-jobs`:按**外部预注册作业清单**选样(定向 CCT,2026-07-27)。

    与 `--from-labels` 的区别 = **选样域**:`_critical_pairs` 只取"同方式跨时长档翻转"
    的键;定向 CCT 要按**模型不确定性**挑键(`scripts/cct_pilot_jobs.py` 产清单),
    两者交集只有约 78%。二分/双验/写档路径**完全共用**,不新开口径。
    """
    out = []
    for ln in jobs_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("battery") != battery:
            continue
        out.append({"sample_id": r["sample_id"], "i": int(r["i"]), "j": int(r["j"]),
                    "id": int(r["id"]), "k_pct": float(r["k_pct"]),
                    # 定向键不保证跨档翻转,故无 labels 括号;分层随行入档供分层统计
                    "dur_stable_max": None, "dur_unstable_min": None,
                    "stratum": r.get("stratum"), "n_seats_blind": r.get("n_seats_blind")})
    out.sort(key=lambda d: (d["sample_id"], d["i"], d["j"], d["id"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--battery", required=True, choices=sorted(ALL_BATTERY))
    ap.add_argument("--from-labels", help="判稳批目录(挑临界故障);与 --from-jobs 二选一")
    ap.add_argument("--from-jobs", help="预注册作业清单 jsonl(定向 CCT;按模型不确定性选样)")
    ap.add_argument("--out", default=None, help="默认 runs/ts_cct_<battery>")
    # -1 = 未显式指定:--from-labels 沿用历史默认 30;--from-jobs 取清单全量(清单本身
    # 就是预注册的抽样结果,再截断会破坏配额与随机性)。0 = 不限。
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--t-end", type=float, default=12.0)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--verify", action="store_true",
                    help="DoD:对每个非删失 CCT 做临界性双验(Te=cct 稳 / Te=cct+0.01 失稳)")
    ap.add_argument("--no-sme-vbus", action="store_true")
    ap.add_argument("--vmin-floor", type=float, default=TS_BASE_VMIN_FLOOR,
                    help="暂稳方式门:基态全网最低电压下限(2026-07-16 裁决,modegate.py)")
    a = ap.parse_args()

    lib = Path(a.lib)
    out = Path(a.out or f"runs/ts_cct_{a.battery}")
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "cct.jsonl"
    # **续跑 done 集必须带 battery**(2026-07-27 对抗审查 major):tag **不含 K%**
    # (`ts_task.join_key` 的注释早已写死"禁止裸 tag 全局 join"),而 cepri36 / cepri36_v2 /
    # cepri36_k50 同库同方式、k50 的 25 条线 = v1 5 条 + v2 20 条 —— 只有 K 不同的两个键
    # tag 逐字相同。实测:2,165 键的定向池里 **308 个裸 tag 被 2 个电池共用 = 616 键
    # (28.5%)**;若几个电池共用一个 --out,后跑的会被**静默跳过**。
    done = set()
    if jsonl.exists():
        for ln in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
                done.add((r.get("battery"), r["tag"]))
            except (ValueError, KeyError):
                continue

    if bool(a.from_labels) == bool(a.from_jobs):
        raise SystemExit("[t4.4] ✋ --from-labels 与 --from-jobs 必须且只能给一个")
    if a.from_jobs:
        cands = _jobs_for_battery(Path(a.from_jobs), a.battery)
        print(f"[t4.4] **定向模式**(--from-jobs):{Path(a.from_jobs).name} 中电池 "
              f"{a.battery} 命中 {len(cands)} 键")
        if not cands:
            raise SystemExit(f"[t4.4] ✋ 清单里没有电池 {a.battery} 的作业,停")
    else:
        cands = _critical_pairs(Path(a.from_labels))
    manifest = {r["sample_id"] for r in csv.DictReader(
        open(lib / "manifest.csv", encoding="utf-8"))}
    cands = [c for c in cands if c["sample_id"] in manifest
             and (lib / "cases" / c["sample_id"]).is_dir()]
    # **暂稳方式门**(2026-07-16 裁决;modegate.py):基态 vmin <门槛的方式,判据②在
    # 故障前就已违反 → 全档失稳、无"稳→失稳翻转"可言,CCT 二分会一路退到下界产伪值。
    # 老标签(门前所产)里含这类方式,此处按门过滤。
    n_pre = len(cands)
    cands = [c for c in cands
             if ts_mode_ok(lib / "cases" / c["sample_id"], floor=a.vmin_floor)]
    if n_pre != len(cands):
        print(f"[t4.4] 方式门(基态 vmin ≥ {a.vmin_floor}):剔除 {n_pre - len(cands)} 个"
              f"低压方式的临界故障(基态即违反判据②稳态分句)")
    if not cands:                 # 方式门/manifest 过滤后清空 → 响亮失败,不静默写空 summary
        raise SystemExit(f"[t4.4] ✋ 过滤后候选为 0(--lib 与 --battery 配错?),停")
    limit = a.limit if a.limit >= 0 else (0 if a.from_jobs else 30)
    if a.from_jobs and limit:     # 清单本身就是预注册抽样,再截断会破坏配额
        raise SystemExit(f"[t4.4] ✋ --from-jobs 下不得给 --limit({limit});"
                         f"清单即预注册抽样结果,截断会破坏分层/族配额")
    if limit:
        cands = cands[:limit]
    jobs = [c for c in cands
            if (a.battery, f"{c['sample_id']}_L{c['i']}-{c['j']}-{c['id']}") not in done]
    print(f"[t4.4] 临界故障 {len(cands)}(待跑 {len(jobs)},已完成跳过 {len(cands)-len(jobs)});"
          f"**单路串行**;t_end={a.t_end}s;二分 [0,{T_MAX}] tol={TOL} 上限 {MAX_SIMS} 次")
    est = len(jobs) * (MAX_SIMS + (2 if a.verify else 0)) * 8
    print(f"[t4.4] 机时估算 ≤ {est/60:.0f} min(按 8s/仿真上限)")

    solver = PsaspSolver()
    t0 = time.time()
    n_done = 0
    for c in jobs:
        tag = f"{c['sample_id']}_L{c['i']}-{c['j']}-{c['id']}"
        work = out / "work" / a.battery / tag      # 按电池分目录:裸 tag 跨电池会撞
        robust_copytree(lib / "cases" / c["sample_id"], work)
        prep_fault_card(work, ALL_BATTERY[a.battery])
        if not a.no_sme_vbus:
            buses = sorted(b.bus_no for b in cards.parse_lp1(Card.load(work / "LF.LP1")))
            sme = work / "ST.SME"
            if not sme.exists() or not sme.read_bytes().strip():
                # 无 SME 交付工程(118 类):从零构造(寻址随 L5 版式二分)
                build_sme_from_scratch(sme, work / "LF.L5")
            add_bus_voltage_records(sme, buses)
        set_s0_duration(work, a.t_end)
        fault = FaultSpec(c["i"], c["j"], c["id"], k_pct=c["k_pct"], ts=TS, te=TS + 0.1)
        t1 = time.time()
        try:
            res = solver.search_cct(work, fault, lo=0.0, hi=T_MAX, tol=TOL,
                                    max_sims=MAX_SIMS, t_end=a.t_end)
        except LicenseError:
            print("[t4.4] ✋ slm_init error:加密狗失效,已中止;重插狗后重跑同命令续接")
            return 2
        row = {"tag": tag, "sample_id": c["sample_id"], "battery": a.battery,
               "i": c["i"], "j": c["j"], "id": c["id"], "k_pct": c["k_pct"],
               "ts": TS, "t_end": a.t_end,
               "cct": (None if math.isnan(res.cct_value) else res.cct_value),
               "is_censored": res.is_censored, "infeasible": res.infeasible,
               # 下侧删失(CCT < 下界,真值未测)—— 与上侧删失对称打标,**不入点值统计**
               "is_censored_low": bool(getattr(res, "is_censored_low", False)),
               "n_sims": res.n_sims,
               "bracket_from_labels": [c["dur_stable_max"], c["dur_unstable_min"]],
               "criteria_version": "v1_20260712", "elapsed_s": round(time.time() - t1, 1)}
        if c.get("stratum") is not None:      # 定向模式:分层信息随行入档(分层报数用)
            row["stratum"] = c["stratum"]
            row["n_seats_blind"] = c.get("n_seats_blind")
        # DoD 临界性双验:cct 稳 + cct+tol 失稳(CCT 定义的直接验证)。
        # **下侧删失必须跳过**(2026-07-27 审查 critical):它的 cct = 未仿真过的下界 0.0,
        # 双验会去跑 `te = ts + 0` 的**零时长故障卡**(write_fault_card 对 te==ts 不报错、
        # 照写 Ts=Te),引擎判"无扰动 → 稳" → **空洞通过**,白烧 2 次仿真且把 G1 灌水。
        if (a.verify and row["cct"] is not None and not res.is_censored
                and not res.infeasible and not row["is_censored_low"] and row["cct"] > 0):
            lo_res = solver.run_transient(work, FaultSpec(
                c["i"], c["j"], c["id"], k_pct=c["k_pct"], ts=TS, te=TS + row["cct"]),
                t_end=a.t_end)
            hi_res = solver.run_transient(work, FaultSpec(
                c["i"], c["j"], c["id"], k_pct=c["k_pct"], ts=TS,
                # 7 位而非 3 位:3 位会把 cct+TOL 再推上引擎 0.01s 栅格的半步平分点,
                # 复现 psasp.py:471 刚修掉的那类伪影(只是方向相反 = 上侧假通过)
                te=round(TS + row["cct"] + TOL, 7)), t_end=a.t_end)
            row["verify"] = {
                "te_cct_stable": bool(lo_res.feasible and lo_res.stable),
                "te_cct_plus_tol_unstable": bool(hi_res.feasible and not hi_res.stable),
                "ok": bool(lo_res.feasible and lo_res.stable and hi_res.feasible
                           and not hi_res.stable)}
        robust_rmtree(work)
        with open(jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        n_done += 1
        v = row.get("verify", {})
        print(f"[{n_done}/{len(jobs)}] {tag}: CCT={row['cct']}"
              f"{' (删失)' if res.is_censored else ''}"
              f"{' (infeasible)' if res.infeasible else ''}"
              f"  {res.n_sims} 次  {row['elapsed_s']}s"
              + (f"  验证 {'OK' if v.get('ok') else 'FAIL'}" if v else ""), flush=True)

    rows = [json.loads(ln) for ln in jsonl.read_text(encoding="utf-8").splitlines()]

    def _usable(r):
        """**可用 = 有点值 ∧ 非 infeasible ∧ 上下都不删失**(2026-07-27 审查 critical)。

        旧 `n_valid` 只排了 `infeasible`:删失行的 `cct` 是哨兵(上侧 = hi = 0.5、
        下侧 = lo = 0.0)、`infeasible=False`,于是被算成"有效",还污染 `cct_min/max`。
        老批 `n_censored=0` 所以从未暴露;定向批的**退化层正是删失该点火的地方**,
        不修则"可用率"恒报 100%,买不买退化层直接判反。
        """
        return (r["cct"] is not None and not r["infeasible"]
                and not r["is_censored"] and not r.get("is_censored_low", False))

    usable = [r for r in rows if _usable(r)]
    valid_legacy = [r for r in rows if r["cct"] is not None and not r["infeasible"]]
    ver = [r for r in rows if r.get("verify")]
    n_ok = sum(1 for r in ver if r["verify"]["ok"])

    def _bucket(key):
        """逐层/逐电池拆分(并池会冒充能力,报数纪律要求分层)。"""
        out_ = {}
        for r in rows:
            k = str(r.get(key))
            b = out_.setdefault(k, {"n": 0, "usable": 0, "censored": 0,
                                    "censored_low": 0, "infeasible": 0,
                                    "verify_n": 0, "verify_ok": 0})
            b["n"] += 1
            b["usable"] += int(_usable(r))
            b["censored"] += int(r["is_censored"])
            b["censored_low"] += int(r.get("is_censored_low", False))
            b["infeasible"] += int(r["infeasible"])
            if r.get("verify"):
                b["verify_n"] += 1
                b["verify_ok"] += int(r["verify"]["ok"])
        return out_

    summary = {
        "battery": a.battery, "lib": str(lib), "n": len(rows),
        # n_valid 保留旧语义(向后兼容旧消费方),**判据一律用 n_usable**
        "n_valid": len(valid_legacy), "n_usable": len(usable),
        "n_censored": sum(1 for r in rows if r["is_censored"]),
        "n_censored_low": sum(1 for r in rows if r.get("is_censored_low", False)),
        "n_infeasible": sum(1 for r in rows if r["infeasible"]),
        "cct_min": min((r["cct"] for r in usable), default=None),
        "cct_max": max((r["cct"] for r in usable), default=None),
        "mean_sims": round(sum(r["n_sims"] for r in rows) / max(len(rows), 1), 2),
        "verify": {"n": len(ver), "ok": n_ok,
                   "rate": round(n_ok / len(ver), 4) if ver else None,
                   "protocol": "Te=cct 稳 且 Te=cct+0.01 失稳(CCT 定义直接验证)"},
        "by_stratum": _bucket("stratum"), "by_battery": _bucket("battery"),
        "serial": True, "t_end": a.t_end,
        "criteria_version": "v1_20260712",
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "cct_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[t4.4] 完成 {len(rows)}:**可用 {len(usable)}** / 上侧删失 "
          f"{summary['n_censored']} / 下侧删失 {summary['n_censored_low']}"
          f" / infeasible {summary['n_infeasible']};验证 {n_ok}/{len(ver)} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
