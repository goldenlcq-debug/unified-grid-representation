"""T4.2 批量判稳标签(方案 10.3 M4;依赖 T2.3/T4.1;7.3 并行豁免条款)。

种子电池故障 × 方式库已解算例(generate_library --keep-cases 留存的 cases/<sid>/)
→ 判稳标签(D21 v1 判据;阶段 2 起注入 SME Type=2 全母线电压曲线 → 机端/逐母线
精确电压口径 + COI 频率)+ infeasible 分流。**并行粗产合规依据**:判稳二值标签
对并发漂移不敏感(7.3 豁免正式条文);CCT/机群等数值敏感标签不走本脚本(T4.4 单路
串行)。临界带样本(二值翻转/贴阈值)由交叉核对与 T4.4 复核兜底。

崩溃可续:labels.jsonl 逐行落盘,重跑自动跳过已完成 tag;STANA.DAT 每例留档
(out/stana/<tag>.dat,供 openclaw 交叉核对与阈值校准,免重跑);**STOUT.DDD 每例留档**
(out/stout/<tag>.ddd,~36KB/例——逐机功角轨迹本体,T4.3 过程监督的数据源,反推格式
已验证:503 步 × (1+n_coord) float64,机序 = ST.SME Coord_Type=1 行序,另存一份
ST.SME 于 out/ 供解码)。狗失效(slm_init error)= 立即中止等用户(铁律)。

  GRIDFM_PSASP_BIN=... python scripts/ts_labels.py \
      --lib E:/AI-2/统一计算大模型/gridfm-data/lib_cepri36 --battery cepri36 \
      [--out runs/ts_labels_cepri36] [--workers 2] [--limit 0] [--t-end 10.0]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from ts_seeds import BATTERY, BATTERY_V2, BATTERY_V3, BATTERY_V4, DURS, TS  # noqa: E402  (电池唯一定义源)

ALL_BATTERY = {**BATTERY, **BATTERY_V2, **BATTERY_V3, **BATTERY_V4}

from gridfm.fsutil import robust_copytree  # noqa: E402
from gridfm.psasp_io import cards  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402
from gridfm.psasp_io.modegate import TS_BASE_VMIN_FLOOR, base_vmin, ts_mode_ok  # noqa: E402
from gridfm.psasp_io.stana import parse_stana  # noqa: E402
from gridfm.psasp_io.stout import (  # noqa: E402
    add_bus_voltage_records, build_sme_from_scratch,
)
from gridfm.solver.base import FaultSpec  # noqa: E402
from gridfm.solver.psasp import (  # noqa: E402
    LicenseError, PsaspSolver, blank_s12 as blank_s12_card, set_s0_duration,
    write_fault_card,
)


def _jobs(lib: Path, battery: dict, limit: int, vmin_floor: float, only_sids=None):
    rows = list(csv.DictReader(open(lib / "manifest.csv", encoding="utf-8")))
    modes = [r for r in rows if r["solvable"] == "1" and (lib / "cases" / r["sample_id"]).is_dir()]
    skipped = sum(1 for r in rows if r["solvable"] == "1") - len(modes)
    if skipped:
        print(f"[t4.2] ⚠ {skipped} 个收敛方式缺 cases/ 目录被跳过(库需 --keep-cases 重产)")
    # **暂稳方式门**(2026-07-16 用户裁决;gridfm/psasp_io/modegate.py):基态 vmin
    # <门槛的方式,判据②的稳态分句在故障前就已违反 → 不论故障如何一律判失稳
    # (标签与故障无关)。这类方式不是合法的暂稳研究起点,不产标签。
    kept = [r for r in modes
            if ts_mode_ok(lib / "cases" / r["sample_id"], floor=vmin_floor)]
    dropped = len(modes) - len(kept)
    if dropped:
        print(f"[t4.2] 方式门(基态 vmin ≥ {vmin_floor}):{len(kept)}/{len(modes)} 方式入选"
              f",剔除 {dropped}(基态就违反判据②稳态分句,标签会与故障无关)")
    modes = kept
    if only_sids is not None:            # 档2 v2:限定 121 抬压方式(29 keep 保留 v1,不重烧)
        modes = [r for r in modes if r["sample_id"] in only_sids]
        print(f"[t4.2] --only-sids 限定:{len(modes)}/{len(kept)} 方式入选(名单 {len(only_sids)} 个)")
    if limit:
        modes = modes[:limit]
    # 时长档:电池可自带 durs(118 补 0.05/0.08 铺开基率梯度,见 ts_seeds);缺省 = DURS
    durs = battery.get("durs", DURS)
    out = []
    for r in modes:
        sid = r["sample_id"]
        for (i_no, j_no, id_no, k) in battery["faults"]:
            for dur in durs:
                tag = f"{sid}_L{i_no}-{j_no}-{id_no}_d{int(dur * 1000):03d}"
                out.append((tag, sid, (i_no, j_no, id_no, k), dur))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True, help="方式库目录(需含 cases/ 与 manifest.csv)")
    ap.add_argument("--battery", required=True, choices=sorted(ALL_BATTERY),
                    help="故障电池(与 ts_seeds 同一定义源;*_v2 = 扩产电池)")
    ap.add_argument("--out", default=None, help="默认 runs/ts_labels_<battery>")
    ap.add_argument("--workers", type=int, default=2,
                    help="并行度(7.3 豁免:仅判稳二值标签可并行粗产)")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个方式(冒烟)")
    ap.add_argument("--t-end", type=float, default=10.0)
    ap.add_argument("--timeout", type=int, default=300,
                    help="单例引擎超时秒数(冒烟实测:并发+GPU 训练下 120s 会把慢例"
                         "误标 infeasible;超时另标 timeout 且重试 1 次)")
    ap.add_argument("--keep-work", action="store_true", help="保留工作目录(默认删,省盘)")
    ap.add_argument("--requeue-infeasible", action="store_true",
                    help="把已有 labels.jsonl 中 infeasible/timeout 行剔除重跑(建议配"
                         " --workers 1:并发下引擎会偶发初始化误报 infeasible,"
                         "2026-07-12 种子重跑实测 6/18,串行复核是收口动作)")
    ap.add_argument("--no-sme-vbus", action="store_true",
                    help="不注入 SME Type=2 全母线电压曲线(阶段 2 起默认注入:机端/"
                         "逐母线精确电压口径 + T4.3 电压轨迹顺产;此开关仅供复现旧口径)")
    ap.add_argument("--vmin-floor", type=float, default=TS_BASE_VMIN_FLOOR,
                    help="暂稳方式门:基态(故障前)全网最低电压下限。基态就 <0.9 的方式"
                         "会被判据②无条件判失稳(标签与故障无关)——2026-07-16 用户裁决"
                         "改为数据门剔除,不动判据(modegate.py)")
    ap.add_argument("--only-sids", default="",
                    help="只标注该 JSON 文件列出的 sample_id(档2 v2:限定 121 抬压方式,"
                         "29 keep 保留 v1 不重烧;缺省=全部过门方式,零行为变化)")
    ap.add_argument("--blank-s12", default="auto", choices=("auto", "on", "off"),
                    help="删 S12 切机扰动卡(生产协议常设决定 2026-07-24)。"
                         "auto(缺省)= 按电池 blank_s12 旗(ts_seeds 定义源);"
                         "on/off = 强制覆盖(off 仅供复现旧批,须在变更记录说明理由)")
    a = ap.parse_args()

    lib = Path(a.lib)
    out = Path(a.out or f"runs/ts_labels_{a.battery}")
    (out / "stana").mkdir(parents=True, exist_ok=True)
    (out / "stout").mkdir(parents=True, exist_ok=True)
    (out / "sme").mkdir(parents=True, exist_ok=True)
    jsonl = out / "labels.jsonl"
    if a.requeue_infeasible and jsonl.exists():
        kept, dropped = [], 0
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("feasible") and not row.get("timeout"):
                kept.append(line)
            else:
                dropped += 1
        jsonl.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        print(f"[t4.2] 剔除 infeasible/timeout {dropped} 行待串行复核(保留 {len(kept)})")
    done = set()
    if jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["tag"])
            except (ValueError, KeyError):
                continue
    only_sids = None
    if a.only_sids:
        _d = json.loads(Path(a.only_sids).read_text(encoding="utf-8"))
        only_sids = set(_d if isinstance(_d, list) else _d.get("sample_ids", []))
    # 删 S12 常设决定:auto = 电池旗;on/off 强制。逐批落进 summary(审计可回溯)
    blank_s12 = (bool(ALL_BATTERY[a.battery].get("blank_s12")) if a.blank_s12 == "auto"
                 else a.blank_s12 == "on")
    print(f"[t4.2] blank_s12 = {blank_s12}(--blank-s12 {a.blank_s12};"
          f"电池旗 {bool(ALL_BATTERY[a.battery].get('blank_s12'))})")
    if not blank_s12 and ALL_BATTERY[a.battery].get("blank_s12"):
        print("[t4.2] ⚠ 该电池按生产协议应删 S12,当前被 --blank-s12 off 强制关闭"
              "——须在变更记录写明理由", flush=True)
    jobs = [j for j in _jobs(lib, ALL_BATTERY[a.battery], a.limit, a.vmin_floor, only_sids)
            if j[0] not in done]
    print(f"[t4.2] 库 {lib.name}:待跑 {len(jobs)}(已完成跳过 {len(done)});"
          f"workers={a.workers}(7.3 豁免)")

    solver = PsaspSolver()
    lock = threading.Lock()
    stop = threading.Event()
    t0 = time.time()
    n_done = 0
    bus_cache: dict = {}

    def lib_buses(sid: str):
        """样本全母线清单(LF.LP1)+ 真实机组数(LF.LP5 行数,= 覆盖门基准)
        + 基态 vmin(方式门的判定量,逐行落盘供审计);按 sid 缓存。"""
        with lock:
            if sid in bus_cache:
                return bus_cache[sid]
        case = lib / "cases" / sid
        buses = sorted(b.bus_no for b in cards.parse_lp1(Card.load(case / "LF.LP1")))
        n_gen = len(cards.parse_lp5(Card.load(case / "LF.LP5")))
        v0 = base_vmin(case)
        with lock:
            bus_cache[sid] = (buses, n_gen, v0)
        return buses, n_gen, v0

    def run_one(job):
        tag, sid, (i_no, j_no, id_no, k), dur = job
        if stop.is_set():
            return None
        fault = FaultSpec(i_no, j_no, id_no, k_pct=k, ts=TS, te=TS + dur)
        t1 = time.time()
        timed_out = False
        for attempt in (1, 2):            # 超时重试 1 次(判稳二值不受重跑漂移影响,7.3)
            work = out / "work" / tag
            robust_copytree(lib / "cases" / sid, work)   # 并行下句柄竞态重试(fsutil)
            t_at = time.time()
            buses, n_gen_case, v_base = lib_buses(sid)
            try:
                if ALL_BATTERY[a.battery].get("blank_s11"):
                    # 交付卡为 K≠0 模板(故障点母线只对原线路有效)——换线路必须清空,
                    # 走 write_fault_card 从零写卡路径(仅 K=0 端部,Nfault 同步)
                    (work / "ST.S11").write_bytes(b"")
                if blank_s12:
                    # **生产协议常设决定(2026-07-24 用户裁定;2026-07-25 入主产线)**:
                    # 判稳批产删除交付模板的 S12 切机卡。配方与 s12_recheck.remove_s12
                    # 同源(ST.S12 置空 + ST.S0 idx8 Ndist 计数清零——引擎按计数读卡,
                    # 卡与计数必须同步,同 Nfault 教训);对无 S12 的模板幂等
                    # (cepri36 实测无 ST.S12 且 Ndist=0)。
                    # 注:勘误实测该卡在全 118 粮仓 22,155 条轨迹中从未动作(翻案 0/453),
                    # 故删卡是"无害且简化模板",非数据修正。
                    blank_s12_card(work)
                if not a.no_sme_vbus:     # 阶段 2:机端/逐母线精确电压口径的数据源
                    sme = work / "ST.SME"
                    if not sme.exists() or not sme.read_bytes().strip():
                        # 118 类交付工程无 ST.SME(曲线配置本就由我方注入):从零构造
                        # 逐机功角(寻址随 L5 版式二分)+ Type=8/11(STANA 产出开关)
                        build_sme_from_scratch(sme, work / "LF.L5")
                    add_bus_voltage_records(sme, buses)
                write_fault_card(work / "ST.S11", fault)
                set_s0_duration(work, a.t_end)   # 07-12 修:t_end 曾不生效,cepri36 5s 窗截断慢失稳
                r = solver.run_transient_file(work, work / "ST.S11", timeout=a.timeout)
            except LicenseError:
                stop.set()
                raise
            timed_out = (not r.feasible) and (time.time() - t_at) >= a.timeout - 5
            if not timed_out:
                break
        det = r.verdict_detail or {}
        stana = work / "STANA.DAT"
        last1s_vmin = None
        if stana.exists():
            shutil.copy2(stana, out / "stana" / f"{tag}.dat")
            series = parse_stana(stana)
            if series.t:
                tl = series.t[-1] - 1.0
                tail = [v for t, v in zip(series.t, series.vmin) if t >= tl]
                last1s_vmin = min(tail) if tail else None
        stout = work / "STOUT.DDD"
        if stout.exists():
            shutil.copy2(stout, out / "stout" / f"{tag}.ddd")
        sme = work / "ST.SME"
        if sme.exists():
            # **逐 tag 留档**:SME 是 .ddd 的解码键,列布局随 Type=2 注入而变
            # (43→82 列)。曾只在目录级写一次(不覆盖)——续跑/requeue 混批时新
            # .ddd 配旧 SME,ts_relabel 静默解析失败退化到已否决的单机瞬时频率口径
            # (2026-07-13 审查确认)。目录级 ST.SME 仍写(向后兼容 + 人读),但以
            # 逐 tag 副本为准。
            shutil.copy2(sme, out / "sme" / f"{tag}.sme")
            shutil.copy2(sme, out / "ST.SME")
        row = {"tag": tag, "sample_id": sid, "battery": a.battery,
               "i": i_no, "j": j_no, "id": id_no, "k_pct": k,
               "ts": TS, "te": round(TS + dur, 4), "dur": dur, "t_end": a.t_end,
               "feasible": r.feasible, "stable": r.stable, "timeout": timed_out,
               "n_gen_case": n_gen_case,     # 覆盖门基准(LF.LP5 机组行数)
               "base_vmin": v_base,          # 方式门判定量(故障前 LF.LP1 全网最低电压)
               "source": det.get("source"),
               "angle_peak_deg": det.get("angle_peak_deg"),
               "vmin_pu": det.get("vmin_pu"),
               "vterm_min_pu": det.get("vterm_min_pu"),
               "vterm_full": det.get("vterm_full"),
               "n_vterm_curves": det.get("n_vterm_curves"),
               "n_vbus_curves": det.get("n_vbus_curves"),
               "v_recover_source": det.get("criteria", {}).get("v_recover_source"),
               "freq_source": det.get("freq_source"),
               "recover_evaluable": det.get("recover_evaluable"),
               "instability_types": det.get("instability_types"),
               "voltage_modes": det.get("voltage_modes"),
               "traj_note": det.get("traj_note"),
               "vlow_hold_s": det.get("vlow_worst_hold_s"),
               "vlow_from_s": det.get("criteria", {}).get("vlow_from_s"),
               # 临界带标记(7.3 串行复核入口):末秒最低电压 <0.9 = openclaw 稳态
               # 判据会翻案的边缘带(阈值校准账见变更记录 07-12)
               "last1s_vmin": last1s_vmin,
               "v_marginal": bool(last1s_vmin is not None and last1s_vmin < 0.9),
               "elapsed_s": round(time.time() - t1, 2)}
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
        return row

    try:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(run_one, j) for j in jobs]
            for f in as_completed(futs):
                row = f.result()          # LicenseError 在此抛出并中止
                if row is None:
                    continue
                with lock:
                    with open(jsonl, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_done += 1
                    if n_done % 20 == 0 or n_done == len(jobs):
                        el = time.time() - t0
                        print(f"[t4.2] {n_done}/{len(jobs)}  {el:.0f}s "
                              f"({el / max(n_done, 1):.1f}s/例)", flush=True)
    except LicenseError:
        print("[t4.2] ✋ slm_init error:加密狗失效,已中止;重插狗后重跑同命令续接",
              flush=True)
        return 2

    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    n_st = sum(1 for r in rows if r["feasible"] and r["stable"])
    n_un = sum(1 for r in rows if r["feasible"] and not r["stable"])
    n_to = sum(1 for r in rows if r.get("timeout"))
    n_inf = sum(1 for r in rows if not r["feasible"] and not r.get("timeout"))
    summary = {"lib": str(lib), "battery": a.battery, "n": len(rows),
               "stable": n_st, "unstable": n_un, "infeasible": n_inf, "timeout": n_to,
               "workers": a.workers,
               # 删 S12 常设决定的执行留痕(审计可回溯;mode=auto/on/off)
               "blank_s12": blank_s12, "blank_s12_mode": a.blank_s12,
               "vmin_floor": a.vmin_floor,   # 方式门(2026-07-16 裁决;modegate.py)
               "criteria_source": "v1_20260712(D21;阶段 2 机端/逐母线精确口径,"
                                  "sme_vbus=" + ("off" if a.no_sme_vbus else "on") + ")",
               "elapsed_s": round(time.time() - t0, 1)}
    (out / "labels_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[t4.2] 完成 {len(rows)}:稳 {n_st} / 失稳 {n_un} / infeasible {n_inf}"
          f" / timeout {n_to} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
