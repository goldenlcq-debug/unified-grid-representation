"""暂稳批产数据体检(2026-07-13 用户裁决"做";只读诊断,不改任何标签/数据)。

清洗政策(同日对话对齐):**管线/求解器伪影激进清洗,物理极端只隔离不删**。本工具
只负责"发现并如实报告"——处置(重跑/隔离)另行按纪律执行(隔离留档 + manifest +
变更记录;阈值调整走 8.9)。

四项逐行体检(对 ts_labels / ts_seeds 输出目录):
1. **anchor 锚定校验**:同一物理量在 STOUT(轨迹)与 STANA(引擎极值摘要)各存一份,
   逐行核对——功角 spread↔col6、SME Type-2 电压 min/max↔col8/col14。超差 = 解码键
   错配(SME 混批)/列错位/引擎输出异常,任何一种都是错标签的前兆。
   容差:角 5e-3°(STANA 打印 3 位小数)+ 1e-6 相对项(大角度打印舍入);压 1e-4 pu
   (打印 5 位小数)。
2. **finite 有限性**:轨迹与 STANA 全部数值必须有限(NaN/Inf 会静默传进 min/max)。
3. **window 窗长**:实际仿真末刻 ≥ 指令 t_end − 0.5s(引擎中途停会以
   recover_evaluable=False 静默混过);STOUT↔STANA 行数一致(judge 已兜底,此处计数)。
4. **t0 初值一致性**:暂态轨迹 t=0 的 Type-2 母线电压 vs LF.LP1 潮流解逐母线对比
   (暂态初始化正确性的独立见证);需 --lib(或算例留档)提供 LP1,缺则如实跳过。
   容差 2e-3 pu(两侧打印精度 + 初始化数值容差)。

输出 <labels>/audit_report.json:逐检查通过/失败计数 + 最差样本 + 逐行失败清单;
退出码:0 = 全过,1 = 有失败行(供管线串接)。

  python scripts/ts_audit.py --labels runs/ts_labels_cepri36 \
      --lib E:/AI-2/统一计算大模型/gridfm-data/lib_cepri36
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402

from gridfm.psasp_io import cards  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402
from gridfm.psasp_io.stana import parse_stana  # noqa: E402
from gridfm.psasp_io.stout import (ANGLE_SENTINEL_DEG,  # noqa: E402
                                   extract_angles, extract_bus_voltage,
                                   max_angle_spread, parse_stout_blocks)

TOL_ANGLE_DEG = 5e-3       # STANA 打印 3 位小数
TOL_ANGLE_REL = 1e-6       # 大角度(±3万°)下打印舍入的相对项
TOL_V_PU = 1e-4            # STANA 打印 5 位小数
TOL_T0_PU = 2e-3           # 轨迹初值 vs LF.LP1 潮流解(两侧打印精度 + 初始化容差)
WINDOW_SLACK_S = 0.5       # 实际窗长允许比指令短的量(末步取整余量)


def _rows_of(root: Path):
    """支持 ts_labels(labels.jsonl)与 ts_seeds(seeds.json)两种目录。"""
    p = root / "labels.jsonl"
    if p.exists():
        return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()]
    p = root / "seeds.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["rows"]
    raise SystemExit(f"{root} 既无 labels.jsonl 也无 seeds.json")


def _lp1_lookup(lib: Path):
    cache = {}

    def get(sid):
        if lib is None or sid is None:
            return None
        if sid not in cache:
            p = lib / "cases" / sid / "LF.LP1"
            cache[sid] = ({b.bus_no: b.v_pu for b in cards.parse_lp1(Card.load(p))}
                          if p.exists() else None)
        return cache[sid]
    return get


def audit_one(root: Path, row: dict, lp1_of) -> dict:
    tag = row["tag"]
    out = {"tag": tag}
    stana_p = root / "stana" / f"{tag}.dat"
    ddd_p = root / "stout" / f"{tag}.ddd"
    sme_p = root / "sme" / f"{tag}.sme"
    if not sme_p.exists():
        sme_p = root / "ST.SME"
    if not (stana_p.exists() and ddd_p.exists() and sme_p.exists()):
        out["skip"] = "missing_archive"
        return out

    series = parse_stana(stana_p)
    try:
        t, blocks = parse_stout_blocks(ddd_p, sme_p)
    except Exception as e:  # noqa: BLE001  解析失败本身就是体检发现
        out["fail"] = {"parse": str(e)}
        return out
    fails = {}

    # 2. finite(先查:NaN 会污染后续比较)
    raw_ok = all(np.isfinite(arr).all() for _, arr in blocks) and np.isfinite(t).all()
    stana_ok = all(np.isfinite(series.t).all() if hasattr(series.t, "all") else
                   all(np.isfinite(x) for x in col)
                   for col in (series.t, series.angle_max, series.vmin,
                               series.vmax or [0.0], series.fmin or [0.0],
                               series.fmax or [0.0]))
    if not (raw_ok and stana_ok):
        fails["finite"] = {"stout_ok": bool(raw_ok), "stana_ok": bool(stana_ok)}

    # 3. window
    t_cmd = float(row.get("t_end") or 0.0)
    t_act = float(t[-1])
    if t_cmd and t_act < t_cmd - WINDOW_SLACK_S:
        fails["window"] = {"commanded_s": t_cmd, "actual_s": round(t_act, 3)}
    if len(t) != len(series.t):
        fails["rows"] = {"stout": len(t), "stana": len(series.t)}

    # 1. anchors(行数一致才逐行比;不一致已在 rows 记账)
    if "finite" not in fails and len(t) == len(series.t) and series.t:
        ang, _pairs = extract_angles(blocks)
        spread = max_angle_spread(ang)
        ref = np.asarray(series.angle_max)
        # **封顶段排除**(2026-07-16 对抗审查 critical;118 实测 47/63 例命中):
        # 引擎把 STANA 的**角差**与 STOUT 的**逐机 δ** 各自封顶到 99999 —— 封的是两个
        # 不同的量,故触顶后 max_angle_spread(δ) 与 STANA 角差列必然不等(实测差
        # 2k~22k°)。这不是解码错误,是真值不可得:锚点在封顶段本就不可比,一律红会
        # 淹没锚点的真实信号(SME 错列/列错位)并让收口门永久拒纳整批。
        # 掩码必须**两侧都看**:STOUT 侧常常先于 STANA 侧触顶(审查反驳者实证)。
        sat_rows = (ref >= ANGLE_SENTINEL_DEG) | (np.abs(ang).max(axis=1) >= ANGLE_SENTINEL_DEG)
        n_sat = int(sat_rows.sum())
        if n_sat:
            out["angle_rows_saturated"] = n_sat        # 如实计数,不静默
        cmp_rows = ~sat_rows
        if cmp_rows.any():
            err_a = float(np.abs(spread[cmp_rows] - ref[cmp_rows]).max())
            tol_a = TOL_ANGLE_DEG + TOL_ANGLE_REL * float(np.abs(ref[cmp_rows]).max())
            if err_a > tol_a:
                fails["anchor_angle"] = {"max_err_deg": round(err_a, 6),
                                         "tol": round(tol_a, 6),
                                         "rows_compared": int(cmp_rows.sum()),
                                         "rows_saturated": n_sat}
        else:
            out["angle_anchor_skipped"] = "all_rows_saturated"
        vbus, buses = extract_bus_voltage(blocks)
        if buses and series.vmax and len(series.vmax) != len(series.t):
            # STANA 末行截断到 9~18 列时扩展列(vmax/fmin/fmax)比 t 短一行——本身就是
            # 要体检出的伪影;不守卫会在下面广播运算崩掉整批(审查独立复现)
            fails["stana_ragged_rows"] = {"t": len(series.t), "vmax": len(series.vmax)}
        elif buses and series.vmax:
            err_lo = float(np.abs(vbus.min(axis=1) - np.asarray(series.vmin)).max())
            err_hi = float(np.abs(vbus.max(axis=1) - np.asarray(series.vmax)).max())
            if max(err_lo, err_hi) > TOL_V_PU:
                fails["anchor_voltage"] = {"vmin_err": round(err_lo, 6),
                                           "vmax_err": round(err_hi, 6),
                                           "tol": TOL_V_PU}
        # 4. t0 初值 vs 潮流解(有 Type-2 曲线且拿得到 LP1 才可评)
        lp1 = lp1_of(row.get("sample_id"))
        if buses and lp1:
            missing = [b for b in buses if b not in lp1]
            if missing:
                fails["t0_bus_mismatch"] = {"buses_not_in_lp1": missing[:5]}
            else:
                v0 = vbus[0]
                ref0 = np.array([lp1[b] for b in buses])
                err0 = float(np.abs(v0 - ref0).max())
                if err0 > TOL_T0_PU:
                    worst = int(np.abs(v0 - ref0).argmax())
                    fails["t0"] = {"max_err_pu": round(err0, 6),
                                   "bus": buses[worst], "tol": TOL_T0_PU}
                out["t0_checked"] = True
        elif buses and lp1 is None:
            out["t0_skipped"] = "no_lp1"
        if not buses:
            out["voltage_skipped"] = "no_type2_curves"

    if fails:
        out["fail"] = fails
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="ts_labels / ts_seeds 输出目录")
    ap.add_argument("--lib", default=None, help="方式库目录(t0 检查需 LF.LP1;缺则跳过)")
    a = ap.parse_args()
    root = Path(a.labels)
    rows = [r for r in _rows_of(root) if r.get("feasible")]
    lp1_of = _lp1_lookup(Path(a.lib) if a.lib else None)

    results = []
    for r in rows:
        try:
            results.append(audit_one(root, r, lp1_of))
        except Exception as e:  # noqa: BLE001  单行体检崩溃本身就是发现,不得炸穿整批
            results.append({"tag": r.get("tag"), "fail": {"audit_crash": str(e)}})
    failed = [r for r in results if r.get("fail")]
    skipped = [r for r in results if r.get("skip")]
    counts = {}
    for r in failed:
        for k in r["fail"]:
            counts[k] = counts.get(k, 0) + 1
    n_t0 = sum(1 for r in results if r.get("t0_checked"))
    report = {
        "n_feasible_rows": len(rows),
        "n_audited": len(results) - len(skipped),
        "n_skipped_missing_archive": len(skipped),
        "n_failed": len(failed),
        "fail_counts_by_check": counts,
        "t0_checked": n_t0,
        "t0_skipped_no_lp1": sum(1 for r in results if r.get("t0_skipped")),
        "no_type2_curves": sum(1 for r in results if r.get("voltage_skipped")),
        # 功角输出封顶(引擎哨兵 99999):锚点在封顶段不可比,如实计数不静默
        "rows_with_angle_saturation": sum(1 for r in results
                                          if r.get("angle_rows_saturated")),
        "angle_anchor_all_saturated": sum(1 for r in results
                                          if r.get("angle_anchor_skipped")),
        "tolerances": {"angle_deg": TOL_ANGLE_DEG, "angle_rel": TOL_ANGLE_REL,
                       "v_pu": TOL_V_PU, "t0_pu": TOL_T0_PU,
                       "window_slack_s": WINDOW_SLACK_S},
        "policy": "只读体检;伪影处置(重跑/隔离)按纪律另行执行,不自动清洗",
        "failures": failed,
    }
    (root / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[audit] {root.name}: 体检 {report['n_audited']} 行,失败 {len(failed)}"
          f"(按检查 {counts});t0 可评 {n_t0};跳过缺档 {len(skipped)}"
          f" → {root / 'audit_report.json'}")
    for r in failed[:10]:
        print(f"  失败 {r['tag']}: {r['fail']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
