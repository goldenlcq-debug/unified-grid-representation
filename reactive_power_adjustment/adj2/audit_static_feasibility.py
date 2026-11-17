# -*- coding: utf-8 -*-
r"""六档健康方式库的静态可行性审计(2026-08-22;只读既有 LF.LP1/LF.LP2,不跑 PSASP)。

口径(文献 000/002/003/005 一致做法):
  电压:逐母线对照**该算例自带**的 VMIN/VMAX(MATPOWER bus 列 12/13;118/300 = 0.94~1.06,3120sp = 0.95~1.12/1.11/1.05,
        ACTIVSg = 0.90~1.10)→ BVR(越限母线占比,002 的定义)与"全母线在带内"的方式数;另报 0.95~1.05 与 0.94~1.06 两把尺子作对照。
  热稳定:支路视在功率 S = max(|S_i|,|S_j|)(LP2,pu×baseMVA)对照 MATPOWER rateA(MVA);支路映射靠转换器写进 L2 的 idname "L{k}_ac"
        (k = MATPOWER 支路行号)。交流线 LP2 与变压器 LP3 同格式,一并审计(idname L{k}_ac / T{k}_2w / T{k}_sf)。rateA=0 的算例(118/300)判"源数据无额定值"。
  python audit_static_feasibility.py
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"E:\统一计算大模型-迁移包\增量-20260902\N-k其他推理器实验-20260905\grid-foundation")
LINE = Path(r"E:\统一计算大模型-迁移包\增量-20260902\大算例样本线-20260820")
MP = Path(r"E:\统一计算大模型-迁移包\增量-20260902\MATPOWER转换-20260818\matpower_cases")
sys.path.insert(0, str(REPO))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
from gridfm.psasp_io import cards  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402

TIERS = {"118": (["mp_case118"], "case118.m"), "300": (["mp_case300"], "case300.m"),
         "3120": (["mp_case3120sp_fix"], "case3120sp.m"),
         "2000": (["mp_activsg2000", "mp_activsg2000_b2"], "case_ACTIVSg2000.m"),
         "25k": (["mp_activsg25k", "mp_activsg25k_b2"], "case_ACTIVSg25k.m"),
         "70k": (["mp_activsg70k", "mp_activsg70k_b2"], "case_ACTIVSg70k.m")}


def read_m(path: Path):
    t = io.open(path, encoding="utf-8", errors="replace").read()
    base = float(re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", t).group(1))
    def mat(name):
        m = re.search(r"mpc\.%s\s*=\s*\[(.*?)\];" % name, t, re.S)
        rows = [r.split("%")[0].replace(";", " ").split() for r in m.group(1).strip().splitlines()]
        rows = [r for r in rows if r]
        return np.array([[float(x) for x in r] for r in rows])
    bus, br = mat("bus"), mat("branch")
    # 与 mp2psasp.preprocess 同步:剔除 status=0 支路与不连通/type4 母线后,转换器按 br_on 行序给 idname 编号 L{k}/T{k}
    # (2026-08-22 修:ACTIVSg25k 有 1 条停运支路,按原始行号对会整体错位 14032 条)
    br_on = br[br[:, 10] > 0] if br.shape[1] > 10 else br
    connected = set(br_on[:, 0].astype(int)) | set(br_on[:, 1].astype(int))
    keep = np.array([(int(b[1]) != 4) and (int(b[0]) in connected) for b in bus])
    bus = bus[keep]
    ext = set(bus[:, 0].astype(int))
    br_on = np.array([b for b in br_on if int(b[0]) in ext and int(b[1]) in ext])
    return base, bus, br_on


def main() -> int:
    out = {"criteria": {"voltage": "per-bus MATPOWER VMIN/VMAX (文献 000/002/003/005 口径)",
                        "thermal": "S/rateA, 交流线(LP2)+变压器(LP3), S=max(|S_i|,|S_j|)×baseMVA"}, "tiers": {}}
    print("(热稳定:额定值<10 MVA 的支路视为可疑不计;支路序号按转换器 br_on 行序)")
    for tier, (libs, mfile) in TIERS.items():
        base, bus, br = read_m(MP / mfile)
        vmin_b, vmax_b = bus[:, 12], bus[:, 11]
        n_bus = len(bus)
        rate = br[:, 5]
        RATE_SUSPECT = 10.0                      # 额定值 <10 MVA 视为可疑(70k 有 56 条 <1 MVA),不计入越限统计、单列计数
        has_rate = rate >= RATE_SUSPECT
        n_suspect = int(((rate > 0) & (rate < RATE_SUSPECT)).sum())
        rows = []
        # 底图本身先审一遍(区分"底图自带越限"与"扰动造成")
        dirs = [(libs[0], LINE / "libs" / libs[0] / "_base_case", True)]
        for lib in libs:
            ci = json.loads((LINE / "libs" / lib / "convert_info.json").read_text(encoding="utf-8"))
            if ci.get("dropped_buses"):
                print("  [warn] %s dropped_buses=%s,母线序可能错位,电压审计按 min(n) 截断" % (lib, ci["dropped_buses"][:5]))
            for d in sorted((LINE / "libs" / lib / "cases").iterdir()):
                if d.is_dir():
                    dirs.append((lib, d, False))
        for lib, d, is_base in dirs:
            if True:
                # 电压:逐母线对照自带限值
                v = np.zeros(n_bus)
                for r in cards.parse_lp1(Card.load(d / "LF.LP1")):
                    if 1 <= r.bus_no <= n_bus:
                        v[r.bus_no - 1] = r.v_pu
                live = v > 0.01
                nv = int(live.sum())
                viol = live & ((v < vmin_b - 1e-4) | (v > vmax_b + 1e-4))
                bvr = viol.sum() / max(nv, 1)
                in95 = bool(np.all((v[live] >= 0.95 - 1e-4) & (v[live] <= 1.05 + 1e-4)))
                in94 = bool(np.all((v[live] >= 0.94 - 1e-4) & (v[live] <= 1.06 + 1e-4)))
                # 热:交流线 LP2 → idname L{k}_ac → rateA[k]
                load_max, n_over, n_checked = None, None, 0
                if has_rate.any() and (d / "LF.LP2").is_file():
                    ratios = []
                    for fname in ("LF.LP2", "LF.LP3"):          # LP3(变压器)与 LP2 同格式,idname T{k}_2w/T{k}_sf
                        if not (d / fname).is_file():
                            continue
                        for fl in cards.parse_lp2(Card.load(d / fname)):
                            m = re.match(r"[LT](\d+)_", (fl.name or "").strip())
                            if not m:
                                continue
                            k = int(m.group(1))
                            if k >= len(rate) or not has_rate[k]:
                                continue
                            s = max(abs(complex(fl.pi, fl.qi)), abs(complex(fl.pj, fl.qj))) * base
                            ratios.append(s / rate[k])
                    if ratios:
                        n_checked = len(ratios)
                        load_max = float(max(ratios))
                        n_over = int(sum(1 for x in ratios if x > 1.0 + 1e-6))
                        p99 = float(np.percentile(ratios, 99))
                rows.append({"lib": lib, "sid": ("_base_case" if is_base else d.name), "is_base": is_base,
                             "n_live": nv, "n_viol_own": int(viol.sum()), "bvr_own": round(float(bvr), 4),
                             "vmin": round(float(v[live].min()), 4), "vmax": round(float(v[live].max()), 4),
                             "in_0.95_1.05": in95, "in_0.94_1.06": in94,
                             "thermal_n_checked": n_checked, "thermal_max_loading": (round(load_max, 3) if load_max is not None else None),
                             "thermal_p99_loading": (round(p99, 3) if load_max is not None else None),
                             "thermal_n_over": n_over,
                             "thermal_frac_over": (round(n_over / n_checked, 5) if n_checked else None)})
        base_row = [r for r in rows if r["is_base"]][0]
        rows = [r for r in rows if not r["is_base"]]
        n = len(rows)
        all_in = sum(1 for r in rows if r["n_viol_own"] == 0)
        bvr_med = float(np.median([r["bvr_own"] for r in rows]))
        n95 = sum(1 for r in rows if r["in_0.95_1.05"]); n94 = sum(1 for r in rows if r["in_0.94_1.06"])
        th = [r for r in rows if r["thermal_max_loading"] is not None]
        th_clean = sum(1 for r in th if r["thermal_n_over"] == 0)
        th_nover_med = float(np.median([r["thermal_n_over"] for r in th])) if th else None
        th_p99_med = float(np.median([r["thermal_p99_loading"] for r in th])) if th else None
        band = "%.2f~%s" % (float(np.min(vmin_b)), "/".join("%.2f" % x for x in sorted(set(np.round(vmax_b, 2)), reverse=True)))
        out["tiers"][tier] = {"n_modes": n, "own_band": band, "n_all_in_own_band": all_in, "bvr_own_median": round(bvr_med, 4),
                              "n_in_0.95_1.05": n95, "n_in_0.94_1.06": n94,
                              "thermal_rateA_available": bool(has_rate.any()), "thermal_rate_suspect_lt10MVA": n_suspect,
                              "thermal_n_lines_checked_median": (int(np.median([r["thermal_n_checked"] for r in th])) if th else 0),
                              "thermal_n_modes_no_overload": th_clean, "thermal_n_over_median": th_nover_med,
                              "thermal_p99_loading_median": th_p99_med,
                              "base_case": base_row, "rows": rows}
        print("%-5s| %4d | %-15s | 底图:越限母线 %d/过载支路 %s | 方式:全母线在带内 %d | BVR中位 %.1f%% | 0.95~1.05 %d | 0.94~1.06 %d | 热(可审 %s 条,可疑 %d): 无过载方式 %s | 过载支路数中位 %s | p99 载流比中位 %s" % (
            tier, n, band, base_row["n_viol_own"], (base_row["thermal_n_over"] if base_row["thermal_n_over"] is not None else "—"),
            all_in, 100 * bvr_med, n95, n94,
            (out["tiers"][tier]["thermal_n_lines_checked_median"] if th else "无额定值"), n_suspect,
            (th_clean if th else "—"), (th_nover_med if th else "—"), ("%.2f" % th_p99_med) if th_p99_med is not None else "—"))
    o = LINE / "调整器跨规模验证-20260821" / "audit_static_feasibility.json"
    o.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("落盘", o)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
