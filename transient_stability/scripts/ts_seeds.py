"""39/CEPRI-36 暂稳种子电池(M4 起步 = T4.2 种子;D16 标签级单路串行)。

每 (算例, 故障支路, 切除时长) 一次暂态:golden fixture 拷到独立工作目录 →
**注入 SME Type-2 全母线电压曲线(阶段 2)** → write_fault_card(从零写卡 + Nfault
同步)→ set_s0_duration(t_end)→ wmudrt → **D21 v1 判据**(机端/逐母线精确电压口径 +
COI 频率;gridfm/psasp_io/stana.judge_files_v1)。产出 <out>/seeds.json + 逐例留档
(stana/ stout/ sme/,供离线重判与机群提取,免重跑)。

CEPRI-36:K=0% 三相接地自清除短路(Add_No=0,实算验证);39:自带 GUI 模板
(K=20%,故障点母线 43)整卡复用只改时序——其 S12 自带一张切机扰动卡(交付原状,
Ndist=1),种子按交付口径保留并在结果中注明。

  python scripts/ts_seeds.py [--out runs/ts_seeds_v1_12s] [--t-end 12]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.fsutil import robust_copytree  # noqa: E402
from gridfm.psasp_io import cards  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402
from gridfm.psasp_io.stout import (  # noqa: E402
    add_bus_voltage_records, build_sme_from_scratch,
)
from gridfm.solver.base import FaultSpec  # noqa: E402
from gridfm.solver.psasp import PsaspSolver, set_s0_duration  # noqa: E402

FIX = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
DURS = (0.1, 0.2, 0.3)
TS = 0.1
BATTERY = {
    "cepri36": {
        "fixture": "cepri36_golden",
        "faults": [(29, 27, 23, 0.0), (3, 8, 11, 0.0), (26, 24, 28, 0.0),
                   (22, 21, 35, 0.0), (9, 8, 32, 0.0)],
        "note": "K=0% 三相接地自清除短路,从零写卡",
    },
    "ieee39": {
        "fixture": "ieee39_golden",
        "faults": [(34, 28, 23, 20.0)],
        "note": "自带 GUI 模板整卡复用(含故障点母线 43);S12 切机扰动卡为交付原状保留",
    },
}

# v2 扩产电池(2026-07-13 用户裁决"可现在进行,多一些不可以少"):两库全部在运交流线
# (golden fixture LF.L2 枚举,mark=1);cepri36_v2 = 25 线扣除 v1 已产 5 线;ieee39_v2 =
# 全 34 线(含 34-28-23——v1 是 K=20% 中间故障,此处 K=0% 端部,是不同故障)。
# ieee39 交付卡自带 K≠0 模板(含故障点母线 43,只对原线路有效)→ 换线路必须
# blank_s11 从零写 K=0 端部卡(write_fault_card 空卡路径,Nfault 同步)。
# **只入 ts_labels 批产,不入 ts_seeds 种子回路**(种子电池保持 v1 定义)。
BATTERY_V2 = {
    "cepri36_v2": {
        "fixture": "cepri36_golden",
        "faults": [(i, j, d, 0.0) for (i, j, d) in
                   [(3, 9, 10), (3, 7, 12), (11, 6, 14), (13, 23, 16), (13, 22, 17),
                    (32, 21, 19), (31, 26, 21), (29, 26, 24), (29, 25, 25),
                    (29, 24, 26), (29, 20, 27), (26, 33, 29), (25, 9, 30),
                    (24, 9, 31), (8, 7, 33), (6, 23, 34), (27, 16, 37),
                    (33, 37, 41), (20, 19, 42), (37, 36, 43)]],
        "note": "v2 扩产:20 条 v1 未覆盖线路,K=0% 端部,从零写卡",
    },
    "ieee39_v2": {
        "fixture": "ieee39_golden",
        "blank_s11": True,
        "faults": [(i, j, d, 0.0) for (i, j, d) in
                   [(7, 4, 10), (6, 5, 11), (6, 39, 12), (5, 4, 13), (4, 3, 14),
                    (3, 9, 15), (40, 39, 16), (40, 37, 17), (37, 36, 18),
                    (36, 35, 19), (2, 30, 1), (35, 34, 20), (34, 33, 21),
                    (34, 31, 22), (34, 28, 23), (25, 34, 24), (33, 32, 25),
                    (22, 33, 26), (28, 27, 27), (27, 26, 28), (26, 25, 29),
                    (9, 2, 2), (24, 23, 30), (23, 22, 31), (23, 21, 32),
                    (23, 20, 33), (21, 20, 34), (30, 19, 3), (30, 24, 4),
                    (19, 8, 5), (19, 32, 6), (8, 7, 7), (8, 36, 8), (7, 6, 9)]],
        "note": "v2 扩产:全 34 线,K=0% 端部,blank_s11 从零写卡(替代交付 K≠0 模板);"
                "S12 切机扰动卡保持交付原状",
    },
}

# v3 电池(2026-07-16,收轮裁决 1+3 合并落地;拓扑维度 1→2 + K≠0 训练故障):
# **例数受暂稳方式门约束**(modegate.py,同日用户裁决):基态 vmin ≥0.905 的方式才产
# —— lib118 90/150 → 21×90×3 = 5,670 例;lib_cepri36 88/102 → 25×88×3 = 6,600 例。
# - ieee118_v1:训练粮仓第二拓扑家族。21 条在运重载/联络线 = golden LF.L2 mark=1 ×
#   LF.LP2 按 |Pi| 排序 top21(含 39-58 双回线 106/107 成对入选,真并联样本);
#   K=0% 端部。交付卡自带 K=50/Add_No=122 模板 → blank_s11 必开(换线残留陈旧
#   Add_No,写卡修复见 07-16 变更记录);交付 S12 扰动卡(母线 119,t=10s,12s 窗内)
#   照 ieee39_v2 先例保持原状。SME 由 ts_labels 自动从零构造(基准版 Type=3)。
# - cepri36_k50:K≠0 训练故障首批(K 通道死通道解封)。25 线 = v1 5 线 + v2 20 线
#   同清单,K=50%(中点,与 K=0 端部对照最大);T4.1 故障点母线写卡(07-16 落地,
#   K=0 vs K=50 实算物理差异验证)。tag 不含 K% → 与 cepri36_v2 同线同方式同时长
#   tag 撞车,跨批 join 必须带 battery(坑 4;出目录独立无碍)。
# **只入 ts_labels 批产**;点火前须用户确认 K 档位与算账(铁律:大额动作先算账)。
BATTERY_V3 = {
    "ieee118_v1": {
        "fixture": "ieee118_golden",
        "blank_s11": True,
        # **生产协议常设决定**(2026-07-24 用户裁定;2026-07-25 入主产线 ts_labels.py):
        # 判稳批产删 S12 切机卡(母线 119,t=10s)+ 同步清零 ST.S0 Ndist。
        # 存量批(ts_labels_ieee118_v{1,2})产于本旗之前、保留 S12,**两者物理等价**:
        # 勘误实测该卡在全 118 粮仓 22,155 条轨迹中从未动作,删卡对照重仿翻案 0/453、
        # STANA 逐字节相同(docs/勘误-S12判稳标签-20260724.md)。故续跑/补测混产无碍。
        "blank_s12": True,
        # **专属时长档**(2026-07-16 冒烟实测后补,纯增量):默认 3 档在 118 上全落进
        # 饱和区(同方式同 21 线实测失稳率 d100=81% / d200=95% / d300=95% → 基率 ~90%,
        # 训练信号弱且模型可走"118→失稳"捷径)。补 0.05/0.08 两档把梯度铺开
        # (实测 38% / 67%),标准 3 档一个不少(可随时只取子集复现旧口径)。
        "durs": (0.05, 0.08, 0.1, 0.2, 0.3),
        "faults": [(i, j, d, 0.0) for (i, j, d) in
                   [(37, 40, 111), (78, 83, 49), (119, 13, 22), (13, 24, 21),
                    (84, 86, 44), (70, 78, 64), (101, 37, 188), (77, 93, 50),
                    (93, 95, 34), (85, 86, 43), (115, 118, 168), (23, 27, 129),
                    (10, 14, 146), (41, 42, 103), (16, 17, 139), (40, 41, 105),
                    (44, 45, 100), (92, 93, 36), (76, 77, 53), (39, 58, 107),
                    (39, 58, 106)]],
        "note": "118 扩产:21 条重载/联络线(|Pi| top21,含 39-58 双回),K=0% 端部,"
                "blank_s11 从零写卡;5 档时长(补 0.05/0.08 铺开基率梯度);"
                "**blank_s12=True(2026-07-25 起,生产协议常设决定;存量批保留 S12,"
                "实测物理等价)**",
    },
    "cepri36_k50": {
        "fixture": "cepri36_golden",
        "faults": [(i, j, d, 50.0) for (i, j, d) in
                   [(29, 27, 23), (3, 8, 11), (26, 24, 28), (22, 21, 35), (9, 8, 32),
                    (3, 9, 10), (3, 7, 12), (11, 6, 14), (13, 23, 16), (13, 22, 17),
                    (32, 21, 19), (31, 26, 21), (29, 26, 24), (29, 25, 25),
                    (29, 24, 26), (29, 20, 27), (26, 33, 29), (25, 9, 30),
                    (24, 9, 31), (8, 7, 33), (6, 23, 34), (27, 16, 37),
                    (33, 37, 41), (20, 19, 42), (37, 36, 43)]],
        "note": "K=50% 中间故障(T4.1 故障点母线写卡):25 线 = v1 5 线 + v2 20 线"
                "同清单,与 K=0 电池构成位置对照",
    },
}

# v4 定向补测电池(2026-07-17 B2 过账;判稳收尾):**只评估、永不入训**。目标 = 频率低/
# 过电压两个稀有子类在留出方式上凑够 500 例报告门(定向口径单独挂门;自然分布口径
# 继续只报数,两口径不混算)。设计依据 = 零机时故障画像(变更记录 2026-07-17):
# f_low 高产线 33-37/37-36(0.3s×K50 实测 P=0.99~1.0)+ 26-33;v_over 高产线
# 27-16/32-21/11-6/6-23(P≈0.44~0.56)。K=25/75 拆两个电池:tag 不含 K%(坑 4),
# 同电池双 K 同批撞 tag。durs 含新档 0.25/0.35/0.40(0.2 低产不取;判据/窗口与存量同)。
# **只跑 v5 划分的 17 个 cepri36 留出方式(ts_labels --only-sids)**,产出天然全落评估侧。
_SUB_LINES = [(33, 37, 41), (37, 36, 43), (26, 33, 29),               # f_low 高产
              (27, 16, 37), (32, 21, 19), (11, 6, 14), (6, 23, 34)]   # v_over 高产
BATTERY_V4 = {
    "cepri36_sub25": {
        "fixture": "cepri36_golden",
        "durs": (0.25, 0.3, 0.35, 0.4),
        "faults": [(i, j, d, 25.0) for (i, j, d) in _SUB_LINES],
        "note": "定向补测 K=25%:7 线 × 4 时长,只评估永不入训,留出方式限定",
    },
    "cepri36_sub75": {
        "fixture": "cepri36_golden",
        "durs": (0.25, 0.3, 0.35, 0.4),
        "faults": [(i, j, d, 75.0) for (i, j, d) in _SUB_LINES],
        "note": "定向补测 K=75%:同 sub25,K 对侧",
    },
}

# v_over 专用补测(2026-07-17 试产校准后追加):sub25 试产实测 v_over 在 K=25 下熄灭
# (5/84,历史高产全在 K=0/K=50——过电压模式与故障点位置强相关);f_low 三线 0.92~1.0
# 命中不受影响。补救 = 回到 K=0/K=50,线扩到 7 条历史高产线(P≈0.44~0.56),时长只用
# 未采样新档 0.15/0.25(与存量 0.1/0.2/0.3 无重复格子,不产重复标签)。同为只评估、
# 永不入训、--only-sids 留出方式。
_VOVER_LINES = [(27, 16, 37), (32, 21, 19), (11, 6, 14), (6, 23, 34),
                (29, 27, 23), (3, 8, 11), (3, 9, 10)]
BATTERY_V4.update({
    "cepri36_vover0": {
        "fixture": "cepri36_golden",
        "durs": (0.15, 0.25),
        "faults": [(i, j, d, 0.0) for (i, j, d) in _VOVER_LINES],
        "note": "v_over 定向(K=0%):7 条历史高产线 × 新时长档,只评估永不入训",
    },
    "cepri36_vover50": {
        "fixture": "cepri36_golden",
        "durs": (0.15, 0.25),
        "faults": [(i, j, d, 50.0) for (i, j, d) in _VOVER_LINES],
        "note": "v_over 定向(K=50%):同 vover0,K 中点",
    },
})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ts_seeds_v1_12s")
    ap.add_argument("--t-end", type=float, default=12.0,
                    help="仿真时长(阶段 2 = 12s:D21 含'故障后 10s'分句,10s 窗不可评)")
    ap.add_argument("--no-sme-vbus", action="store_true",
                    help="不注入 SME Type-2 全母线电压(仅供复现旧口径)")
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    for sub in ("stana", "stout", "sme"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    s = PsaspSolver()
    rows = []
    t0 = time.time()
    for case_name, spec in BATTERY.items():
        src = FIX / spec["fixture"]
        for i_no, j_no, id_no, k in spec["faults"]:
            for dur in DURS:
                tag = f"{case_name}_L{i_no}-{j_no}-{id_no}_d{int(dur * 1000):03d}"
                work = out / "work" / tag
                robust_copytree(src, work)
                if not a.no_sme_vbus:      # 阶段 2:机端/逐母线精确电压口径的数据源
                    buses = sorted(b.bus_no for b in
                                   cards.parse_lp1(Card.load(work / "LF.LP1")))
                    sme = work / "ST.SME"
                    if not sme.exists() or not sme.read_bytes().strip():
                        # 无 SME 交付工程(118 类):从零构造(寻址随 L5 版式二分)
                        build_sme_from_scratch(sme, work / "LF.L5")
                    add_bus_voltage_records(sme, buses)
                fault = FaultSpec(i_no, j_no, id_no, k_pct=k, ts=TS, te=TS + dur)
                t1 = time.time()
                set_s0_duration(work, a.t_end)
                ts_res = s.run_transient(work, fault, t_end=a.t_end)
                det = ts_res.verdict_detail or {}
                for name, sub, ext in (("STANA.DAT", "stana", "dat"),
                                       ("STOUT.DDD", "stout", "ddd"),
                                       ("ST.SME", "sme", "sme")):
                    p = work / name
                    if p.exists():
                        shutil.copy2(p, out / sub / f"{tag}.{ext}")
                rows.append({
                    "tag": tag,
                    "case": case_name, "i": i_no, "j": j_no, "id": id_no, "k_pct": k,
                    "ts": TS, "te": round(TS + dur, 4), "dur": dur, "t_end": a.t_end,
                    "feasible": ts_res.feasible, "stable": ts_res.stable,
                    "source": det.get("source"),
                    "instability_types": det.get("instability_types"),
                    "voltage_modes": det.get("voltage_modes"),
                    "angle_peak_deg": det.get("angle_peak_deg"),
                    "vmin_pu": det.get("vmin_pu"),
                    "vterm_min_pu": det.get("vterm_min_pu"),
                    "vterm_full": det.get("vterm_full"),
                    "recover_evaluable": det.get("recover_evaluable"),
                    "freq_source": det.get("freq_source"),
                    "v_recover_source": det.get("criteria", {}).get("v_recover_source"),
                    "criteria_version": det.get("criteria_version"),
                    "elapsed_s": round(time.time() - t1, 2), "work": str(work),
                })
                r = rows[-1]
                print(f"[{tag}] feasible={r['feasible']} stable={r['stable']} "
                      f"类型 {r['instability_types']} 角峰 {r['angle_peak_deg']}° "
                      f"机端最低 {r['vterm_min_pu']} ({r['elapsed_s']}s)", flush=True)
                if not a.keep_work:
                    shutil.rmtree(work, ignore_errors=True)

    n_stab = sum(1 for r in rows if r["feasible"] and r["stable"])
    n_unst = sum(1 for r in rows if r["feasible"] and not r["stable"])
    n_inf = sum(1 for r in rows if not r["feasible"])
    rep = {"rows": rows, "notes": {k: v["note"] for k, v in BATTERY.items()},
           "criteria_source": "D21 v1_20260712(阶段 2 精确口径:机端/逐母线电压 + COI 频率)",
           "t_end": a.t_end, "sme_vbus": not a.no_sme_vbus,
           "summary": {"n": len(rows), "stable": n_stab, "unstable": n_unst,
                       "infeasible": n_inf},
           "elapsed_s": round(time.time() - t0, 1)}
    (out / "seeds.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print(f"[seeds] {len(rows)} 例:稳 {n_stab} / 失稳 {n_unst} / infeasible {n_inf}  "
          f"用时 {rep['elapsed_s']}s")
    print(f"[seeds] {out / 'seeds.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
