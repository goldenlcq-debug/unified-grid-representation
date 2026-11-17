# -*- coding: utf-8 -*-
"""两版推理器分网架横向总表(2026-08-30 第 10 轮;用户指令"列一个总表")。

两版各取目前效果最好的一版:
  A = E13  带"后四步物理求解器"(推理器内 4 步失配修正 refine_steps=4 + refine_qv;v1 底座;
           §3.27(c) 判 7 条过 6;§3.30(d) 用户裁定"不作产品模式,只作物理上界参照")
  B = fkE_gE30k  无求解器(现生产件:fkE 三种子权重 + gate30000;fork′ 底座;定档 20260828)
对照 = PSASP(唯一裁判:精度真值 = 它,故其"误差"恒为 0;只比时间);MATPOWER/FDPF 作时间参照列。

精度口径(与 make_master_table.py 同源):误差只在 D15 类型化未知量位置统计(θ:PQ+PV;v:PQ);
θ 用折回误差;平均 = 三种子平均;最大 = 三种子最坏(上界)。
时间口径:GPU-fp32+确定性、空载、缓存热(位置编码已缓存)、单工况 = 特征 + 前向,5 次中位;
PSASP = 进程壁钟(含引擎启动 ~1.4 s + gb18030 IO),基态工况;MATPOWER = 库内 tic/toc。
**直流/152k 数据不入本表(用户 2026-08-30 指令"直流先搁置")。**
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
D = ROOT / "runs" / "diag_5060"
NAME = {"mp_case118": "IEEE-118(MATPOWER 底图)", "mp_case300": "IEEE-300",
        "mp_activsg2000": "ACTIVSg-2000", "mp_activsg10k": "ACTIVSg-10k",
        "mp_activsg25k": "ACTIVSg-25k", "mp_activsg70k": "ACTIVSg-70k",
        "lib118": "IEEE-118(老方式库)", "lib_ieee39": "IEEE-39(老方式库)", "lib_cepri36": "CEPRI-36(老方式库)"}
# E13 修正步是否被覆盖率闸放行(§3.27(c) 原表"修正步"列)
E13_REFINE = {"mp_case118": "生效", "mp_case300": "跳过", "mp_activsg2000": "生效", "mp_activsg10k": "生效",
              "mp_activsg25k": "生效", "mp_activsg70k": "生效", "lib118": "跳过", "lib_ieee39": "跳过",
              "lib_cepri36": "跳过"}
SCALE_FAM = {"mp_case300", "mp_activsg2000", "mp_activsg10k", "mp_activsg25k", "mp_activsg70k"}


def load_master(tag):
    rows = json.loads((D / f"master_table_{tag}.json").read_text(encoding="utf-8"))
    return {(r["fam"], r["status"]): r for r in rows}


def load_speed(name):
    p = D / name
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {r["fam"]: r for r in d["rows"]}


def load_psasp():
    d = json.loads((D / "speed_psasp-20260828.json").read_text(encoding="utf-8"))
    return {r["fam"]: r for r in d["rows"]}


def load_matpower():
    out = {}
    m = {"case300": "mp_case300", "case_ACTIVSg2000": "mp_activsg2000", "case_ACTIVSg10k": "mp_activsg10k",
         "case_ACTIVSg25k": "mp_activsg25k", "case_ACTIVSg70k": "mp_activsg70k"}
    for line in (D / "speed_matpower-20260828.txt").read_text(encoding="utf-8").splitlines():
        for k, v in m.items():
            if line.startswith(k + " "):
                out[v] = float(line.split("median_s=")[1].split()[0])
    return out


def deg(x):
    return math.degrees(x)


def main():
    A = load_master("e13")
    B = load_master("fke_ge30k")
    sA = load_speed("speed_scale_e13-20260830.json")
    sB = load_speed("speed_scale_fke_ge30k-20260830.json")
    sB_cpu = load_speed("speed_scale-cpu-20260828.json")      # E22 架构同 fkE_gE30k(CPU 口径,v1 底座)
    ps = load_psasp()
    mp = load_matpower()

    order = ["mp_case118", "mp_case300", "mp_activsg2000", "mp_activsg10k", "mp_activsg25k", "mp_activsg70k",
             "lib118", "lib_ieee39", "lib_cepri36"]
    keys = sorted(set(A) | set(B), key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1]))

    lines = []
    lines.append("## 表 1 精度:预测偏差的平均值 / 最大值(θ 单位 °,v 单位 pu;PSASP 为真值,其误差恒 0)\n")
    lines.append("| 网架 | 节点 | 训练状态 | 方式 | **A·E13** θ 平均 | θ 最大 | v 平均 | v 最大 | 修正步 | **B·fkE_gE30k** θ 平均 | θ 最大 | v 平均 | v 最大 |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|")
    for k in keys:
        a, b = A.get(k), B.get(k)
        fam, st = k
        n = (a or b)["n"]
        nodes = (a or b)["nodes"]

        def cell(r, f, fmt):
            return (fmt % r[f]) if r else "—"
        lines.append(
            f"| {NAME.get(fam, fam)} | {nodes} | {st} | {n} | "
            f"{(('%.3f' % deg(a['th_mean'])) if a else '—')} | {(('%.3f' % deg(a['th_max'])) if a else '—')} | "
            f"{cell(a, 'v_mean', '%.6f')} | {cell(a, 'v_max', '%.5f')} | {E13_REFINE.get(fam, '?')} | "
            f"{(('%.3f' % deg(b['th_mean'])) if b else '—')} | {(('%.3f' % deg(b['th_max'])) if b else '—')} | "
            f"{cell(b, 'v_mean', '%.6f')} | {cell(b, 'v_max', '%.5f')} |")

    lines.append("\n## 表 2 时间:单工况计算时间(秒;GPU-fp32+确定性、空载、缓存热;PSASP = 进程壁钟含启动+IO)\n")
    lines.append("| 网架 | 节点 | **A·E13**(特征+前向) | 其中前向 | **B·fkE_gE30k**(特征+前向) | 其中前向 | B 的 CPU 口径† | **PSASP** 壁钟 | PSASP 迭代 | MATPOWER 8.1 | 自研 FDPF-6‡ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for fam in ["mp_case300", "mp_activsg2000", "mp_activsg10k", "mp_activsg25k", "mp_activsg70k"]:
        a, b, c, p = sA.get(fam), sB.get(fam), sB_cpu.get(fam), ps.get(fam)
        lines.append(
            f"| {NAME[fam]} | {(a or b or c)['n']} | "
            f"{('%.3f' % a['ours_total_s']) if a else '待测'} | {('%.3f' % a['ours_forward_s']) if a else '—'} | "
            f"{('%.3f' % b['ours_total_s']) if b else '待测'} | {('%.3f' % b['ours_forward_s']) if b else '—'} | "
            f"{('%.2f' % c['ours_total_s']) if c else '—'} | "
            f"{('%.2f' % p['wall_s']) if p else '—'} | {p['iters'] if p else '—'} | "
            f"{('%.3f' % mp[fam]) if fam in mp else '—'} | {('%.3f' % b['fdpf_total_s']) if b else ('%.3f' % a['fdpf_total_s'] if a else '—')} |")

    md = "\n".join(lines)
    print(md)
    return md


if __name__ == "__main__":
    main()
