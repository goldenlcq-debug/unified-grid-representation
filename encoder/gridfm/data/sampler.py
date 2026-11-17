"""方式采样器(方案 T3.1 / 7.2)。在骨架上采样运行方式变量,经真实求解器过滤收敛样本。

v0 方式变量:负荷水平(整体缩放 0.8~1.15 + 分区扰动)、发电有功再分配(按同因子缩放,
平衡机吸收失配)、机端电压设定微扰。新能源等值(D18)为配置侧改造清单,不改卡。
不收敛样本按比例保留作可解性负类(3.4.3 / 8.5)。
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..fsutil import robust_copytree
from ..psasp_io.case import PsaspCase
from ..psasp_io.catalog import Catalog
from ..schema.assemble import build_sample
from ..schema.sample import GridSample
from ..solver.base import SolverRunner


@dataclass
class SampleSpec:
    load_factor: float
    v_jitter: float
    seed: int


# 多样方式的离散负荷档(标签):同档多方式 = 真"一对多"(2026-07-14 用户裁决④)
DIVERSE_LF_LEVELS = (0.95, 1.05, 1.15)


def sample_mode_diverse(case: PsaspCase, rng: np.random.Generator,
                        amp: float = 1.0,
                        lf_levels: tuple = DIVERSE_LF_LEVELS,
                        decommit_frac: float = 0.3) -> SampleSpec:
    """多模态方式采样(2026-07-14 用户裁决④:G1"平均化失败"演示的数据前提)。

    与 sample_mode 的本质区别:**同一标签(离散负荷档)对应多种真正不同的方式**——
    - 负荷档从 lf_levels 离散取(标签 = 档位值;逐负荷 ±3%·amp 抖动保多样不改档);
    - **开机组合**:非平衡机按 decommit_frac 概率"停机"(Pg=0,mark 不动 = 空载并网,
      不改拓扑语义;PV 母线电压支撑保留,v0 简化如实入档);
    - **出力分配**:目标总出力 = 档位 × 基态总出力,在开机机组间按 Dirichlet(α=1)
      随机参与度分配,[Pmin,Pmax] 截断后残差按剩余裕度轮次消化(消化不完留给平衡机);
    - 机端电压 ±0.03·amp(比基础采样器 ±0.02 略宽,画像多样性)。

    同档两次采样的逐机出力向量天差地别(单测锁定)——确定性回归对同档条件均值化,
    输出"哪个模式都不是"的平均调度,这才是 6.6 第一档要演示的对象。
    不动 sample_mode(同 seed 复现逐位不变)。
    """
    lf = float(lf_levels[int(rng.integers(0, len(lf_levels)))])
    for ld in case.loads():
        j = float(rng.uniform(1.0 - 0.03 * amp, 1.0 + 0.03 * amp))
        ld.pl = ld.pl * lf * j
        ld.ql = ld.ql * lf * j

    gens = [g for g in case.gens() if g.type_code != 0]
    base_total = sum(g.pg for g in gens)
    target = base_total * lf
    on = [g for g in gens if rng.random() >= decommit_frac]
    if not on:                                   # 全停保护:至少留一台
        on = [gens[int(rng.integers(0, len(gens)))]]
    for g in gens:
        if g not in on:
            g.pg = 0.0
    w = rng.dirichlet(np.ones(len(on)))
    alloc = w * target
    for g, p in zip(on, alloc):
        g.pg = float(p)
    # [Pmin,Pmax] 截断 + 残差按剩余裕度轮次消化(≤4 轮,消化不完平衡机吸收)
    for _ in range(4):
        resid = target - sum(min(max(g.pg, g.pmin), g.pmax if g.pmax > 0 else g.pg)
                             for g in on)
        for g in on:
            g.pg = min(max(g.pg, g.pmin), g.pmax if g.pmax > 0 else g.pg)
        if abs(resid) < 1e-6:
            break
        room = [max((g.pmax if g.pmax > 0 else g.pg + abs(resid)) - g.pg, 0.0)
                if resid > 0 else max(g.pg - g.pmin, 0.0) for g in on]
        tot = sum(room)
        if tot <= 1e-9:
            break
        for g, r in zip(on, room):
            g.pg = g.pg + resid * (r / tot)

    v_jit = 0.0
    for g in gens:
        dv = float(rng.uniform(-0.03 * amp, 0.03 * amp))
        g.v0 = g.v0 + dv
        v_jit = max(v_jit, abs(dv))
    return SampleSpec(load_factor=lf, v_jitter=v_jit, seed=int(rng.integers(0, 2**31)))


def sample_mode(case: PsaspCase, rng: np.random.Generator, amp: float = 1.0) -> SampleSpec:
    """在 case(已加载)上原地施加一组方式变量。返回本次采样规格(供标签/复现)。

    amp = 采样幅度系数(2026-07-11 用户批复:大网按规模收紧,防收敛率崩塌吃掉批产预算;
    实测方式收敛率 86%@150 → 59%@700)。amp=1.0 时各区间与旧实现完全相同(逐位兼容);
    amp<1 等比收窄:负荷整体缩放 U(1−0.2a, 1+0.15a)、逐负荷扰动 ±5%a、机端电压 ±0.02a。
    """
    lf = float(rng.uniform(1.0 - 0.20 * amp, 1.0 + 0.15 * amp))
    # 负荷:整体缩放 + 每负荷扰动
    for ld in case.loads():
        j = float(rng.uniform(1.0 - 0.05 * amp, 1.0 + 0.05 * amp))
        ld.pl = ld.pl * lf * j
        ld.ql = ld.ql * lf * j
    # 发电:非平衡机 Pg 按同因子缩放,截断于 [Pmin, Pmax];机端电压微扰
    v_jit = 0.0
    for g in case.gens():
        if g.type_code == 0:      # 平衡机不设 Pg(由潮流决定)
            continue
        newp = g.pg * lf
        if g.pmax > 0:
            newp = min(newp, g.pmax)
        newp = max(newp, g.pmin)
        g.pg = newp
        dv = float(rng.uniform(-0.02 * amp, 0.02 * amp))
        g.v0 = g.v0 + dv
        v_jit = max(v_jit, abs(dv))
    return SampleSpec(load_factor=lf, v_jitter=v_jit, seed=int(rng.integers(0, 2**31)))


def generate_library(base_case_dir: Path, solver: SolverRunner, n: int, out_dir: Path,
                     work_dir: Path, seed: int = 0, family_id: str = "ieee118",
                     keep_divergent_frac: float = 0.08,
                     max_attempts: Optional[int] = None,
                     v_floor: Optional[float] = 0.85,
                     amp: float = 1.0,
                     keep_cases: bool = False,
                     id_start: int = 0,
                     diverse: bool = False) -> dict:
    """生成方式库,直到收集到 **n 个收敛样本**(或尝试次数超 max_attempts,默认 3n)。

    实测教训(2026-07-09):旧实现只跑固定 n 次迭代,发散即丢——合成 500 节点网的方式
    发散率约 20%,库大小随规模缩水,家族间样本数不齐。改为"目标收敛数"语义。

    收敛样本正常入库;按 keep_divergent_frac 比例额外保留不收敛样本作负类(状态量置
    missing、可解性=0),负类不计入 n。keep_divergent_frac=0 时纯收敛库(冻结测试集用)。

    **电压带验收(2026-07-11 用户批准,阈值 0.85)**:收敛但存在有效母线 v < v_floor 的
    样本(合成器病态弱电压口袋 / 失电孤岛 v=0)**不入主库、不计入 n**,封存到
    `out_dir/_lowv_quarantine/`(带独立 manifest,留档供困难样本集/热启动窄门,7.2)。
    v_floor=None 关闭验收(既有库复现用)。现有冻结集不受影响(本函数只管新生产)。

    **keep_cases(2026-07-11,T4.2 前置)**:入主库的每个样本把**已解算例目录**整目录
    留存到 `out_dir/cases/<sample_id>/`(含 LF.* 输入卡与 LP*/CAL 结果卡)——T4.2 批量
    判稳需要每方式原始算例文件(交接文档实现路径 2);不影响 RNG 消耗序列,同 seed 复现
    的 .npz 与不开 keep_cases 逐位一致。低压封存/丢弃样本不留存。

    **id_start(2026-07-14,外协包 C 号段隔离护栏)**:sample_id 序号 = id_start + 尝试序
    (`{family_id}_{id_start+i:05d}`)。外协扩产按任务书指派号段(如包 C = 20000 起),
    与主线 0 段物理不相交;默认 0 与旧命名逐位一致,**不消耗 RNG**,同 seed 复现不受影响。
    返回 {n_converged, n_divergent_kept, n_lowv_quarantined, n_total, n_attempts, out_dir}。
    """
    base_case_dir = Path(base_case_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_dir)
    rng = np.random.default_rng(seed)
    cap = max_attempts if max_attempts is not None else max(3 * n, n + 10)

    n_conv = 0
    n_div_kept = 0
    n_lowv = 0
    attempts = 0
    manifest = []
    lowv_manifest = []
    lowv_dir = out_dir / "_lowv_quarantine"
    while n_conv < n and attempts < cap:
        i = attempts
        attempts += 1
        # 独立工作目录(单库内串行;并行只在家族级,7.0 纪律 7)
        # 家族级并行下 Windows 句柄异步释放会让 rmtree/copytree 偶发瞬态失败
        # (07-09 批产丢家族)→ 重试兜底,耗尽仍失败即抛(不静默跳过)
        robust_copytree(base_case_dir, work_dir)
        case = PsaspCase(work_dir)
        # diverse = 多模态方式(裁决④,离散负荷档 × 随机开机/参与度);默认走基础采样器
        spec = (sample_mode_diverse if diverse else sample_mode)(case, rng, amp=amp)
        case.save_inplace()

        pf = solver.run_powerflow(work_dir)
        converged = pf.converged
        keep = converged or (rng.random() < keep_divergent_frac)
        if not keep:
            continue

        # 重新加载已解算例(结果卡已更新)装配档案
        solved = PsaspCase(work_dir)
        cat = Catalog(solved)
        sid = f"{family_id}_{id_start + i:05d}"
        gs = build_sample(solved, cat, sample_id=sid, family_id=family_id)
        if not converged:
            # 负类:状态量置 missing、可解性标签=0(装配已按无结果处理;此处显式标注)
            gs.meta["solvable"] = 0
        else:
            gs.meta["solvable"] = 1
        gs.meta["load_factor"] = spec.load_factor

        # 电压带验收(只对收敛样本;发散负类状态本就 missing,不适用)
        if converged and v_floor is not None:
            vmin = _min_valid_voltage(gs)
            if np.isfinite(vmin) and vmin < v_floor:
                lowv_dir.mkdir(parents=True, exist_ok=True)
                gs.save_npz(lowv_dir / f"{sid}.npz")
                lowv_manifest.append({"sample_id": sid, "family_id": family_id,
                                      "solvable": 1, "load_factor": spec.load_factor,
                                      "min_v_pu": float(vmin)})
                n_lowv += 1
                continue                      # 封存留档,不入主库、不计入 n

        gs.save_npz(out_dir / f"{sid}.npz")
        if keep_cases:
            robust_copytree(work_dir, out_dir / "cases" / sid)
        manifest.append({"sample_id": sid, "family_id": family_id,
                         "solvable": gs.meta["solvable"], "load_factor": spec.load_factor})
        if converged:
            n_conv += 1
        else:
            n_div_kept += 1

    _write_manifest(out_dir / "manifest.csv", manifest)
    if lowv_manifest:
        _write_manifest(lowv_dir / "manifest.csv", lowv_manifest)
    return {"n_converged": n_conv, "n_divergent_kept": n_div_kept,
            "n_lowv_quarantined": n_lowv,
            "n_total": len(manifest), "n_attempts": attempts, "out_dir": str(out_dir)}


def _min_valid_voltage(gs) -> float:
    """样本内有效母线(非 NULL、状态非 missing)的最低电压(pu)。无有效母线返回 nan。"""
    from ..schema.fields import NODE_TABLE, Group
    from ..schema.sample import group_bit_index
    off_v = NODE_TABLE.offset_of("v")
    off_null = NODE_TABLE.offset_of("is_null")
    sbit = group_bit_index(Group.STATE)
    ok = (~gs.node_missing[:, sbit]) & (gs.node_feat[:, off_null] < 0.5)
    if not ok.any():
        return float("nan")
    return float(gs.node_feat[ok, off_v].min())


def _write_manifest(path: Path, rows: List[dict]) -> None:
    import csv
    if not rows:
        path.write_text("sample_id,family_id,solvable,load_factor\n", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
