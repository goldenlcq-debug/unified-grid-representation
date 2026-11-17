"""统计-几何合成电网生成器(方案 7.2"进阶:统计合成方法"的 v0 实现)。

物理含义的三层保障:
1. 拓扑统计:母线为平面地理散点;最小生成树保连通(短边优先=地理局部性),
   再按边长从短到长补 Delaunay 边至目标平均度(~2.8-3.2)——复现真实电网
   "稀疏、网状、局部连接、少量枢纽"的结构特征(RT-nestedSmallWorld/地理法思路)。
2. 电气参数:全部从真实施主算例(118)的记录克隆/分布采样——线路 X 按几何边长
   缩放施主分布(长线抗大),R 按施主 X/R 比,B 随长度;发电机/负荷整条克隆真实
   记录(限值/电压设定/功率因数天然真实),仅改挂接与出力。
3. 运行点:负载率对齐施主(~65-75%),出力按容量分摊 + 单一平衡机;
   **真实 PSASP 牛拉收敛闭环校验,发散自动降载重试**——不收敛不出厂。

v0 限定:单电压等级(施主 118 即如此)、无变压器/动态参数(潮流侧算例);
分层电压等级与动态配置是后续升级项。
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

from ..fsutil import robust_copytree
from ..psasp_io.cards import GenView, _find_idname_index
from ..psasp_io.lowlevel import Card, Record
from .repair import (
    DEFAULT_REPAIR_CONFIG,
    RepairAction,
    RepairObservation,
    RepairTrace,
    RuleRepairConfig,
)


def _dc_solve(ei: np.ndarray, ej: np.ndarray, xs: np.ndarray, n: int,
              keep: np.ndarray, p_inj: np.ndarray) -> np.ndarray:
    """直流潮流 B'θ = P 的稀疏求解(消去平衡机行列)。返回 theta[keep]。

    B' 装配:每条支路 w=1/x 贡献 (i,i)+w (j,j)+w (i,j)-w (j,i)-w;重边(并联回路)由
    coo→csr 的 sum_duplicates 天然叠加,与稠密 `Bm[i,j] -= w` 累加口径逐位一致。
    """
    w = 1.0 / xs
    rows = np.concatenate([ei, ej, ei, ej])
    cols = np.concatenate([ei, ej, ej, ei])
    vals = np.concatenate([w, w, -w, -w])
    B = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    Bk = B[keep][:, keep].tocsc()
    theta = spsolve(Bk, p_inj[keep])
    if not np.all(np.isfinite(theta)):
        raise np.linalg.LinAlgError("直流潮流 B' 奇异(网架不连通?)")
    return theta


def _dc_solve_dense_legacy(
    edges, xs: np.ndarray, n: int, keep: np.ndarray, p_inj: np.ndarray
) -> np.ndarray:
    """2026-07-09 9/9 历史回归使用的稠密 B' 实现；只供固定历史复现。"""
    matrix = np.zeros((n, n))
    for k, (i, j) in enumerate(edges):
        weight = 1.0 / xs[k]
        matrix[i, i] += weight
        matrix[j, j] += weight
        matrix[i, j] -= weight
        matrix[j, i] -= weight
    return np.linalg.solve(matrix[np.ix_(keep, keep)], p_inj[keep])


def _append(card: Card, recs, ending: str = "\r\n") -> None:
    for r in recs:
        card.elements.append(r)
        card.endings.append(ending)


# ---------------- 拓扑:地理散点 + MST + Delaunay 补边 ----------------

def synth_topology(n: int, avg_degree: float, rng: np.random.Generator):
    """返回 (pts[n,2], edges[(i,j)] 0-based, lengths)。MST⊂Delaunay 保证连通+局部。"""
    pts = rng.random((n, 2))
    tri = Delaunay(pts)
    dedges = set()
    for simplex in tri.simplices:
        for a in range(3):
            i, j = int(simplex[a]), int(simplex[(a + 1) % 3])
            dedges.add((min(i, j), max(i, j)))
    dedges = sorted(dedges)
    lengths = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in dedges])

    # MST(在 Delaunay 边集上)= 骨架边(承担大范围功率转移 → 配强线)
    ii = np.array([e[0] for e in dedges]); jj = np.array([e[1] for e in dedges])
    mst = minimum_spanning_tree(coo_matrix((lengths, (ii, jj)), shape=(n, n)))
    mst_edges = {(min(a, b), max(a, b)) for a, b in zip(*mst.nonzero())}

    # 按边长从短到长补边至目标平均度
    target_m = int(round(avg_degree * n / 2))
    chosen = set(mst_edges)
    order = np.argsort(lengths)
    for k in order:
        if len(chosen) >= target_m:
            break
        chosen.add(dedges[k])
    edges = sorted(chosen)
    elens = np.array([np.linalg.norm(pts[i] - pts[j]) for i, j in edges])
    is_backbone = np.array([e in mst_edges for e in edges])
    return pts, edges, elens, is_backbone


# ---------------- 主生成器 ----------------

def synth_case(n_buses: int, donor_dir: Path, out_dir: Path, seed: int = 0,
               avg_degree: float = 3.0, load_level: float = 0.60,
               gen_frac: float = 0.24, load_frac: float = 0.75,
               q_comp: float = 0.5, dc_backend: str = "sparse") -> dict:
    """q_comp:负荷无功就地补偿系数(净 Q = 施主 Q × q_comp;Q 运不远,
    真实电网在负荷侧就地补偿——PV 支撑稀疏的合成网必须体现这一点,实测教训)。"""
    if dc_backend not in {"sparse", "dense_legacy"}:
        raise ValueError(f"未知 dc_backend:{dc_backend}")
    donor_dir, out_dir = Path(donor_dir), Path(out_dir)
    rng = np.random.default_rng(seed)
    robust_copytree(donor_dir, out_dir)     # 家族级并行:Windows 句柄竞态重试(fsutil)
    for f in ("LF.CAL", "LF.LP1", "LF.LP2", "LF.LP3", "LF.LP5", "LF.LP6", "LF.P1D"):
        (out_dir / f).unlink(missing_ok=True)

    # ---- 施主统计 ----
    dl1 = Card.load(donor_dir / "LF.L1"); dl2 = Card.load(donor_dir / "LF.L2")
    dl5 = Card.load(donor_dir / "LF.L5"); dl6 = Card.load(donor_dir / "LF.L6")
    dbus = Card.load(donor_dir / "LF.BUS"); dphy = Card.load(donor_dir / "PHYBUS.LFI")
    d_lines = [r for r in dl2.records
               if r.get_int(0) == 1 and r.get_str(_find_idname_index(r)).endswith("_ac")]
    dX = np.array([r.get_float(5) for r in d_lines])
    dXR = np.array([r.get_float(5) / max(r.get_float(4), 1e-6) for r in d_lines])
    dB = np.array([r.get_float(6) for r in d_lines])
    d_gens = list(dl5.records)
    d_slack = next(r for r in d_gens if r.get_int(2) == 0)
    d_pv = [r for r in d_gens if r.get_int(2) != 0]
    d_loads = list(dl6.records)
    tmpl_bus = next(r for r in dl1.records if not r.get_str(0).upper().startswith("NULL"))

    # ---- 拓扑 ----
    pts, edges, elens, is_backbone = synth_topology(n_buses, avg_degree, rng)
    mean_len = elens.mean()

    # ---- 母线卡(L1/BUS/PHYBUS 全新生成,克隆模板保格式)----
    def new_l1(i):
        r = Record(list(tmpl_bus.cells))
        r.set_str(0, f"S-bus-{i+1}"); r.set_str(7, f"S-bus-{i+1}")
        return r
    l1 = Card.load(out_dir / "LF.L1"); l1.elements = [new_l1(i) for i in range(n_buses)]
    l1.endings = ["\r\n"] * n_buses
    tb = Record(list(dbus.records[0].cells))
    lbus = Card.load(out_dir / "LF.BUS")
    lbus.elements = []; lbus.endings = []
    tp = Record(list(dphy.records[0].cells))
    phy = Card.load(out_dir / "PHYBUS.LFI")
    phy.elements = []; phy.endings = []
    for i in range(n_buses):
        rb = Record(list(tb.cells)); rb.set_str(0, f"S-bus-{i+1}")
        lbus.elements.append(rb); lbus.endings.append("\r\n")
        rp = Record(list(tp.cells)); rp.set_str(1, f"S-bus-{i+1}"); rp.set_int(2, i + 1)
        phy.elements.append(rp); phy.endings.append("\r\n")

    # ---- 线路参数:单位长度线路常数(物理第一性)----
    #      X = x/km × 边长 × 轻抖动(校准到施主中位数),B ∝ X(同一物理量纲)。
    #      实测教训:对 X、B 独立双重随机会造出"长弱桥/短强边配巨充电"怪胎,
    #      织物无主干且电压贴边;按公里数走则长边自然弱(物理),强弱有序。
    x_hi = float(np.percentile(dX, 95))              # X 上限(防长弱线压垮电压)
    b_hi = float(np.percentile(dB, 90))              # B 上限(防充电无功盈余顶爆电压)
    x_per_len = float(np.median(dX)) / float(np.median(elens))
    b_per_x = float(np.median(dB)) / float(np.median(dX))
    m = len(edges)
    xs = np.empty(m); rs = np.empty(m); bs = np.empty(m)
    for k in range(m):
        jitter = float(np.exp(rng.normal(0.0, 0.25)))
        x = float(np.clip(x_per_len * elens[k] * jitter, 5e-4, x_hi))
        xr = float(rng.choice(dXR))
        xs[k] = x; rs[k] = x / max(xr, 1.0)
        bs[k] = float(np.clip(b_per_x * x, 0.0, b_hi))

    # ---- 机组:克隆真实记录(限值/设定真实),按度数优先落枢纽母线 ----
    deg = np.zeros(n_buses)
    for i, j in edges:
        deg[i] += 1; deg[j] += 1
    n_gen = max(3, int(round(gen_frac * n_buses)))
    # PV 布点 = 枢纽一半(大机组在枢纽)+ 最远点采样一半(电压支撑必须覆盖全网,
    # Q 运不远,纯枢纽布点会让大片 PQ 区域电压塌陷——NR 白盒实测教训)。
    # 真实规划约束:电厂至少两回送出线(N-1),布机候选一律 度≥2(实测:度1叶子挂
    # 3pu 机组会让 PSASP 牛拉在该点过冲发散)。
    eligible = set(np.where(deg >= 2)[0].tolist())
    n_hub = n_gen // 2
    hubs = [b for b in np.argsort(-deg).tolist() if b in eligible][:n_hub]
    rest = [b for b in sorted(eligible) if b not in set(hubs)]
    spread: List[int] = []
    cand = pts[rest]
    picked = pts[hubs] if hubs else pts[:1]
    for _ in range(min(n_gen - n_hub, len(rest))):
        dmin = np.min(np.linalg.norm(cand[:, None, :] - picked[None, :, :], axis=2), axis=1)
        k = int(np.argmax(dmin))
        spread.append(rest[k])
        picked = np.vstack([picked, cand[k]])
        cand = np.delete(cand, k, axis=0); rest = rest[:k] + rest[k + 1:]
    gen_buses = hubs + spread
    picks = [d_slack] + [d_pv[int(rng.integers(0, len(d_pv)))] for _ in range(len(gen_buses) - 1)]
    # 大机组配大枢纽(真实规律):机组按容量降序 × 母线按度降序 配对;平衡机=最大机落最大枢纽
    picks.sort(key=lambda r: -r.get_float(9))
    gen_buses.sort(key=lambda b: -deg[b])
    n_gen = len(gen_buses)
    gens: List[Record] = []
    for gi, (b, src) in enumerate(zip(gen_buses, picks)):
        r = Record(list(src.cells))
        r.set_int(0, 1); r.set_int(1, int(b) + 1)
        # 初值归平!克隆记录的 V0/θ 是施主自己那张网的解值,搬到新拓扑上是错乱初值:
        # 实测 30° 初始角差过 X=0.04 强线 = 12 pu 初始失配,牛拉必发散。
        r.set_float(6, 0.0)                                  # θ 初值 = 0
        # V0 在写线路卡后按"母线最强邻线刚度"统一回填(见下),此处先占位
        r.set_float(5, 1.000)
        r.set_str(_find_idname_index(r), f"S-gen-{gi+1}")
        gens.append(r)
    cap = np.array([r.get_float(9) for r in gens])       # Pmax

    # ---- 负荷:克隆真实记录,缩放到目标负载率 ----
    n_load = max(3, int(round(load_frac * n_buses)))
    load_buses = rng.choice(n_buses, size=n_load, replace=False)
    # 单负荷限幅:剔除施主分布尾部大负荷(防大负荷落在弱支线末端压垮局部电压)
    p_cap = float(np.percentile([r.get_float(4) for r in d_loads], 85))
    d_loads_capped = [r for r in d_loads if r.get_float(4) <= p_cap] or d_loads
    lsrc = [d_loads_capped[int(rng.integers(0, len(d_loads_capped)))] for _ in range(n_load)]
    raw_p = np.array([r.get_float(4) for r in lsrc])
    target_total = load_level * cap.sum()
    lscale = target_total / max(raw_p.sum(), 1e-6)
    l6 = Card.load(out_dir / "LF.L6"); l6.elements = []; l6.endings = []
    for li, (b, src) in enumerate(zip(load_buses, lsrc)):
        r = Record(list(src.cells))
        r.set_int(0, 1); r.set_int(1, int(b) + 1); r.set_int(2, 0)
        r.set_int(3, 1)      # 强制 PQ 型!施主部分负荷 Type=-1(定电压型),克隆到随机母线
        #                      会与 PV 机组电压设定冲突 → 雅可比病态发散(实测教训)
        r.set_float(4, src.get_float(4) * lscale)
        r.set_float(5, src.get_float(5) * lscale * q_comp)  # 净 Q = 施主 Q × 就地补偿系数
        r.set_float(6, 1.0); r.set_float(7, 0.0)          # V/θ 初值归平(同上教训)
        r.set_str(_find_idname_index(r), f"S-load-{li+1}")
        l6.elements.append(r); l6.endings.append("\r\n")

    # ---- 调度:Voronoi 分区自平衡(真实电网"分区平衡"直接建模)----
    # 每个负荷归属最近机组的地理分区,机组出力 = 本分区负荷 × 1.03(留网损);
    # 越限部分按剩余容量再分配。跨区大转移由构造消灭 → 走廊压力最小。
    # (实测教训:纯容量比例造成大范围转移,N=1000 加固需求爆炸;
    #  "3跳本地负荷加权"因多机共享邻域双计数,反而更糟,均已撤回)
    disp = target_total * 1.03
    gen_bus_idx = np.array([r.get_int(1) - 1 for r in gens])
    load_at = np.zeros(n_buses)
    for r in l6.records:
        load_at[r.get_int(1) - 1] += r.get_float(4)
    # 逐负荷母线找最近机组(欧氏,地理散点),区负荷求和
    d2 = np.linalg.norm(pts[:, None, :] - pts[gen_bus_idx][None, :, :], axis=2)  # [n,ngen]
    nearest = np.argmin(d2, axis=1)
    region_load = np.zeros(len(gens))
    for b in range(n_buses):
        if load_at[b] > 0:
            region_load[nearest[b]] += load_at[b]
    # 一轮分配 + 越限盈余按剩余容量再分配(平衡机不参与定出力)
    pg_arr = np.zeros(len(gens))
    is_pv = np.array([r.get_int(2) != 0 for r in gens])
    want = region_load * 1.03
    pg_arr[is_pv] = np.minimum(want[is_pv], cap[is_pv])
    surplus = float(want[is_pv].sum() - pg_arr[is_pv].sum())
    headroom = np.where(is_pv, cap - pg_arr, 0.0)
    if surplus > 1e-9 and headroom.sum() > 1e-9:
        pg_arr += headroom * min(1.0, surplus / headroom.sum())
    for r, pg, pv_flag in zip(gens, pg_arr, is_pv):
        if pv_flag:
            r.set_float(3, float(np.clip(pg, r.get_float(10), None)))
    l5 = Card.load(out_dir / "LF.L5"); l5.elements = list(gens)
    l5.endings = ["\r\n"] * len(gens)

    # ---- 网架规划迭代(物理关键步!):直流潮流校核线路角差,超稳定裕度的走廊
    #      并联加固(X/2、R/2、B×2 = 增加一回同型线路;真实电网大电厂配强送出通道)。
    #      独立采样 X 忽略"线路强度∝输送功率"的规律,实测大机组灌弱节点必崩(NR 白盒诊断)。
    p_inj = np.zeros(n_buses)
    slack_bus = None
    for r in gens:
        if r.get_int(2) == 0:
            slack_bus = r.get_int(1) - 1
        else:
            p_inj[r.get_int(1) - 1] += r.get_float(3)
    for r in l6.records:
        p_inj[r.get_int(1) - 1] -= r.get_float(4)
    p_inj[slack_bus] = -(p_inj.sum() - p_inj[slack_bus])   # 直流无损,平衡机兜差

    DELTA_MAX = 0.10                                        # 单线角差规划裕度(rad)。
    # 0.3 rad 实测教训:DC 可行但线上无功损耗 θ²/x 高达数 pu,AC 电压塌;
    # 真实电网单线角差通常仅几度,收紧到 0.10 使加固环建出真实的多回走廊。
    n_reinforced = 0
    halvings = np.zeros(m, dtype=np.int64)   # 每线加固轮数封顶 3(=8 回等效)。实测教训:
    # 无封顶时热点走廊被砍到 X/32,PV 设定差经近零阻抗在平启动产生 ~33pu 无功涌流,牛拉过冲死
    MAX_HALVINGS = 5   # x/32 下限;配合 PV 设定统一 1.00(平启动零涌流)刚性风险已消
    # 稀疏化(2026-07-09):此前每轮构造 N×N 稠密 Bm 并 np.linalg.solve,
    # N=10k 时 800MB/O(N³) 直接不可行;改为 coo→csr 装配 + spsolve(B 为电网 B' 矩阵,极稀疏)。
    ei_arr = np.array([e[0] for e in edges], dtype=np.int64)
    ej_arr = np.array([e[1] for e in edges], dtype=np.int64)
    keep = np.array([b for b in range(n_buses) if b != slack_bus], dtype=np.int64)
    for _round in range(12):
        theta = np.zeros(n_buses)
        if dc_backend == "dense_legacy":
            theta[keep] = _dc_solve_dense_legacy(
                edges, xs, n_buses, keep, p_inj)
        else:
            theta[keep] = _dc_solve(
                ei_arr, ej_arr, xs, n_buses, keep, p_inj)
        dth = np.abs(theta[ei_arr] - theta[ej_arr])
        viol = (dth > DELTA_MAX) & (halvings < MAX_HALVINGS)
        if not viol.any():
            break
        xs[viol] *= 0.5; rs[viol] *= 0.5                    # 并联同型一回
        # B 保持不变 = 每回配套等容高抗全补偿充电(真实 EHV 多回走廊标准做法;
        # 实测教训:B×2 随多轮加固指数增长 → N=1000 充电爆表,过压 1.14、Q 失配 25pu)
        halvings[viol] += 1
        n_reinforced += int(viol.sum())

    # ---- 写线路卡(加固后参数)----
    tmpl_line = d_lines[0]
    l2 = Card.load(out_dir / "LF.L2"); l2.elements = []; l2.endings = []
    for k, (i, j) in enumerate(edges):
        r = Record(list(tmpl_line.cells))
        r.set_int(0, 1); r.set_int(1, i + 1); r.set_int(2, j + 1); r.set_int(3, 100 + k)
        r.set_float(4, float(rs[k])); r.set_float(5, float(xs[k]))
        r.set_float(6, float(bs[k]))
        r.set_str(_find_idname_index(r), f"SL{k}_ac")
        l2.elements.append(r); l2.endings.append("\r\n")

    # ---- PV 设定按邻线刚度分配(两种失效模式分治,均为实测教训):
    #      强化走廊旁(min 邻线 x<0.006):V0=1.00 —— 平启动 ΔV/x 无功涌流为零;
    #      其余机组:V0=1.03 —— 撑起大网电压面貌(全 1.00 时 500+ 节点电压下垂塌陷)。
    min_x_at = np.full(n_buses, np.inf)
    for k, (i, j) in enumerate(edges):
        min_x_at[i] = min(min_x_at[i], xs[k]); min_x_at[j] = min(min_x_at[j], xs[k])
    for r in gens:
        b = r.get_int(1) - 1
        stiff = min_x_at[b] < 0.006
        if r.get_int(2) == 0:
            r.set_float(5, 1.010 if stiff else 1.035)
        else:
            r.set_float(5, 1.000 if stiff else 1.030)

    # ---- L3 置空(v0 无变压器)+ L0 计数 ----
    l3 = Card.load(out_dir / "LF.L3"); l3.elements = []; l3.endings = []
    l0 = Card.load(out_dir / "LF.L0")
    h = l0.records[0]
    h.set_int(0, n_buses); h.set_int(1, len(edges)); h.set_int(2, 0)
    h.set_int(4, n_gen); h.set_int(5, n_load)

    for card, name in ((l1, "LF.L1"), (l2, "LF.L2"), (l3, "LF.L3"), (l5, "LF.L5"),
                       (l6, "LF.L6"), (lbus, "LF.BUS"), (phy, "PHYBUS.LFI"), (l0, "LF.L0")):
        card.save(out_dir / name)

    return {"n_buses": n_buses, "n_lines": len(edges), "n_gens": n_gen, "n_loads": n_load,
            "avg_degree": 2 * len(edges) / n_buses, "n_reinforced": n_reinforced,
            "out_dir": str(out_dir)}


def psasp_referee(out_dir: Path, solver) -> dict:
    """唯一裁判 = PSASP(本项目全部算例以 PSASP 为准,不引入其他仿真平台)。

    返回 {converged, vmin, vmax, v(1-based 母线号索引到 0-based 数组)}。
    发散时不给电压画像(LF.LP1 可能含溢出星号),由 nrpf 白盒诊断补位(仅诊断,无裁决权)。
    """
    import numpy as _np
    pf = solver.run_powerflow(Path(out_dir))
    if not pf.converged or not pf.bus_voltage:
        return {"converged": False}
    n = max(r.bus_no for r in pf.bus_voltage)
    v = _np.zeros(n)
    for r in pf.bus_voltage:
        v[r.bus_no - 1] = r.v_pu
    vv = v[v > 0.01]
    return {"converged": True, "vmin": float(vv.min()), "vmax": float(vv.max()), "v": v}


def _trace_psasp(trace: Optional[RepairTrace], stage: str, round_index: int,
                 result: dict) -> None:
    if trace is None:
        return
    trace.add_observation(RepairObservation(
        stage=stage,
        round_index=round_index,
        source="psasp",
        converged=bool(result["converged"]),
        vmin=result.get("vmin"),
        vmax=result.get("vmax"),
        voltage=result.get("v"),
    ))


def _trace_nr(trace: Optional[RepairTrace], stage: str, round_index: int,
              result) -> None:
    if trace is None:
        return
    diagnosis = dict(getattr(result, "diag", {}))
    diagnosis.update(
        iters=getattr(result, "iters", None),
        max_mismatch=getattr(result, "max_mismatch", None),
        theta=getattr(result, "theta", None),
    )
    trace.add_observation(RepairObservation(
        stage=stage,
        round_index=round_index,
        source="nrpf",
        converged=bool(getattr(result, "converged", False)),
        voltage=getattr(result, "v", None),
        diagnosis=diagnosis,
    ))


def _ac_reactive_planning(out_dir: Path, solver, max_rounds: Optional[int] = None,
                          protected_buses=(),
                          config: RuleRepairConfig = DEFAULT_REPAIR_CONFIG,
                          trace: Optional[RepairTrace] = None,
                          stage: str = "q_primary") -> dict:
    """AC 无功规划修复环(双向,真实规划做法):
    低压母线 → 就地补偿(加并联电容);过压母线 → 装电抗器吸收(大网充电盈余——实测教训)。
    **裁判 = PSASP**(D16:唯一平台);PSASP 发散时用 nrpf 白盒拿病灶电压画像(仅诊断)。
    只修改已有 LF.L6 行；无行母线记入 missing_comp_buses，不静默新建记录。"""
    from ..psasp_io.case import PsaspCase
    max_rounds = config.q_max_rounds if max_rounds is None else max_rounds
    protected = frozenset(config.with_protected(protected_buses).protected_buses)
    missing_all = set()
    protected_skipped = set()
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        case = PsaspCase(out_dir)
        rf = psasp_referee(out_dir, solver)
        _trace_psasp(trace, stage, rounds - 1, rf)
        if rf["converged"]:
            if (rf["vmin"] >= config.voltage_target and
                    rf["vmax"] <= config.voltage_max_soft):
                return {"nr_ok": True, "q_rounds": rounds - 1,
                        "missing_comp_buses": sorted(missing_all),
                        "protected_skipped": sorted(protected_skipped)}
            vvec = rf["v"]
        else:
            from ..eval.nrpf import solve_nr
            nr = solve_nr(case, max_iter=40)
            _trace_nr(trace, stage, rounds - 1, nr)
            vvec = nr.v   # 白盒诊断(无裁决权)
        # 比例控制器(实测教训:定步长"加电抗/砍电容"会双向对打振荡):
        # 每母线修正量 dQ = gain × 越限量,双向作用于同一补偿记录,逐轮收敛
        adj = {}
        for b in range(len(vvec)):
            bus = b + 1
            if bus in protected:
                if (vvec[b] > config.voltage_high_trigger or
                        config.voltage_island < vvec[b] < config.voltage_low_trigger):
                    protected_skipped.add(bus)
                    if trace is not None:
                        trace.add_action(RepairAction(
                            action_type="adjust_q",
                            stage=stage,
                            round_index=rounds - 1,
                            object_kind="bus",
                            object_id=bus,
                            field="LF.L6.QL",
                            before=None,
                            after=None,
                            delta=None,
                            trigger=(
                                "over_voltage" if vvec[b] > config.voltage_high_trigger
                                else "under_voltage"),
                            status="skipped_protected",
                            metadata={"voltage": float(vvec[b])},
                        ))
                continue
            if vvec[b] > config.voltage_high_trigger:
                adj[bus] = float(np.clip(
                    config.q_gain * (vvec[b] - config.voltage_high_trigger),
                    0.0, config.q_step_cap))    # 加感(电抗器)
            elif config.voltage_island < vvec[b] < config.voltage_low_trigger:
                adj[bus] = float(np.clip(
                    -config.q_gain * (config.voltage_low_trigger - vvec[b]),
                    -config.q_step_cap, 0.0))  # 加容(电容)
        l6 = Card.load(out_dir / "LF.L6")
        touched = set()
        for r in l6.records:
            b = r.get_int(1)
            if b in adj and b not in touched:      # 每母线只调一条记录(补偿设备)
                before = r.get_float(5)
                after = before + adj[b]
                r.set_float(5, after)
                touched.add(b)
                if trace is not None:
                    trace.add_action(RepairAction(
                        action_type="adjust_q",
                        stage=stage,
                        round_index=rounds - 1,
                        object_kind="bus",
                        object_id=b,
                        field="LF.L6.QL",
                        before=before,
                        after=after,
                        delta=adj[b],
                        trigger=("over_voltage" if adj[b] > 0 else "under_voltage"),
                        metadata={"voltage": float(vvec[b - 1])},
                    ))
        missing_comp = [b for b in adj if b not in touched]
        if config.allow_new_comp_rows and missing_comp and l6.records:
            tmpl = l6.records[0]
            for b in missing_comp:
                r = Record(list(tmpl.cells))
                r.set_int(0, 1)
                r.set_int(1, b)
                r.set_int(2, 90)
                r.set_int(3, 1)
                r.set_float(4, 0.0)
                r.set_float(5, adj[b])
                r.set_float(6, 1.0)
                r.set_float(7, 0.0)
                r.set_str(_find_idname_index(r), f"RX-{b}")
                _append(l6, [r])
                touched.add(b)
                if trace is not None:
                    trace.add_action(RepairAction(
                        action_type="adjust_q",
                        stage=stage,
                        round_index=rounds - 1,
                        object_kind="bus",
                        object_id=b,
                        field="LF.L6.QL",
                        before=0.0,
                        after=adj[b],
                        delta=adj[b],
                        trigger=(
                            "over_voltage" if adj[b] > 0 else "under_voltage"),
                        status="applied",
                        metadata={"experimental_new_record": True},
                    ))
            l0 = Card.load(out_dir / "LF.L0")
            l0.records[0].set_int(5, len(l6.records))
            l0.save(out_dir / "LF.L0")
            missing_comp = [b for b in missing_comp if b not in touched]
        if trace is not None:
            for b in missing_comp:
                trace.add_action(RepairAction(
                    action_type="adjust_q",
                    stage=stage,
                    round_index=rounds - 1,
                    object_kind="bus",
                    object_id=b,
                    field="LF.L6.QL",
                    before=None,
                    after=None,
                    delta=adj[b],
                    trigger=("over_voltage" if adj[b] > 0 else "under_voltage"),
                    status="skipped_missing_record",
                    metadata={"voltage": float(vvec[b - 1])},
                ))
        missing_all.update(missing_comp)
        l6.save(out_dir / "LF.L6")
    return {"nr_ok": False, "q_rounds": rounds,
            "missing_comp_buses": sorted(missing_all),
            "protected_skipped": sorted(protected_skipped)}


def _p_redispatch(out_dir: Path, solver, max_rounds: Optional[int] = None,
                  protected_buses=(),
                  config: RuleRepairConfig = DEFAULT_REPAIR_CONFIG,
                  trace: Optional[RepairTrace] = None,
                  stage: str = "p_redispatch") -> dict:
    """安全再调度(真实运行操作):Q 控制器治不了有功输送瓶颈——白盒残差 dP>0 的机组
    送不出(削减),dP<0 的母线缺电(抬升其 2 跳内机组)。裁判 = PSASP。"""
    from ..psasp_io.case import PsaspCase
    from ..eval.nrpf import build_ybus, solve_nr
    from scipy.sparse import coo_matrix as _coo
    from scipy.sparse.csgraph import dijkstra as _dij
    max_rounds = config.p_max_rounds if max_rounds is None else max_rounds
    protected = frozenset(config.with_protected(protected_buses).protected_buses)
    for rounds in range(1, max_rounds + 1):
        case = PsaspCase(out_dir)
        rf = psasp_referee(out_dir, solver)
        _trace_psasp(trace, stage, rounds - 1, rf)
        if rf["converged"]:
            return {"p_ok": True, "p_rounds": rounds - 1}
        nr = solve_nr(case, max_iter=40)          # 停机点残差 = 病灶指示
        _trace_nr(trace, stage, rounds - 1, nr)
        n = case.n_buses
        Y = build_ybus(case, n)
        vv = nr.v * np.exp(1j * nr.theta)
        S = vv * np.conj(Y @ vv)
        Pinj = np.zeros(n)
        slack = None
        gen_rows = {}
        protected_gen_buses = set()
        l5 = Card.load(Path(out_dir) / "LF.L5")
        for row, r in enumerate(l5.records):
            gen = GenView(r, row)
            b = gen.bs_no
            if gen.type_code == 0:
                slack = b
            else:
                Pinj[b - 1] += gen.pg
                if b not in protected:
                    gen_rows[b] = gen
                else:
                    protected_gen_buses.add(b)
        l6 = Card.load(Path(out_dir) / "LF.L6")
        for r in l6.records:
            Pinj[r.get_int(1) - 1] -= r.get_float(4)
        dP = Pinj - S.real
        if slack:
            dP[slack - 1] = 0.0
        # 图上 2 跳邻接(找缺电母线附近的机组)
        lines = [(l.i_no - 1, l.j_no - 1) for l in case.aclines() if l.mark == 1]
        A1 = _coo((np.ones(len(lines)), tuple(zip(*lines))), shape=(n, n))
        moved = 0.0
        for b in np.where(dP > config.p_residual_threshold)[0]:  # 送不出:削减该机
            gen = gen_rows.get(b + 1)
            if gen is None and b + 1 in protected_gen_buses and trace is not None:
                trace.add_action(RepairAction(
                    action_type="redispatch_p",
                    stage=stage,
                    round_index=rounds - 1,
                    object_kind="generator_bus",
                    object_id=b + 1,
                    field="LF.L5.PG",
                    before=None,
                    after=None,
                    delta=None,
                    trigger="positive_p_residual",
                    status="skipped_protected",
                    metadata={"dP": float(dP[b])},
                ))
            if gen is not None:
                cut = min(
                    config.p_adjust_fraction * dP[b],
                    gen.pg - gen.pmin)
                if cut > 0:
                    before = gen.pg
                    after = before - cut
                    gen.pg = after
                    moved += cut
                    if trace is not None:
                        trace.add_action(RepairAction(
                            action_type="redispatch_p",
                            stage=stage,
                            round_index=rounds - 1,
                            object_kind="generator_bus",
                            object_id=b + 1,
                            field="LF.L5.PG",
                            before=before,
                            after=after,
                            delta=-cut,
                            trigger="positive_p_residual",
                            lower=gen.pmin,
                            upper=gen.pmax,
                            metadata={"dP": float(dP[b])},
                        ))
        starving = np.where(dP < -config.p_residual_threshold)[0]
        if starving.size:
            hops = _dij(A1, directed=False, unweighted=True,
                        indices=starving, limit=config.p_search_hops)
            for k, b in enumerate(starving):
                near = np.where(np.isfinite(hops[k]))[0]
                for nb in near:
                    if int(nb) + 1 in protected_gen_buses:
                        if trace is not None:
                            trace.add_action(RepairAction(
                                action_type="redispatch_p",
                                stage=stage,
                                round_index=rounds - 1,
                                object_kind="generator_bus",
                                object_id=int(nb) + 1,
                                field="LF.L5.PG",
                                before=None,
                                after=None,
                                delta=None,
                                trigger="nearby_negative_p_residual",
                                status="skipped_protected",
                                metadata={
                                    "source_bus": int(b) + 1,
                                    "dP": float(dP[b]),
                                },
                            ))
                        continue
                    gen = gen_rows.get(int(nb) + 1)
                    if gen is not None:
                        cap_r = gen.pmax
                        add = min(
                            config.p_adjust_fraction * (-dP[b]),
                            cap_r - gen.pg)
                        if add > 0:
                            before = gen.pg
                            after = before + add
                            gen.pg = after
                            if trace is not None:
                                trace.add_action(RepairAction(
                                    action_type="redispatch_p",
                                    stage=stage,
                                    round_index=rounds - 1,
                                    object_kind="generator_bus",
                                    object_id=int(nb) + 1,
                                    field="LF.L5.PG",
                                    before=before,
                                    after=after,
                                    delta=add,
                                    trigger="nearby_negative_p_residual",
                                    lower=gen.pmin,
                                    upper=cap_r,
                                    metadata={
                                        "source_bus": int(b) + 1,
                                        "dP": float(dP[b]),
                                    },
                                ))
                            break
        l5.save(Path(out_dir) / "LF.L5")
        if moved < 1e-6 and not starving.size:
            break
    return {"p_ok": False, "p_rounds": rounds}


def _snapshot_case_files(out_dir: Path, names) -> dict:
    """读取方式变量卡原始字节，供引擎/写卡异常时逐位回滚。"""
    out_dir = Path(out_dir)
    return {
        out_dir / name: (out_dir / name).read_bytes()
        for name in names if (out_dir / name).exists()
    }


def _restore_case_files(snapshots: dict) -> None:
    """恢复 _snapshot_case_files 产生的逐文件字节快照。"""
    for path, content in snapshots.items():
        path.write_bytes(content)


def _repair_pipeline_impl(out_dir: Path, solver, protected_buses=(),
                          config: RuleRepairConfig = DEFAULT_REPAIR_CONFIG,
                          trace: Optional[RepairTrace] = None) -> dict:
    """标准修复流水线(实测三档 3/3 配方):无功规划 → 安全再调度 → 无功规划复修。
    全程以 PSASP 为裁判(唯一平台)。返回前对**当前最终卡片**重新终审；中间任一阶段
    曾经收敛都不能代替最终裁决。"""
    out_dir = Path(out_dir)
    protected = frozenset(config.with_protected(protected_buses).protected_buses)
    rq = _ac_reactive_planning(
        out_dir, solver, protected_buses=protected, config=config,
        trace=trace, stage="q_primary")
    info = {"q_rounds": rq["q_rounds"], "p_rounds": 0,
            "missing_comp_buses": list(rq.get("missing_comp_buses", [])),
            "protected_skipped": list(rq.get("protected_skipped", []))}
    if not rq["nr_ok"]:
        rp = _p_redispatch(
            out_dir, solver, protected_buses=protected, config=config,
            trace=trace, stage="p_redispatch")
        rq2 = _ac_reactive_planning(
            out_dir, solver, max_rounds=config.q_recheck_rounds,
            protected_buses=protected, config=config,
            trace=trace, stage="q_recheck")
        info.update(p_rounds=rp["p_rounds"], q_rounds=rq["q_rounds"] + rq2["q_rounds"])
        info["missing_comp_buses"] = sorted(set(info["missing_comp_buses"]) |
                                             set(rq2.get("missing_comp_buses", [])))
        info["protected_skipped"] = sorted(set(info["protected_skipped"]) |
                                           set(rq2.get("protected_skipped", [])))
    final = psasp_referee(out_dir, solver)
    _trace_psasp(trace, "final_psasp", 0, final)
    info["nr_ok"] = bool(final["converged"])
    info["final_converged"] = bool(final["converged"])
    if final["converged"]:
        info["final_vmin"] = final["vmin"]
        info["final_vmax"] = final["vmax"]
    if trace is not None:
        trace.finish(
            "final_psasp_converged" if final["converged"]
            else "final_psasp_diverged",
            bool(final["converged"]),
        )
        info["trace"] = trace.to_dict()
    return info


def repair_pipeline(out_dir: Path, solver, protected_buses=(),
                    config: RuleRepairConfig = DEFAULT_REPAIR_CONFIG,
                    trace: Optional[RepairTrace] = None) -> dict:
    """事务化规则修复：异常时恢复本轮进入前的 LF.L5/LF.L6。

    正常的 PSASP 不收敛是算法结果，不触发回滚；引擎异常、许可证异常、写卡异常和人工中断
    会抛出异常，并在传播前恢复方式变量卡，避免留下未经终审的半成品。
    """
    out_dir = Path(out_dir)
    trace = trace or RepairTrace.for_config(config.with_protected(protected_buses))
    snapshots = _snapshot_case_files(out_dir, ("LF.L0", "LF.L5", "LF.L6"))
    try:
        return _repair_pipeline_impl(
            out_dir, solver, protected_buses=protected_buses, config=config,
            trace=trace)
    except BaseException as exc:
        _restore_case_files(snapshots)
        trace.metadata["exception"] = type(exc).__name__
        trace.finish("exception_rollback", None)
        raise


def repair_case_convergent(
    out_dir: Path,
    solver,
    max_repair: Optional[int] = None,
    protected_buses=(),
    config: RuleRepairConfig = DEFAULT_REPAIR_CONFIG,
    trace: Optional[RepairTrace] = None,
) -> dict:
    """既有方式完整修复：Q→P→Q，失败后按固定倍率降载并重试，最终裁判只认 PSASP。

    与 ``synth_convergent`` 不同，本函数不生成底图；它是 A3 评测器和 A4 受控扰动的统一入口。
    ``max_repair=N`` 表示初次规则链失败后最多执行 N 次降载。每次降载后的
    ``repair_pipeline`` 都含当前卡片的 PSASP 终审，绝不留下未验证修改。
    """
    out_dir = Path(out_dir)
    max_repair = config.max_load_repairs if max_repair is None else max_repair
    if max_repair < 0:
        raise ValueError("max_repair 不得为负")
    effective_config = config.with_protected(protected_buses)
    trace = trace or RepairTrace.for_config(effective_config)
    info = repair_pipeline(
        out_dir, solver, protected_buses=protected_buses,
        config=config, trace=trace)
    if info["nr_ok"]:
        info.update(converged=True, repairs=0, trace=trace.to_dict())
        return info

    for attempt in range(1, max_repair + 1):
        snapshots = _snapshot_case_files(out_dir, ("LF.L6",))
        try:
            l6 = Card.load(out_dir / "LF.L6")
            for r in l6.records:
                bus = r.get_int(1)
                for col, field_name in ((4, "LF.L6.PL"), (5, "LF.L6.QL")):
                    before = r.get_float(col)
                    after = before * config.load_scale
                    r.set_float(col, after)
                    trace.add_action(RepairAction(
                        action_type="scale_load",
                        stage="scale_load",
                        round_index=attempt - 1,
                        object_kind="bus",
                        object_id=bus,
                        field=field_name,
                        before=before,
                        after=after,
                        delta=after - before,
                        trigger="repair_pipeline_diverged",
                        metadata={"scale": config.load_scale},
                    ))
            l6.save(out_dir / "LF.L6")
            info = repair_pipeline(
                out_dir, solver, protected_buses=protected_buses,
                config=config, trace=trace)
        except BaseException as exc:
            _restore_case_files(snapshots)
            trace.metadata["exception"] = type(exc).__name__
            trace.finish("load_scale_exception_rollback", None)
            raise
        if info["nr_ok"]:
            info.update(
                converged=True, repairs=attempt, trace=trace.to_dict())
            return info
    trace.finish("repair_budget_exhausted", False)
    info.update(
        nr_ok=False, converged=False, repairs=max_repair,
        trace=trace.to_dict())
    return info


def synth_convergent(n_buses: int, donor_dir: Path, out_dir: Path, solver, seed: int = 0,
                     max_repair: Optional[int] = None, protected_buses=(),
                     config: RuleRepairConfig = DEFAULT_REPAIR_CONFIG,
                     trace: Optional[RepairTrace] = None, **kw) -> dict:
    """生成 → 修复流水线(PSASP 裁判)→ PSASP 终审;终审发散再降载重试。不收敛不出厂。
    max_repair=N 表示初始修复后最多再降载修复 N 次；每次卡片修改后都有对应的 PSASP
    终审，不在最后一次裁决后留下未经验证的修改。全链路唯一仿真平台 = PSASP(D16);
    nrpf 仅在 PSASP 发散时提供白盒电压画像,无裁决权。"""
    max_repair = config.max_load_repairs if max_repair is None else max_repair
    effective_config = config.with_protected(protected_buses)
    trace = trace or RepairTrace.for_config(effective_config)
    info = synth_case(n_buses, donor_dir, out_dir, seed=seed, **kw)
    info.update(repair_pipeline(
        out_dir, solver, protected_buses=protected_buses, config=config,
        trace=trace))
    for attempt in range(max_repair + 1):
        pf = solver.run_powerflow(Path(out_dir))
        if pf.converged and pf.bus_voltage:
            pvs = [r.v_pu for r in pf.bus_voltage if r.v_pu > 0.01]
            direct = {
                "converged": True,
                "vmin": min(pvs),
                "vmax": max(pvs),
                "v": np.array([r.v_pu for r in pf.bus_voltage]),
            }
        else:
            direct = {"converged": False}
        _trace_psasp(trace, "synth_final_psasp", attempt, direct)
        if pf.converged:
            vs = [r.v_pu for r in pf.bus_voltage if r.v_pu > 0.01]
            trace.finish("synth_converged", True)
            info.update(
                converged=True, repairs=attempt, vmin=min(vs), vmax=max(vs),
                trace=trace.to_dict())
            return info
        if attempt == max_repair:
            break
        snapshots = _snapshot_case_files(Path(out_dir), ("LF.L6",))
        try:
            l6 = Card.load(Path(out_dir) / "LF.L6")
            for r in l6.records:
                bus = r.get_int(1)
                for col, field_name in ((4, "LF.L6.PL"), (5, "LF.L6.QL")):
                    before = r.get_float(col)
                    after = before * config.load_scale
                    r.set_float(col, after)
                    trace.add_action(RepairAction(
                        action_type="scale_load",
                        stage="scale_load",
                        round_index=attempt,
                        object_kind="bus",
                        object_id=bus,
                        field=field_name,
                        before=before,
                        after=after,
                        delta=after - before,
                        trigger="repair_pipeline_diverged",
                        metadata={"scale": config.load_scale},
                    ))
            l6.save(Path(out_dir) / "LF.L6")
            info.update(repair_pipeline(
                out_dir, solver, protected_buses=protected_buses, config=config,
                trace=trace))
        except BaseException as exc:
            _restore_case_files(snapshots)
            trace.metadata["exception"] = type(exc).__name__
            trace.finish("load_scale_exception_rollback", None)
            raise
    trace.finish("repair_budget_exhausted", False)
    info.update(
        converged=False, repairs=max_repair, trace=trace.to_dict())
    return info


def synth_family(n_buses: int, donor_dir: Path, out_dir: Path, solver, seed: int = 0,
                 seed_retries: int = 4, base_v_floor: Optional[float] = 0.87, **kw) -> dict:
    """批产用:换种子重试直到出一个**收敛且电压健康**的家族。返回 info(含 seed_used/seed_tries)。

    实测教训(2026-07-09):同一配方下 N=1000 的收敛与种子相关(seed 0/1/2 过,903 不过),
    批产不能靠运气——不收敛就换种子,而不是降标准或换平台(PSASP 是唯一裁判)。

    **底图电压带验收(2026-07-11)**:2k 批产实测两例病态底图(s6001 基态 0.805、
    s6004 基态 0.623),方式几乎全被 0.85 验收线封存(0/50、0/101)——底图只查收敛
    不查电压,坏底图会白烧一整个家族的方式预算。底图须 vmin ≥ base_v_floor(None=关),
    不达标同样换种子。**0.87 初值**:= 方式验收线 0.85 + 裕量;实测健康底图(≥0.892,
    含最高产的 s6002)与病态底图(≤0.805)之间分离带宽,0.90 会误杀 0.892 的好底图。
    """
    last = {}
    rejected = []
    for t in range(seed_retries + 1):
        s = seed + 10_000 * t
        last = synth_convergent(n_buses, donor_dir, out_dir, solver, seed=s, **kw)
        if last.get("converged"):
            vmin = last.get("vmin")
            if base_v_floor is not None and vmin is not None and vmin < base_v_floor:
                rejected.append({"seed": s, "vmin": float(vmin)})
                continue                       # 电压病态底图:换种子,不出厂
            last.update(seed_used=s, seed_tries=t + 1, low_v_rejected=rejected)
            return last
    last.update(seed_used=None, seed_tries=seed_retries + 1, converged=False,
                low_v_rejected=rejected)
    return last
