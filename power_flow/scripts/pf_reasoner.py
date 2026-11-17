"""潮流任务推理器训练(P2 起步;方案 5.1/5.2/5.3;D13/D15)。

编码器冻结(v1 或随机初始化 = 5.3 组1/组2),只训推理器(~1.1M 参数)。
输入 = D15 未解方式编码(泄漏单测锁定);损失 = 类型化未知量 L1(归一化空间);
评估 = 标幺 MAE(内=val / 外=transfer)+ 任务掩码零学习阶梯(L0/L1/L2)与技巧分——
"注入量可见后 θ 是否突破同信息零学习"在此裁决(预训练侧 θ 上限≈1.0 的正道出口)。

  E:/anaconda3/envs/grid_gpu/python.exe scripts/pf_reasoner.py \
      --root E:/AI-2/统一计算大模型/gridfm-data --device cuda [--encoder v1|random] [--resume]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from gridfm.data.dataset import load_families, load_library  # noqa: E402
from gridfm.data.graph import sample_to_data  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.reason.pf_physics import (  # noqa: E402
    _ybus_terms, dc_theta, dc_theta_base_delta, dc_theta_lossy, dc_theta_lossy_v, pf_phys_arrays,
    pf_phys_denorm, pf_phys_loss, pf_phys_residual, pf_phys_row_scale, qv_hint,
)
from gridfm.reason.pf_task import (  # noqa: E402
    PF_FIELDS, PF_SIDE_DIM, PFReasoner, pf_input_view, pf_loss, pf_side_features,
    pf_supervised_cols, pf_theta_hinge_loss, pf_zero_learning_baseline,
)
from gridfm.reason.pf_theta_edge import (  # noqa: E402
    EDGE_SIDE_DIM, REFINE_NEGB, pf_edge_bias_loss, pf_edge_loss, pf_edge_side_features,
    pf_edge_susceptance, pf_refine_free, pf_theta_ref,
)
from gridfm.schema.fields import NODE_TABLE as _NODE_TABLE  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402

DEFAULT_LIBS = ["lib118", "lib_ieee39", "lib_cepri36"]

# eqp_cand / eqq_cand(2026-08-22 第 3 轮):E12 覆盖率闸的分母("本该有方程"的节点),纯掩码、不进损失
PHYS_KEYS = ("eqp_clean", "eqq_clean", "eqp_def", "eqq_def", "eqp_cand", "eqq_cand")
_TH_COL = _NODE_TABLE.offset_of("theta")


BASE_DELTA_REL_MAX = 0.08     # §21.1b:样本注入偏离基态 Σ|ΔP|/Σ|P_b| 超此值 → 线性化失效,退回 θ_dcL
_BASE_HINT_CACHE: dict = {}
_BASE_HINT_WARNED: set = set()
BASE_CACHE_DIR = Path(__file__).resolve().parent.parent / "runs" / "base_cache"


def load_base_hint(family_id: str, n_nodes: int):
    """E18(预注册 §21):按家族惰性加载基态提示常量 (th_b, pinj_b);机器级缓存 `runs/base_cache/<family>.npz`
    (由 `scripts/probes_5060/build_base_cache.py` 从各语料根的 `_base_case` 一次性提取,不写语料目录)。
    缺文件 / 节点数不符 → 返回 None(调用方退回 lossyv 行为),每家族只警告一次。"""
    if family_id in _BASE_HINT_CACHE:
        e = _BASE_HINT_CACHE[family_id]
        return e if (e is not None and e[0].shape[0] == n_nodes) else None
    f = BASE_CACHE_DIR / f"{family_id}.npz"
    if not f.exists():
        _BASE_HINT_CACHE[family_id] = None
        if family_id not in _BASE_HINT_WARNED:
            _BASE_HINT_WARNED.add(family_id)
            print(f"[pf] 基态提示:家族 {family_id} 无基态缓存,该家族退回不带基态的口径"
                  f"(lossybv → lossyv;--edge-base bdelta → acv)", flush=True)
        return None
    z = np.load(f)
    e = (z["theta"].astype(np.float64), z["pinj"].astype(np.float64))
    _BASE_HINT_CACHE[family_id] = e
    if e[0].shape[0] != n_nodes:
        if family_id not in _BASE_HINT_WARNED:
            _BASE_HINT_WARNED.add(family_id)
            print(f"[pf] 基态提示:家族 {family_id} 基态节点数 {e[0].shape[0]} ≠ 样本 {n_nodes},该家族退回", flush=True)
        return None
    return e


def attach_family_extra(raws, roots):
    """把家族 sidecar(`families/*/*/family_extra.npz`:变比 / 移相角)挂到原始样本上:
    `raw.branch_tap` / `raw.branch_shift_rad`(2026-08-22,预注册 §14.1)。没有 sidecar 的家族不挂,
    物理损失保持旧的保守口径(变压器相邻行作废)。逐样本核验支路数与变压器标志位对齐(行序错位即拒)。
    返回 (挂上的样本数, 命中的家族列表)。"""
    from gridfm.schema.fields import BRANCH_TABLE, BRANCH_TYPES
    cache = {}
    for root in roots:
        for p in sorted(Path(root).glob("families/*/*/family_extra.npz")):
            z = np.load(p)
            cache[p.parent.name] = (z["branch_tap"].astype(np.float32),
                                    z["branch_shift_rad"].astype(np.float32),
                                    z["branch_is_xfmr"].astype(bool))
    boff = BRANCH_TABLE.offset_of
    n_hit, fams = 0, set()
    for s in raws:
        e = cache.get(s.family_id)
        if e is None:
            continue
        tap, shift, isx = e
        if len(tap) != s.n_branches:
            raise SystemExit(f"[pf] sidecar 支路数 {len(tap)} ≠ 样本 {s.sample_id} 支路数 {s.n_branches}")
        oh = s.branch_feat[:, boff("branch_type"):boff("branch_type") + 6]
        bt = np.asarray(BRANCH_TYPES)[np.argmax(oh, axis=1)]
        if not np.array_equal(np.isin(bt, ["xfmr_2w", "xfmr_3w", "phase_shifter"]), isx):
            raise SystemExit(f"[pf] sidecar 变压器标志与样本 {s.sample_id} 的 branch_type 不一致(行序错位)")
        s.branch_tap, s.branch_shift_rad = tap, shift
        n_hit += 1
        fams.add(s.family_id)
    return n_hit, sorted(fams)


def split_holdout(samples, holdout_ids):
    """按 sample_id 把库内留出样本从训练池分出(顺序保持;不在清单内的全部留在训练池)。"""
    keep, held = [], []
    for s in samples:
        (held if s.sample_id in holdout_ids else keep).append(s)
    return keep, held


def resolve_edge_base(edge_base: str, n_nodes: int) -> str:
    """E20(预注册-E20 §0):`edge_base` 的取值解析,返回 "acv" 或 "bdelta"。

    "gate<N>" = 按网架规模的数据级门控(架构规则 5;门控量 n_nodes 为给定量):
    n_nodes ≥ N → "bdelta"(样本级 rel 门控在 base_delta_seed 里原样叠加),否则 "acv"(逐位同现行路径)。
    阈值编码在字符串里、经检查点 config 原样透传 —— 不新增 `_pf_data` 形参,plumbing REQUIRED 不变。"""
    if edge_base in ("acv", "bdelta"):
        return edge_base
    if isinstance(edge_base, str) and edge_base.startswith("gate") and edge_base[4:].isdigit():
        return "bdelta" if int(n_nodes) >= int(edge_base[4:]) else "acv"
    raise ValueError(f"edge_base 取值非法:{edge_base!r}(合法:acv | bdelta | gate<N>)")


def base_delta_seed(raw):
    """门控后的基态提示 θ_bΔ [N,3],不满足门控返回 None(调用方退回 DC 口径)。

    E18(§21.1b)与 E19b(§22.1)共用。门控**全部是给定量**:①本家族有基态缓存且节点数相符;
    ②有 sidecar 变比(= 真实家族;合成族的方式是独立重采样,rel 小也可能线性化崩,e18_offline 有实例);
    ③Σ|ΔP|/Σ|P_b| ≤ BASE_DELTA_REL_MAX;④解出的提示至少一个有效位。
    2026-08-24 由 `_pf_data` 内联版重构抽出,逐位等价(E19 单测锁定)。"""
    base = load_base_hint(raw.family_id, raw.n_nodes)
    if base is None or getattr(raw, "branch_tap", None) is None:
        return None
    _noff = _NODE_TABLE.offset_of
    _pm = (raw.node_feat[:, _noff("pg")] - raw.node_feat[:, _noff("pl")]).astype(np.float64)
    _rel = float(np.abs(_pm - base[1]).sum() / max(np.abs(base[1]).sum(), 1e-9))
    if _rel > BASE_DELTA_REL_MAX:
        return None
    seed = dc_theta_base_delta(raw, base[0], base[1])
    return seed if (seed[:, 1] > 0.5).any() else None


def edge_base_auto_choice(raw, dc_arr, seed_arr, vh, tap, shift, shz, score_mode="A",
                          return_scores=False):
    """E21 失配自门控(预注册-E21 §1):对两个候选边基线(θ_acv = dc_arr、θ_bΔ = seed_arr)各算一次
    免标签物理自洽分,小者胜,返回 "acv"/"bdelta"。seed_arr=None(基态不可用)→ "acv",与 bdelta 语义逐位一致。

    主分 A = AC 干净 P 失配:V̂ 沿用 Q–V 提示(PQ 取提示、无效取 1.0;PV/平衡机取给定 V,
    与 dc_theta_lossy_v 的 v_hat 同规则),Ybus 取 pf_phys_arrays(含 sidecar 变比与 shz 口径),
    在 eqp_clean ∩ 两候选均 valid 的节点上取 mean|ΔP|。备分 B = DC B′ 残差(仅 G1 判 A 失败才启用)。
    **无泄漏**:eqp_clean 不含平衡机行(其 pg 为被监督量,被掩掉);PQ 的 v 只用提示;θ 用候选(计算型);
    单测锁定(扰动全部被监督量,选择与分值逐位不变)。"""
    if seed_arr is None:
        return ("acv", None) if return_scores else "acv"
    noff = _NODE_TABLE.offset_of
    n = raw.n_nodes
    if vh is None:
        vh = qv_hint(raw, tap=tap, shift=shift, dc=dc_arr, shunt_zero_harmless=shz)
    tcol = noff("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    is_pq = types == 0
    v_hat = np.where(is_pq, np.where(vh[:, 1] > 0.5, vh[:, 0].astype(np.float64), 1.0),
                     raw.node_feat[:, noff("v")].astype(np.float64))
    v_hat = np.where(np.isfinite(v_hat) & (v_hat > 0.5), v_hat, 1.0)
    pinj = (raw.node_feat[:, noff("pg")] - raw.node_feat[:, noff("pl")]).astype(np.float64)
    arr = pf_phys_arrays(raw, tap=tap, shift=shift, shunt_zero_harmless=shz)
    both = (dc_arr[:, 1] > 0.5) & (seed_arr[:, 1] > 0.5)
    mask = arr["eqp_clean"] & both
    if not mask.any():
        return ("acv", None) if return_scores else "acv"
    row, col = arr["index"][0], arr["index"][1]
    g = arr["g"].astype(np.float64)
    b = arr["b"].astype(np.float64)

    def _score(theta_arr):
        th = np.where(theta_arr[:, 1] > 0.5, theta_arr[:, 0].astype(np.float64), 0.0)
        if score_mode == "A":
            vre, vim = v_hat * np.cos(th), v_hat * np.sin(th)
            ire = np.bincount(row, weights=g * vre[col] - b * vim[col], minlength=n)
            iim = np.bincount(row, weights=g * vim[col] + b * vre[col], minlength=n)
            dp = pinj - (vre * ire + vim * iim)
        else:                                   # 备分 B:DC B′ 残差(纯 θ,免 V)
            boff2 = __import__("gridfm.schema.fields", fromlist=["BRANCH_TABLE"]).BRANCH_TABLE.offset_of
            x = raw.branch_feat[:, boff2("x")].astype(np.float64)
            mark = raw.branch_feat[:, boff2("mark")]
            oh = raw.branch_feat[:, boff2("branch_type"):boff2("branch_type") + 6]
            from gridfm.schema.fields import BRANCH_TYPES as _BT
            bt = np.asarray(_BT)[np.argmax(oh, axis=1)]
            use = (mark == 1) & (bt != "shunt_comp") & (np.abs(x) > 1e-9)
            ei, ej = raw.edge_index
            i_, j_ = ei[use].astype(np.int64), ej[use].astype(np.int64)
            bb = 1.0 / x[use]
            flow = bb * (th[i_] - th[j_])
            dp = pinj - (np.bincount(i_, weights=flow, minlength=n)
                         - np.bincount(j_, weights=flow, minlength=n))
        return float(np.abs(dp[mask]).mean())

    sa, sb = _score(dc_arr), _score(seed_arr)
    pick = "bdelta" if sb < sa else "acv"
    return (pick, {"acv": sa, "bdelta": sb}) if return_scores else pick


def _freeze_ybus(t):
    """提速二期 B:把 _ybus_terms 结果的数组置只读后返回(共享去重防串改;消费方全为只读访问)。"""
    if t is not None:
        for v in t.values():
            if isinstance(v, np.ndarray):
                v.flags.writeable = False
    return t


def _pf_data(sample, cfg, view="field", raw=None, dc=False, phys=False, edge=False,
             vhint=False, vstat=None, negb="sign", shz=False, edge_base="acv"):
    """raw = 同一样本的原始(未归一化)版:phys=True 附物理残差常量(杠杆 a),
    dc=True 把 dc_theta(raw)[N,2] 拼进侧通道(杠杆 d;θ 为 rad 恒等范式,原始即
    归一化;无泄漏声明与单测见 pf_physics.dc_theta)。edge=True 附 θ 边势重建的
    边侧特征与节点侧常量(pf_theta_edge,2026-08-21)。三者都要求给 raw。
    negb = E12 修正步 B′ 的串补口径(pf_edge_susceptance;默认 "sign" = FDPF 标准,探针 B v2 选定)。
    shz = E15 `--shunt-zero-harmless`(并补记录不牵连方程、r≠0 并补按 1/z 计;同时作用于 Q–V 提示与物理常量;
    评估脚本须从检查点 config 的 `shunt_zero_harmless` 取值透传,默认 False = 旧检查点逐位不变)。
    edge_base = E19b `--edge-base`("acv" = 现行,边侧 Δθ 基线取 dc_arr;"bdelta" = 取门控后的 θ_bΔ,
    基态信息进边路径,§22.1)。**只作用于边侧特征**,Q–V 提示与参考可达仍用 dc_arr;
    评估脚本须从检查点 config 的 `edge_base` 透传,默认 "acv" = 旧检查点逐位不变。"""
    v, target, lm = pf_input_view(sample, view=view)
    data = sample_to_data(v, cfg.lape_k, cfg.rwse_steps, random_dim=cfg.random_dim,
                          branch_random_dim=cfg.branch_random_dim)
    data.target = torch.from_numpy(target)
    data.loss_mask = torch.from_numpy(lm)
    dc_arr = None
    tap = getattr(raw, "branch_tap", None) if raw is not None else None
    shift = getattr(raw, "branch_shift_rad", None) if raw is not None else None
    side_arr = None
    seed = None                      # 侧通道用的 θ_bΔ(仅 lossybv 会解出);edge 段复用它免重复 splu
    ybus_T = None                    # 提速二期 B:同输入 _ybus_terms 全 _pf_data 只算一次(惰性,只读)
    if (dc or edge) and raw is not None:
        # lossy:带网损 DC [N,3](θ, valid, fallback);其前两列与 dc_theta 同义,边侧特征亦由它派生;
        # lossyv(E16,§20 修订版):双轨 —— 侧通道节点 θ 仍用带网损 DC(side_arr),
        # 边侧 Δθ 基线 / Q–V 提示 / 参考可达(dc_arr)用 AC 一阶修正的 θ_acv;
        # 后续 qv_hint(dc=dc_arr) 即"Q–V 提示用修正 θ 重算"(用户批准的第二次 Q–V 解);
        # 提示整体无效的样本 θ_acv 逐位退回带网损 DC(老库输入不变)。
        if dc in ("lossyv", "lossybv"):
            # E18(§21.1b 修订版,lossybv):**只换侧通道** —— 节点级 θ = θ_bΔ(基态+ΔDC+损耗增量,
            # 把 70k 的全网功率分摊锚定);**边侧 Δθ 基线 / Q–V 提示 / 参考与 E16(lossyv)逐位相同**
            # (离线闸拦下了"种子版 acv":AC 基态的边差再走 AC 修正 = V 因子加两遍,边/V 全线劣化)。
            # 样本级退回规则(纯给定量):Σ|ΔP|/Σ|P_b| > 0.08 → 线性化失效(合成族重采样方式),
            # 退回 θ_dcL 且 fallback=1(模型可见);无基态缓存/基态解不出同样退回。阈值只由 train/val 分布定
            # (真实族 ≤0.063 全优,合成族 ≥0.091 起好坏参半;e18_offline.json)。
            seed = base_delta_seed(raw) if dc == "lossybv" else None
            d_lossy = None                       # 提速二期 A:带网损 DC 只算一次,传给 lossy_v 复用(逐位同值)
            if seed is not None:
                side_arr = seed
            else:
                d_lossy = dc_theta_lossy(raw)
                side_arr = d_lossy
                if dc == "lossybv":
                    side_arr = d_lossy.copy()
                    side_arr[:, 2] = 1.0                      # 基态提示缺席/退回:fallback 位亮(模型可见)
            # 提速二期 B:同输入的 _ybus_terms 只算一次(替代 lossy_v 内部 Q–V、v_hint、物理常量的重复计算)
            if ybus_T is None and raw.n_branches:
                ybus_T = _freeze_ybus(_ybus_terms(raw, tap, shift, shz))
            dc_arr = dc_theta_lossy_v(raw, tap=tap, shift=shift, shunt_zero_harmless=shz,
                                      d_lossy=d_lossy, ybus_t=ybus_T)
        elif str(dc).startswith("lossy"):
            dc_arr = dc_theta_lossy(raw, final_check=(dc == "lossyg"))
        else:
            dc_arr = dc_theta(raw)
    if side_arr is None:
        side_arr = dc_arr
    vh_arr = None                    # E21:自门控打分复用 v_hint 的 Q–V 提示(没算过则选择器内自算)
    if view == "group":
        side = pf_side_features(sample)
        if dc:
            assert raw is not None, "--dc-theta 需要原始样本"
            d_ = side_arr
            if dc == "bounded":
                th_, ok_ = d_[:, 0], d_[:, 1]
                d_ = np.stack([np.sin(th_) * ok_, np.cos(th_) * ok_,
                               np.tanh(th_ / 2.0) * ok_, ok_], axis=1).astype(np.float32)
            elif dc == "lossy2":
                # 预注册 §13.3(E8):侧通道只供 (θ_dcL, valid),**去掉 fallback 位**。
                # 归因见结果档 §3.4d:fallback 位在训练集约 4% 为 1、合成 700/1000 档 37.5% 为 1,
                # 而侧通道投影为全部头共享 → v/pg 外集越界。边侧 Δθ_dc 仍由 θ_dcL 派生(不变)。
                d_ = d_[:, :2]
            side = np.concatenate([side, d_], axis=1)
        if vhint:
            # 电压物理提示(预注册 §14.2):侧通道 3 列 = (z 化的 v̂·valid, valid, fallback);
            # 残差基 data.v_hint = z 化 v̂(无效处 0),data.v_hint_ok = valid
            assert raw is not None and vstat is not None, "--v-hint 需要原始样本与 v 的归一化统计"
            if ybus_T is None and raw.n_branches:
                ybus_T = _freeze_ybus(_ybus_terms(raw, tap, shift, shz))
            vh = qv_hint(raw, tap=tap, shift=shift, dc=dc_arr, shunt_zero_harmless=shz, ybus_t=ybus_T)
            vh_arr = vh
            zv = ((vh[:, 0] - vstat[0]) / vstat[1]) * vh[:, 1]
            side = np.concatenate([side, np.stack([zv, vh[:, 1], vh[:, 2]], axis=1).astype(np.float32)], axis=1)
            data.v_hint = torch.from_numpy(zv.astype(np.float32))
            data.v_hint_ok = torch.from_numpy(vh[:, 1] > 0.5)
        data.side = torch.from_numpy(side)
    elif vhint:
        raise ValueError("--v-hint 需要 A 口径(--view group)")
    if edge:
        assert raw is not None, "--theta-edge 需要原始样本"
        edge_dc = dc_arr
        eb_req = edge_base
        _sd_auto = None
        if eb_req == "auto":
            # E21(预注册-E21 §1):失配自门控 —— 逐样本对 θ_acv / θ_bΔ 打免标签物理自洽分择优;
            # 基态不可用(缓存缺席 / rel 门 / 无 sidecar)→ "acv",与现行 bdelta 退回语义逐位一致
            _sd_auto = seed if (dc == "lossybv" and seed is not None) else base_delta_seed(raw)
            eb_req = edge_base_auto_choice(raw, dc_arr, _sd_auto, vh_arr, tap, shift, shz)
        if resolve_edge_base(eb_req, raw.n_nodes) == "bdelta":
            # E19b(§22.1):**只换边基线** —— 边侧 Δθ_dc 与 p_dc 由 θ_bΔ 派生(E18 的教训:基态信息
            # 挂在节点侧通道上吃不到,必须进边积分通路)。门控不通过的样本逐位退回 dc_arr(θ_acv);
            # **Q–V 提示(qv_hint)与参考可达(pf_theta_ref)仍用 dc_arr,一列不动** —— 前者是 E16 已验证
            # 资产,后者的可达性不应随基态提示的有无而变。
            # 与 lossybv 同开时复用侧通道已解出的那份(dc_theta_base_delta 每次都要逐分量 splu);
            # auto 路径进来时复用选择器已解出的 _sd_auto,免二次求解
            _sd = _sd_auto if _sd_auto is not None else (
                seed if (dc == "lossybv" and seed is not None) else base_delta_seed(raw))
            if _sd is not None:
                edge_dc = _sd
        es, eu = pf_edge_side_features(raw, edge_dc[:, :2] if edge_dc is not None else None)
        tr, rm, reach, avail = pf_theta_ref(raw, dc_arr[:, :2] if dc_arr is not None else None)
        data.edge_side = torch.from_numpy(es)
        data.edge_use = torch.from_numpy(eu)
        data.theta_ref = torch.from_numpy(tr)
        data.ref_mask = torch.from_numpy(rm)
        data.reach_mask = torch.from_numpy(reach)
        data.theta_avail = torch.from_numpy(avail)
        eb = pf_edge_susceptance(raw, negb)                          # 失配修正步的 B′ 权重(§13.1 / §15.2)
        data.edge_b = torch.from_numpy(eb)
        data.refine_free = torch.from_numpy(pf_refine_free(raw, eb, rm, reach))   # 修正步自由节点(其余 Dirichlet)
        if edge_dc is not None:
            # E25(预注册 §28.4b):θ 铰链的对手 = 自家边基线的节点场(推理器 edge_dc_residual 的
            # 免费起点;lossyv 下提示整体无效样本已按语义退回 θ_dcL)。既有数组切片,零新计算;
            # 只在训练损失(--theta-hinge-weight > 0)时被读取,前向与评估不碰。
            data.theta_hint = torch.from_numpy(np.ascontiguousarray(edge_dc[:, 0], dtype=np.float32))
            data.theta_hint_ok = torch.from_numpy(edge_dc[:, 1] > 0.5)
    if phys:
        assert raw is not None, "物理残差需要原始样本"
        if ybus_T is None and raw.n_branches:
            ybus_T = _freeze_ybus(_ybus_terms(raw, tap, shift, shz))
        arr = pf_phys_arrays(raw, tap=tap, shift=shift, shunt_zero_harmless=shz, ybus_t=ybus_T)
        data.phys_index = torch.from_numpy(arr["index"])
        data.phys_g = torch.from_numpy(arr["g"])
        data.phys_b = torch.from_numpy(arr["b"])
        for k in PHYS_KEYS:
            setattr(data, k, torch.from_numpy(arr[k]))
    return data


def apply_train_dropout(datas, rng, hint_p: float = 0.0, refine_p: float = 0.0):
    """E13 训练期数据级扰动(只在训练循环调用,评估不用;用训练 RNG,可复现):
    hint_p:按样本把电压提示整体置无效——侧通道末三列 (z·valid, valid, fallback) → (0, 0, 1),残差基 0、v_hint_ok 全 False
           (与 E10b"提示缺席"完全同一种输入,让残差头学会无提示档);
    refine_p:按样本把 refine_free 置空 → 修正步的覆盖率闸对该图跳过(学习部分在没有物理步时也要能用)。
    返回 (提示失效样本数, 修正关闭样本数)。"""
    n_h = n_r = 0
    for d in datas:
        if hint_p and hasattr(d, "v_hint") and rng.random() < hint_p:
            d.side[:, -3:] = torch.tensor([0.0, 0.0, 1.0], dtype=d.side.dtype)
            d.v_hint = torch.zeros_like(d.v_hint)
            d.v_hint_ok = torch.zeros_like(d.v_hint_ok)
            n_h += 1
        if refine_p and hasattr(d, "refine_free") and rng.random() < refine_p:
            d.refine_free = torch.zeros_like(d.refine_free)
            n_r += 1
    return n_h, n_r


def _phys_terms(pred, batch, dn):
    """AMP 外 fp32 失配(CUDA index_add 原子加,训练容忍非确定序)。"""
    dp, dq = pf_phys_residual(pred.float(), batch.target.float(), batch.loss_mask,
                              batch.phys_index, batch.phys_g, batch.phys_b, dn)
    return dp, dq, {k: getattr(batch, k) for k in PHYS_KEYS}


def _forward(enc, reasoner, batch, use_amp, amp_dtype, dev, field_weights=None,
             train_enc=False, edge_weight=1.0, return_edge=False, edge_bias_weight=0.0, dn=None,
             refine_aux_weight=0.0, hinge_weight=0.0):
    """返回 (pred, loss);return_edge=True 时返回 (pred, loss, delta)(非 theta_edge 推理器 delta=None,诊断用)。
    theta_edge 推理器:loss 含 edge_weight × 逐边 Δθ L1;
    theta_wrap 推理器:θ 监督项用周期误差(pf_loss theta_periodic)。旧推理器逐位同旧。
    hinge_weight(E25,预注册 §28):>0 时加图级非对称铰链 pf_theta_hinge_loss(θ 不得劣于
    自家零训练边基线 theta_hint);0 = 完全短路,数值路径与现行逐位一致。"""
    theta_edge = getattr(reasoner, "theta_edge", False)
    theta_wrap = getattr(reasoner, "theta_wrap", False)
    with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
        if train_enc:
            h, g, e = enc(batch)
        else:
            with torch.no_grad():
                h, g, e = enc(batch)
        extra = {}
        if theta_edge:
            extra = dict(edge_side=batch.edge_side, edge_use=batch.edge_use,
                         theta_ref=batch.theta_ref, ref_mask=batch.ref_mask,
                         reach_mask=batch.reach_mask, return_edge=True)
        if getattr(reasoner, "v_hint", False):
            # 用 getattr:批里没带提示时交给推理器的配对判错(ValueError),与其它开关一致
            extra["v_hint"] = getattr(batch, "v_hint", None)
            extra["v_hint_ok"] = getattr(batch, "v_hint_ok", None)
        if getattr(reasoner, "refine_steps", 0):
            if dn is None:
                raise SystemExit("[pf] --refine-steps 需要物理常量(--phys-weight > 0 才会装载 dn)")
            extra["phys"] = dict(index=batch.phys_index, g=batch.phys_g, b=batch.phys_b,
                                 denorm=dn, target=batch.target, loss_mask=batch.loss_mask,
                                 eqp_clean=batch.eqp_clean, eqq_clean=batch.eqq_clean,
                                 eqq_cand=batch.eqq_cand, edge_b=batch.edge_b,
                                 refine_free=batch.refine_free)
        out = reasoner(h, g, e, batch.batch, batch.batch[batch.edge_index[0]],
                       side=getattr(batch, "side", None),
                       edge_index=batch.edge_index if (len(reasoner.mp) or theta_edge) else None,
                       **extra)
        pred, delta = out if theta_edge else (out, None)
        loss = pf_loss(pred, batch.target, batch.loss_mask, field_weights=field_weights,
                       theta_periodic=theta_wrap)
        if refine_aux_weight > 0 and getattr(reasoner, "refine_steps", 0):
            # E13:修正前的 θ⁰ / V⁰ 同样受监督(同口径 L1),防止模型把活儿全推给物理步
            loss = loss + refine_aux_weight * pf_loss(reasoner.pre_refine_pred, batch.target, batch.loss_mask,
                                                      field_weights=field_weights, theta_periodic=theta_wrap)
        if delta is not None and edge_weight > 0:
            loss = loss + edge_weight * pf_edge_loss(delta, batch.edge_index, batch.edge_use,
                                                    batch.target[:, _TH_COL], batch.theta_avail)
        if delta is not None and edge_bias_weight > 0:
            loss = loss + edge_bias_weight * pf_edge_bias_loss(
                delta, batch.edge_index, batch.edge_use, batch.target[:, _TH_COL], batch.theta_avail,
                batch.batch[batch.edge_index[0]], int(batch.num_graphs))
        if hinge_weight > 0:
            th_hint = getattr(batch, "theta_hint", None)
            th_ok = getattr(batch, "theta_hint_ok", None)
            if th_hint is None or th_ok is None:
                raise ValueError("--theta-hinge-weight > 0 需要 _pf_data 附 theta_hint"
                                 "(edge 路径且 dc 基线在场;见预注册 §28.4b)")
            h, n_act, n_g = pf_theta_hinge_loss(
                pred[:, PF_FIELDS.index("theta")], batch.target[:, _TH_COL], th_hint, th_ok,
                batch.loss_mask[:, _TH_COL], batch.batch, int(batch.num_graphs))
            reasoner.hinge_stats = (h.detach(), n_act, n_g)      # 打印步才 .item(),不加同步税
            loss = loss + hinge_weight * h
    if return_edge:
        return pred, loss, delta
    return pred, loss


@torch.no_grad()
def eval_pu_mae(enc, reasoner, samples, cfg, norm, dev, use_amp, amp_dtype, bs=16,
                view="field", raws=None, dn=None, dc=False, phys=False, edge=False,
                vhint=False, vstat=None, negb="sign", shz=False, edge_base="acv"):
    """任务协议下逐字段标幺 MAE(只在类型化未知量位置)。

    raws+dn 给定时并报物理失配诊断(干净/定义方程平均 |失配|,标幺;
    预测的物理自洽度,与真值 MAE 相互独立)。"""
    cols = pf_supervised_cols()
    abs_err = {f: 0.0 for f in PF_FIELDS}
    counts = {f: 0 for f in PF_FIELDS}
    phys_sum = {"phys_clean": 0.0, "phys_def": 0.0}
    phys_cnt = {"phys_clean": 0, "phys_def": 0}
    enc_was_training = enc.training           # e2e 臂:评估期编码器必须 eval(dropout)
    enc.eval()
    reasoner.eval()
    for i in range(0, len(samples), bs):
        rr = raws[i:i + bs] if raws is not None else [None] * bs
        datas = [_pf_data(s, cfg, view, raw=r, dc=dc, phys=phys, edge=edge, vhint=vhint, vstat=vstat, negb=negb,
                          shz=shz, edge_base=edge_base)
                 for s, r in zip(samples[i:i + bs], rr)]
        batch = Batch.from_data_list(datas).to(dev)
        pred, _ = _forward(enc, reasoner, batch, use_amp, amp_dtype, dev, dn=dn)
        if phys:
            dp, dq, masks = _phys_terms(pred, batch, dn)
            for key, terms in (("phys_clean", ((dp, masks["eqp_clean"]),
                                               (dq, masks["eqq_clean"]))),
                               ("phys_def", ((dp, masks["eqp_def"]),
                                             (dq, masks["eqq_def"])))):
                for x, m in terms:
                    phys_sum[key] += float(x[m].abs().sum())
                    phys_cnt[key] += int(m.sum())
        pred = pred.float().cpu().numpy()
        target = batch.target.float().cpu().numpy()
        lm = batch.loss_mask.cpu().numpy()
        pred_full = target.copy()
        for k, f in enumerate(PF_FIELDS):
            pred_full[:, cols[f]] = pred[:, k]
        pred_pu = norm.inverse_transform_node(pred_full)
        target_pu = norm.inverse_transform_node(target)
        for f in PF_FIELDS:
            m = lm[:, cols[f]]
            if m.any():
                abs_err[f] += float(np.abs(pred_pu[m, cols[f]] - target_pu[m, cols[f]]).sum())
                counts[f] += int(m.sum())
    reasoner.train()
    if enc_was_training:
        enc.train()
    out = {f: (abs_err[f] / counts[f]) if counts[f] else float("nan") for f in PF_FIELDS}
    if phys:
        for k in phys_sum:
            out[k] = phys_sum[k] / max(phys_cnt[k], 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--v1-dir", default="runs/encoder_v1")
    ap.add_argument("--encoder", choices=["v1", "random", "e2e"], default="v1",
                    help="v1 = 5.3 组1(冻结预训练);random = 组2(冻结随机初始化);"
                         "e2e = 组3(同架构端到端从头联训,编码器随任务损失更新;"
                         "审查 A1 补试,不触碰 v1 冻结档)")
    ap.add_argument("--view", choices=["field", "group"], default="field",
                    help="D15 口径对照(07-11 裁决点 3):field=B 现行字段级;"
                         "group=A 整 STATE 组遮蔽+给定量推理器侧直供")
    ap.add_argument("--phys-weight", type=float, default=0.0,
                    help="物理残差损失权重(干净方程:PQ P/Q + PV P;θ 杠杆 a,"
                         "0=关闭与旧行为逐位一致;方案执行清单 1a,权重只在 val 判)")
    ap.add_argument("--phys-defeq-weight", type=float, default=0.0,
                    help="定义方程(PV Q/平衡机 P Q)权重;因样本 qg/平衡机 pg 存卡值"
                         "非解值(变更记录 07-11),与 L1 监督相拉扯,v0 默认 0 待裁决")
    ap.add_argument("--theta-loss-weight", type=float, default=1.0,
                    help="θ 字段级损失权重(杠杆 b;1.0=旧行为逐位一致)")
    ap.add_argument("--theta-head", action="store_true",
                    help="独立 θ 头(杠杆 b;防共享头末层挤压 θ 梯度)")
    ap.add_argument("--gnn-layers", type=int, default=0,
                    help="推理器侧薄消息传递层数(杠杆 c;补 A 口径传播算力,0=关)")
    ap.add_argument("--dc-theta", action="store_true",
                    help="DC-θ 特征直供侧通道(杠杆 d,2026-07-12 用户批准 B 方案;"
                         "报告并列 DC 裸值基线行,门禁仍挂 L2)")
    ap.add_argument("--v-hint", action="store_true",
                    help="电压物理提示 + v 残差头(预注册 §14.2,E10;Q–V 半步 4 轮,sidecar 变比 + 并补 −1/x;"
                         "需 --view group)")
    ap.add_argument("--refine-steps", type=int, default=0,
                    help="势重建后展开几步快速解耦潮流式 P–θ 修正(预注册 §13.1 E6 / §15.2 E12;0 = 关,与旧检查点逐位兼容)。"
                         "需 --theta-edge 且 --phys-weight > 0(要用物理常量算失配)")
    ap.add_argument("--refine-qv", action="store_true",
                    help="E12:每个修正步后附加一次 Q–V 半步(B″·ΔV = ΔQ/V,β 可学习;预注册 §15.2;运行名后缀 qv)")
    ap.add_argument("--refine-aux-weight", type=float, default=0.0,
                    help="E13:修正前 θ⁰/V⁰ 的辅助 L1 监督权重(0 = 关,与 E12 逐位一致;运行名后缀 _aux<w>)")
    ap.add_argument("--hint-dropout", type=float, default=0.0,
                    help="E13:训练时按样本以此概率把电压提示整体置无效(valid=0、fallback=1),让残差头学会无提示档(老方式库);评估不用")
    ap.add_argument("--refine-dropout", type=float, default=0.0,
                    help="E13:训练时按样本以此概率关闭修正步(refine_free 置空 → 覆盖率闸跳过),让学习部分保持独立可用;评估不用")
    ap.add_argument("--refine-ls", action="store_true",
                    help="E13:P–θ 步残差回溯线搜索(α×{1,½,¼} 取第一个让 mean|ΔP| 下降的;都不降则跳过;运行名后缀 _ls)")
    ap.add_argument("--shunt-zero-harmless", action="store_true",
                    help="(2026-08-25 起为默认行为,保留只为兼容旧命令行/队列脚本;要关用 --no-shunt-zero-harmless)")
    ap.add_argument("--no-shunt-zero-harmless", action="store_true",
                    help="退回 E15 之前的并补口径(并补记录牵连相邻方程作废、r≠0 并补只取 −1/x)。"
                         "**只在复现 pre-E15 归档臂时用**;正常训练不要加 —— 该口径在 mp_case300 上"
                         "会丢 15 个节点方程、2 个节点带 ~1.3e-3 pu 系统偏置(见结果档 §3.31(e))")
    ap.add_argument("--phys-rownorm", action="store_true",
                    help="E15b(预注册 §19.2):物理残差损失逐行归一化——ΔP/ΔQ 除以该行 max(|B_ii|, 1) 再做 L1(结果档 §3.22(4) 的 E9b;"
                         "评估打印的 |ΔPQ| 仍为原始标幺值);需 --phys-weight > 0;默认关;运行名后缀 _prn")
    ap.add_argument("--refine-negb", choices=list(REFINE_NEGB), default="sign",
                    help="E12 修正步 B′ 里串补(x<0)的口径:sign = 带符号 1/x(FDPF 标准,探针 B v2 选定,默认)/"
                         "zero = 置 0 并把孤块按 Dirichlet 跳过(旧 pf_edge_susceptance 行为,探针实测第 2 步即发散)/ abs")
    ap.add_argument("--dc-encoding", choices=["raw", "bounded", "lossy", "lossyg", "lossy2", "lossyv", "lossybv"],
                    default="raw",
                    help="DC-θ 表示:raw=[θ_dc, valid];bounded=[sin, cos, tanh(θ/2), valid]"
                         "(外集大网 DC 崩坏时裸值幅度支配侧投影,有界编码抑制,"
                         "同一批准杠杆的表示层选项);lossy=[θ_dcL, valid, fallback] 带网损 DC;"
                         "lossyg=lossy + 末轮结果补做 (a)(b) 复核(结果档 §3.11 修法 1;"
                         "合成语料上会多回退 0.2~0.4%% 的样本,真实尺度 0,故与 lossy 分开计);"
                         "lossy2=lossy 去掉 fallback 位(侧通道只留 θ_dcL 与 valid,预注册 §13.3);"
                         "lossyv=V 修正 DC(E16,预注册 §20,用户 2026-08-23 批准:逐边 Δθ_dcL÷V̂_iV̂_j×t 再最小二乘积分,"
                         "Q–V 提示用 dcv θ 重算一次)"
                         "(迭代型物理先验,2026-08-21,须用户批准后方可用于训练)")
    ap.add_argument("--theta-edge", action="store_true",
                    help="θ 边势重建(2026-08-21 5060 线,pf_theta_edge):逐支路 Δθ 头(以 Δθ_dc 为"
                         "局部基线)+ 图上最小二乘积分(平衡机给定角为边界);须 --view group 与原始样本")
    ap.add_argument("--theta-wrap", action="store_true",
                    help="θ 输出折回 (−π, π](存档真值为 ±π 截断值),θ 监督项改周期误差 "
                         "wrap(pred−target);评估口径不变(原样 MAE 对存档 θ)")
    ap.add_argument("--edge-loss-weight", type=float, default=1.0,
                    help="--theta-edge 的逐边 Δθ L1 权重(0 = 只经势重建后的 θ 监督)")
    ap.add_argument("--no-edge-dc-residual", action="store_true",
                    help="--theta-edge 时不以 Δθ_dc 作边头的残差基线(消融)")
    ap.add_argument("--edge-zero-init", action="store_true",
                    help="E19a(§22.1):边头末层零初始化 → 步 0 的 δ 恒等于边基线,训练从零训练地板出发"
                         "(需 --theta-edge 且不带 --no-edge-dc-residual)")
    ap.add_argument("--edge-base", default="acv",
                    help="E19b(§22.1):边侧 Δθ 基线的来源。acv = 现行(dc_arr,与 --dc-encoding 同源);"
                         "bdelta = 门控后的基态提示 θ_bΔ(基态信息进边路径;门控不通过逐样本退回 acv);"
                         "gate<N>(E20,预注册-E20):n_nodes ≥ N 的样本走 bdelta、其余逐位 acv。"
                         "只换边侧,Q–V 提示与参考可达不动")
    ap.add_argument("--theta-hinge-weight", type=float, default=0.0,
                    help="E25(预注册 §28):θ 图级非对称铰链权重 —— 每图 max(0, 模型 wrap-MAE −"
                         " 自家边基线 wrap-MAE),只在模型劣于自家零训练基线时罚(对手无梯度,"
                         "梯度朝真值);0 = 关(现行为逐位)。只作用于训练损失,推理零改动")
    ap.add_argument("--edge-bias-weight", type=float, default=0.0,
                    help="--theta-edge 的逐图\"沿真实 Δθ 方向平均偏差\"惩罚权重(压系统性收缩;0 = 关)")
    ap.add_argument("--edge-head-local", action="store_true",
                    help="--theta-edge 的局部物理边头(E4):只看两端给定量 + 预测电压 + 边侧物理量,"
                         "不看节点 token / 上下文 / 边 token(与规模无关)")
    ap.add_argument("--extra-root", action="append", default=[],
                    help="额外语料根(可重复):并入其 families/train 到训练集(如 gridfm-data-large);不碰其 transfer")
    ap.add_argument("--val-holdout", action="append", default=[],
                    help="库内留出清单 json(可重复,含 val_holdout_sample_ids):这些 sample_id 从训练池移入 val")
    ap.add_argument("--extra-weight", type=float, default=1.0,
                    help="--extra-root 样本的采样权重倍数(1 = 与主语料按样本数均匀;>1 上采样大算例)")
    ap.add_argument("--out", default=None,
                    help="默认 runs/pf_reasoner_<encoder>[_group][_phys<w>]")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--queries", type=int, default=16)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    # 2026-08-25(PSASP 规范版本核查后):并补正解口径**改为命令行默认开**。翻的是 CLI 默认而非库默认 ——
    # 库默认仍是 False,归档检查点的复评由评估脚本从 ckpt config 取值,`tests/test_pf_physics_e15.py`
    # 的"默认 = 旧口径逐位不变"守卫原样保留。这里只堵"新训练忘加开关就静默用错物理"这一个口子。
    # 必须放在 parse_args 之后、运行名后缀/续训守卫/config 落盘之前(它们都读这个值)。
    a.shunt_zero_harmless = not a.no_shunt_zero_harmless
    use_phys = a.phys_weight > 0 or a.phys_defeq_weight > 0
    use_raw = use_phys or a.dc_theta or a.theta_edge or a.v_hint
    dc_mode = a.dc_encoding if a.dc_theta else False
    if a.dc_theta and a.view != "group":
        raise SystemExit("[pf] --dc-theta 只在 A 口径(--view group)下定义")
    if a.theta_edge and a.view != "group":
        raise SystemExit("[pf] --theta-edge 只在 A 口径(--view group)下定义(平衡机给定角经侧通道)")
    if a.refine_steps and not (a.theta_edge and a.phys_weight > 0):
        raise SystemExit("[pf] --refine-steps 需要同时开 --theta-edge 与 --phys-weight>0"
                         "(修正步作用在势重建输出上,且要用物理常量算失配)")
    if a.refine_qv and not a.refine_steps:
        raise SystemExit("[pf] --refine-qv 需要 --refine-steps > 0")
    if a.refine_ls and not a.refine_steps:
        raise SystemExit("[pf] --refine-ls 需要 --refine-steps > 0")
    if (a.refine_aux_weight or a.refine_dropout) and not a.refine_steps:
        raise SystemExit("[pf] --refine-aux-weight / --refine-dropout 需要 --refine-steps > 0")
    if a.hint_dropout and not a.v_hint:
        raise SystemExit("[pf] --hint-dropout 需要 --v-hint")
    if a.phys_rownorm and not use_phys:
        raise SystemExit("[pf] --phys-rownorm 需要 --phys-weight > 0(归一化的是物理残差损失)")
    if not a.theta_edge and (a.edge_zero_init or a.edge_base != "acv"):
        raise SystemExit("[pf] --edge-zero-init / --edge-base 需要 --theta-edge(两者只作用于 θ 边路径)")
    if a.edge_base == "auto":
        raise SystemExit("[pf] --edge-base auto(E21 失配自门控)仅限评估路径(经检查点 config 注入);"
                         "训练用法未预注册(预注册-E21 §1)")
    resolve_edge_base(a.edge_base, 0)          # 取值合法性即刻校验(acv | bdelta | gate<N>),非法直接报错
    if a.edge_zero_init and a.no_edge_dc_residual:
        raise SystemExit("[pf] --edge-zero-init 与 --no-edge-dc-residual 互斥(零初始化的意义是恒等于残差基线)")
    if not a.theta_edge and (a.edge_loss_weight != 1.0 or a.no_edge_dc_residual or a.edge_bias_weight != 0.0
                             or a.edge_head_local):
        raise SystemExit("[pf] --edge-loss-weight / --no-edge-dc-residual / --edge-bias-weight / --edge-head-local "
                         "只在 --theta-edge 下有意义")
    fw = {"theta": a.theta_loss_weight} if a.theta_loss_weight != 1.0 else None
    suffix = "" if a.view == "field" else f"_{a.view}"
    if use_phys:
        suffix += f"_phys{a.phys_weight:g}"
        if a.phys_defeq_weight > 0:
            suffix += f"_def{a.phys_defeq_weight:g}"
    if a.theta_loss_weight != 1.0:
        suffix += f"_tw{a.theta_loss_weight:g}"
    if a.theta_head:
        suffix += "_th"
    if a.gnn_layers:
        suffix += f"_gnn{a.gnn_layers}"
    if a.dc_theta:
        suffix += "_dc" + {"raw": "", "bounded": "b", "lossy": "L", "lossyg": "Lg", "lossy2": "L2", "lossyv": "Lv",
                           "lossybv": "Lbv"}[a.dc_encoding]
    if a.theta_edge:
        suffix += "_edge" + ("" if a.edge_loss_weight == 1.0 else f"{a.edge_loss_weight:g}") \
                  + ("nr" if a.no_edge_dc_residual else "") \
                  + ("_zi" if a.edge_zero_init else "") \
                  + ("_bE" if a.edge_base == "bdelta" else
                     (f"_gE{int(a.edge_base[4:]) // 1000}k" if a.edge_base.startswith("gate")
                      and int(a.edge_base[4:]) % 1000 == 0 else
                      f"_gE{a.edge_base[4:]}" if a.edge_base.startswith("gate") else "")) \
                  + ("_vh" if a.v_hint else "") \
                  + ("_shz" if a.shunt_zero_harmless else "") \
                  + (f"_rf{a.refine_steps}" + ("qv" if a.refine_qv else "") if a.refine_steps else "") \
                  + ("_ls" if a.refine_ls else "") \
                  + (f"_aux{a.refine_aux_weight:g}" if a.refine_aux_weight else "") \
                  + (f"_hd{a.hint_dropout:g}" if a.hint_dropout else "") \
                  + (f"_rd{a.refine_dropout:g}" if a.refine_dropout else "") \
                  + ("_prn" if a.phys_rownorm else "") \
                  + (f"_eb{a.edge_bias_weight:g}" if a.edge_bias_weight else "") \
                  + ("_loc" if a.edge_head_local else "")
    if a.theta_wrap:
        suffix += "_wrap"
    out = Path(a.out or f"runs/pf_reasoner_{a.encoder}{suffix}")
    out.mkdir(parents=True, exist_ok=True)

    # 写保护(2026-07-11 评审):非续训却指向已有检查点目录 = 静默覆写归档基线。
    # runs/ 不入 git,覆写不可逆——启动即拒绝,除非 --resume 且口径/编码器相符(下方再校验)。
    existing = sorted((out / "ckpt").glob("step_*.pt"))
    if existing and not a.resume:
        raise SystemExit(f"[pf] {out}/ckpt 已有 {len(existing)} 个检查点且未 --resume,"
                         f"拒绝覆写归档基线;换 --out 或显式 --resume")

    v1 = Path(a.v1_dir)
    ck = torch.load(v1 / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in allowed})
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(a.seed)
    enc, _dec = build_models(cfg)
    if a.encoder == "v1":
        enc.load_state_dict(ck["model_state"])
    train_enc = a.encoder == "e2e"
    for p in enc.parameters():
        p.requires_grad_(train_enc)
    enc.to(dev)
    enc.train() if train_enc else enc.eval()
    n_enc = sum(p.numel() for p in enc.parameters())
    print(f"[pf] 编码器 = {a.encoder}"
          + (f"(端到端可训 {n_enc/1e6:.2f}M,随机初始化 seed={a.seed}"
             if train_enc else "(冻结")
          + f",d{cfg.d}/L{cfg.n_layers}/K{cfg.n_global});"
          f"normalizer = 冻结档;口径 = {a.view}"
          + ("(A:整 STATE 组遮蔽 + 给定量侧通道)" if a.view == "group" else "(B:字段级)"))

    norm = Normalizer.load(v1 / "normalizer.npz")
    dn = torch.from_numpy(pf_phys_denorm(norm)).to(dev) if use_phys else None
    _vc = _NODE_TABLE.offset_of("v")
    vstat = (float(norm.node_mean[_vc]), float(norm.node_std[_vc])) if a.v_hint else None
    if use_phys:
        print(f"[pf] 物理残差:干净方程 w={a.phys_weight:g} / 定义方程 w={a.phys_defeq_weight:g}"
              f"(标幺 L1;Ybus 行不完整节点自动排除)"
              + (";逐行归一化 ÷max(|B_ii|,1)(E15b --phys-rownorm)" if a.phys_rownorm else ""))
    # 2026-08-25(PSASP 规范版本核查后):并补正解口径**改为默认开**。翻的是命令行默认值而非库默认值 ——
    # 库默认仍是 False,归档检查点的复评由评估脚本从 ckpt config 取值,`tests/test_pf_physics_e15.py`
    # 的"默认 = 旧口径逐位不变"守卫也原样保留。这里只堵"新训练忘加开关就静默用错物理"这一个口子。
    if a.shunt_zero_harmless:
        print("[pf] 并补正解口径(E15,现为默认):并补记录不牵连相邻方程,r≠0 并补按 1/z 计"
              "(只影响 mp_case300 的 17 条与 cepri36 的 2 条占位记录)")
    else:
        print("[pf] !! --no-shunt-zero-harmless:退回 pre-E15 并补口径 —— 仅供复现归档臂,"
              "mp_case300 会丢 15 个节点方程,勿用于新实验", flush=True)
    root = Path(a.root)
    train_raw = load_families(root, "train")
    for lib in DEFAULT_LIBS:
        train_raw += load_library(root / lib)
    val_raw = load_families(root, "val")
    xfer_raw = load_families(root, "transfer")
    n_main = len(train_raw)
    # 额外语料(2026-08-21 大规模算例):只并入其 train;库内留出清单中的样本移入 val
    holdout_ids = set()
    for hp in a.val_holdout:
        holdout_ids |= set(json.loads(Path(hp).read_text(encoding="utf-8"))["val_holdout_sample_ids"])
    extra_raw = []
    for er in a.extra_root:
        extra_raw += load_families(Path(er), "train")
    # 2026-08-21 第 2 轮修法(结果档 §3.11 第 5 项):跨 root 零交集断言。
    # 口径说明(2026-08-21 20:05 修正):**不能**按 family_id 判——`mp_activsg2000` 本来就同时在
    # 交付语料与本机扩产语料里,那是同一底图的**不同方式**,设计如此(样本级零重复已在两轮语料核验中确认)。
    # 真正要拦的是:(a) 同一 sample_id 被喂两遍(权重口径失真);(b) train 家族与 transfer 家族相交(泄漏,
    # CLAUDE.md 第 6 条)。
    ids_all = [s.sample_id for s in train_raw] + [s.sample_id for s in extra_raw]
    if len(ids_all) != len(set(ids_all)):
        from collections import Counter
        dup = [k for k, v in Counter(ids_all).items() if v > 1]
        raise SystemExit(f"[pf] 训练池有 {len(dup)} 个重复 sample_id(跨 root 喂了两遍):{sorted(dup)[:5]}")
    fam_train = {s.family_id for s in train_raw} | {s.family_id for s in extra_raw}
    fam_xfer = {s.family_id for s in xfer_raw}
    leak = fam_train & fam_xfer
    if leak:
        raise SystemExit(f"[pf] train 与 transfer 家族相交(泄漏):{sorted(leak)[:5]}")
    print(f"[pf] 零交集断言 PASS:训练池 {len(ids_all)} 个样本无重复 id;"
          f"train {len(fam_train)} 族与 transfer {len(fam_xfer)} 族不相交")
    extra_train, extra_val = split_holdout(extra_raw, holdout_ids)
    # 2026-08-21 审查:留出清单必须全部命中,且留出 id 不得出现在主语料训练池(静默漏传 root/清单 = 口径错)
    hit = {s.sample_id for s in extra_val}
    missing = holdout_ids - hit
    if missing:
        raise SystemExit(f"[pf] 留出清单有 {len(missing)} 个 sample_id 未在 --extra-root 语料中命中:{sorted(missing)[:5]}")
    _main_keep, main_held = split_holdout(train_raw, holdout_ids)
    if main_held:
        raise SystemExit(f"[pf] 留出 id 出现在 --root 训练池:{[s.sample_id for s in main_held][:5]}")
    train_raw += extra_train
    val_raw += extra_val
    _n_tap, _tap_fams = attach_family_extra(train_raw + val_raw + xfer_raw, [a.root] + list(a.extra_root))
    if a.edge_base != "acv":
        # E19b(bdelta):全部非老库家族须有基态缓存;E20(gate<N>):只有 n_nodes ≥ N 的家族会走 θ_bΔ,只查它们
        _thr = 0 if a.edge_base == "bdelta" else int(a.edge_base[4:])
        _sizes: dict[str, int] = {}
        for r in train_raw + val_raw + xfer_raw:
            _sizes[r.family_id] = max(_sizes.get(r.family_id, 0), r.n_nodes)
        _fams_e = sorted(f for f, nn in _sizes.items() if nn >= _thr)
        _miss_e = [f for f in _fams_e if f not in ("ieee118", "ieee39", "cepri36")   # 老库 family_id(非目录名)
                   and not (BASE_CACHE_DIR / f"{f}.npz").exists()]
        if _miss_e:
            raise SystemExit(f"[pf] --edge-base {a.edge_base} 要求门控内家族有基态缓存;缺 {len(_miss_e)} 个:{_miss_e[:6]}…"
                             f"\n先跑 scripts/probes_5060/build_base_cache.py")
        print(f"[pf] --edge-base {a.edge_base}(§22.1/预注册-E20):基态缓存齐全(门控内 {len(_fams_e)} 家族);"
              f"**边侧 Δθ 基线** = θ_bΔ(门控:n_nodes≥{_thr} 且真实家族且 rel≤{BASE_DELTA_REL_MAX:g},否则逐样本退回 acv);"
              f"Q–V 提示与参考可达仍用 {a.dc_encoding} 口径,一列不动", flush=True)
    if a.dc_encoding == "lossybv":
        _fams_all = sorted({r.family_id for r in train_raw + val_raw + xfer_raw})
        _miss = [f for f in _fams_all if f not in ("ieee118", "ieee39", "cepri36")   # 老库样本的 family_id(非目录名)
                 and not (BASE_CACHE_DIR / f"{f}.npz").exists()]
        if _miss:
            raise SystemExit(f"[pf] --dc-encoding lossybv 要求全部非老库家族有基态缓存;缺 {len(_miss)} 个:{_miss[:6]}…"
                             f"\n先跑 scripts/probes_5060/build_base_cache.py")
        print(f"[pf] E18 lossybv(§21.1b):基态缓存齐全({len(_fams_all)} 家族);侧通道 = θ_bΔ"
              f"(真实家族且 rel≤{BASE_DELTA_REL_MAX:g},退回样本 fallback=1);边侧/Q–V 与 lossyv 逐位相同", flush=True)
    print(f"[pf] 家族 sidecar(变比/移相)挂载:{_n_tap} 个样本,家族 {_tap_fams};"
          f"物理损失按 j 侧变比 + 并补 −1/x 建 Ybus(2026-08-22 修正,无 sidecar 的家族保持保守口径)")
    if a.extra_root:
        print(f"[pf] 额外语料 {a.extra_root}:train +{len(extra_train)} / val +{len(extra_val)}"
              f"(留出清单 {len(holdout_ids)} 条,命中 {len(extra_val)});采样权重 ×{a.extra_weight:g}")
    sample_p = None
    if a.extra_root and a.extra_weight != 1.0:
        w = np.ones(len(train_raw)); w[n_main:] = a.extra_weight
        sample_p = w / w.sum()

    def prep(ss):
        o = [copy.deepcopy(s) for s in ss]
        for s in o:
            norm.transform(s)
        return o
    train, val, xfer = prep(train_raw), prep(val_raw), prep(xfer_raw)
    print(f"[pf] train {len(train)} / val {len(val)} / transfer {len(xfer)}")

    dc_dim = 0 if not a.dc_theta else {"raw": 2, "bounded": 4, "lossy": 3, "lossyg": 3, "lossy2": 2, "lossyv": 3,
                                       "lossybv": 3}[a.dc_encoding]
    side_dim = (PF_SIDE_DIM + dc_dim + (3 if a.v_hint else 0)) if a.view == "group" else 0
    reasoner = PFReasoner(d=cfg.d, n_queries=a.queries, n_layers=a.layers,
                          side_dim=side_dim,
                          gnn_layers=a.gnn_layers, theta_head=a.theta_head,
                          edge_side_dim=EDGE_SIDE_DIM if a.theta_edge else 0,
                          theta_edge=a.theta_edge, theta_wrap=a.theta_wrap,
                          edge_dc_residual=not a.no_edge_dc_residual,
                          edge_head_zero_init=a.edge_zero_init,
                          edge_head_local=a.edge_head_local,
                          refine_steps=a.refine_steps, v_hint=a.v_hint, refine_qv=a.refine_qv,
                          refine_ls=a.refine_ls).to(dev)
    print(f"[pf] 推理器参数 {reasoner.num_params()/1e6:.2f}M(D9 目标 1e6 量级)"
          + (f";杠杆:tw={a.theta_loss_weight:g} th={a.theta_head} gnn={a.gnn_layers}"
             if (fw or a.theta_head or a.gnn_layers) else "")
          + (f";θ 边势重建 edge_w={a.edge_loss_weight:g} dc_residual={not a.no_edge_dc_residual}"
             if a.theta_edge else "")
          + (";θ 输出折回 (−π,π] + 周期监督" if a.theta_wrap else "")
          + (f";E25 θ 铰链 hinge_w={a.theta_hinge_weight:g}(对手=自家边基线,图级非对称)"
             if a.theta_hinge_weight else "")
          + (f";E12 修正步 K={a.refine_steps} qv={a.refine_qv} negb={a.refine_negb}" if a.refine_steps else "")
          + (f";E13 aux={a.refine_aux_weight:g} hint_dropout={a.hint_dropout:g} refine_dropout={a.refine_dropout:g} ls={a.refine_ls}"
             if (a.refine_aux_weight or a.hint_dropout or a.refine_dropout or a.refine_ls) else ""))
    trainable = list(reasoner.parameters()) + (list(enc.parameters()) if train_enc else [])
    opt = torch.optim.Adam(trainable, lr=a.lr)
    use_amp = dev.type == "cuda"
    amp_dtype = torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    rng = np.random.default_rng(a.seed)
    start, best_val, bad = 0, float("inf"), 0
    ckpts = sorted((out / "ckpt").glob("step_*.pt"))
    if a.resume and ckpts:
        rck = torch.load(ckpts[-1], map_location=dev, weights_only=False)
        ck_view, ck_enc = rck.get("view", "field"), rck.get("encoder", a.encoder)
        if ck_view != a.view:
            raise SystemExit(f"[pf] 检查点口径 {ck_view} 与 --view {a.view} 不符")
        if ck_enc != a.encoder:
            raise SystemExit(f"[pf] 检查点编码器 {ck_enc} 与 --encoder {a.encoder} 不符"
                             f"(会破坏 5.3 组1/组2 可比性)")
        ck_pw = (rck.get("phys_weight", 0.0), rck.get("phys_defeq_weight", 0.0))
        if ck_pw != (a.phys_weight, a.phys_defeq_weight):
            raise SystemExit(f"[pf] 检查点残差权重 {ck_pw} 与 --phys-weight/"
                             f"--phys-defeq-weight ({a.phys_weight}, {a.phys_defeq_weight}) "
                             f"不符(损失定义中途变更会污染扫描结论)")
        ck_cfg = rck.get("config", {})
        ck_lv = (rck.get("theta_loss_weight", 1.0), ck_cfg.get("theta_head", False),
                 ck_cfg.get("gnn_layers", 0), ck_cfg.get("dc_theta", False),
                 ck_cfg.get("theta_edge", False), ck_cfg.get("theta_wrap", False),
                 ck_cfg.get("edge_dc_residual", True), ck_cfg.get("edge_loss_weight", 1.0),
                 ck_cfg.get("edge_head_zero_init", False), ck_cfg.get("edge_base", "acv"),
                 ck_cfg.get("theta_hinge_weight", 0.0),
                 ck_cfg.get("edge_bias_weight", 0.0), ck_cfg.get("edge_head_local", False),
                 ck_cfg.get("refine_steps", 0), ck_cfg.get("v_hint", False),
                 ck_cfg.get("refine_qv", False), ck_cfg.get("refine_negb", "sign"),
                 ck_cfg.get("refine_aux_weight", 0.0), ck_cfg.get("hint_dropout", 0.0), ck_cfg.get("refine_dropout", 0.0),
                 ck_cfg.get("refine_ls", False),
                 ck_cfg.get("shunt_zero_harmless", False), ck_cfg.get("phys_rownorm", False),
                 (ck_cfg.get("dc_encoding", "raw") if ck_cfg.get("dc_theta", False) else None),
                 # 2026-08-21 15:00 之前写出的检查点没有这三个字段:缺失时按本次取值(不拒绝续训)
                 list(ck_cfg.get("extra_root", [str(Path(p)) for p in a.extra_root])),
                 list(ck_cfg.get("val_holdout", [str(Path(p)) for p in a.val_holdout])),
                 ck_cfg.get("extra_weight", a.extra_weight))
        now_lv = (a.theta_loss_weight, a.theta_head, a.gnn_layers, a.dc_theta,
                  a.theta_edge, a.theta_wrap, not a.no_edge_dc_residual, a.edge_loss_weight,
                  a.edge_zero_init, a.edge_base,
                  a.theta_hinge_weight,
                  a.edge_bias_weight, a.edge_head_local, a.refine_steps, a.v_hint,
                  a.refine_qv, a.refine_negb, a.refine_aux_weight, a.hint_dropout, a.refine_dropout, a.refine_ls,
                  a.shunt_zero_harmless, a.phys_rownorm,
                  (a.dc_encoding if a.dc_theta else None),
                  [str(Path(p)) for p in a.extra_root], [str(Path(p)) for p in a.val_holdout], a.extra_weight)
        if ck_lv != now_lv:
            raise SystemExit(f"[pf] 检查点杠杆配置 {ck_lv} 与本次 {now_lv} 不符")
        reasoner.load_state_dict(rck["reasoner_state"])
        if train_enc:
            if "encoder_state" not in rck:
                raise SystemExit("[pf] e2e 续训要求检查点含 encoder_state,旧检查点不可续")
            enc.load_state_dict(rck["encoder_state"])
        opt.load_state_dict(rck["optimizer_state"])
        start, best_val, bad = rck["step"], rck["best_val"], rck["bad"]
        rng.bit_generator.state = rck["rng"]
        print(f"[pf] 续训自 step {start}")

    warm = max(1, int(0.05 * a.steps))
    hist = []
    for step in range(start, a.steps):
        lr = (a.lr * (step + 1) / warm if step < warm else
              a.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, a.steps - warm - 1))))
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = rng.choice(len(train), size=min(a.batch, len(train)), replace=False, p=sample_p)
        datas = [_pf_data(train[int(i)], cfg, a.view,
                          raw=train_raw[int(i)] if use_raw else None,
                          dc=dc_mode, phys=use_phys, edge=a.theta_edge,
                          vhint=a.v_hint, vstat=vstat, negb=a.refine_negb, shz=a.shunt_zero_harmless,
                          edge_base=a.edge_base) for i in idx]
        if a.hint_dropout or a.refine_dropout:
            apply_train_dropout(datas, rng, a.hint_dropout, a.refine_dropout)
        batch = Batch.from_data_list(datas).to(dev)
        opt.zero_grad()
        pred, loss = _forward(enc, reasoner, batch, use_amp, amp_dtype, dev, field_weights=fw, dn=dn,
                              train_enc=train_enc, edge_weight=a.edge_loss_weight,
                              edge_bias_weight=a.edge_bias_weight, refine_aux_weight=a.refine_aux_weight,
                              hinge_weight=a.theta_hinge_weight)
        if use_phys:
            dp, dq, masks = _phys_terms(pred, batch, dn)
            rs = pf_phys_row_scale(batch.phys_index, batch.phys_b, dp.shape[0]) if a.phys_rownorm else None
            ploss, _pm = pf_phys_loss(dp, dq, masks, a.phys_weight, a.phys_defeq_weight, row_scale=rs)
            loss = loss + ploss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()
        hist.append(float(loss.detach()))
        if (step + 1) % max(1, a.steps // 20) == 0:
            hs = getattr(reasoner, "hinge_stats", None) if a.theta_hinge_weight else None
            print(f"[pf] step {step+1}/{a.steps} lr={lr:.2e} loss={hist[-1]:.5f}"
                  + (f" hinge={float(hs[0]):.5f} act={int(hs[1])}/{hs[2]}" if hs is not None else "")
                  + (f" refine={getattr(reasoner, 'refine_stats', {})} alpha={[round(float(x), 3) for x in reasoner.refine_alpha]}"
                     + (f" beta={[round(float(x), 3) for x in reasoner.refine_beta]}" if a.refine_qv else "")
                     if a.refine_steps else ""), flush=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            mae = eval_pu_mae(enc, reasoner, val, cfg, norm, dev, use_amp, amp_dtype,
                              view=a.view, raws=val_raw if use_raw else None, dn=dn,
                              dc=dc_mode, phys=use_phys, edge=a.theta_edge,
                              vhint=a.v_hint, vstat=vstat, negb=a.refine_negb, shz=a.shunt_zero_harmless,
                              edge_base=a.edge_base)
            score = mae["theta"] + 10 * mae["v"]        # 单标量早停锚(θ 主 + v 加权)
            print(f"[pf] eval step {step+1}: " +
                  " ".join(f"{f}={mae[f]:.5f}" for f in PF_FIELDS) +
                  (f" |ΔPQ|clean={mae['phys_clean']:.4f} def={mae['phys_def']:.4f}"
                   if use_phys else "") +
                  f" score={score:.5f} best={best_val:.5f}", flush=True)
            (out / "ckpt").mkdir(parents=True, exist_ok=True)
            state = {"reasoner_state": reasoner.state_dict(), "optimizer_state": opt.state_dict(),
                     **({"encoder_state": enc.state_dict()} if train_enc else {}),
                     "step": step + 1, "best_val": min(best_val, score), "bad": bad,
                     "rng": rng.bit_generator.state, "encoder": a.encoder, "view": a.view,
                     "phys_weight": a.phys_weight, "phys_defeq_weight": a.phys_defeq_weight,
                     "theta_loss_weight": a.theta_loss_weight,
                     "config": {"queries": a.queries, "layers": a.layers, "d": cfg.d,
                                "theta_head": a.theta_head, "gnn_layers": a.gnn_layers,
                                "dc_theta": a.dc_theta, "dc_encoding": a.dc_encoding,
                                "theta_edge": a.theta_edge, "theta_wrap": a.theta_wrap,
                                "edge_dc_residual": not a.no_edge_dc_residual,
                                "edge_head_zero_init": a.edge_zero_init, "edge_base": a.edge_base,
                                "theta_hinge_weight": a.theta_hinge_weight,
                                "edge_loss_weight": a.edge_loss_weight,
                                "edge_bias_weight": a.edge_bias_weight,
                                "edge_head_local": a.edge_head_local,
                                "refine_steps": a.refine_steps, "v_hint": a.v_hint,
                                "refine_qv": a.refine_qv, "refine_negb": a.refine_negb,
                                "refine_aux_weight": a.refine_aux_weight, "hint_dropout": a.hint_dropout,
                                "refine_dropout": a.refine_dropout, "refine_ls": a.refine_ls,
                                "shunt_zero_harmless": a.shunt_zero_harmless, "phys_rownorm": a.phys_rownorm,
                                "phys_tap_families": _tap_fams, "seed": a.seed,
                                "v1_dir": str(a.v1_dir),   # E26(§29):编码器底座凭据(评估侧必须同底座;防"假凭据")
                                "extra_root": [str(Path(p)) for p in a.extra_root],
                                "val_holdout": [str(Path(p)) for p in a.val_holdout],
                                "extra_weight": a.extra_weight, "n_train": len(train), "n_val": len(val),
                                "lr": a.lr, "batch": a.batch, "theta_loss_weight": a.theta_loss_weight}}
            tmp = out / "ckpt" / f"step_{step+1:06d}.pt.tmp"
            torch.save(state, tmp); tmp.replace(out / "ckpt" / f"step_{step+1:06d}.pt")
            if score < best_val - 1e-6:
                best_val, bad = score, 0
                torch.save(state, out / "ckpt" / "best.pt.tmp")
                (out / "ckpt" / "best.pt.tmp").replace(out / "ckpt" / "best.pt")
            else:
                bad += 1
                if bad >= a.patience:
                    print(f"[pf] 早停于 step {step+1}")
                    break

    # ---- 终评:best 检查点;任务掩码零学习阶梯 + 技巧分 ----
    bck = torch.load(out / "ckpt" / "best.pt", map_location=dev, weights_only=False)
    b_view, b_enc = bck.get("view", "field"), bck.get("encoder", a.encoder)
    if b_view != a.view or b_enc != a.encoder:
        raise SystemExit(f"[pf] best.pt(口径 {b_view}/编码器 {b_enc})与 "
                         f"--view {a.view}/--encoder {a.encoder} 不符,请换 --out 或清理旧检查点")
    b_pw = (bck.get("phys_weight", 0.0), bck.get("phys_defeq_weight", 0.0))
    if b_pw != (a.phys_weight, a.phys_defeq_weight):
        raise SystemExit(f"[pf] best.pt 残差权重 {b_pw} 与本次配置不符,请换 --out")
    b_cfg = bck.get("config", {})
    b_lv = (bck.get("theta_loss_weight", 1.0), b_cfg.get("theta_head", False),
            b_cfg.get("gnn_layers", 0), b_cfg.get("dc_theta", False),
            b_cfg.get("theta_edge", False), b_cfg.get("theta_wrap", False),
            b_cfg.get("edge_dc_residual", True), b_cfg.get("edge_loss_weight", 1.0),
            b_cfg.get("edge_head_zero_init", False), b_cfg.get("edge_base", "acv"),
            b_cfg.get("theta_hinge_weight", 0.0),
            b_cfg.get("edge_bias_weight", 0.0), b_cfg.get("edge_head_local", False),
            b_cfg.get("refine_steps", 0), b_cfg.get("refine_qv", False), b_cfg.get("refine_negb", "sign"),
            b_cfg.get("refine_aux_weight", 0.0), b_cfg.get("hint_dropout", 0.0), b_cfg.get("refine_dropout", 0.0),
            b_cfg.get("refine_ls", False), b_cfg.get("v_hint", False),
            b_cfg.get("shunt_zero_harmless", False), b_cfg.get("phys_rownorm", False),
            (b_cfg.get("dc_encoding", "raw") if b_cfg.get("dc_theta", False) else None))
    if b_lv != (a.theta_loss_weight, a.theta_head, a.gnn_layers, a.dc_theta,
                a.theta_edge, a.theta_wrap, not a.no_edge_dc_residual, a.edge_loss_weight,
                a.edge_zero_init, a.edge_base,
                a.theta_hinge_weight,
                a.edge_bias_weight, a.edge_head_local, a.refine_steps, a.refine_qv, a.refine_negb,
                a.refine_aux_weight, a.hint_dropout, a.refine_dropout, a.refine_ls, a.v_hint,
                a.shunt_zero_harmless, a.phys_rownorm,
                (a.dc_encoding if a.dc_theta else None)):
        raise SystemExit(f"[pf] best.pt 杠杆配置 {b_lv} 与本次不符,请换 --out")
    reasoner.load_state_dict(bck["reasoner_state"])
    if train_enc:
        if "encoder_state" not in bck:
            raise SystemExit("[pf] e2e 终评要求 best.pt 含 encoder_state")
        enc.load_state_dict(bck["encoder_state"])
        enc.eval()
    mae_in = eval_pu_mae(enc, reasoner, val, cfg, norm, dev, use_amp, amp_dtype, view=a.view,
                         raws=val_raw if use_raw else None, dn=dn,
                         dc=dc_mode, phys=use_phys, edge=a.theta_edge, vhint=a.v_hint, vstat=vstat, negb=a.refine_negb,
                         shz=a.shunt_zero_harmless, edge_base=a.edge_base)
    mae_out = eval_pu_mae(enc, reasoner, xfer, cfg, norm, dev, use_amp, amp_dtype, view=a.view,
                          raws=xfer_raw if use_raw else None, dn=dn,
                          dc=dc_mode, phys=use_phys, edge=a.theta_edge, vhint=a.v_hint, vstat=vstat, negb=a.refine_negb,
                          shz=a.shunt_zero_harmless, edge_base=a.edge_base)
    lad_in = pf_zero_learning_baseline(val_raw)
    lad_out = pf_zero_learning_baseline(xfer_raw)
    dc_row = None
    if a.dc_theta:
        # 预注册基线行:DC-θ 裸值 MAE(θ 任务掩码位置;模型对 DC 的增量透明,门禁仍挂 L2)
        from gridfm.schema.fields import NODE_TABLE as _NT
        from gridfm.train.masking import _pf_typed_mask as _mask
        th_c = _NT.offset_of("theta")
        dc_row = {}
        for name, rr_ in (("in", val_raw), ("out", xfer_raw)):
            errs, errs_l, errs_v, n_fb = [], [], [], 0
            for s_ in rr_:
                _mf, _tg, lmr = _mask(s_.node_feat, s_.node_missing)
                d_ = dc_theta(s_)
                m_ = lmr[:, th_c] & (d_[:, 1] > 0.5)
                if m_.any():
                    errs.append(np.abs(d_[m_, 0] - s_.node_feat[m_, th_c]))
                if a.dc_encoding.startswith("lossy"):
                    dl_ = dc_theta_lossy(s_, final_check=(a.dc_encoding == "lossyg"))
                    ml_ = lmr[:, th_c] & (dl_[:, 1] > 0.5)
                    if ml_.any():
                        errs_l.append(np.abs(dl_[ml_, 0] - s_.node_feat[ml_, th_c]))
                        n_fb += int(dl_[ml_, 2].max() > 0.5)     # 只计有 θ 监督的样本(不可解样本不计)
                if a.dc_encoding == "lossyv":
                    dv_ = dc_theta_lossy_v(s_, tap=getattr(s_, "branch_tap", None),
                                           shift=getattr(s_, "branch_shift_rad", None),
                                           shunt_zero_harmless=a.shunt_zero_harmless)
                    mv_ = lmr[:, th_c] & (dv_[:, 1] > 0.5)
                    if mv_.any():
                        errs_v.append(np.abs(dv_[mv_, 0] - s_.node_feat[mv_, th_c]))
            dc_row[name] = float(np.concatenate(errs).mean()) if errs else float("nan")
            if a.dc_encoding.startswith("lossy"):
                dc_row[name + "_lossy"] = float(np.concatenate(errs_l).mean()) if errs_l else float("nan")
                dc_row[name + "_lossy_fallback_samples"] = n_fb
            if a.dc_encoding == "lossyv":
                dc_row[name + "_dcv"] = float(np.concatenate(errs_v).mean()) if errs_v else float("nan")
        print(f"[pf] DC-θ 裸值基线:内 {dc_row['in']:.4f} / 外 {dc_row['out']:.4f}(标幺 MAE)"
              + (f";带网损 DC:内 {dc_row['in_lossy']:.4f} / 外 {dc_row['out_lossy']:.4f}"
                 f"(回退样本 {dc_row['in_lossy_fallback_samples']} / {dc_row['out_lossy_fallback_samples']})"
                 if a.dc_encoding.startswith("lossy") else "")
              + (f";dcv 裸值:内 {dc_row['in_dcv']:.4f} / 外 {dc_row['out_dcv']:.4f}"
                 if a.dc_encoding == "lossyv" else ""))

    print(f"\n[任务掩码阶梯(D15 协议,与 8.4 随机课程阶梯不可混排)]")
    print(f"{'':<14}{'θ内':>9}{'θ外':>9}{'v内':>10}{'v外':>10}{'pg内':>10}{'qg内':>10}")
    for lvl in ("L0", "L1", "L2"):
        print(f"{lvl:<14}{lad_in['theta'][lvl]:>9.4f}{lad_out['theta'][lvl]:>9.4f}"
              f"{lad_in['v'][lvl]:>10.5f}{lad_out['v'][lvl]:>10.5f}"
              f"{lad_in['pg'][lvl]:>10.5f}{lad_in['qg'][lvl]:>10.5f}")
    print(f"{'模型':<13}{mae_in['theta']:>9.4f}{mae_out['theta']:>9.4f}"
          f"{mae_in['v']:>10.5f}{mae_out['v']:>10.5f}"
          f"{mae_in['pg']:>10.5f}{mae_in['qg']:>10.5f}")
    skills = {}
    for f in PF_FIELDS:
        si = lad_in[f]["L2"] / mae_in[f] if mae_in[f] > 0 else float("nan")
        so = lad_out[f]["L2"] / mae_out[f] if mae_out[f] > 0 else float("nan")
        skills[f] = {"skill_in": si, "skill_out": so,
                     "survival": so / si if (np.isfinite(si) and si > 0) else float("nan")}
        print(f"{f:>6}: 技巧分 内 {si:.2f} / 外 {so:.2f} / 存活 "
              f"{skills[f]['survival']:.2f}")

    rep = {"encoder": a.encoder, "view": a.view, "steps": a.steps, "best_step": bck["step"],
           "dc_theta": a.dc_theta, "dc_theta_raw_mae": dc_row,
           "theta_edge": a.theta_edge, "theta_wrap": a.theta_wrap,
           "edge_loss_weight": a.edge_loss_weight, "edge_dc_residual": not a.no_edge_dc_residual,
           "edge_bias_weight": a.edge_bias_weight, "edge_head_local": a.edge_head_local,
           "seed": a.seed, "lr": a.lr, "batch": a.batch, "theta_loss_weight": a.theta_loss_weight,
           "theta_head": a.theta_head, "gnn_layers": a.gnn_layers, "dc_encoding": a.dc_encoding,
           "refine_steps": a.refine_steps, "refine_qv": a.refine_qv, "refine_negb": a.refine_negb, "v_hint": a.v_hint,
           "refine_aux_weight": a.refine_aux_weight, "hint_dropout": a.hint_dropout, "refine_dropout": a.refine_dropout,
           "refine_ls": a.refine_ls,
           "shunt_zero_harmless": a.shunt_zero_harmless, "phys_rownorm": a.phys_rownorm,
           "refine_alpha": ([float(x) for x in reasoner.refine_alpha] if a.refine_steps else None),
           "refine_beta": ([float(x) for x in reasoner.refine_beta] if a.refine_qv else None),
           "n_train": len(train), "n_val": len(val), "n_transfer": len(xfer),
           "extra_root": list(a.extra_root), "val_holdout": list(a.val_holdout), "extra_weight": a.extra_weight,
           "n_extra_train": len(extra_train), "n_extra_val": len(extra_val),
           "phys_weight": a.phys_weight, "phys_defeq_weight": a.phys_defeq_weight,
           "reasoner_params": reasoner.num_params(),
           "encoder_params": n_enc, "encoder_trainable": train_enc,
           "mae_in_pu": mae_in, "mae_out_pu": mae_out,
           "ladder_task_in": lad_in, "ladder_task_out": lad_out, "skills": skills,
           "train_loss_first_last": [hist[0] if hist else None, hist[-1] if hist else None]}
    (out / "pf_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"[pf] 报告 {out / 'pf_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
