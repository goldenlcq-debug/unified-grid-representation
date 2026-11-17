# -*- coding: utf-8 -*-
"""θ 边势重建(2026-08-21,5060 线;推理器侧,D13):逐支路 Δθ 预测 → 图上最小二乘积分 → 折回 (−π, π]。

**动机(2026-08-21 当前语料逐节点诊断,见 docs/结果档)**:
1. 存档 θ = deg2rad(PSASP ±180° 截断值),未 unwrap:transfer 1000 节点档 58.6% 监督节点被 ±π
   缠绕,解缠绕后物理相角 p90 = 7.5 rad。绝对 θ 在大网上不是可外推的回归目标。
2. 逐支路 |Δθ| 跨规模同分布(150→1000 节点均值 0.04~0.056 rad、p99 0.17~0.27),且 DC 的
   **局部**精度极好(|Δθ_dc − Δθ| 每支路 0.013~0.027 rad),"DC 大网崩坏"= 系统性局部小误差沿
   30~140 跳路径累积 + 真值缠绕。
因此:让推理器学局部量 Δθ_ij(以 Δθ_dc 为残差基线),θ 由"势重建"(最小二乘积分,平衡机给定角
为 Dirichlet 边界)得到,输出层折回 (−π, π] 与存档真值同域。评估口径(原样 MAE 对存档 θ)不变。

**D15 / 红线自查**:
- 边侧特征只读 拓扑、静态 r/x/b_half、投退 mark、给定注入与平衡机给定角(经 dc_theta,
  2026-07-12 已批准的计算型特征),不碰被监督量与支路潮流;单测锁定(扰动被监督量不改特征);
- Δθ 真值 = wrap(θ_j − θ_i) 只在损失侧使用;
- 边头 = 逐边共享 MLP;势重建 = 现算线性求解(无可学习 N 依赖参数,等价于图上的固定算子);
- 不动编码器、不动 normalizer、不改 pf_input_view / pf_side_features。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .. import perf_cache
from ..schema.fields import assert_branch_mark, BRANCH_TABLE, BRANCH_TYPES, NODE_TABLE, Group
from ..schema.sample import GridSample, group_bit_index

TWO_PI = 2.0 * np.pi
_SLACK = 2
_TH = NODE_TABLE.offset_of("theta")

# 逐有向边侧特征列(与 graph.sample_to_data 的对称化布局对齐:前 M 条 i→j,后 M 条 j→i)
EDGE_SIDE_FIELDS = ("use", "dtheta_dc", "dc_valid", "log10_x", "log10_r", "b_half", "p_dc")
EDGE_SIDE_DIM = len(EDGE_SIDE_FIELDS)
DTH_DC_COL = EDGE_SIDE_FIELDS.index("dtheta_dc")


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """折回 [−π, π];对 x 的梯度恒为 1(round 项截断梯度)。"""
    return x - TWO_PI * torch.round(x / TWO_PI).detach()


def wrap_np(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) + np.pi) % TWO_PI - np.pi


def use_edges(raw: GridSample) -> np.ndarray:
    """Δθ 有物理定义的支路:在运、非并补、|x|>1e-9、两端非 NULL(取舍与 dc_theta 同款再加 NULL 排除)。"""
    boff, noff = BRANCH_TABLE.offset_of, NODE_TABLE.offset_of
    if raw.n_branches == 0:
        return np.zeros(0, dtype=bool)
    x = raw.branch_feat[:, boff("x")].astype(np.float64)
    mark = raw.branch_feat[:, boff("mark")]
    assert_branch_mark(mark, "pf_theta_edge.use_edges")
    oh = raw.branch_feat[:, boff("branch_type"):boff("branch_type") + 6]
    bt = np.asarray(BRANCH_TYPES)[np.argmax(oh, axis=1)]
    null = raw.node_feat[:, noff("is_null")] > 0.5
    i, j = raw.edge_index
    return (mark == 1) & (bt != "shunt_comp") & (np.abs(x) > 1e-9) & ~null[i] & ~null[j]


def pf_edge_side_features(raw: GridSample, dc: Optional[np.ndarray] = None,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """逐有向边侧特征 [2M, EDGE_SIDE_DIM] f32 与 use 位 [2M] bool(原始样本;不进编码器)。

    dc = pf_physics.dc_theta(raw)[N,2](可传入复用);None 时现算。
    列:use、Δθ_dc(=θ_dc[dst]−θ_dc[src],两端 valid 且 use 时有效,否则 0)、该有效位、
    log10|x|、log10 r(裁 [−6,6])、b_half(裁 ±10)、DC 支路潮流 Δθ_dc/x(裁 ±10)。
    反向边 Δθ_dc 与 p_dc 取负,其余相同。
    """
    from .pf_physics import dc_theta
    boff = BRANCH_TABLE.offset_of
    m = raw.n_branches
    if m == 0:
        return np.zeros((0, EDGE_SIDE_DIM), dtype=np.float32), np.zeros(0, dtype=bool)
    if dc is None:
        dc = dc_theta(raw)
    use = use_edges(raw)
    i, j = raw.edge_index[0].astype(np.int64), raw.edge_index[1].astype(np.int64)
    x = raw.branch_feat[:, boff("x")].astype(np.float64)
    r = raw.branch_feat[:, boff("r")].astype(np.float64)
    bh = raw.branch_feat[:, boff("b_half")].astype(np.float64)
    ok = use & (dc[i, 1] > 0.5) & (dc[j, 1] > 0.5)
    dth = np.where(ok, dc[j, 0].astype(np.float64) - dc[i, 0].astype(np.float64), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_dc = np.where(ok & (np.abs(x) > 1e-9), dth / np.where(np.abs(x) > 1e-9, x, 1.0), 0.0)
    p_dc = np.clip(p_dc, -10.0, 10.0)
    lx = np.clip(np.log10(np.maximum(np.abs(x), 1e-6)), -6.0, 6.0)
    lr = np.clip(np.log10(np.maximum(np.abs(r), 1e-6)), -6.0, 6.0)
    bhc = np.clip(bh, -10.0, 10.0)
    fwd = np.stack([use.astype(np.float64), dth, ok.astype(np.float64), lx, lr, bhc, p_dc], axis=1)
    rev = fwd.copy()
    rev[:, DTH_DC_COL] *= -1.0
    rev[:, EDGE_SIDE_FIELDS.index("p_dc")] *= -1.0
    es = np.concatenate([fwd, rev], axis=0).astype(np.float32)
    eu = np.concatenate([use, use], axis=0)
    return es, eu


REFINE_NEGB = ("zero", "sign", "abs")


def pf_edge_susceptance(raw: GridSample, negb: str = "zero") -> np.ndarray:
    """逐有向边电纳 b_k = 1/x_k [2M] f32(不可用支路给 0),与 graph.sample_to_data 的对称化同序。

    失配修正步(§13.1)的 B′ 权重。只读静态 x 与投退位 —— 与 `dc_theta` 同类的给定量,无泄漏。
    negb = 串补(x<0)支路的口径(2026-08-22 第 3 轮,探针 `probe_bprime_conn` 实测 ACTIVSg 10k/25k/70k 各有
    193/503/1365 条在运串补):
      "zero"(默认,旧行为):按 0 计 —— B′ 保持半正定,但只靠串补接到参考角的节点/小簇在 B′ 里成为无 Dirichlet 的孤块
                 (10k 163 / 25k 468 / 70k 1292 个节点),调用方须用 `pf_refine_free` 把它们按 Dirichlet 跳过;
      "sign":带符号 1/x(FDPF 标准 B′,XB 格式);"abs":|1/x|。两者保持连通,直接 LU 不要求半正定。
    """
    boff = BRANCH_TABLE.offset_of
    if negb not in REFINE_NEGB:
        raise ValueError(f"negb 应为 {REFINE_NEGB},得到 {negb!r}")
    if raw.n_branches == 0:
        return np.zeros(0, dtype=np.float32)
    x = raw.branch_feat[:, boff("x")].astype(np.float64)
    use = use_edges(raw)
    b = np.where(use & (np.abs(x) > 1e-9), 1.0 / np.where(np.abs(x) > 1e-9, x, 1.0), 0.0)
    if negb == "zero":
        # 串补(x<0)会让 B′ 失去半正定性 → 该支路按 0 计(只影响修正步的权重,不影响任何监督量)
        b = np.where(b > 0, b, 0.0)
    elif negb == "abs":
        b = np.abs(b)
    return np.concatenate([b, b]).astype(np.float32)


def pf_refine_free(raw: GridSample, edge_b: np.ndarray, ref_mask: np.ndarray, reach_mask: np.ndarray) -> np.ndarray:
    """E12 修正步的自由节点集 [N] bool:经 **b≠0** 支路与某个有效参考角(ref_mask)连通、且 reach、非参考。

    其余节点(参考角、不可达、以及只靠被置 0 的串补支路接入的"孤块")在修正步里一律 Dirichlet(Δθ ≡ 0),
    这样 B′_ff 一定非奇异(每个自由分量都含 Dirichlet 节点),不需要任何岭。只读静态拓扑与给定量,无泄漏。
    edge_b 取 `pf_edge_susceptance` 的前 M 项或全部 2M 项均可(只看非零性)。
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    n = raw.n_nodes
    m = raw.n_branches
    if m == 0 or not np.any(ref_mask):
        return np.zeros(n, dtype=bool)
    b = np.asarray(edge_b)[:m]
    i, j = raw.edge_index[0].astype(np.int64), raw.edge_index[1].astype(np.int64)
    k = b != 0
    A = sp.coo_matrix((np.ones(int(k.sum()) * 2), (np.r_[i[k], j[k]], np.r_[j[k], i[k]])), shape=(n, n)).tocsr()
    nc, comp = connected_components(A, directed=False)
    has_ref = np.bincount(comp[np.asarray(ref_mask, dtype=bool)], minlength=nc) > 0
    return has_ref[comp] & np.asarray(reach_mask, dtype=bool) & ~np.asarray(ref_mask, dtype=bool)


def pf_theta_ref(raw: GridSample, dc: Optional[np.ndarray] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """势重建的节点侧常量(原始样本):
    theta_ref[N] f32 = 平衡机给定角(其余 0);ref_mask[N] = 平衡机且 STATE 非 missing 且非 NULL;
    reach_mask[N] = 经在运支路可达某平衡机(= dc_theta valid 位);
    theta_avail[N] = 存档 θ 可用(非 NULL、STATE 非 missing;给定或被监督,仅供损失侧 Δθ 真值掩码)。
    """
    from .pf_physics import dc_theta
    noff = NODE_TABLE.offset_of
    if dc is None:
        dc = dc_theta(raw)
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    tcol = noff("node_type")
    types = np.argmax(raw.node_feat[:, tcol:tcol + 3], axis=1)
    null = raw.node_feat[:, noff("is_null")] > 0.5
    smiss = raw.node_missing[:, group_bit_index(Group.STATE)]
    ref_mask = (types == _SLACK) & ~null & ~smiss
    theta_ref = np.where(ref_mask, raw.node_feat[:, _TH], 0.0).astype(np.float32)
    # reach = 经 use 支路与某个 **有效参考角**(ref_mask)连通(与 dc_theta 的 valid 位不同:后者不看
    # 平衡机 STATE missing;2026-08-21 审查:两者不一致时势重建会给出零均值势而不回退)
    n = raw.n_nodes
    reach = np.zeros(n, dtype=bool)
    use = use_edges(raw)
    if ref_mask.any() and use.any():
        i, j = raw.edge_index[0][use].astype(np.int64), raw.edge_index[1][use].astype(np.int64)
        A = sp.coo_matrix((np.ones(i.size * 2), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
        nc, comp = connected_components(A, directed=False)
        has_ref = np.bincount(comp[ref_mask], minlength=nc) > 0
        reach = has_ref[comp] & ~null & (dc[:, 1] > 0.5)
    avail = ~null & ~smiss
    return theta_ref, ref_mask, reach, avail


DENSE_MAX_NODES = 4096     # 单图节点数超过此值改走稀疏 CG(稠密法方程 N² fp64:4096 → 134 MB/图)
# 2026-08-21 第 2 轮修法(结果档 §3.11 第 4 项):稠密法方程按 **整批** 分配 A[B,nmax,nmax],
# 只看单图 nmax 会漏掉"batch 大 × 图中等"的组合——batch 8 含 2000 节点图时 A 就有 244 MB、
# torch.linalg.solve 的工作区再翻一倍(≈0.5 GB/步,实测常驻 + 峰值与审查估计一致)。
# 故 auto 判据加一条整批元素数上限。阈值 24e6 fp64 = 192 MB:
#   当前语料 batch 16 × 1000 节点 = 16.0e6 → 仍走稠密(合成语料各臂逐位不变);
#   大语料   batch  8 × 2000 节点 = 32.0e6 → 转稀疏(省 ~0.5 GB/步)。
# 两条路径解的是**同一个**线性系统(稠密 LU vs scipy 稀疏 LU,均 fp64),实测 |Δθ| ≤ 5.96e-8,
# 小于该量级下 fp32 输出的分辨率(6.2e-8),即对外可见结果不变。
DENSE_MAX_BATCH_ELEMS = 24_000_000


def _laplacian_matvec(x_full, src, dst, w):
    """y = L_w x(图拉普拉斯,按边 index_add,纯 torch;D17 合规)。"""
    diff = w * (x_full[src] - x_full[dst])
    y = torch.zeros_like(x_full)
    y.index_add_(0, src, diff)
    y.index_add_(0, dst, -diff)
    return y


def _pcg_solve(rhs, src, dst, w, free, diag, ridge_vec, iters, tol, check_every: int = 25):
    """在自由节点上解 (L_w + diag(ridge))_ff x = rhs,Jacobi 预条件 CG(fp64)。rhs/返回值为全节点向量(固定节点为 0)。

    2026-08-21 修订:原实现每轮两次 GPU→CPU 同步(pAp / 残差判停)+ tol 1e-10,真实电网(直径 ~100 跳)需数千轮,
    万节点图一步训练 >20 s。改为每 check_every 轮同步一次判停、pAp 用张量运算保护。"""
    n = rhs.numel()
    x = torch.zeros_like(rhs)
    free_f = free.to(rhs.dtype)
    def A(v):
        v = v * free_f
        return (_laplacian_matvec(v, src, dst, w) + ridge_vec * v) * free_f
    r = (rhs - A(x)) * free_f
    minv = torch.where(free, 1.0 / diag.clamp_min(1e-30), torch.zeros_like(diag))
    z = minv * r
    p = z.clone()
    rz = torch.dot(r, z)
    b_norm = torch.linalg.norm(rhs * free_f).clamp_min(1e-30)
    tiny = torch.finfo(rhs.dtype).tiny
    for k in range(1, iters + 1):
        Ap = A(p)
        pAp = torch.dot(p, Ap).clamp_min(tiny)        # 非正(数值退化)时 alpha→巨大?用 clamp 保安全,下方判停兜底
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        z = minv * r
        rz_new = torch.dot(r, z)
        p = z + (rz_new / rz.clamp_min(tiny)) * p
        rz = rz_new
        if k % check_every == 0 or k == iters:
            res = float(torch.linalg.norm(r))
            if not np.isfinite(res) or res <= tol * float(b_norm):
                break
    return x * free_f


SPARSE_SOLVER = "splu"      # "splu" = CPU scipy 直接稀疏 LU(默认);"pcg" = GPU Jacobi-PCG(备用)


class _CpuLU:
    """CPU 上 scipy.sparse.linalg.splu 封装:自由节点子系统 (L_w + diag(ridge))_ff 的 LU,前向/反向各解一次。

    2026-08-21 实测:GPU 上的 PCG 每步需数千个微小 kernel,多进程分时下一个 10k 图的 batch 前向 40 s;
    CPU 直接 LU 10k 节点 ~0.1 s 且与 GPU 争用无关。只用 numpy/scipy(D17 合规)。"""

    def __init__(self, src, dst, w, free, ridge_vec):
        import scipy.sparse as sp
        from scipy.sparse.linalg import splu
        n = free.numel()
        s = src.detach().cpu().numpy().astype(np.int64)
        d = dst.detach().cpu().numpy().astype(np.int64)
        ww = w.detach().cpu().numpy().astype(np.float64)
        rg = ridge_vec.detach().cpu().numpy().astype(np.float64)
        L = sp.coo_matrix((np.r_[ww, ww, -ww, -ww], (np.r_[s, d, s, d], np.r_[s, d, d, s])), shape=(n, n)).tocsr()
        L = L + sp.diags(rg)
        self.free_idx = np.where(free.detach().cpu().numpy())[0]
        self.n = n
        A = L[self.free_idx][:, self.free_idx].tocsc()
        # 2026-08-21 第 2 轮修法(结果档 §3.11 第 3 项):原先**无条件**加 1e-12·I,与稠密路径
        # "岭只加在 ridge_mask 行"的语义不一致——有参考角的分量本该拿到精确解,却被塞了一点收缩偏差。
        # 改为:先按原样分解;只有真奇异(无参考角且调用方没给岭的分量)才用极小岭兜底。
        # 实测该微岭的影响 |Δθ| ≤ 2.3e-9(10k)/1.9e-9(25k)/2.1e-7(70k),与 fp32 输出分辨率
        # (6.8e-8 / 9.5e-8 / 1.7e-7)同量级或更小,故本项对归档结果无可见影响。
        try:
            self.lu = splu(A)
        except RuntimeError:
            self.lu = splu(A + 1e-12 * sp.identity(A.shape[0], format="csc"))

    def solve(self, rhs_full: torch.Tensor) -> torch.Tensor:
        r = rhs_full.detach().cpu().numpy().astype(np.float64)
        x = np.zeros(self.n, dtype=np.float64)
        if self.free_idx.size:
            x[self.free_idx] = self.lu.solve(r[self.free_idx])
        return torch.from_numpy(x).to(rhs_full.device, rhs_full.dtype)


def _cpu_lu(src, dst, w, free, ridge_vec) -> "_CpuLU":
    """势重建法方程 LU 的缓存入口(预注册 §2 A 项;`GRIDFM_LU_CACHE=1` 才启用,默认关 = 直接现算)。

    键 = edge_index(src,dst)+ 权重 w(主线 = edge_use,0/1 给定量,不含学习量)+ free(= ~ref_mask)
    + ridge_vec(岭掩码与岭值合一)的内容哈希 —— 即矩阵 (L_w + diag(ridge))_ff 的全部决定量;
    禁 family_id/文件名。首次命中自检与拒绝见 perf_cache.cached_factor(G2)。"""
    if not perf_cache.lu_cache_enabled():
        return _CpuLU(src, dst, w, free, ridge_vec)
    key = perf_cache.content_key(
        "pot_lw",
        src.detach().cpu().numpy().astype(np.int64),
        dst.detach().cpu().numpy().astype(np.int64),
        w.detach().cpu().numpy().astype(np.float64),
        free.detach().cpu().numpy().astype(bool),
        ridge_vec.detach().cpu().numpy().astype(np.float64),
    )

    def _build():
        return _CpuLU(src, dst, w, free, ridge_vec)

    def _probe(obj):
        return obj.lu.solve(np.ones(obj.free_idx.size)) if obj.free_idx.size else np.zeros(0)

    return perf_cache.cached_factor(key, _build, _probe)


class _SparsePotential(torch.autograd.Function):
    """稀疏势重建:前向解 A x = rhs;反向隐式微分(对称系统,再解一次 A λ = g),∂θ/∂δ_k = w_k(λ[dst]−λ[src])。
    求解器:SPARSE_SOLVER="splu"(CPU 直接 LU,默认)或 "pcg"(GPU)。只对 δ 求导(w 视为常量)。"""

    @staticmethod
    def forward(ctx, delta, src, dst, w, free, fixed_vals, ridge_vec, iters, tol):
        dt = delta.dtype
        free_f = free.to(dt)
        # b = Σ_k w_k δ_k (e_dst − e_src);Dirichlet:rhs_f = b_f − (L x_s)_f
        b = torch.zeros_like(fixed_vals)
        wd = w * delta
        b.index_add_(0, dst, wd)
        b.index_add_(0, src, -wd)
        rhs = (b - _laplacian_matvec(fixed_vals, src, dst, w)) * free_f
        diag = torch.zeros_like(fixed_vals).index_add_(0, src, w).index_add_(0, dst, w) + ridge_vec
        lu = None
        if SPARSE_SOLVER == "splu":
            lu = _cpu_lu(src, dst, w, free, ridge_vec)
            x = lu.solve(rhs)
        else:
            x = _pcg_solve(rhs, src, dst, w, free, diag, ridge_vec, iters, tol)
        theta = x + fixed_vals
        ctx.save_for_backward(src, dst, w, free, diag, ridge_vec)
        ctx.iters, ctx.tol, ctx.lu = iters, tol, lu
        return theta

    @staticmethod
    def backward(ctx, g):
        src, dst, w, free, diag, ridge_vec = ctx.saved_tensors
        gf = g * free.to(g.dtype)
        if ctx.lu is not None:
            lam = ctx.lu.solve(gf)
        else:
            lam = _pcg_solve(gf, src, dst, w, free, diag, ridge_vec, ctx.iters, ctx.tol)
        grad_delta = w * (lam[dst] - lam[src])
        return grad_delta, None, None, None, None, None, None, None, None


def potential_reconstruct(delta: torch.Tensor, edge_index: torch.Tensor, edge_w: torch.Tensor,
                          node_batch: torch.Tensor, theta_ref: torch.Tensor, ref_mask: torch.Tensor,
                          num_graphs: Optional[int] = None, ridge: float = 1e-6,
                          dtype: torch.dtype = torch.float64,
                          ridge_mask: Optional[torch.Tensor] = None,
                          mode: str = "auto", cg_iters: int = 2000, cg_tol: float = 1e-7,
                          nmax: Optional[int] = None, has_ref: Optional[bool] = None,
                          has_ridge: Optional[bool] = None) -> torch.Tensor:
    """图上势重建:θ = argmin Σ_k w_k (θ[dst_k] − θ[src_k] − δ_k)²,s.t. θ[ref] = θ_ref。

    逐图稠密法方程(加权拉普拉斯 + Dirichlet 行替换),torch.linalg.solve(默认 fp64,
    路径图 N=1000 时条件数 ~1e6,fp32 不够)。
    岭:**只**加在 ridge_mask 为 True 的节点行(应传"所在分量无参考角"的节点;None = 全部行,
    仅供无 reach 信息时兜底)。含参考角的分量经 Dirichlet 行替换后已非奇异,岭为 0 → 精确解;
    若对其加岭,偏差 ≈ −ε·L_ff⁻¹θ 随有效电阻(~N²)与 |θ| 放大(2026-08-21 审查实测 N=1000 时
    ε=1e-6 偏 0.02~0.2 rad)。无参考角的分量:岭解 = 零均值有界势(调用方按 reach 回退)。
    返回 θ[N] fp32。可微(对 δ 与 w)。
    """
    dev = delta.device
    with torch.autocast(device_type=dev.type, enabled=False):
        d = delta.to(dtype)
        w = edge_w.to(dtype)
        n = node_batch.numel()
        B = int(node_batch.max().item()) + 1 if num_graphs is None else int(num_graphs)
        counts = torch.zeros(B, dtype=torch.long, device=dev).index_add_(
            0, node_batch, torch.ones_like(node_batch))     # 同步税:bincount 在 CUDA 上隐式同步
        off = torch.cumsum(counts, 0) - counts
        loc = torch.arange(n, device=dev) - off[node_batch]
        nmax = int(counts.max().item()) if nmax is None else int(nmax)   # 同步税:调用方可传入,免 .item() 同步
        src, dst = edge_index[0], edge_index[1]
        dense_elems = B * nmax * nmax
        if mode == "sparse" or (mode == "auto" and (nmax > DENSE_MAX_NODES
                                                    or dense_elems > DENSE_MAX_BATCH_ELEMS)):
            # 稀疏路径:全批 Jacobi-PCG + 隐式微分(只对 δ 可微,w 视为常量);O(E) 内存,万节点级
            free = ~ref_mask
            fixed_vals = torch.where(ref_mask, theta_ref.to(dtype), torch.zeros(n, dtype=dtype, device=dev))
            if ridge_mask is None:
                ridge_vec = torch.full((n,), float(ridge), dtype=dtype, device=dev)
            else:
                ridge_vec = torch.where(ridge_mask, torch.full((n,), float(ridge), dtype=dtype, device=dev),
                                        torch.zeros(n, dtype=dtype, device=dev))
            out = _SparsePotential.apply(d, src, dst, w.detach(), free, fixed_vals, ridge_vec,
                                         int(cg_iters), float(cg_tol))
            return out.float()
        g = node_batch[src]
        ls, ld = loc[src], loc[dst]
        A = torch.zeros(B, nmax, nmax, dtype=dtype, device=dev)
        A = A.index_put((g, ls, ls), w, accumulate=True)
        A = A.index_put((g, ld, ld), w, accumulate=True)
        A = A.index_put((g, ls, ld), -w, accumulate=True)
        A = A.index_put((g, ld, ls), -w, accumulate=True)
        b = torch.zeros(B, nmax, dtype=dtype, device=dev)
        b = b.index_put((g, ld), w * d, accumulate=True)
        b = b.index_put((g, ls), -(w * d), accumulate=True)
        eye = torch.eye(nmax, dtype=dtype, device=dev)
        if ridge_mask is None:
            A = A + ridge * eye
        elif (has_ridge if has_ridge is not None else bool(ridge_mask.any())):
            rdiag = torch.zeros(B, nmax, dtype=dtype, device=dev)
            rdiag[node_batch[ridge_mask], loc[ridge_mask]] = ridge
            A = A + torch.diag_embed(rdiag)
        ar = torch.arange(nmax, device=dev)
        fixed = ar[None, :] >= counts[:, None]                  # padding 行
        refd = torch.zeros(B, nmax, dtype=dtype, device=dev)
        if (has_ref if has_ref is not None else bool(ref_mask.any())):
            gref, lref = node_batch[ref_mask], loc[ref_mask]
            fixed = fixed.clone()
            fixed[gref, lref] = True
            refd[gref, lref] = theta_ref[ref_mask].to(dtype)
        A = torch.where(fixed[:, :, None], eye[None].expand(B, -1, -1), A)
        b = torch.where(fixed, refd, b)
        theta = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
        out = theta[node_batch, loc]
    return out.float()          # 恒 fp32:θ 可达 ±13 rad,fp16 分辨率(0.008)不够


def pf_edge_bias_loss(delta: torch.Tensor, edge_index: torch.Tensor, edge_use: torch.Tensor,
                      theta_target: torch.Tensor, theta_avail: torch.Tensor,
                      edge_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """逐图"沿真实 Δθ 方向的平均偏差"惩罚:mean_g |mean_k∈g (δ_k − Δθ_k)·sign(Δθ_k)|。

    动机(2026-08-21 E2 诊断):逐边 L1 在带噪提示上的最优解是向均值收缩(斜率 −0.18 val / −0.29 transfer),
    零均值误差沿路径只按 √hop 累积,而系统性收缩按 hop 线性累积 → 大网 θ 崩坏。本项直接压平均收缩。
    """
    src, dst = edge_index[0], edge_index[1]
    m = edge_use & theta_avail[src] & theta_avail[dst]
    if not bool(m.any()):
        return delta.new_zeros(())
    dtrue = wrap_angle(theta_target[dst] - theta_target[src]).float()
    e_al = (delta.float() - dtrue) * torch.sign(dtrue)
    g = edge_batch[m]
    s = torch.zeros(num_graphs, dtype=torch.float32, device=delta.device).index_add_(0, g, e_al[m])
    c = torch.zeros(num_graphs, dtype=torch.float32, device=delta.device).index_add_(
        0, g, torch.ones_like(e_al[m]))
    has = c > 0
    return (s[has] / c[has]).abs().mean()


def pf_edge_loss(delta: torch.Tensor, edge_index: torch.Tensor, edge_use: torch.Tensor,
                 theta_target: torch.Tensor, theta_avail: torch.Tensor) -> torch.Tensor:
    """逐边 L1:|δ_k − wrap(θ_t[dst] − θ_t[src])|,只在 use 且两端存档 θ 可用的边上。"""
    src, dst = edge_index[0], edge_index[1]
    m = edge_use & theta_avail[src] & theta_avail[dst]
    if not bool(m.any()):
        return delta.new_zeros(())
    dtrue = wrap_angle(theta_target[dst] - theta_target[src])
    return (delta[m].float() - dtrue[m].float()).abs().mean()


# ---------------------------------------------------------------------------
# 失配修正步(预注册 §13.1,E6;2026-08-21 第 4 轮)
# ---------------------------------------------------------------------------

class _NodeSolve(torch.autograd.Function):
    """解 (L_w + diag(ridge))_ff x_f = rhs_f(固定节点 x=0),对 rhs 可微。

    与 `_SparsePotential` 的区别:右端项直接给在**节点**上(失配 ΔP),不是由边上的 δ 求散度。
    A 对称 → 反向 = 用同一个 LU 再解一次(隐式微分)。w 视为常量(静态 1/x)。
    """

    @staticmethod
    def forward(ctx, rhs, src, dst, w, free, ridge_vec, iters, tol):
        free_f = free.to(rhs.dtype)
        r = rhs * free_f
        diag = torch.zeros_like(rhs).index_add_(0, src, w).index_add_(0, dst, w) + ridge_vec
        lu = None
        if SPARSE_SOLVER == "splu":
            lu = _cpu_lu(src, dst, w, free, ridge_vec)
            x = lu.solve(r)
        else:
            x = _pcg_solve(r, src, dst, w, free, diag, ridge_vec, iters, tol)
        ctx.save_for_backward(src, dst, w, free, diag, ridge_vec)
        ctx.iters, ctx.tol, ctx.lu = iters, tol, lu
        return x

    @staticmethod
    def backward(ctx, g):
        src, dst, w, free, diag, ridge_vec = ctx.saved_tensors
        gf = g * free.to(g.dtype)
        if ctx.lu is not None:
            lam = ctx.lu.solve(gf)
        else:
            lam = _pcg_solve(gf, src, dst, w, free, diag, ridge_vec, ctx.iters, ctx.tol)
        return lam * free.to(g.dtype), None, None, None, None, None, None, None


def mismatch_correction(dp: torch.Tensor, edge_index: torch.Tensor, edge_b: torch.Tensor,
                        node_batch: torch.Tensor, ref_mask: torch.Tensor,
                        num_graphs: Optional[int] = None, ridge: float = 1e-6,
                        dtype: torch.dtype = torch.float64,
                        ridge_mask: Optional[torch.Tensor] = None,
                        mode: str = "auto", cg_iters: int = 2000,
                        cg_tol: float = 1e-7) -> torch.Tensor:
    """一步 DC-Newton 失配修正:解 B′·Δθ = ΔP(平衡机/参考节点 Δθ = 0),返回 Δθ[N] fp32。

    用途(预注册 §13.1):势重建给出的 θ⁰ 是"逐边 Δθ 的最小二乘积分",误差会沿长路径累积
    (实测 7 万节点上 +0.0028 rad/跳)。ΔP 是**全局**信息、B′⁻¹ 是**全局**算子,一步就能把
    累积偏差整体拉回;这正是快速解耦潮流的 P–θ 半迭代。

    - dp[N]:节点有功失配(标幺),**调用方负责只在"干净 P 方程"节点上非零**
      (PQ 的 P、PV 的 P;平衡机 P 未知、Ybus 行不完整的节点都必须置 0);
    - edge_b[E]:支路电纳 1/x(静态给定量),不在运/不可用的边给 0;
    - ref_mask[N]:Dirichlet 节点(平衡机),那里 Δθ ≡ 0;
    - ridge_mask:与 `potential_reconstruct` 同约定——只对"所在分量无参考角"的行加岭,
      有参考分量得精确解(§5b 的教训)。

    **无泄漏**:本函数只做线性代数;ΔP 由调用方用"预测量 + 给定量"算出,不读任何被监督量的存档值。
    """
    dev = dp.device
    with torch.autocast(device_type=dev.type, enabled=False):
        rhs = dp.to(dtype)
        w = edge_b.to(dtype)
        n = node_batch.numel()
        B = int(node_batch.max().item()) + 1 if num_graphs is None else int(num_graphs)
        counts = torch.bincount(node_batch, minlength=B)
        off = torch.cumsum(counts, 0) - counts
        loc = torch.arange(n, device=dev) - off[node_batch]
        nmax = int(counts.max().item())
        src, dst = edge_index[0], edge_index[1]
        free = ~ref_mask
        if ridge_mask is None:
            ridge_vec = torch.full((n,), float(ridge), dtype=dtype, device=dev)
        else:
            ridge_vec = torch.where(ridge_mask, torch.full((n,), float(ridge), dtype=dtype, device=dev),
                                    torch.zeros(n, dtype=dtype, device=dev))
        if mode == "sparse" or (mode == "auto" and (nmax > DENSE_MAX_NODES
                                                    or B * nmax * nmax > DENSE_MAX_BATCH_ELEMS)):
            out = _NodeSolve.apply(rhs, src, dst, w, free, ridge_vec, int(cg_iters), float(cg_tol))
            return out.float()
        g = node_batch[src]
        ls, ld = loc[src], loc[dst]
        A = torch.zeros(B, nmax, nmax, dtype=dtype, device=dev)
        A = A.index_put((g, ls, ls), w, accumulate=True)
        A = A.index_put((g, ld, ld), w, accumulate=True)
        A = A.index_put((g, ls, ld), -w, accumulate=True)
        A = A.index_put((g, ld, ls), -w, accumulate=True)
        b = torch.zeros(B, nmax, dtype=dtype, device=dev)
        b = b.index_put((node_batch, loc), rhs, accumulate=True)
        eye = torch.eye(nmax, dtype=dtype, device=dev)
        if ridge_mask is None:
            A = A + ridge * eye
        elif bool(ridge_mask.any()):
            rdiag = torch.zeros(B, nmax, dtype=dtype, device=dev)
            rdiag[node_batch[ridge_mask], loc[ridge_mask]] = ridge
            A = A + torch.diag_embed(rdiag)
        ar = torch.arange(nmax, device=dev)
        fixed = ar[None, :] >= counts[:, None]                  # padding 行
        if bool(ref_mask.any()):
            fixed = fixed.clone()
            fixed[node_batch[ref_mask], loc[ref_mask]] = True   # 参考节点 Δθ ≡ 0
        A = torch.where(fixed[:, :, None], eye[None].expand(B, -1, -1), A)
        b = torch.where(fixed, torch.zeros_like(b), b)
        x = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
        out = x[node_batch, loc]
    return out.float()


# ---------------------------------------------------------------------------
# 通用稀疏直接解(E12 的 Q–V 半步;预注册 §15.2,2026-08-22 第 3 轮)
# ---------------------------------------------------------------------------

class _CooLU:
    """CPU scipy splu:一般稀疏方阵 A(COO 给定)在自由节点子块 A_ff 上的 LU;前向解 A_ff x = r,反向解 A_ffᵀ λ = g。"""

    def __init__(self, rows, cols, vals, free, ridge: float):
        import scipy.sparse as sp
        from scipy.sparse.linalg import splu
        n = free.numel()
        r = rows.detach().cpu().numpy().astype(np.int64)
        c = cols.detach().cpu().numpy().astype(np.int64)
        v = vals.detach().cpu().numpy().astype(np.float64)
        A = sp.coo_matrix((v, (r, c)), shape=(n, n)).tocsr()
        self.free_idx = np.where(free.detach().cpu().numpy())[0]
        self.n = n
        Af = A[self.free_idx][:, self.free_idx].tocsc()
        if ridge:
            Af = Af + float(ridge) * sp.identity(Af.shape[0], format="csc")
        self.lu = splu(Af)

    def solve(self, rhs_full: torch.Tensor, trans: str = "N") -> torch.Tensor:
        r = rhs_full.detach().cpu().numpy().astype(np.float64)
        x = np.zeros(self.n, dtype=np.float64)
        if self.free_idx.size:
            x[self.free_idx] = self.lu.solve(r[self.free_idx], trans=trans)
        return torch.from_numpy(x).to(rhs_full.device, rhs_full.dtype)


class _CooSolve(torch.autograd.Function):
    """x_f = A_ff⁻¹ r_f(其余 x=0);对 r 可微(隐式微分:λ = A_ffᵀ⁻¹ g_f);A 视为常量。"""

    @staticmethod
    def forward(ctx, rhs, rows, cols, vals, free, ridge):
        lu = _CooLU(rows, cols, vals, free, ridge)
        x = lu.solve(rhs * free.to(rhs.dtype))
        ctx.lu, ctx.free = lu, free
        return x

    @staticmethod
    def backward(ctx, g):
        lam = ctx.lu.solve(g * ctx.free.to(g.dtype), trans="T")
        return lam * ctx.free.to(g.dtype), None, None, None, None, None


def coo_solve(rhs: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, vals: torch.Tensor,
              free: torch.Tensor, ridge: float = 0.0, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """解 A_ff x_f = rhs_f(A 由 COO 三元组给出,重复项求和;free 之外 x=0),fp64 直接稀疏 LU,对 rhs 可微。
    返回 x[N],dtype 随 rhs(推理器传 fp32 → fp32;双精度输入保持 fp64,便于 gradcheck)。"""
    dev = rhs.device
    with torch.autocast(device_type=dev.type, enabled=False):
        out = _CooSolve.apply(rhs.to(dtype), rows, cols, vals.to(dtype), free, float(ridge))
    return out.to(rhs.dtype)


def qv_correction(dq_over_v: torch.Tensor, phys_index: torch.Tensor, phys_b: torch.Tensor,
                  free: torch.Tensor, ridge: float = 1e-8) -> torch.Tensor:
    """快速解耦潮流的 Q–V 半步(E12 `--refine-qv`):解 B″·ΔV = ΔQ / V,B″ = −Im Y 在 free(干净 Q 方程的 PQ 节点)上的子块,
    其余节点 ΔV ≡ 0(Dirichlet:PV / 平衡机电压给定,方程不完整的 PQ 行不动)。与 `pf_physics.qv_hint` 同一线性系统
    (那里是零训练提示、numpy;这里可微、在推理器里用模型自己的 V/θ 起步)。返回 ΔV[N] fp32。

    **无泄漏**:只做线性代数;右端由调用方用"预测量 + 给定量"算出。phys_b 来自 `pf_phys_arrays`(j 侧变比 + 并补 −1/x)。
    """
    return coo_solve(dq_over_v, phys_index[0], phys_index[1], -phys_b, free, ridge=ridge)
