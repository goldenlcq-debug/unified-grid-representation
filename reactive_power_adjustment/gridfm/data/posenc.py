"""位置/结构编码离线计算(方案 3.2 / T3.4)。数据生成阶段按拓扑算一次并随样本落盘。

- LaPE:对称归一化拉普拉斯 L_sym = I - D^{-1/2} A D^{-1/2}(v0 边权=1);按连通分量分别特征分解,
  取非零特征值最小的 k 个特征向量;k_eff = min(k, N_c-1),不足补零并附逐向量有效位;
  **重根子空间固定种子规范化**(方案 3.2:|λi-λj|<1e-6 视为重根)+ 确定性定号。
- RWSE:随机游走返回概率 diag(P^t), t=1..steps,P = D^{-1} A。天然无符号/基歧义。
- component_id:连通分量归属(按规模降序编号,0=最大分量)。

孤岛/孤立节点(如 NULL 母线)= 单节点分量,k_eff=0,LaPE 全零 + 有效位 0。

稀疏化(2026-07-09,解锁 5k~10k 规模):全程 scipy.sparse,不再构造 N×N 稠密矩阵。
- LaPE:分量内 Lanczos(eigsh 取 2I−L 的最大端,免 LU 分解;确定性起始向量 v0);
  小分量与巨型简并谱退化回稠密 eigh。
- RWSE:diag(P^t) = diag(S^t)(S = D^{-1/2} A D^{-1/2} 对称相似变换),稀疏幂迭代取对角;
  按行块乘并计 nnz,超帽时退化为分块稠密列迭代(始终精确,只是慢)。

**环境地雷(实测,别改回去)**:本机 scipy 1.8.1 的 SuperLU 会随堆状态随机访问违例
(同进程先算 5041 节点再算 10000 节点必段错误),因此 LaPE **不得使用 eigsh 的移位反演**
(sigma=... 会触发 LU 分解)。见 _eig_lowest 的详细说明。

**规范化为什么是必需的**(2026-07-09 FABLE 评审):严格图自同构的孪生节点,其反对称模落在
简并子空间内(真孪生:λ=1 或 1+1/d)。稠密 eigh 与稀疏 eigsh 在简并子空间内返回的基是任意的,
特征分解器版本一换、窗口边界一切穿重根组,孪生身份就静默互换。规范化把基钉死为
"本征空间投影 × 固定探针 → QR",只依赖本征空间本身(与求解器无关),两条路径逐位可比。
窗口边界切穿重根组时**自动扩容取到整组**再截断,否则稀疏路径只看到组的一部分,规范化结果不同。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import ArpackError, eigsh   # ArpackNoConvergence 是 ArpackError 子类

# 分量节点数 ≤ 此值走稠密 eigh(ARPACK 在小矩阵上既不快也不稳)
LAPE_DENSE_MAX = 64
# 重根判据(方案 3.2:|λi-λj| < 1e-6 视为同一子空间)
LAPE_DEGEN_TOL = 1e-6
# 重根子空间规范化探针种子(固定 → 跨进程/跨求解器一致)
LAPE_CANON_SEED = 20260709
# 可规范化的最大重根子空间维数。超出者**整体弃用**(该列起留零 + 有效位 0)。
# 理由有二:①代价——星形/辐射网的 λ=1 重数可达 N-2,取全整组等于退回 O(N³) 稠密;
# ②语义——巨型简并子空间里根本不存在"有信息的规范基",任何选择都是任意的,
# 硬塞进 LaPE 只会让模型骑着特征分解器的数值噪声认人(FABLE 评审的静默互换风险)。
# 对称打破在这些节点上交给随机通道(3.2-3),这正是它存在的理由。
LAPE_MAX_DEGEN = 32
# ARPACK 在"要取的模数占分量规模一大半"时既慢又易不收敛 → 直接走稠密
LAPE_SPARSE_FRAC = 0.5
# ARPACK 失败后的稠密兜底上限(eigh 在 4096² 约 130MB / 数十秒;再大则明确报错不猜)
LAPE_DENSE_RESCUE_MAX = 4096
# RWSE 稀疏幂迭代 nnz 上限(超出退化为分块稠密,防幂次填充爆内存)
RWSE_NNZ_CAP = 8_000_000
RWSE_BLOCK = 256          # 稠密兜底的列块宽
RWSE_ROW_BLOCK = 512      # 稀疏幂乘的行块宽(限住单次乘法的瞬时分配)


def _adjacency(edge_index: np.ndarray, n: int) -> sp.csr_matrix:
    """对称邻接稀疏矩阵(去自环、重边计 1)。edge_index:[2, M],0-based。"""
    i, j = edge_index[0], edge_index[1]
    mask = i != j
    i, j = i[mask], j[mask]
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    A = sp.coo_matrix((np.ones(rows.size, dtype=np.float64), (rows, cols)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A.data = np.ones_like(A.data)      # 重边计 1(并联双回在图上是一条边)
    return A


def _component_labels(A: sp.csr_matrix, n: int) -> np.ndarray:
    ncomp, labels = connected_components(A, directed=False)
    # 按分量规模降序重编号(0 = 最大分量)
    sizes = np.bincount(labels, minlength=ncomp)
    order = np.argsort(-sizes, kind="stable")
    remap = np.empty(ncomp, dtype=np.int64)
    remap[order] = np.arange(ncomp)
    return remap[labels]


def _sign_normalize(vec: np.ndarray, rel_tol: float = 1e-6) -> np.ndarray:
    """确定性定号:绝对值最大分量为正(并列取首个)。

    实测教训(2026-07-09):对称图(环/方格网)的特征向量存在**解析等幅**分量,
    裸 `argmax(|v|)` 由浮点噪声裁决落在哪个并列项 → 稠密/稀疏两条路径取到相反符号。
    改为"取 |v| ≥ (1-rel_tol)·max|v| 的并列带内最小索引",把解析并列一网打尽后再按索引裁决。
    """
    a = np.abs(vec)
    m = float(a.max()) if a.size else 0.0
    if m <= 0.0:
        return vec
    idx = int(np.flatnonzero(a >= m * (1.0 - rel_tol))[0])
    return -vec if vec[idx] < 0 else vec


def _norm_laplacian(Ac: sp.csr_matrix) -> Tuple[sp.csr_matrix, np.ndarray]:
    """返回 (L_sym, u0)。u0 = D^{1/2}·1 归一化 = 连通分量上 L_sym 的解析零模。"""
    nc = Ac.shape[0]
    deg = np.asarray(Ac.sum(axis=1)).ravel()
    dinv_sqrt = np.zeros(nc)
    nz = deg > 0
    dinv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    Dm = sp.diags(dinv_sqrt)
    L = sp.identity(nc, format="csr") - (Dm @ Ac @ Dm)
    L = ((L + L.T) * 0.5).tocsr()       # 数值对称化
    u0 = np.zeros(nc)
    u0[nz] = np.sqrt(deg[nz])
    nrm = np.linalg.norm(u0)
    if nrm > 0:
        u0 /= nrm
    return L, u0


def _eig_dense(L: sp.csr_matrix, kk: int):
    Ld = L.toarray()
    Ld = (Ld + Ld.T) * 0.5
    evals, evecs = np.linalg.eigh(Ld)
    return evals[:kk], evecs[:, :kk]


def _eig_lowest(L: sp.csr_matrix, kk: int, dense: bool):
    """返回按升序排列的最小 kk 个 (evals, evecs)。dense=True 走 eigh(全谱后截断)。

    稀疏路径取 **M = 2I − L 的最大代数特征值**(σ(L_sym) ⊂ [0,2] ⇒ M 的最大端 = L 的最小端)。
    为什么不用移位反演(sigma<0,ARPACK 的常规做法):

    - 它要对 L − σI 做 LU 分解,即 SuperLU。**本机 scipy 1.8.1 的 SuperLU 会随堆状态
      随机性地访问违例**(实测:同进程内先算一次 5041 节点、再算 10000 节点必段错误;
      单独各自跑却都正常;纯 scipy 复现同样成立)。段错误不可 catch,批产会静默半途而废。
    - M 只需矩阵-向量乘,不做任何分解。实测 10000 节点方格网:1.68s / 残差 1.8e-14,
      比移位反演更快,与稠密 eigh 特征值全同(稠密要 106s)。

    λ = 2 − w 存在相消(w≈2, λ≈2.5e-4),故不用它:特征向量拿到后直接用 **Rayleigh 商**
    λ_i = vᵢᵀ L vᵢ 现算,无相消,与稠密路径的分组容差(1e-6)口径一致。

    起始向量 v0 必须显式给定——ARPACK 默认随机 v0 会让简并子空间内返回的基不可复现。
    """
    nc = L.shape[0]
    if dense:
        return _eig_dense(L, kk)
    assert kk < nc, f"ARPACK 要求 k < n(kk={kk}, nc={nc});调用方须先切稠密路径"
    v0 = np.ones(nc) / np.sqrt(nc)
    M = (2.0 * sp.identity(nc, format="csr") - L).tocsr()
    try:
        _w, evecs = eigsh(M, k=kk, which="LA", v0=v0, tol=0.0)
    except ArpackError:
        # 高度简并谱(如星形网 λ=1 重数 N−2)可能不收敛。加大 Krylov 子空间重试一次;
        # 仍失败则退稠密——**绝不返回不可复现的基**。
        try:
            ncv = min(nc - 1, max(2 * kk + 1, 40))
            _w, evecs = eigsh(M, k=kk, which="LA", v0=v0, ncv=ncv, maxiter=10_000, tol=0.0)
        except ArpackError as e:
            if nc <= LAPE_DENSE_RESCUE_MAX:
                return _eig_dense(L, kk)
            raise RuntimeError(
                f"LaPE:ARPACK 在 nc={nc} 分量上失败({e}),且规模超出稠密兜底上限 "
                f"{LAPE_DENSE_RESCUE_MAX}。拒绝返回不可复现的基。"
            ) from e
    evals = np.einsum("ij,ij->j", evecs, L @ evecs)     # Rayleigh 商(免 2−w 相消)
    order = np.argsort(evals, kind="stable")
    return evals[order], evecs[:, order]


def _canonical_group(V: np.ndarray, gidx: int) -> Optional[np.ndarray]:
    """重根子空间固定种子规范化(方案 3.2)。失败返回 None(调用方弃用该组)。

    V:[nc, r] 本征空间的任意标准正交基。返回**只依赖本征空间本身**的规范基:
    投影算子 P = V Vᵀ 与基无关 → 对固定探针 G 投影后 QR,得到确定基与确定列序。
    r=1 时无基不定性,直接返回。

    探针投影秩亏(概率 0)时**不得回退到 V**:V 是求解器给的任意基,稠密/稀疏两路不同,
    正是要消除的东西。改为换种子重试;仍秩亏则返回 None,该组整体弃用(留零 + 有效位 0)。
    """
    nc, r = V.shape
    if r <= 1:
        return V
    for attempt in range(3):
        rng = np.random.default_rng(LAPE_CANON_SEED + 1000 * attempt + gidx)
        G = rng.standard_normal((nc, r))
        Pg = V @ (V.T @ G)              # 投影到本征空间(投影算子与基无关)
        Q, R = np.linalg.qr(Pg)
        d = np.diag(R)
        if np.min(np.abs(d)) >= 1e-10:
            signs = np.sign(d)
            signs[signs == 0] = 1.0
            return Q * signs            # 固定 QR 的符号不定性
    return None


def _group_bounds(evals: np.ndarray, tol: float = LAPE_DEGEN_TOL):
    """按重根容差把升序特征值切成组,返回 [(start, end), ...](end 不含)。"""
    if evals.size == 0:
        return []
    cuts = np.where(np.diff(evals) > tol)[0] + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [evals.size]])
    return list(zip(starts.tolist(), ends.tolist()))


def _drop_trivial(evals: np.ndarray, evecs: np.ndarray, u0: np.ndarray):
    """剔除平凡零模(解析已知 u0):按与 u0 的重叠识别,而非按索引——
    近奇异分量上 λ0 与 λ1 可能同时接近 0,按索引会误删真模。"""
    ov = np.abs(evecs.T @ u0)
    j0 = int(np.argmax(ov))
    keep = np.ones(evecs.shape[1], dtype=bool)
    keep[j0] = False
    return evals[keep], evecs[:, keep]


def _component_lape(Ac: sp.csr_matrix, k: int) -> Tuple[np.ndarray, int]:
    """单连通分量的 LaPE。返回 (vecs[nc, k_eff], n_valid)。

    n_valid = 左起可用列数;其余列留零 + 有效位 0(重根组过大,无可规范化的基)。

    取模数固定为 k_eff + 1(含平凡零模)+ LAPE_MAX_DEGEN:多取的这一段专门用来
    **判断窗口边界的重根组有没有取全**。若边界组延伸到取模数末端,其维数必 > LAPE_MAX_DEGEN
    (因为组起点 s < k_eff),按上文约定整组弃用——于是稀疏路径无需无界增长,
    且与稠密路径(看得见全谱)得到完全一致的取舍。
    """
    nc = Ac.shape[0]
    k_eff = min(k, nc - 1)
    if k_eff <= 0:
        return np.zeros((nc, 0)), 0
    L, u0 = _norm_laplacian(Ac)

    want = k_eff + 1 + LAPE_MAX_DEGEN
    # want 接近 nc 时 ARPACK 既慢又易在简并谱上不收敛(且 k<n 是硬约束)→ 直接稠密
    dense = (nc <= LAPE_DENSE_MAX) or (want > int(LAPE_SPARSE_FRAC * nc))
    evals, evecs = _eig_lowest(L, nc if dense else want, dense)
    ev, vv = _drop_trivial(evals, evecs, u0)

    out = np.zeros((nc, k_eff))
    n_valid = 0
    for gidx, (s, e) in enumerate(_group_bounds(ev)):
        if s >= k_eff:
            break
        # 组延伸到取模数末端 ⇒ 其维数必 > LAPE_MAX_DEGEN(因 s < k_eff),与稠密路径判定一致
        if (e - s) > LAPE_MAX_DEGEN:                      # 巨型重根组:该列起全部弃用
            break
        Q = _canonical_group(vv[:, s:e], gidx)            # 整组规范化后再截断
        if Q is None:                                     # 规范化失败:同样整组弃用
            break
        take = min(e, k_eff) - s
        for c in range(take):
            out[:, s + c] = _sign_normalize(Q[:, c])
        n_valid = s + take
    return out, n_valid


def compute_lape(edge_index: np.ndarray, n: int, k: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (lape[N,k] float32, valid[N,k] bool, component_id[N] int64)。"""
    A = _adjacency(edge_index, n)
    comp = _component_labels(A, n)
    lape = np.zeros((n, k), dtype=np.float32)
    valid = np.zeros((n, k), dtype=bool)

    for c in np.unique(comp):
        nodes = np.where(comp == c)[0]
        if nodes.size <= 1:
            continue                     # 孤立节点:LaPE 全零、有效位 0
        Ac = A[nodes][:, nodes]
        vecs, n_valid = _component_lape(Ac.tocsr(), k)
        if n_valid:
            cols = np.arange(n_valid)
            lape[np.ix_(nodes, cols)] = vecs[:, :n_valid].astype(np.float32)
            valid[np.ix_(nodes, cols)] = True

    return lape, valid, comp


def _matmul_capped(St: sp.csr_matrix, S: sp.csr_matrix, cap: int):
    """按行块计算 St @ S,累计 nnz 超帽即放弃(返回 None)。

    为什么不能整块乘:scipy 的 csr@csr 先做符号阶段算出结果精确 nnz,再一次性分配。
    星形网(n=10000)的 S² 精确 nnz≈1e8 → 单个 CSR 1.2GB 已经落地,事后再查 nnz 无意义。
    按行块乘把瞬时分配压到 block×n,且顺带拿到**精确** nnz(而非路径数上界——
    实测教训:上界在 100×100 方格网上比真值大 4 倍,会把正常电网误踢进分块兜底,慢 20 倍)。
    """
    n = St.shape[0]
    parts, total = [], 0
    for r0 in range(0, n, RWSE_ROW_BLOCK):
        P = (St[r0:r0 + RWSE_ROW_BLOCK] @ S).tocsr()
        total += P.nnz
        if total > cap:
            return None
        parts.append(P)
    return parts[0] if len(parts) == 1 else sp.vstack(parts, format="csr")


def _rwse_sparse(S: sp.csr_matrix, n: int, steps: int):
    """返回 [n, steps];填充超帽时返回 None(交给分块稠密路径重算)。"""
    out = np.zeros((n, steps), dtype=np.float64)
    St = S.copy()
    for t in range(steps):
        out[:, t] = St.diagonal()
        if t + 1 < steps:
            St = _matmul_capped(St, S, RWSE_NNZ_CAP)
            if St is None:
                return None
    return out


def _rwse_blocked(S: sp.csr_matrix, n: int, steps: int) -> np.ndarray:
    """精确但慢的兜底:按列块推进 S^t·e_j,只取对角。内存 O(n × block)。"""
    out = np.zeros((n, steps), dtype=np.float64)
    for start in range(0, n, RWSE_BLOCK):
        cols = np.arange(start, min(start + RWSE_BLOCK, n))
        X = np.zeros((n, cols.size))
        X[cols, np.arange(cols.size)] = 1.0
        for t in range(steps):
            X = S @ X
            out[cols, t] = X[cols, np.arange(cols.size)]
    return out


def compute_rwse(edge_index: np.ndarray, n: int, steps: int = 16) -> np.ndarray:
    """随机游走返回概率 diag(P^t), t=1..steps。返回 [N, steps] float32。

    P = D^{-1}A 非对称,但 P = D^{-1/2} S D^{1/2}(S = D^{-1/2} A D^{-1/2} 对称)
    ⇒ diag(P^t) ≡ diag(S^t)。用对称 S 做稀疏幂迭代,数值更稳且省一半存储。
    """
    A = _adjacency(edge_index, n)
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv_sqrt = np.zeros(n)
    nz = deg > 0
    dinv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    Dm = sp.diags(dinv_sqrt)
    S = (Dm @ A @ Dm).tocsr()            # 孤立节点:整行整列为 0 → diag 恒 0(与稠密口径一致)

    out = _rwse_sparse(S, n, steps)
    if out is None:
        out = _rwse_blocked(S, n, steps)
    return out.astype(np.float32)


def lape_sign_flip_augment(lape: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """训练期增广:每个特征向量列独立以 0.5 概率乘 -1(方案 3.2)。"""
    signs = rng.choice([-1.0, 1.0], size=lape.shape[1]).astype(np.float32)
    return lape * signs[None, :]
