"""推理提速缓存(预注册 docs/预注册-提速LU与PE缓存-20260825.md;2026-08-25 第 7 轮)。

两个开关,**默认全关 = 旧行为逐位不变**(训练/评估/队列不设环境变量时本模块零介入):
  - `GRIDFM_LU_CACHE=1`:稀疏 LU 因子按**内容哈希**进程内缓存(势重建 L_w、带网损 DC 的 B′、
    Q–V 的 B″、dcv 的积分拉普拉斯;FDPF 对照同享)。条目数上限 `GRIDFM_LU_CACHE_MAX`(默认 16,LRU)。
  - `GRIDFM_POSENC_DIR=<dir>`:位置编码落盘缓存目录(gridfm/data/graph.py 使用;本模块只提供键与指纹)。

纪律(预注册 §2c C 类,唯一翻车路径 = 缓存键取错):
  1. 键 = 决定矩阵内容的数组的 sha256(**dtype + shape + 字节**),按站点加 tag 前缀;
     **禁止** family_id / 文件名 / 样本序号参与键。
  2. G2 首次命中自检:每个条目第一次命中时,现算一份因子,对确定右端(全 1)各解一次,
     max|Δ| ≤ 1e-12 才放行;不过则**拒绝该缓存条目**(换用现算的,记 verify_reject)。
  3. 关闭时(默认)`cached_splu` 就是 `splu(build())`,与旧代码逐位同一条路。
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np

# G2 首次命中自检的判限(预注册 §3 G2;绝对值)
VERIFY_TOL = 1e-12
# 位置编码落盘 schema 版本(G3 指纹字段之一;数组布局变更时必须 +1)
POSENC_SCHEMA = 1


def lu_cache_enabled() -> bool:
    return os.environ.get("GRIDFM_LU_CACHE", "0") == "1"


def posenc_dir() -> str:
    return os.environ.get("GRIDFM_POSENC_DIR", "")


def content_key(tag: str, *parts) -> str:
    """内容哈希键:tag(站点名,防跨站点撞键)+ 各数组的 dtype/shape/字节 + 标量 repr。
    None 记入哨兵(键含"哪些参数缺席"的信息)。"""
    h = hashlib.sha256(tag.encode("utf-8"))
    for p in parts:
        if p is None:
            h.update(b"\x00<None>")
        elif isinstance(p, np.ndarray):
            a = np.ascontiguousarray(p)
            h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        else:
            h.update(repr(p).encode())
        h.update(b"\x00")
    return h.hexdigest()


class _Entry:
    __slots__ = ("obj", "verified")

    def __init__(self, obj):
        self.obj = obj
        self.verified = False


_CACHE: "OrderedDict[str, _Entry]" = OrderedDict()
_STATS = {"miss": 0, "hit": 0, "verify_pass": 0, "verify_reject": 0}


def _max_entries() -> int:
    return int(os.environ.get("GRIDFM_LU_CACHE_MAX", "16"))


def clear() -> None:
    _CACHE.clear()
    for k in _STATS:
        _STATS[k] = 0


def stats() -> dict:
    return dict(_STATS, entries=len(_CACHE))


def cached_factor(key: str, build: Callable[[], object],
                  probe: Callable[[object], np.ndarray]):
    """通用因子缓存。build() 现算因子对象;probe(obj) 用确定右端解一次(G2 对拍用)。
    未开启 → 直接 build()(与旧代码同一条路);未命中 → build 后入缓存;
    首次命中 → 现算对拍(max|Δ| ≤ VERIFY_TOL),不过则拒绝缓存条目、改用现算。"""
    if not lu_cache_enabled():
        return build()
    ent = _CACHE.get(key)
    if ent is None:
        obj = build()
        while len(_CACHE) >= _max_entries():
            _CACHE.popitem(last=False)
        _CACHE[key] = _Entry(obj)
        _STATS["miss"] += 1
        return obj
    _CACHE.move_to_end(key)
    _STATS["hit"] += 1
    if not ent.verified:
        fresh = build()
        try:
            a = np.asarray(probe(ent.obj), dtype=np.float64)
            b = np.asarray(probe(fresh), dtype=np.float64)
            ok = (a.shape == b.shape) and (a.size == 0 or bool(np.max(np.abs(a - b)) <= VERIFY_TOL))
        except Exception:
            ok = False
        if not ok:
            _STATS["verify_reject"] += 1
            ent.obj = fresh          # 拒绝旧条目;现算的重新走首次命中自检
            ent.verified = False
            return fresh
        _STATS["verify_pass"] += 1
        ent.verified = True
    return ent.obj


def cached_splu(tag: str, key_parts, build_csc: Callable[[], object]):
    """scipy splu 的缓存封装。key_parts = 决定矩阵内容的数组/标量序列(内容哈希,禁文件名);
    build_csc() 现构 CSC 矩阵(未开启/未命中/首次命中对拍时才会被调用)。返回 SuperLU。"""
    from scipy.sparse.linalg import splu
    if not lu_cache_enabled():
        return splu(build_csc())
    key = content_key(tag, *key_parts)

    def _build():
        return splu(build_csc())

    def _probe(lu):
        n = lu.shape[0]
        return lu.solve(np.ones(n)) if n else np.zeros(0)

    return cached_factor(key, _build, _probe)


# ---------------------------------------------------------------------------
# 位置编码落盘(G3 指纹;gridfm/data/graph.py 调用)
# ---------------------------------------------------------------------------

def posenc_fingerprint(lape_k: int, rwse_steps: int) -> dict:
    import scipy
    return {"schema": POSENC_SCHEMA, "lape_k": int(lape_k), "rwse_steps": int(rwse_steps),
            "numpy": np.__version__, "scipy": scipy.__version__}


def posenc_path(dirpath: str, edge_index: np.ndarray, n: int, lape_k: int, rwse_steps: int) -> str:
    key = content_key("posenc", np.int64(n), np.int64(lape_k), np.int64(rwse_steps), edge_index)
    return os.path.join(dirpath, f"posenc_{key}.npz")


def posenc_load(path: str, lape_k: int, rwse_steps: int, n: int):
    """指纹逐项核对(G3):schema / lape_k / rwse_steps / numpy / scipy / n,任一不匹配返回 None(重算)。"""
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            fp = posenc_fingerprint(lape_k, rwse_steps)
            for k, v in fp.items():
                if str(z[f"fp_{k}"]) != str(v):
                    return None
            if int(z["fp_n"]) != int(n):
                return None
            lape, valid, rwse = z["lape"], z["valid"], z["rwse"]
    except Exception:
        return None
    if lape.shape != (n, lape_k) or valid.shape != (n, lape_k) or rwse.shape != (n, rwse_steps):
        return None
    return lape, valid, rwse


def posenc_save(path: str, lape: np.ndarray, valid: np.ndarray, rwse: np.ndarray,
                lape_k: int, rwse_steps: int, n: int) -> None:
    """原子写(tmp + os.replace),多进程并发安全;失败静默放弃(缓存缺席≠错误)。"""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fp = {f"fp_{k}": str(v) for k, v in posenc_fingerprint(lape_k, rwse_steps).items()}
        tmp = f"{path}.{os.getpid()}.tmp.npz"   # np.savez 会自动补 .npz,临时名必须已带后缀
        np.savez(tmp, lape=lape, valid=valid, rwse=rwse, fp_n=int(n), **fp)
        os.replace(tmp, path)
    except OSError:
        pass
