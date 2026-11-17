"""文件系统健壮操作(Windows 并行批产的竞态兜底;2026-07-13 待办清账)。

背景:家族级并行批产(gen_families,6 进程)下 `shutil.rmtree` + `copytree` 会偶发
`PermissionError/OSError`——Windows 上文件句柄的释放是异步的(杀软扫描、索引器、
刚退出的 PSASP 进程尚未完全收尸),`rmtree` 返回后目录项可能仍短暂存在,紧接着的
`copytree` 撞上 `FileExistsError`;或反过来 `rmtree` 撞上被占用的文件。整批因此
损失家族(07-09 批产 83/84,变更记录)。

策略:重试 + 指数退避,只兜底**瞬态**错误;重试耗尽仍失败即抛出(不静默降级——
静默跳过家族 = 数据集组成静默漂移,正是 D16 纪律①要防的)。
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

RETRIES = 5
BASE_DELAY_S = 0.3
_TRANSIENT = (PermissionError, FileExistsError, FileNotFoundError, OSError, shutil.Error)


def robust_rmtree(path, retries: int = RETRIES, base_delay: float = BASE_DELAY_S) -> None:
    """删目录树;瞬态占用重试。目录不存在视为成功。"""
    path = Path(path)
    for k in range(retries):
        if not path.exists():
            return
        try:
            shutil.rmtree(path)
            return
        except _TRANSIENT:
            if k == retries - 1:
                raise
            time.sleep(base_delay * (2 ** k))


def robust_copytree(src, dst, retries: int = RETRIES,
                    base_delay: float = BASE_DELAY_S) -> Path:
    """dst 存在则先删,再整拷;瞬态错误重试(每轮重新清理 dst,避免半拷残留)。"""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"源目录不存在:{src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    for k in range(retries):
        try:
            robust_rmtree(dst, retries=retries, base_delay=base_delay)
            shutil.copytree(src, dst)
            return dst
        except _TRANSIENT:
            if k == retries - 1:
                raise
            time.sleep(base_delay * (2 ** k))
    return dst           # 不可达(循环内必 return 或 raise)


def robust_rename(src, dst, retries: int = RETRIES, base_delay: float = BASE_DELAY_S) -> Path:
    """重命名/移动目录树;瞬态占用重试。

    [外协包 F,2026-08-28 实测补入] 本模块开头列的三种异步句柄释放(杀软扫描、索引器、
    刚退出的 PSASP 进程尚未收尸)同样会打中 `Path.rename`,而此前只有 rmtree/copytree
    做了兜底。实证:护栏轮 `f_g6_guard_s5001_blend05` 在把 work 目录晋升为 delivered 时,
    前两个需求成功、第三个抛
        PermissionError: [WinError 32] 另一个程序正在使用此文件
    整轮就此中止(该机 Defender 实时保护为开启状态)。属**瞬态**,重试即可;
    与 rmtree 同策略:重试耗尽仍失败即抛出,不静默降级(静默跳过 = 交付集合静默漂移)。
    """
    src, dst = Path(src), Path(dst)
    for k in range(retries):
        try:
            src.rename(dst)
            return dst
        except _TRANSIENT:
            if k == retries - 1:
                raise
            time.sleep(base_delay * (2 ** k))
    return dst           # 不可达(循环内必 return 或 raise)
