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
