# -*- coding: utf-8 -*-
r"""PSASP 加密狗"不响应"守卫（2026-09-05，N-k 其他推理器线）。

事实：IEEE300 N-0 跑分第 19 例，引擎没写 LF.CAL、lfreport.lis 末尾是"软件狗不响应，请检查加密狗!"；
仓库 `gridfm.solver.psasp._license_dead` 只认 "slm_init error"，这条没被识别，被当成"不收敛"记进结果，
跑分器随后在阶梯里因 vmin=None 崩溃。这类瞬时故障必须与真发散区分：
  · 判不收敛 且 报告文件里有加密狗故障文字 → 等 wait 秒重试（引擎每次解算前自清场，重试安全）；
  · 重试用尽仍是狗故障 → 抛 LicenseError 熔断（与仓库口径一致：继续 = 制造假结果）；
  · 判不收敛但报告里没有狗故障文字 → 原样返回（真发散）。
用法：
    rf = guarded(lambda: psasp_referee(work, solver), work, lambda r: bool(r.get("converged")), tag=...)
    pf = guarded(lambda: solver.run_powerflow(work, timeout=t), work, lambda p: bool(p.converged), tag=...)
"""
from __future__ import annotations

import time
from pathlib import Path

from gridfm.solver.psasp import LicenseError

DONGLE_PATTERNS = ("软件狗不响应", "请检查加密狗", "slm_init")
REPORT_FILES = ("lfreport.lis", "LFERR.LIS")


def dongle_failed(case_dir) -> bool:
    for name in REPORT_FILES:
        p = Path(case_dir) / name
        if not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="gb18030", errors="replace")
        except OSError:
            continue
        if any(k in txt for k in DONGLE_PATTERNS):
            return True
    return False


def guarded(call, case_dir, is_converged, retries: int = 3, wait: float = 20.0, tag: str = ""):
    """call() 解一次；不收敛且是狗故障 → 等 wait 秒重试；重试用尽仍是狗故障 → LicenseError。"""
    last = None
    for i in range(retries + 1):
        last = call()
        if is_converged(last) or not dongle_failed(case_dir):
            return last
        print(f"[psasp-guard] {tag} 加密狗不响应（第 {i + 1} 次）；{wait:.0f}s 后重试", flush=True)
        time.sleep(wait)
    raise LicenseError(f"加密狗不响应，重试 {retries} 次仍失败：{tag}")
