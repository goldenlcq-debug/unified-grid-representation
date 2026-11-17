"""P1 校准修复单测(scripts/calibrate_ts.py 纯函数:温度拟合/ECE/召回-复核)。

锁定:温度缩放拟合能复原已知 T;完美校准 ECE≈0;召回-复核比例单调且神谕排序最省。
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("calibrate_ts", REPO / "scripts" / "calibrate_ts.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


def test_fit_temperature_recovers_known():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, size=8000)          # 真 logit
    p = cal._sig(z)
    y = (rng.uniform(size=z.size) < p).astype(float)  # 由真概率生成标签 → 真 T=1
    T = cal.fit_temperature(z, y)
    assert abs(T - 1.0) < 0.15
    # 人为把 logit 放大 3× (过度自信) → 拟合应给出 T≈3 把它降回
    T3 = cal.fit_temperature(z * 3.0, y)
    assert 2.3 < T3 < 3.8


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=p.size) < p).astype(float)  # 完美校准
    assert cal.ece(p, y, n_bins=15) < 0.02
    # 系统性过高:p 全 0.99 但真实一半 → ECE 大
    assert cal.ece(np.full(1000, 0.99), np.r_[np.ones(500), np.zeros(500)]) > 0.4


def test_recall_cost_monotone_and_oracle_cheapest():
    y = np.r_[np.ones(300), np.zeros(700)]
    oracle = y.copy()                        # 神谕分数 = 标签
    rc = cal.recall_cost(oracle, y, recalls=(0.9, 0.95, 0.99))
    # 神谕:达 99% 召回只需复核≈失稳占比 0.3(把 300 个失稳排最前)
    assert rc[0.99] <= 0.31
    assert rc[0.9] <= rc[0.95] <= rc[0.99]   # 召回越高复核越多
    rng = np.random.default_rng(2)
    rand = rng.uniform(size=y.size)          # 随机分数:达 R 召回需复核≈R
    rr = cal.recall_cost(rand, y, recalls=(0.9,))
    assert rr[0.9] > rc[0.9]                  # 随机远差于神谕


def test_recall_cost_no_positives():
    assert cal.recall_cost(np.arange(10.0), np.zeros(10))[0.9] is None
