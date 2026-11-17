"""动态参数/能量函数侧通道特征单测(审查 A2 强制配套)。

泄漏红线:特征 = 合法输入(v0.3 给定列 + 故障参数 fbr/dur)的确定函数;签名只吃
(full, fault_branch_row, dur),结构上无从访问暂态产物。此处锁:v0 视图拒收(防静默
退化成假 0)、确定性、**读 DATALIB 正确列而非 legacy 假 0 列**(A2 的实体)、dur²
能量代理、防旁读。
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from gridfm.psasp_io.case import PsaspCase
from gridfm.psasp_io.catalog import Catalog
from gridfm.reason.dyn_side import (
    DYN_FEATURE_NAMES, compute_dyn_side_features, dyn_zbus)
from gridfm.schema.assemble import build_sample
from gridfm.schema.fields import NODE_TABLE_V03

FIX = Path(__file__).parent / "fixtures" / "cepri36_golden"


def _full():
    case = PsaspCase(FIX)
    return build_sample(case, Catalog(case), sample_id="dyn_test", family_id="cepri36",
                        schema="v0.3")


def test_v0_view_rejected():
    """25 列 v0 视图必须响亮拒绝——静默算出全 0 特征比报错危险。"""
    case = PsaspCase(FIX)
    v0 = build_sample(case, Catalog(case), sample_id="t", family_id="cepri36", schema="v0")
    with pytest.raises(ValueError):
        compute_dyn_side_features(v0, 0, 0.2)


def test_features_deterministic_and_finite():
    full = _full()
    a = compute_dyn_side_features(full, 0, 0.2)
    b = compute_dyn_side_features(full, 0, 0.2)
    assert set(a) == set(DYN_FEATURE_NAMES)
    assert a == b, "同输入必给同输出(确定函数,无 RNG)"
    assert all(np.isfinite(v) for v in a.values())


def test_reads_datalib_columns_not_legacy():
    """A2 的实体:cepri36 legacy tj 列是内联假 0,正确值在 tj_lib——特征必须取自后者,
    且对 legacy tj 列的扰动逐位不变(fault_severity 的 fs_total_inertia 在 cepri36 上
    恒 0,正是因为它读了 legacy 列)。"""
    full = _full()
    gm = full.node_feat[:, NODE_TABLE_V03.offset_of("has_gen")] > 0.5
    assert (full.node_feat[gm, NODE_TABLE_V03.offset_of("tj")] == 0.0).all(), \
        "前提:cepri36 legacy tj 列为内联假 0(若不再成立,本测与 A2 动因需复核)"
    f = compute_dyn_side_features(full, 0, 0.2)
    assert f["dyn_tj_sum"] > 0, "tj_lib 正确值必须进特征(legacy 列会给 0)"
    full2 = copy.deepcopy(full)
    full2.node_feat[:, NODE_TABLE_V03.offset_of("tj")] += 7.7   # 扰动 legacy 假 0 列
    assert compute_dyn_side_features(full2, 0, 0.2) == f, "特征不得依赖 legacy tj 列"


def test_cached_zbus_bitwise_identical_to_inline():
    """缓存 Zbus 路径(按方式复用)与内联现算路径必须逐位一致——否则批产提速会污染
    实验对照(同一批三种子须同一份特征)。"""
    full = _full()
    zc = dyn_zbus(full)
    for fbr in range(min(6, full.n_branches)):
        for dur in (0.1, 0.2, 0.35):
            a = compute_dyn_side_features(full, fbr, dur)
            b = compute_dyn_side_features(full, fbr, dur, zbus=zc)
            assert a == b, f"缓存/内联 Zbus 特征不一致 @ fbr={fbr} dur={dur}"


def test_acc_proxy_scales_with_dur_squared():
    """能量代理 = pg_near·dur²/(2Tj_near):dur 翻倍 → 该项 ×4,其余特征不随 dur 变。"""
    full = _full()
    a = compute_dyn_side_features(full, 0, 0.2)
    b = compute_dyn_side_features(full, 0, 0.4)
    if a["dyn_acc_proxy"] > 0:
        assert abs(b["dyn_acc_proxy"] / a["dyn_acc_proxy"] - 4.0) < 1e-6
    for k in DYN_FEATURE_NAMES:
        if k != "dyn_acc_proxy":
            assert a[k] == b[k], f"{k} 不该随 dur 变"


def test_invariant_to_unused_columns():
    """不使用的列(出力界、电压)扰动 → 特征逐位不变(防旁读;电气距离只由拓扑+阻抗定)。"""
    full = _full()
    base = compute_dyn_side_features(full, 3, 0.2)
    full2 = copy.deepcopy(full)
    for name in ("pmax", "qmax", "pmin", "qmin", "v", "theta"):
        full2.node_feat[:, NODE_TABLE_V03.offset_of(name)] += 0.37
    assert compute_dyn_side_features(full2, 3, 0.2) == base


def test_fault_position_sensitivity():
    """换故障支路 → 电气距离类特征应变(近区惯量/最近机距离),全网聚合量不变。"""
    full = _full()
    a = compute_dyn_side_features(full, 0, 0.2)
    diffs = []
    for fbr in range(1, min(8, full.n_branches)):
        b = compute_dyn_side_features(full, fbr, 0.2)
        assert b["dyn_tj_sum"] == a["dyn_tj_sum"], "全网惯量与故障位置无关"
        diffs.append(abs(b["dyn_near_zdist"] - a["dyn_near_zdist"])
                     + abs(b["dyn_prox_tj"] - a["dyn_prox_tj"]))
    assert max(diffs) > 1e-9, "电气距离类特征应对故障位置敏感"
