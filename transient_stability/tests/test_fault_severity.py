"""故障电气严重度侧通道特征单测(W2 强制配套)。

泄漏红线:特征 = 合法输入(已解档案 node/branch_feat + 故障参数 fbr/K)的**确定函数**;
函数签名只吃 (raw, fault_branch_row, k_frac),结构上无从访问任何暂态产物(标签结果字段)→
无泄漏由构造保证。此处锁:确定性、Ybus 对称、对静态阻抗/故障位置/故障前电压的敏感、
K% 拆线的边界一致(K=0 = 母线 i)。
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from gridfm.psasp_io.case import PsaspCase
from gridfm.psasp_io.catalog import Catalog
from gridfm.reason.fault_severity import (
    build_ybus, compute_fault_severity_features, _zbus, _z_th_at_fault, FEATURE_NAMES,
)
from gridfm.schema.assemble import build_sample
from gridfm.schema.fields import NODE_TABLE, BRANCH_TABLE

FIX = Path(__file__).parent / "fixtures" / "cepri36_golden"


def _sample():
    case = PsaspCase(FIX)
    return build_sample(case, Catalog(case), sample_id="fs_test", family_id="cepri36")


def test_ybus_symmetric_and_sized():
    raw = _sample()
    Y = build_ybus(raw)
    assert Y.shape == (raw.n_nodes, raw.n_nodes)
    assert np.allclose(Y, Y.T), "无相移变压器/线路网络 Ybus 必对称"


def test_features_deterministic_and_finite():
    raw = _sample()
    a = compute_fault_severity_features(raw, 0, 0.2)
    b = compute_fault_severity_features(raw, 0, 0.2)
    assert set(a) == set(FEATURE_NAMES)
    assert a == b, "同输入必给同输出(确定函数,无 RNG)"
    assert all(np.isfinite(v) for v in a.values())


def test_z_th_k0_equals_bus_diagonal():
    """K=0(线首)戴维南阻抗 = Zbus[i,i];K∈(0,1) 拆线后一般不同。"""
    raw = _sample()
    Y = build_ybus(raw)
    Z = _zbus(Y)
    ei, ej = raw.edge_index
    fbr = 0
    i, j = int(ei[fbr]), int(ej[fbr])
    zL = complex(raw.branch_feat[fbr, BRANCH_TABLE.offset_of("r")],
                 raw.branch_feat[fbr, BRANCH_TABLE.offset_of("x")])
    z0 = _z_th_at_fault(Y, Z, i, j, zL, 0.0)
    assert abs(z0 - Z[i, i]) < 1e-9
    zmid = _z_th_at_fault(Y, Z, i, j, zL, 0.5)
    assert abs(zmid - Z[i, i]) > 1e-9, "线中点故障 Z_th 应不同于线首(拆线加内节点)"


def test_sensitive_to_line_impedance():
    """加大某线电抗 → 该网络 Z_th 变(结构量对静态阻抗敏感;正确性 sanity)。"""
    raw = _sample()
    base = compute_fault_severity_features(raw, 0, 0.0)
    raw2 = copy.deepcopy(raw)
    xcol = BRANCH_TABLE.offset_of("x")
    raw2.branch_feat[:, xcol] *= 2.0        # 全网电抗翻倍
    hi = compute_fault_severity_features(raw2, 0, 0.0)
    assert abs(hi["fs_zth_mag"] - base["fs_zth_mag"]) > 1e-6, "阻抗变 Z_th 必变"


def test_sensitive_to_prefault_fault_bus_voltage():
    """改故障母线故障前电压 v → fs_v_fault 与 scc_current/capacity 变(V_i 依赖,运行点信号来源)。"""
    raw = _sample()
    ei = raw.edge_index[0]
    i = int(ei[0])
    vcol = NODE_TABLE.offset_of("v")
    base = compute_fault_severity_features(raw, 0, 0.0)
    raw2 = copy.deepcopy(raw)
    raw2.node_feat[i, vcol] = float(raw2.node_feat[i, vcol]) * 0.8 + 0.05   # 压低故障母线电压
    lo = compute_fault_severity_features(raw2, 0, 0.0)
    assert lo["fs_v_fault"] != base["fs_v_fault"]
    assert abs(lo["fs_scc_current"] - base["fs_scc_current"]) > 1e-6, "V_i 变 → scc 必变"
    # 纯结构量(zth/zline)不因电压变(它们不读 v)
    assert abs(lo["fs_zth_mag"] - base["fs_zth_mag"]) < 1e-9, "Z_th 是结构量,不随故障前电压变"


def test_features_invariant_to_unused_static_bounds():
    """不使用的静态列(pmax/qmax 出力上下限)扰动 → 特征逐位不变(只读该读的量,无旁读)。"""
    raw = _sample()
    base = compute_fault_severity_features(raw, 3, 0.2)
    raw2 = copy.deepcopy(raw)
    for name in ("pmax", "qmax", "pmin", "qmin"):
        raw2.node_feat[:, NODE_TABLE.offset_of(name)] += 3.3
    after = compute_fault_severity_features(raw2, 3, 0.2)
    assert after == base, "特征不得依赖未使用的静态出力界(防旁读)"
