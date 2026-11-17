"""判稳推理器 v0 泄漏与掩码单测(交接 15 版任务 ③ 强制配套)。

泄漏红线(与潮流任务不同):判稳输入 = 已解方式标准档案 + 故障参数,**暂态产物
(标签行一切结果字段)不得进入任何输入张量**。核心测试 = 打乱标签行全部结果字段,
输入侧(故障支路行、事件参数、档案样本)必须逐位不变。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gridfm.psasp_io.case import PsaspCase
from gridfm.psasp_io.catalog import Catalog
from gridfm.reason.ts_task import (
    CCT_T_MAX, EVENT_PARAM_DIM, TS_FAULT_KEYS, TS_SUBTYPES, TSReasoner,
    build_ts_example, cct_join_tag, fault_edge_pointer, gen_node_mask,
    group_multi_hot, ts_event_params, ts_loss,
)
from gridfm.schema.assemble import build_sample

FIX = Path(__file__).parent / "fixtures" / "cepri36_golden"


@pytest.fixture(scope="module")
def sample_and_catalog():
    case = PsaspCase(FIX)
    cat = Catalog(case)
    s = build_sample(case, cat, sample_id="ts_fix", family_id="cepri36")
    return s, cat


def _label_row(**over):
    row = {
        "tag": "ts_fix_L29-27-23_d100", "sample_id": "ts_fix", "battery": "cepri36",
        "i": 29, "j": 27, "id": 23, "k_pct": 0.0, "ts": 0.1, "te": 0.2, "dur": 0.1,
        "t_end": 12.0, "feasible": True, "timeout": False,
        # ---- 以下全部为暂态产物(结果字段),泄漏测试的打乱对象 ----
        "stable": False, "instability_types": ["angle"], "voltage_modes": [],
        "angle_peak_deg": 612.0, "vmin_pu": 0.42, "vterm_min_pu": 0.55,
        "last1s_vmin": 0.91, "recover_evaluable": True, "freq_source": "coi_sustained",
        "v_recover_source": "sme_terminal_exact", "criteria_version": "v1_20260712",
    }
    row.update(over)
    return row


def test_ts_inputs_invariant_to_transient_outcomes(sample_and_catalog):
    """核心泄漏单测:打乱一切结果字段,输入侧必须逐位不变(输入只准依赖故障参数)。"""
    s, cat = sample_and_catalog
    feat_before = s.node_feat.copy()
    row_a = _label_row()
    ex_a = build_ts_example(row_a, s, cat)
    # 打乱全部非故障键结果字段(feasible/timeout 是分流门,单独保持)
    scrambled = {}
    for k, v in row_a.items():
        if k in TS_FAULT_KEYS or k in ("feasible", "timeout"):
            scrambled[k] = v
        elif isinstance(v, bool):
            scrambled[k] = not v
        elif isinstance(v, (int, float)):
            scrambled[k] = (v or 1.0) * -7.77
        elif isinstance(v, list):
            scrambled[k] = ["frequency", "voltage"]
        else:
            scrambled[k] = "scrambled"
    ex_b = build_ts_example(scrambled, s, cat)
    assert ex_a["fault_branch_row"] == ex_b["fault_branch_row"]
    np.testing.assert_array_equal(ex_a["event_params"], ex_b["event_params"])
    # 档案样本不得被就地修改(编码器路输入 = 样本本身)
    np.testing.assert_array_equal(s.node_feat, feat_before)
    # 结果字段确实进了标签侧(证明打乱生效而非例子退化)
    assert ex_a["y_unstable"] != ex_b["y_unstable"] or not np.array_equal(
        ex_a["y_subtypes"], ex_b["y_subtypes"])


def test_ts_event_params_only_fault_params():
    """事件侧特征 = [K/100, Ts, Te, dur] 恰好四维,不含任何其他字段。"""
    row = _label_row(k_pct=20.0, ts=0.1, te=0.35, dur=0.25,
                     angle_peak_deg=9999.0, vmin_pu=-5.0)
    v = ts_event_params(row)
    assert v.shape == (EVENT_PARAM_DIM,)
    np.testing.assert_allclose(v, [0.20, 0.1, 0.35, 0.25], rtol=1e-6)


def test_ts_infeasible_rows_dropped(sample_and_catalog):
    s, cat = sample_and_catalog
    assert build_ts_example(_label_row(feasible=False), s, cat) is None
    assert build_ts_example(_label_row(timeout=True), s, cat) is None


def test_group_multi_hot_alignment_and_guard(sample_and_catalog):
    """(bus, g) → 节点行 bus−1,必须命中 has_gen 行;非发电机母线即 raise。"""
    s, _cat = sample_and_catalog
    gm = gen_node_mask(s)
    gen_bus = int(np.where(gm)[0][0]) + 1          # 任一发电机母线(1 起)
    nongen_bus = int(np.where(~gm)[0][0]) + 1
    y, breakup = group_multi_hot(
        {"tag": "t", "unstable_group": [[gen_bus, 1]], "breakup": False}, s)
    assert y[gen_bus - 1] == 1.0 and y.sum() == 1.0 and not breakup
    with pytest.raises(ValueError):
        group_multi_hot({"tag": "t", "unstable_group": [[nongen_bus, 1]],
                         "breakup": False}, s)


def test_cct_join_tag_strips_duration_suffix():
    assert cct_join_tag("cepri36_00000_L3-8-11_d100") == "cepri36_00000_L3-8-11"
    assert cct_join_tag("cepri36_00000_L3-8-11") == "cepri36_00000_L3-8-11"


def _loss_batch(**over):
    """ts_loss 的最小批(2 图:图0 两节点,图1 一节点;图0 首节点为发电机)。"""
    d = {
        "y_unstable": torch.tensor([1.0, 0.0]),
        "y_subtypes": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0, 0.0]]),
        "y_cct": torch.tensor([0.2, 0.0]),
        "cct_valid": torch.tensor([True, False]),
        "cct_censored": torch.tensor([False, False]),
        "group_valid": torch.tensor([True, False]),
        "gen_mask": torch.tensor([True, False, True]),
        "y_group": torch.tensor([1.0, 0.0, 0.0]),
        "batch": torch.tensor([0, 0, 1]),
    }
    d.update(over)
    return SimpleNamespace(**d)


def test_ts_loss_type_and_group_masks():
    """类型头仅失稳行计损;机群头仅有效行 × 发电机位——被掩码位置放 NaN 仍须有限。"""
    s = torch.tensor([0.3, -0.2])
    t = torch.tensor([[0.1, -0.1, 0.2, 0.0, 0.1], [float("nan")] * 5])  # 稳行 NaN
    c = torch.tensor([0.19, float("nan")])                     # 无 CCT 标签行为 NaN
    g = torch.tensor([0.5, float("nan"), float("nan")])        # 非发电机/无效图行为 NaN
    loss, diag = ts_loss(s, t, c, g, _loss_batch())
    assert math.isfinite(float(loss)), "掩码位置的 NaN 不得进入损失"
    assert {"stable", "type", "cct", "group"} <= set(diag)


def test_ts_loss_all_stable_skips_type_and_group():
    b = _loss_batch(y_unstable=torch.tensor([0.0, 0.0]),
                    group_valid=torch.tensor([False, False]),
                    cct_valid=torch.tensor([False, False]))
    s = torch.tensor([0.0, 0.0])
    t = torch.full((2, 5), float("nan"))
    c = torch.full((2,), float("nan"))
    g = torch.full((3,), float("nan"))
    loss, diag = ts_loss(s, t, c, g, b)
    assert math.isfinite(float(loss))
    assert "type" not in diag and "group" not in diag and "cct" not in diag


def test_ts_loss_censored_cct_one_sided_hinge():
    """删失:预测 ≥ t_max 零罚;预测 < t_max 才罚(禁把 t_max 当精确真值)。"""
    b = _loss_batch(cct_valid=torch.tensor([True, False]),
                    cct_censored=torch.tensor([True, False]),
                    y_cct=torch.tensor([CCT_T_MAX, 0.0]),
                    y_unstable=torch.tensor([0.0, 0.0]),
                    group_valid=torch.tensor([False, False]))
    zeros = (torch.zeros(2), torch.zeros(2, 5), torch.zeros(3))
    above = torch.tensor([CCT_T_MAX + 0.1, 0.0])
    below = torch.tensor([CCT_T_MAX - 0.2, 0.0])
    _, diag_hi = ts_loss(zeros[0], zeros[1], above, zeros[2], b)
    _, diag_lo = ts_loss(zeros[0], zeros[1], below, zeros[2], b)
    assert diag_hi["cct"] == pytest.approx(0.0, abs=1e-6)
    assert diag_lo["cct"] > 0.1


def test_ts_loss_subtype_pos_weight_default_bitwise_identical():
    """B1 守卫:不给 type_pos_weight(或给 None)= 历史行为逐位不变。"""
    s = torch.tensor([0.3, -0.2])
    t = torch.tensor([[0.1, -0.1, 0.2, 0.0, 0.1], [0.0] * 5])
    c = torch.tensor([0.19, 0.0])
    g = torch.tensor([0.5, 0.0, 0.0])
    l0, d0 = ts_loss(s, t, c, g, _loss_batch())
    l1, d1 = ts_loss(s, t, c, g, _loss_batch(), weights={"type_pos_weight": None})
    assert float(l0) == float(l1) and d0["type"] == d1["type"]
    ones, d_ones = ts_loss(s, t, c, g, _loss_batch(),
                           weights={"type_pos_weight": [1.0] * 5})
    assert d_ones["type"] == pytest.approx(d0["type"], rel=1e-6), \
        "全 1 权重必须与无权重一致(pos_weight 只乘正类项)"


def test_ts_loss_group_pos_weight_default_and_scaling():
    """B6 守卫:group_pos_weight 缺省/None/1.0 = 历史行为逐位不变;>1 时含正标签的
    机群损失增大(图0 首节点为发电机且 y_group=1)。"""
    s = torch.tensor([0.0, 0.0])
    t = torch.zeros(2, 5)
    c = torch.tensor([0.19, 0.0])
    g = torch.tensor([0.5, 0.0, 0.0])
    b = _loss_batch()
    _, d0 = ts_loss(s, t, c, g, b)
    _, d_none = ts_loss(s, t, c, g, b, weights={"group_pos_weight": None})
    _, d_one = ts_loss(s, t, c, g, b, weights={"group_pos_weight": 1.0})
    assert d_none["group"] == d0["group"]
    assert d_one["group"] == pytest.approx(d0["group"], rel=1e-6)
    _, d_w = ts_loss(s, t, c, g, b, weights={"group_pos_weight": 3.0})
    assert d_w["group"] > d0["group"], "正标签存在时加权必须放大机群损失"


def test_eval_only_batches_lock():
    """B2 守卫:定向补测批(BATTERY_V4 产物)只准评估、永不入训的代码级锁——
    ①默认装载(only_batches=None)不放行;②白名单不点名不放行;③显式点名才放行(评估器路径);
    ④训练入口 assert 对点名响亮拒绝;⑤普通批行为不变。"""
    import importlib.util
    from pathlib import Path as _P
    spec = importlib.util.spec_from_file_location(
        "tsr_lock", _P(__file__).resolve().parent.parent / "scripts" / "ts_reasoner.py")
    tsr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tsr)
    sub = "ts_labels_cepri36_sub25"
    assert sub in tsr.EVAL_ONLY_BATCHES
    assert not tsr._batch_allowed(sub, None), "默认装载不得包含定向批"
    assert not tsr._batch_allowed(sub, ["ts_labels_cepri36"]), "未点名不得包含"
    assert tsr._batch_allowed(sub, [sub]), "评估器显式点名必须放行"
    assert tsr._batch_allowed("ts_labels_cepri36", None), "普通批默认放行(行为不变)"
    assert not tsr._batch_allowed("ts_labels_cepri36", ["ts_labels_ieee39"]), "普通批白名单不变"
    import pytest as _pt
    with _pt.raises(SystemExit):
        tsr.assert_no_eval_only_batches(["ts_labels_cepri36", sub])
    tsr.assert_no_eval_only_batches(["ts_labels_cepri36"])   # 无定向批不拦
    tsr.assert_no_eval_only_batches(None)                    # 缺省不拦


def test_cct_quality_gate_predicate():
    """B3 守卫:双验质量门——verify.ok=False 拒;ok=True 或无 verify(旧批/删失)放行。"""
    import importlib.util
    from pathlib import Path as _P
    spec = importlib.util.spec_from_file_location(
        "tsr_gate", _P(__file__).resolve().parent.parent / "scripts" / "ts_reasoner.py")
    tsr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tsr)
    assert tsr._cct_quality_ok({"cct": 0.1, "verify": {"ok": True}})
    assert not tsr._cct_quality_ok({"cct": 0.055, "verify": {"ok": False}})
    assert tsr._cct_quality_ok({"cct": 0.5, "is_censored": True})   # 删失行无 verify
    assert tsr._cct_quality_ok({"cct": 0.1})                        # 旧批无 verify


def test_ts_loss_subtype_pos_weight_scales_positive_term_only():
    """B1 守卫:正类权重只放大有正标签的子类损失;全负标签子类不受影响。"""
    s = torch.tensor([0.0, 0.0])
    t = torch.zeros(2, 5)
    c = torch.tensor([0.19, 0.0])
    g = torch.zeros(3)
    b = _loss_batch()          # 图0 失稳,y_subtypes 仅第 0 类(angle)为正
    _, d_base = ts_loss(s, t, c, g, b)
    _, d_w0 = ts_loss(s, t, c, g, b, weights={"type_pos_weight": [3.0, 1, 1, 1, 1]})
    assert d_w0["type"] > d_base["type"], "正类所在子类加权后损失必须增大"
    _, d_w4 = ts_loss(s, t, c, g, b, weights={"type_pos_weight": [1, 1, 1, 1, 3.0]})
    assert d_w4["type"] == pytest.approx(d_base["type"], rel=1e-6), \
        "全负子类(f_low 无正标签)加权不得改变损失"


def test_fault_edge_pointer_two_graphs():
    """两图批(图0 三条边、图1 两条边,均对称化前计):段起点折算正确。"""
    edge_batch = torch.tensor([0] * 6 + [1] * 4)   # 对称化后 2M 行
    ptr = fault_edge_pointer(edge_batch, torch.tensor([2, 1]))
    assert ptr.tolist() == [2, 6 + 1]


def test_ts_reasoner_forward_shapes_and_n_independence():
    """前向形状 + 红线 3:同一权重服务不同 N/M 的图(输出逐 token,无固定 N 头)。"""
    torch.manual_seed(0)
    model = TSReasoner(d=16, n_queries=4, n_layers=1, n_heads=2)
    for n, m, b in ((5, 6, 2), (9, 12, 3)):
        h = torch.randn(n * b, 16)
        e = torch.randn(2 * m * b, 16)
        g = torch.randn(b, 2, 16)
        node_batch = torch.arange(b).repeat_interleave(n)
        edge_batch = torch.arange(b).repeat_interleave(2 * m)
        ptr = fault_edge_pointer(edge_batch, torch.zeros(b, dtype=torch.long))
        ev = torch.zeros(b, EVENT_PARAM_DIM)
        s, t, c, gr = model(h, g, e, node_batch, edge_batch, ptr, ev)
        assert s.shape == (b,) and t.shape == (b, len(TS_SUBTYPES))
        assert c.shape == (b,) and gr.shape == (n * b,)


def test_ts_subtype_labels_mapping():
    """细分五路标签推导(2026-07-14 裁决:低压两段合并;过压/高频/低频独立)。"""
    from gridfm.reason.ts_task import ts_subtype_labels

    # 低压(恢复段触发)+ 过压共存
    y = ts_subtype_labels(_label_row(
        instability_types=["voltage"], voltage_modes=["low_recover", "overvoltage"]))
    np.testing.assert_array_equal(y, [0, 1, 1, 0, 0])
    # 低压末态段同样归"低电压"(合并语义)
    y = ts_subtype_labels(_label_row(
        instability_types=["voltage"], voltage_modes=["low_steady"]))
    np.testing.assert_array_equal(y, [0, 1, 0, 0, 0])
    # voltage 触发但 modes 缺席(旧批)→ 保守记低电压
    y = ts_subtype_labels(_label_row(instability_types=["voltage"], voltage_modes=[]))
    np.testing.assert_array_equal(y, [0, 1, 0, 0, 0])
    # 频率方向 · 主口径 = 持续段时长(freq_hold_*),不是全窗极值
    y = ts_subtype_labels(_label_row(instability_types=["frequency"],
                                     freq_hold_high_s=0.8, freq_hold_low_s=0.0,
                                     fmin_pu=0.999, fmax_pu=1.046))
    np.testing.assert_array_equal(y, [0, 0, 0, 1, 0])
    # 关键场景:首摆瞬时冲高(不持续)+ 随后持续跌低 —— 极值代理会误标两个方向,
    # 严格口径只认持续段:必须只标 f_low
    y = ts_subtype_labels(_label_row(instability_types=["frequency"],
                                     freq_hold_high_s=0.04, freq_hold_low_s=0.62,
                                     fmin_pu=0.96, fmax_pu=1.05))
    np.testing.assert_array_equal(y, [0, 0, 0, 0, 1])
    # 双向都持续(罕见)→ 两位同置
    y = ts_subtype_labels(_label_row(instability_types=["frequency"],
                                     freq_hold_high_s=0.7, freq_hold_low_s=0.55,
                                     fmin_pu=0.96, fmax_pu=1.05))
    np.testing.assert_array_equal(y, [0, 0, 0, 1, 1])
    # 字段缺席(旧批未重判)→ 退极值代理,如实退化
    y = ts_subtype_labels(_label_row(instability_types=["frequency"],
                                     fmin_pu=0.97, fmax_pu=1.03))
    np.testing.assert_array_equal(y, [0, 0, 0, 1, 1])
    # 功角 + 频率共存
    y = ts_subtype_labels(_label_row(instability_types=["angle", "frequency"],
                                     freq_hold_high_s=0.0, freq_hold_low_s=0.9,
                                     fmin_pu=0.96, fmax_pu=1.0))
    np.testing.assert_array_equal(y, [1, 0, 0, 0, 1])
    # 稳例全零
    y = ts_subtype_labels(_label_row(instability_types=[]))
    np.testing.assert_array_equal(y, [0, 0, 0, 0, 0])


# ---------------- 2026-07-15 对抗审查修复的回归锁 ----------------

def test_join_key_disambiguates_by_battery():
    """tag 不含 K%:ieee39 v1(K=20)与 v2(K=0)的同线同方式同时长 tag 逐字相同。
    跨批 join 必须带 battery,否则 CCT 真值会挂到另一个电池的例子上(实测 300 撞)。"""
    from gridfm.reason.ts_task import join_key

    v1 = {"battery": "ieee39", "tag": "ieee39_00007_L34-28-23_d200", "k_pct": 20.0}
    v2 = {"battery": "ieee39_v2", "tag": "ieee39_00007_L34-28-23_d200", "k_pct": 0.0}
    assert v1["tag"] == v2["tag"], "前提:tag 确实撞车(不含 K%)"
    assert join_key(v1) != join_key(v2), "join 键必须能区分两个不同电池的同名 tag"
    # 时长后缀被剥掉(cct.jsonl 的 tag 不带 _dXXX),但 battery 仍在
    assert join_key(v1) == ("ieee39", "ieee39_00007_L34-28-23")


def test_group_label_excludes_fallback_rows(sample_and_catalog):
    """breakup / onset_at_edge 行的 unstable_group 是"窗口起始提名"语义(非末态失步)
    → 不得产训练标签(否则两种语义混教一个头);但旗标须如实携带。"""
    s, cat = sample_and_catalog
    gm = gen_node_mask(s)
    gen_bus = int(np.where(gm)[0][0]) + 1
    row = _label_row(stable=False, instability_types=["angle"])

    normal = {"tag": row["tag"], "unstable_group": [[gen_bus, 1]], "breakup": False}
    ex = build_ts_example(row, s, cat, coh_row=normal)
    assert ex["group_valid"] and ex["y_group"][gen_bus - 1] == 1.0

    for fb in ({"breakup": True, "onset_at_edge": False},
               {"breakup": False, "onset_at_edge": True}):
        coh = {"tag": row["tag"], "unstable_group": [[gen_bus, 1]], **fb}
        ex = build_ts_example(row, s, cat, coh_row=coh)
        assert not ex["group_valid"], f"回退行不得产机群标签:{fb}"
        assert ex["y_group"] is None
        assert ex["breakup"] == fb["breakup"], "breakup 旗标须如实携带(供计数)"


def _mk_batch(d: Path, n_raw=3, n_v1=3, summary_n=3, audit=None, ver=None, feasible=True):
    """造一个标签批目录(收口门的被测对象)。"""
    from gridfm.psasp_io.stana import CRITERIA_VERSION
    ver = ver or CRITERIA_VERSION
    d.mkdir(parents=True, exist_ok=True)
    raw = [{"tag": f"t{i}", "feasible": feasible, "timeout": False} for i in range(n_raw)]
    v1 = [{"tag": f"t{i}", "feasible": feasible, "timeout": False,
           "criteria_version": ver} for i in range(n_v1)]
    (d / "labels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in raw) + "\n", encoding="utf-8")
    (d / "labels_v1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in v1) + "\n", encoding="utf-8")
    (d / "labels_summary.json").write_text(
        json.dumps({"n": summary_n}), encoding="utf-8")
    a = audit if audit is not None else {"n_failed": 0, "n_audited": n_raw,
                                         "n_feasible_rows": n_raw,
                                         "n_skipped_missing_archive": 0,
                                         "t0_checked": n_raw}   # t0 覆盖门(见下)
    (d / "audit_report.json").write_text(json.dumps(a), encoding="utf-8")
    return d


def test_closeout_gate_rejects_in_flight_and_unaudited(tmp_path):
    """收口门(审查后重写)必须挡住的四类批 —— 每一类都曾能骗过旧门。"""
    import importlib.util
    from gridfm.psasp_io.stana import CRITERIA_VERSION
    spec = importlib.util.spec_from_file_location(
        "tsr_gate", Path(__file__).parent.parent / "scripts" / "ts_reasoner.py")
    tsr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tsr)
    gate = tsr._closed_out

    ok, why, rows, _ = gate(_mk_batch(tmp_path / "good"))
    assert ok, f"干净批必须放行,却被拒:{why}"
    assert len(rows) == 3

    # ⑤ t0 初值校验没跑(ts_audit 缺 --lib):n_failed=0 / n_audited==n_feasible /
    #    n_skipped=0 **全绿**,但 t0_checked=0 —— t0 是**唯一**拿轨迹去对独立事实源
    #    (LF.LP1 潮流解)的检查,其余锚点都是 STOUT↔STANA 同源互对(同源错误对它们
    #    不可见)。丢了它 = 又一个"0 行体检也算全绿"(2026-07-14 对抗审查 major)。
    d = _mk_batch(tmp_path / "no_t0",
                  audit={"n_failed": 0, "n_audited": 3, "n_feasible_rows": 3,
                         "n_skipped_missing_archive": 0, "t0_checked": 0})
    ok, why, _, _ = gate(d)
    assert not ok and "t0" in why, f"缺 t0 覆盖必须拒纳,实际:{ok}/{why}"

    # ① 在产批:labels.jsonl 长到 5 行,labels_v1/summary 还停在 3 行
    #    旧门:summary.n(3) == len(labels_v1)(3) → **放行**(时序反转)
    d = _mk_batch(tmp_path / "inflight", n_raw=5, n_v1=3, summary_n=3,
                  audit={"n_failed": 0, "n_audited": 3, "n_feasible_rows": 3,
                         "n_skipped_missing_archive": 0, "t0_checked": 3})
    ok, why, _, _ = gate(d)
    assert not ok and "tag 集不一致" in why, f"在产批必须拒纳,实际:{ok}/{why}"

    # ② 体检 0 行(全部缺归档):n_failed=0 但 n_audited=0
    #    旧门:只看 n_failed==0 → **放行**(以为全绿,其实一行没查)
    d = _mk_batch(tmp_path / "unaudited",
                  audit={"n_failed": 0, "n_audited": 0, "n_feasible_rows": 3,
                         "n_skipped_missing_archive": 3, "t0_checked": 0})
    ok, why, _, _ = gate(d)
    assert not ok and "跳过" in why, f"缺归档批必须拒纳,实际:{ok}/{why}"

    # ③ 生产未收尾:summary.n 与 labels.jsonl 行数不符(进程被杀/熔断时 summary 不写)
    d = _mk_batch(tmp_path / "nosummary", n_raw=3, n_v1=3, summary_n=999)
    ok, why, _, _ = gate(d)
    assert not ok and "summary" in why

    # ④ 判据版本陈旧(判据/schema 变更后未重判)——含"v0 冒充 v1"的直通行
    d = _mk_batch(tmp_path / "oldver", ver="v1_20260712")
    ok, why, _, _ = gate(d)
    assert not ok and "版本" in why
    d = _mk_batch(tmp_path / "passthrough", ver="v0_passthrough")
    ok, why, _, _ = gate(d)
    assert not ok, "feasible 行盖 v0_passthrough 戳(缺归档)的批必须拒纳"
    assert CRITERIA_VERSION in why


# ---------------- 拓扑维度 2:逐训练家族留出(2026-07-16) ----------------

def _tsr_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tsr_split", Path(__file__).parent.parent / "scripts" / "ts_reasoner.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_libs_carry_two_training_families_and_out_family():
    """拓扑维度 1→2:ieee118 成为第二个训练家族;ieee39 整族仍是 [外](D14 铁律)。
    新批(cepri36_k50 / ieee118_v1)必须挂进 LIBS,否则烧完的机时没有装入训练的路径。"""
    tsr = _tsr_module()
    assert tsr.OUT_FAMILIES == ("ieee39",)
    train_fams = [f for f in tsr.LIBS if f not in tsr.OUT_FAMILIES]
    assert set(train_fams) == {"cepri36", "ieee118"}, "训练粮仓须有两个拓扑家族"
    assert "ts_labels_ieee118_v1" in tsr.LIBS["ieee118"][1]
    assert "ts_labels_ieee118_v2" in tsr.LIBS["ieee118"][1], \
        "档2 重产批(121 抬压方式 12,705 例)须挂入,否则 11h 机时无装载路径"
    assert "ts_labels_cepri36_k50" in tsr.LIBS["cepri36"][1], "K=50 电池须入训(K 通道解封)"
    assert tsr.LIBS["ieee118"][0] == "lib118"


def test_ablate_graph_flag_is_off_by_default_and_actually_masks():
    """诊断开关 ablate_graph:①默认关 → 前向**逐位不变**(诊断代码不得污染生产口径);
    ②打开 → 档案的节点/支路/全局三段被屏蔽,只留事件 token(输出必然改变,且与
    "把这三段换成任意噪声"的结果一致 —— 证明它们真的没被读到)。"""
    torch.manual_seed(0)
    d, n, m, K, B = 32, 12, 20, 4, 2
    r = TSReasoner(d=d, n_queries=4, n_layers=1, event_dim=EVENT_PARAM_DIM).eval()
    h = torch.randn(n * B, d)
    e = torch.randn(m * B, d)
    g = torch.randn(B, K, d)
    nb = torch.repeat_interleave(torch.arange(B), n)
    eb = torch.repeat_interleave(torch.arange(B), m)
    ptr = torch.tensor([0, m])
    ep = torch.randn(B, EVENT_PARAM_DIM)
    with torch.no_grad():
        s0 = r(h, g, e, nb, eb, ptr, ep)[0]
        s_def = r(h, g, e, nb, eb, ptr, ep, ablate_graph=False)[0]
        s_abl = r(h, g, e, nb, eb, ptr, ep, ablate_graph=True)[0]
        # 把被屏蔽的三段整体换成噪声:开关打开时输出必须不变(= 真的没读)
        h2, e2, g2 = torch.randn_like(h), torch.randn_like(e), torch.randn_like(g)
        # 注:事件 token 由 e[ptr] 参与构造,故 e 的**故障行**必须保持原值
        e2[ptr] = e[ptr]
        s_abl2 = r(h2, g2, e2, nb, eb, ptr, ep, ablate_graph=True)[0]
    assert torch.equal(s0, s_def), "默认关:必须与不带该参数逐位一致"
    assert not torch.allclose(s0, s_abl), "打开后输出应当改变(图信息被切断)"
    assert torch.allclose(s_abl, s_abl2, atol=1e-5), \
        "屏蔽段换成噪声不改变输出 → 证明节点/支路/全局三段确实没被读到"


def test_ablate_event_edge_cuts_the_fault_branch_bypass():
    """ablate_event_edge:事件 token 由 e[fault_edge_ptr](故障支路完整档案嵌入,一条
    绕过档案交叉注意力的图信息旁路)拼事件参数构成。2026-07-14 对抗审查 critical:
    只用 ablate_graph 不干净(旁路未断)。本开关把 e[fault_edge_ptr] 置零 → 只留
    [K,Ts,Te,dur]。①默认关逐位不变;②与 ablate_graph 合用 = 零图信息,输出应只依赖
    事件参数(把整个 e 换成噪声、但 fault 行仍被开关清零 → 输出不变)。"""
    torch.manual_seed(1)
    d, n, m, K, B = 32, 12, 20, 4, 3
    r = TSReasoner(d=d, n_queries=4, n_layers=1, event_dim=EVENT_PARAM_DIM).eval()
    h = torch.randn(n * B, d)
    e = torch.randn(m * B, d)
    g = torch.randn(B, K, d)
    nb = torch.repeat_interleave(torch.arange(B), n)
    eb = torch.repeat_interleave(torch.arange(B), m)
    ptr = torch.tensor([0, m, 2 * m])
    ep = torch.randn(B, EVENT_PARAM_DIM)
    with torch.no_grad():
        s_def = r(h, g, e, nb, eb, ptr, ep)[0]
        s_off = r(h, g, e, nb, eb, ptr, ep, ablate_event_edge=False)[0]
        s_on = r(h, g, e, nb, eb, ptr, ep, ablate_event_edge=True)[0]
        # 零图信息:ablate_graph + ablate_event_edge 一起 → 只剩事件参数
        s_zero1 = r(h, g, e, nb, eb, ptr, ep,
                    ablate_graph=True, ablate_event_edge=True)[0]
        s_zero2 = r(torch.randn_like(h), torch.randn_like(g), torch.randn_like(e),
                    nb, eb, ptr, ep, ablate_graph=True, ablate_event_edge=True)[0]
    assert torch.equal(s_def, s_off), "默认关:逐位一致"
    assert not torch.allclose(s_def, s_on), "打开后输出改变(故障支路旁路被切断)"
    assert torch.allclose(s_zero1, s_zero2, atol=1e-5), \
        "零图信息档:整个档案换噪声不改变输出 → 只剩 [K,Ts,Te,dur]"


def test_average_precision_ties_give_no_free_skill():
    """2026-07-14 对抗审查 major(实测复现):AP 对并列分数按 stable-sort 保留原数组顺序,
    而 [外] 的排列 = 标签批拼接序(与真标签强相关)→ **恒定预测器凭空刷出判别力**
    (实测 [外] 基率 0.6635,恒定预测器 AP 0.6781 → 白得 1.022×)。
    修后:恒定分数的 AP 必须**恰等于基率**;打乱输入顺序不得改变 AP。"""
    from gridfm.reason.ts_task import average_precision as ap

    rng = np.random.default_rng(0)
    # 最恶劣情形:标签按批拼接(前段全负、后段全正),分数全同
    y = np.concatenate([np.zeros(300), np.ones(700)])
    const = np.ones(1000)
    assert ap(y, const) == pytest.approx(y.mean()), "恒定分数 = 零信息,AP 必须 = 基率"
    y2 = np.concatenate([np.ones(700), np.zeros(300)])          # 反序拼接
    assert ap(y2, const) == pytest.approx(y.mean()), "AP 不得依赖样本排列"
    # 部分并列:真实排序信息仍必须被计入(不能把 AP 打平)
    score = np.concatenate([np.zeros(300), np.ones(700)])       # 完美排序
    assert ap(y, score) == pytest.approx(1.0)
    perm = rng.permutation(1000)
    assert ap(y[perm], score[perm]) == pytest.approx(1.0), "置换不变"
    # 随机分数 → AP 应在基率附近(不得系统性高于基率)
    r = rng.random(1000)
    assert abs(ap(y, r) - y.mean()) < 0.05


def test_family_seed_is_process_stable():
    """**逐族 rng 的种子分量必须跨进程稳定**:内置 hash(str) 逐进程随机化
    (PYTHONHASHSEED)→ 同 --seed 在不同进程会切出不同留出集 → 续训校验必崩,
    且重跑时上一轮的留出方式会漂进 train(静默泄漏)。锚:固定值。"""
    tsr = _tsr_module()
    assert tsr.family_seed("cepri36") == 1755639859
    assert tsr.family_seed("ieee118") == 962468902
    assert tsr.family_seed("cepri36") != tsr.family_seed("ieee118")
    # 子进程复算(独立 PYTHONHASHSEED)必须给出同一值
    import subprocess
    import sys as _sys
    code = ("import hashlib;"
            "print(int.from_bytes(hashlib.sha256(b'cepri36').digest()[:4],'big'))")
    got = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True)
    assert int(got.stdout.strip()) == tsr.family_seed("cepri36")


def test_per_family_holdout_covers_every_training_family():
    """逐族留出:每个训练家族都必须有自己的留出方式(合并切分会让某族的留出占比
    随方式数漂移、甚至整族缺席 [内] → "同拓扑内是否成立"在那一族上无从检验)。
    这里复算 main() 的划分逻辑(与实现同源:每族独立 rng = [seed, family_seed(fam)])。"""
    tsr = _tsr_module()
    examples = ([{"family": "cepri36", "sample_id": f"c{i:03d}"} for i in range(88)
                 for _ in range(3)]
                + [{"family": "ieee118", "sample_id": f"e{i:03d}"} for i in range(90)
                   for _ in range(5)]
                + [{"family": "ieee39", "sample_id": f"n{i:03d}"} for i in range(100)])
    seed, frac = 0, 0.2
    train_fams = [f for f in tsr.LIBS if f not in tsr.OUT_FAMILIES]
    held_by_fam = {}
    for fam in train_fams:
        ids = sorted({x["sample_id"] for x in examples if x["family"] == fam})
        rng_f = np.random.default_rng([seed, tsr.family_seed(fam)])
        held_by_fam[fam] = sorted(rng_f.permutation(ids)[:max(1, int(len(ids) * frac))].tolist())
    assert set(held_by_fam) == {"cepri36", "ieee118"}, "两个训练家族都要留出"
    assert len(held_by_fam["cepri36"]) == 17 and len(held_by_fam["ieee118"]) == 18
    for fam, ids in held_by_fam.items():
        assert all(i.startswith("c" if fam == "cepri36" else "e") for i in ids), \
            "留出 id 不得跨族串味"
    held_all = {f: set(v) for f, v in held_by_fam.items()}
    train = [x for x in examples if x["family"] in held_all
             and x["sample_id"] not in held_all[x["family"]]]
    in_ex = [x for x in examples if x["family"] in held_all
             and x["sample_id"] in held_all[x["family"]]]
    out_ex = [x for x in examples if x["family"] in tsr.OUT_FAMILIES]
    assert not any(x["family"] in tsr.OUT_FAMILIES for x in train + in_ex), \
        "D14:[外] 整族禁入 train 与 [内]"
    assert len(out_ex) == 100
    assert {x["family"] for x in train} == {"cepri36", "ieee118"}
    assert {x["family"] for x in in_ex} == {"cepri36", "ieee118"}, "[内] 必须两族都有"


def test_random_split_leaks_modes_and_protects_out_family():
    """W4.2 文献对表口径(--split random):训练家族内**行级**随机切分。必须
    ①**方式泄漏**——同一 sample_id 的行同时落 train 与 [内](这正是文献随机切分"偏乐观"
    的来源,与生产口径 mode_holdout 的本质区别);②[外] ieee39 整族**不进** train/[内]
    (D14 铁律,随机切分只动训练家族,out_exs 口径不变);③**确定性**(同 seed 同族切分一致)。
    复算 main() 的 random 分支(与实现同源:rng=[seed, family_seed(fam), 20260715];
    fam_exs 先按 tag 排序,与装载顺序无关)。"""
    tsr = _tsr_module()

    def mk(fam, n_modes, rows_per):     # 每方式 rows_per 行(不同时长档),tag 全局唯一
        return [{"family": fam, "sample_id": f"{fam}_{i:03d}",
                 "tag": f"{fam}_{i:03d}_L1-2-3_d{r:03d}"}
                for i in range(n_modes) for r in range(rows_per)]

    examples = mk("cepri36", 20, 5) + mk("ieee118", 20, 5) + mk("ieee39", 30, 5)
    seed, frac, SALT = 0, 0.2, 20260715
    train_fams = [f for f in tsr.LIBS if f not in tsr.OUT_FAMILIES]

    def split_once():
        hbf, tr, ie = {}, [], []
        for fam in train_fams:
            fam_exs = sorted((x for x in examples if x["family"] == fam), key=lambda x: x["tag"])
            rng_f = np.random.default_rng([seed, tsr.family_seed(fam), SALT])
            n_held = max(1, int(len(fam_exs) * frac))
            held_pos = set(rng_f.permutation(len(fam_exs))[:n_held].tolist())
            for k, x in enumerate(fam_exs):
                (ie if k in held_pos else tr).append(x)
            hbf[fam] = sorted(fam_exs[k]["tag"] for k in held_pos)
        return hbf, tr, ie

    held_by_fam, train_exs, in_exs = split_once()
    out_exs = [x for x in examples if x["family"] in tsr.OUT_FAMILIES]

    # ② D14:[外] 整族不进 train / [内];out 口径不变(30 方式 × 5 行)
    assert not any(x["family"] in tsr.OUT_FAMILIES for x in train_exs + in_exs)
    assert len(out_exs) == 150
    assert set(held_by_fam) == {"cepri36", "ieee118"}, "只切训练家族"
    # 行级:[内] = 20% 家族行(每族 100 行 → 20 留出),两族合 40
    assert len(in_exs) == 40
    # ① 方式泄漏:同一 sample_id 的行同时在 train 与 [内](否则退化成方式留出)
    leaked = {x["sample_id"] for x in train_exs} & {x["sample_id"] for x in in_exs}
    assert leaked, "行级随机切分必须制造方式泄漏"
    # ③ 确定性:重算逐位一致
    held_by_fam2, _, _ = split_once()
    assert held_by_fam2 == held_by_fam, "同 seed 同族切分必须确定"


def test_load_superseded_reads_list_dict_and_absence(tmp_path):
    """W1.3:superseded_sids.json 读取。无文件=空集(向后兼容,老批逐位不变);
    列表与 {"sample_ids":[...]} 两式都解析。批内文件只读不改。"""
    tsr = _tsr_module()
    d = tmp_path / "batch"
    d.mkdir()
    assert tsr._load_superseded(d) == set(), "无该文件必须返回空集(不影响老批)"
    f = d / "superseded_sids.json"
    payload = json.dumps(["ieee118_00004", "ieee118_00007"])
    f.write_text(payload, encoding="utf-8")
    assert tsr._load_superseded(d) == {"ieee118_00004", "ieee118_00007"}
    assert f.read_text(encoding="utf-8") == payload, "读取不得改动批内文件(留档纪律)"
    f.write_text(json.dumps({"sample_ids": ["ieee118_00010"]}), encoding="utf-8")
    assert tsr._load_superseded(d) == {"ieee118_00010"}


def test_superseded_filter_excludes_and_counts_before_no_sample():
    """W1.3 同源 spec-lock:装载行循环里 superseded 剔除并计数,且排在缺样本门**之前**
    (作废方式即便不在库也如实计入 skipped_superseded);非作废行不受影响。
    复算 load_examples 行循环的 superseded 分支(与实现同源)。"""
    superseded = {"ieee118_00004", "ieee118_00007"}
    sids = {"ieee118_00004", "ieee118_00099"}          # 库里有(00007 不在库)
    rows = [{"sample_id": s} for s in
            ["ieee118_00004", "ieee118_00004", "ieee118_00007", "ieee118_00099", "ieee118_99999"]]
    cov = {"rows": 0, "skipped_superseded": 0, "skipped_no_sample": 0}
    kept = []
    for r in rows:
        cov["rows"] += 1
        sid = r["sample_id"]
        if sid in superseded:
            cov["skipped_superseded"] += 1
            continue
        if sid not in sids:
            cov["skipped_no_sample"] += 1
            continue
        kept.append(sid)
    assert cov["skipped_superseded"] == 3, "00004×2 + 00007×1 = 3 行作废(00007 不在库也计)"
    assert cov["skipped_no_sample"] == 1, "99999 缺样本(在 superseded 之外)"
    assert kept == ["ieee118_00099"], "仅非作废且在库的方式留下"
    # 无 superseded(空集)时,行为 = 补丁前:全部走原缺样本门
    cov0 = {"skipped_superseded": 0, "skipped_no_sample": 0}
    for r in rows:
        sid = r["sample_id"]
        if sid in set():
            cov0["skipped_superseded"] += 1; continue
        if sid not in sids:
            cov0["skipped_no_sample"] += 1
    assert cov0["skipped_superseded"] == 0 and cov0["skipped_no_sample"] == 2


def test_ts_data_event_ext_concat(sample_and_catalog):
    """W5 侧通道(D13):ex 带 event_ext → 事件向量拼宽且前 4 维逐位不变;
    不带 → 4 维,与接线前行为逐位一致(默认路径零变化)。"""
    from gridfm.train.pretrain import TrainConfig
    tsr = _tsr_module()
    s, cat = sample_and_catalog
    ex = build_ts_example(_label_row(), s, cat)
    cfg = TrainConfig()
    d0 = tsr._ts_data(ex, s, s, cfg)
    assert d0.event_params.shape == (1, EVENT_PARAM_DIM)
    ex["event_ext"] = np.arange(9, dtype=np.float32)
    d1 = tsr._ts_data(ex, s, s, cfg)
    assert d1.event_params.shape == (1, EVENT_PARAM_DIM + 9)
    np.testing.assert_array_equal(d1.event_params[0, :EVENT_PARAM_DIM].numpy(),
                                  d0.event_params[0].numpy())
    np.testing.assert_array_equal(d1.event_params[0, EVENT_PARAM_DIM:].numpy(),
                                  np.arange(9, dtype=np.float32))


def test_flat_held_ids_checkpoint_is_rejected():
    """held_ids schema 由扁平 list 改 {family:[ids]} → 旧检查点必须被拒(续训 = 混口径:
    粮仓与划分都变了)。校验键 = json.dumps(..., sort_keys=True),对两种 schema 天然不等。"""
    old = json.dumps(["c001", "c002"], sort_keys=True)
    new = json.dumps({"cepri36": ["c001", "c002"]}, sort_keys=True)
    assert old != new
    same = json.dumps({"ieee118": ["e1"], "cepri36": ["c1"]}, sort_keys=True)
    same2 = json.dumps({"cepri36": ["c1"], "ieee118": ["e1"]}, sort_keys=True)
    assert same == same2, "键序不同但内容相同的划分必须判等(否则同配置续训会被误拒)"


def test_resolve_out_families_guards():
    """双外族实验(2026-07-17):--out-families 校验。①默认=("ieee39",) 逐位兼容;
    ②ieee39 不许移出(D14 铁律);③未知家族拒;④不许覆盖全部家族;⑤cepri36 可加入且去重保序。"""
    import pytest as _pt
    tsr = _tsr_module()
    assert tsr.resolve_out_families(list(tsr.OUT_FAMILIES)) == ("ieee39",)
    assert tsr.resolve_out_families(["cepri36", "ieee39", "cepri36"]) == ("cepri36", "ieee39")
    with _pt.raises(SystemExit):
        tsr.resolve_out_families(["cepri36"])                  # 丢了 ieee39
    with _pt.raises(SystemExit):
        tsr.resolve_out_families(["ieee39", "nosuch"])         # 未知家族
    with _pt.raises(SystemExit):
        tsr.resolve_out_families(["ieee39", "cepri36", "ieee118"])   # 无训练家族


def test_train_mode_ablation_nested_and_holdout_untouched():
    """方式消融旗(2026-07-31 方式扩容重看)三条硬性质,任一破 = 剂量-响应设计作废:
    ①嵌套:同 mode-seed 下 frac 0.25 的保留集 ⊂ frac 0.5 ⊂ frac 0.75(各臂比的是"同一批
    方式被逐步砍",不是各臂各抽——后者把抽样方差混进剂量效应);②确定性:同参数重算逐位同;
    ③按 round 取 k 且每族 ≥1。消融逻辑与主脚本同源(default_rng([mode_seed, family_seed])
    + 排序 sid + permutation 前 k)——此处按同一算法独立重算,防主脚本静默改抽样。"""
    import numpy as _np
    tsr = _tsr_module()
    sids = {"cepri36": [f"c{i:05d}" for i in range(71)],
            "ieee118": [f"e{i:05d}" for i in range(120)]}
    def kept(frac, seed=20260731):
        out = {}
        for fam, ss in sids.items():
            ss = sorted(ss)
            perm = _np.random.default_rng([seed, tsr.family_seed(fam)]).permutation(len(ss))
            k = max(1, int(round(frac * len(ss))))
            out[fam] = sorted(ss[i] for i in perm[:k])
        return out
    k25, k50, k75 = kept(0.25), kept(0.5), kept(0.75)
    for fam in sids:
        assert set(k25[fam]) < set(k50[fam]) < set(k75[fam]), f"{fam} 嵌套性破"
        assert len(k25[fam]) == max(1, round(0.25 * len(sids[fam])))
    assert kept(0.5) == k50, "同参数重算不确定"
    assert kept(0.5, seed=1) != k50, "mode-seed 未生效"
    one = kept(0.01)
    assert all(len(v) >= 1 for v in one.values()), "极小 frac 下每族必须 ≥1 方式"


def test_train_mode_ablation_per_family_scope():
    """--train-mode-families(2026-07-31 审查改法):①只砍指定族,其余族全保留;
    ②被砍族的保留集与全族消融时逐位相同(嵌套/复算不受 scope 影响);③非训练家族名必须拒。
    与主脚本同源算法独立重算(default_rng([mode_seed, family_seed]) + 排序 + permutation)。"""
    import numpy as _np
    tsr = _tsr_module()
    sids = {"cepri36": [f"c{i:05d}" for i in range(71)],
            "ieee118": [f"e{i:05d}" for i in range(120)]}
    def kept(frac, fams):
        out = {}
        for fam, ss in sids.items():
            ss = sorted(ss)
            if fam not in fams:
                out[fam] = ss
                continue
            perm = _np.random.default_rng([20260731, tsr.family_seed(fam)]).permutation(len(ss))
            out[fam] = sorted(ss[i] for i in perm[:max(1, round(frac * len(ss)))])
        return out
    k = kept(0.5, {"ieee118"})
    assert k["cepri36"] == sorted(sids["cepri36"]), "未指定族必须全保留"
    assert len(k["ieee118"]) == 60
    assert k["ieee118"] == kept(0.5, {"ieee118", "cepri36"})["ieee118"], \
        "被砍族保留集不得因 scope 而变(嵌套/跨臂可比性)"


def test_cct_bracket_math_and_classify():
    """CCT 括号(判稳收尾,2026-08-01):①宽度 = 0.5/2^(n_sims-1),n_sims=7 → 0.0078125
    (与二分实现同源:1 次探上界 + 6 次二分);②删失行无括号;③分类三态:dur<lo 判稳、
    dur≥hi 判不稳、括号内不可判 —— 试点实测 9/9"矛盾"全落括号内,该三态是其修法。"""
    import importlib.util as _il
    from pathlib import Path as _P
    sp = _il.spec_from_file_location("cbr", str(_P(__file__).parent.parent / "scripts" / "cct_brackets.py"))
    cbr = _il.module_from_spec(sp)
    sp.loader.exec_module(cbr)
    assert cbr.bracket_width(7) == 0.0078125
    assert cbr.bracket_width(1) == 0.5
    rows = cbr.derive([
        {"cct": 0.078125, "n_sims": 7, "infeasible": False, "is_censored": False},
        {"cct": 0.5, "n_sims": 1, "infeasible": False, "is_censored": True},
        {"cct": None, "n_sims": 0, "infeasible": True, "is_censored": False},
    ])
    assert rows[0]["cct_hi"] == 0.078125 + 0.0078125
    assert rows[1]["cct_hi"] is None and rows[1]["cct_lo"] == 0.5
    assert rows[2]["cct_lo"] is None
    lo, hi = rows[0]["cct_lo"], rows[0]["cct_hi"]
    assert cbr.classify(0.05, lo, hi) == "rule_stable"
    assert cbr.classify(0.08, lo, hi) == "in_bracket"   # 试点 9 条不自洽的原型
    assert cbr.classify(0.10, lo, hi) == "rule_unstable"


def test_batch_overrides_invariants():
    """跨库批 BATCH_OVERRIDES(2026-08-01 OOD 定向修补)四不变量:
    ①family 必须在 LIBS;②批名不得与 LIBS 各族批名/EVAL_ONLY_BATCHES 撞名(撞名会双装);
    ③labels_dir 必须在本仓 runs/ 下(治理:训练批不落归档、不落 evalonly);
    ④lib 名不得等于该族默认库(等于则不需要覆盖,声明错误)。"""
    tsr = _tsr_module()
    all_libs_batches = {b for _l, bs in tsr.LIBS.values() for b in bs}
    for name, o in tsr.BATCH_OVERRIDES.items():
        assert o["family"] in tsr.LIBS, name
        assert name not in all_libs_batches, f"{name} 与 LIBS 批名撞"
        assert name not in tsr.EVAL_ONLY_BATCHES, f"{name} 与评估专用批撞"
        assert str(o["labels_dir"]).replace("\\", "/").endswith(f"runs/{name}"), name
        assert o["lib"] != tsr.LIBS[o["family"]][0], f"{name} 覆盖库与族默认库相同"


def test_holdout_exempt_keeps_held_ids_stable():
    """holdout_exempt(2026-08-01):豁免例不入留出候选池——加入豁免批前后
    held_ids 必须逐位相同,且豁免 sid 永不出现在 held 里(全进训练侧)。"""
    tsr = _tsr_module()
    base = [{"family": "ieee118", "sample_id": f"ieee118_{i:05d}"} for i in range(50)]
    h0 = tsr.mode_holdout_by_family(base, ["ieee118"], 0.2, 17260)
    extra = [{"family": "ieee118", "sample_id": f"ieee118_3{i:04d}", "holdout_exempt": True}
             for i in range(30)]
    h1 = tsr.mode_holdout_by_family(base + extra, ["ieee118"], 0.2, 17260)
    assert h0 == h1, "豁免批加入后 held_ids 漂移"
    assert not any(s.startswith("ieee118_3") for s in h1["ieee118"])
