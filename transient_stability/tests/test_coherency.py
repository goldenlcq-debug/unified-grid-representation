"""T4.3 失稳机群提取单测(方案 7.3;v1.1 末态失步口径,2026-07-13 用户裁决"按推荐优化")。

v1.1 规则:超前失步 = 末 1s 相对 COI 中位 > +500°;落后失步 = 低于非超前机中位基线
−500°(基线法消等权 COI 均摊伪影);标签 = 超前 ∪ 落后;v0 窗口先摆开提名 + 摇摆序
降级为诊断(稳控排序仍消费)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridfm.psasp_io.coherency import (coi_reference, extract_coherency,
                                       extract_from_files, first_angle_violation)

FIX39 = Path(__file__).parent / "fixtures" / "ieee39_stout"
FIX36 = Path(__file__).parent / "fixtures" / "cepri36_stout"


def _swing(n_gen=5, n=1200, dt=0.01, breakaway=(0, 1), rate=120.0):
    """合成:breakaway 机组以 rate °/s 单调拉开,其余机组在 ±5° 内摇摆。"""
    t = np.arange(n) * dt
    a = 5.0 * np.sin(2 * np.pi * 1.2 * t)[:, None] * np.ones((1, n_gen))
    for g in breakaway:
        a[:, g] += rate * t
    pairs = [(10 + i, i + 1) for i in range(n_gen)]
    return t, a, pairs


def test_first_violation_and_advance_group():
    t, a, pairs = _swing()
    t_star = first_angle_violation(t, a)
    assert t_star is not None and abs(t_star - 500.0 / 120.0) < 0.05
    res = extract_coherency(t, a, pairs)
    assert res.t_star == t_star and not res.needs_review
    # 末态:两台拉开机 rel≈+864°(1440*0.6),远超 +500 → 超前失步群
    assert res.advance_group == [(10, 1), (11, 2)]
    assert res.unstable_group == [(10, 1), (11, 2)] and not res.lag_group
    assert res.rest_group == [(12, 3), (13, 4), (14, 5)]
    assert res.direction == {"10-1": "advance", "11-2": "advance"}
    # v0 窗口提名与摇摆序作诊断保留:先摆开的两台排最前
    assert set(res.v0_window_group) == {(10, 1), (11, 2)}
    assert set(res.swing_order[:2]) == {(10, 1), (11, 2)}


def test_swingback_machine_excluded_late_joiner_included():
    """v1.1 对 v0 的两处修正(用户裁决"按推荐优化"的动因):
    ①首摆越 90° 但回摆的机不入群(v0 误纳,实测 186/420);
    ②窗口外才走失的机纳入群(v0 漏纳,实测 141/420)。"""
    n, dt = 1200, 0.01
    t = np.arange(n) * dt
    a = np.zeros((n, 4))
    a[:, 0] = 900.0 * t                                   # 真失步:单调拉爆
    # 回摆机:首摆冲到 +200°(t≈1s)后摆回 0(v0 窗口 t*≈0.56s 内峰值 >90° 会提名它)
    a[:, 1] = 200.0 * np.sin(np.pi * t / 2.0) * (t < 2.0)
    # 迟到机:前 6s 同步,6s 后才拉开,末态 +3000°
    a[:, 2] = np.where(t > 6.0, 500.0 * (t - 6.0), 0.0)
    pairs = [(1, 1), (2, 2), (3, 3), (4, 4)]
    res = extract_coherency(t, a, pairs)
    assert (1, 1) in res.advance_group, "真失步机必入"
    assert (3, 3) in res.advance_group, "迟到机末态 +3000° 应纳入(v0 漏纳)"
    assert (2, 2) not in res.unstable_group, "回摆机末态 ≈0 应剔除(v0 误纳)"
    assert (4, 4) in res.rest_group and (2, 2) in res.rest_group


def test_laggard_via_baseline_not_coi_artifact():
    """落后失步用中位基线判,不用相对 COI 负阈:单机大幅超前会把其余机整体压负
    (等权 COI 均摊伪影),中位基线对 <50% 失步鲁棒。"""
    n, dt = 1200, 0.01
    t = np.arange(n) * dt
    a = np.zeros((n, 5))
    a[:, 0] = 2000.0 * t                                  # 超前失步(末态 +24000)
    a[:, 1] = -600.0 * t                                  # 真落后失步(末态 −7200)
    pairs = [(10 + i, i + 1) for i in range(5)]
    res = extract_coherency(t, a, pairs)
    assert res.advance_group == [(10, 1)]
    assert res.lag_group == [(11, 2)], "均摊伪影下同步机不得混入落后群"
    assert set(res.unstable_group) == {(10, 1), (11, 2)}
    assert res.direction["11-2"] == "lag"
    assert res.rest_group == [(12, 3), (13, 4), (14, 5)]


def test_chain_bridge_does_not_hide_outlier():
    """审查 critical:邻差链式衔接(0/400/800)把偏离基线 >500° 的机并入最大簇——
    成员资格必须对全部机(含簇内)判,标签不得取决于无关桥接机的存在。"""
    n, dt = 1200, 0.01
    t = np.arange(n) * dt
    a = np.zeros((n, 8))
    a[:, 0] = 900.0 * t                                   # 另一台真失步(保证组非空)
    a[:, 6] = np.where(t > 1.0, 400.0 * np.minimum(t - 1.0, 1.0), 0.0)   # 桥接机 +400°
    a[:, 7] = np.where(t > 1.0, 800.0 * np.minimum(t - 1.0, 1.0), 0.0)   # 链尾 +800°
    pairs = [(10 + i, i + 1) for i in range(8)]
    res = extract_coherency(t, a, pairs)
    assert (17, 8) in res.advance_group, "链尾机偏基线 +800° 必须入群(桥接不得遮蔽)"
    assert (16, 7) not in res.unstable_group, "桥接机 +400° 在带内,不入群"
    assert (10, 1) in res.advance_group


def test_runaway_pair_cluster_triggers_majority_gate():
    """审查 major:两台连锁机末态相距 <500° 构成 size-2 最大簇(不过半)——
    不得把基线锚到失步机对上,须走解列回退(多数门)。"""
    n, dt = 1200, 0.01
    t = np.arange(n) * dt
    a = np.zeros((n, 8))
    a[:, 0] = -800.0 * t                                  # 起始失步机(负向,t*≈0.63)
    # 机 1 速率取到与机 0 末态仅差 ~100°(−9500 vs −9600):恰构成 size-2 失步对簇
    for k, rate in ((1, -950.0), (2, 300.0), (3, -300.0), (4, 700.0),
                    (5, -450.0), (6, 550.0), (7, 150.0)):
        a[:, k] = np.where(t > 2.0, rate * (t - 2.0), 0.0)   # 2s 后全散,0/1 恰成对
    pairs = [(10 + i, i + 1) for i in range(8)]
    res = extract_coherency(t, a, pairs)
    assert res.breakup, "无过半同步核 = 全网解列,不得用 size-2 失步对作基线"
    assert (10, 1) in res.unstable_group, "起始失步机必须在机群里"
    assert res.direction["10-1"] == "lag", "负向起始,方向按末态符号"


def test_saturated_angles_in_sync_core_refuse_to_label():
    """2026-07-16 critical(118 实测复现):引擎功角输出封顶 99999 后恒定 →
    多台封顶机彼此角差恒 0,伪装成"同步群";19 机中 10 台封顶恰好过半 → 基线锚到
    **最发散的那群机**上,其余 9 台正常机反被判失步(标签整体反转,且全程静默)。
    守卫 = 同步群含封顶机即拒产标签(封顶机的 δ 真值不可得)。"""
    n, dt = 1200, 0.01
    t = np.arange(n) * dt
    a = np.zeros((n, 19))
    for k in range(10):                                   # 10 台快速滑极 → 打到 99999 封顶
        a[:, k] = np.minimum(9000.0 * t * (1 + 0.01 * k), 99999.0)
    for k in range(10, 19):                               # 9 台慢速滑极(45k 量级,未封顶)
        a[:, k] = 3500.0 * t + 20.0 * k                   # ≈ 118 实测的 45k 那群
    pairs = [(10 + i, i + 1) for i in range(19)]
    # 危险确已构成:10 台封顶机末态逐位相同(邻差 0)→ 最大簇 10/19 过半 → 若无守卫,
    # 基线 = 99999,9 台慢机偏离 −57000° 被判失步(而真相恰恰相反)
    tail = a[t >= t[-1] - 1.0]
    assert np.allclose(np.median(tail[:, :10], axis=0), 99999.0), \
        "10 台封顶机的末 1s 中位逐位相同(邻差 0)= 伪同步群"
    res = extract_coherency(t, a, pairs)
    assert res.n_saturated == 10, "封顶机计数如实携带"
    assert res.needs_review and not res.unstable_group, "拒产标签(基线不可信)"
    assert "saturated" in res.note
    assert not res.advance_group and not res.lag_group


def test_saturation_outside_sync_core_still_labels():
    """封顶机不在同步群时不拦:它们必然 |偏离基线| ≫500° → 正确判为失步机,
    真值不可得不影响成员资格(只拒"基准本身不可信"的行)。"""
    n, dt = 1200, 0.01
    t = np.arange(n) * dt
    a = np.zeros((n, 9))
    a[:, 0] = np.minimum(20000.0 * t, 99999.0)            # 1 台封顶(少数)
    for k in range(1, 9):                                 # 8 台同步核(角度平稳)
        a[:, k] = 10.0 * k + 2.0 * np.sin(t)
    pairs = [(10 + i, i + 1) for i in range(9)]
    res = extract_coherency(t, a, pairs)
    assert res.n_saturated == 1
    assert not res.needs_review
    assert res.unstable_group == [(10, 1)], "封顶机 = 失步机;8 台同步核作基线"


def test_negative_onset_fallback_not_inverted():
    """审查 major:负向先摆开(落后型起始)在回退分支不得反转——正向单侧 COI 提名会
    把起始机漏掉、同步机集体入选;双侧中位参考提名修复。"""
    n, dt = 1200, 0.01
    t = np.arange(n) * dt
    a = np.zeros((n, 8))
    a[:, 0] = -900.0 * t                                  # 负向俯冲起始机
    for k, rate in ((1, 400.0), (2, -300.0), (3, 700.0), (4, -450.0),
                    (5, 550.0), (6, 150.0), (7, -650.0)):
        a[:, k] = np.where(t > 4.0, rate * (t - 4.0), 0.0)   # 4s 后连锁全散
    pairs = [(10 + i, i + 1) for i in range(8)]
    res = extract_coherency(t, a, pairs)
    assert res.breakup
    assert res.unstable_group == [(10, 1)], "机群 = 负向起始机,不是 7 台同步机"
    assert res.direction["10-1"] == "lag"


def test_late_accelerating_machine_caught_by_end_check():
    """审查 major:末段(末 ~1s)才起滑的机,末 1s 中位 ≈ 末刻 − r/2,单用中位会漏;
    末刻副判必须抓住(且不依赖'整组为空'的回退门)。"""
    n, dt = 1200, 0.01
    t = np.arange(n) * dt
    a = np.zeros((n, 6))
    a[:, 0] = 900.0 * t                                   # 早失步机(t*≈0.56,组非空)
    a[:, 5] = np.where(t > 11.3, 2000.0 * (t - 11.3), 0.0)  # 末段起滑:末刻 +1400,中位 ~+400
    pairs = [(10 + i, i + 1) for i in range(6)]
    res = extract_coherency(t, a, pairs)
    assert (10, 1) in res.advance_group
    assert (15, 6) in res.advance_group, "末段加速失步机必须被末刻副判抓住"
    assert not res.needs_review


def test_breakup_falls_back_to_onset_nomination():
    """全网连锁解列(cepri36 实测 80/420):末态所有机彼此散开(最大簇并列),
    "谁最终失步"不适定 → 回退 v0 窗口先摆开提名(起始失步机)并打 breakup。"""
    n, dt = 1200, 0.01
    t = np.arange(n) * dt
    a = np.zeros((n, 5))
    a[:, 0] = 1000.0 * t                                  # 起始失步机(先摆开)
    for k, rate in ((1, 400.0), (2, -300.0), (3, -700.0), (4, 150.0)):
        a[:, k] = np.where(t > 3.0, rate * (t - 3.0), 0.0)   # 3s 后连锁,末态全散
    pairs = [(10 + i, i + 1) for i in range(5)]
    res = extract_coherency(t, a, pairs)
    assert res.breakup and not res.needs_review
    assert res.unstable_group == [(10, 1)], "机群 = 起始失步机(窗口提名)"
    assert res.direction == {"10-1": "advance"}
    assert "连锁解列" in res.note


def test_stable_case_returns_none():
    t = np.arange(600) * 0.01
    a = 30.0 * np.sin(2 * np.pi * 1.0 * t)[:, None] * np.ones((1, 4))
    res = extract_coherency(t, a, [(1, 1), (2, 2), (3, 3), (4, 4)])
    assert res.t_star is None and res.note == "not_angle_unstable"
    assert not res.unstable_group


def test_needs_review_when_all_resync():
    """功角差曾越 500° 但末态全部追回(滑极后再同步/临界多摆)→ needs_review,
    不静默产空标签。"""
    n, dt = 1200, 0.01
    t = np.arange(n) * dt
    a = np.zeros((n, 4))
    # 首摆冲到 +520° 后回落到 60° 并保持(末态相对角 <500)
    a[:, 0] = np.where(t < 2.0, 260.0 * t * (t < 2.0) * 2.0 * (1 - t / 4.0), 60.0)
    a[:, 0] = np.where(t < 2.0, 520.0 * np.sin(np.pi * t / 4.0), 60.0)
    pairs = [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert first_angle_violation(t, a) is not None, "确曾越限(前提)"
    res = extract_coherency(t, a, pairs)
    assert res.needs_review and not res.unstable_group
    assert "末态无失步机" in res.note


def test_partial_angle_coverage_refuses_to_label():
    """**覆盖门**(2026-07-13 审查):SME 功角记录来自算例 GUI 模板,可能缺机且无人校验。
    基准 = 算例真实机数(LF.LP5 行数),不足即拒产标签(needs_review)。"""
    t, a, pairs = _swing(n_gen=5)
    res = extract_coherency(t, a, pairs, n_gen_case=6)      # 算例 6 机,SME 只配 5 机
    assert res.needs_review and res.t_star is None
    assert not res.unstable_group
    assert "5/6" in res.note
    ok = extract_coherency(t, a, pairs, n_gen_case=5)
    assert not ok.needs_review and ok.advance_group == [(10, 1), (11, 2)]


def test_coi_weights_equal_default():
    a = np.array([[0.0, 100.0], [10.0, 110.0]])
    np.testing.assert_allclose(coi_reference(a), [50.0, 60.0])
    np.testing.assert_allclose(coi_reference(a, weights=np.array([3.0, 1.0])),
                               [25.0, 35.0])
    # 非法权(0/NaN)回落等权(与 stout.coi_frequency 同一退化口径)
    np.testing.assert_allclose(coi_reference(a, weights=np.array([0.0, np.nan])),
                               [50.0, 60.0])


def test_extract_from_files_ieee39_unstable_fixture():
    """真轨迹:ieee39 L34-28-23 d200(失稳例)。人工读图:(17,2)/(16,3) 甩出、
    (9,10) 掉队——v1.1 标签应与人工判读一致(v0 只标先摆开的 (17,2),(16,3))。"""
    res = extract_from_files(FIX39 / "STOUT.DDD", FIX39 / "ST.SME")
    assert res.t_star is not None and res.n_gen == 10
    assert (17, 2) in res.advance_group
    assert (9, 10) in res.lag_group, "真实落后失步机应入 v1.1 标签(v0 不含)"
    assert set(res.unstable_group) >= {(17, 2), (9, 10)}
    assert len(res.rest_group) >= 6
    assert set(res.unstable_group) & set(res.rest_group) == set()


def test_extract_from_files_stable_fixture_no_group():
    """cepri36 夹具(稳例,角峰 ~100°)→ 非功角失稳,不产机群标签。"""
    res = extract_from_files(FIX36 / "STOUT.DDD", FIX36 / "ST.SME")
    assert res.t_star is None and res.note == "not_angle_unstable"
