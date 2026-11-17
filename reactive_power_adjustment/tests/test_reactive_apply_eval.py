"""无功验收/规则修法 · 纯函数单测(方向二选项 B;无 PSASP)。

锁拓扑辅助:_bus_adjacency(读 LF.L2 邻接)、_nearest_gens(图跳数最近机组)、
_extreme_bus(越限最深母线,低/高压两向)。reactive_rule_arm/trivial_arm/main 含 PSASP,
S2 点火验证不进单测。
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from gridfm.psasp_io.cards import AclineView
from gridfm.psasp_io.lowlevel import Card
from scripts.reactive_apply_eval import (
    _bus_adjacency, _extreme_bus, _nearest_gens, apply_mixed, as_mixed, fix_is_empty,
    hops_from, mixed_fix, read_shunt_marks, reactive_rule_arm, restore_vset,
    set_shunt_marks, shunt_records,
)
from scripts.reactive_labels import read_ctrl

FIX = Path(__file__).parent / "fixtures" / "cepri36_golden"


def _case(tmp_path: Path) -> Path:
    """可写算例副本(单测只改卡片、不调 PSASP)。"""
    work = tmp_path / "case"
    shutil.copytree(FIX, work)
    return work


def _shunt_l2_rows(work: Path):
    """LF.L2 里的并联补偿记录 [(记录序, 母线号, X)]。"""
    l2 = Card.load(work / "LF.L2")
    return [(i, AclineView(r, i).i_no, r.get_float(AclineView.X))
            for i, r in enumerate(l2.records)
            if AclineView(r, i).branch_type == "shunt_comp"]


def test_bus_adjacency_symmetric_nonempty():
    adj = _bus_adjacency(FIX)
    assert len(adj) > 0
    for i, nbrs in adj.items():
        for j in nbrs:
            assert i in adj[j], "邻接对称"


def test_nearest_gens_returns_valid_ordered_gens():
    adj = _bus_adjacency(FIX)
    _s, gens = read_ctrl(Card.load(FIX / "LF.L5"))
    seed = list(gens.values())[0]["bus"]
    ng = _nearest_gens(adj, seed, gens, 3)
    assert 1 <= len(ng) <= 3
    assert all(gi in gens for gi in ng), "返回的都是有效机组记录序"
    assert len(set(ng)) == len(ng), "不重复"
    # seed 本身是机组则排第一
    assert gens[ng[0]]["bus"] == seed


def test_nearest_gens_k_capped_and_empty_safe():
    adj = _bus_adjacency(FIX)
    _s, gens = read_ctrl(Card.load(FIX / "LF.L5"))
    seed = list(gens.values())[0]["bus"]
    all_g = _nearest_gens(adj, seed, gens, 999)
    assert len(all_g) <= len(gens)
    assert _nearest_gens({}, 99999, gens, 3) == []       # 孤立种子不炸


# ---------------- 混合动作契约(v_set 连续 + shunt 离散) ----------------

def test_mixed_fix_contract_and_backward_compat():
    f = mixed_fix({7: 0.03}, [20, 21])
    assert f == {"dvset": {7: 0.03}, "shunt": [20, 21]}
    assert not fix_is_empty(f)
    assert fix_is_empty(mixed_fix()), "空修法 = 方式本就健康"
    assert not fix_is_empty(None), "None = 修不动,不是空修法"
    assert not fix_is_empty(mixed_fix({}, [20])), "只投并补也是实修法"
    # 老契约 {母线: Δ} 自动升级(v0/v2 标签与预测仍可读)
    assert as_mixed({7: 0.03}) == {"dvset": {7: 0.03}, "shunt": []}
    assert as_mixed(None) is None
    assert as_mixed(f) == f


def test_hops_from_bfs_distances():
    adj = {1: {2}, 2: {1, 3}, 3: {2}}
    assert hops_from(adj, 1) == {1: 0, 2: 1, 3: 2}
    assert hops_from(adj, 99) == {99: 0}, "孤立种子不炸"


# ---------------- 并联补偿(第二杠杆)读写 ----------------

def test_shunt_records_selects_out_of_service_by_sign(tmp_path):
    work = _case(tmp_path)
    rows = _shunt_l2_rows(work)
    assert rows, "夹具应含并联补偿记录"
    assert shunt_records(work, capacitive=True) == [], "全在运 → 无可投候选"
    cap = next(i for i, _b, x in rows if x < 0)
    ind = next(i for i, _b, x in rows if x > 0)
    set_shunt_marks(work, {cap: 0, ind: 0})              # 置停运 → 成为候选
    got_cap = shunt_records(work, capacitive=True)
    got_ind = shunt_records(work, capacitive=False)
    assert [i for i, _b in got_cap] == [cap], "容性(X<0,抬压)只进容性候选"
    assert ind in [i for i, _b in got_ind] and cap not in [i for i, _b in got_ind]


def test_shunt_marks_read_write_roundtrip(tmp_path):
    work = _case(tmp_path)
    idxs = [i for i, _b, _x in _shunt_l2_rows(work)][:2]
    orig = read_shunt_marks(work, idxs)
    assert set(orig.values()) == {1}
    set_shunt_marks(work, {i: 0 for i in idxs})
    assert set(read_shunt_marks(work, idxs).values()) == {0}
    set_shunt_marks(work, orig)                          # 复位
    assert read_shunt_marks(work, idxs) == orig
    set_shunt_marks(work, {})                            # 空写不炸、不改盘


def test_apply_mixed_invests_shunt_then_writes_dvset(tmp_path):
    work = _case(tmp_path)
    cap_idx, cap_bus, _x = next(t for t in _shunt_l2_rows(work) if t[2] < 0)
    set_shunt_marks(work, {cap_idx: 0})                  # 造一个可投候选
    _s, gens = read_ctrl(Card.load(work / "LF.L5"))
    gi = list(gens)[0]
    gbus, v_before = gens[gi]["bus"], gens[gi]["view"].v0
    applied = apply_mixed(work, mixed_fix({gbus: 0.02}, [cap_bus]))
    assert applied["shunt"] == [cap_bus]
    assert read_shunt_marks(work, [cap_idx])[cap_idx] == 1, "并补已投运(mark=1)"
    _s2, gens2 = read_ctrl(Card.load(work / "LF.L5"))
    assert gens2[gi]["view"].v0 == pytest.approx(v_before + 0.02, abs=1e-6)
    assert applied["dvset"][gbus] == pytest.approx(0.02, abs=1e-6), "返回实际生效量"


def test_apply_mixed_shunt_only_leaves_vset_untouched(tmp_path):
    work = _case(tmp_path)
    cap_idx, cap_bus, _x = next(t for t in _shunt_l2_rows(work) if t[2] < 0)
    set_shunt_marks(work, {cap_idx: 0})
    _s, gens = read_ctrl(Card.load(work / "LF.L5"))
    before = {gi: g["view"].v0 for gi, g in gens.items()}
    applied = apply_mixed(work, mixed_fix({}, [cap_bus]))
    _s2, gens2 = read_ctrl(Card.load(work / "LF.L5"))
    assert {gi: g["view"].v0 for gi, g in gens2.items()} == before
    assert applied["dvset"] == {} and applied["shunt"] == [cap_bus]


def test_apply_mixed_ignores_bus_without_available_shunt(tmp_path):
    """预测挑了个没有停运并补的母线:如实不投(不伪造动作),不抛异常。"""
    work = _case(tmp_path)
    applied = apply_mixed(work, mixed_fix({}, [99999]))
    assert applied["shunt"] == []


def test_restore_vset_puts_back_original_settings(tmp_path):
    work = _case(tmp_path)
    _s, gens = read_ctrl(Card.load(work / "LF.L5"))
    orig = {gi: g["view"].v0 for gi, g in gens.items()}
    gbus = gens[list(gens)[0]]["bus"]
    apply_mixed(work, mixed_fix({gbus: 0.05}, []))
    restore_vset(work, orig)
    _s2, gens2 = read_ctrl(Card.load(work / "LF.L5"))
    assert {gi: g["view"].v0 for gi, g in gens2.items()} == orig


# ---------------- 规则两级杠杆:用假裁判测状态管理(不调 PSASP) ----------------

def _prep_caps(work: Path, n: int):
    """造 n 个"停运的容性并补"候选(夹具本身并补全在运、且只有 1 个容性)。
    返回 [(记录序, 母线号)]。"""
    l2 = Card.load(work / "LF.L2")
    rows = [i for i, r in enumerate(l2.records) if AclineView(r, i).branch_type == "shunt_comp"]
    picked = []
    for i in rows[:n]:
        av = AclineView(l2.records[i], i)
        av.mark = 0                                   # 停运 → 成为候选
        l2.records[i].set_float(AclineView.X, -1.0)   # 容性(X<0,抬压)
        picked.append((i, av.i_no))
    l2.save(work / "LF.L2")
    return picked


class _FakeGrid:
    """不调 PSASP 的物理代理裁判:vmin = 基准 + 已投并补台数×增益 + Δv_set 总量×增益。

    真实 PSASP 的电压响应当然不是线性的,但这里要测的不是物理,而是 reactive_rule_arm
    两级杠杆的**状态管理**:每一档 k 是否都从同一病态起点重来、失败后是否把并补 mark
    与 v_set 完整复位。裁判只要**如实反映当前盘面**,这些性质就能被测出来。
    """

    def __init__(self, work: Path, cap_idxs, seed_bus: int, n_buses: int,
                 base_vmin: float, gain_shunt: float, gain_vset: float,
                 vset_needs_shunt: bool = False) -> None:
        self.work, self.cap_idxs = work, list(cap_idxs)
        self.seed_bus, self.n_buses = seed_bus, n_buses
        self.base_vmin = base_vmin
        self.gain_shunt, self.gain_vset = gain_shunt, gain_vset
        # vset_needs_shunt = 复刻已查明的"规则敌对"机理:近端机无功已饱和,不先投并补
        # 就抬不动电压 —— 这正是 68 个方式 v_set 单杠杆一个都修不动的物理成因
        self.vset_needs_shunt = vset_needs_shunt
        _s, gens = read_ctrl(Card.load(work / "LF.L5"))
        self.base_v0 = {gi: g["view"].v0 for gi, g in gens.items()}
        self.calls = 0

    def n_invested(self) -> int:
        marks = read_shunt_marks(self.work, self.cap_idxs)
        return sum(1 for m in marks.values() if m == 1)

    def total_dvset(self) -> float:
        _s, gens = read_ctrl(Card.load(self.work / "LF.L5"))
        return sum(gens[gi]["view"].v0 - v for gi, v in self.base_v0.items() if gi in gens)

    def __call__(self, work, solver):
        self.calls += 1
        n_inv = self.n_invested()
        eff_vset = 0.0 if (self.vset_needs_shunt and n_inv == 0) else self.gain_vset
        vmin = (self.base_vmin + self.gain_shunt * n_inv
                + eff_vset * self.total_dvset())
        v = np.full(self.n_buses, 1.0)
        v[self.seed_bus - 1] = vmin
        return {"converged": True, "vmin": float(vmin), "vmax": 1.0, "v": v}


def _scenario(tmp_path, monkeypatch, n_caps, base_vmin, gain_shunt, gain_vset,
              vset_needs_shunt=False):
    work = _case(tmp_path)
    caps = _prep_caps(work, n_caps)
    _s, gens = read_ctrl(Card.load(work / "LF.L5"))
    seed_bus = gens[list(gens)[0]]["bus"]
    n_buses = max(int(b) for b in _bus_adjacency(work)) + 5
    fake = _FakeGrid(work, [i for i, _b in caps], seed_bus, n_buses,
                     base_vmin, gain_shunt, gain_vset, vset_needs_shunt)
    monkeypatch.setattr("scripts.reactive_apply_eval._referee", fake)
    return work, caps, fake


def test_rule_arm_first_lever_alone_leaves_shunts_untouched(tmp_path, monkeypatch):
    """v_set 一级就修好时,不得动并补(最小干预)。"""
    work, caps, _f = _scenario(tmp_path, monkeypatch, 3, 0.90, 0.0, 3.0)
    fix = reactive_rule_arm(work, None, 0.93, 1.08, use_shunt=True)
    assert fix is not None and fix["shunt"] == [] and fix["dvset"]
    assert set(read_shunt_marks(work, [i for i, _b in caps]).values()) == {0}


def test_rule_arm_shunt_only_rescue(tmp_path, monkeypatch):
    """v_set 修不动、投 1 台并补即合格(去风险实测里 13 例中的 9 例是这种)。"""
    work, caps, _f = _scenario(tmp_path, monkeypatch, 3, 0.90, 0.05, 0.0)
    fix = reactive_rule_arm(work, None, 0.93, 1.08, use_shunt=True)
    assert fix is not None and fix["dvset"] == {}
    assert len(fix["shunt"]) == 1 and fix["shunt"][0] in [b for _i, b in caps]
    assert sum(read_shunt_marks(work, [i for i, _b in caps]).values()) == 1


def test_rule_arm_escalates_shunt_count_until_healthy(tmp_path, monkeypatch):
    """1 台不够就投 2 台:档位递增,且每档从同一起点重来(不是累加到 1+2=3 台)。"""
    work, caps, _f = _scenario(tmp_path, monkeypatch, 3, 0.90, 0.02, 0.0)
    fix = reactive_rule_arm(work, None, 0.93, 1.08, use_shunt=True)
    assert fix is not None and len(fix["shunt"]) == 2, "0.90+2*0.02=0.94 才够"
    assert sum(read_shunt_marks(work, [i for i, _b in caps]).values()) == 2


def test_rule_arm_combines_shunt_then_vset(tmp_path, monkeypatch):
    """"规则敌对"方式的解锁路径:近端机 Q 饱和 → v_set 单杠杆一个都修不动;
    先投 1 台并补解饱和,再跑 v_set 贪心才修得好 → 返回混合动作。
    这正是第二杠杆要解锁那 68 个方式打标的机理。"""
    work, caps, _f = _scenario(tmp_path, monkeypatch, 3, 0.90, 0.02, 1.0,
                               vset_needs_shunt=True)
    fix = reactive_rule_arm(work, None, 0.93, 1.08, use_shunt=True)
    assert fix is not None, "双杠杆应当修得动"
    assert len(fix["shunt"]) == 1 and fix["dvset"], \
        "0.90+0.02(并补解饱和)+0.02(抬 1 机一步)=0.94"
    assert sum(read_shunt_marks(work, [i for i, _b in caps]).values()) == 1


def test_rule_arm_restores_all_state_when_unfixable(tmp_path, monkeypatch):
    """**最关键的一条**:双杠杆都修不动时返回 None,并把并补 mark 与 v_set 完整复位
    —— 否则打标脚本会把"投了并补的盘面"当作下一次的病态起点,污染后续全部标签。"""
    work, caps, fake = _scenario(tmp_path, monkeypatch, 3, 0.90, 0.0, 0.0)
    before_v = dict(fake.base_v0)
    fix = reactive_rule_arm(work, None, 0.93, 1.08, use_shunt=True)
    assert fix is None
    assert set(read_shunt_marks(work, [i for i, _b in caps]).values()) == {0}, "并补全复位"
    _s, gens = read_ctrl(Card.load(work / "LF.L5"))
    assert {gi: g["view"].v0 for gi, g in gens.items()} == before_v, "v_set 全复位"


def test_rule_arm_default_stays_single_lever(tmp_path, monkeypatch):
    """缺省 use_shunt=False 必须与已定论的 0.61 老师逐位同构:不碰并补。"""
    work, caps, fake = _scenario(tmp_path, monkeypatch, 3, 0.90, 0.05, 0.0)
    fix = reactive_rule_arm(work, None, 0.93, 1.08)
    assert fix is None, "单杠杆修不动(gain_vset=0)"
    assert set(read_shunt_marks(work, [i for i, _b in caps]).values()) == {0}


def test_rule_arm_empty_fix_when_already_healthy(tmp_path, monkeypatch):
    work, _caps, _f = _scenario(tmp_path, monkeypatch, 3, 0.99, 0.0, 0.0)
    fix = reactive_rule_arm(work, None, 0.93, 1.08, use_shunt=True)
    assert fix_is_empty(fix), "本就健康 = 空修法,不是修不动"


def test_extreme_bus_low_and_high():
    rf = {"converged": True, "v": np.array([1.00, 0.90, 1.12, 0.005, 0.95])}
    assert _extreme_bus(rf, low=True) == 2, "最低压母线(1 基;跳过失电 0.005)"
    assert _extreme_bus(rf, low=False) == 3, "最高压母线"
    assert _extreme_bus({"converged": True, "v": np.array([0.005, 0.004])}, low=True) is None
    assert _extreme_bus({"converged": False}, low=True) is None
