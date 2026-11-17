"""统计-几何合成器离线结构测试(不依赖 PSASP/pandapower;收敛性由修复流水线+终审覆盖)。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

import gridfm.data.synth as synth_mod
from gridfm.data.repair import DEFAULT_REPAIR_CONFIG, RepairTrace
from gridfm.solver.base import PFResult
from gridfm.psasp_io.case import PsaspCase
from gridfm.psasp_io.catalog import Catalog
from gridfm.psasp_io.lowlevel import Card
from gridfm.schema.assemble import build_sample
from gridfm.data.posenc import compute_lape
from gridfm.data.synth import synth_case, synth_topology


def test_topology_stats():
    rng = np.random.default_rng(0)
    pts, edges, elens, is_backbone = synth_topology(200, 3.0, rng)
    deg = np.zeros(200)
    for i, j in edges:
        deg[i] += 1; deg[j] += 1
    assert abs(deg.mean() - 3.0) < 0.15          # 目标平均度
    assert deg.min() >= 1                         # MST 保证连通(无孤立点)
    assert is_backbone.sum() == 199               # MST 边数 = n-1


def test_sparse_and_legacy_dense_dc_solvers_match():
    rng = np.random.default_rng(7)
    _pts, edges, _lengths, _backbone = synth_topology(40, 3.0, rng)
    xs = 0.02 + rng.random(len(edges)) * 0.2
    p = rng.normal(size=40)
    p -= p.mean()
    keep = np.arange(1, 40)
    ei = np.array([edge[0] for edge in edges])
    ej = np.array([edge[1] for edge in edges])
    sparse = synth_mod._dc_solve(ei, ej, xs, 40, keep, p)
    dense = synth_mod._dc_solve_dense_legacy(edges, xs, 40, keep, p)
    assert np.allclose(sparse, dense, rtol=1e-10, atol=1e-10)


def test_synth_case_structure(tmp_path, case_dir):
    info = synth_case(160, case_dir, tmp_path / "s", seed=1)
    c = PsaspCase(tmp_path / "s")
    assert c.n_buses == 160
    assert len(c.aclines()) == info["n_lines"]
    assert len(c.xfmrs()) == 0                    # v0 无变压器
    # 平衡机唯一
    slacks = [g for g in c.gens() if g.type_code == 0]
    assert len(slacks) == 1
    # N-1 布机规则:每台机组母线度 >= 2
    deg = {}
    for l in c.aclines():
        deg[l.i_no] = deg.get(l.i_no, 0) + 1
        deg[l.j_no] = deg.get(l.j_no, 0) + 1
    for g in c.gens():
        assert deg.get(g.bs_no, 0) >= 2, f"机组母线 {g.bs_no} 度<2 违反 N-1 布机"
    # 负荷全为 PQ 型、初值归平
    for ld in c.loads():
        assert ld.rec.get_int(3) == 1
    # L0 计数一致
    h = Card.load(tmp_path / "s" / "LF.L0").records[0]
    assert h.get_int(0) == 160
    assert h.get_int(1) == len(c.aclines())
    assert h.get_int(4) == len(c.gens())
    assert h.get_int(5) == len(c.loads())
    # 下游可消化:目录页/档案/单一主分量
    cat = Catalog(c)
    s = build_sample(c, cat, family_id="synth160")
    assert not np.isnan(s.node_feat).any()
    _l, _v, comp = compute_lape(s.edge_index, s.n_nodes, k=8)
    assert np.bincount(comp).max() == 160         # 全连通(合成网无 NULL 占位)


def test_line_constants_physical(tmp_path, case_dir):
    """线路常数物理性:X∝长度(同一网内相关),X/R 在施主范围,B∝X。"""
    synth_case(160, case_dir, tmp_path / "s2", seed=2)
    c = PsaspCase(tmp_path / "s2")
    lines = [l for l in c.aclines() if l.mark == 1]
    xs = np.array([l.x for l in lines])
    rs = np.array([l.r for l in lines])
    bs = np.array([l.b_half for l in lines])
    assert (xs > 0).all() and xs.max() < 0.5
    xr = xs / np.maximum(rs, 1e-9)
    assert np.median(xr) > 1.0                    # X/R>1(输电网特征)
    # B 与 X 正相关(同为长度的函数)
    corr = np.corrcoef(xs, bs)[0, 1]
    assert corr > 0.5


class _AlwaysDivergedSolver:
    def __init__(self):
        self.calls = 0

    def run_powerflow(self, _case_dir):
        self.calls += 1
        return PFResult(converged=False)


def test_repair_pipeline_uses_final_psasp_verdict(monkeypatch, tmp_path):
    """中间 P 环曾成功也不能代替最终卡片的 PSASP 终审。"""
    q_results = iter([
        {"nr_ok": False, "q_rounds": 8},
        {"nr_ok": False, "q_rounds": 4},
    ])
    monkeypatch.setattr(synth_mod, "_ac_reactive_planning",
                        lambda *_a, **_k: next(q_results))
    monkeypatch.setattr(synth_mod, "_p_redispatch",
                        lambda *_a, **_k: {"p_ok": True, "p_rounds": 1})
    solver = _AlwaysDivergedSolver()

    result = synth_mod.repair_pipeline(tmp_path, solver)

    assert result["nr_ok"] is False
    assert result["final_converged"] is False
    assert result["trace"]["termination_reason"] == "final_psasp_diverged"
    assert result["trace"]["observations"][-1]["stage"] == "final_psasp"
    assert solver.calls == 1


def test_synth_convergent_does_not_modify_after_last_verdict(monkeypatch, tmp_path):
    """max_repair=N 只能产生 N 次降载，最后一次修改必须有对应终审。"""
    class _Record:
        def __init__(self):
            self.values = {4: 1.0, 5: 1.0}

        def get_int(self, idx):
            assert idx == 1
            return 1

        def get_float(self, idx):
            return self.values[idx]

        def set_float(self, idx, value):
            self.values[idx] = value

    class _Card:
        loads = 0
        saves = 0
        record = _Record()

        @classmethod
        def load(cls, _path):
            cls.loads += 1
            return cls()

        def __init__(self):
            self.records = [self.record]

        def save(self, _path):
            type(self).saves += 1

    repair_calls = []
    monkeypatch.setattr(synth_mod, "Card", _Card)
    monkeypatch.setattr(synth_mod, "synth_case", lambda *_a, **_k: {})
    monkeypatch.setattr(
        synth_mod, "repair_pipeline",
        lambda *_a, **_k: repair_calls.append(1) or
        {"nr_ok": False, "q_rounds": 0, "p_rounds": 0},
    )
    solver = _AlwaysDivergedSolver()
    trace = RepairTrace.for_config(DEFAULT_REPAIR_CONFIG)

    result = synth_mod.synth_convergent(
        10, tmp_path / "donor", tmp_path / "out", solver, max_repair=2,
        trace=trace)

    assert result["converged"] is False
    assert result["repairs"] == 2
    assert solver.calls == 3             # 初始终审 + 两次降载后的终审
    assert len(repair_calls) == 3        # 初始修复 + 两次降载后的修复
    assert _Card.loads == 2
    assert _Card.saves == 2
    assert _Card.record.values[4] == 0.85 ** 2
    assert _Card.record.values[5] == 0.85 ** 2
    assert result["trace"]["termination_reason"] == "repair_budget_exhausted"
    assert len(result["trace"]["observations"]) == 3
    assert [a["field"] for a in result["trace"]["actions"]] == [
        "LF.L6.PL", "LF.L6.QL", "LF.L6.PL", "LF.L6.QL"]
    assert trace.replay_touched_values() == {
        "bus:1:LF.L6.PL": 0.85 ** 2,
        "bus:1:LF.L6.QL": 0.85 ** 2,
    }


def test_ac_repair_existing_rows_only_and_protected(monkeypatch, tmp_path, case_dir):
    """无 L6 行的病灶只记账不新建；保护母线即使低压也不修改。"""
    work = tmp_path / "case"
    shutil.copytree(case_dir, work)
    before = Card.load(work / "LF.L6")
    count_before = len(before.records)
    q2_before = next(r.get_float(5) for r in before.records if r.get_int(1) == 2)

    v_bad = np.ones(119)
    v_bad[0] = 0.90                 # 母线 1 无 L6 行，应记 missing
    v_bad[1] = 0.90                 # 母线 2 有 L6 行，但被保护
    verdicts = iter([
        {"converged": True, "vmin": 0.90, "vmax": 1.0, "v": v_bad},
        {"converged": True, "vmin": 0.95, "vmax": 1.0, "v": np.ones(119)},
    ])
    monkeypatch.setattr(synth_mod, "psasp_referee",
                        lambda *_a, **_k: next(verdicts))

    trace = RepairTrace.for_config(DEFAULT_REPAIR_CONFIG.with_protected({2}))
    result = synth_mod._ac_reactive_planning(
        work, object(), max_rounds=2, protected_buses={2}, trace=trace)

    after = Card.load(work / "LF.L6")
    q2_after = next(r.get_float(5) for r in after.records if r.get_int(1) == 2)
    assert result["nr_ok"] is True
    assert result["missing_comp_buses"] == [1]
    assert result["protected_skipped"] == [2]
    assert len(after.records) == count_before
    assert q2_after == q2_before
    assert len(trace.observations) == 2
    assert any(
        a["object_id"] == 1 and a["status"] == "skipped_missing_record"
        for a in trace.actions)
    assert any(
        a["object_id"] == 2 and a["status"] == "skipped_protected"
        for a in trace.actions)


def test_ac_repair_reads_unified_rule_config(monkeypatch, tmp_path, case_dir):
    """自定义目标经统一配置生效，不依赖 synth.py 内的重复常量。"""
    work = tmp_path / "case"
    shutil.copytree(case_dir, work)
    l6_before = (work / "LF.L6").read_bytes()
    v = np.ones(119)
    v[0] = 0.90
    monkeypatch.setattr(
        synth_mod, "psasp_referee",
        lambda *_a, **_k: {
            "converged": True, "vmin": 0.90, "vmax": 1.0, "v": v})
    cfg = replace(DEFAULT_REPAIR_CONFIG, voltage_target=0.90)

    result = synth_mod._ac_reactive_planning(work, object(), config=cfg)

    assert result["nr_ok"] is True
    assert result["q_rounds"] == 0
    assert (work / "LF.L6").read_bytes() == l6_before


def test_ac_repair_experimental_new_row_is_explicit_and_updates_l0(
        monkeypatch, tmp_path, case_dir):
    """历史复现开关可新建纯补偿行；默认关闭，且 L0 计数必须同步。"""
    work = tmp_path / "case"
    shutil.copytree(case_dir, work)
    l6_before = Card.load(work / "LF.L6")
    count_before = len(l6_before.records)
    v_bad = np.ones(119)
    v_bad[0] = 0.90  # 母线 1 在 golden LF.L6 中无行
    verdicts = iter([
        {"converged": True, "vmin": 0.90, "vmax": 1.0, "v": v_bad},
        {"converged": True, "vmin": 0.95, "vmax": 1.0, "v": np.ones(119)},
    ])
    monkeypatch.setattr(
        synth_mod, "psasp_referee", lambda *_a, **_k: next(verdicts))
    cfg = replace(DEFAULT_REPAIR_CONFIG, allow_new_comp_rows=True)
    trace = RepairTrace.for_config(cfg)

    result = synth_mod._ac_reactive_planning(
        work, object(), max_rounds=2, config=cfg, trace=trace)

    l6_after = Card.load(work / "LF.L6")
    l0_after = Card.load(work / "LF.L0")
    assert result["nr_ok"] is True
    assert result["missing_comp_buses"] == []
    assert len(l6_after.records) == count_before + 1
    assert l0_after.records[0].get_int(5) == len(l6_after.records)
    new_action = next(
        action for action in trace.actions
        if action["metadata"].get("experimental_new_record"))
    assert new_action["object_id"] == 1
    assert new_action["status"] == "applied"


def test_p_redispatch_does_not_move_protected_generator(
        monkeypatch, tmp_path, case_dir):
    """保护名单同时约束有功再调度，不只约束无功环。"""
    import gridfm.eval.nrpf as nrpf_mod

    work = tmp_path / "case"
    shutil.copytree(case_dir, work)
    before = Card.load(work / "LF.L5")
    protected_bus = 118
    pg_before = next(
        r.get_float(3) for r in before.records if r.get_int(1) == protected_bus)
    other_before = {
        r.get_int(1): r.get_float(3)
        for r in before.records
        if r.get_int(2) != 0 and r.get_int(1) != protected_bus
    }
    monkeypatch.setattr(
        synth_mod, "psasp_referee", lambda *_a, **_k: {"converged": False})
    monkeypatch.setattr(
        nrpf_mod, "build_ybus", lambda _case, n: np.zeros((n, n), complex))
    monkeypatch.setattr(
        nrpf_mod, "solve_nr",
        lambda case, max_iter=40: SimpleNamespace(
            v=np.ones(case.n_buses), theta=np.zeros(case.n_buses)),
    )
    import scipy.sparse.csgraph as csgraph_mod
    real_dijkstra = csgraph_mod.dijkstra
    limits = []

    def tracked_dijkstra(*args, **kwargs):
        limits.append(kwargs.get("limit"))
        return real_dijkstra(*args, **kwargs)

    monkeypatch.setattr(csgraph_mod, "dijkstra", tracked_dijkstra)

    trace = RepairTrace.for_config(
        DEFAULT_REPAIR_CONFIG.with_protected({protected_bus}))
    synth_mod._p_redispatch(
        work, object(), max_rounds=1, protected_buses={protected_bus},
        trace=trace)

    after = Card.load(work / "LF.L5")
    pg_after = next(
        r.get_float(3) for r in after.records if r.get_int(1) == protected_bus)
    other_after = {
        r.get_int(1): r.get_float(3)
        for r in after.records
        if r.get_int(2) != 0 and r.get_int(1) != protected_bus
    }
    assert pg_after == pg_before
    assert any(other_after[b] != pg for b, pg in other_before.items())
    assert trace.observations[0]["source"] == "psasp"
    assert trace.observations[1]["source"] == "nrpf"
    assert all(
        a["object_id"] != protected_bus
        for a in trace.actions if a["status"] == "applied")
    assert any(
        a["object_id"] == protected_bus and a["status"] == "skipped_protected"
        for a in trace.actions)
    assert any(a["action_type"] == "redispatch_p" for a in trace.actions)
    assert limits and all(limit == DEFAULT_REPAIR_CONFIG.p_search_hops for limit in limits)
    for action in trace.actions:
        if action["action_type"] != "redispatch_p" or action["status"] != "applied":
            continue
        before_pg, after_pg = action["before"], action["after"]
        lower, upper = action["lower"], action["upper"]
        if lower <= before_pg <= upper:
            assert lower <= after_pg <= upper
        elif before_pg > upper:
            assert after_pg <= before_pg
        else:
            assert after_pg >= before_pg


def test_p_redispatch_supports_genno_l5_layout(monkeypatch, tmp_path):
    """CEPRI/39 的 +1 GenNo 版式必须按真实 Pg/Pmin/Pmax 列读写。"""
    import gridfm.eval.nrpf as nrpf_mod
    from gridfm.psasp_io.cards import GenView

    source = Path(__file__).parent / "fixtures" / "cepri36_golden"
    work = tmp_path / "cepri36"
    shutil.copytree(source, work)
    before = Card.load(work / "LF.L5")
    before_types = [
        GenView(record, row).type_code
        for row, record in enumerate(before.records)
    ]
    monkeypatch.setattr(
        synth_mod, "psasp_referee", lambda *_a, **_k: {"converged": False})
    monkeypatch.setattr(
        nrpf_mod, "build_ybus", lambda _case, n: np.zeros((n, n), complex))
    monkeypatch.setattr(
        nrpf_mod, "solve_nr",
        lambda case, max_iter=40: SimpleNamespace(
            v=np.ones(case.n_buses), theta=np.zeros(case.n_buses)),
    )

    trace = RepairTrace.for_config(DEFAULT_REPAIR_CONFIG)
    synth_mod._p_redispatch(
        work, object(), max_rounds=1, trace=trace)

    after = Card.load(work / "LF.L5")
    gens = [GenView(record, row) for row, record in enumerate(after.records)]
    assert [gen.type_code for gen in gens] == before_types
    assert all(gen._off == 1 for gen in gens)
    assert any(
        action["action_type"] == "redispatch_p"
        and action["status"] == "applied"
        for action in trace.actions)


def test_repair_pipeline_rolls_back_cards_on_exception(
        monkeypatch, tmp_path, case_dir):
    """引擎/写卡异常传播前恢复进入本轮时的 L5/L6 字节。"""
    work = tmp_path / "case"
    shutil.copytree(case_dir, work)
    l0_before = (work / "LF.L0").read_bytes()
    l5_before = (work / "LF.L5").read_bytes()
    l6_before = (work / "LF.L6").read_bytes()

    def broken_q(out_dir, *_a, **_k):
        (out_dir / "LF.L0").write_bytes(b"broken-l0")
        (out_dir / "LF.L5").write_bytes(b"broken-l5")
        (out_dir / "LF.L6").write_bytes(b"broken-l6")
        raise RuntimeError("engine failed")

    monkeypatch.setattr(synth_mod, "_ac_reactive_planning", broken_q)
    with pytest.raises(RuntimeError, match="engine failed"):
        synth_mod.repair_pipeline(work, object())

    assert (work / "LF.L0").read_bytes() == l0_before
    assert (work / "LF.L5").read_bytes() == l5_before
    assert (work / "LF.L6").read_bytes() == l6_before


def test_synth_convergent_rolls_back_load_scale_on_exception(
        monkeypatch, tmp_path, case_dir):
    """降载后的规则/引擎异常不得留下未经终审的 LF.L6。"""
    work = tmp_path / "case"
    shutil.copytree(case_dir, work)
    l6_before = (work / "LF.L6").read_bytes()
    results = iter([
        {"nr_ok": False, "q_rounds": 0, "p_rounds": 0},
        RuntimeError("repair failed"),
    ])

    def fake_repair(*_a, **_k):
        result = next(results)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(synth_mod, "synth_case", lambda *_a, **_k: {})
    monkeypatch.setattr(synth_mod, "repair_pipeline", fake_repair)
    with pytest.raises(RuntimeError, match="repair failed"):
        synth_mod.synth_convergent(
            118, case_dir, work, _AlwaysDivergedSolver(), max_repair=1)

    assert (work / "LF.L6").read_bytes() == l6_before


def test_repair_case_convergent_applies_bounded_load_fallback(
        monkeypatch, tmp_path, case_dir):
    """既有方式入口执行完整 Q→P→Q→降载链，且成功降载有对应终审结果。"""
    work = tmp_path / "case"
    shutil.copytree(case_dir, work)
    before = Card.load(work / "LF.L6").records[0].get_float(4)
    calls = []

    def scripted_pipeline(*_args, **kwargs):
        calls.append(1)
        ok = len(calls) == 2
        kwargs["trace"].finish(
            "final_psasp_converged" if ok else "final_psasp_diverged", ok)
        return {"nr_ok": ok, "final_converged": ok}

    monkeypatch.setattr(synth_mod, "repair_pipeline", scripted_pipeline)
    trace = RepairTrace.for_config(DEFAULT_REPAIR_CONFIG)
    result = synth_mod.repair_case_convergent(
        work, object(), max_repair=2, trace=trace)

    after = Card.load(work / "LF.L6").records[0].get_float(4)
    assert result["converged"] is True
    assert result["repairs"] == 1
    assert len(calls) == 2
    assert after == pytest.approx(before * DEFAULT_REPAIR_CONFIG.load_scale)
    assert any(
        action["action_type"] == "scale_load"
        for action in result["trace"]["actions"])


def test_repair_case_convergent_rolls_back_failed_load_transaction(
        monkeypatch, tmp_path, case_dir):
    work = tmp_path / "case"
    shutil.copytree(case_dir, work)
    before = (work / "LF.L6").read_bytes()
    calls = []

    def broken_second(*_args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return {"nr_ok": False, "final_converged": False}
        raise RuntimeError("engine failed after scale")

    monkeypatch.setattr(synth_mod, "repair_pipeline", broken_second)
    with pytest.raises(RuntimeError, match="engine failed after scale"):
        synth_mod.repair_case_convergent(
            work, object(), max_repair=1)
    assert (work / "LF.L6").read_bytes() == before
