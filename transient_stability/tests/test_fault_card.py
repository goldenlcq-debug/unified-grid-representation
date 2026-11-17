"""故障写卡 v1(T4.1 反推验证落地;变更记录 2026-07-11):从零写卡 + ST.S0 Nfault 同步。

离线测试(不跑 PSASP;引擎级验证已由 CEPRI-36 实算完成:Nfault=0 时卡被静默忽略,
补计数后三相短路 vmin 0.0001 / 永久断线 amax 168.9°)。
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gridfm.psasp_io.case import PsaspCase
from gridfm.psasp_io.lowlevel import Card
from gridfm.solver.base import FaultSpec
from gridfm.solver.psasp import sync_s0_nfault, write_fault_card

FIX = Path(__file__).parent / "fixtures"


def _copy(tmp_path, name):
    dst = tmp_path / "case"
    shutil.copytree(FIX / name, dst)
    return dst


def test_write_from_scratch_on_empty_s11(tmp_path):
    """CEPRI-36 自带空 S11:从零写卡 + Nfault 0→1。"""
    case = _copy(tmp_path, "cepri36_golden")
    assert Card.load(case / "ST.S0").records[0].get_int(7) == 0
    write_fault_card(case / "ST.S11", FaultSpec(29, 27, 23, k_pct=0.0, ts=0.1, te=0.2))
    faults = PsaspCase(case).faults()
    assert len(faults) == 1
    f = faults[0]
    assert (f.i_no, f.j_no, f.id_no) == (29, 27, 23)
    assert f.k_pct == 0.0 and f.phases == (1, 1, 1)
    assert f.rec.get_float(f.TS) == pytest.approx(0.1)
    assert f.rec.get_float(f.TE) == pytest.approx(0.2)
    assert f.rec.get_int(f.ADD_NO) == 0
    assert f.rec.n_fields() >= 17, "须含 Add_Name_2 字段(ST.S11.txt 字段表)"
    assert Card.load(case / "ST.S0").records[0].get_int(7) == 1


def test_write_from_scratch_midline_adds_fault_point_bus(tmp_path):
    """T4.1 完整范围:空卡 K=50 → 四处落笔(ST.S1/LF.L1 追加、Add_No、S0.Nbus),
    LF.L0/LF.L2 字节不动(线路不拆分,引擎按 K% 内部完成——39/118 模板解剖实证)。"""
    case = _copy(tmp_path, "cepri36_golden")
    n0 = len(Card.load(case / "ST.S1").records)
    assert len(Card.load(case / "LF.L1").records) == n0
    assert Card.load(case / "ST.S0").records[0].get_int(0) == n0
    l0_before = (case / "LF.L0").read_bytes()
    l2_before = (case / "LF.L2").read_bytes()
    write_fault_card(case / "ST.S11", FaultSpec(29, 27, 23, k_pct=50.0, ts=0.1, te=0.2))
    s1 = Card.load(case / "ST.S1").records
    l1 = Card.load(case / "LF.L1").records
    assert len(s1) == n0 + 1 and len(l1) == n0 + 1
    assert s1[-1].get_str(0) == l1[-1].get_str(0)
    assert s1[-1].get_str(0).endswith("_50_00"), "命名照 39 模板版式 {线名}_{K整数}_{两位小数}"
    assert l1[-1].n_fields() == 8, "故障点母线行 7 字段收尾(无 Bus_Name)+ 尾空单元格"
    assert l1[-1].get_float(1) == 0.0, "Vbase=0(占位母线,模板一致)"
    f = PsaspCase(case).faults()[0]
    assert f.k_pct == 50.0
    assert f.add_no == n0 + 1, "Add_No = 新增母线行号(行号即编号,ST.S1.txt)"
    assert Card.load(case / "ST.S0").records[0].get_int(0) == n0 + 1, "S0.Nbus 同步"
    assert Card.load(case / "ST.S0").records[0].get_int(7) == 1, "Nfault 同步"
    assert (case / "LF.L0").read_bytes() == l0_before, "LF.L0 NBB 不计故障母线"
    assert (case / "LF.L2").read_bytes() == l2_before, "线路卡不拆分"


def test_midline_rewrite_is_idempotent(tmp_path):
    """CCT 二分对同一 case_dir 反复写卡:同线同 K 不得重复追加母线(名字幂等复用行号)。"""
    case = _copy(tmp_path, "cepri36_golden")
    write_fault_card(case / "ST.S11", FaultSpec(29, 27, 23, k_pct=50.0, ts=0.1, te=0.2))
    s1_bytes = (case / "ST.S1").read_bytes()
    l1_bytes = (case / "LF.L1").read_bytes()
    s0_bytes = (case / "ST.S0").read_bytes()
    write_fault_card(case / "ST.S11", FaultSpec(29, 27, 23, k_pct=50.0, ts=0.1, te=0.3))
    assert (case / "ST.S1").read_bytes() == s1_bytes
    assert (case / "LF.L1").read_bytes() == l1_bytes
    assert (case / "ST.S0").read_bytes() == s0_bytes
    f = PsaspCase(case).faults()[0]
    assert f.rec.get_float(f.TE) == pytest.approx(0.3)
    assert f.add_no == len(Card.load(case / "ST.S1").records)


def test_existing_template_add_no_follows_fault(tmp_path):
    """118 交付卡自带 K=50/Add_No=122(名 'AC155_10' 名实不符——引擎不校验名字):
    复用路径改写故障必须重置 Add_No——K=0 清零(修复:此前不碰该字段,残留陈旧指向);
    换线 K=50 则注册新母线并重指(陈旧母线保留为占位,与 NULL 行同性质)。"""
    case = _copy(tmp_path, "ieee118_golden")
    assert PsaspCase(case).faults()[0].add_no == 122  # 交付原状
    write_fault_card(case / "ST.S11", FaultSpec(37, 40, 111, k_pct=0.0, ts=0.1, te=0.2))
    assert PsaspCase(case).faults()[0].add_no == 0
    write_fault_card(case / "ST.S11", FaultSpec(37, 40, 111, k_pct=50.0, ts=0.1, te=0.2))
    f = PsaspCase(case).faults()[0]
    assert f.add_no == 123
    assert Card.load(case / "ST.S1").records[122].get_str(0) == "AC111_50_00"
    assert Card.load(case / "ST.S0").records[0].get_int(0) == 123


def test_midline_k_out_of_range_or_unknown_line_rejected(tmp_path):
    """K=100/负值/未知支路必须拒绝(端部语义用 K=0;未知线 = 故障落空,响亮失败)。"""
    case = _copy(tmp_path, "cepri36_golden")
    for bad in (100.0, 120.0, -5.0):
        with pytest.raises(ValueError):
            write_fault_card(case / "ST.S11", FaultSpec(29, 27, 23, k_pct=bad))
    with pytest.raises(ValueError):
        write_fault_card(case / "ST.S11", FaultSpec(1, 2, 999, k_pct=50.0))


def test_midline_bus_keeps_catalog_alignment(tmp_path):
    """交接坑 1:新增母线后 LF 侧母线数 +1,行号对齐(节点行 = 计算母线编号 −1)对
    "末尾追加"必须稳定——既有节点 token_row 逐一不移,新末行 = 新编号。"""
    from gridfm.psasp_io.catalog import Catalog

    case = _copy(tmp_path, "cepri36_golden")
    cat0 = Catalog(PsaspCase(case))
    rows_before = {e.name: e.token_row for e in cat0.nodes if not e.is_null}
    write_fault_card(case / "ST.S11", FaultSpec(29, 27, 23, k_pct=50.0, ts=0.1, te=0.2))
    cat1 = Catalog(PsaspCase(case))
    assert cat1.n_nodes == cat0.n_nodes + 1
    assert cat1.n_branches == cat0.n_branches, "支路 token 不受故障母线影响"
    for name, row in rows_before.items():
        assert cat1.node_row_by_name(name) == row, f"既有节点 {name} 行号移位"
    assert cat1.nodes[-1].token_row == cat0.n_nodes + 1
    assert cat1.nodes[-1].vbase == 0.0


def test_write_on_existing_template_keeps_count(tmp_path):
    """39 的 GUI 模板路径:改字段不改行数,Nfault 保持 1 且 S0 其余字节不动。"""
    case = _copy(tmp_path, "ieee39_golden")
    before = (case / "ST.S0").read_bytes()
    assert Card.load(case / "ST.S0").records[0].get_int(7) == 1
    write_fault_card(case / "ST.S11", FaultSpec(34, 28, 23, k_pct=20.0, te=0.25))
    f = PsaspCase(case).faults()[0]
    assert f.rec.get_float(f.TE) == pytest.approx(0.25)
    assert Card.load(case / "ST.S0").records[0].get_int(7) == 1
    assert (case / "ST.S0").read_bytes() == before, "同值同宽回写必须字节级一致"


def test_sync_nfault_missing_s0_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sync_s0_nfault(tmp_path, 1)


def test_set_s0_duration_field14(tmp_path):
    """ST.S0 第 14 字段 = 暂态仿真时长(07-12 修:t_end 曾为死参数,5s 窗截断慢失稳)。"""
    import shutil

    import pytest

    from gridfm.psasp_io.lowlevel import Card
    from gridfm.solver.psasp import set_s0_duration

    fx = Path(__file__).parent / "fixtures" / "cepri36_golden"
    for f in ("ST.S0",):
        shutil.copy2(fx / f, tmp_path / f)
    before = Card.load(tmp_path / "ST.S0").records[0]
    vals_before = [before.get_int(i) for i in (0, 1, 2, 7)]
    assert before.get_int(13) == 5                      # cepri36 交付窗长
    set_s0_duration(tmp_path, 10.0)
    after = Card.load(tmp_path / "ST.S0").records[0]
    assert after.get_int(13) == 10
    assert [after.get_int(i) for i in (0, 1, 2, 7)] == vals_before, "其他字段不得动"
    with pytest.raises(ValueError):
        set_s0_duration(tmp_path, 7.5)                  # v0 只支持整数秒
