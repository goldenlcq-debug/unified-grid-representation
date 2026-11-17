"""ts_cct v2 电池支持(2026-07-14 B2 前置补丁;离线,不跑 PSASP)。

补丁两半:①--battery choices 纳入 BATTERY_V2;②blank_s11 电池拷贝算例后清空
ST.S11(与 ts_labels 同款——ieee39 交付卡为 K≠0 GUI 模板,换线路不清空 = 残行
污染仿真)。B2 发包 pin 必须 ≥ 本补丁提交。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

spec = importlib.util.spec_from_file_location("ts_cct", REPO / "scripts" / "ts_cct.py")
ts_cct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts_cct)


def test_all_battery_includes_v2():
    for b in ("cepri36", "ieee39", "cepri36_v2", "ieee39_v2"):
        assert b in ts_cct.ALL_BATTERY, f"--battery 缺 {b}"
    assert ts_cct.ALL_BATTERY["ieee39_v2"].get("blank_s11") is True
    assert not ts_cct.ALL_BATTERY["cepri36_v2"].get("blank_s11")


def test_prep_fault_card_blanks_template_only_for_blank_s11(tmp_path):
    template = b"1, 34, 28, 23, ... K!=0 GUI template rows ...,\n"
    for battery, expect_blank in (("ieee39_v2", True), ("cepri36_v2", False),
                                  ("cepri36", False), ("ieee39", False)):
        work = tmp_path / battery
        work.mkdir()
        (work / "ST.S11").write_bytes(template)
        ts_cct.prep_fault_card(work, ts_cct.ALL_BATTERY[battery])
        content = (work / "ST.S11").read_bytes()
        if expect_blank:
            assert content == b"", f"{battery}: 模板残行未清空"
        else:
            assert content == template, f"{battery}: 不该动既有卡"
