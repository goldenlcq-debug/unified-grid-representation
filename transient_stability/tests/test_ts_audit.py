"""数据体检工具单测(2026-07-13 用户裁决"做";伪影必须被抓,好数据不误伤)。"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIX = Path(__file__).parent / "fixtures" / "cepri36_stout"

spec = importlib.util.spec_from_file_location("ts_audit", REPO / "scripts" / "ts_audit.py")
ts_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts_audit)


def _mk_batch(tmp_path, tag="c_000_L1-2-3_d100", t_end=5.0):
    """用 cepri36 夹具搭一个最小批目录(1 行,全 Type=1 配置)。"""
    root = tmp_path / "batch"
    for sub in ("stana", "stout", "sme"):
        (root / sub).mkdir(parents=True)
    shutil.copy2(FIX / "STANA.DAT", root / "stana" / f"{tag}.dat")
    shutil.copy2(FIX / "STOUT.DDD", root / "stout" / f"{tag}.ddd")
    shutil.copy2(FIX / "ST.SME", root / "sme" / f"{tag}.sme")
    row = {"tag": tag, "sample_id": "c_000", "feasible": True, "t_end": t_end}
    (root / "labels.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return root, row


def _audit(root, row):
    return ts_audit.audit_one(root, row, lambda sid: None)


def test_clean_batch_passes(tmp_path):
    root, row = _mk_batch(tmp_path)
    out = _audit(root, row)
    assert "fail" not in out and "skip" not in out
    assert out.get("voltage_skipped") == "no_type2_curves", "夹具无 Type-2,如实跳过"


def test_anchor_tamper_caught(tmp_path):
    """篡改 STANA col6 一行 → anchor_angle 必须抓到(解码错列/引擎异常的哨兵)。"""
    root, row = _mk_batch(tmp_path)
    p = root / "stana" / f"{row['tag']}.dat"
    lines = p.read_text(encoding="gb18030").splitlines()
    cells = lines[100].split(",")
    cells[6] = f"{float(cells[6]) + 1.0:15.3f}"          # 加 1°(远超 5e-3 容差)
    lines[100] = ",".join(cells)
    p.write_text("\n".join(lines), encoding="gb18030")
    out = _audit(root, row)
    assert "anchor_angle" in out.get("fail", {})


def test_nan_caught(tmp_path):
    root, row = _mk_batch(tmp_path)
    p = root / "stout" / f"{row['tag']}.ddd"
    raw = np.fromfile(p, dtype="<f8")
    raw[1000] = np.nan
    raw.tofile(p)
    out = _audit(root, row)
    assert "finite" in out.get("fail", {})


def test_truncated_window_caught(tmp_path):
    """整行截断(尺寸自洽但窗长不足)→ window + rows 双抓。"""
    root, row = _mk_batch(tmp_path, t_end=5.0)
    p = root / "stout" / f"{row['tag']}.ddd"
    raw = np.fromfile(p, dtype="<f8").reshape(-1, 9)
    raw[:300].tofile(p)                                   # 5s 窗截到 ~3s
    out = _audit(root, row)
    assert "window" in out.get("fail", {}) and "rows" in out["fail"]


def test_missing_archive_skipped_not_failed(tmp_path):
    root, row = _mk_batch(tmp_path)
    (root / "stout" / f"{row['tag']}.ddd").unlink()
    out = _audit(root, row)
    assert out.get("skip") == "missing_archive"


# ---------------- 功角封顶(2026-07-16 对抗审查 critical) ----------------

def _mk_saturated_batch(tmp_path, tamper_row=None):
    """合成一份"半程封顶"的批(2 机):前 300 步正常、后 300 步 δ1 打到 99999。

    引擎实况(118 实测):STANA 封的是**角差**、STOUT 封的是**每台机的 δ** —— 两个
    不同的量 → 封顶后 max_angle_spread(δ) 与 STANA 角差列必然不等(实测差 2k~22k°)。
    tamper_row:在**未封顶段**注入一个真实解码错误,验证豁免没把锚点检查弄瞎。
    """
    root = tmp_path / "sat"
    for sub in ("stana", "stout", "sme"):
        (root / sub).mkdir(parents=True)
    tag = "s_000_L1-2-3_d100"
    n = 600
    t = np.arange(n) * 0.01
    d1 = np.minimum(400.0 * t * 60.0, 99999.0)            # 后半程触顶
    d2 = 10.0 + 0.0 * t
    np.stack([t, d1, d2], axis=1).astype("<f8").tofile(root / "stout" / f"{tag}.ddd")
    (root / "sme" / f"{tag}.sme").write_bytes((
        "  1, 1,     2,     1,     2,     1,     5,     2,     5,     2,     0,"
        "     0,     0,     0,     0,     0,     0,     0,'发电机功角',\r\n"
        "  1, 8,   101,  1101,     0,     0,     0,     0,     0,     0,     0,"
        "     0,     0,     0,     0,     0,     0,     0,'',\r\n").encode("gb18030"))
    lines = []
    for i in range(n):
        spread = min(abs(d1[i] - d2[i]), 99999.0)         # STANA 对**角差**封顶
        if tamper_row is not None and i == tamper_row:
            spread += 1.0                                  # 未封顶段的真解码错误
        c = ["0.0"] * 28
        c[0] = f"{t[i]:.5f}"
        c[6] = f"{spread:.3f}"
        c[8] = "1.00000"
        c[12] = "1.00000"
        c[14] = "1.00000"
        c[18] = "1.00000"
        lines.append(",".join(c) + ",")
    (root / "stana" / f"{tag}.dat").write_bytes(
        ("\r\n".join(lines) + "\r\n").encode("gb18030"))
    row = {"tag": tag, "sample_id": "s_000", "feasible": True, "t_end": 6.0}
    (root / "labels.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return root, row


def test_saturated_rows_exempt_from_angle_anchor(tmp_path):
    """封顶段的锚点不可比(真值不可得)→ 豁免 + 如实计数,不得记 fail。
    未修前:118 实测 47/63 例体检必红 → 收口门永久拒纳整批(审查 critical)。"""
    root, row = _mk_saturated_batch(tmp_path)
    out = _audit(root, row)
    assert "fail" not in out, f"封顶段不得判红:{out.get('fail')}"
    assert out.get("angle_rows_saturated", 0) > 0, "封顶行数必须如实计数(不静默)"
    assert out["angle_rows_saturated"] < 600, "只封后半程(前半程仍逐行比对)"


def test_saturation_exemption_does_not_blind_anchor(tmp_path):
    """豁免只能作用于封顶段:未封顶段的真实解码错误照抓(否则等于把锚点关了)。"""
    root, row = _mk_saturated_batch(tmp_path, tamper_row=100)
    out = _audit(root, row)
    assert "anchor_angle" in out.get("fail", {}), "未封顶段的错列/篡改必须照抓"
    assert out["fail"]["anchor_angle"]["rows_saturated"] > 0


def test_ragged_stana_last_row_reported_not_crash(tmp_path):
    """STANA 末行截断到 9~18 列(引擎被杀/拷贝竞态):扩展列比 t 短一行——必须记
    stana_ragged_rows,不得广播崩溃炸穿整批(2026-07-13 审查独立复现)。"""
    root = tmp_path / "b3"
    for sub in ("stana", "stout", "sme"):
        (root / sub).mkdir(parents=True)
    tag = "r_000_L1-2-3_d100"
    n, buses = 5, [2, 5]
    t = np.arange(n) * 0.01
    m = np.column_stack([t, 10 + 0 * t, np.full(n, 1.0), np.full(n, 1.0)])
    m.astype("<f8").tofile(root / "stout" / f"{tag}.ddd")
    sme = ("  1, 1,     2,     1,     2,     1,     0,     0,     0,     0,     0,"
           "     0,     0,     0,     0,     0,     0,     0,     0,'a',\r\n"
           "  1, 2,     2,     5,     0,     0,     0,     0,     0,     0,     0,"
           "     0,     0,     0,     0,     0,     0,     0,     0,'v',\r\n")
    (root / "sme" / f"{tag}.sme").write_bytes(sme.encode("gb18030"))
    rows_txt = []
    for i in range(n):
        c = ["%12.5f" % t[i]] + ["0"] * 25
        c[6] = "%15.3f" % 0.0
        c[8] = "%10.5f" % 1.0
        c[14] = "%10.5f" % 1.0
        rows_txt.append(",".join(c) + ",")
    rows_txt[-1] = ",".join(rows_txt[-1].split(",")[:12])   # 末行截到 12 列
    (root / "stana" / f"{tag}.dat").write_text("\n".join(rows_txt), encoding="gb18030")
    row = {"tag": tag, "sample_id": "r_000", "feasible": True, "t_end": 0.04}
    out = ts_audit.audit_one(root, row, lambda sid: None)
    assert "stana_ragged_rows" in out.get("fail", {}), "参差行必须被报告而非崩溃"


def test_t0_check_against_lp1(tmp_path):
    """t0 检查:合成 Type-2 批 + 假 LP1 —— 一致放行,超差抓住。"""
    root = tmp_path / "b2"
    for sub in ("stana", "stout", "sme"):
        (root / sub).mkdir(parents=True)
    tag = "s_000_L1-2-3_d100"
    n, buses = 11, [2, 5, 9]
    t = np.arange(n) * 0.01
    v0 = {2: 1.02, 5: 0.98, 9: 1.00}
    vcols = np.stack([np.full(n, v0[b]) for b in buses], axis=1)
    ang = np.stack([10 + 0 * t, 20 + 0 * t], axis=1)
    m = np.column_stack([t, ang, vcols])
    m.astype("<f8").tofile(root / "stout" / f"{tag}.ddd")
    sme = ("  1, 1,     2,     1,     2,     1,     5,     2,     5,     2,     0,"
           "     0,     0,     0,     0,     0,     0,     0,     0,'ang',\r\n"
           "  1, 2,     2,     5,     9,     0,     0,     0,     0,     0,     0,"
           "     0,     0,     0,     0,     0,     0,     0,     0,'v',\r\n")
    (root / "sme" / f"{tag}.sme").write_bytes(sme.encode("gb18030"))
    stana_rows = []
    for i in range(n):
        c = ["%12.5f" % t[i]] + ["0"] * 25
        c[6] = "%15.3f" % 10.0                            # spread = 20-10
        c[8] = "%10.5f" % 0.98
        c[14] = "%10.5f" % 1.02
        stana_rows.append(",".join(c) + ",")
    (root / "stana" / f"{tag}.dat").write_text("\n".join(stana_rows),
                                               encoding="gb18030")
    row = {"tag": tag, "sample_id": "s_000", "feasible": True, "t_end": 0.1}
    out = ts_audit.audit_one(root, row, lambda sid: dict(v0))
    assert "fail" not in out and out.get("t0_checked")
    bad = dict(v0)
    bad[5] = 0.93                                          # LP1 与轨迹初值差 0.05
    out2 = ts_audit.audit_one(root, row, lambda sid: bad)
    assert "t0" in out2.get("fail", {}) and out2["fail"]["t0"]["bus"] == 5
