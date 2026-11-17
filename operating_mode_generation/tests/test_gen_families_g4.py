"""G4 库生产配方单测(D14 转正强制单测 + 2026-07-18 批准修正配方锁定)。

离线,不跑 PSASP。D14 转正(交接-G线 §2-2):G4 起 train/val/transfer 无共同家族,
不再享单拓扑豁免——计划级(_plan 种子段)与数据级(audit_families_disjoint 读磁盘
manifest)双重断言。配方(_family_params)锁定 2026-07-18 呈批批准的修正:
diverse 全 split / amp transfer 收紧 / keep_cases 只在评估 split。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _mod():
    spec = importlib.util.spec_from_file_location(
        "gen_families", Path(__file__).parent.parent / "scripts" / "gen_families.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


GF = _mod()


# ---------------- D14 计划级:三 split 家族/种子段两两不相交 ----------------

def test_d14_plan_family_disjoint():
    jobs = GF._plan(64, 8, 4, 100, 50)
    by: dict = {}
    seeds: dict = {}
    for split, fid, _n, seed, _modes, _kd in jobs:
        by.setdefault(split, set()).add(fid)
        seeds.setdefault(split, []).append(seed)
    assert by["train"] and by["val"] and by["transfer"]
    assert not by["train"] & by["val"]
    assert not by["train"] & by["transfer"]
    assert not by["val"] & by["transfer"]
    # 种子段隔离(train 1000+ / val 2000+ / transfer 3000+)
    assert max(seeds["train"]) < 2000 <= min(seeds["val"])
    assert max(seeds["val"]) < 3000 <= min(seeds["transfer"])


def test_plan_synth_only_no_real_family_collision():
    # 计划只产 synth* 家族;真实家族(lib118 等)仅经步骤 3 语料并入 train,不可能撞名
    jobs = GF._plan(64, 8, 4, 100, 50)
    assert all(fid.startswith("synth") for _s, fid, _n, _sd, _m, _kd in jobs)


def test_transfer_pure_convergent_frozen():
    # 冻结迁移测试集 = 纯收敛库(keep_div=0);train/val 保留 8% 发散负类
    jobs = GF._plan(2, 2, 1, 10, 5)
    for split, _fid, _n, _seed, _modes, kd in jobs:
        assert kd == (0.0 if split == "transfer" else 0.08)


# ---------------- 修正配方锁定(2026-07-18 呈批批准) ----------------

def test_family_params_recipe_locked():
    # 2026-07-18 两次实测修正后的定稿配方(dict 单一出处):
    # transfer diverse=False(大网 0/50 全发散→§8.4 原语义 sample_mode,transfer2k 先例);
    # train/val = diverse-lite + 停机 0.15(三臂标定@270 合成基态:sample_mode 20/20、
    # 停机0 20/20、停机0.15 12/20、停机0.30 ~10%——真凶=停机重分配剂量效应)
    # S2(2026-07-18):停机自适应阶梯 0.15→0.08→0.0(固定 0.15 实测 270 档 1/13 达标)
    tv = {"amp": 1.0, "keep_cases": False, "diverse": True,
          "dispatch": "proportional", "decommit": (0.15, 0.08, 0.0)}
    assert GF._family_params("train", 150) == tv
    assert GF._family_params("train", 270) == tv
    assert GF._family_params("val", 210) == {**tv, "keep_cases": True}
    xf = {"keep_cases": True, "diverse": False, "dispatch": None, "decommit": None}
    assert GF._family_params("transfer", 400) == {**xf, "amp": 0.7}
    assert GF._family_params("transfer", 700) == {**xf, "amp": 0.5}
    assert GF._family_params("transfer", 1000) == {**xf, "amp": 0.5}


# ---------------- D14 数据级:磁盘 manifest 审计 ----------------

def _mk_family(root: Path, split: str, fid: str, family_col: str = None) -> None:
    d = root / split / fid
    d.mkdir(parents=True, exist_ok=True)
    fam = family_col if family_col is not None else fid
    (d / "manifest.csv").write_text(
        f"sample_id,family_id,solvable,load_factor\n{fid}_00000,{fam},1,1.05\n",
        encoding="utf-8")


def test_audit_families_disjoint_pass_and_violation(tmp_path):
    _mk_family(tmp_path, "train", "synth150_s1000")
    _mk_family(tmp_path, "val", "synth150_s2000")
    _mk_family(tmp_path, "transfer", "synth400_s3000")
    fams = GF.audit_families_disjoint(tmp_path)
    assert fams["train"] == {"synth150_s1000"}
    assert fams["transfer"] == {"synth400_s3000"}
    # 制造违规:val 里出现 train 家族 → 必须抛
    _mk_family(tmp_path, "val", "synth150_s1000")
    with pytest.raises(AssertionError):
        GF.audit_families_disjoint(tmp_path)


def test_audit_reads_family_id_column_not_dirname(tmp_path):
    # 审计以 manifest 的 family_id 列为准(独立事实源),而非目录名
    _mk_family(tmp_path, "train", "dirA", family_col="synth150_s1000")
    _mk_family(tmp_path, "val", "dirB", family_col="synth150_s1000")
    with pytest.raises(AssertionError):
        GF.audit_families_disjoint(tmp_path)
