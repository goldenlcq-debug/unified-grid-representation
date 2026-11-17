"""无功调整标签生成 · 纯函数单测(方向二选项 B;无 PSASP)。

锁:read_ctrl(读 v_set/无功界,复用 GenView 版式自适应)、vset_eff_bounds、apply_dvset
(写 Δv_set 裁剪到有效界并返回实际生效量)、sample_vset_perturbation(降 v_set 造低压)、
双边判据 _is_healthy_2s/_is_violated_2s。PSASP 驱动(generate_bfam/main)试产点火,不进单测。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gridfm.psasp_io.lowlevel import Card
from gridfm.reason.reactive_task import VIO_HIGH, VIO_LOW
from scripts.reactive_labels import (
    VSET_HI, VSET_LO, _is_healthy_2s, _is_violated_2s, apply_dvset, read_ctrl,
    sample_vset_perturbation, vset_eff_bounds,
)

FIX = Path(__file__).parent / "fixtures" / "cepri36_golden"


def _l5():
    return Card.load(FIX / "LF.L5")


def test_read_ctrl_reads_vset_and_qlimits():
    slack, gens = read_ctrl(_l5())
    assert slack is not None
    assert len(gens) >= 2
    for g in gens.values():
        assert "v0" in g and "qmax" in g and "qmin" in g
        assert 0.8 < g["v0"] < 1.2, "机端电压设定应在合理带内"
        assert g["qmax"] >= g["qmin"]


def test_vset_eff_bounds_band_and_widen():
    assert vset_eff_bounds(1.03) == (VSET_LO, VSET_HI)
    lo, hi = vset_eff_bounds(1.15)                 # cur 在带外 → 放宽含 cur
    assert lo == VSET_LO and hi == 1.15
    lo, hi = vset_eff_bounds(0.90)
    assert lo == 0.90 and hi == VSET_HI


def test_apply_dvset_writes_and_clips_returns_effective():
    slack, gens = read_ctrl(_l5())
    gi = list(gens)[0]
    v0 = gens[gi]["view"].v0
    applied = apply_dvset(gens, {gi: 0.02})
    assert gens[gi]["view"].v0 == np.float32(v0 + 0.02).item() or \
        abs(gens[gi]["view"].v0 - (v0 + 0.02)) < 1e-4
    assert abs(applied[gi] - 0.02) < 1e-4, "带内施加,实际生效 = 名义"
    # 越上界裁剪:实际生效 < 名义
    gens[gi]["view"].v0 = VSET_HI
    applied = apply_dvset(gens, {gi: 0.5})
    assert applied[gi] < 0.5 and gens[gi]["view"].v0 <= VSET_HI + 1e-6


def test_sample_vset_perturbation_lowers_vset():
    _s, gens = read_ctrl(_l5())
    rng = np.random.default_rng(0)
    d = sample_vset_perturbation(gens, rng, k=2, d_lo=0.03, d_hi=0.08, sign=-1)
    assert len(d) == 2
    for dv in d.values():
        assert -0.08 <= dv <= -0.03, "降 v_set 造低压(sign=-1)"
    # 确定性
    assert sample_vset_perturbation(gens, np.random.default_rng(0), 2) == d
    # k 截断
    assert len(sample_vset_perturbation(gens, rng, k=999)) == len(gens)


def test_bilateral_criteria():
    assert _is_healthy_2s({"converged": True, "vmin": 0.94, "vmax": 1.06})
    assert not _is_healthy_2s({"converged": True, "vmin": 0.92, "vmax": 1.06}), "低压不健康"
    assert not _is_healthy_2s({"converged": True, "vmin": 0.94, "vmax": 1.10}), "过压不健康(双边)"
    assert not _is_healthy_2s({"converged": False})
    assert _is_violated_2s({"converged": True, "vmin": 0.92, "vmax": 1.06})   # 低压越限
    assert _is_violated_2s({"converged": True, "vmin": 0.94, "vmax": 1.10})   # 过压越限
    assert not _is_violated_2s({"converged": True, "vmin": 0.94, "vmax": 1.06})
    # VIO_HIGH=1.08:vmax 1.075 健康、1.085 越限
    assert _is_healthy_2s({"converged": True, "vmin": 0.94, "vmax": 1.075})
    assert _is_violated_2s({"converged": True, "vmin": 0.94, "vmax": 1.085})
    assert VIO_LOW == 0.93 and VIO_HIGH == 1.08
