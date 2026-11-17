"""c_label v1 单测(2026-07-19 用户定稿:load_level + online_count)。离线,不跑 PSASP。"""
from __future__ import annotations

import numpy as np
import pytest

from gridfm.data.c_label import (LABEL_DIM, LabelNorm, base_pl_sum, encode,
                                 fit_norm, match, recompute)
from gridfm.schema.fields import NODE_TABLE
from gridfm.psasp_io.case import PsaspCase
from gridfm.psasp_io.catalog import Catalog
from gridfm.schema.assemble import build_sample


@pytest.fixture(scope="module")
def base_sample(case_dir):
    c = PsaspCase(case_dir)
    return build_sample(c, Catalog(c), sample_id="base", family_id="ieee118")


def test_recompute_load_level_and_online(base_sample):
    base = base_sample
    b = base_pl_sum(base)
    assert b > 0
    # 自身对基态回算:load_level ≈ 1.0
    lab = recompute(base, b)
    assert abs(lab["load_level"] - 1.0) < 1e-6
    assert lab["online_count"] <= lab["n_gen"]

    # 构造"负荷 ×1.2"方式 → load_level ≈ 1.2
    import copy
    hot = copy.deepcopy(base)
    c = {k: NODE_TABLE.offset_of(v) for k, v in
         [("has_load", "has_load"), ("null", "is_null"), ("pl", "pl")]}
    m = (hot.node_feat[:, c["has_load"]] > 0.5) & (hot.node_feat[:, c["null"]] < 0.5)
    hot.node_feat[m, c["pl"]] *= 1.2
    assert abs(recompute(hot, b)["load_level"] - 1.2) < 1e-5


def test_online_count_tracks_pg_zero(base_sample):
    import copy
    from gridfm.data.c_label import _cols, _nonslack_gen_mask
    b = base_pl_sum(base_sample)
    n0 = recompute(base_sample, b)["online_count"]
    # 把两台非平衡机 pg 置 0(停机)→ online_count 减 2
    s = copy.deepcopy(base_sample)
    c = _cols()
    gens = np.where(_nonslack_gen_mask(s.node_feat, c)
                    & (s.node_feat[:, c["pg"]] > 1e-6))[0]
    assert len(gens) >= 2
    s.node_feat[gens[:2], c["pg"]] = 0.0
    assert recompute(s, b)["online_count"] == n0 - 2


def test_encode_missing_and_cfg():
    norm = LabelNorm(1.0, 0.1, 0.8, 0.1)
    lab = {"load_level": 1.1, "online_count": 40, "n_gen": 50}
    # 全指定:值位非零、缺失位 0
    v = encode(lab, norm)
    assert v.shape == (LABEL_DIM,)
    assert abs(v[0] - (1.1 - 1.0) / 0.1) < 1e-5           # load 标准化
    assert abs(v[1] - (40 / 50 - 0.8) / 0.1) < 1e-5       # online 占比标准化
    assert v[2] == 0.0 and v[3] == 0.0
    # 逐字段缺失:值位 0、缺失位 1
    v2 = encode(lab, norm, missing=("load_level",))
    assert v2[0] == 0.0 and v2[2] == 1.0 and v2[3] == 0.0
    # CFG 无条件(label=None):两字段皆缺失
    v3 = encode(None, norm)
    assert v3[2] == 1.0 and v3[3] == 1.0 and v3[0] == 0.0 and v3[1] == 0.0


def test_encode_online_scale_invariant():
    """online 用占比标准化 → 不同规模网络同占比得同编码(规模无关,可跨家族条件)。"""
    norm = LabelNorm(1.0, 0.1, 0.8, 0.1)
    small = encode({"load_level": 1.0, "online_count": 40, "n_gen": 50}, norm)
    big = encode({"load_level": 1.0, "online_count": 80, "n_gen": 100}, norm)
    assert abs(small[1] - big[1]) < 1e-6                  # 同占比 0.8 → 同编码


def test_fit_norm():
    labels = [{"load_level": 0.95, "online_count": 45, "n_gen": 50},
              {"load_level": 1.05, "online_count": 40, "n_gen": 50},
              {"load_level": 1.15, "online_count": 35, "n_gen": 50}]
    norm = fit_norm(labels)
    assert abs(norm.load_mean - 1.05) < 1e-6
    assert norm.load_std > 0 and norm.online_frac_std > 0
    # 往返
    assert LabelNorm.from_dict(norm.to_dict()).load_mean == norm.load_mean


def test_match_tolerances_and_missing():
    # load_level 相对 ≤5%、online_count ±1
    g = {"load_level": 1.03, "online_count": 41}
    assert match(g, {"load_level": 1.0}) == {"load_level": True}     # 1.03 vs 1.0 = 3% ≤5% → True
    assert match({"load_level": 1.04, "online_count": 41},
                 {"load_level": 1.0})["load_level"] is True          # 4% ≤5%
    assert match({"load_level": 1.07, "online_count": 41},
                 {"load_level": 1.0})["load_level"] is False         # 7% >5%
    assert match(g, {"online_count": 40})["online_count"] is True    # |41-40|=1 ≤1
    assert match(g, {"online_count": 39})["online_count"] is False   # |41-39|=2 >1
    # prompt 未指定字段不计入(语义一致率只在已指定字段)
    assert "online_count" not in match(g, {"load_level": 1.0})
