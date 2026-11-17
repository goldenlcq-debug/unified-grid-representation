"""c_label v1 方式标签(方案 6.3① / 附录 C.3;2026-07-19 用户定稿)。

**v1 字段(两个,均自动回算 → (方式,标签) 零人工标注):**
- `load_level`:连续。回算 = ΣPl(方式) / ΣPl_base(家族基态)。容差 = 相对偏差 ≤5%。
- `online_count`:整数。回算 = 非平衡机 pg>0 的台数。容差 = ±1 台。

每字段配**缺失标记**;推理时用户未指定的字段一律置缺失(不填默认值),语义一致率
只在**已指定**字段上算(8.8)。断面/新能源/区域字段需领域定义(断面清单/新能源机组
规则/多区网络),合成家族无事实源,留 v2(用户 2026-07-19 裁决 v1 只含此两字段)。

**编码(AdaLN 条件向量,8 维,与 v0 denoiser LABEL_DIM 同宽,最小改动):**
  [0] load_level 标准化值(缺失时 0)
  [1] online 标准化值(缺失时 0)
  [2] load_level 缺失位(1=缺失)
  [3] online 缺失位
  [4..7] 预留(0)
CFG 无条件分支 = 两字段皆缺失(逐字段 dropout 亦复用此编码)。标准化统计随扩散检查点
固化(fit 于训练语料;编码不含身份、跨家族可比 —— online 用 pg>0 占比,规模无关)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..schema.fields import NODE_TABLE
from ..schema.sample import GridSample

LABEL_DIM = 8
_SLACK = 2                     # node_type one-hot 中 slack 的下标(与 g_diag_a 同款)
_PG_ON_EPS = 1e-6             # pg>此值 = 开机(空载并网 pg≈0 判停机)

FIELDS = ("load_level", "online_count")
TOLERANCE = {"load_level": ("relative", 0.05), "online_count": ("abs", 1)}


def _cols():
    o = NODE_TABLE.offset_of
    return dict(ntype=o("node_type"), null=o("is_null"), has_gen=o("has_gen"),
                has_load=o("has_load"), pg=o("pg"), pl=o("pl"))


def _nonslack_gen_mask(feat: np.ndarray, c: dict) -> np.ndarray:
    types = np.argmax(feat[:, c["ntype"]:c["ntype"] + 3], axis=1)
    not_null = feat[:, c["null"]] < 0.5
    has_gen = feat[:, c["has_gen"]] > 0.5
    return has_gen & not_null & (types != _SLACK)


def base_pl_sum(base_gs: GridSample) -> float:
    """家族基态总负荷 ΣPl_base(load_level 分母)。base_gs = _base_case 装配的档案。"""
    c = _cols()
    m = (base_gs.node_feat[:, c["has_load"]] > 0.5) & (base_gs.node_feat[:, c["null"]] < 0.5)
    return float(base_gs.node_feat[m, c["pl"]].sum())


def recompute(gs: GridSample, base_pl_sum_val: float) -> Dict[str, float]:
    """从方式档案回算 c_label v1(pu 空间;gs 为未归一化物理样本)。

    load_level 需家族基态总负荷 base_pl_sum_val(> 0);online_count 自足。
    """
    c = _cols()
    feat = gs.node_feat
    load_mask = (feat[:, c["has_load"]] > 0.5) & (feat[:, c["null"]] < 0.5)
    pl_sum = float(feat[load_mask, c["pl"]].sum())
    gen_ns = _nonslack_gen_mask(feat, c)
    n_gen = int(gen_ns.sum())
    online = int((gen_ns & (feat[:, c["pg"]] > _PG_ON_EPS)).sum())
    assert base_pl_sum_val > 0, "base_pl_sum 必须 > 0(家族基态总负荷)"
    return {"load_level": pl_sum / base_pl_sum_val,
            "online_count": online, "n_gen": n_gen}


@dataclass
class LabelNorm:
    """标准化统计(fit 于训练语料;随检查点固化)。online 走占比(规模无关)。"""
    load_mean: float
    load_std: float
    online_frac_mean: float
    online_frac_std: float

    def to_dict(self) -> dict:
        return {"load_mean": self.load_mean, "load_std": self.load_std,
                "online_frac_mean": self.online_frac_mean,
                "online_frac_std": self.online_frac_std}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelNorm":
        return cls(d["load_mean"], d["load_std"],
                   d["online_frac_mean"], d["online_frac_std"])


def fit_norm(labels) -> LabelNorm:
    """labels = recompute 输出列表(含 online_count/n_gen/load_level)。"""
    ll = np.array([x["load_level"] for x in labels], dtype=float)
    of = np.array([x["online_count"] / max(x["n_gen"], 1) for x in labels], dtype=float)
    return LabelNorm(float(ll.mean()), float(ll.std() + 1e-8),
                     float(of.mean()), float(of.std() + 1e-8))


def encode(label: Optional[Dict[str, float]], norm: LabelNorm,
           missing=("",), n_gen: Optional[int] = None) -> np.ndarray:
    """→ 8 维 AdaLN 条件向量。label=None 或 missing 含字段名 → 该字段置缺失。

    missing:要遮蔽的字段集合(逐字段 dropout / 推理未指定);全遮蔽 = CFG 无条件。
    online 用占比 online_count/n_gen 标准化(规模无关);n_gen 缺省取 label['n_gen']。
    """
    v = np.zeros(LABEL_DIM, dtype=np.float32)
    if label is None:
        v[2] = v[3] = 1.0                       # 两字段皆缺失(CFG 无条件)
        return v
    ng = n_gen if n_gen is not None else label.get("n_gen")
    if "load_level" in missing or label.get("load_level") is None:
        v[2] = 1.0
    else:
        v[0] = (label["load_level"] - norm.load_mean) / norm.load_std
    if "online_count" in missing or label.get("online_count") is None or not ng:
        v[3] = 1.0
    else:
        frac = label["online_count"] / max(ng, 1)
        v[1] = (frac - norm.online_frac_mean) / norm.online_frac_std
    return v


def match(gen_label: Dict[str, float], prompt: Dict[str, float]) -> Dict[str, bool]:
    """语义一致率:逐字段比对生成方式回算值 vs prompt(只在 prompt 已指定字段上)。"""
    out = {}
    for f in FIELDS:
        if prompt.get(f) is None:
            continue
        kind, tol = TOLERANCE[f]
        g, p = gen_label[f], prompt[f]
        if kind == "relative":
            out[f] = abs(g - p) <= tol * abs(p) if p != 0 else abs(g) <= tol
        else:
            out[f] = abs(g - p) <= tol
    return out
