# -*- coding: utf-8 -*-
"""E18(预注册 §21):从各语料根的 `families/*/*/_base_case`(PSASP 已解卡片)提取基态提示常量,
落机器级缓存 `runs/base_cache/<family>.npz`(theta / pinj / v / n_nodes)。**不写语料目录**(数据纪律 §4.4-3)。

同名家族出现在多个根(mp_activsg2000 在交付与扩产两处):两份基态逐位一致则跳过,不一致按先到的保留并告警。
用法(grid-foundation/ 下):python scripts/probes_5060/build_base_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.psasp_io.catalog import Catalog  # noqa: E402
from gridfm.schema.assemble import build_sample  # noqa: E402
from gridfm.schema.fields import NODE_TABLE  # noqa: E402

noff = NODE_TABLE.offset_of
D = ROOT.parent / "大规模算例语料交付-20260821" / "大规模算例语料交付-20260821"
ROOTS = [ROOT.parent / "gridfm-data", D / "gridfm-data-large", ROOT.parent / "gridfm-data-large-ext"]
OUT = ROOT / "runs" / "base_cache"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    n_new = n_same = n_conflict = 0
    for root in ROOTS:
        for bd in sorted(root.glob("families/*/*/_base_case")):
            fam = bd.parent.name
            f = OUT / f"{fam}.npz"
            try:
                c = PsaspCase(bd)
                g = build_sample(c, Catalog(c), sample_id="base", family_id=fam)
            except Exception as e:
                print(f"[base] {fam} @ {bd}:提取失败 {e}", flush=True)
                continue
            th = g.node_feat[:, noff("theta")].astype(np.float64)
            pinj = (g.node_feat[:, noff("pg")] - g.node_feat[:, noff("pl")]).astype(np.float64)
            v = g.node_feat[:, noff("v")].astype(np.float64)
            # §6-2 暴露面自检(2026-08-24 潮流清场修复件后加):基态经"不经 run_powerflow 直接装配"取得,
            # 须验卡片与解自洽 —— 真值代入干净行失配应在存档精度量级(刚性支路 b~1e4 上 1e-5 舍入 → ≤1e-2);
            # 粗大失配(>0.05)= 疑似 donor 残留解,拒绝入缓存。
            try:
                import copy as _copy, torch as _torch
                from gridfm.data.normalize import Normalizer as _N
                from gridfm.train.masking import _pf_typed_mask as _msk
                from gridfm.reason.pf_physics import pf_phys_arrays as _pa, pf_phys_denorm as _pd, pf_phys_residual as _pr
                from gridfm.reason.pf_task import PF_FIELDS as _PF
                _arr = _pa(g, shunt_zero_harmless=True)
                _norm = _N().fit([g]); _s = _copy.deepcopy(g); _norm.transform(_s)
                _mf, _tg, _lm = _msk(_s.node_feat, _s.node_missing)
                _t = _torch.from_numpy(_tg); _pred = _t[:, [noff(f) for f in _PF]].clone()
                _dp, _dq = _pr(_pred, _t, _torch.from_numpy(_lm), _torch.from_numpy(_arr["index"]),
                               _torch.from_numpy(_arr["g"]), _torch.from_numpy(_arr["b"]), _torch.from_numpy(_pd(_norm)))
                _mx = max(float(_dp[_arr["eqp_clean"]].abs().max()), float(_dq[_arr["eqq_clean"]].abs().max()))
                if _mx > 0.05:
                    print(f"[base] 拒绝 {fam}:真值代入失配 {_mx:.3f} > 0.05(疑似残留解,§6-2)", flush=True)
                    continue
            except Exception as _e:
                print(f"[base] {fam} 自检异常(照存,人工复核):{_e}", flush=True)
                _mx = float("nan")
            if f.exists():
                z = np.load(f)
                if z["theta"].shape == th.shape and np.array_equal(z["theta"], th) and np.array_equal(z["pinj"], pinj):
                    n_same += 1
                else:
                    n_conflict += 1
                    print(f"[base] 冲突:{fam} 在 {root.name} 的基态与已缓存不一致,保留先到的(检查语料!)", flush=True)
                continue
            np.savez_compressed(f, theta=th, pinj=pinj, v=v, n_nodes=np.int64(g.n_nodes),
                                phys_mismatch_max=np.float64(_mx))
            n_new += 1
            print(f"[base] {fam:<24} N={g.n_nodes:<6} ← {bd}", flush=True)
    print(f"[base] 完成:新建 {n_new},同名一致跳过 {n_same},冲突 {n_conflict};缓存目录 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
