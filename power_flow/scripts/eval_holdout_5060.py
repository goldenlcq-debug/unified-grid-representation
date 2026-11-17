#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实尺度语料"库内留出"无偏评估(2026-08-21 第 2 轮固化;原为临时脚本,现落为正式脚本)。

只评估 `--val-holdout` 清单里的方式(训练期从未见过,但与训练家族同库),按家族分组
报逐字段标幺 MAE 与零学习 L2 技巧分。口径与 `scripts/eval_pf_ckpt.py` / 训练终评
**逐位同源**(直接复用 pf_reasoner.eval_pu_mae 与 pf_zero_learning_baseline)。

与 eval_pf_ckpt.py 的区别:后者把留出并进 val(和 448 个合成样本混池),本脚本
**只看留出的真实尺度方式**,并分家族拆开——这是"没见过的真实大网上到底多准"的无偏读数。

用法(grid-foundation/ 下):
  D=../大规模算例语料交付-20260821/大规模算例语料交付-20260821
  python scripts/eval_holdout_5060.py --bs 1 --device cpu \
    --extra-root "$D/gridfm-data-large" --extra-root ../gridfm-data-large-ext \
    --val-holdout "$D/val留出清单-mp_activsg2000.json" \
    --val-holdout ../gridfm-data-large-ext/val留出清单-mp_activsg10k.json \
    --ckpt runs/pf_reasoner_5060_R1dcL_edge0_loc_wrap_L_s1/ckpt/best.pt \
    --out runs/diag_5060/holdout44.json

θ 单位 = 弧度。万节点图请用 --bs 1;CPU fp32 更省显存(不与训练抢 GPU)。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pf_reasoner as M  # noqa: E402  (复用 eval_pu_mae,口径同源)
from gridfm.data.dataset import family_files  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.reason.pf_physics import dc_theta, dc_theta_lossy, pf_phys_denorm  # noqa: E402
from gridfm.reason.pf_task import (  # noqa: E402
    PF_FIELDS, PF_SIDE_DIM, PFReasoner, pf_zero_learning_baseline,
)
from gridfm.reason.pf_theta_edge import EDGE_SIDE_DIM  # noqa: E402
from gridfm.schema.fields import NODE_TABLE  # noqa: E402
from gridfm.schema.sample import GridSample  # noqa: E402
from gridfm.train.masking import _pf_typed_mask  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402


def load_holdout(extra_roots, holdout_paths):
    """只加载命中留出清单的 npz(路径集合与 load_families 同源:同一 manifest 驱动)。"""
    ids = set()
    for hp in holdout_paths:
        ids |= set(json.loads(Path(hp).read_text(encoding="utf-8"))["val_holdout_sample_ids"])
    raws = []
    for er in extra_roots:
        for p in family_files(Path(er), "train"):
            if Path(p).stem in ids:
                raws.append(GridSample.load_npz(p))
    missing = ids - {s.sample_id for s in raws}
    if missing:
        raise SystemExit(f"[holdout] 清单有 {len(missing)} 个 id 未命中:{sorted(missing)[:5]}")
    return raws


def dc_bare_rows(grp):
    """DC 裸值行(预注册 §3 口径:"DC 裸值行照报")——模型必须显著优于这条**白送的**物理提示
    才算学到东西;只报技巧分(对零学习 L2)会高估模型贡献。返回 (普通DC, 带网损DC, 回退数, 有监督样本数)。"""
    th = NODE_TABLE.offset_of("theta")
    e_dc, e_lo, n_fb, n_sup = [], [], 0, 0
    for s in grp:
        _mf, _tg, lm = _pf_typed_mask(s.node_feat, s.node_missing)
        d = dc_theta(s)
        m = lm[:, th] & (d[:, 1] > 0.5)
        if m.any():
            e_dc.append(np.abs(d[m, 0] - s.node_feat[m, th]))
        dl = dc_theta_lossy(s)
        ml = lm[:, th] & (dl[:, 1] > 0.5)
        if ml.any():
            e_lo.append(np.abs(dl[ml, 0] - s.node_feat[ml, th]))
            n_sup += 1
            n_fb += int(dl[ml, 2].max() > 0.5)
    f = lambda e: float(np.concatenate(e).mean()) if e else float("nan")   # noqa: E731
    return f(e_dc), f(e_lo), n_fb, n_sup


def build_from_ckpt(ckpt_path, v1_dir, dev, seed=0):
    """按检查点配置重建(冻结 V1 编码器 + 推理器);与 eval_pf_ckpt.py 逐位同源。"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    rc = ck["config"]
    view, encoder = ck.get("view", "field"), ck.get("encoder", "v1")
    dc = bool(rc.get("dc_theta", False))
    dc_enc = rc.get("dc_encoding", "raw")
    phys_w, phys_def = ck.get("phys_weight", 0.0), ck.get("phys_defeq_weight", 0.0)
    use_phys = phys_w > 0 or phys_def > 0 or bool(rc.get("refine_steps", 0))
    theta_edge = bool(rc.get("theta_edge", False))

    vck = torch.load(Path(v1_dir) / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in vck["config"].items() if k in allowed})
    torch.manual_seed(seed)
    enc, _ = build_models(cfg)
    if encoder == "v1":
        enc.load_state_dict(vck["model_state"])
    elif encoder == "e2e":
        enc.load_state_dict(ck["encoder_state"])
    enc.to(dev).eval()
    norm = Normalizer.load(Path(v1_dir) / "normalizer.npz")

    dc_dim = 0 if not dc else {"raw": 2, "bounded": 4, "lossy": 3, "lossyg": 3, "lossy2": 2, "lossyv": 3, "lossybv": 3}[dc_enc]
    v_hint = bool(rc.get("v_hint", False))
    side_dim = (PF_SIDE_DIM + dc_dim + (3 if v_hint else 0)) if view == "group" else 0
    reasoner = PFReasoner(d=rc.get("d", cfg.d), n_queries=rc["queries"], n_layers=rc["layers"],
                          side_dim=side_dim, gnn_layers=rc.get("gnn_layers", 0),
                          theta_head=rc.get("theta_head", False),
                          edge_side_dim=EDGE_SIDE_DIM if theta_edge else 0,
                          theta_edge=theta_edge, theta_wrap=bool(rc.get("theta_wrap", False)),
                          edge_dc_residual=rc.get("edge_dc_residual", True),
                          edge_head_local=rc.get("edge_head_local", False),
                          refine_steps=rc.get("refine_steps", 0), refine_qv=bool(rc.get("refine_qv", False)),
                          refine_ls=bool(rc.get("refine_ls", False)),
                          v_hint=bool(rc.get("v_hint", False))).to(dev)
    reasoner.load_state_dict(ck["reasoner_state"])
    reasoner.eval()
    _vc = NODE_TABLE.offset_of("v")
    meta = dict(step=ck.get("step"), view=view, encoder=encoder, dc=dc, dc_enc=dc_enc,
                use_phys=use_phys, theta_edge=theta_edge, v_hint=v_hint,
                vstat=(float(norm.node_mean[_vc]), float(norm.node_std[_vc])) if v_hint else None,
                dc_mode=(dc_enc if dc else False),
                use_raw=(use_phys or dc or theta_edge or v_hint), negb=rc.get("refine_negb", "sign"),
                shz=bool(rc.get("shunt_zero_harmless", False)),
                edge_base=rc.get("edge_base", "acv"))
    return enc, reasoner, cfg, norm, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-root", action="append", required=True)
    ap.add_argument("--val-holdout", action="append", required=True)
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--v1-dir", default=str(HERE.parent / "runs" / "encoder_v1"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    raws = load_holdout(a.extra_root, a.val_holdout)
    _n_tap, _fams = M.attach_family_extra(raws, list(a.extra_root))
    print(f"[holdout] 家族 sidecar 挂载 {_n_tap} 个样本 {_fams}")
    fams = sorted({s.family_id for s in raws})
    print(f"holdout samples: {len(raws)} {fams}")
    lad = pf_zero_learning_baseline(raws)
    print(f"L2 theta/v on holdout: {lad['theta']['L2']:.4f} {lad['v']['L2']:.5f}")

    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    groups = [(f, [s for s in raws if s.family_id == f]) for f in fams] + [("ALL", raws)]
    rep = {"n_holdout": len(raws), "families": fams,
           "ladder": {f: lad[f] for f in PF_FIELDS}, "dc_bare": {}, "models": {}}
    for gname, grp in groups:
        dc_, lo_, fb_, sup_ = dc_bare_rows(grp)
        rep["dc_bare"][gname] = {"dc_raw": dc_, "dc_lossy": lo_,
                                 "lossy_fallback_samples": fb_, "n_supervised": sup_}
        print(f"{'DC 裸值':<32} {gname:<15} n={len(grp)} 普通DC {dc_:.4f} 带网损DC {lo_:.4f} "
              f"回退 {fb_}/{sup_}")

    for ckpt in a.ckpt:
        enc, reasoner, cfg, norm, meta = build_from_ckpt(ckpt, a.v1_dir, dev, a.seed)
        label = Path(ckpt).resolve().parent.parent.name.replace("pf_reasoner_5060_", "")
        dn = torch.from_numpy(pf_phys_denorm(norm)).to(dev) if meta["use_phys"] else None
        use_amp = dev.type == "cuda"
        rep["models"][label] = {"ckpt": str(ckpt), "step": meta["step"], "groups": {}}
        for gname, grp in groups:
            gnorm = [copy.deepcopy(s) for s in grp]
            for s in gnorm:
                norm.transform(s)
            mae = M.eval_pu_mae(enc, reasoner, gnorm, cfg, norm, dev, use_amp, torch.float16,
                                bs=a.bs, view=meta["view"],
                                raws=grp if meta["use_raw"] else None, dn=dn,
                                dc=meta["dc_mode"], phys=meta["use_phys"], edge=meta["theta_edge"],
                                vhint=meta["v_hint"], vstat=meta["vstat"], negb=meta["negb"], shz=meta["shz"],
                                edge_base=meta["edge_base"])
            print(f"{label:<32} {gname:<15} n={len(grp)} theta {mae['theta']:.4f} v {mae['v']:.5f} "
                  f"pg {mae['pg']:.3f} qg {mae['qg']:.3f} phys {mae.get('phys_clean', float('nan')):.3f}")
            rec = {k: float(v) for k, v in mae.items()}
            rec["n"] = len(grp)
            if gname == "ALL":
                rec["skill_theta"] = float(lad["theta"]["L2"] / mae["theta"]) if mae["theta"] > 0 else float("nan")
                rec["skill_v"] = float(lad["v"]["L2"] / mae["v"]) if mae["v"] > 0 else float("nan")
            rep["models"][label]["groups"][gname] = rec
        del enc, reasoner
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[holdout] 报告 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
