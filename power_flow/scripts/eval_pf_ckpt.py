#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""潮流任务推理器 · 仅评估(不训练、不写检查点)。

载入一个 pf_reasoner 检查点,在 val(内,同分布留出家族)与 transfer(外,400~1000 节点
冻结大网)上复算:逐字段标幺 MAE、任务掩码零学习阶梯(L0/L1/L2)、技巧分
(= 阶梯 L2 / 模型 MAE;>1 才算比"同信息零学习"强)、DC-θ 裸值基线行。
评估口径与 scripts/pf_reasoner.py 终评逐位同源(直接复用其函数),结果可与历史
pf_report.json 直接对照。

用法(在 grid-foundation/ 目录下):
  python scripts/eval_pf_ckpt.py --root ../gridfm-data \\
      --ckpt runs/pf_baselines/pf_reasoner_v1_group_R1_dc/ckpt/best.pt --device cuda
  可选:--out report.json  把结果落盘;--seed N  只对 --encoder random 臂有意义
  (随机初始化编码器须用与训练时相同的 seed 才能复现)。

θ 的单位:弧度(rad)。θ MAE 1.0 ≈ 57°。
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

import pf_reasoner as M  # noqa: E402  (复用 eval_pu_mae / _pf_data 等,口径同源)
from gridfm.data.dataset import load_families  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.reason.pf_physics import dc_theta, dc_theta_lossy, pf_phys_denorm  # noqa: E402
from gridfm.reason.pf_task import (  # noqa: E402
    PF_FIELDS, PF_SIDE_DIM, PFReasoner, pf_zero_learning_baseline,
)
from gridfm.reason.pf_theta_edge import EDGE_SIDE_DIM  # noqa: E402
from gridfm.schema.fields import NODE_TABLE  # noqa: E402
from gridfm.train.masking import _pf_typed_mask  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="gridfm-data 目录(含 families/)")
    ap.add_argument("--ckpt", required=True, help="pf_reasoner 检查点 best.pt")
    ap.add_argument("--v1-dir", default=str(HERE.parent / "runs" / "encoder_v1"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bs", type=int, default=16, help="评估批大小(万节点图请用 1)")
    ap.add_argument("--extra-root", action="append", default=[],
                    help="与训练同款:额外语料根,其 families/train 中命中 --val-holdout 清单的样本并入 val")
    ap.add_argument("--val-holdout", action="append", default=[], help="库内留出清单 json(可重复)")
    ap.add_argument("--families", default=None,
                    help="只评估这些 family_id(逗号分隔;加载前按目录名过滤,省内存)")
    ap.add_argument("--splits", default="val,transfer",
                    help="要评估哪些 split(逗号分隔,默认 val,transfer;单跑一档用于分家族对账)")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    rc = ck["config"]
    view = ck.get("view", "field")
    encoder = ck.get("encoder", "v1")
    dc = bool(rc.get("dc_theta", False))
    dc_enc = rc.get("dc_encoding", "raw")
    dc_mode = dc_enc if dc else False
    phys_w, phys_def = ck.get("phys_weight", 0.0), ck.get("phys_defeq_weight", 0.0)
    use_phys = phys_w > 0 or phys_def > 0 or bool(rc.get("refine_steps", 0))
    theta_edge = bool(rc.get("theta_edge", False))
    theta_wrap = bool(rc.get("theta_wrap", False))
    use_raw = use_phys or dc or theta_edge or bool(rc.get("v_hint", False))
    print(f"[eval] 检查点 {a.ckpt}\n       step={ck.get('step')} encoder={encoder} view={view} "
          f"dc_theta={dc}({dc_enc}) phys_w={phys_w:g} theta_head={rc.get('theta_head', False)} "
          f"gnn_layers={rc.get('gnn_layers', 0)} queries={rc['queries']} layers={rc['layers']} "
          f"theta_edge={theta_edge} theta_wrap={theta_wrap}")

    # ---- 编码器(三臂口径与训练脚本一致)----
    v1 = Path(a.v1_dir)
    vck = torch.load(v1 / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in vck["config"].items() if k in allowed})
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(a.seed)
    enc, _ = build_models(cfg)
    if encoder == "v1":
        enc.load_state_dict(vck["model_state"])
    elif encoder == "e2e":
        enc.load_state_dict(ck["encoder_state"])
    elif encoder == "random":
        print(f"[eval] random 臂:按 seed={a.seed} 重建随机初始化编码器(须与训练时一致)")
    enc.to(dev).eval()
    norm = Normalizer.load(v1 / "normalizer.npz")

    # ---- 推理器 ----
    dc_dim = 0 if not dc else {"raw": 2, "bounded": 4, "lossy": 3, "lossyg": 3, "lossy2": 2, "lossyv": 3, "lossybv": 3}[dc_enc]
    v_hint = bool(rc.get("v_hint", False))
    side_dim = (PF_SIDE_DIM + dc_dim + (3 if v_hint else 0)) if view == "group" else 0
    reasoner = PFReasoner(d=rc.get("d", cfg.d), n_queries=rc["queries"], n_layers=rc["layers"],
                          side_dim=side_dim, gnn_layers=rc.get("gnn_layers", 0),
                          theta_head=rc.get("theta_head", False),
                          edge_side_dim=EDGE_SIDE_DIM if theta_edge else 0,
                          theta_edge=theta_edge, theta_wrap=theta_wrap,
                          edge_dc_residual=rc.get("edge_dc_residual", True),
                          edge_head_local=rc.get("edge_head_local", False),
                          refine_steps=rc.get("refine_steps", 0), refine_qv=bool(rc.get("refine_qv", False)),
                          refine_ls=bool(rc.get("refine_ls", False)),
                          v_hint=bool(rc.get("v_hint", False))).to(dev)
    reasoner.load_state_dict(ck["reasoner_state"])
    reasoner.eval()

    # ---- 数据(只读留出集;train 不加载)----
    root = Path(a.root)
    fams = set(a.families.split(",")) if a.families else None

    def _load(split):
        """--families 在加载前按家族目录名过滤(与 diag_theta_nodes.py 同款,万节点档省内存)。"""
        if split not in a.splits.split(","):
            return []
        if fams is None:
            return load_families(root, split)
        from gridfm.data.dataset import family_files
        from gridfm.schema.sample import GridSample
        return [GridSample.load_npz(q) for q in family_files(root, split)
                if Path(q).parent.name in fams]

    val_raw = _load("val")
    xfer_raw = _load("transfer")
    if a.extra_root:
        holdout_ids = set()
        for hp in a.val_holdout:
            holdout_ids |= set(json.loads(Path(hp).read_text(encoding="utf-8"))["val_holdout_sample_ids"])
        extra_raw = []
        for er in a.extra_root:
            extra_raw += load_families(Path(er), "train")
        _keep, extra_val = M.split_holdout(extra_raw, holdout_ids)
        missing = holdout_ids - {s.sample_id for s in extra_val}
        if missing:
            raise SystemExit(f"[eval] 留出清单有 {len(missing)} 个 id 未命中:{sorted(missing)[:5]}")
        val_raw += extra_val
        print(f"[eval] 额外语料留出并入 val:+{len(extra_val)}(与训练口径同源)")

    _n_tap, _fams = M.attach_family_extra(val_raw + xfer_raw, [a.root] + list(a.extra_root))
    print(f"[eval] 家族 sidecar 挂载 {_n_tap} 个样本 {_fams}(物理失配口径:j 侧变比 + 并补 −1/x,2026-08-22)")
    _vc = NODE_TABLE.offset_of("v")
    vstat = (float(norm.node_mean[_vc]), float(norm.node_std[_vc])) if v_hint else None

    def prep(ss):
        o = [copy.deepcopy(s) for s in ss]
        for s in o:
            norm.transform(s)
        return o
    val, xfer = prep(val_raw), prep(xfer_raw)
    print(f"[eval] val {len(val)} / transfer {len(xfer)}  device={dev}")

    dn = torch.from_numpy(pf_phys_denorm(norm)).to(dev) if use_phys else None
    use_amp = dev.type == "cuda"
    amp_dtype = torch.float16
    mae_in = M.eval_pu_mae(enc, reasoner, val, cfg, norm, dev, use_amp, amp_dtype, bs=a.bs, view=view,
                           raws=val_raw if use_raw else None, dn=dn, dc=dc_mode, phys=use_phys,
                           edge=theta_edge, vhint=v_hint, vstat=vstat, negb=rc.get("refine_negb", "sign"),
                           shz=bool(rc.get("shunt_zero_harmless", False)),
                           edge_base=rc.get("edge_base", "acv"))
    mae_out = M.eval_pu_mae(enc, reasoner, xfer, cfg, norm, dev, use_amp, amp_dtype, bs=a.bs, view=view,
                            raws=xfer_raw if use_raw else None, dn=dn, dc=dc_mode, phys=use_phys,
                            edge=theta_edge, vhint=v_hint, vstat=vstat, negb=rc.get("refine_negb", "sign"),
                           shz=bool(rc.get("shunt_zero_harmless", False)),
                           edge_base=rc.get("edge_base", "acv"))
    lad_in = pf_zero_learning_baseline(val_raw)
    lad_out = pf_zero_learning_baseline(xfer_raw)

    dc_row = None
    if dc:
        th_c = NODE_TABLE.offset_of("theta")
        dc_row = {}
        for name, rr_ in (("in", val_raw), ("out", xfer_raw)):
            errs = []
            errs_l, n_fb, n_sup = [], 0, 0
            for s_ in rr_:
                _mf, _tg, lmr = _pf_typed_mask(s_.node_feat, s_.node_missing)
                d_ = dc_theta(s_)
                m_ = lmr[:, th_c] & (d_[:, 1] > 0.5)
                if m_.any():
                    errs.append(np.abs(d_[m_, 0] - s_.node_feat[m_, th_c]))
                if dc_enc.startswith("lossy"):
                    dl_ = dc_theta_lossy(s_, final_check=(dc_enc == "lossyg"))
                    ml_ = lmr[:, th_c] & (dl_[:, 1] > 0.5)
                    if ml_.any():
                        errs_l.append(np.abs(dl_[ml_, 0] - s_.node_feat[ml_, th_c]))
                        n_sup += 1
                        n_fb += int(dl_[ml_, 2].max() > 0.5)
            dc_row[name] = float(np.concatenate(errs).mean()) if errs else float("nan")
            if dc_enc.startswith("lossy"):
                dc_row[name + "_lossy"] = float(np.concatenate(errs_l).mean()) if errs_l else float("nan")
                dc_row[name + "_lossy_fallback_samples"] = n_fb
                dc_row[name + "_lossy_supervised_samples"] = n_sup

    print(f"\n{'':<12}{'θ内':>9}{'θ外':>9}{'v内':>10}{'v外':>10}{'pg内':>10}{'pg外':>10}{'qg内':>10}{'qg外':>10}")
    for lvl in ("L0", "L1", "L2"):
        print(f"{'零学习 '+lvl:<10}{lad_in['theta'][lvl]:>9.4f}{lad_out['theta'][lvl]:>9.4f}"
              f"{lad_in['v'][lvl]:>10.5f}{lad_out['v'][lvl]:>10.5f}"
              f"{lad_in['pg'][lvl]:>10.5f}{lad_out['pg'][lvl]:>10.5f}"
              f"{lad_in['qg'][lvl]:>10.5f}{lad_out['qg'][lvl]:>10.5f}")
    if dc_row:
        print(f"{'DC-θ 裸值':<9}{dc_row['in']:>9.4f}{dc_row['out']:>9.4f}")
        if dc_enc.startswith("lossy"):
            print(f"{'带网损DC':<9}{dc_row['in_lossy']:>9.4f}{dc_row['out_lossy']:>9.4f}"
                  f"   回退样本 内 {dc_row['in_lossy_fallback_samples']}/{dc_row['in_lossy_supervised_samples']}"
                  f" / 外 {dc_row['out_lossy_fallback_samples']}/{dc_row['out_lossy_supervised_samples']}")
    print(f"{'模型 MAE':<10}{mae_in['theta']:>9.4f}{mae_out['theta']:>9.4f}"
          f"{mae_in['v']:>10.5f}{mae_out['v']:>10.5f}"
          f"{mae_in['pg']:>10.5f}{mae_out['pg']:>10.5f}"
          f"{mae_in['qg']:>10.5f}{mae_out['qg']:>10.5f}")
    skills = {}
    for f in PF_FIELDS:
        si = lad_in[f]["L2"] / mae_in[f] if mae_in[f] > 0 else float("nan")
        so = lad_out[f]["L2"] / mae_out[f] if mae_out[f] > 0 else float("nan")
        skills[f] = {"skill_in": si, "skill_out": so,
                     "survival": so / si if (np.isfinite(si) and si > 0) else float("nan")}
        print(f"{f:>6}: 技巧分 内 {si:.2f} / 外 {so:.2f} / 存活 {skills[f]['survival']:.2f}")
    print(f"\nθ 换算:内 {np.degrees(mae_in['theta']):.1f}° / 外 {np.degrees(mae_out['theta']):.1f}°"
          f"(零学习 L2:内 {np.degrees(lad_in['theta']['L2']):.1f}° / 外 {np.degrees(lad_out['theta']['L2']):.1f}°)")
    if use_phys:
        print(f"物理失配 |ΔPQ| 干净方程 内 {mae_in['phys_clean']:.4f} / 外 {mae_out['phys_clean']:.4f}")

    rep = {"ckpt": str(a.ckpt), "step": ck.get("step"), "encoder": encoder, "view": view,
           "dc_theta": dc, "dc_encoding": dc_enc, "dc_theta_raw_mae": dc_row,
           "theta_edge": theta_edge, "theta_wrap": theta_wrap, "v_hint": v_hint,
           "phys_tap_families": _fams,
           "phys_weight": phys_w, "phys_defeq_weight": phys_def, "reasoner_config": rc,
           "n_val": len(val), "n_transfer": len(xfer),
           "mae_in_pu": mae_in, "mae_out_pu": mae_out,
           "ladder_task_in": lad_in, "ladder_task_out": lad_out, "skills": skills}
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[eval] 报告 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
