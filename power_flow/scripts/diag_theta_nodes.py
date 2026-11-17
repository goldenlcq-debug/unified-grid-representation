#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""潮流任务推理器 · 逐节点 θ 误差诊断(README §7-1;只读 val / transfer,不训练、不选模)。

模型装载、数据视图、前向与反归一化全部照抄 scripts/eval_pf_ckpt.py / pf_reasoner.eval_pu_mae,
差别只在:逐节点误差在 sum 之前被截获,并附上
  * hop      —— 离平衡机的图距离(在运支路上 BFS,支路取舍与 pf_physics.dc_theta 同款);
  * theta_u  —— 沿 BFS 树把存档 θ 解缠绕后的"物理相角"(假设任一在运支路两端 |Δθ|<π;
                非树边上若 |Δθ_u|>π 计为自检失败,按样本统计);
  * wrapped  —— 存档 θ 与 theta_u 不同(即存档值被 ±π 截断过)的节点;
  * theta_dc —— DC 潮流 θ(pf_physics.dc_theta)与 valid 位。
输出:总表 / 分 tier / 分箱(hop、|θ_true|、|θ_u|、|θ_dc−θ_u|)的
  raw_mae   = mean|pred − θ_archived|          (= 正式评估口径,须与 eval json 对账)
  wrap_mae  = mean|wrap(pred − θ_archived)|    (周期误差)
  unwrap_mae= mean|pred − θ_u|                 (对物理相角的误差)
  wout_mae  = mean|wrap(pred) − θ_archived|    (若输出层加 wrap 后的正式口径误差)
以及离散度比 std(pred)/std(truth)、相关系数、hop 斜率、DC 裸值四口径、L2 阶梯复算。

用法(grid-foundation/ 下):
  python scripts/diag_theta_nodes.py --root ../gridfm-data \\
      --ckpt runs/pf_baselines/pf_reasoner_v1_group_R1_dc/ckpt/best.pt --device cuda \\
      --ref runs/pf_baselines/pf_reasoner_v1_group_R1_dc/eval_on_current_corpus.json \\
      --out runs/diag/R1_dc_baseline.json [--dump runs/diag/R1_dc_baseline_nodes.npz]
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
from scipy.sparse.csgraph import breadth_first_order, connected_components, shortest_path  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

import pf_reasoner as M  # noqa: E402
from gridfm.data.dataset import load_families  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.reason.pf_physics import dc_theta, dc_theta_lossy, pf_phys_denorm  # noqa: E402
from gridfm.reason.pf_task import PF_FIELDS, PF_SIDE_DIM, PFReasoner, pf_supervised_cols  # noqa: E402
from gridfm.reason.pf_theta_edge import EDGE_SIDE_DIM  # noqa: E402
from gridfm.schema.fields import BRANCH_TABLE, BRANCH_TYPES, NODE_TABLE  # noqa: E402
from gridfm.train.masking import _pf_typed_mask  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402

TWO_PI = 2.0 * np.pi
_SLACK = 2
TH = NODE_TABLE.offset_of("theta")


def wrap(x):
    """把角度折回 (-π, π]。"""
    return (np.asarray(x, dtype=np.float64) + np.pi) % TWO_PI - np.pi


def _use_edges(raw):
    """在运、非并补、|x|>1e-9 的支路(与 dc_theta 同款)。"""
    boff = BRANCH_TABLE.offset_of
    x = raw.branch_feat[:, boff("x")].astype(np.float64)
    mark = raw.branch_feat[:, boff("mark")]
    oh = raw.branch_feat[:, boff("branch_type"):boff("branch_type") + 6]
    bt = np.asarray(BRANCH_TYPES)[np.argmax(oh, axis=1)] if raw.n_branches else np.array([])
    return (mark == 1) & (bt != "shunt_comp") & (np.abs(x) > 1e-9)


def graph_unwrap(raw):
    """沿平衡机出发的 BFS 树解缠绕存档 θ。

    返回 theta_u[N] f64、hop[N] int(不可达 −1)、n_bad_cycle(非树边 |Δθ_u|>π 条数)、n_use_edges。
    """
    n = raw.n_nodes
    noff = NODE_TABLE.offset_of
    theta = raw.node_feat[:, TH].astype(np.float64)
    types = np.argmax(raw.node_feat[:, noff("node_type"):noff("node_type") + 3], axis=1)
    null = raw.node_feat[:, noff("is_null")] > 0.5
    theta_u = theta.copy()
    hop = np.full(n, -1, dtype=np.int64)
    use = _use_edges(raw)
    if not use.any():
        return theta_u, hop, 0, 0
    i, j = raw.edge_index[0][use].astype(np.int64), raw.edge_index[1][use].astype(np.int64)
    A = sp.coo_matrix((np.ones(i.size * 2), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    _nc, comp = connected_components(A, directed=False)
    for c in np.unique(comp):
        nodes = np.where((comp == c) & ~null)[0]
        if nodes.size < 2:
            continue
        slacks = nodes[types[nodes] == _SLACK]
        if slacks.size == 0:
            continue
        s0 = int(slacks[0])
        order, pred = breadth_first_order(A, s0, directed=False, return_predecessors=True)
        dist = shortest_path(A, directed=False, unweighted=True, indices=s0)
        for v in order[1:]:
            p = pred[v]
            theta_u[v] = theta_u[p] + wrap(theta[v] - theta[p])
        reach = np.isfinite(dist)
        hop[reach] = dist[reach].astype(np.int64)
    both = (hop[i] >= 0) & (hop[j] >= 0)
    n_bad = int((np.abs(theta_u[j[both]] - theta_u[i[both]]) > np.pi).sum())
    return theta_u, hop, n_bad, int(use.sum())


def _tier(family_id: str) -> int:
    """规模档:synthNNN_* → NNN;mp_activsgNNk → NNN×1000;mp_activsgNNNN / mp_caseNNN → NNNN;其它 −1。"""
    fid = family_id or ""
    m = re.match(r"synth(\d+)_", fid)
    if m:
        return int(m.group(1))
    m = re.match(r"mp_activsg(\d+)k", fid)
    if m:
        return int(m.group(1)) * 1000
    m = re.match(r"mp_(?:activsg|case)(\d+)", fid)
    return int(m.group(1)) if m else -1


def load_model(ckpt_path, v1_dir, device, seed=0):
    """与 eval_pf_ckpt.py 逐位同源的模型装载。"""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    rc = ck["config"]
    view = ck.get("view", "field")
    encoder = ck.get("encoder", "v1")
    dc = bool(rc.get("dc_theta", False))
    dc_enc = rc.get("dc_encoding", "raw")
    dc_mode = dc_enc if dc else False
    phys_w, phys_def = ck.get("phys_weight", 0.0), ck.get("phys_defeq_weight", 0.0)
    use_phys = phys_w > 0 or phys_def > 0 or bool(rc.get("refine_steps", 0))

    v1 = Path(v1_dir)
    vck = torch.load(v1 / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in vck["config"].items() if k in allowed})
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(seed)
    enc, _ = build_models(cfg)
    if encoder == "v1":
        enc.load_state_dict(vck["model_state"])
    elif encoder == "e2e":
        enc.load_state_dict(ck["encoder_state"])
    enc.to(dev).eval()
    norm = Normalizer.load(v1 / "normalizer.npz")

    dc_dim = 0 if not dc else {"raw": 2, "bounded": 4, "lossy": 3, "lossyg": 3, "lossy2": 2, "lossyv": 3, "lossybv": 3}[dc_enc]
    v_hint = bool(rc.get("v_hint", False))
    side_dim = (PF_SIDE_DIM + dc_dim + (3 if v_hint else 0)) if view == "group" else 0
    theta_edge = bool(rc.get("theta_edge", False))
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
    dn = torch.from_numpy(pf_phys_denorm(norm)).to(dev) if use_phys else None
    _vc = NODE_TABLE.offset_of("v")
    info = dict(step=ck.get("step"), encoder=encoder, view=view, dc=dc, dc_mode=dc_mode,
                use_phys=use_phys, theta_edge=theta_edge, config=rc, v_hint=v_hint,
                vstat=(float(norm.node_mean[_vc]), float(norm.node_std[_vc])) if v_hint else None,
                negb=rc.get("refine_negb", "sign"), shz=bool(rc.get("shunt_zero_harmless", False)),
                edge_base=rc.get("edge_base", "acv"))
    return enc, reasoner, cfg, norm, dev, dn, info


@torch.no_grad()
def collect_nodes(enc, reasoner, samples, raws, cfg, norm, dev, info, split, bs=16, dn=None):
    """逐节点记录(只保留 θ 被监督的节点;前向与 eval_pu_mae 逐位同源)。"""
    cols = pf_supervised_cols()
    use_amp = dev.type == "cuda"
    amp_dtype = torch.float16
    rec = {k: [] for k in ("sample", "tier", "n_nodes", "node", "ntype", "theta_true", "theta_u",
                           "hop", "theta_pred", "theta_dc", "dc_valid", "theta_slack")}
    per_sample = []
    noff = NODE_TABLE.offset_of
    for i in range(0, len(samples), bs):
        rr = raws[i:i + bs]
        datas = [M._pf_data(s, cfg, info["view"], raw=r, dc=info["dc_mode"], phys=info["use_phys"],
                            edge=info["theta_edge"], vhint=info["v_hint"], vstat=info["vstat"], negb=info["negb"],
                            shz=info["shz"], edge_base=info["edge_base"])
                 for s, r in zip(samples[i:i + bs], rr)]
        batch = Batch.from_data_list(datas).to(dev)
        pred, _, delta = M._forward(enc, reasoner, batch, use_amp, amp_dtype, dev, return_edge=True, dn=dn)   # E12:修正步要物理常量
        pred = pred.float().cpu().numpy()
        target = batch.target.float().cpu().numpy()
        lm = batch.loss_mask.cpu().numpy()
        gidx = batch.batch.cpu().numpy()
        pred_full = target.copy()
        for k, f in enumerate(PF_FIELDS):
            pred_full[:, cols[f]] = pred[:, k]
        pred_pu = norm.inverse_transform_node(pred_full)
        target_pu = norm.inverse_transform_node(target)
        if delta is not None:
            ei = batch.edge_index.cpu().numpy()
            d_np = delta.float().cpu().numpy()
            es = batch.edge_side.cpu().numpy()
            eu = batch.edge_use.cpu().numpy()
            ptr = batch.ptr.cpu().numpy()
        for g, raw in enumerate(rr):
            rows = np.where(gidx == g)[0]
            m = lm[rows, TH]
            theta_u, hop, n_bad, n_use = graph_unwrap(raw)
            if delta is not None:
                esel = np.where(gidx[ei[0]] == g)[0]
                s_loc, d_loc = ei[0, esel] - ptr[g], ei[1, esel] - ptr[g]
                use_e = eu[esel]
                avail_e = (~raw.node_missing[:, 3])[s_loc] & (~raw.node_missing[:, 3])[d_loc]
                keep = use_e & avail_e
                if keep.any():
                    EREC = rec.setdefault("_edges", {k: [] for k in (
                        "sample", "tier", "delta", "dtrue", "dth_dc", "dc_ok", "p_dc", "hop_min", "dpred")})
                    EREC["sample"].append(np.full(keep.sum(), i + g))
                    EREC["tier"].append(np.full(keep.sum(), _tier(raw.family_id)))
                    EREC["delta"].append(d_np[esel][keep].astype(np.float64))
                    EREC["dtrue"].append((theta_u[d_loc] - theta_u[s_loc])[keep])
                    EREC["dth_dc"].append(es[esel, 1][keep].astype(np.float64))
                    EREC["dc_ok"].append(es[esel, 2][keep] > 0.5)
                    EREC["p_dc"].append(es[esel, 6][keep].astype(np.float64))
                    EREC["hop_min"].append(np.minimum(hop[s_loc], hop[d_loc])[keep])
                    th_p = pred_pu[rows, TH].astype(np.float64)
                    EREC["dpred"].append(wrap(th_p[d_loc] - th_p[s_loc])[keep])
            # 节点级 DC 列按检查点口径取(lossy 检查点 → 带网损 DC 前两列,与边侧 Δθ_dc 同源)
            _dm = str(info.get("dc_mode"))
            d_ = (dc_theta_lossy(raw, final_check=(_dm == "lossyg"))[:, :2]
                  if _dm.startswith("lossy") else dc_theta(raw))
            types = np.argmax(raw.node_feat[:, noff("node_type"):noff("node_type") + 3], axis=1)
            slack_rows = np.where(types == _SLACK)[0]
            th_slack = float(raw.node_feat[slack_rows[0], TH]) if slack_rows.size else 0.0
            wrapped = np.abs(theta_u - raw.node_feat[:, TH]) > 1e-6
            per_sample.append(dict(split=split, idx=i + g, family=raw.family_id, sample_id=raw.sample_id,
                                   tier=_tier(raw.family_id), n_nodes=raw.n_nodes,
                                   n_masked=int(m.sum()), n_wrapped=int((wrapped & m).sum()),
                                   n_bad_cycle=n_bad, n_use_edges=n_use,
                                   max_abs_theta_u=float(np.abs(theta_u[m]).max()) if m.any() else 0.0,
                                   max_abs_theta=float(np.abs(raw.node_feat[m, TH]).max()) if m.any() else 0.0))
            if not m.any():
                continue
            sel = rows[m]
            loc = np.where(m)[0]
            rec["sample"].append(np.full(loc.size, i + g))
            rec["tier"].append(np.full(loc.size, _tier(raw.family_id)))
            rec["n_nodes"].append(np.full(loc.size, raw.n_nodes))
            rec["node"].append(loc)
            rec["ntype"].append(types[loc])
            rec["theta_true"].append(target_pu[sel, TH].astype(np.float64))
            rec["theta_u"].append(theta_u[loc])
            rec["hop"].append(hop[loc])
            rec["theta_pred"].append(pred_pu[sel, TH].astype(np.float64))
            rec["theta_dc"].append(d_[loc, 0].astype(np.float64))
            rec["dc_valid"].append(d_[loc, 1] > 0.5)
            rec["theta_slack"].append(np.full(loc.size, th_slack))
    edges = rec.pop("_edges", None)
    out = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rec.items()}
    out["wrapped"] = np.abs(out["theta_u"] - out["theta_true"]) > 1e-6
    if edges is not None:
        out["_edges"] = {k: np.concatenate(v) for k, v in edges.items()}
    return out, per_sample


def _edge_stats(E, sel):
    sel = np.asarray(sel, dtype=bool)
    n = int(sel.sum())
    if n == 0:
        return {"n": 0}
    d, t, dc, ok, dp = E["delta"][sel], E["dtrue"][sel], E["dth_dc"][sel], E["dc_ok"][sel], E["dpred"][sel]
    e = d - t
    sg = np.sign(t)
    o = {"n": n,
         "delta_mae": float(np.abs(e).mean()),
         # 正反向边成对出现,原始符号偏差恒 0;改报"沿真实 Δθ 方向的偏差":<0 = 收缩(估小),>0 = 放大
         "delta_bias": float((e * sg).mean()),
         "dc_bias_aligned": float(((dc - t) * sg)[ok].mean()) if ok.any() else float("nan"),
         "delta_p90": float(np.percentile(np.abs(e), 90)),
         "dc_mae": float(np.abs(dc[ok] - t[ok]).mean()) if ok.any() else float("nan"),
         "dc_bias": float(((dc - t) * sg)[ok].mean()) if ok.any() else float("nan"),
         "corr_err_delta_vs_dc": float(np.corrcoef(e[ok], (dc - t)[ok])[0, 1]) if ok.sum() > 2 else float("nan"),
         "mean_abs_correction": float(np.abs(d - dc)[ok].mean()) if ok.any() else float("nan"),
         "dtrue_mean_abs": float(np.abs(t).mean()),
         "ls_residual_mae": float(np.abs(dp - d).mean()),          # 势重建后边差与 δ 的不一致(回路不自洽)
         "recon_edge_mae": float(np.abs(dp - t).mean()),           # 重建后边差对真实 Δθ 的误差
         }
    # δ 误差对真实 Δθ 的线性回归(斜率 <0:收缩;>0:放大)
    if t.std() > 0:
        o["slope_err_vs_dtrue"] = float(np.polyfit(t, e, 1)[0])
    return o


def summarize_edges(E):
    out = {"all": _edge_stats(E, np.ones(E["delta"].size, dtype=bool))}
    out["by_tier"] = {int(t): _edge_stats(E, E["tier"] == t) for t in np.unique(E["tier"])}
    adc = np.abs(E["dth_dc"])
    out["by_abs_dth_dc"] = {f"[{lo:g},{hi:g})": _edge_stats(E, E["dc_ok"] & (adc >= lo) & (adc < hi))
                            for lo, hi in zip([0, 0.02, 0.05, 0.1, 0.2, 0.5], [0.02, 0.05, 0.1, 0.2, 0.5, 10])}
    ap = np.abs(E["p_dc"])
    out["by_abs_p_dc"] = {f"[{lo:g},{hi:g})": _edge_stats(E, E["dc_ok"] & (ap >= lo) & (ap < hi))
                          for lo, hi in zip([0, 0.25, 0.5, 1, 2, 4], [0.25, 0.5, 1, 2, 4, 100])}
    hm = E["hop_min"]
    out["by_hop_min"] = {f"[{lo:g},{hi:g})": _edge_stats(E, (hm >= lo) & (hm < hi))
                         for lo, hi in zip([0, 3, 6, 11, 21, 41], [3, 6, 11, 21, 41, 10000])}
    return out


def print_edges(split, SE):
    K = [("n", 8, 0), ("dtrue_mean_abs", 8, 4), ("delta_mae", 9, 4), ("delta_bias", 10, 4), ("delta_p90", 9, 4),
         ("dc_mae", 8, 4), ("dc_bias", 9, 4), ("mean_abs_correction", 9, 4), ("corr_err_delta_vs_dc", 8, 3),
         ("slope_err_vs_dtrue", 8, 3), ("ls_residual_mae", 9, 4), ("recon_edge_mae", 9, 4)]
    hdr = "".join(f"{k:>{w}}" for k, w, _ in K)
    print(f"\n----- {split} · 逐边(δ 对真实 Δθ_u;dc = Δθ_dc 对真实;correction = |δ − Δθ_dc|) -----")
    print(f"{'':<14}{hdr}")
    print(f"{'全部':<14}{_fmt(SE['all'], K)}")
    for t, s in sorted(SE["by_tier"].items()):
        print(f"{'tier '+str(t):<14}{_fmt(s, K)}")
    for name in ("by_abs_dth_dc", "by_abs_p_dc", "by_hop_min"):
        print(f"  -- 分箱 {name}:")
        for b, s in SE[name].items():
            if s.get("n", 0):
                print(f"     {b:<11}{_fmt(s, K)}")


def _stats(R, sel):
    """一组节点上的全部口径。"""
    sel = np.asarray(sel, dtype=bool)
    n = int(sel.sum())
    if n == 0:
        return {"n": 0}
    p, t, u = R["theta_pred"][sel], R["theta_true"][sel], R["theta_u"][sel]
    dc, okdc = R["theta_dc"][sel], R["dc_valid"][sel]
    sl = R["theta_slack"][sel]
    w = R["wrapped"][sel]
    e_raw = np.abs(p - t)
    o = {
        "n": n,
        "frac_wrapped": float(w.mean()),
        "raw_mae": float(e_raw.mean()),
        "wrap_mae": float(np.abs(wrap(p - t)).mean()),
        "unwrap_mae": float(np.abs(p - u).mean()),
        "wout_mae": float(np.abs(wrap(p) - t).mean()),
        "raw_mae_on_wrapped": float(e_raw[w].mean()) if w.any() else float("nan"),
        "raw_mae_on_unwrapped": float(e_raw[~w].mean()) if (~w).any() else float("nan"),
        "wrapped_share_of_raw_mae": float(e_raw[w].sum() / e_raw.sum()) if e_raw.sum() > 0 else 0.0,
        "bias_vs_u": float((p - u).mean()),
        "std_true": float(t.std()), "std_u": float(u.std()), "std_pred": float(p.std()),
        "disp_ratio_vs_true": float(p.std() / t.std()) if t.std() > 0 else float("nan"),
        "disp_ratio_vs_u": float(p.std() / u.std()) if u.std() > 0 else float("nan"),
        "corr_vs_true": float(np.corrcoef(p, t)[0, 1]) if n > 2 and t.std() > 0 and p.std() > 0 else float("nan"),
        "corr_vs_u": float(np.corrcoef(p, u)[0, 1]) if n > 2 and u.std() > 0 and p.std() > 0 else float("nan"),
        # L2 阶梯复算(θ 在 PQ+PV 全遮 → 调和插值 = 平衡机给定角常数)
        "L2_raw_mae": float(np.abs(sl - t).mean()),
        "L2_unwrap_mae": float(np.abs(sl - u).mean()),
        "L2_wrap_mae": float(np.abs(wrap(sl - t)).mean()),
        # DC 裸值四口径(valid 位)
        "dc_n": int(okdc.sum()),
        "dc_raw_mae": float(np.abs(dc[okdc] - t[okdc]).mean()) if okdc.any() else float("nan"),
        "dc_wrap_mae": float(np.abs(wrap(dc[okdc] - t[okdc])).mean()) if okdc.any() else float("nan"),
        "dc_unwrap_mae": float(np.abs(dc[okdc] - u[okdc]).mean()) if okdc.any() else float("nan"),
        "dc_wout_mae": float(np.abs(wrap(dc[okdc]) - t[okdc]).mean()) if okdc.any() else float("nan"),
        "dc_corr_vs_u": float(np.corrcoef(dc[okdc], u[okdc])[0, 1]) if okdc.sum() > 2 and u[okdc].std() > 0 else float("nan"),
    }
    h = R["hop"][sel]
    hv = h >= 0
    if hv.sum() > 10 and h[hv].std() > 0:
        o["hop_slope_unwrap_err"] = float(np.polyfit(h[hv], np.abs(p - u)[hv], 1)[0])
        o["hop_slope_raw_err"] = float(np.polyfit(h[hv], e_raw[hv], 1)[0])
        o["hop_max"] = int(h[hv].max())
        o["hop_mean"] = float(h[hv].mean())
    return o


def _binned(R, sel, key_vals, edges, label):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = sel & (key_vals >= lo) & (key_vals < hi)
        s = _stats(R, b)
        s["bin"] = f"[{lo:g},{hi:g})"
        rows.append(s)
    return {"key": label, "rows": rows}


def summarize(R, split):
    out = {}
    allsel = np.ones(R["theta_true"].size, dtype=bool)
    out["all"] = _stats(R, allsel)
    out["by_tier"] = {int(t): _stats(R, R["tier"] == t) for t in np.unique(R["tier"])}
    out["by_type"] = {{0: "PQ", 1: "PV", 2: "Slack"}[int(t)]: _stats(R, R["ntype"] == t)
                      for t in np.unique(R["ntype"])}
    hop_edges = np.array([0, 1, 3, 6, 11, 21, 41, 10_000])
    out["bins"] = {
        "hop": _binned(R, allsel & (R["hop"] >= 0), R["hop"], hop_edges, "hop"),
        "abs_theta_true": _binned(R, allsel, np.abs(R["theta_true"]),
                                  np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.9, 3.2]), "|θ_archived|"),
        "abs_theta_u": _binned(R, allsel, np.abs(R["theta_u"]),
                               np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 100.0]), "|θ_u|"),
        "abs_dc_err_u": _binned(R, allsel & R["dc_valid"], np.abs(R["theta_dc"] - R["theta_u"]),
                                np.array([0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 1000.0]), "|θ_dc−θ_u|"),
    }
    for t in np.unique(R["tier"]):
        ts = R["tier"] == t
        out["by_tier"][int(t)]["bins_hop"] = _binned(R, ts & (R["hop"] >= 0), R["hop"], hop_edges, "hop")
    # 预注册 §9:正式口径取"全部方式";含缠绕节点的方式(70k 的 3 个)另并报一行"剔除后"的数字。
    # 纯统计口径添加(只是换一个节点子集调 _stats),不改任何既有指标的算法。
    bad = np.unique(R["sample"][R["wrapped"]])
    clean = ~np.isin(R["sample"], bad)
    out["wrapped_sample_ids"] = [int(x) for x in bad]
    out["by_tier_clean_samples"] = {int(t): _stats(R, (R["tier"] == t) & clean)
                                    for t in np.unique(R["tier"])}
    return out


def _fmt(s, keys):
    return "".join(f"{s.get(k, float('nan')):>{w}.{p}f}" if isinstance(s.get(k, None), float)
                   else f"{str(s.get(k, '')):>{w}}" for k, w, p in keys)


def print_report(split, S, per_sample):
    K = [("n", 8, 0), ("frac_wrapped", 8, 3), ("raw_mae", 9, 4), ("wrap_mae", 9, 4), ("unwrap_mae", 10, 4),
         ("wout_mae", 9, 4), ("disp_ratio_vs_u", 8, 3), ("corr_vs_u", 8, 3), ("L2_raw_mae", 9, 4),
         ("L2_unwrap_mae", 10, 4), ("dc_raw_mae", 9, 4), ("dc_unwrap_mae", 10, 4)]
    hdr = "".join(f"{k:>{w}}" for k, w, _ in K)
    print(f"\n===== {split} =====")
    print(f"{'':<12}{hdr}")
    print(f"{'全部':<12}{_fmt(S['all'], K)}")
    for t, s in sorted(S["by_tier"].items()):
        print(f"{'tier '+str(t):<12}{_fmt(s, K)}")
    for t, s in sorted(S.get("by_tier_clean_samples", {}).items()):
        if s.get("n", 0) and s["n"] != S["by_tier"][t]["n"]:      # 只在真有缠绕方式被剔除时并报
            print(f"{'  ↳剔缠绕':<12}{_fmt(s, K)}")
    for t, s in S["by_type"].items():
        print(f"{t:<12}{_fmt(s, K)}")
    a = S["all"]
    print(f"  缠绕节点占比 {a['frac_wrapped']:.3f};缠绕节点 raw_mae {a['raw_mae_on_wrapped']:.4f} vs 非缠绕 "
          f"{a['raw_mae_on_unwrapped']:.4f};缠绕节点贡献了 raw_mae 的 {a['wrapped_share_of_raw_mae']:.1%}")
    print(f"  离散度比 std(pred)/std(θ_u) {a['disp_ratio_vs_u']:.3f}(对存档 {a['disp_ratio_vs_true']:.3f});"
          f"corr(pred,θ_u) {a['corr_vs_u']:.3f}(对存档 {a['corr_vs_true']:.3f});偏差 mean(pred−θ_u) {a['bias_vs_u']:+.4f}")
    if "hop_slope_unwrap_err" in a:
        print(f"  hop 斜率:unwrap 误差 {a['hop_slope_unwrap_err']:+.4f} rad/跳,raw 误差 {a['hop_slope_raw_err']:+.4f} rad/跳;"
              f"hop 均值 {a['hop_mean']:.1f} / 最大 {a['hop_max']}")
    print(f"  DC 裸值:raw {a['dc_raw_mae']:.4f} / wrap {a['dc_wrap_mae']:.4f} / unwrap {a['dc_unwrap_mae']:.4f} / "
          f"wrap(dc) {a['dc_wout_mae']:.4f};corr(dc,θ_u) {a['dc_corr_vs_u']:.3f}")
    for name, B in S["bins"].items():
        print(f"  -- 分箱 {B['key']}:")
        print(f"     {'bin':<12}{hdr}")
        for r in B["rows"]:
            if r.get("n", 0):
                print(f"     {r['bin']:<12}{_fmt(r, K)}")
    bad = [p for p in per_sample if p["n_bad_cycle"] > 0]
    nw = sum(1 for p in per_sample if p["n_wrapped"] > 0)
    print(f"  样本:{len(per_sample)} 个,含缠绕节点的 {nw} 个;解缠绕自检失败(非树边 |Δθ_u|>π)的样本 {len(bad)} 个"
          + (f",例如 {[(p['family'], p['sample_id'], p['n_bad_cycle']) for p in bad[:5]]}" if bad else ""))
    mx = max((p["max_abs_theta_u"] for p in per_sample), default=0.0)
    print(f"  max|θ_u| = {mx:.3f} rad({np.degrees(mx):.0f}°),max|θ_archived| = "
          f"{max((p['max_abs_theta'] for p in per_sample), default=0.0):.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--v1-dir", default=str(HERE.parent / "runs" / "encoder_v1"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", default="val,transfer")
    ap.add_argument("--bs", type=int, default=16, help="评估批大小(万节点图请用 1)")
    ap.add_argument("--families", default=None, help="只评估这些 family_id(逗号分隔)")
    ap.add_argument("--ref", default=None, help="eval_on_current_corpus.json,用于对账")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", default=None, help="逐节点 npz 落盘")
    a = ap.parse_args()

    enc, reasoner, cfg, norm, dev, dn, info = load_model(a.ckpt, a.v1_dir, a.device, a.seed)
    print(f"[diag] ckpt {a.ckpt} step={info['step']} encoder={info['encoder']} view={info['view']} "
          f"dc={info['dc_mode']} phys={info['use_phys']} device={dev}"
          + ("  [节点级 DC 列 = 带网损 DC;--ref 的 DC 对账跳过]" if info.get("dc_mode") == "lossy" else ""))
    ref = json.loads(Path(a.ref).read_text(encoding="utf-8")) if a.ref else None

    root = Path(a.root)
    report = {"ckpt": str(a.ckpt), "step": info["step"], "config": info["config"], "splits": {}, "checks": {}}
    dumps = {}
    for split in a.splits.split(","):
        if a.families:                                   # 加载前按家族目录名过滤(省内存)
            from gridfm.data.dataset import family_files
            keep = set(a.families.split(","))
            paths = [p for p in family_files(root, split) if Path(p).parent.name in keep]
            from gridfm.schema.sample import GridSample as _GS
            raws = [_GS.load_npz(p) for p in paths]
        else:
            raws = load_families(root, split)
        M.attach_family_extra(raws, [root])
        samples = [copy.deepcopy(s) for s in raws]
        for s in samples:
            norm.transform(s)
        R, per_sample = collect_nodes(enc, reasoner, samples, raws, cfg, norm, dev, info, split, bs=a.bs, dn=dn)
        E = R.pop("_edges", None)
        S = summarize(R, split)
        print_report(split, S, per_sample)
        report["splits"][split] = {"summary": S, "per_sample": per_sample}
        if E is not None:
            SE = summarize_edges(E)
            print_edges(split, SE)
            report["splits"][split]["edges"] = SE
            if a.dump:
                for k, v in E.items():
                    dumps[f"{split}/edge_{k}"] = v
        if ref:
            key = "in" if split == "val" else "out"
            ref_mae = ref["mae_%s_pu" % key]["theta"]
            ref_l2 = ref["ladder_task_%s" % key]["theta"]["L2"]
            ref_dc = (ref.get("dc_theta_raw_mae") or {}).get(key)
            if info.get("dc_mode") == "lossy":
                ref_dc = (ref.get("dc_theta_raw_mae") or {}).get(key + "_lossy")   # 同口径对账
            chk = {"raw_mae": S["all"]["raw_mae"], "ref_mae": ref_mae, "diff": S["all"]["raw_mae"] - ref_mae,
                   "L2": S["all"]["L2_raw_mae"], "ref_L2": ref_l2, "diff_L2": S["all"]["L2_raw_mae"] - ref_l2,
                   "dc": S["all"]["dc_raw_mae"], "ref_dc": ref_dc,
                   "diff_dc": (S["all"]["dc_raw_mae"] - ref_dc) if ref_dc is not None else None}
            report["checks"][split] = chk
            print(f"  [对账] raw_mae {chk['raw_mae']:.4f} vs ref {ref_mae:.4f}(Δ {chk['diff']:+.5f});"
                  f"L2 {chk['L2']:.4f} vs {ref_l2:.4f}(Δ {chk['diff_L2']:+.5f})"
                  + (f";DC {chk['dc']:.4f} vs {ref_dc:.4f}(Δ {chk['diff_dc']:+.5f})" if ref_dc is not None else ""))
        if a.dump:
            for k, v in R.items():
                dumps[f"{split}/{k}"] = v
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        print(f"[diag] 报告 {a.out}")
    if a.dump:
        Path(a.dump).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.dump, **dumps)
        print(f"[diag] 逐节点 {a.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
