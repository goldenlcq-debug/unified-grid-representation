#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按**网架(family)× 训练状态**出总表:电压 v 与功角 θ 的平均偏差、p99、最大偏差。

与既有脚本的区别:`eval_pf_ckpt.py` / `eval_holdout_5060.py` 只报平均 MAE(这是判据口径),
本脚本额外报 **p99 与最大偏差**,并把每个网架的"训练见过 / 同拓扑留出未见 / 跨拓扑冻结外集"
标注清楚——用于横向看效果,不改任何判据。

口径:
- 误差只在 **D15 类型化未知量**位置统计(θ:PQ+PV;v:PQ),与正式 MAE 同源(复用 `_pf_typed_mask`);
- 反归一化后按物理量纲(θ 弧度 / 度,v 标幺)统计;
- 并报"带网损 DC 裸值"作为**零训练**参照(θ);
- 最大偏差是**逐节点**的最坏值(不是逐样本平均),它对单点异常敏感,故同时给 p99。

用法(grid-foundation/ 下):
  python scripts/eval_by_family_5060.py --ckpt runs/<run>/ckpt/best.pt \
     --set "合成val:../gridfm-data:val" --set "合成外集:../gridfm-data:transfer" \
     --set "大网train:$D/gridfm-data-large:train" --set "大网外集:$D/gridfm-data-large:transfer" \
     --val-holdout "$D/val留出清单-mp_activsg2000.json" --bs 8 --device cuda --out runs/diag_5060/byfam.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import fields as dc_fields
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

import pf_reasoner as M  # noqa: E402
from gridfm.data.dataset import family_files  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.reason.pf_physics import dc_theta, dc_theta_lossy, pf_phys_denorm  # noqa: E402
from gridfm.reason.pf_task import (  # noqa: E402
    PF_SIDE_DIM, PFReasoner, pf_supervised_cols,
)
from gridfm.reason.pf_theta_edge import EDGE_SIDE_DIM  # noqa: E402
from gridfm.schema.fields import NODE_TABLE  # noqa: E402
from gridfm.schema.sample import GridSample  # noqa: E402
from gridfm.train.masking import _pf_typed_mask  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402

TH, VC = NODE_TABLE.offset_of("theta"), NODE_TABLE.offset_of("v")


def build_from_ckpt(ckpt_path, v1_dir, dev, seed=0):
    """与 eval_pf_ckpt.py 逐位同源的装载。"""
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
    reasoner = PFReasoner(d=rc.get("d", cfg.d), n_queries=rc["queries"], n_layers=rc["layers"],
                          side_dim=(PF_SIDE_DIM + dc_dim + (3 if v_hint else 0)) if view == "group" else 0,
                          gnn_layers=rc.get("gnn_layers", 0), theta_head=rc.get("theta_head", False),
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
    meta = dict(step=ck.get("step"), view=view, dc_mode=(dc_enc if dc else False),
                use_phys=use_phys, theta_edge=theta_edge, v_hint=v_hint,
                vstat=(float(norm.node_mean[_vc]), float(norm.node_std[_vc])) if v_hint else None,
                use_raw=(use_phys or dc or theta_edge or v_hint), negb=rc.get("refine_negb", "sign"),
                shz=bool(rc.get("shunt_zero_harmless", False)),
                edge_base=rc.get("edge_base", "acv"))
    return enc, reasoner, cfg, norm, meta


@torch.no_grad()
def errors(enc, reasoner, raws, norm, cfg, dev, meta, dn, bs):
    """逐节点 |误差|(物理量纲):θ rad、v pu。返回 (err_theta[], err_v[])。"""
    cols = pf_supervised_cols()
    use_amp = dev.type == "cuda"
    et, ev, etw = [], [], []
    for i in range(0, len(raws), bs):
        rr = raws[i:i + bs]
        ss = []
        for q in rr:
            s = copy.deepcopy(q)
            norm.transform(s)
            ss.append(s)
        datas = [M._pf_data(s, cfg, meta["view"], raw=q, dc=meta["dc_mode"],
                            phys=meta["use_phys"], edge=meta["theta_edge"],
                            vhint=meta["v_hint"], vstat=meta["vstat"], negb=meta["negb"], shz=meta["shz"],
                            edge_base=meta["edge_base"])
                 for s, q in zip(ss, rr)]
        batch = Batch.from_data_list(datas).to(dev)
        pred, _ = M._forward(enc, reasoner, batch, use_amp, torch.float16, dev, dn=dn)
        pred = pred.float().cpu().numpy()
        target = batch.target.float().cpu().numpy()
        lm = batch.loss_mask.cpu().numpy()
        full = target.copy()
        for k, f in enumerate(("pg", "qg", "v", "theta")):
            full[:, cols[f]] = pred[:, k]
        pu_p, pu_t = norm.inverse_transform_node(full), norm.inverse_transform_node(target)
        for c, acc in ((TH, et), (VC, ev)):
            m = lm[:, c]
            if m.any():
                acc.append(np.abs(pu_p[m, c] - pu_t[m, c]).astype(np.float64))
        m = lm[:, TH]
        if m.any():
            # 折回口径:存档 θ 是 PSASP 的 ±180° 截断值,合成 700/1000 档最多 58.6% 的监督节点被缠绕,
            # 原样最大偏差会被 2π 的伪误差顶满。折回误差 = |wrap(pred − 存档)|,反映真实偏差。
            d = pu_p[m, TH] - pu_t[m, TH]
            etw.append(np.abs((d + np.pi) % (2 * np.pi) - np.pi).astype(np.float64))
    return (np.concatenate(et) if et else np.zeros(0),
            np.concatenate(ev) if ev else np.zeros(0),
            np.concatenate(etw) if etw else np.zeros(0))


def dc_theta_err(raws, lossy: bool):
    """带网损 / 普通 DC 的逐节点 |θ 误差|(零训练参照)。"""
    out = []
    for s in raws:
        _mf, _tg, lm = _pf_typed_mask(s.node_feat, s.node_missing)
        d = dc_theta_lossy(s) if lossy else dc_theta(s)
        m = lm[:, TH] & (d[:, 1] > 0.5)
        if m.any():
            out.append(np.abs(d[m, 0] - s.node_feat[m, TH]).astype(np.float64))
    return np.concatenate(out) if out else np.zeros(0)


def stat(e):
    if e.size == 0:
        return dict(n=0)
    return dict(n=int(e.size), mean=float(e.mean()), p99=float(np.percentile(e, 99)),
                max=float(e.max()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True,
                    help='"标签:根目录:split",可重复')
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--val-holdout", action="append", default=[],
                    help="库内留出清单 json:命中的样本单列为「同拓扑·留出未见」")
    ap.add_argument("--families", default=None, help="只看这些 family_id(逗号分隔)")
    ap.add_argument("--limit", type=int, default=0, help="每个家族最多取几个方式(0 = 全部)")
    ap.add_argument("--v1-dir", default=str(HERE.parent / "runs" / "encoder_v1"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--big-bs", type=int, default=1, help="节点数 > big-n 的家族改用此批大小")
    ap.add_argument("--big-n", type=int, default=4096)
    ap.add_argument("--dc-ref", choices=["lossy", "both", "none"], default="lossy",
                    help="零训练参照:带网损 DC(默认) / 两种都算 / 不算。万节点档算普通 DC 很贵")
    ap.add_argument("--limit-big", type=int, default=0,
                    help="只对节点数 > big-n 的家族限量(0 = 不限);7 万节点档用它控制耗时")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    hold = set()
    for hp in a.val_holdout:
        hold |= set(json.loads(Path(hp).read_text(encoding="utf-8"))["val_holdout_sample_ids"])
    keep_fam = set(a.families.split(",")) if a.families else None

    groups = []            # (标签, family, 训练状态, [raw...])
    for spec in a.set:
        tag, root, split = spec.split(":", 2)
        byfam = {}
        rp = Path(root)
        if not (rp / "families").exists() and (rp / "manifest.csv").exists():
            # 方式库布局(lib118 / lib_ieee39 / lib_cepri36):整个目录就是一个家族
            from gridfm.data.dataset import library_files
            byfam[rp.name] = list(library_files(rp))
        else:
            for p in family_files(rp, split or None):
                byfam.setdefault(Path(p).parent.name, []).append(p)
        for fam, ps in sorted(byfam.items()):
            if keep_fam and fam not in keep_fam:
                continue
            seen = [q for q in ps if Path(q).stem not in hold]
            held = [q for q in ps if Path(q).stem in hold]
            for sub, status in ((seen, tag), (held, "同拓扑·留出未见")):
                if not sub:
                    continue
                if a.limit:
                    sub = sub[:a.limit]
                if a.limit_big:
                    probe = GridSample.load_npz(sub[0])
                    if probe.n_nodes > a.big_n:
                        sub = sub[:a.limit_big]
                raws_g = [GridSample.load_npz(q) for q in sub]
                M.attach_family_extra(raws_g, [rp])
                groups.append((tag, fam, status, raws_g))
    print(f"[byfam] 分组 {len(groups)} 个,样本合计 {sum(len(g[3]) for g in groups)}", flush=True)

    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    rep = {"groups": []}
    dcs = {}
    for tag, fam, status, raws in groups:
        key = (fam, status)
        dcs[key] = dict(lossy=stat(dc_theta_err(raws, True)) if a.dc_ref in ("lossy", "both") else dict(n=0),
                        raw=stat(dc_theta_err(raws, False)) if a.dc_ref == "both" else dict(n=0))
    for ckpt in a.ckpt:
        enc, reasoner, cfg, norm, meta = build_from_ckpt(ckpt, a.v1_dir, dev, a.seed)
        label = Path(ckpt).resolve().parent.parent.name.replace("pf_reasoner_5060_", "")
        dn = torch.from_numpy(pf_phys_denorm(norm)).to(dev) if meta["use_phys"] else None
        for tag, fam, status, raws in groups:
            n_nodes = int(np.mean([q.n_nodes for q in raws]))
            bs = a.big_bs if n_nodes > a.big_n else a.bs
            t0 = time.time()
            et, ev, etw = errors(enc, reasoner, raws, norm, cfg, dev, meta, dn, bs)
            d = dcs[(fam, status)]
            rec = dict(model=label, set=tag, family=fam, status=status, n_samples=len(raws),
                       n_nodes=n_nodes, theta=stat(et), theta_wrap=stat(etw), v=stat(ev),
                       dc_lossy_theta=d["lossy"], dc_raw_theta=d["raw"],
                       secs=round(time.time() - t0, 1))
            rep["groups"].append(rec)
            th, vv = rec["theta"], rec["v"]
            print(f"{label:<26}{fam:<16}{status:<16}n={len(raws):<4}节点{n_nodes:<6}"
                  f"θ 平均 {th.get('mean', float('nan')):.4f} p99 {th.get('p99', float('nan')):.3f} "
                  f"最大 {th.get('max', float('nan')):.3f}(折回后 {rec['theta_wrap'].get('max', float('nan')):.3f}) rad | "
                  f"v 平均 {vv.get('mean', float('nan')):.5f} p99 {vv.get('p99', float('nan')):.4f} "
                  f"最大 {vv.get('max', float('nan')):.4f} pu | {rec['secs']}s", flush=True)
        del enc, reasoner
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[byfam] 报告 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
