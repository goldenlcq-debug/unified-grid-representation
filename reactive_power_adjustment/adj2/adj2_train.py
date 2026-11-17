# -*- coding: utf-8 -*-
r"""调整器效果提升线(2026-08-22):有功/无功调整推理器训练(V1 编码器冻结只训推理头;GPU;与 PSASP 错时)。

输入 = adj2_labelgen 产的标签库(adjust_labels.jsonl + manifest.jsonl + cases/<tag>.npz,可多库混训、跨档混批);
划分 = 按源方式 sid 留出(同一方式的多条标签不跨集;考卷源方式在造标签时已剔除)。

变体(--variant):
  有功  v07   = AdjustReasoner(side_dim=5):侧通道 [低压深度, 机组上/下行余量比, 平衡机上/下行裕度(全图广播)](adj2_common.active_side_feats)
               + 损失 = 指针 BCE(可加权) + ΔP L1(规范解位) + w_slack × 平衡机一致性罚(预测净增量超出平衡机裕度带的部分);
        v06re = 老结构(side_dim=1 越限深度)只换新数据(对照:分离"数据"与"结构"两个因素)。
  无功  v6    = v5 三头(指针 + 4 档 Δv_set 分类 + 并补指针)+ **台数头**(图级,监督规范解机组数,类别均衡加权)
               → 解码 = 台数头定 k、指针排序取前 k(最小干预正则的结构落点);
        v5re  = v5 配方(三头、指针正类加权 auto、档位均衡、并补加权)只换新数据(对照)。
LaPE/RWSE 每个拓扑只算一次(gridfm 缓存)并把 Data 缓存在内存,每步不重算;随机特征列沿样本持久化值(与推理口径一致)。
--dyn-fill(2026-08-23):动态列填法开关(adj2_common.apply_dyn_fill),zero=现状 / mean=缺失动态组机组节点填归一化均值;写进检查点。

  python adj2_train.py --task active --variant v07 --labels ../libs/labels_p_118_reverse ../libs/labels_p_2000_reverse --out ../调整器效果提升-20260822/runs/train_p_v07_s0 --seed 0
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adj2_common as C  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from gridfm.data.graph import sample_to_data  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.reason.adjust_task import AdjustReasoner, build_adjust_example, unit_span, viol_depth  # noqa: E402
from gridfm.reason.reactive_task import (  # noqa: E402
    DV_LEVELS, ReactiveShuntReasoner, build_reactive_example, react_side, reactive_shunt_loss, unit_span_vset)
from gridfm.reason.ts_task import gen_node_mask  # noqa: E402
from gridfm.schema.sample import GridSample  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402

K_MAX = 12


# ---------------------------------------------------------------- 模型
class ReactiveCountReasoner(ReactiveShuntReasoner):
    """v6:v5 三头 + 图级台数头 head_k(0..K_MAX 分类;0 = 纯并补)。主干/三头参数集合与 v5 完全同构。"""

    def __init__(self, d, n_queries=16, n_layers=2, n_heads=4, ffn_mult=2, side_dim=3, n_dv_levels=4, n_k_max=K_MAX):
        super().__init__(d=d, n_queries=n_queries, n_layers=n_layers, n_heads=n_heads, ffn_mult=ffn_mult,
                         side_dim=side_dim, n_dv_levels=n_dv_levels)
        self.n_k_max = int(n_k_max)
        self.head_k = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, self.n_k_max + 1))

    def forward(self, h, g, e, node_batch, edge_batch, depth, n_graphs=None):
        feat = self.node_feat(h, g, e, node_batch, edge_batch, depth)
        dv = self.head_dp(feat)
        nb = node_batch
        B = int(n_graphs) if n_graphs is not None else (int(nb.max()) + 1 if nb.numel() else 0)
        f32 = feat.float()
        pooled = torch.zeros(B, f32.shape[-1], dtype=torch.float32, device=feat.device)
        pooled.index_add_(0, nb, f32)
        cnt = torch.zeros(B, dtype=torch.float32, device=feat.device)
        cnt.index_add_(0, nb, torch.ones_like(nb, dtype=torch.float32))
        pooled = (pooled / cnt.clamp_min(1.0).unsqueeze(-1)).to(feat.dtype)
        return (self.head_ptr(feat).squeeze(-1), dv, self.head_shunt(feat).squeeze(-1), self.head_k(pooled))


K_MAX_ACTIVE = 32


class ActiveCountReasoner(AdjustReasoner):
    """v07d:有功双头 + 图级台数头(0..K_MAX_ACTIVE 分类),解码 top-k(k>=1)+ 平衡机裕度投影。"""

    def __init__(self, d, n_queries=16, n_layers=2, n_heads=4, ffn_mult=2, side_dim=6, n_k_max=K_MAX_ACTIVE):
        super().__init__(d=d, n_queries=n_queries, n_layers=n_layers, n_heads=n_heads, ffn_mult=ffn_mult, side_dim=side_dim)
        self.n_k_max = int(n_k_max)
        self.head_k = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, self.n_k_max + 1))

    def forward(self, h, g, e, node_batch, edge_batch, depth, n_graphs=None):
        feat = self.node_feat(h, g, e, node_batch, edge_batch, depth)
        nb = node_batch
        B = int(n_graphs) if n_graphs is not None else (int(nb.max()) + 1 if nb.numel() else 0)
        f32 = feat.float()
        pooled = torch.zeros(B, f32.shape[-1], dtype=torch.float32, device=feat.device)
        pooled.index_add_(0, nb, f32)
        cnt = torch.zeros(B, dtype=torch.float32, device=feat.device)
        cnt.index_add_(0, nb, torch.ones_like(nb, dtype=torch.float32))
        pooled = (pooled / cnt.clamp_min(1.0).unsqueeze(-1)).to(feat.dtype)
        return (self.head_ptr(feat).squeeze(-1), self.head_dp(feat).squeeze(-1), self.head_k(pooled))


# 变体表:side_dim / 是否带越限邻近度特征(prox) / 是否带台数头
ACTIVE_VARIANTS = {"v06re": dict(side=1, prox=False, count=False), "v07": dict(side=5, prox=False, count=False),
                   "v07d": dict(side=6, prox=True, count=True)}
REACTIVE_VARIANTS = {"v5re": dict(side=3, prox=False, count=False), "v6": dict(side=3, prox=False, count=True),
                     "v6d": dict(side=4, prox=True, count=True),
                     # v6e(2026-08-24 A 项):结构与 v6d 完全同构(head_k 仍 13 类输出),仅把 head_k 的标签语义
                     # 从"台数 0..12(clamp)"换成"阶梯档位 0..12"(0=纯并补,1..12=C.RUNG_TABLE 的 (k,mag));
                     # 动因:y_k clamp 12 把 21 条 17~32 台的大网标签砍掉、台数与幅度不联合 → 首步 IEEE118 10%/2000 局部 20%。
                     "v6e": dict(side=4, prox=True, count=True)}


def build_model(task: str, variant: str, d: int, queries: int, layers: int):
    if task == "active":
        v = ACTIVE_VARIANTS[variant]
        if v["count"]:
            return (ActiveCountReasoner(d=d, n_queries=queries, n_layers=layers, side_dim=v["side"]),
                    {"side_dim": v["side"], "prox": v["prox"], "k_max": K_MAX_ACTIVE})
        return AdjustReasoner(d=d, n_queries=queries, n_layers=layers, side_dim=v["side"]), {"side_dim": v["side"], "prox": v["prox"]}
    v = REACTIVE_VARIANTS[variant]
    if v["count"]:
        return (ReactiveCountReasoner(d=d, n_queries=queries, n_layers=layers, side_dim=v["side"], n_dv_levels=len(DV_LEVELS)),
                {"side_dim": v["side"], "with_shunt": True, "n_dv_levels": len(DV_LEVELS), "k_max": K_MAX, "prox": v["prox"]})
    return (ReactiveShuntReasoner(d=d, n_queries=queries, n_layers=layers, side_dim=v["side"], n_dv_levels=len(DV_LEVELS)),
            {"side_dim": v["side"], "with_shunt": True, "n_dv_levels": len(DV_LEVELS), "prox": v["prox"]})


def side_feats(task: str, variant: str, raw, low: float = 0.93, high: float = 1.08) -> np.ndarray:
    """侧通道(训练与跑分器共用同一函数,口径不会漂)。"""
    if task == "active":
        v = ACTIVE_VARIANTS[variant]
        base = C.active_side_feats(raw, low) if v["side"] >= 5 else viol_depth(raw, low)[:, None]
        if v["prox"]:
            base = np.concatenate([base, C.violation_proximity(raw, low)[:, None]], axis=1)
        return base.astype(np.float32)
    v = REACTIVE_VARIANTS[variant]
    base = react_side(raw, low, high, with_shunt=True)
    if v["prox"]:
        base = np.concatenate([base, C.violation_proximity(raw, low, high)[:, None]], axis=1)
    return base.astype(np.float32)


# ---------------------------------------------------------------- 数据
def _sid_seed(sid: str) -> int:
    return int.from_bytes(hashlib.sha256(sid.encode("utf-8")).digest()[:4], "big")


def load_examples(task: str, labels_dirs):
    examples, raw_by_tag, n_missing, n_bad = [], {}, 0, 0
    for lab in labels_dirs:
        lab = Path(lab)
        if not (lab / "adjust_labels.jsonl").is_file():
            print("[adj2-train] 标签库缺失,跳过: %s" % lab)
            continue
        rows = [json.loads(ln) for ln in (lab / "adjust_labels.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in (lab / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            m = json.loads(ln)
            row = rows[m["row_index"]]
            npz = lab / "cases" / ("%s.npz" % m["tag"])
            if not npz.exists():
                n_missing += 1
                continue
            raw = GridSample.load_npz(npz)
            try:
                ex = (build_adjust_example(row, raw, m["tag"]) if task == "active"
                      else build_reactive_example(row, raw, m["tag"]))
            except ValueError as e:
                n_bad += 1
                print("[adj2-train] 标签错位跳过 %s: %s" % (m["tag"], e))
                continue
            if ex is None:
                continue
            ex["tier"] = row.get("tier")
            ex["family"] = row.get("family")
            ex["mechanism"] = row.get("mechanism")
            ex["batch"] = lab.name
            ex["rung_src"] = (row.get("canonical_fix") or {}).get("source", "")   # v6e 档位标签来源
            raw_by_tag[m["tag"]] = raw
            examples.append(ex)
    return examples, raw_by_tag, n_missing, n_bad


def make_data(task: str, variant: str, ex, raw, normed, cfg):
    data = sample_to_data(normed, cfg.lape_k, cfg.rwse_steps, random_dim=cfg.random_dim,
                          branch_random_dim=cfg.branch_random_dim)
    gm = gen_node_mask(raw)
    if task == "active":
        # 固定出力机组(pmax<=pmin,MATPOWER 口径不可调)不进指针候选:2000 档 116/391 台、3120 档 24/247 台(2026-08-22 实测)
        from gridfm.schema.fields import NODE_TABLE as _NT
        pmax = raw.node_feat[:, _NT.offset_of("pmax")]
        pmin = raw.node_feat[:, _NT.offset_of("pmin")]
        gm = gm & (pmax > pmin + 1e-9)
    data.gen_mask = torch.from_numpy(gm)
    data.y_units = torch.from_numpy(ex["y_units"])
    data.y_dp = torch.from_numpy(ex["y_dp"])
    data.y_dp_mask = torch.from_numpy(ex["y_dp_mask"])
    data.depth = torch.from_numpy(side_feats(task, variant, raw))
    if task == "active":
        data.y_k = torch.tensor([min(int(ex["y_dp_mask"].sum()), K_MAX_ACTIVE)], dtype=torch.long)
        data.span = torch.from_numpy(unit_span(raw))
        up, dn = C.slack_headroom_from_raw(raw)
        big = 1e4
        data.slack_up = torch.tensor([min(up, big)], dtype=torch.float32)
        data.slack_dn = torch.tensor([min(dn, big)], dtype=torch.float32)
    else:
        data.shunt_cand = torch.from_numpy(ex["shunt_cand"])
        data.y_shunt = torch.from_numpy(ex["y_shunt"])
        data.y_dv_cls = torch.from_numpy(ex["y_dv_cls"])
        if variant == "v6e":
            data.y_k = torch.tensor([C.rung_index(ex.get("rung_src", ""))], dtype=torch.long)
        else:
            data.y_k = torch.tensor([min(int(ex["y_dp_mask"].sum()), K_MAX)], dtype=torch.long)
    return data


# ---------------------------------------------------------------- 损失
def active_loss(ptr, dp, batch, w_ptr, w_dp, w_slack, ptr_pw, tol=C.SLACK_TOL, k_logits=None, w_k=1.0, k_cw=None):
    gm = batch.gen_mask
    pw = torch.as_tensor(float(ptr_pw), device=ptr.device, dtype=ptr.dtype) if ptr_pw else None
    l_ptr = F.binary_cross_entropy_with_logits(ptr[gm], batch.y_units[gm], pos_weight=pw)
    m = batch.y_dp_mask
    l_dp = F.l1_loss(dp[m].float(), batch.y_dp[m]) if bool(m.any()) else torch.zeros((), device=ptr.device)
    l_sl = torch.zeros((), device=ptr.device)
    if w_slack > 0:
        # 预测净增量 net_b = Σ_gens σ(ptr)·dp·span;罚超出 [−(up+tol), dn+tol] 的部分,按图内机组平均幅度归一
        nb = batch.batch
        B = int(batch.num_graphs)
        contrib = (torch.sigmoid(ptr.float()) * dp.float() * batch.span.float()) * gm.float()
        net = torch.zeros(B, device=ptr.device).index_add_(0, nb, contrib)
        span_sum = torch.zeros(B, device=ptr.device).index_add_(0, nb, batch.span.float() * gm.float())
        n_gen = torch.zeros(B, device=ptr.device).index_add_(0, nb, gm.float())
        mean_span = span_sum / n_gen.clamp_min(1.0)
        up = batch.slack_up.float().clamp_min(0.0) + tol
        dn = batch.slack_dn.float().clamp_min(0.0) + tol
        over = F.relu(net - dn) + F.relu(-net - up)
        l_sl = (over / mean_span.clamp_min(1e-3)).mean()
    total = w_ptr * l_ptr + w_dp * l_dp + w_slack * l_sl
    diag = {"total": float(total.detach()), "ptr": float(l_ptr.detach()), "dp": float(l_dp.detach()), "slack": float(l_sl.detach())}
    if k_logits is not None:
        kw = k_cw.to(k_logits.device).float() if k_cw is not None else None
        l_k = F.cross_entropy(k_logits.float(), batch.y_k.view(-1), weight=kw)
        total = total + w_k * l_k
        diag["k"] = float(l_k.detach())
        diag["total"] = float(total.detach())
    return total, diag


def reactive_loss_v6(heads, batch, w_ptr, w_dp, w_shunt, w_k, shunt_pw, dv_cw, ptr_pw, k_cw, with_k: bool):
    if with_k:
        ptr, dv, sh, kl = heads
    else:
        ptr, dv, sh = heads
        kl = None
    total, diag = reactive_shunt_loss(ptr, dv, sh, batch, w_ptr, w_dp, w_shunt, shunt_pw, dv_cw, ptr_pw)
    if kl is not None:
        kw = k_cw.to(kl.device).float() if k_cw is not None else None
        l_k = F.cross_entropy(kl.float(), batch.y_k.view(-1), weight=kw)
        total = total + w_k * l_k
        diag = dict(diag)
        diag["k"] = float(l_k.detach())
        diag["total"] = float(total.detach())
    return total, diag


# ---------------------------------------------------------------- 评估(监督口径;功能性验收走 adj2_eval + PSASP)
@torch.no_grad()
def eval_sup(task, variant, enc, reasoner, exs, data_by_tag, dev, use_amp, loss_fn, bs=8):
    reasoner.eval()
    tp = fp = fn = 0
    losses, n_sel, n_true, k_hit, n_graph, dv_hit, dv_n = [], [], [], 0, 0, 0, 0
    for i in range(0, len(exs), bs):
        chunk = exs[i:i + bs]
        batch = Batch.from_data_list([data_by_tag[x["tag"]] for x in chunk]).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            h, g, e = enc(batch)
            heads = reasoner(h, g, e, batch.batch, batch.batch[batch.edge_index[0]], batch.depth,
                             **({"n_graphs": int(batch.num_graphs)} if variant in ("v6", "v6d", "v6e", "v07d") else {}))
        loss, _ = loss_fn(heads, batch)
        losses.append(float(loss))
        ptr = heads[0]
        gm = batch.gen_mask
        prob = torch.sigmoid(ptr.float())
        nb = batch.batch
        for b in range(int(batch.num_graphs)):
            rows = torch.where((nb == b) & gm)[0]
            if not len(rows):
                continue
            tru = set(torch.where((nb == b) & gm & (batch.y_units > 0.5))[0].tolist())
            if variant in ("v6", "v6d", "v6e", "v07d"):
                kl_ = heads[3] if task == "reactive" else heads[2]
                kp = int(kl_[b].float().argmax())
                order = rows[torch.argsort(prob[rows], descending=True)]
                sel = set(order[:max(kp, 1)].tolist())
                kt = int(batch.y_k[b])
                k_hit += int(kp == kt)
            else:
                sel = set(rows[prob[rows] > 0.5].tolist())
                if not sel:
                    sel = {int(rows[int(torch.argmax(prob[rows]))])}
            n_graph += 1
            n_sel.append(len(sel)); n_true.append(len(tru))
            tp += len(sel & tru); fp += len(sel - tru); fn += len(tru - sel)
        m = batch.y_dp_mask
        if task == "reactive" and bool(m.any()):
            dv_hit += int((heads[1][m].float().argmax(-1) == batch.y_dv_cls[m]).sum()); dv_n += int(m.sum())
    reasoner.train()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    out = {"loss": float(np.mean(losses)) if losses else float("nan"), "ptr_precision": prec, "ptr_recall": rec,
           "ptr_f1": 2 * prec * rec / max(prec + rec, 1e-9), "n": len(exs),
           "n_sel_mean": float(np.mean(n_sel)) if n_sel else 0.0, "n_true_mean": float(np.mean(n_true)) if n_true else 0.0}
    if variant in ("v6", "v6d", "v6e", "v07d"):
        out["k_acc"] = k_hit / max(n_graph, 1)
    if task == "reactive":
        out["dv_cls_acc"] = dv_hit / max(dv_n, 1)
    return out


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["active", "reactive"])
    ap.add_argument("--variant", required=True, choices=["v07", "v06re", "v07d", "v6", "v5re", "v6d", "v6e"])
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v1-dir", default=str(C.REPO / "runs" / "encoder_v1"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--queries", type=int, default=16)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--w-ptr", type=float, default=1.0)
    ap.add_argument("--w-dp", type=float, default=1.0)
    ap.add_argument("--w-shunt", type=float, default=1.0)
    ap.add_argument("--w-k", type=float, default=1.0)
    ap.add_argument("--w-slack", type=float, default=0.5, help="有功 v07:平衡机一致性罚权重(v06re 固定 0)")
    ap.add_argument("--ptr-pos-weight", default="auto", help="机组指针正类权重:none / auto(负正比,封顶 --ptr-pw-cap)/ 数值")
    ap.add_argument("--ptr-pw-cap", type=float, default=10.0)
    ap.add_argument("--shunt-pos-weight", default="auto")
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--select-by", default="loss", choices=["loss", "f1"],
                    help="best.pt 选点:loss=留出总损失最小(缺省,仓里先例);f1=留出指针 F1 最大(第二轮发现台数头/档位头的 CE 把留出 loss 抬高,"
                         "而指针 F1 还在涨——按 loss 选会错过指针头最好的点)")
    ap.add_argument("--dyn-fill", default="zero", choices=list(C.DYN_FILL_MODES),
                    help="动态列填法(数据级规则开关,不动编码器;2026-08-22 用户裁决做对照):zero=现状(常量伪值);"
                         "mean=动态组缺失的机组节点四列 tj/xdp/sh/ph 填归一化均值(z=0);mean_all=所有缺失节点都填(对照项)。"
                         "写进检查点 config.dyn_fill,跑分器 --dyn-fill auto 按它对齐")
    a = ap.parse_args()
    if a.task == "active" and a.variant not in ACTIVE_VARIANTS:
        raise SystemExit("有功变体只能是 %s" % list(ACTIVE_VARIANTS))
    if a.task == "reactive" and a.variant not in REACTIVE_VARIANTS:
        raise SystemExit("无功变体只能是 %s" % list(REACTIVE_VARIANTS))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "ckpt" / "best.pt").is_file():
        raise SystemExit("[adj2-train] %s/ckpt 已有检查点,拒绝覆写;换 --out" % out)
    w_slack = a.w_slack if a.variant in ("v07", "v07d") else 0.0
    with_count = (a.task == "active" and ACTIVE_VARIANTS[a.variant]["count"]) or (a.task == "reactive" and REACTIVE_VARIANTS[a.variant]["count"])

    v1 = Path(a.v1_dir)
    ck = torch.load(v1 / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in allowed})
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    enc, _dec = build_models(cfg)
    enc.load_state_dict(ck["model_state"])
    for p in enc.parameters():
        p.requires_grad_(False)
    enc.to(dev).eval()
    norm = Normalizer.load(v1 / "normalizer.npz")
    print("[adj2-train] 编码器 = v1(冻结 d%d/L%d) 设备 %s" % (cfg.d, cfg.n_layers, dev), flush=True)

    t0 = time.time()
    examples, raw_by_tag, n_missing, n_bad = load_examples(a.task, a.labels)
    if not examples:
        raise SystemExit("[adj2-train] 无训练例")
    data_by_tag = {}
    n_dyn_filled = 0
    for x in examples:
        raw = raw_by_tag[x["tag"]]
        normed = copy.deepcopy(raw); norm.transform(normed)
        n_dyn_filled += C.apply_dyn_fill(normed, raw, a.dyn_fill)
        data_by_tag[x["tag"]] = make_data(a.task, a.variant, x, raw, normed, cfg)
    print("[adj2-train] 例 %d(缺落料 %d,错位 %d);Data 缓存 %.0fs;dyn_fill=%s(改动节点 %d)"
          % (len(examples), n_missing, n_bad, time.time() - t0, a.dyn_fill, n_dyn_filled), flush=True)

    fams = sorted({x["battery"] for x in examples})
    held_by_fam = {}
    for fam in fams:
        sids = sorted({x["sid"] for x in examples if x["battery"] == fam})
        rng = np.random.default_rng(a.seed * 1_000_003 + _sid_seed(fam))
        k = max(1, int(round(a.holdout_frac * len(sids))))
        held_by_fam[fam] = sorted(rng.permutation(sids)[:k].tolist())
    held = {s for ss in held_by_fam.values() for s in ss}
    train_exs = [x for x in examples if x["sid"] not in held]
    eval_exs = [x for x in examples if x["sid"] in held]
    by_batch = {}
    for x in examples:
        by_batch[x["batch"]] = by_batch.get(x["batch"], 0) + 1
    print("[adj2-train] train %d / 留出 %d(%s);按库 %s" % (len(train_exs), len(eval_exs),
          " + ".join("%s %d 方式" % (f, len(held_by_fam[f])) for f in fams), by_batch), flush=True)
    if not eval_exs:
        raise SystemExit("[adj2-train] 留出集为空")

    # 指针正类权重
    ptr_pw = None
    if str(a.ptr_pos_weight).lower() != "none":
        if str(a.ptr_pos_weight).lower() == "auto":
            npos = float(sum(float(x["y_units"].sum()) for x in train_exs))
            ncand = float(sum(float(gen_node_mask(raw_by_tag[x["tag"]]).sum()) for x in train_exs))
            ptr_pw = min((ncand - npos) / max(npos, 1.0), a.ptr_pw_cap)
            print("[adj2-train] 指针正类权重 auto = %.2f(正例 %.0f/%.0f = %.4f;封顶 %.1f)" % (ptr_pw, npos, ncand, npos / max(ncand, 1), a.ptr_pw_cap))
        else:
            ptr_pw = float(a.ptr_pos_weight)
    shunt_pw, dv_cw, k_cw = None, None, None
    if a.task == "active" and with_count:
        kc = np.zeros(K_MAX_ACTIVE + 1)
        for x in train_exs:
            kc[min(int(x["y_dp_mask"].sum()), K_MAX_ACTIVE)] += 1
        pk = kc > 0
        wk = np.zeros_like(kc)
        wk[pk] = kc[pk].sum() / (pk.sum() * kc[pk])
        k_cw = torch.tensor(wk, dtype=torch.float32, device=dev)
        print("[adj2-train] 有功规范解机组数分布 %s" % {int(i): int(v) for i, v in enumerate(kc) if v})
    if a.task == "reactive":
        npos = float(sum(float(x["y_shunt"].sum()) for x in train_exs))
        ncand = float(sum(float(x["shunt_cand"].sum()) for x in train_exs))
        if str(a.shunt_pos_weight).lower() == "auto":
            shunt_pw = (ncand - npos) / max(npos, 1.0) if npos > 0 else None
        elif str(a.shunt_pos_weight).lower() != "none":
            shunt_pw = float(a.shunt_pos_weight)
        print("[adj2-train] 并补候选 %.0f 正例 %.0f → 正类权重 %s" % (ncand, npos, shunt_pw))
        cnt = np.zeros(len(DV_LEVELS))
        for x in train_exs:
            for k_ in x["y_dv_cls"][x["y_dp_mask"]].tolist():
                cnt[k_] += 1
        present = cnt > 0
        w = np.zeros_like(cnt); w[present] = cnt[present].sum() / (present.sum() * cnt[present])
        dv_cw = torch.tensor(w, dtype=torch.float32, device=dev)
        print("[adj2-train] 档位计数 %s → 均衡权重 %s" % (cnt.astype(int).tolist(), np.round(w, 3).tolist()))
        kc = np.zeros(K_MAX + 1)
        for x in train_exs:
            if a.variant == "v6e":
                kc[C.rung_index(x.get("rung_src", ""))] += 1
            else:
                kc[min(int(x["y_dp_mask"].sum()), K_MAX)] += 1
        pk = kc > 0
        wk = np.zeros_like(kc); wk[pk] = kc[pk].sum() / (pk.sum() * kc[pk])
        k_cw = torch.tensor(wk, dtype=torch.float32, device=dev)
        print("[adj2-train] %s分布 %s" % ("规范解档位" if a.variant == "v6e" else "规范解机组数",
                                          {int(i): int(v) for i, v in enumerate(kc) if v}))

    reasoner, mcfg = build_model(a.task, a.variant, cfg.d, a.queries, a.layers)
    reasoner = reasoner.to(dev)
    print("[adj2-train] 推理器 %s 参数 %.2fM %s" % (a.variant, sum(p.numel() for p in reasoner.parameters()) / 1e6, mcfg), flush=True)

    def loss_fn(heads, batch):
        if a.task == "active":
            return active_loss(heads[0], heads[1], batch, a.w_ptr, a.w_dp, w_slack, ptr_pw,
                               k_logits=(heads[2] if with_count else None), w_k=a.w_k, k_cw=k_cw)
        return reactive_loss_v6(heads, batch, a.w_ptr, a.w_dp, a.w_shunt, a.w_k, shunt_pw, dv_cw, ptr_pw, k_cw,
                                with_k=with_count)

    opt = torch.optim.Adam(reasoner.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    use_amp = dev.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    rng = np.random.default_rng(a.seed)
    best, bad = float("inf"), 0
    warm = max(1, int(0.05 * a.steps))
    t1 = time.time()
    for step in range(a.steps):
        lr = (a.lr * (step + 1) / warm if step < warm else
              a.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, a.steps - warm - 1))))
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = rng.choice(len(train_exs), size=min(a.batch, len(train_exs)), replace=False)
        batch = Batch.from_data_list([data_by_tag[train_exs[int(i)]["tag"]] for i in idx]).to(dev)
        opt.zero_grad()
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                h, g, e = enc(batch)
            heads = reasoner(h, g, e, batch.batch, batch.batch[batch.edge_index[0]], batch.depth,
                             **({"n_graphs": int(batch.num_graphs)} if with_count else {}))
            loss, diag = loss_fn(heads, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(reasoner.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if (step + 1) % max(1, a.steps // 10) == 0:
            print("[adj2-train] step %d/%d loss=%.4f %s (%.0fs)" % (step + 1, a.steps, diag["total"],
                  " ".join("%s=%.4f" % (k, v) for k, v in diag.items() if k != "total"), time.time() - t1), flush=True)
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            m = eval_sup(a.task, a.variant, enc, reasoner, eval_exs, data_by_tag, dev, use_amp, loss_fn)
            print("[adj2-train] eval step %d: loss=%.4f ptrF1=%.3f P=%.3f R=%.3f sel=%.1f/true=%.1f%s%s" % (
                step + 1, m["loss"], m["ptr_f1"], m["ptr_precision"], m["ptr_recall"], m["n_sel_mean"], m["n_true_mean"],
                (" kAcc=%.3f" % m["k_acc"]) if "k_acc" in m else "", (" dvAcc=%.3f" % m["dv_cls_acc"]) if "dv_cls_acc" in m else ""), flush=True)
            (out / "ckpt").mkdir(parents=True, exist_ok=True)
            state = {"reasoner_state": reasoner.state_dict(), "step": step + 1, "held_by_fam": held_by_fam, "seed": a.seed,
                     "config": {"task": a.task, "variant": a.variant, "queries": a.queries, "layers": a.layers, "d": cfg.d,
                                "w_ptr": a.w_ptr, "w_dp": a.w_dp, "w_shunt": a.w_shunt, "w_k": a.w_k, "w_slack": w_slack,
                                "ptr_pos_weight": ptr_pw, "shunt_pos_weight": shunt_pw, "dv_levels": list(DV_LEVELS),
                                "labels": [str(Path(x).name) for x in a.labels], "select_by": a.select_by,
                                "dyn_fill": a.dyn_fill, **mcfg}}
            with (out / "eval_history.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"step": step + 1, **m}, ensure_ascii=False) + "\n")
            score = m["loss"] if a.select_by == "loss" else -float(m["ptr_f1"])
            if score < best - 1e-6:
                best, bad = score, 0
                torch.save(state, out / "ckpt" / "best.pt")
            else:
                bad += 1
                if bad >= a.patience:
                    print("[adj2-train] 早停于 step %d" % (step + 1), flush=True)
                    break
    bck = torch.load(out / "ckpt" / "best.pt", map_location=dev, weights_only=False)
    reasoner.load_state_dict(bck["reasoner_state"])
    m_final = eval_sup(a.task, a.variant, enc, reasoner, eval_exs, data_by_tag, dev, use_amp, loss_fn)
    rep = {"task": a.task, "variant": a.variant, "n_examples": len(examples), "n_train": len(train_exs), "n_eval": len(eval_exs),
           "by_batch": by_batch, "held_by_fam": held_by_fam, "metrics_eval": m_final, "best_step": bck["step"], "seed": a.seed,
           "reasoner_params": sum(p.numel() for p in reasoner.parameters()), "config": bck["config"],
           "select_by": a.select_by, "dyn_fill": a.dyn_fill, "n_dyn_filled_nodes": n_dyn_filled,
           "gpu_secs": round(time.time() - t1, 1), "peak_gpu_mb": (round(torch.cuda.max_memory_allocated() / 1e6) if use_amp else None),
           "note": "监督口径;功能性验收(考卷三臂 + 实际干预 + 合理性)走 adj2_eval + PSASP"}
    (out / "train_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[adj2-train] 报告 %s best_step=%d eval=%s" % (out / "train_report.json", bck["step"], json.dumps(m_final)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
