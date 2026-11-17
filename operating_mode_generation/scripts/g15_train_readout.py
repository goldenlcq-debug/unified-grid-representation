"""[G] G1.5:读出解码器训练(方案 6.5 触发情形一,用户批准方案 (a);编码器冻结不动)。

监督 = **全位置读出**(诊断 A 判定 v1 解码器 readout 通路从未受监督);损失 = recon_loss
同款(Huber/BCE 逐列,组权重 state=2.0 其余 1.0,3.4.1 初值);潜码噪声增广 σ∈{0,0.1,0.2}
相对尺度(3.4.2 三前提②,补缺席)。数据 = lib118_diverse 训练 240 方式(留出 60 不碰;
训练内再切 val 做早停)。潜码全部**预计算缓存**(编码器冻结,一次前向)。

产物:runs/decoder_g15/readout.pt(**新目录;v1 原检查点原样保留,铁律**)。

  python scripts/g15_train_readout.py --root ..\\gridfm-data [--steps 3000]
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import gridfm  # noqa: E402
assert gridfm.__file__.startswith(str(REPO)), f"窜仓:{gridfm.__file__}"

from gridfm.data.cond_views import cgen_view  # noqa: E402
from gridfm.data.dataset import load_library  # noqa: E402
from gridfm.data.graph import sample_to_data  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.models.decoder.readout import ReadoutDecoder  # noqa: E402
from gridfm.models.decoder.recon import GROUP_RANGES, recon_loss  # noqa: E402
from gridfm.schema.fields import MASKABLE_GROUPS, NODE_TABLE  # noqa: E402
from gridfm.schema.sample import group_bit_index  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402

SIGMAS = (0.0, 0.1, 0.2)          # 3.4.2 潜码噪声增广档
GROUP_W = {"topology": 1.0, "static": 1.0, "dynamic": 1.0, "state": 2.0}   # 3.4.1 初值


def _loss_mask(sample) -> np.ndarray:
    """全位置读出监督掩码:各可遮蔽组在"该组非 missing"的行上全列计损失。"""
    n = sample.n_nodes
    m = np.zeros((n, NODE_TABLE.total_dim), dtype=bool)
    for g in MASKABLE_GROUPS:
        s, e = GROUP_RANGES[g]
        rows = ~sample.node_missing[:, group_bit_index(g)]
        m[rows, s:e] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lib-name", default="lib118_diverse")
    ap.add_argument("--v1-dir", default="runs/encoder_v1")
    ap.add_argument("--out", default="runs/decoder_g15_cgen",
                    help="c_gen 去泄漏版新目录;旧 runs/decoder_g15(泄漏版)留作对照臂")
    ap.add_argument("--test-every", type=int, default=5, help="留出切法同 g1/诊断A,留出不碰")
    ap.add_argument("--val-every", type=int, default=10, help="训练集内每 N 取 1 做早停 val")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--corpus", choices=("single", "g4"), default="single",
                    help="single=lib118_diverse 单库(v0.x 口径);"
                         "g4=多家族训练语料(corpus.py 单一出处,防泄漏断言内置)")
    ap.add_argument("--upweight-fams", default=None,
                    help="[外协包 F] 逐家族上采样权重,格式 fam1:w1,fam2:w2(fam 匹配 family_id)。"
                         "与 --upweight-lib 叠乘。用于把大网族(2000/3120)的曝光提到目标占比,"
                         "而不改变小网语料本身——历史 08-04 教训:直接扩语料伤同规模基本盘。")
    ap.add_argument("--upweight-lib", type=float, default=1.0,
                    help="g4 语料下 lib118(family_id=ieee118)样本的采样权重倍数——对冲"
                         "稀释(lib118 仅占语料 3.6%),恢复真网读出精度;1.0=不加权")
    ap.add_argument("--layers", type=int, default=2, help="ReadoutDecoder MPNN 层数(默认 2)")
    ap.add_argument("--ffn-mult", type=int, default=2, help="MPNN FFN 扩张倍数(默认 2)")
    ap.add_argument("--hidden", type=int, default=0,
                    help="分组头隐藏宽度(默认 0=2×d);加大专治 pg/qg 双调度冲突")
    a = ap.parse_args()
    out = Path(a.out)
    (out).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")

    lib = Path(a.root) / a.lib_name
    corpus_prov = None
    if a.corpus == "g4":
        from gridfm.data.corpus import g4_train_corpus
        pool, corpus_prov = g4_train_corpus(Path(a.root), a.lib_name, a.test_every)
        src_desc = f"g4_train({corpus_prov['fingerprint']})"
        n_heldout = corpus_prov["lib_pool"]["n_holdout_excluded"]
        print(f"[g15] G4 语料:{corpus_prov['n_samples']} 样本 = "
              f"{corpus_prov['n_synth_families']} 合成家族 + {a.lib_name} 训练份 "
              f"{corpus_prov['lib_pool']['n_train']};留出家族/规模档零载入(断言强制)")
    else:
        lf_by_sid, solvable = {}, {}
        with open(lib / "manifest.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                lf_by_sid[r["sample_id"]] = float(r["load_factor"])
                solvable[r["sample_id"]] = r["solvable"] == "1"
        samples = [s for s in load_library(lib) if solvable.get(s.sample_id)]
        samples.sort(key=lambda s: lf_by_sid[s.sample_id])
        test_idx = set(range(0, len(samples), a.test_every))
        pool = [s for i, s in enumerate(samples) if i not in test_idx]  # 240,留出 60 不碰
        src_desc = lib.name
        n_heldout = len(test_idx)
    val_idx = set(range(0, len(pool), a.val_every))
    train_s = [s for i, s in enumerate(pool) if i not in val_idx]
    val_s = [s for i, s in enumerate(pool) if i in val_idx]
    print(f"[g15] train {len(train_s)} / val {len(val_s)} / 留出 {n_heldout}(不碰)")

    # 指纹实算 + 与就位三查钉定值断言(审查修复:此前为硬编码常量,信任根失义;
    # 若确需换 v1 重训,改此钉定值——差异在 diff 里响亮可见)
    fp_v1 = hashlib.blake2b((Path(a.v1_dir) / "v1.pt").read_bytes(),
                            digest_size=16).hexdigest()
    assert fp_v1 == "ccf4260f2a8f5db5b6650a33b17d7a0f", \
        f"v1.pt 指纹 {fp_v1} 与就位三查钉定值不符,拒绝训练"
    ck = torch.load(Path(a.v1_dir) / "v1.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items()
                         if k in {f.name for f in dc_fields(TrainConfig)}})
    enc, _ = build_models(cfg)
    enc.load_state_dict(ck["model_state"])
    enc.eval()
    enc.to(dev)                    # G4:6640 样本编码前向上 GPU(v0.2 仅 240 走 CPU 够快;
    #                                缓存 .cpu() 落 CPU 内存,不占 6GB 显存,assemble 再逐 batch 回 GPU)
    norm = Normalizer.load(Path(a.v1_dir) / "normalizer.npz")

    # ---- 潜码预计算缓存(编码器冻结):h_full / e_ctopo / target / loss_mask / edge_index ----
    @torch.no_grad()
    def precompute(ss):
        cache = []
        for i in range(0, len(ss), a.batch):
            chunk = ss[i:i + a.batch]
            normed = []
            for raw in chunk:
                s = copy.deepcopy(raw)
                norm.transform(s)
                normed.append(s)
            d_full = [sample_to_data(v, cfg.lape_k, cfg.rwse_steps,
                                     random_dim=cfg.random_dim,
                                     branch_random_dim=cfg.branch_random_dim)
                      for v in normed]
            d_ct = [sample_to_data(cgen_view(v), cfg.lape_k, cfg.rwse_steps,
                                   random_dim=cfg.random_dim,
                                   branch_random_dim=cfg.branch_random_dim)
                    for v in normed]
            bf = Batch.from_data_list(d_full).to(dev)
            bc = Batch.from_data_list(d_ct).to(dev)
            hf, _g1, _ef = enc(bf)
            _hc, _g2, ec = enc(bc)
            hf, ec = hf.cpu(), ec.cpu()          # 落 CPU 缓存(6640 样本 ~4GB 显存会 OOM)
            # 累积偏移切片(G4 多家族混规模批;均一规模时与旧 j*n 切片逐位同款)
            n_off = e_off = 0
            for j, (raw, s_n) in enumerate(zip(chunk, normed)):
                nj = raw.n_nodes
                mj = d_full[j].edge_index.shape[1]
                cache.append({
                    "h": hf[n_off:n_off + nj].clone(),
                    "e": ec[e_off:e_off + mj].clone(),
                    "ei": d_full[j].edge_index.clone(),
                    "target": torch.from_numpy(s_n.node_feat.copy()),
                    "lm": torch.from_numpy(_loss_mask(raw)),
                    "fam": raw.family_id,
                })
                n_off += nj
                e_off += mj
        return cache
    t0 = time.time()
    tr_cache = precompute(train_s)
    va_cache = precompute(val_s)
    print(f"[g15] 潜码缓存完成({time.time() - t0:.0f}s);"
          f"h[{tr_cache[0]['h'].shape}] e[{tr_cache[0]['e'].shape}]")

    def assemble(items, sigmas=None):
        """拼 batch(edge_index 加节点偏移);sigmas 非 None 时对 h 加相对尺度噪声。"""
        hs, es, eis, tgs, lms = [], [], [], [], []
        off = 0
        for k, it in enumerate(items):
            h = it["h"]
            if sigmas is not None and sigmas[k] > 0:
                h = h + sigmas[k] * h.std(dim=0, keepdim=True) * torch.randn_like(h)
            hs.append(h)
            es.append(it["e"])
            eis.append(it["ei"] + off)
            tgs.append(it["target"])
            lms.append(it["lm"])
            off += h.shape[0]
        return (torch.cat(hs).to(dev), torch.cat(es).to(dev),
                torch.cat(eis, dim=1).to(dev), torch.cat(tgs).to(dev),
                torch.cat(lms).to(dev))

    hidden = a.hidden or (2 * cfg.d)
    dec = ReadoutDecoder(cfg.d, n_layers=a.layers, ffn_mult=a.ffn_mult, hidden=hidden).to(dev)
    print(f"[g15] ReadoutDecoder 参数 {dec.num_params()/1e6:.2f}M"
          f"({a.layers} 层 MPNN,ffn×{a.ffn_mult},head hidden {hidden})")
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, weight_decay=1e-4)

    @torch.no_grad()
    def eval_cache(cache):
        dec.eval()
        tot = 0.0
        for i in range(0, len(cache), a.batch):
            h, e, ei, tg, lm = assemble(cache[i:i + a.batch])
            pred = dec(h, e, ei)
            tot += float(recon_loss(pred, tg, lm, GROUP_W)) * min(a.batch, len(cache) - i)
        dec.train()
        return tot / len(cache)

    # lib118 上采样权重(对冲稀释;family_id=ieee118)。归一为概率,训练按 p 抽 batch。
    samp_p = None
    fam_w = {}
    if a.upweight_fams:
        for kv in a.upweight_fams.split(","):
            k, _, vv = kv.strip().partition(":")
            if k:
                fam_w[k] = float(vv)
    if a.upweight_lib != 1.0 or fam_w:
        w = np.array([(a.upweight_lib if it.get("fam") == "ieee118" else 1.0)
                      * fam_w.get(it.get("fam"), 1.0) for it in tr_cache], dtype=float)
        samp_p = w / w.sum()
        n_up = int((w > 1.0).sum())
        eff_frac = float(w[w > 1.0].sum() / w.sum()) if n_up else 0.0
        print(f"[g15] 上采样(lib118 x{a.upweight_lib}; 逐族 {fam_w or '无'}):"
              f"{n_up}/{len(tr_cache)} 样本 → 曝光占比 "
              f"{n_up/max(len(tr_cache),1):.1%} 提到 {eff_frac:.1%}")
        for f, ww in sorted(fam_w.items()):
            n_f = sum(1 for it in tr_cache if it.get("fam") == f)
            sh = float(sum(ww for it in tr_cache if it.get("fam") == f) / w.sum()) if n_f else 0.0
            print(f"[g15]   族 {f}: {n_f} 样本 x{ww} → 曝光 {sh:.1%}")

    best, bad, hist = float("inf"), 0, []
    t0 = time.time()
    for step in range(a.steps):
        idx = rng.choice(len(tr_cache), size=a.batch, replace=False, p=samp_p)
        sig = rng.choice(SIGMAS, size=a.batch)
        h, e, ei, tg, lm = assemble([tr_cache[i] for i in idx], sigmas=sig)
        loss = recon_loss(dec(h, e, ei), tg, lm, GROUP_W)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        hist.append(float(loss))
        if (step + 1) % a.eval_every == 0:
            v = eval_cache(va_cache)
            print(f"[g15] step {step+1}/{a.steps} train {np.mean(hist[-a.eval_every:]):.4f}"
                  f" val {v:.4f} best {best:.4f}", flush=True)
            if v < best - 1e-5:
                best, bad = v, 0
                torch.save({"decoder_state": dec.state_dict(),
                            "config": {"d": cfg.d, "n_layers": a.layers,
                                       "ffn_mult": a.ffn_mult, "hidden": hidden,
                                       "arch": f"readout_mpnn{a.layers}"},
                            "train": {"lib": src_desc, "n_train": len(train_s),
                                      "n_val": len(val_s), "sigmas": list(SIGMAS),
                                      "group_weight": GROUP_W, "seed": a.seed,
                                      "upweight_lib": a.upweight_lib,
                                      "step": step + 1, "val_loss": v,
                                      "corpus_provenance": corpus_prov},
                            "v1_fingerprint": fp_v1},
                           out / "readout.pt")
            else:
                bad += 1
                if bad >= a.patience:
                    print(f"[g15] 早停于 step {step+1}")
                    break
    print(f"[g15] 训练完成 {time.time() - t0:.0f}s;best val {best:.4f}"
          f" → {out / 'readout.pt'}")
    (out / "train_report.json").write_text(json.dumps(
        {"best_val_loss": best, "steps_run": len(hist),
         "train_loss_first_last": [hist[0], hist[-1]],
         "note": "全位置读出监督 + 潜码噪声增广 σ∈{0,0.1,0.2};v1 解码器原样保留"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
