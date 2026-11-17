"""[G] G1-扩散主体训练(方案 6.2-6.4;2026-07-17 用户批准的账:L=8 d=192,
60k 步 batch 16,AdamW 2e-4 + EMA 0.999,DDPM T=1000 cosine ε-pred)。

数据流(6.3②):冻结编码器对 240 训练方式各干两份活——完整编码取节点 token 场 z
(latent_stats_v1 标准化后作扩散状态)、c_topo 视图编码取骨架条件(节点 token 拼接
注入 + 支路 token e_ct 作边上下文)。c_label v0 = lf 档 one-hot+缺失位;条件 dropout
= 整体 15% + 逐字段 20%(本库唯一字段 = lf 档),c_topo 永不丢弃。留出 60 永不入;
训练内 val 24 做早停监控(EMA 权重、固定噪声口径)。

产物:runs/diffusion_g1/denoiser.pt(EMA + 原始权重;记录 v1/normalizer/latent_stats
三指纹)+ train_report.json。编码器/normalizer/v1/readout 检查点全部只读。

  python scripts/g_diffusion_train.py --root ..\\gridfm-data [--steps 60000]
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
import torch.nn.functional as F
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
from gridfm.models.diffusion import DenoiserV0, GaussianDiffusion  # noqa: E402
from gridfm.models.diffusion.denoiser import LF_TIERS, label_vec  # noqa: E402
from gridfm.data import c_label as CL  # noqa: E402
from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.psasp_io.catalog import Catalog  # noqa: E402
from gridfm.schema.assemble import build_sample  # noqa: E402
from gridfm.schema.fields import NODE_TABLE, Group  # noqa: E402
from gridfm.schema.sample import group_bit_index  # noqa: E402
from gridfm.train.masking import COL_GROUP  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402


def _fp(path: Path) -> str:
    return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()


def _base_pl_sum_for_family(root: Path, fid: str) -> float:
    """家族基态总负荷 ΣPl_base(c_label load_level 分母)。synth 取 _base_case,
    ieee118 取 118 golden 夹具(lib118_diverse 的合成源)。"""
    case_dir = (REPO / "tests" / "fixtures" / "ieee118_golden" if fid == "ieee118"
                else Path(root) / "families" / "train" / fid / "_base_case")
    c = PsaspCase(case_dir)
    base_gs = build_sample(c, Catalog(c), sample_id=f"{fid}_base", family_id=fid)
    return CL.base_pl_sum(base_gs)


def _git_head(repo: Path) -> str:
    """代码版本入检查点(审查修复:产物须可对账到提交;有未提交改动如实标 dirty)。"""
    try:
        import subprocess
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                           capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                               capture_output=True, text=True).stdout.strip()
        return (h[:12] + "+dirty") if dirty else h[:12]
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lib-name", default="lib118_diverse")
    ap.add_argument("--v1-dir", default="runs/encoder_v1")
    ap.add_argument("--latent-stats", default="runs/latent_stats/latent_stats.npz")
    ap.add_argument("--expect-latent-version", default=None,
                    help="断言 latent_stats meta.version(如 latent_stats_v2)——审查修复:"
                         "v1/v2 共享编码器+normalizer 指纹,仅凭指纹三验无法拦 v1 顶 v2 误用")
    ap.add_argument("--corpus", choices=("single", "g4"), default="single",
                    help="single=lib118_diverse 单库(v0.x,c_label=lf 档);"
                         "g4=多家族语料 + c_label v1(load_level+online_count)")
    ap.add_argument("--out", default="runs/diffusion_g1_cgen",
                    help="c_gen 去泄漏版新目录;旧 runs/diffusion_g1(泄漏版)留作对照臂")
    ap.add_argument("--test-every", type=int, default=5)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--upweight-fams", default=None,
                    help="[外协包 F] 逐家族上采样权重,格式 fam1:w1,fam2:w2。大网族在语料中只占"
                         "约 6%,不上采样等于不存在;而直接扩语料伤同规模基本盘是本线 08-04 的"
                         "实测教训,上采样是可控替代(与 g15_train_readout 同款机制)。")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    # 条件 dropout(6.3② 档内:整体 10%~20%);单字段库 v0.1 取 10%+10%(复合 ≈19%)
    ap.add_argument("--drop-whole", type=float, default=0.10)
    ap.add_argument("--drop-field", type=float, default=0.10)
    # v0.2 解码感知辅助损失(修生成 v_set 系统性偏低):x0_hat 解码→字段损失,v/v_set 加权,
    # 按 alpha_bar[t] 加权(低 t 高 SNR 主导);解码器冻结,梯度只回去噪器。0=关(v0.1)。
    ap.add_argument("--decode-aware", type=float, default=0.0, help="解码感知损失权重 λ")
    ap.add_argument("--v-weight", type=float, default=6.0, help="解码损失中 v/v_set 列权重")
    ap.add_argument("--readout-ckpt", default="runs/decoder_g15_cgen/readout.pt",
                    help="解码感知用的冻结读出解码器")
    ap.add_argument("--clip-x0", type=float, default=4.0, help="解码感知 x0_hat 夹取(同采样)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true",
                    help="从 out/denoiser.pt(best-ckpt)续训——兜底 2060 间歇 CUDA 崩溃;"
                         "opt 状态重置(Adam 快速回暖);无 ckpt 时等同全新训练")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available())
                       else "cpu")

    # ---- 指纹三验(7.4):latent_stats 与 v1 必须同源 ----
    fp_v1 = _fp(Path(a.v1_dir) / "v1.pt")
    fp_nm = _fp(Path(a.v1_dir) / "normalizer.npz")
    fp_ls = _fp(Path(a.latent_stats))
    stats = np.load(a.latent_stats, allow_pickle=False)
    meta = json.loads(str(stats["meta"]))
    assert meta["v1_fingerprint"] == fp_v1, \
        f"latent_stats 与 v1 不同源:{meta['v1_fingerprint']} != {fp_v1}"
    assert meta["normalizer_fingerprint"] == fp_nm, \
        "latent_stats 与 normalizer 不同源(v1/normalizer 配套强制)"
    # 审查修复(critical g_diffusion_train.py:111):版本断言。v1(single)/v2(g4) 共享
    # 同一冻结编码器与 normalizer,指纹三验对两者都通过——必须显式核对语料版本,
    # 否则 latent_stats_v1 会静默顶替 v2 训练 G4 扩散(升版纪律洞)。
    if a.expect_latent_version is not None:
        assert meta.get("version") == a.expect_latent_version, \
            f"latent_stats 版本 {meta.get('version')} != 期望 {a.expect_latent_version}"
    z_mean = torch.from_numpy(stats["mean"]).to(dev)
    z_std = torch.from_numpy(stats["std"]).to(dev)
    print(f"[gdiff] 指纹:v1 {fp_v1} / normalizer {fp_nm} / latent_stats {fp_ls}"
          f"(version={meta['version']};期望 {a.expect_latent_version or '未指定'})")

    lib = Path(a.root) / a.lib_name
    lf_by_sid = {}
    if a.corpus == "g4":
        from gridfm.data.corpus import g4_train_corpus
        pool, corpus_prov = g4_train_corpus(Path(a.root), a.lib_name, a.test_every)
        val_idx = set(range(0, len(pool), a.val_every))
        train_s = [s for i, s in enumerate(pool) if i not in val_idx]
        val_s = [s for i, s in enumerate(pool) if i in val_idx]
        print(f"[gdiff] G4 语料:{corpus_prov['n_samples']} = "
              f"{corpus_prov['n_synth_families']} 合成家族 + lib118 训练份;"
              f"train {len(train_s)} / val {len(val_s)};指纹 {corpus_prov['fingerprint']}")
    else:
        corpus_prov = None
        solvable = {}
        with open(lib / "manifest.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                lf_by_sid[r["sample_id"]] = float(r["load_factor"])
                solvable[r["sample_id"]] = r["solvable"] == "1"
        samples = [s for s in load_library(lib) if solvable.get(s.sample_id)]
        samples.sort(key=lambda s: lf_by_sid[s.sample_id])
        test_idx = set(range(0, len(samples), a.test_every))
        pool = [s for i, s in enumerate(samples) if i not in test_idx]
        val_idx = set(range(0, len(pool), a.val_every))
        train_s = [s for i, s in enumerate(pool) if i not in val_idx]
        val_s = [s for i, s in enumerate(pool) if i in val_idx]
        print(f"[gdiff] train {len(train_s)} / val {len(val_s)} / 留出 {len(test_idx)}(不碰)")

    ck = torch.load(Path(a.v1_dir) / "v1.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items()
                         if k in {f.name for f in dc_fields(TrainConfig)}})
    enc, _ = build_models(cfg)
    enc.load_state_dict(ck["model_state"])
    enc.eval()
    enc.to(dev)                          # G4:6640 样本编码前向上 GPU(缓存落 CPU 防 OOM)
    z_mean_c, z_std_c = z_mean.cpu(), z_std.cpu()
    norm = Normalizer.load(Path(a.v1_dir) / "normalizer.npz")

    # ---- 条件标签设置 ----
    if a.corpus == "g4":
        fam_ids = sorted({s.family_id for s in train_s + val_s})
        base_pls = {fid: _base_pl_sum_for_family(a.root, fid) for fid in fam_ids}
        label_of = {s.sample_id: CL.recompute(s, base_pls[s.family_id])
                    for s in train_s + val_s}
        label_norm = CL.fit_norm([label_of[s.sample_id] for s in train_s])
        tier_of = None
        print(f"[gdiff] c_label v1:{len(fam_ids)} 家族基态载入;"
              f"load μ={label_norm.load_mean:.3f}±{label_norm.load_std:.3f} "
              f"online_frac μ={label_norm.online_frac_mean:.3f}±{label_norm.online_frac_std:.3f}")
    else:
        tier_of = {v: i for i, v in enumerate(LF_TIERS)}
        label_of, label_norm = None, None

    def _label_missings(n, rng_):
        """条件 dropout → 每样本缺失字段集。single:lf 单字段(whole==field);
        g4:drop-whole 两字段皆缺,否则逐字段独立缺失(load_level/online_count)。"""
        out = []
        for _k in range(n):
            if a.corpus == "g4":
                if rng_.random() < a.drop_whole:
                    out.append({"load_level", "online_count"})
                else:
                    ms = set()
                    if rng_.random() < a.drop_field:
                        ms.add("load_level")
                    if rng_.random() < a.drop_field:
                        ms.add("online_count")
                    out.append(ms)
            else:
                drop = (rng_.random() < a.drop_whole) or (rng_.random() < a.drop_field)
                out.append({"lf"} if drop else set())
        return out

    def _encode(item, missing):
        """→ 8 维标签向量。g4 走 c_label.encode;single 走 lf 档 label_vec。"""
        if a.corpus == "g4":
            return torch.from_numpy(
                CL.encode(item["label"], label_norm, missing=tuple(missing))).to(dev)
        return label_vec(None if missing else item["tier"], device=dev)

    # ---- 解码感知(v0.2):冻结读出解码器 + STATE 列/电压加权 ----
    sbit = group_bit_index(Group.STATE)
    scols = torch.tensor(np.where(COL_GROUP == sbit)[0], device=dev)
    dec_frozen = None
    col_w_dec = None
    if a.decode_aware > 0:
        rk = torch.load(a.readout_ckpt, map_location="cpu", weights_only=False)
        fp_r = rk.get("v1_fingerprint")
        assert fp_r == fp_v1, f"readout ckpt 与 v1 指纹不符:{fp_r} != {fp_v1}"
        dec_frozen = ReadoutDecoder(rk["config"]["d"],
                                    n_layers=rk["config"]["n_layers"]).to(dev)
        dec_frozen.load_state_dict(rk["decoder_state"])
        dec_frozen.eval()
        for p in dec_frozen.parameters():
            p.requires_grad_(False)
        col_w_dec = torch.ones(NODE_TABLE.total_dim, device=dev)
        for name in ("v", "v_set"):
            col_w_dec[NODE_TABLE.offset_of(name)] = a.v_weight
        col_w_dec = col_w_dec[scols]           # 只在 STATE 列上算
        print(f"[gdiff] 解码感知 λ={a.decode_aware} v/v_set 权重={a.v_weight}"
              f"(冻结读出解码器 {a.readout_ckpt};指纹校验通过)")

    # ---- 潜码/条件预计算缓存(冻结编码器一次前向;全部驻显存,~50MB)----
    @torch.no_grad()
    def precompute(ss):
        cache = []
        for i in range(0, len(ss), 8):
            chunk = ss[i:i + 8]
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
            hf, _g1, _ef = enc(Batch.from_data_list(d_full).to(dev))
            hc, _g2, ec = enc(Batch.from_data_list(d_ct).to(dev))
            hf, hc, ec = hf.cpu(), hc.cpu(), ec.cpu()   # 缓存落 CPU(6640 样本 ~5GB 显存会 OOM)
            # 累积偏移切片(G4 混规模批;均一规模时与旧 j*n 逐位同款)
            n_off = e_off = 0
            for j, raw in enumerate(chunk):
                nj = raw.n_nodes
                mj = d_full[j].edge_index.shape[1]
                z0 = (hf[n_off:n_off + nj] - z_mean_c) / z_std_c
                item = {
                    "z0": z0,
                    "ct": hc[n_off:n_off + nj],
                    "ec": ec[e_off:e_off + mj],
                    "ei": d_full[j].edge_index,
                    "fam": raw.family_id,      # [外协包 F] 逐族上采样要用(原实现未存)
                }
                if a.corpus == "g4":
                    item["label"] = label_of[raw.sample_id]
                else:
                    item["tier"] = tier_of[lf_by_sid[raw.sample_id]]
                if a.decode_aware > 0:
                    # 目标 = 真方式 STATE(标准化空间);vmask = STATE 非 missing 行
                    item["target"] = torch.from_numpy(normed[j].node_feat.copy())
                    item["vmask"] = torch.from_numpy(~raw.node_missing[:, sbit])
                n_off += nj
                e_off += mj
                cache.append(item)
        return cache
    t0 = time.time()
    tr, va = precompute(train_s), precompute(val_s)
    print(f"[gdiff] 缓存完成({time.time() - t0:.0f}s);"
          f"z0[{tr[0]['z0'].shape}] ec[{tr[0]['ec'].shape}]")

    def assemble(items, missings):
        """missings[k] = 第 k 样本要遮蔽的字段集(条件 dropout / CFG)。CPU 缓存逐 batch 上 dev。"""
        zs, cts, ecs, eis, labs, npg, tgs, vms = [], [], [], [], [], [], [], []
        off = 0
        for it, miss in zip(items, missings):
            zs.append(it["z0"])
            cts.append(it["ct"])
            ecs.append(it["ec"])
            eis.append(it["ei"] + off)
            labs.append(_encode(it, miss))
            npg.append(it["z0"].shape[0])
            if a.decode_aware > 0:
                tgs.append(it["target"])
                vms.append(it["vmask"])
            off += it["z0"].shape[0]
        target = torch.cat(tgs).to(dev) if tgs else None
        vmask = torch.cat(vms).to(dev) if vms else None
        return (torch.cat(zs).to(dev), torch.cat(cts).to(dev), torch.cat(ecs).to(dev),
                torch.cat(eis, dim=1).to(dev), torch.stack(labs),
                torch.tensor(npg, dtype=torch.long, device=dev), target, vmask)

    diff = GaussianDiffusion(a.T).to(dev)
    net = DenoiserV0(cfg.d, n_layers=a.layers).to(dev)
    ema = copy.deepcopy(net)
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"[gdiff] DenoiserV0 L={a.layers} d={cfg.d}:{net.num_params()/1e6:.2f}M 参数")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)

    @torch.no_grad()
    def ema_update():
        for pe, pn in zip(ema.parameters(), net.parameters()):
            pe.lerp_(pn, 1.0 - a.ema)
        for be, bn in zip(ema.buffers(), net.buffers()):
            be.copy_(bn)

    @torch.no_grad()
    def eval_val():
        """EMA 权重、固定噪声/固定 t(逐次评估可比)。条件口径 = 全条件。
        decode_aware>0 时早停指标 = εMSE + λ·解码损失(与训练目标一致,反映电压修复)。"""
        g = torch.Generator(device="cpu").manual_seed(1234)
        tot, cnt = 0.0, 0
        for i in range(0, len(va), a.batch):
            items = va[i:i + a.batch]
            z0, ct, ec, ei, lab, npg, target, vmask = assemble(
                items, [set() for _ in items])          # 全条件(无遮蔽)
            t = torch.randint(1, a.T + 1, (len(items),), generator=g).to(dev)
            t_node = t.repeat_interleave(npg)
            eps = torch.randn(z0.shape, generator=g).to(dev)
            z_t = diff.q_sample(z0, t_node, eps)
            pred = ema(z_t, ct, ec, ei, t, lab, npg)
            batch_loss = float(((pred - eps) ** 2).mean())
            if a.decode_aware > 0:
                ab_node = diff.alpha_bar[t_node][:, None]
                x0_hat = ((z_t - (1 - ab_node).sqrt() * pred)
                          / ab_node.sqrt()).clamp(-a.clip_x0, a.clip_x0)
                fields = dec_frozen(x0_hat * z_std + z_mean, ec, ei)
                per_node = (F.huber_loss(fields[:, scols], target[:, scols],
                                         delta=1.0, reduction="none") * col_w_dec).sum(1)
                w_node = ab_node.squeeze(1) * vmask.float()
                batch_loss += a.decode_aware * float(
                    (per_node * w_node).sum() / w_node.sum().clamp(min=1.0))
            tot += batch_loss * len(items)
            cnt += len(items)
        return tot / cnt

    # [外协包 F] 逐族上采样概率(不传 --upweight-fams 时 samp_p=None,与原实现逐位一致)
    samp_p = None
    if a.upweight_fams:
        _fw = {}
        for kv in a.upweight_fams.split(","):
            k, _, vv = kv.strip().partition(":")
            if k:
                _fw[k] = float(vv)
        _w = np.array([_fw.get(it.get("fam"), 1.0) for it in tr], dtype=float)
        samp_p = _w / _w.sum()
        for _f, _ww in sorted(_fw.items()):
            _n = sum(1 for it in tr if it.get("fam") == _f)
            _share = float(_ww * _n / _w.sum()) if _n else 0.0
            print(f"[diff] 上采样 族 {_f}: {_n} 样本 x{_ww} → 曝光 {_share:.1%}", flush=True)

    best, bad, hist = float("inf"), 0, []
    start_step = 0
    if a.resume and (out / "denoiser.pt").exists():
        rk = torch.load(out / "denoiser.pt", map_location=dev, weights_only=False)
        net.load_state_dict(rk["raw_state"]); ema.load_state_dict(rk["ema_state"])
        start_step = int(rk["train"]["step"]); best = float(rk["train"]["val_ema_mse"])
        # [外协包 F,2026-08-28] 释放检查点字典的 GPU 张量引用:load_state_dict 已拷贝完,
        # rk 若存活到训练循环会多驻留一份权重并改变分配序,实测诱发 WDDM 换页(1.51s/步)
        del rk; torch.cuda.empty_cache()
        print(f"[gdiff] 续训:从 step {start_step} best {best:.4f} 恢复"
              f"(opt 重置,兜底 2060 间歇崩溃)", flush=True)
    step = start_step - 1
    t0 = time.time()
    for step in range(start_step, a.steps):
        idx = rng.choice(len(tr), size=a.batch, replace=False, p=samp_p)
        items = [tr[i] for i in idx]
        # 条件 dropout(6.3②):整体 drop-whole 与逐字段 drop-field(g4 两字段各自独立)
        missings = _label_missings(a.batch, rng)
        z0, ct, ec, ei, lab, npg, target, vmask = assemble(items, missings)
        t = torch.randint(1, a.T + 1, (a.batch,), device=dev)
        t_node = t.repeat_interleave(npg)
        eps = torch.randn_like(z0)
        z_t = diff.q_sample(z0, t_node, eps)
        eps_pred = net(z_t, ct, ec, ei, t, lab, npg)
        loss_eps = ((eps_pred - eps) ** 2).mean()
        loss = loss_eps
        loss_dec_val = 0.0
        if a.decode_aware > 0:
            ab_node = diff.alpha_bar[t_node][:, None]        # [ΣN,1]
            x0_hat = ((z_t - (1 - ab_node).sqrt() * eps_pred)
                      / ab_node.sqrt()).clamp(-a.clip_x0, a.clip_x0)
            fields = dec_frozen(x0_hat * z_std + z_mean, ec, ei)   # [ΣN,25]
            per_node = (F.huber_loss(fields[:, scols], target[:, scols],
                                     delta=1.0, reduction="none") * col_w_dec).sum(1)
            w_node = ab_node.squeeze(1) * vmask.float()      # SNR × STATE 有效行
            loss_dec = (per_node * w_node).sum() / w_node.sum().clamp(min=1.0)
            loss = loss_eps + a.decode_aware * loss_dec
            loss_dec_val = float(loss_dec)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ema_update()
        hist.append(float(loss_eps))
        if (step + 1) % a.eval_every == 0:
            v = eval_val()
            el = time.time() - t0
            print(f"[gdiff] step {step+1}/{a.steps}"
                  f" εMSE {np.mean(hist[-a.eval_every:]):.4f}"
                  f"{f' dec {loss_dec_val:.4f}' if a.decode_aware>0 else ''}"
                  f" valEMA {v:.4f} best {best:.4f} ({el:.0f}s)", flush=True)
            if v < best - 1e-5:
                best, bad = v, 0
                torch.save({
                    "ema_state": ema.state_dict(),
                    "raw_state": net.state_dict(),
                    "config": {"d": cfg.d, "n_layers": a.layers, "T": a.T,
                               "arch": "adaln_mpnn_v0",
                               "label_schema": ("c_label_v1:load_level+online_count"
                                                if a.corpus == "g4"
                                                else "lf_tier_onehot3+missing1+reserve4"),
                               "label_norm": (label_norm.to_dict()
                                              if a.corpus == "g4" else None),
                               "lf_tiers": list(LF_TIERS),
                               "drop_whole": a.drop_whole,
                               "drop_field": a.drop_field,
                               "decode_aware": a.decode_aware,
                               "v_weight": a.v_weight if a.decode_aware > 0 else None,
                               "clip_x0": a.clip_x0},
                    "fingerprints": {"v1": fp_v1, "normalizer": fp_nm,
                                     "latent_stats": fp_ls,
                                     "latent_version": meta.get("version"),
                                     "corpus_provenance": meta.get("corpus_provenance"),
                                     # C-1(2026-08-03 第十任审查):上行是 latent_stats
                                     # 采集时的语料出处,非本次训练语料;真值在下行
                                     "train_corpus_provenance": corpus_prov,
                                     # [外协包 F] lib118_diverse 未随包,缺库时记 None
                                     # (原实现无条件取指纹 → 训练跑完却在写报告时崩)
                                     "lib_manifest": (_fp(lib / "manifest.csv")
                                                      if (lib / "manifest.csv").is_file() else None)},
                    "git_head": _git_head(REPO),
                    "train": {"lib": lib.name, "n_train": len(train_s),
                              "n_val": len(val_s), "seed": a.seed,
                              "split": f"lf 排序 test_every={a.test_every}"
                                       f" 留出不入,val_every={a.val_every}",
                              "step": step + 1, "val_ema_mse": v},
                }, out / "denoiser.pt")
            else:
                bad += 1
                if bad >= a.patience:
                    print(f"[gdiff] 早停于 step {step+1}")
                    break
    el = time.time() - t0
    print(f"[gdiff] 训练完成 {el:.0f}s;best valEMA {best:.4f} → {out / 'denoiser.pt'}")
    (out / "train_report.json").write_text(json.dumps(
        {"best_val_ema_mse": best, "steps_run": step + 1, "elapsed_s": round(el, 1),
         "train_loss_first_last": ([hist[0], hist[-1]] if hist else [None, None]),
         "git_head": _git_head(REPO),
         "latent_version": meta.get("version"),
         "corpus_provenance": meta.get("corpus_provenance"),
         "train_corpus_provenance": corpus_prov,
         "basis": ("lib118_diverse 单家族([内],D14 单拓扑豁免;不作泛化主张)"
                   if meta.get("version") == "latent_stats_v1"
                   else f"G4 多家族语料({meta.get('version')};跨家族真泛化口径)"),
         "note": "ε-MSE;val 用 EMA 权重+固定噪声;条件 dropout 整体 15%+逐字段 20%;"
                 f"z={meta.get('version')} 标准化节点 token 场;启停注记见 denoiser.py"},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
