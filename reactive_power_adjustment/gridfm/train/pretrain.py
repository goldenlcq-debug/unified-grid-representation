"""预训练循环(方案 T5.4 / 3.4)。掩码重构 + 全局物理量(占位)辅助;检查点原子写入 + resume。

训练配方(方案 10.2 初值表):AdamW(wd 0.01) / lr 3e-4 / **cosine + 5% warmup** / grad clip 1.0 /
早停 = 验证掩码损失连续 N 次 eval 无改善。

每步从样本池**随机不放回采样** batch_graphs 个图(2026-07-09 修复:此前固定取前 batch_graphs 个,
等价于只在极小子集上训练,损失趋零纯属记忆)。

在线增广(方案 3.2):①LaPE 逐列 ±1 符号翻转;②随机特征通道重采样。
验证/推理一律关闭增广,用持久化随机特征 + 确定性定号,保证逐位可复现(7.4)。

检查点字段(10.5):model/decoder/optimizer/scaler state、global_step、rng(numpy 全局/Generator/torch)、
早停状态、config、schema_version。原子保存:先写 .tmp 再 os.replace。
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from ..data.graph import edge_input_dim, feature_slices, node_input_dim, sample_to_data
from ..models.decoder.recon import COL_IS_BCE_V03, GROUP_RANGES_V03, ReconDecoder, recon_loss
from ..models.encoder.gps import GridEncoder
from ..models.heads.tag_recon import TagReconHead
from ..schema.fields import BRANCH_TABLE, Group, NODE_TABLE, NODE_TABLE_V03, SCHEMA_VERSION
from ..schema.sample import GridSample, group_bit_index
from .masking import MaskConfig, apply_node_mask

# 验证集掩码固定种子偏移:每次 eval 复用同一批掩码 → 验证损失跨 eval 可比(方案 8.4"同一随机种子掩码协议")
_VAL_MASK_SEED_OFFSET = 10_007
# 支路四流列(解算潮流 pi/qi/pj/qj;泄漏修复线分叉课程的遮蔽面,mark 不在内)
_BRANCH_FLOW_COLS = tuple(BRANCH_TABLE.offset_of(n) for n in ("pi", "qi", "pj", "qj"))


@dataclass
class TrainConfig:
    d: int = 48
    n_layers: int = 3
    n_global: int = 8
    n_heads: int = 4
    ffn_mult: int = 2
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    batch_graphs: int = 4
    max_steps: int = 30
    seed: int = 0
    lape_k: int = 16
    rwse_steps: int = 16
    random_dim: int = 8
    branch_random_dim: int = 8
    device: str = "cpu"          # 'cpu' | 'cuda'(2060 用 fp16)
    amp_dtype: str = "none"      # 'none' | 'float16' | 'bfloat16'
    # schedule(方案 10.2)
    warmup_frac: float = 0.05
    lr_min_frac: float = 0.0     # cosine 末端 lr = lr × lr_min_frac
    # 验证 + 早停(方案 10.2)
    eval_every: int = 0          # 0 = 不做验证(需 val_samples 才生效)
    early_stop_patience: int = 20
    early_stop_min_delta: float = 0.0
    # 在线增广(方案 3.2)
    aug_lape_sign_flip: bool = True
    aug_random_resample: bool = True
    # D19(方案 v2.1 §3.4.3):A1 标签重构辅助权重(0=关;操作带标定起扫 0.05);
    # A2 随机通道投影独立化 + 排除 weight decay + 可学习增益(结构开关,重训才可改变)
    tag_recon_weight: float = 0.0
    rand_proj_split: bool = False
    # 潮流定向缺失课程档概率(方案 3.4.1 字段级实现,2026-07-11 用户批准实验)。
    # **只作用于训练步**(augment=True);验证/门禁掩码流一律 MaskConfig() 默认,逐位不变
    mask_pf_typed: float = 0.0
    # 连通片 v/theta 课程档概率(v2 S2 新目标,2026-07-16 用户点火;同上只作用于训练步)
    mask_region: float = 0.0
    # 泄漏修复线分叉课程(预注册 docs/预注册-泄漏修复线-解析基线与支路遮蔽分叉-20260827.md §3.2):
    # 以此概率对整样本遮蔽支路四流列(pi/qi/pj/qj 置 0 + branch_masked STATE 位;mark 不遮,
    # 只遮不重构 —— 解码器与损失不动)。默认 0 = V1 路径逐位不变;p≥1 短路零 rng 消耗 →
    # 节点掩码流亦逐位不变。只作用于训练步(augment=True),验证/门禁掩码流照旧。
    mask_branch_state: float = 0.0
    # §8.4 阶梯 L3 / 5.3 组2:编码器保持随机初始化并冻结,只训解码器(+辅助头)。
    # 剥离"预训练所学"与"架构先验";不用于任何保留性产物(D11 只认 P1-全量验收冻结)
    freeze_encoder: bool = False
    # 编码器 v2(A1,2026-08-06):"v0"(默认)= v1 栈全默认参路径,逐位不变红线;
    # "v0.3" = 43 节点列(NODE_TABLE_V03),语料须 load_npz(full_schema=True) 加载,
    # 宽度不符在 apply_node_mask 断言处响亮报错
    schema: str = "v0"
    # 编码器梯度检查点(A5 前置件,2026-08-09):万节点单图显存件;默认 False =
    # v1 逐位不变(只在训练态生效,eval/no_grad 恒走原路径,test_grad_checkpoint 锁定)
    grad_checkpoint: bool = False
    group_weight: Dict[str, float] = field(
        default_factory=lambda: {"topology": 1.0, "static": 1.0, "dynamic": 1.0, "state": 2.0}
    )


def _schema_spec(schema: str):
    """schema → (节点表, 解码 group_ranges, 逐列 BCE 表)。

    v0 返回 (NODE_TABLE, None, None):下游 apply_node_mask/ReconDecoder/recon_loss
    一律收到默认参 → v1 路径逐位不变(A1 红线)。未知值响亮报错,防半截 schema 静默入训。
    """
    if schema == "v0":
        return NODE_TABLE, None, None
    if schema == "v0.3":
        return NODE_TABLE_V03, GROUP_RANGES_V03, COL_IS_BCE_V03
    raise ValueError(f"未知 schema {schema!r}(可选 'v0' | 'v0.3')")


def lr_at(cfg: TrainConfig, step: int) -> float:
    """cosine + warmup(方案 10.2)。step 为 0-based 全局步。

    warmup:step 0 → lr/warm,step warm-1 → lr(峰值)。
    cosine:step warm → lr,step max_steps-1 → lr × lr_min_frac(末步恰好落到地板)。
    """
    total = max(1, cfg.max_steps)
    warm = int(round(cfg.warmup_frac * total))
    if warm > 0 and step < warm:
        return cfg.lr * float(step + 1) / float(warm)
    prog = float(step - warm) / float(max(1, total - warm - 1))
    prog = min(1.0, max(0.0, prog))
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return cfg.lr * (cfg.lr_min_frac + (1.0 - cfg.lr_min_frac) * cos)


def _masked_data(sample: GridSample, cfg: TrainConfig, rng: np.random.Generator,
                 augment: bool = True) -> Data:
    """对已归一化样本施掩码,构造带 target/loss_mask 的 Data。augment=False 时为确定性编码。

    掩码课程扩展(mask_pf_typed)只在训练步(augment=True)生效;augment=False(验证/门禁/
    阶梯)一律 MaskConfig() 协议默认——保证评估掩码流跨实验逐位不变(8.9)。
    """
    mcfg = (MaskConfig(p_pf_typed=cfg.mask_pf_typed, p_region=cfg.mask_region)
            if (augment and (cfg.mask_pf_typed > 0 or cfg.mask_region > 0)) else MaskConfig())
    table, _, _ = _schema_spec(cfg.schema)
    masked_feat, target, loss_mask = apply_node_mask(
        sample.node_feat, sample.node_missing, mcfg, rng,
        edge_index=sample.edge_index,
        table=None if cfg.schema == "v0" else table,
    )
    # 泄漏修复线分叉课程(预注册 20260827 §3.2):支路四流列遮蔽,只遮不重构。
    # 默认 mask_branch_state=0 时本块整体短路(V1 路径逐位不变);p≥1 不消耗 rng
    # (节点掩码流与关闭时逐位同);仅 0<p<1 才抽一次(本轮只用 p∈{0,1})。
    bf, bm = sample.branch_feat, sample.branch_masked
    if augment and cfg.mask_branch_state > 0 and (
            cfg.mask_branch_state >= 1.0 or rng.random() < cfg.mask_branch_state):
        bf = bf.copy()
        bm = bm.copy()
        for c in _BRANCH_FLOW_COLS:
            bf[:, c] = 0.0
        bm[:, group_bit_index(Group.STATE)] = True
    masked = GridSample(
        node_feat=masked_feat, branch_feat=bf, edge_index=sample.edge_index,
        node_missing=sample.node_missing, node_masked=sample.node_masked,
        branch_missing=sample.branch_missing, branch_masked=bm,
        node_random=sample.node_random, branch_random=sample.branch_random,
        sample_id=sample.sample_id, family_id=sample.family_id,
    )
    # 随机特征(节点+支路):默认用样本持久化值;训练期按方案 3.2 做重采样增广
    rf = sample.random_features(cfg.random_dim)
    brf = sample.branch_random_features(cfg.branch_random_dim)
    if augment and cfg.aug_random_resample:
        if cfg.random_dim > 0:
            rf = rng.standard_normal((sample.n_nodes, cfg.random_dim)).astype(np.float32)
        if cfg.branch_random_dim > 0:
            brf = rng.standard_normal((sample.n_branches, cfg.branch_random_dim)).astype(np.float32)

    data = sample_to_data(masked, cfg.lape_k, cfg.rwse_steps,
                          random_dim=cfg.random_dim, random_feat=rf,
                          branch_random_dim=cfg.branch_random_dim, branch_random_feat=brf)
    if augment and cfg.aug_lape_sign_flip and cfg.lape_k > 0:
        sl = feature_slices(sample.node_feat.shape[1], cfg.lape_k, cfg.rwse_steps, cfg.random_dim)
        signs = rng.choice([-1.0, 1.0], size=cfg.lape_k).astype(np.float32)
        data.x[:, sl["lape"]] *= torch.from_numpy(signs)
    data.target = torch.from_numpy(target)
    data.loss_mask = torch.from_numpy(loss_mask)
    if cfg.tag_recon_weight > 0:
        # D19-A1 目标 = 本次前向实际输入的随机标签(增广期为重采样值,eval 期为持久化值);
        # 支路目标按 sample_to_data 的对称化约定复制正反两份
        data.node_rand_target = torch.from_numpy(np.ascontiguousarray(rf))
        data.edge_rand_target = torch.from_numpy(np.concatenate([brf, brf], axis=0))
    return data


def build_models(cfg: TrainConfig):
    table, group_ranges, _ = _schema_spec(cfg.schema)   # v0 → (NODE_TABLE, None, ·) 逐位不变
    enc = GridEncoder(
        node_in=node_input_dim(table.total_dim, cfg.lape_k, cfg.rwse_steps, cfg.random_dim),
        edge_in=edge_input_dim(BRANCH_TABLE.total_dim, cfg.branch_random_dim),
        d=cfg.d, n_layers=cfg.n_layers, n_global=cfg.n_global,
        n_heads=cfg.n_heads, ffn_mult=cfg.ffn_mult,
        node_rand_dim=cfg.random_dim if cfg.rand_proj_split else 0,
        edge_rand_dim=cfg.branch_random_dim if cfg.rand_proj_split else 0,
        grad_checkpoint=cfg.grad_checkpoint,
    )
    dec = ReconDecoder(cfg.d, group_ranges=group_ranges)
    return enc, dec


def build_tag_head(cfg: TrainConfig):
    """D19-A1 辅助头(tag_recon_weight>0 时);训练期专用,不进冻结产物。"""
    if cfg.tag_recon_weight <= 0:
        return None
    return TagReconHead(cfg.d, node_rand_dim=cfg.random_dim, edge_rand_dim=cfg.branch_random_dim)


def build_optimizer(enc, dec, cfg: TrainConfig, tag_head=None):
    """AdamW;D19-A2:随机通道投影 + 增益单列 wd=0 参数组,其余照旧 cfg.weight_decay。

    freeze_encoder(阶梯 L3/组2)时编码器参数不进优化器(解码器/辅助头照常)。"""
    enc_params = [] if cfg.freeze_encoder else list(enc.parameters())
    no_decay = (enc.no_decay_parameters()
                if (enc_params and hasattr(enc, "no_decay_parameters")) else [])
    no_decay_ids = {id(p) for p in no_decay}
    main = [p for p in enc_params + list(dec.parameters())
            if id(p) not in no_decay_ids]
    if tag_head is not None:
        main += list(tag_head.parameters())
    groups = [{"params": main, "weight_decay": cfg.weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=cfg.lr)


def save_checkpoint(path: Path, enc, dec, opt, step: int, cfg: TrainConfig,
                    scaler=None, rng: Optional[np.random.Generator] = None,
                    best_val: float = float("inf"), bad_evals: int = 0,
                    tag_head=None) -> None:
    path = Path(path)
    payload = {
        "model_state": enc.state_dict(),
        "decoder_state": dec.state_dict(),
        "tag_head_state": tag_head.state_dict() if tag_head is not None else None,
        "optimizer_state": opt.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "global_step": step,
        "rng_numpy": np.random.get_state(),
        "rng_gen": rng.bit_generator.state if rng is not None else None,
        "rng_torch": torch.get_rng_state(),
        "best_val": best_val,
        "bad_evals": bad_evals,
        "config": asdict(cfg),
        "schema_version": SCHEMA_VERSION,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # 原子替换


def load_checkpoint_full(path: Path, enc, dec, opt=None, scaler=None,
                         rng: Optional[np.random.Generator] = None, tag_head=None) -> dict:
    """恢复全部可复现状态(含 scaler / Generator / 早停计数),返回原始 ckpt 字典。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    enc.load_state_dict(ckpt["model_state"])
    dec.load_state_dict(ckpt["decoder_state"])
    if tag_head is not None and ckpt.get("tag_head_state") is not None:
        tag_head.load_state_dict(ckpt["tag_head_state"])
    if opt is not None and "optimizer_state" in ckpt:
        opt.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    if rng is not None and ckpt.get("rng_gen") is not None:
        rng.bit_generator.state = ckpt["rng_gen"]
    torch.set_rng_state(ckpt["rng_torch"])
    np.random.set_state(ckpt["rng_numpy"])
    return ckpt


def load_checkpoint(path: Path, enc, dec, opt=None) -> int:
    """兼容旧签名:仅恢复模型/优化器/全局 rng,返回 global_step。"""
    return int(load_checkpoint_full(path, enc, dec, opt)["global_step"])


def _latest_ckpt(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    cks = sorted(ckpt_dir.glob("step_*.pt"))
    return cks[-1] if cks else None


def _forward_loss(enc, dec, batch, cfg: TrainConfig, total_cols: int,
                  dev: torch.device, amp_dtype, use_amp: bool, tag_head=None):
    """返回 (总损失, 掩码重构分量, 标签重构分量或 None)。总损失 = recon + w_tag × tag(D19-A1)。"""
    _, group_ranges, col_is_bce = _schema_spec(cfg.schema)   # v0 → (·, None, None) 逐位不变
    with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
        h, _g, e = enc(batch)
        pred = dec(h, total_cols)
        recon = recon_loss(pred.float(), batch.target.float(), batch.loss_mask, cfg.group_weight,
                           group_ranges=group_ranges, col_is_bce=col_is_bce)
        tag = None
        if tag_head is not None and cfg.tag_recon_weight > 0:
            tag = tag_head.loss(h.float(), e.float(),
                                batch.node_rand_target.float(), batch.edge_rand_target.float())
        loss = recon if tag is None else recon + cfg.tag_recon_weight * tag
    return loss, recon, tag


@torch.no_grad()
def eval_loss(enc, dec, samples: Sequence[GridSample], cfg: TrainConfig,
              dev: Optional[torch.device] = None) -> float:
    """验证集掩码重构损失(确定性:固定掩码种子、关增广)。按被遮蔽元素数加权取均值。"""
    if not samples:
        return float("nan")
    dev = dev or next(enc.parameters()).device
    use_amp = cfg.amp_dtype in ("float16", "bfloat16") and dev.type == "cuda"
    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16
    total_cols = _schema_spec(cfg.schema)[0].total_dim
    was_training = enc.training
    enc.eval(); dec.eval()

    rng_v = np.random.default_rng(cfg.seed + _VAL_MASK_SEED_OFFSET)
    bs = max(1, min(cfg.batch_graphs, len(samples)))
    acc, n_elem = 0.0, 0
    for i in range(0, len(samples), bs):
        chunk = samples[i:i + bs]
        batch = Batch.from_data_list(
            [_masked_data(s, cfg, rng_v, augment=False) for s in chunk]
        ).to(dev)
        # 验证指标 = 纯掩码重构分量(与历史 val 口径可比;D19 辅助项由 8.2/8.3 门禁直接检验)
        _loss, recon, _tag = _forward_loss(enc, dec, batch, cfg, total_cols, dev, amp_dtype, use_amp)
        k = int(batch.loss_mask.sum().item())
        if k > 0:
            acc += float(recon.detach()) * k
            n_elem += k
    if was_training:
        enc.train(); dec.train()
    return acc / max(n_elem, 1)


def run_pretrain(samples: List[GridSample], cfg: TrainConfig,
                 ckpt_dir: Optional[Path] = None, resume: bool = False,
                 save_every: int = 0, return_models: bool = False,
                 val_samples: Optional[List[GridSample]] = None,
                 return_history: bool = False, log_every: int = 0):
    """返回 losses;return_models 追加 (enc, dec);return_history 追加 val_history[(step, loss)]。"""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = torch.device(cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu")
    enc, dec = build_models(cfg)
    enc.to(dev); dec.to(dev)
    tag_head = build_tag_head(cfg)                    # D19-A1(tag_recon_weight>0 时)
    if tag_head is not None:
        tag_head.to(dev)
    opt = build_optimizer(enc, dec, cfg, tag_head)    # D19-A2 参数组在内
    use_amp = cfg.amp_dtype in ("float16", "bfloat16") and dev.type == "cuda"
    amp_dtype = torch.float16 if cfg.amp_dtype == "float16" else torch.bfloat16
    _fp16 = use_amp and cfg.amp_dtype == "float16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=_fp16)      # torch>=2.4 新 API
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=_fp16)

    start = 0
    best_val, bad_evals = float("inf"), 0
    if resume and ckpt_dir is not None:
        last = _latest_ckpt(Path(ckpt_dir))
        if last is not None:
            ck = load_checkpoint_full(last, enc, dec, opt, scaler, rng, tag_head)
            start = int(ck["global_step"])
            best_val = float(ck.get("best_val", float("inf")))
            bad_evals = int(ck.get("bad_evals", 0))

    enc.train(); dec.train()
    if tag_head is not None:
        tag_head.train()
    losses: List[float] = []
    val_history: List[Tuple[int, float]] = []
    total_cols = _schema_spec(cfg.schema)[0].total_dim
    if cfg.freeze_encoder:                      # 阶梯 L3/组2:随机初始化编码器冻结
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
        params = list(dec.parameters())
    else:
        params = list(enc.parameters()) + list(dec.parameters())
    if tag_head is not None:
        params += list(tag_head.parameters())

    n_pool = len(samples)
    if n_pool == 0:
        raise ValueError("样本池为空")
    bs = min(cfg.batch_graphs, n_pool)
    do_eval = bool(cfg.eval_every) and bool(val_samples)

    for step in range(start, cfg.max_steps):
        for pg in opt.param_groups:                       # cosine + warmup
            pg["lr"] = lr_at(cfg, step)
        # 每步随机不放回采样(修复:此前 samples[:bs] 固定前 bs 个)
        idx = rng.choice(n_pool, size=bs, replace=False)
        pool = [_masked_data(samples[int(i)], cfg, rng, augment=True) for i in idx]
        batch = Batch.from_data_list(pool).to(dev)
        opt.zero_grad()
        loss, recon, tag = _forward_loss(enc, dec, batch, cfg, total_cols, dev, amp_dtype,
                                         use_amp, tag_head)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            tag_s = f" tag={float(tag.detach()):.5f}" if tag is not None else ""
            print(f"[train] step {step + 1}/{cfg.max_steps} lr={lr_at(cfg, step):.2e} "
                  f"loss={losses[-1]:.5f} recon={float(recon.detach()):.5f}{tag_s}", flush=True)

        stop = False
        if do_eval and (step + 1) % cfg.eval_every == 0:
            vl = eval_loss(enc, dec, val_samples, cfg, dev)
            val_history.append((step + 1, vl))
            if vl < best_val - cfg.early_stop_min_delta:
                best_val, bad_evals = vl, 0
                if ckpt_dir is not None:
                    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
                    save_checkpoint(Path(ckpt_dir) / "best.pt", enc, dec, opt, step + 1, cfg,
                                    scaler, rng, best_val, bad_evals, tag_head)
            else:
                bad_evals += 1
            if log_every:
                print(f"[eval ] step {step + 1} val={vl:.5f} best={best_val:.5f} "
                      f"bad={bad_evals}/{cfg.early_stop_patience}", flush=True)
            stop = bad_evals >= cfg.early_stop_patience

        if save_every and ckpt_dir is not None and ((step + 1) % save_every == 0 or stop):
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            save_checkpoint(Path(ckpt_dir) / f"step_{step + 1:06d}.pt", enc, dec, opt, step + 1,
                            cfg, scaler, rng, best_val, bad_evals, tag_head)
        if stop:
            print(f"[train] 早停于 step {step + 1}(val 连续 {bad_evals} 次 eval 无改善)", flush=True)
            break

    out: list = [losses]
    if return_models:
        out += [enc, dec]
    if return_history:
        out += [val_history]
    return out[0] if len(out) == 1 else tuple(out)
