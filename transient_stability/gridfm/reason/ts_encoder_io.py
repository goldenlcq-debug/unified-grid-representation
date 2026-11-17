"""判稳 run 的**编码器装载口**(R2 编码器解冻起必需;2026-07-25)。

**为什么必须集中**:R2 之前判稳线所有下游消费方都写死"从 runs/encoder_v1/v1.pt 重建
编码器"——因为编码器自预训练冻结至今、每个 run 都配同一套权重。R2 起 run 可以带**自己的**
编码器(任务分叉副本),此时下游若仍从 v1.pt 重建,就会把"best 步推理器 + 冻结编码器"
拼在一起打分:**数字仍然出得来,而且看着像"解冻有害"**——这是本轮最贵的假阴性来源。

**两个假凭据(2026-07-25 核实,勿用于判断是否解冻)**:
1. `ts_report.json` 的 `encoder` 字段写的是 `--encoder` 的值(默认 `"v1"`),解冻后**照写
   "v1"**。故真凭据 = 新键 `unfreeze_encoder`(报告)/ `unfreeze`(检查点)。
2. `ts_screen.py` 里 `assert _blake(v1.pt) == FP_V1` 断的是**共享资产未被覆写**(铁律),
   与"这个 run 该配哪个编码器"无关,解冻后照样绿灯。

**拒绝运行的两种错配**(响亮失败优于静默出数):
- 报告标解冻臂(`unfreeze_encoder != none`)但检查点里没有 `encoder_state`;
- 报告与检查点的臂名不一致。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from gridfm.train.pretrain import build_models

UNFREEZE_CHOICES = ("none", "last2", "all")

# train()/eval() 会改变数值行为的模块类型(有则 enc.train() 不是无害操作,须在评估前切回)
_TRAIN_SENSITIVE = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                    nn.AlphaDropout, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                    nn.SyncBatchNorm, nn.InstanceNorm1d, nn.InstanceNorm2d)


def train_mode_sensitive(enc) -> list:
    """返回 enc 内 train/eval 语义不同的子模块名(空 = enc.train() 数值上是 no-op)。

    凭据用:R2 表第 1 条提醒"注意 dropout 语义变化"——本函数把它从"记忆中的断言"
    变成**当场可见的清单**,而不是靠人记得 GridEncoder 有没有 dropout。

    **必须同时看 MultiheadAttention**(2026-07-25 对抗审查):它不是 nn.Dropout 实例,
    但内部带 dropout 概率,train 模式下生效 —— 只按 isinstance(nn.Dropout) 扫会漏报,
    凭据反而变成误导。故对 MHA 与任何带 >0 dropout 属性的模块单独判。"""
    out = []
    for nm, m in enc.named_modules():
        label = nm or "<root>"
        if isinstance(m, _TRAIN_SENSITIVE):
            p = getattr(m, "p", None)
            if isinstance(m, nn.modules.dropout._DropoutNd) and not p:
                continue                       # p=0 的 Dropout 数值上是恒等
            out.append(f"{label}:{type(m).__name__}")
            continue
        d = getattr(m, "dropout", None)         # MultiheadAttention 等:float 属性
        if isinstance(d, (int, float)) and d > 0:
            out.append(f"{label}:{type(m).__name__}(dropout={d})")
    return out


def encoder_trainable(enc, mode: str) -> list:
    """按解冻臂名返回 [(参数名, 参数)]。

    - `none` :空(历史行为,编码器全冻)
    - `last2`:末两层 GPS + `ln_out`(直接喂推理器的读出 LayerNorm)
    - `all`  :全部编码器参数(含 node/edge 投影与 global_token)
    """
    if mode == "none":
        return []
    if mode == "all":
        return list(enc.named_parameters())
    if mode == "last2":
        n = len(enc.layers)
        pref = tuple(f"layers.{i}." for i in range(max(0, n - 2), n))
        return [(nm, p) for nm, p in enc.named_parameters()
                if nm.startswith(pref) or nm.startswith("ln_out.")]
    raise ValueError(f"未知解冻臂 {mode!r}(可选 {UNFREEZE_CHOICES})")


def apply_unfreeze(enc, mode: str, verbose: bool = True) -> list:
    """就地设置 requires_grad 与 train/eval 模式;返回可训 [(名, 参数)]。**凭据 = 打印**。"""
    if mode not in UNFREEZE_CHOICES:
        raise ValueError(f"未知解冻臂 {mode!r}(可选 {UNFREEZE_CHOICES})")
    for p in enc.parameters():
        p.requires_grad_(False)
    tr = encoder_trainable(enc, mode)
    for _nm, p in tr:
        p.requires_grad_(True)
    enc.eval() if mode == "none" else enc.train()
    if verbose:
        tot = sum(p.numel() for p in enc.parameters())
        ntr = sum(p.numel() for _nm, p in tr)
        sens = train_mode_sensitive(enc)
        mods = sorted({nm.rsplit(".", 1)[0] if "." in nm else nm for nm, _p in tr})
        print(f"[enc] 解冻臂={mode} 模式={'train' if enc.training else 'eval'};"
              f"**编码器可训参数 {ntr}/{tot}**({ntr / max(tot, 1):.1%});"
              f"张量 {len(tr)} 个,子模块 {len(mods)} 个")
        if mods:
            print(f"[enc]   可训子模块:{', '.join(mods)}")
        print(f"[enc]   train/eval 敏感子模块(dropout/BN)= "
              f"{sens if sens else '无(train() 数值上等价 eval())'}")
    return tr


def encoder_fingerprint(enc_or_state) -> str:
    """编码器**权重**指纹(blake2b(16),按参数名排序拼字节)。

    与"来源打印"互补:来源只说**从哪个文件装的**,指纹说**装进去的到底是哪套权重**。
    训练末把它写进 ts_report.encoder_state_fingerprint,下游装完复算比对 —— 编码器错配
    从此是机制上不可能,而不是靠约定。"""
    sd = enc_or_state.state_dict() if hasattr(enc_or_state, "state_dict") else enc_or_state
    h = hashlib.blake2b(digest_size=16)
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode("utf-8"))
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def encoder_drift(enc, ref_state) -> dict:
    """编码器相对参考权重(v1)的**漂移量**——"解冻到底动了多少"的凭据。

    没有这一行就无法区分两种完全不同的结局:①编码器几乎没动(1e-5 × 4000 步太小 /
    梯度被裁没了)→ 结论只能是"这个配方下解冻没发生",**不能**说"解冻结构性无效";
    ②编码器动了很多但指标没涨 → 才轮得到"解冻对判稳无用"这个结论。
    返回相对 L2(‖Δ‖/‖ref‖)与漂移最大的若干张量。"""
    sd = enc.state_dict()
    num = den = 0.0
    per = []
    for k, ref in ref_state.items():
        if k not in sd:
            continue
        d = (sd[k].detach().cpu().float() - ref.detach().cpu().float())
        n2, r2 = float((d ** 2).sum()), float((ref.detach().cpu().float() ** 2).sum())
        num += n2
        den += r2
        if r2 > 0:
            per.append((k, (n2 / r2) ** 0.5))
    per.sort(key=lambda kv: -kv[1])
    return {"rel_l2_overall": round((num / den) ** 0.5 if den else 0.0, 6),
            "top_drifted_tensors": [{"name": k, "rel_l2": round(v, 6)} for k, v in per[:8]],
            "n_tensors_moved": sum(1 for _k, v in per if v > 0)}


def run_unfreeze_arm(rep: dict) -> str:
    """报告里的**真**解冻臂(`encoder` 字段是假凭据,见模块 docstring)。"""
    return str(rep.get("unfreeze_encoder", "none"))


def _load_periodic_ckpt(run: Path, at_step: int, dev):
    """载入**非 best** 的周期检查点(同步数配对出数用),并当场核对 step。

    为什么要单开一个口:`best.pt` 是"各座按自己早停点"的产物,拿它比两臂就同时混进了
    "停点不同"这个污染(R4 §2 实测:118 acc 端点 +3.67pp,同步数只剩 +0.13pp)。
    要在**同一 step** 上比,必须能按步号装检查点 —— 而周期检查点里 `reasoner_state` 与
    `encoder_state` 是同一步落盘的,**天然配对**,不会重演 R2 第 4 条那个
    "best 步推理器 + 末步编码器"的错配。
    """
    d = run / "ckpt"
    p = d / f"step_{int(at_step):06d}.pt"
    if not p.exists():
        avail = sorted(int(q.stem.split("_")[1]) for q in d.glob("step_*.pt"))
        near = [s for s in avail if abs(s - int(at_step)) <= 2000]
        raise SystemExit(
            f"[enc] ✋ {run.name}:无 step {at_step} 的检查点({p.name} 不存在)——"
            f"该座共 {len(avail)} 个周期检查点"
            + (f",范围 {avail[0]}..{avail[-1]}" if avail else "")
            + (f";附近可用 {near}" if near else ";附近无可用步点(多半是 --keep-last 剪掉了)")
            + "。**不要用 best.pt 顶替**:那就退回端点口径、把停点污染放回来了")
    best = torch.load(p, map_location=dev or "cpu", weights_only=False)
    got = int(best.get("step", -1))
    if got != int(at_step):
        raise SystemExit(
            f"[enc] ✋ {run.name}:{p.name} 里记的 step={got} ≠ 请求的 {at_step} —— "
            f"文件名与内容不同源,拒绝出数")
    return best, p


def build_encoder_for_run(run, ck, cfg, dev=None, best=None, rep=None, verbose=True,
                          at_step=None):
    """按 run 的实际出身装编码器,并**打印来源**。返回 (enc, rep, best, source)。

    参数:
      run  —— run 目录(含 ts_report.json 与 ckpt/best.pt)
      ck   —— 已 load 的 runs/encoder_v1/v1.pt(共享冻结基座,永不覆写)
      cfg  —— TrainConfig(自 v1.pt 的 config 还原)
      best —— 可选,已 load 的 best.pt(免二次 IO);None 则本函数载入
      at_step —— 可选,**改装该步的周期检查点**而不是 best.pt(同步数配对出数用)。
        给了它,指纹门的**基准会从"报告里那条(= best 步)"改挂到"该检查点自身的
        encoder_state"** —— 否则解冻臂必然报错配(报告记的是 best 步的权重)。
        为补回这一点松动,解冻臂另加一条"装出来的编码器不得与冻结 v1 逐位相同"的硬检查。
    """
    run = Path(run)
    if rep is None:
        rep = json.loads((run / "ts_report.json").read_text(encoding="utf-8"))
    ck_path = "ckpt/best.pt"
    if at_step is not None:
        if best is not None:
            raise SystemExit("[enc] ✋ best 与 at_step 不可同时给(会装错一套权重)")
        best, _p = _load_periodic_ckpt(run, at_step, dev)
        ck_path = f"ckpt/{_p.name}"
    elif best is None:
        best = torch.load(run / "ckpt" / "best.pt",
                          map_location=dev or "cpu", weights_only=False)
    arm = run_unfreeze_arm(rep)
    ck_arm = str(best.get("unfreeze", "none"))
    if arm != ck_arm:
        raise SystemExit(
            f"[enc] ✋ {run.name}:报告解冻臂 {arm!r} 与检查点 {ck_arm!r} 不一致——"
            f"报告与权重不同源,拒绝出数")
    if arm != "none" and "encoder_state" not in best:
        raise SystemExit(
            f"[enc] ✋ {run.name}:报告标解冻臂 {arm!r} 但 {ck_path} 无 encoder_state——"
            f"若继续将用**冻结 v1 编码器 + 微调后推理器**打分(错配,系统性偏低,"
            f"会伪造'解冻有害');拒绝运行")
    if arm == "none" and "encoder_state" in best:
        raise SystemExit(
            f"[enc] ✋ {run.name}:报告标冻结臂但 {ck_path} 带 encoder_state——口径不明,拒绝出数")

    enc, _dec = build_models(cfg)
    if arm != "none":
        enc.load_state_dict(best["encoder_state"])
        src = (f"run ckpt({run.name}/{ck_path} step {best.get('step')},解冻臂 {arm})")
    elif rep.get("encoder") == "v1":
        enc.load_state_dict(ck["model_state"])
        src = "v1.pt(共享冻结基座)"
    else:                       # L3 对照组:冻结随机初始化(同 seed 复现)
        torch.manual_seed(int(run.name.rsplit("_s", 1)[1]))
        enc, _dec = build_models(cfg)
        src = f"随机初始化(encoder={rep.get('encoder')!r},seed={run.name.rsplit('_s', 1)[1]})"
    enc.eval()
    # **权重指纹复算**:报告里若有 encoder_state_fingerprint,装完必须对得上。
    # 这把"编码器错配"从"约定不会发生"变成"机制上发生即停"。
    want = rep.get("encoder_state_fingerprint")
    basis = "ts_report.encoder_state_fingerprint"
    if at_step is not None and arm != "none":
        # 报告里那条指纹是 **best 步**的权重,拿它门这一步必然误判为"错配"。
        # 基准改挂**该检查点自身的 encoder_state**——门仍是机制门(验 load_state_dict
        # 真的装进去了、键/形状没错),只是基准换了源,并在下面补一条"确实不是 v1"。
        want = encoder_fingerprint(best["encoder_state"])
        basis = f"{ck_path} 自身的 encoder_state"
    got = encoder_fingerprint(enc) if want else None
    if want and got != want:
        raise SystemExit(
            f"[enc] ✋ {run.name}:装载后的编码器权重指纹 {got} ≠ {basis} 记录的 {want} —— "
            f"**编码器错配**(报告的数字出自另一套权重),拒绝出数")
    if at_step is not None and arm != "none" and got == encoder_fingerprint(ck["model_state"]):
        raise SystemExit(
            f"[enc] ✋ {run.name}:{ck_path} 的 encoder_state 与冻结 v1 **逐位相同** —— "
            f"标着解冻臂却没动过编码器,该步检查点不可信,拒绝出数")
    if dev is not None:
        enc.to(dev)
    if verbose:
        print(f"[enc] {run.name}:**编码器来源 = {src}**"
              + (f";权重指纹 {got}(与 {basis} 一致 ✓)" if want else ""), flush=True)
    return enc, rep, best, src
