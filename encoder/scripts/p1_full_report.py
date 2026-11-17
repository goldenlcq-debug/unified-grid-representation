"""P1-全量(2060 缩小版)多家族混训 + 三门禁报告(方案 8.1/8.2/8.3/8.4;交接清单第 ⑦ 项)。

划分(方案 8.4 / D14,家族级防泄漏):
  train    = 合成训练规模档(150~270) 全部家族 + 真实家族(118,+CEPRI-36 若已接入)
  val [内] = 合成同分布留出家族(150~270)      ← 8.4 分母 / 检索·孪生门禁评估集
  transfer [跨] = 合成留出规模档(400~1000)冻结集 ← 8.4 分子

三门禁(P1-全量 ★,8.1):
  ① 检索(8.2):留出方式 top-1 轨道命中 ≥99%
  ② 孪生(8.3):val 样本上构造严格真孪生,五判据(可分性/跨视图/机制消融)
  ③ 迁移(8.4):状态通道标幺 MAE 退化倍数,V、θ 各 ≤2×
全过 → 编码器按 D11 冻结为 v1;不过 → 归因(容量 or 多样性),这本身就是关键结论。

  E:/anaconda3/envs/grid_gpu/python.exe scripts/p1_full_report.py \
      --root E:/AI-2/统一计算大模型/gridfm-data --lib E:/AI-2/统一计算大模型/gridfm-data/lib118 \
      --lib E:/AI-2/统一计算大模型/gridfm-data/lib_ieee39 --lib E:/AI-2/统一计算大模型/gridfm-data/lib_cepri36 \
      --steps 4000 --device cuda --d 192 --layers 8 --global-tokens 16 --batch 16 --out runs/p1_full
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows GBK 控制台打不出 ✅/❌ 会 UnicodeEncodeError 崩掉末段报告(increment_ladder 实测)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.data.dataset import family_files, load_families, load_library  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.eval.transfer import content_hash, format_report, transfer_report  # noqa: E402
from gridfm.eval.twins import add_branch_twin, add_node_twin, branch_twin_separability, twin_probe  # noqa: E402
from gridfm.probes.retrieval import _valid_node_mask, retrieval_probe  # noqa: E402
from gridfm.schema.fields import NODE_TABLE  # noqa: E402
from gridfm.train.pretrain import TrainConfig, run_pretrain  # noqa: E402


def _gen_bus(sample):
    ei = sample.edge_index
    off = NODE_TABLE.offset_of
    for a in range(sample.n_nodes):
        if sample.node_feat[a, off("is_null")] > 0.5:
            continue
        if int((ei[0] == a).sum() + (ei[1] == a).sum()) >= 2:
            return a
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="数据根(含 families/{train,val,transfer})")
    ap.add_argument("--lib118", default=None, help="真实 118 方式库(并入主训;等价 --lib,历史别名)")
    ap.add_argument("--lib", action="append", default=[],
                    help="真实方式库目录,可重复(lib118/lib_ieee39/lib_cepri36 等,全部并入主训)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--global-tokens", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/p1_full")
    ap.add_argument("--from-ckpt", default=None,
                    help="只跑三门禁:从检查点加载 enc/dec,跳过训练(超参须与检查点一致)")
    ap.add_argument("--tag-recon-weight", type=float, default=0.0,
                    help="D19-A1 标签重构辅助权重(0=关;起扫初值 0.05,只在 val 上标定)")
    ap.add_argument("--rand-no-decay", action="store_true",
                    help="D19-A2:随机通道投影独立化 + 排除 weight decay + 可学习增益")
    ap.add_argument("--pf-typed-frac", type=float, default=0.0,
                    help="潮流定向缺失课程档概率(3.4.1 字段级;只作用于训练步,评估口径不变)")
    ap.add_argument("--resume", action="store_true",
                    help="从 --out/ckpt 最新 step_*.pt 续训(GPU 长跑瞬时故障恢复;10.5)")
    a = ap.parse_args()

    root = Path(a.root)
    train_raw = load_families(root, "train")
    val_raw = load_families(root, "val")
    xfer_raw = load_families(root, "transfer")
    libs = ([a.lib118] if a.lib118 else []) + list(a.lib)
    for lib in libs:                                  # 真实家族并入主训(118/39/CEPRI-36)
        train_raw += load_library(Path(lib))
    fams = lambda ss: sorted({s.family_id for s in ss})  # noqa: E731
    print(f"[p1] train {len(train_raw)}(家族 {len(fams(train_raw))}) / "
          f"val {len(val_raw)}(家族 {len(fams(val_raw))}) / "
          f"transfer {len(xfer_raw)}(家族 {len(fams(xfer_raw))})")
    if not train_raw or not val_raw:
        raise SystemExit("[p1] 缺 train 或 val 家族——先跑 gen_families.py")

    # D14 家族级防泄漏:三集家族**必须**两两不相交(2026-07-09 审查:此前只打印不校验,
    # 目录布局或 lib118 家族名一旦冲突就静默泄漏,门禁全在漏题上认证)。
    ft, fv, fx = set(fams(train_raw)), set(fams(val_raw)), set(fams(xfer_raw))
    leaks = (ft & fv) | (ft & fx) | (fv & fx)
    if leaks:
        raise SystemExit(f"[p1] D14 违规:家族跨集 {sorted(leaks)} —— 检查 families/ 布局与 --lib 家族名")

    # 归一化只在训练集拟合(防泄漏),施加到全部集。
    # --from-ckpt(只跑门禁)时**加载训练期归一化器**而非重拟合(2026-07-10 评审:
    # 重拟合会在 --lib/数据变动时静默漂移三门禁口径,且覆盖销毁原 artifact)。
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    norm_path = out / "normalizer.npz"
    if a.from_ckpt and norm_path.exists():
        norm = Normalizer.load(norm_path)
        print(f"[p1] --from-ckpt:加载训练期归一化器 {norm_path}(不重拟合)")
    else:
        norm = Normalizer().fit(train_raw)
        norm.save(norm_path)
    def prep(ss):
        out2 = [copy.deepcopy(s) for s in ss]
        for s in out2:
            norm.transform(s)
        return out2
    train, val, xfer = prep(train_raw), prep(val_raw), prep(xfer_raw)

    amp = "float16" if a.device == "cuda" else "none"
    cfg = TrainConfig(d=a.d, n_layers=a.layers, n_global=a.global_tokens,
                      batch_graphs=min(a.batch, len(train)), max_steps=a.steps,
                      seed=a.seed, device=a.device, amp_dtype=amp,
                      eval_every=a.eval_every, early_stop_patience=a.patience,
                      tag_recon_weight=a.tag_recon_weight, rand_proj_split=a.rand_no_decay,
                      mask_pf_typed=a.pf_typed_frac)
    if a.from_ckpt:
        import torch
        from gridfm.train.pretrain import build_models
        ck = torch.load(a.from_ckpt, map_location="cpu", weights_only=False)
        enc, dec = build_models(cfg)
        enc.load_state_dict(ck["model_state"]); dec.load_state_dict(ck["decoder_state"])
        vh = []
        print(f"[p1] 从检查点复跑门禁:{a.from_ckpt} @ step {ck['global_step']} "
              f"(best_val {ck.get('best_val', float('nan')):.5f})")
    else:
        losses, enc, dec, vh = run_pretrain(
            train, cfg, ckpt_dir=out / "ckpt", save_every=max(1, a.steps // 6),
            resume=a.resume,
            return_models=True, val_samples=val, return_history=True,
            log_every=max(1, a.steps // 20))
        print(f"[p1] 训练损失 {losses[0]:.4f} -> {losses[-1]:.4f}"
              + (f";验证最优 {min(v for _s, v in vh):.5f}" if vh else ""))

    # 探针段在 CPU(retrieval/twins 的 Batch 未 .to(dev));**enc 与 dec 必须同设备**
    enc.to("cpu").eval(); dec.to("cpu").eval()
    from dataclasses import asdict
    rep = {"config": {"d": a.d, "L": a.layers, "K": a.global_tokens, "steps": a.steps,
                      "params_M": round(enc.num_params() / 1e6, 2),
                      "train_config": asdict(cfg),      # 完整可归因(含 D19 键/seed,评审建议)
                      "libs": libs},
           "families": {"train": fams(train_raw), "val": fams(val_raw), "transfer": fams(xfer_raw)}}

    # ---- 门禁① 检索(8.2)。**跨家族分层抽样**(2026-07-09 审查:val 按目录序拼接,
    # val[:20] 只覆盖第一个家族 → 20 个近重复拓扑,99% 门禁被架空)。轮转:每家族取第 k 个方式。----
    by_fam: dict = {}
    for s in val:
        by_fam.setdefault(s.family_id, []).append(s)
    probes, k = [], 0
    while len(probes) < 20 and any(len(v) > k for v in by_fam.values()):
        for fid in sorted(by_fam):
            if len(by_fam[fid]) > k and len(probes) < 20:
                probes.append(by_fam[fid][k])
        k += 1
    accs = [retrieval_probe(enc, s, np.random.default_rng(i), lape_k=cfg.lape_k,
                            rwse_steps=cfg.rwse_steps, random_dim=cfg.random_dim,
                            branch_random_dim=cfg.branch_random_dim)
            for i, s in enumerate(probes)]
    accs = [x for x in accs if np.isfinite(x)]
    retr = float(np.mean(accs)) if accs else float("nan")
    nvalid = float(np.mean([_valid_node_mask(s).sum() for s in probes])) if probes else 1.0
    rep["gate1_retrieval"] = {"mean_top1": retr, "min": float(np.min(accs)) if accs else None,
                              "n_samples": len(accs), "n_families": len({s.family_id for s in probes}),
                              "random_baseline": 1.0 / max(nvalid, 1.0),
                              "threshold": 0.99, "pass": bool(retr >= 0.99)}
    print(f"[门禁①检索] top-1 {retr:.4f} (n={len(accs)}, 覆盖 {rep['gate1_retrieval']['n_families']} 家族, "
          f"基线 {1.0/max(nvalid,1.0):.3f}) {'✅' if retr >= 0.99 else '❌'}")

    # ---- 门禁② 孪生(8.3,跨家族:每个 val 家族(至多 4 个)各构造节点+支路严格真孪生)----
    # base 已归一化;孪生构造是逐字节复制 → 直接得到归一化空间的严格孪生,**不得再次 transform**
    # (twin_probe 只算 token 余弦,无需物理量)。
    node_reps, branch_reps = [], []
    for fid in sorted(by_fam)[:4]:
        base = by_fam[fid][0]
        ntw, npair = add_node_twin(base, _gen_bus(base))
        tp = twin_probe(enc, ntw, [npair], random_dim=cfg.random_dim,
                        branch_random_dim=cfg.branch_random_dim, lape_k=cfg.lape_k,
                        rwse_steps=cfg.rwse_steps)
        tp["family_id"] = fid
        node_reps.append(tp)
        btw, bpair = add_branch_twin(base, 0)
        bt = branch_twin_separability(enc, btw, [bpair], random_dim=cfg.random_dim,
                                      branch_random_dim=cfg.branch_random_dim,
                                      lape_k=cfg.lape_k, rwse_steps=cfg.rwse_steps)
        bt["family_id"] = fid
        branch_reps.append(bt)
    premise_ok = all(t["lape_identical"] for t in node_reps)
    # 2026-07-11 用户批准(§8.3 修订):节点①②改读出口径裁决;裸余弦裕度/裸 xview 降诊断项
    twin_pass = bool(node_reps and premise_ok
                     and all(t["decode_twin_ok"] and t["decode_xview_ok"]
                             and t["overuse_ok"] and t["mechanism_ok"] for t in node_reps)
                     and all(b["separable"] and b["mechanism_ok"] for b in branch_reps))
    rep["gate2_twin"] = {"node": node_reps, "branch": branch_reps,
                         "lape_premise_ok": premise_ok, "pass": twin_pass}
    dec_twin = float(np.mean([t["decode_twin_rate"] for t in node_reps])) if node_reps else float("nan")
    dec_xv = float(np.mean([t["decode_xview_rate"] for t in node_reps])) if node_reps else float("nan")
    share = max(t["rand_share_median"] for t in node_reps) if node_reps else float("nan")
    sep_min = min(t["sep_min"] for t in node_reps) if node_reps else float("nan")
    abl_max = max(t["ablated_sep_max"] for t in node_reps) if node_reps else float("nan")
    drift = float(np.median([t["self_drift_median"] for t in node_reps])) if node_reps else float("nan")
    xview = float(np.mean([t["xview_hit_rate"] for t in node_reps])) if node_reps else float("nan")
    bsep = min(b["sep_min"] for b in branch_reps) if branch_reps else float("nan")
    babl = max(b["ablated_sep_max"] for b in branch_reps) if branch_reps else float("nan")
    rep["gate2_twin"]["summary"] = {"decode_twin_rate": dec_twin, "decode_xview_rate": dec_xv,
                                    "rand_share_max": share,
                                    "node_sep_min": sep_min, "self_drift_median": drift,
                                    "margin_ratio": sep_min / drift if drift else float("nan"),
                                    "xview_hit_rate": xview, "node_ablated_max": abl_max,
                                    "branch_sep_min": bsep, "branch_ablated_max": babl}
    print(f"[门禁②孪生] {len(node_reps)} 家族  LaPE前提 {'✓' if premise_ok else '✗(换锚点)'}")
    print(f"  ①可解码区分(读出口径): {dec_twin:.4f}(需 ≥0.99) "
          f"{'✅' if all(t['decode_twin_ok'] for t in node_reps) else '❌'}"
          f"  重采样移动量 {share:.3f}(≤0.3) {'✅' if all(t['overuse_ok'] for t in node_reps) else '❌'}")
    print(f"  ②读出空间跨视图身份: {dec_xv:.4f}(需 ≥0.99) "
          f"{'✅' if all(t['decode_xview_ok'] for t in node_reps) else '❌'}")
    print(f"  ③机制消融: 节点 {abl_max:.1e} / 支路 {babl:.1e}(须 <1e-3)"
          f" {'✅' if all(t['mechanism_ok'] for t in node_reps) and all(b['mechanism_ok'] for b in branch_reps) else '❌'}")
    print(f"  支路可分性: sep_min {bsep:.2e} {'✅' if all(b['separable'] for b in branch_reps) else '❌'}")
    print(f"  [诊断] 裸 sep_min {sep_min:.2e} / 自漂移 {drift:.2e} / 裕度比 "
          f"{sep_min/drift if drift else float('nan'):.3f} / 裸 xview {xview:.3f}")
    print(f"  → 门禁② {'✅' if twin_pass else '❌'}")

    # ---- 门禁③ 迁移(8.4 增量阶梯,D20;内=val 分母,外=transfer 分子)----
    # 969 个样本(含 1000 节点图)前向 → 搬回 GPU;state_field_mae 会把 batch .to(dev)
    # raw 集(未归一化,标幺)供 L0/L1/L2 零学习基线;掩码流与模型评估逐位同源(同 seed 同序)
    if xfer:
        import torch
        gdev = torch.device("cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu")
        enc.to(gdev); dec.to(gdev)
        tr = transfer_report(enc, dec, val, xfer, cfg, norm, dev=gdev,
                             in_raw=val_raw, out_raw=xfer_raw)
        enc.to("cpu"); dec.to("cpu")
        rep["gate3_transfer"] = tr
        # 冻结哈希 = **实际被评估的文件集**(family_files 与 load_families 严格同源,manifest 驱动;
        # 2026-07-09 审查:裸 glob 会盖住 manifest 外文件,且扁平布局下哈希静默为 None)。
        xfer_paths = family_files(root, "transfer")
        rep["transfer_content_hash"] = content_hash(xfer_paths)
        rep["transfer_n_files"] = len(xfer_paths)
        # 完整性:批产报告若记有失败/不足的 transfer 家族 → 冻结集不完整,标警示(8.9 冻结纪律)
        br = root / "families" / "build_report.json"
        incomplete = []
        if br.exists():
            bj = json.loads(br.read_text(encoding="utf-8"))
            incomplete = [r["family_id"] for r in bj.get("families", [])
                          if r.get("split") == "transfer" and not r.get("ok")]
        rep["transfer_incomplete_families"] = incomplete
        if incomplete:
            print(f"[警示] transfer 冻结集不完整:{incomplete} —— 哈希只证当前字节,补产后须重新冻结")
        print(f"[冻结] transfer {len(xfer_paths)} 文件  哈希 {rep['transfer_content_hash']}")
        print("[门禁③迁移]\n" + format_report(tr))
    else:
        rep["gate3_transfer"] = {"pass": False, "reason": "无 transfer 家族"}
        print("[门禁③迁移] ❌ 无 transfer 家族(批产未完成?)")

    all_pass = (rep["gate1_retrieval"]["pass"] and rep["gate2_twin"]["pass"]
                and rep["gate3_transfer"].get("passed", rep["gate3_transfer"].get("pass", False)))
    rep["all_gates_pass"] = bool(all_pass)
    (out / "p1_full_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[p1] 三门禁{'全过 → 可按 D11 冻结为 v1' if all_pass else '未全过 → 归因(容量/多样性)'}")
    print(f"[p1] 报告 {out / 'p1_full_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
