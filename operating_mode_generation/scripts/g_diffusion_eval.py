"""[G] 诊断 C(6.5 端到端)+ CFG 网格 + 单变量对照 + 多样性/记忆化 + 泄漏对照臂。

留出 60 方式:c_gen 条件(去泄漏骨架编码,审查修复)+ c_label(lf 档)→ DDIM 采样 z
→ latent_stats 逆变换 → G1.5 读出解码器 → 写回样本 case → PSASP LF 试算。
--regressor-ckpt 走确定性回归(无采样无 CFG),其余协议逐位相同(6.6 单变量对照)。
--leaky 走 c_topo 泄漏视图(评旧泄漏口径产物,量化泄漏贡献)。

指标(并报,收敛率单独看会误判):
- 收敛率(PSASP 默认迭代上限口径,非 8.8 的 k=5/10)/ 电压带内率(vmin≥0.85);
- **双底版**:cases/<sid>(初值=真解,天花板)+ golden 骨架(g1_baseline 同款,
  新方式无真值 case 的诚实起步);
- 标签一致率:就近归档 + **容差判定**(|lf̂−档| ≤ tol,边缘档大偏差不再计一致);
- 字段 MAE vs 留出真值(仅参考,条件一对多);
- **多样性**(同条件 k 个 z_T 生成方式两两 L1)+ **记忆化检验**(生成→最近训练方式
  L1 vs 训练方式间典型 L1;9.8M 参数 > 训练潜码总量,必查背样本)。

  python scripts/g_diffusion_eval.py --root ..\\gridfm-data
      [--cfg-grid 1.0,1.5,2,2.5,3,3.5,4] [--regressor-ckpt ...] [--leaky ...]
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
sys.path.insert(0, str(REPO / "scripts"))
import gridfm  # noqa: E402
assert gridfm.__file__.startswith(str(REPO)), f"窜仓:{gridfm.__file__}"

from g1_baseline import writeback_and_trial  # noqa: E402

from gridfm.data.cond_views import cgen_view, ctopo_view  # noqa: E402
from gridfm.data.dataset import load_library  # noqa: E402
from gridfm.data.graph import sample_to_data  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.fsutil import robust_rmtree  # noqa: E402
from gridfm.models.decoder.readout import ReadoutDecoder  # noqa: E402
from gridfm.models.diffusion import (DenoiserV0, GaussianDiffusion,  # noqa: E402
                                     LatentRegressorV0)
from gridfm.models.diffusion.denoiser import LF_TIERS, label_vec  # noqa: E402
from gridfm.schema.fields import NODE_TABLE, Group  # noqa: E402
from gridfm.schema.sample import group_bit_index  # noqa: E402
from gridfm.solver.psasp import LicenseError, PsaspSolver  # noqa: E402
from gridfm.train.masking import COL_GROUP  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402

MODE_FIELDS = ("pl", "ql", "pg", "v_set")


def _fp(path: Path) -> str:
    return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--lib-name", default="lib118_diverse")
    ap.add_argument("--v1-dir", default="runs/encoder_v1")
    ap.add_argument("--denoiser-ckpt", default="runs/diffusion_g1_cgen/denoiser.pt")
    ap.add_argument("--regressor-ckpt", default=None,
                    help="给出时评单变量对照(确定性回归,无 CFG),协议其余逐位相同")
    ap.add_argument("--readout-ckpt", default="runs/decoder_g15_cgen/readout.pt")
    ap.add_argument("--latent-stats", default="runs/latent_stats/latent_stats.npz")
    ap.add_argument("--golden-base",
                    default=str(REPO / "tests" / "fixtures" / "ieee118_golden"),
                    help="诚实底版(g1_baseline 同款 golden 骨架;新方式无真值 case)")
    ap.add_argument("--leaky", action="store_true",
                    help="泄漏对照臂:用 c_topo 泄漏视图评旧泄漏口径产物")
    ap.add_argument("--out", default=None)
    ap.add_argument("--test-every", type=int, default=5)
    ap.add_argument("--cfg-grid", default="1.0,1.5,2,2.5,3,3.5,4")
    ap.add_argument("--ddim-steps", type=int, default=250,
                    help="采样诊断定档:50 步残余漂移致 z0 std>2,250 步 + clip4 收敛")
    ap.add_argument("--clip-x0", type=float, default=4.0,
                    help="DDIM 静态阈值:标准化潜码 p99.9=3.96/absmax5.34;clip4→z0 std"
                         "1.08(健康),clip8→std2.4(残余漂移,0/60收敛);采样诊断定档")
    ap.add_argument("--tol", type=float, default=0.03, help="标签一致容差(|lf̂−档|)")
    ap.add_argument("--vset-calib", type=float, default=0.0,
                    help="发电机电压设点校准(加到生成 v_set;修系统性偏低,须 val 测得,"
                         "§6.3③ 交付后处理;0=不校准)")
    ap.add_argument("--div-k", type=int, default=10, help="多样性:每条件 z_T 种子数")
    ap.add_argument("--div-psasp", type=int, default=15,
                    help="多样性 PSASP 抽样条件数(验证多样样本仍收敛)")
    ap.add_argument("--seed", type=int, default=0, help="固定 z_T(base),各 CFG 点共用")
    ap.add_argument("--skip-trial", action="store_true")
    ap.add_argument("--skip-diversity", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    is_reg = a.regressor_ckpt is not None
    view_fn = ctopo_view if a.leaky else cgen_view
    arm = "leaky" if a.leaky else "cgen"
    out = Path(a.out) if a.out else Path(
        f"runs/{'latent_reg_g1' if is_reg else 'diffusion_g1'}"
        f"{'' if a.leaky else '_cgen'}/{'eval' if is_reg else 'diag_c'}")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 冻结件与指纹链 ----
    fp_v1 = _fp(Path(a.v1_dir) / "v1.pt")
    fp_nm = _fp(Path(a.v1_dir) / "normalizer.npz")
    fp_ls = _fp(Path(a.latent_stats))
    stats = np.load(a.latent_stats, allow_pickle=False)
    z_mean = torch.from_numpy(stats["mean"]).to(dev)
    z_std = torch.from_numpy(stats["std"]).to(dev)

    gen_path = Path(a.regressor_ckpt if is_reg else a.denoiser_ckpt)
    gk = torch.load(gen_path, map_location="cpu", weights_only=False)
    assert gk["fingerprints"]["v1"] == fp_v1, "生成器 ckpt 与 v1 指纹不符"
    assert gk["fingerprints"]["latent_stats"] == fp_ls, "生成器 ckpt 与 latent_stats 不符"
    assert gk["fingerprints"].get("normalizer", fp_nm) == fp_nm, "生成器 ckpt 与 normalizer 不符"
    cls = LatentRegressorV0 if is_reg else DenoiserV0
    net = cls(gk["config"]["d"], n_layers=gk["config"]["n_layers"]).to(dev)
    net.load_state_dict(gk.get("ema_state") or gk["model_state"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    rk = torch.load(a.readout_ckpt, map_location="cpu", weights_only=False)
    assert rk.get("v1_fingerprint") == fp_v1, "readout ckpt 与 v1 指纹不符"
    dec = ReadoutDecoder(rk["config"]["d"], n_layers=rk["config"]["n_layers"]).to(dev)
    dec.load_state_dict(rk["decoder_state"])
    dec.eval()

    ck = torch.load(Path(a.v1_dir) / "v1.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items()
                         if k in {f.name for f in dc_fields(TrainConfig)}})
    enc, _ = build_models(cfg)
    enc.load_state_dict(ck["model_state"])
    enc.eval()
    norm = Normalizer.load(Path(a.v1_dir) / "normalizer.npz")
    print(f"[diagC] 臂={arm} 生成器={'回归' if is_reg else f'扩散DDIM{a.ddim_steps}'}"
          f" ckpt step {gk['train']['step']}"
          f" valEMA {gk['train'].get('val_ema_mse', gk['train'].get('val_mse')):.4f};"
          "指纹链全通过", flush=True)

    # ---- 留出 60 + 训练 240(记忆化对照)----
    lib = Path(a.root) / a.lib_name
    lf_by_sid, solvable = {}, {}
    with open(lib / "manifest.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lf_by_sid[r["sample_id"]] = float(r["load_factor"])
            solvable[r["sample_id"]] = r["solvable"] == "1"
    samples = [s for s in load_library(lib) if solvable.get(s.sample_id)]
    samples.sort(key=lambda s: lf_by_sid[s.sample_id])
    test_idx = set(range(0, len(samples), a.test_every))
    test_s = [s for i, s in enumerate(samples) if i in test_idx]
    train_s = [s for i, s in enumerate(samples) if i not in test_idx]
    print(f"[diagC] 留出 {len(test_s)} / 训练 {len(train_s)}(记忆化对照)")

    tier_of = {v: i for i, v in enumerate(LF_TIERS)}
    s0 = test_s[0]
    tcol = NODE_TABLE.offset_of("node_type")
    types = np.argmax(s0.node_feat[:, tcol:tcol + 3], axis=1)
    is_null = s0.node_feat[:, NODE_TABLE.offset_of("is_null")] > 0.5
    has_gen = (s0.node_feat[:, NODE_TABLE.offset_of("has_gen")] > 0.5) & ~is_null
    has_load = (s0.node_feat[:, NODE_TABLE.offset_of("has_load")] > 0.5) & ~is_null
    gen_ns = has_gen & (types != 2)
    row_mask = {"pl": has_load, "ql": has_load, "pg": gen_ns, "v_set": gen_ns}
    col_of = {f: NODE_TABLE.offset_of(f) for f in MODE_FIELDS}

    def mode_vec(pu):
        """方式特征向量(pu):负荷母线 pl/ql ⊕ 机组 pg/v_set,记忆化/多样性距离用。"""
        return np.concatenate([pu[row_mask[f], col_of[f]] for f in MODE_FIELDS])

    train_vecs = np.stack([mode_vec(s.node_feat) for s in train_s])   # [240, D]
    # 训练方式间典型 L1(抽样对角外)
    rr = np.random.default_rng(0)
    pi_ = rr.integers(0, len(train_vecs), 400)
    pj_ = rr.integers(0, len(train_vecs), 400)
    msk = pi_ != pj_
    inter_train_l1 = float(np.median(np.abs(train_vecs[pi_[msk]]
                                            - train_vecs[pj_[msk]]).mean(1)))

    def nearest_train_l1(vec):
        return float(np.abs(train_vecs - vec).mean(1).min())

    # ---- 条件编码(chosen view,60 方式整批)----
    normed, ct_datas = [], []
    for raw in test_s:
        s = copy.deepcopy(raw)
        norm.transform(s)
        normed.append(s)
        ct_datas.append(sample_to_data(view_fn(s), cfg.lape_k, cfg.rwse_steps,
                                       random_dim=cfg.random_dim,
                                       branch_random_dim=cfg.branch_random_dim))
    with torch.no_grad():
        hc, _g, ec = enc(Batch.from_data_list(ct_datas))
    n = test_s[0].n_nodes
    hc, ec = hc.to(dev), ec.to(dev)
    ei = torch.cat([ct_datas[j].edge_index + j * n for j in range(len(test_s))],
                   dim=1).to(dev)
    tiers = [tier_of[lf_by_sid[s.sample_id]] for s in test_s]
    lab_c = torch.stack([label_vec(t, device=dev) for t in tiers])
    lab_u = torch.stack([label_vec(None, device=dev)] * len(test_s))
    npg = torch.full((len(test_s),), n, dtype=torch.long, device=dev)
    base_total = {}
    for raw in test_s:
        tot = float(raw.node_feat[has_load, col_of["pl"]].sum())
        base_total[raw.sample_id] = tot / lf_by_sid[raw.sample_id]
    truth_nearest = float(np.mean([nearest_train_l1(mode_vec(s.node_feat))
                                   for s in test_s]))  # 真值新方式的最近训练距离(基线)

    diff = GaussianDiffusion(gk["config"]["T"]).to(dev) if not is_reg else None

    @torch.no_grad()
    def generate(cfg_scale, z_T):
        if is_reg:
            z0 = net(hc, ec, ei, lab_c, npg)
        else:
            def eps_fn(z_t, t):
                tt = torch.full((len(test_s),), t, dtype=torch.long, device=dev)
                e_c = net(z_t, hc, ec, ei, tt, lab_c, npg)
                e_u = net(z_t, hc, ec, ei, tt, lab_u, npg)
                return e_u + cfg_scale * (e_c - e_u)
            z0 = diff.ddim_sample(eps_fn, z_T, steps=a.ddim_steps, clip_x0=a.clip_x0)
        h = z0 * z_std + z_mean
        return dec(h, ec, ei).cpu().numpy()

    def z_T_seed(seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn((len(test_s) * n, cfg.d), generator=g).to(dev)

    solver = None if a.skip_trial else PsaspSolver()
    sbit = group_bit_index(Group.STATE)
    scols = np.where(COL_GROUP == sbit)[0]

    def decode_to_pu(dmat, j):
        full = normed[j].node_feat.copy()
        full[:, scols] = dmat[j * n:(j + 1) * n, scols]
        pu = norm.inverse_transform_node(full)
        if a.vset_calib != 0.0:                 # 设点校准:所有下游路径一致应用
            pu[:, col_of["v_set"]] = pu[:, col_of["v_set"]] + a.vset_calib
        return pu

    def evaluate(dmat, tag, base_fn, do_trial=True):
        err = {f: [] for f in MODE_FIELDS}
        n_conv = n_band = n_lab_near = n_lab_tol = 0
        lf_hat_abs = []
        rows = []
        t0 = time.time()
        for j, raw in enumerate(test_s):
            pu = decode_to_pu(dmat, j)          # 已含设点校准(decode_to_pu 内应用)
            for f in MODE_FIELDS:
                mk = row_mask[f]
                err[f].append(float(np.abs(pu[mk, col_of[f]]
                                           - raw.node_feat[mk, col_of[f]]).mean()))
            lf_hat = float(pu[has_load, col_of["pl"]].sum()) / base_total[raw.sample_id]
            near = min(range(len(LF_TIERS)), key=lambda k: abs(LF_TIERS[k] - lf_hat))
            dev_lf = abs(lf_hat - LF_TIERS[tiers[j]])
            n_lab_near += int(near == tiers[j])
            n_lab_tol += int(dev_lf <= a.tol)
            lf_hat_abs.append(dev_lf)
            row = {"sample_id": raw.sample_id, "lf": lf_by_sid[raw.sample_id],
                   "lf_hat": round(lf_hat, 4)}
            if solver is not None and do_trial:
                pred = np.stack([pu[:, col_of[f]] for f in MODE_FIELDS], axis=1)
                work = out / "work" / f"{tag}_{raw.sample_id}"
                try:
                    conv, vmin, vmax = writeback_and_trial(
                        pred, base_fn(raw), work, solver)
                except LicenseError:
                    print("[diagC] slm_init error:加密狗失效,中止")
                    raise
                n_conv += int(conv)
                n_band += int(conv and vmin is not None and vmin >= 0.85)
                row.update({"converged": conv, "vmin": vmin, "vmax": vmax})
                if not a.keep_work:
                    robust_rmtree(work)
            rows.append(row)
        res = {"tag": tag, "n": len(test_s),
               "mae_pu": {f: round(float(np.mean(v)), 5) for f, v in err.items()},
               "label_consistency_nearest": round(n_lab_near / len(test_s), 4),
               "label_consistency_tol": round(n_lab_tol / len(test_s), 4),
               "lf_hat_mean_abs_dev": round(float(np.mean(lf_hat_abs)), 4)}
        if solver is not None and do_trial:
            res.update({"converged": n_conv, "conv_rate": round(n_conv / len(test_s), 4),
                        "conv_and_vband085": n_band,
                        "band_rate": round(n_band / len(test_s), 4),
                        "elapsed_s": round(time.time() - t0, 1)})
        res["rows"] = rows
        print(f"[diagC] {tag}: 收敛 {res.get('converged','-')}/{len(test_s)}"
              f" 带内 {res.get('conv_and_vband085','-')}"
              f" 标签(近/容差){n_lab_near}/{n_lab_tol}"
              f" | MAE pg {res['mae_pu']['pg']:.4f} v_set {res['mae_pu']['v_set']:.4f}",
              flush=True)
        return res

    cases_fn = lambda raw: lib / "cases" / raw.sample_id
    golden_fn = lambda raw: Path(a.golden_base)

    results = []
    best_cfg, best_dmat = None, None
    best_key = (-1.0, -1.0)          # (band_rate, label_consistency_tol),越大越好
    if is_reg:
        dmat = generate(None, None)
        results.append({"baseline": "cases",
                        **evaluate(dmat, "reg_cases", cases_fn)})
        results.append({"baseline": "golden",
                        **evaluate(dmat, "reg_golden", golden_fn)})
        best_dmat = dmat
    else:
        z_base = z_T_seed(a.seed)
        for c in [float(x) for x in a.cfg_grid.split(",")]:
            t0 = time.time()
            dmat = generate(c, z_base)
            print(f"[diagC] cfg={c}: 采样+解码 {time.time()-t0:.1f}s", flush=True)
            r = {"baseline": "cases", "cfg": c,
                 **evaluate(dmat, f"cfg{c}_cases", cases_fn)}
            results.append(r)
            key = (r.get("band_rate", 0.0), r["label_consistency_tol"])
            if key > best_key:
                best_key, best_cfg, best_dmat = key, c, dmat
        print(f"[diagC] best CFG = {best_cfg}(带内率 {best_key[0]} 容差一致 {best_key[1]})",
              flush=True)
        # 最佳 CFG 上补 golden 诚实底版
        results.append({"baseline": "golden", "cfg": best_cfg,
                        **evaluate(best_dmat, f"cfg{best_cfg}_golden", golden_fn)})

    # ---- 多样性 + 记忆化(best CFG;decode-only 全量 + PSASP 抽样)----
    diversity = None
    if not a.skip_diversity:
        cfg_use = best_cfg if not is_reg else None
        seed_dmats = []
        for s in range(a.div_k):
            zt = None if is_reg else z_T_seed(1000 + s)
            seed_dmats.append(generate(cfg_use, zt))
        # 每条件 k 个方式向量
        per_cond_div, all_gen_nearest = [], []
        for j in range(len(test_s)):
            vecs = [mode_vec(decode_to_pu(dm, j)) for dm in seed_dmats]
            V = np.stack(vecs)
            dsum, dcnt = 0.0, 0
            for x in range(len(V)):
                for y in range(x + 1, len(V)):
                    dsum += float(np.abs(V[x] - V[y]).mean()); dcnt += 1
            per_cond_div.append(dsum / max(1, dcnt))
            all_gen_nearest.extend(nearest_train_l1(v) for v in vecs)
        gen_nearest_mean = float(np.mean(all_gen_nearest))
        diversity = {
            "div_k": a.div_k,
            "mean_pairwise_l1_per_condition": round(float(np.mean(per_cond_div)), 5),
            "inter_train_mode_l1_median": round(inter_train_l1, 5),
            "gen_nearest_train_l1_mean": round(gen_nearest_mean, 5),
            "truth_nearest_train_l1_mean": round(truth_nearest, 5),
            "memorization_flag": bool(gen_nearest_mean < 0.5 * truth_nearest
                                      and gen_nearest_mean < 0.3 * inter_train_l1),
        }
        print(f"[diagC] 多样性(同条件k={a.div_k}两两L1)"
              f"{diversity['mean_pairwise_l1_per_condition']:.4f}"
              f" | 训练间典型 {inter_train_l1:.4f}"
              f" | 生成→最近训练 {gen_nearest_mean:.4f}"
              f" vs 真值→最近训练 {truth_nearest:.4f}"
              f" | 记忆化={diversity['memorization_flag']}", flush=True)
        # PSASP 抽样:验证多样样本仍收敛(前 div-psasp 条件 × 前 min(k,8) 种子)
        if solver is not None and not a.skip_trial:
            ksub = min(a.div_k, 8)
            n_c = n_b = n_t = 0
            for j in range(min(a.div_psasp, len(test_s))):
                for s in range(ksub):
                    pu = decode_to_pu(seed_dmats[s], j)
                    pred = np.stack([pu[:, col_of[f]] for f in MODE_FIELDS], axis=1)
                    work = out / "work" / f"div_{j}_{s}"
                    conv, vmin, vmax = writeback_and_trial(
                        pred, cases_fn(test_s[j]), work, solver)
                    n_c += int(conv)
                    n_b += int(conv and vmin is not None and vmin >= 0.85)
                    n_t += 1
                    if not a.keep_work:
                        robust_rmtree(work)
            diversity["psasp_sample"] = {"n": n_t, "converged": n_c, "band": n_b,
                                         "conds": min(a.div_psasp, len(test_s)),
                                         "seeds": ksub}
            print(f"[diagC] 多样样本 PSASP 抽样:收敛 {n_c}/{n_t} 带内 {n_b}", flush=True)

    report = {
        "task": ("6.6 单变量对照(潜空间回归)评估" if is_reg
                 else "G1 诊断 C(6.5):扩散端到端 + CFG 网格") + f" [{arm} 臂]",
        "arm": arm, "generator_ckpt": str(gen_path), "readout_ckpt": a.readout_ckpt,
        "lib": lib.name, "n_holdout": len(test_s), "seed_zT_base": a.seed,
        "ddim_steps": None if is_reg else a.ddim_steps,
        "best_cfg": best_cfg, "vset_calib": a.vset_calib,
        "fingerprints": gk["fingerprints"],
        "diversity_memorization": diversity,
        "notes": [
            "评估集 = lib118_diverse 留出 60 方式([内],D14 单拓扑豁免,同拓扑;不作泛化主张)",
            f"条件视图 = {'c_topo(泄漏对照臂)' if a.leaky else 'c_gen(去泄漏)'};"
            "各 CFG 点共用同一 z_T(隔离 CFG 效应)",
            "双底版:cases=初值真解天花板 / golden=g1_baseline 同款骨架(新方式诚实起步);"
            "启停从真值卡继承",
            "conv_rate = PSASP 默认迭代上限口径,非 8.8 的裸收敛率 k=5/10(迭代档现工具链"
            "不可设,偏离如实标注);标签一致率为裸口径(无 R=2 兜底,非 8.8 交付口径)",
            "记忆化:生成→最近训练 L1 显著小于 真值→最近训练 且 << 训练间典型 = 背样本预警",
        ],
        "results": results,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(f"[diagC] 报告 {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
