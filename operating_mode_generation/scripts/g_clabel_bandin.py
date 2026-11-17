"""[G] c_label 校准后分层带内(选项2 步骤④⑤;PSASP 唯一裁决)。

对三测试家族(s2007 上限 / 中档 / s2006 下限)按 CFG 网格,比较**未校准**(cond=真 c_label)
vs **校准后**(load 条件按仿射重映射 + v_set 常数偏置修正)的 PSASP 收敛/带内/c_label 一致率。
**分层报,不报单家族/不报混合口径**。冻结件:DDIM250+clip4 / decoder_g15_cgen_g4_up6 /
latent_stats_v2。校准参数读 g_clabel_calib.py 产出的 calib.json。
**写回协议默认 = vset_base(2026-07-22 定档;双集合确认 VAL +18.1pp / transfer +16.7pp,
变更记录-G线 2026-07-20 条)。gen4 = 旧协议,仅作历史对照,勿用于交付数字。**

用法(cwd=仓库):
  python scripts/g_clabel_bandin.py --calib-json runs/g_clabel_calib/calib.json
      [--n 20] [--cfgs 1.0,1.5,2.0,2.5] [--modes uncal,cal]
      [--test-fams synth210_s2007,synth150_s2005,synth180_s2006]
数据根经 GRIDFM_DATA_ROOT 注入(默认 仓库父目录/gridfm-data)。
"""
import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("GRIDFM_DATA_ROOT", str(REPO.parent / "gridfm-data")))
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import gridfm  # noqa
assert gridfm.__file__.startswith(str(REPO)), f"窜仓:{gridfm.__file__}"

from dataclasses import fields as dc_fields  # noqa
from torch_geometric.data import Batch  # noqa
from g1_baseline import writeback_and_trial  # noqa
from gridfm.data import c_label as CL  # noqa
from gridfm.data.cond_views import cgen_view  # noqa
from gridfm.data.corpus import load_lib_lf_sorted  # noqa
from gridfm.data.graph import sample_to_data  # noqa
from gridfm.data.normalize import Normalizer  # noqa
from gridfm.models.decoder.readout import ReadoutDecoder  # noqa
from gridfm.models.diffusion import DenoiserV0, GaussianDiffusion  # noqa
from gridfm.psasp_io.case import PsaspCase  # noqa
from gridfm.psasp_io.catalog import Catalog  # noqa
from gridfm.schema.assemble import build_sample  # noqa
from gridfm.schema.fields import NODE_TABLE, Group  # noqa
from gridfm.schema.sample import group_bit_index  # noqa
from gridfm.solver.psasp import PsaspSolver  # noqa
from gridfm.train.masking import COL_GROUP  # noqa
from gridfm.train.pretrain import TrainConfig, build_models  # noqa

FP_V1 = "ccf4260f2a8f5db5b6650a33b17d7a0f"
FP_NM = "12281dd483206efe09fdf19475472ebc"
MODE_FIELDS = ("pl", "ql", "pg", "v_set")
# 交付判据(2026-07-20 用户裁):**双边**。旧口径只卡下限,而 v_set 写回无钳位 + cal 全局 +0.0255pu
# 会整体上推电压剖面,把可见的低压违限换成不可见的高压违限(实测 cal 越上限率 52~92% vs uncal 0~26%)。
VMIN_LO, VMAX_HI = 0.85, 1.10


def _blake(p):
    return hashlib.blake2b(Path(p).read_bytes(), digest_size=16).hexdigest()


def load_stack(dev, denoiser_path="runs/diffusion_g1_cgen_g4/denoiser.pt"):
    v1 = REPO / "runs/encoder_v1"
    assert _blake(v1 / "v1.pt") == FP_V1 and _blake(v1 / "normalizer.npz") == FP_NM, "指纹不符"
    gk = torch.load(REPO / denoiser_path, map_location="cpu", weights_only=False)
    fps = gk.get("fingerprints", {})
    assert fps.get("latent_version") == "latent_stats_v2"
    st_path = REPO / "runs/latent_stats_v2/latent_stats.npz"
    stats = np.load(st_path, allow_pickle=False)
    assert json.loads(str(stats["meta"]))["version"] == "latent_stats_v2"
    assert _blake(st_path) == fps.get("latent_stats"), "latent_stats 与训练不符"
    lnorm = CL.LabelNorm.from_dict(gk["config"]["label_norm"])
    ck = torch.load(v1 / "v1.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items()
                         if k in {f.name for f in dc_fields(TrainConfig)}})
    enc, _ = build_models(cfg); enc.load_state_dict(ck["model_state"]); enc.eval(); enc.to(dev)
    norm = Normalizer.load(v1 / "normalizer.npz")
    net = DenoiserV0(gk["config"]["d"], n_layers=gk["config"]["n_layers"]).to(dev)
    net.load_state_dict(gk["ema_state"]); net.eval()
    z_mean = torch.from_numpy(stats["mean"]).to(dev); z_std = torch.from_numpy(stats["std"]).to(dev)
    # [外协包 F,2026-08-25] 读出解码器路径改为可经环境变量覆盖,**缺省值与原实现逐字相同**
    # (不设 CLI 是为了不改本文件的判据段与函数签名)。判据 VMIN_LO/VMAX_HI 与判定式一字未动。
    _ro = os.environ.get("GRIDFM_READOUT_CKPT") or str(REPO / "runs/decoder_g15_cgen_g4_up6/readout.pt")
    rk = torch.load(_ro, map_location="cpu",
                    weights_only=False)
    cc = rk["config"]
    dec = ReadoutDecoder(cc["d"], n_layers=cc["n_layers"], ffn_mult=cc.get("ffn_mult", 2),
                         hidden=cc.get("hidden") or 2 * cc["d"]).to(dev)
    dec.load_state_dict(rk["decoder_state"]); dec.eval()
    diff = GaussianDiffusion(gk["config"]["T"]).to(dev)
    print(f"[bandin] stack OK: denoiser step {gk['train']['step']} valEMA "
          f"{gk['train']['val_ema_mse']:.4f}; latent {fps.get('latent_version')}")
    return dict(enc=enc, norm=norm, net=net, dec=dec, diff=diff, lnorm=lnorm,
                z_mean=z_mean, z_std=z_std, d=gk["config"]["d"], cfg=cfg)


def prep_family(fam, N, S, dev, split="val"):
    cfg = S["cfg"]
    famdir = DATA / "families" / split / fam
    test_s, _ = load_lib_lf_sorted(famdir); test_s = test_s[:N]
    bc = PsaspCase(famdir / "_base_case")
    bsample = build_sample(bc, Catalog(bc), sample_id="b", family_id=fam)
    base_pl = CL.base_pl_sum(bsample)
    base_feat = bsample.node_feat.copy()          # 基态逐节点(pg_only 写回协议要用)
    assert base_feat.shape[0] == test_s[0].n_nodes, \
        f"基态节点数 {base_feat.shape[0]} != 方式 {test_s[0].n_nodes}({fam})"
    true_labels = [CL.recompute(s, base_pl) for s in test_s]
    normed, ctd = [], []
    for raw in test_s:
        s = copy.deepcopy(raw); S["norm"].transform(s); normed.append(s)
        ctd.append(sample_to_data(cgen_view(s), cfg.lape_k, cfg.rwse_steps,
                                  random_dim=cfg.random_dim, branch_random_dim=cfg.branch_random_dim))
    with torch.no_grad():
        hc, _g, ec = S["enc"](Batch.from_data_list(ctd).to(dev))
    n = test_s[0].n_nodes
    ei = torch.cat([ctd[j].edge_index + j * n for j in range(N)], dim=1).to(dev)
    npg = torch.full((N,), n, dtype=torch.long, device=dev)
    torch.manual_seed(0)
    z_T = torch.randn(N * n, S["d"], device=dev)
    o = NODE_TABLE.offset_of
    f0 = test_s[0].node_feat
    types = np.argmax(f0[:, o("node_type"):o("node_type") + 3], axis=1)
    gen_ns = (f0[:, o("has_gen")] > 0.5) & (f0[:, o("is_null")] < 0.5) & (types != 2)
    return dict(fam=fam, famdir=famdir, test_s=test_s, normed=normed, true_labels=true_labels,
                base_pl=base_pl, base_feat=base_feat, hc=hc, ec=ec, ei=ei, npg=npg, z_T=z_T,
                n=n, N=N, gen_ns=gen_ns, vcol=o("v_set"))


def build_labels(F, mode, cp, lnorm, dev):
    """mode=uncal → 真 c_label;mode=cal → load 按仿射逆映射重构,online 保持真值。"""
    aL, bL = cp["load_affine"]["a"], cp["load_affine"]["b"]
    supp = cp["_support"]
    labs = []
    for j in range(F["N"]):
        tl = F["true_labels"][j]
        if mode == "cal":
            req = float(np.clip((tl["load_level"] - bL) / aL, supp[0], supp[1]))
            lab = {"load_level": req, "online_count": tl["online_count"], "n_gen": tl["n_gen"]}
        else:
            lab = {"load_level": tl["load_level"], "online_count": tl["online_count"],
                   "n_gen": tl["n_gen"]}
        labs.append(torch.from_numpy(CL.encode(lab, lnorm)).to(dev))
    return torch.stack(labs)


def run_cell(F, cfg_scale, mode, cp, S, dev, scols, solver, workroot, wb="gen4"):
    net, dec, diff = S["net"], S["dec"], S["diff"]
    z_mean, z_std, lnorm = S["z_mean"], S["z_std"], S["lnorm"]
    N, n = F["N"], F["n"]
    lab_c = build_labels(F, mode, cp, lnorm, dev)
    lab_u = torch.stack([torch.from_numpy(CL.encode(None, lnorm)).to(dev)] * N)
    vcorr = -cp["vset_bias_const"] if mode == "cal" else 0.0  # 补 gen 系统性偏低

    @torch.no_grad()
    def eps_fn(z_t, t):
        tt = torch.full((N,), t, dtype=torch.long, device=dev)
        e_c = net(z_t, F["hc"], F["ec"], F["ei"], tt, lab_c, F["npg"])
        e_u = net(z_t, F["hc"], F["ec"], F["ei"], tt, lab_u, F["npg"])
        return e_u + cfg_scale * (e_c - e_u)

    with torch.no_grad():
        z0 = diff.ddim_sample(eps_fn, F["z_T"], steps=250, clip_x0=4.0)
        z0std = float(z0.std())
        fields = dec(z0 * z_std + z_mean, F["ec"], F["ei"]).cpu().numpy()

    col = {f: NODE_TABLE.offset_of(f) for f in MODE_FIELDS}
    n_conv = n_band = n_band2 = n_over = 0
    ll_err, oc_err, gll_all, goc_all, rows = [], [], [], [], []
    for j, raw in enumerate(F["test_s"]):
        full = F["normed"][j].node_feat.copy()
        full[:, scols] = fields[j * n:(j + 1) * n][:, scols]
        pu = S["norm"].inverse_transform_node(full)
        if wb == "vset_base":
            # 变体(B):只把 v_set 换成基态常数(除过压病根),pl/ql/pg 仍用生成值
            # —— 保留解码器三字段与 pg 之间可能存在的"互相自洽"。
            pu[:, col["v_set"]] = F["base_feat"][:, col["v_set"]]
        elif wb == "pg_only":
            # 字段信息账(2026-07-20 坐实):pl/ql/v_set 的解码输出比"基态×标签"平凡公式更差
            # (pl 差3.15x、ql 差11x~607x、v_set 仅好12%),只有 pg 值得生成 → 三路改平凡公式,
            # 一次性移除伪误差;v_set 回基态常数亦自动消除过压(不再叠 vcorr)。
            bf = F["base_feat"]
            tgt = F["true_labels"][j]["load_level"]           # 交付目标 = 真负荷水平(条件已给定)
            b_pl, b_ql = bf[:, col["pl"]], bf[:, col["ql"]]
            new_pl = b_pl * tgt
            pf = np.divide(b_ql, b_pl, out=np.zeros_like(b_ql), where=np.abs(b_pl) > 1e-9)
            pu[:, col["pl"]] = new_pl
            pu[:, col["ql"]] = new_pl * pf                    # 功率因数投影(实测 MAE 1.8e-5)
            pu[:, col["v_set"]] = bf[:, col["v_set"]]         # 基态常数
        elif vcorr:
            pu[F["gen_ns"], F["vcol"]] += vcorr
        pred_pu = np.stack([pu[:, col[f]] for f in MODE_FIELDS], axis=1)
        work = workroot / F["fam"] / f"{mode}_{cfg_scale}" / raw.sample_id
        conv, vmin, vmax = writeback_and_trial(pred_pu, F["famdir"] / "cases" / raw.sample_id,
                                               work, solver)
        inb = bool(conv and vmin is not None and vmin >= VMIN_LO)          # 旧单边(对照)
        inb2 = bool(inb and vmax is not None and vmax <= VMAX_HI)          # 新双边(交付口径)
        n_conv += int(conv); n_band += int(inb); n_band2 += int(inb2)
        n_over += int(bool(conv and vmax is not None and vmax > VMAX_HI))
        g = copy.deepcopy(raw); g.node_feat = pu
        gl = CL.recompute(g, F["base_pl"])
        ll_err.append(abs(gl["load_level"] - F["true_labels"][j]["load_level"]))
        oc_err.append(abs(gl["online_count"] - F["true_labels"][j]["online_count"]))
        gll_all.append(gl["load_level"]); goc_all.append(gl["online_count"])
        rows.append({"sid": raw.sample_id, "conv": bool(conv),
                     "vmin": None if vmin is None else round(float(vmin), 4),
                     "vmax": None if vmax is None else round(float(vmax), 4),
                     "gen_load": round(gl["load_level"], 4), "gen_online": gl["online_count"]})
        shutil.rmtree(work, ignore_errors=True)
    return {"n": N, "converged": n_conv, "conv_rate": round(n_conv / N, 3),
            "band": n_band, "band_rate": round(n_band / N, 3),
            "band_2s": n_band2, "band_2s_rate": round(n_band2 / N, 3),
            "over_vmax": n_over, "vmin_lo": VMIN_LO, "vmax_hi": VMAX_HI,
            "load_mae": round(float(np.mean(ll_err)), 4),
            "online_mae": round(float(np.mean(oc_err)), 3),
            "gen_load_mean": round(float(np.mean(gll_all)), 4),
            "gen_online_mean": round(float(np.mean(goc_all)), 2),
            "vcorr_applied": round(float(vcorr), 4), "z0std": round(z0std, 3), "rows": rows}


def resolve_cp(calib, c, fam, calib_mode, supp):
    """按 calib_mode 取该 (CFG,家族) 生效仿射 + v_set 偏置。pooled=全局;perfam=逐家族自校准。"""
    base = calib["per_cfg"][f"{c}"]
    if calib_mode == "perfam":
        la = base["load_affine_per_family"][fam]
        vb = base.get("vset_bias_perfam", {}).get(fam, base["vset_bias_const"])
    else:
        la = base["load_affine"]
        vb = base["vset_bias_const"]
    return {"load_affine": {"a": la["a"], "b": la["b"]}, "vset_bias_const": vb, "_support": supp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-json", default="runs/g_clabel_calib/calib.json")
    ap.add_argument("--denoiser", default="runs/diffusion_g1_cgen_g4/denoiser.pt",
                    help="扩散去噪器 ckpt(seed0=默认;seed1=..._s1,seed2=..._s2)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--cfgs", default="1.0,1.5,2.0,2.5")
    ap.add_argument("--modes", default="uncal,cal")
    ap.add_argument("--split", default="val", help="families 子集:val | transfer(冻结留出,出口口径)")
    ap.add_argument("--writeback", default="vset_base", choices=["gen4", "vset_base", "pg_only"],
                    help="vset_base=默认交付协议(2026-07-22 定档:仅 v_set←基态常数,除过压,保 pl/pg 自洽);"
                         "gen4=四字段全用生成值(旧协议,仅作历史对照);"
                         "pg_only=仅 pg 用生成,pl←base×load_level / ql←功率因数投影 / v_set←基态常数"
                         "(字段信息账 2026-07-20)")
    ap.add_argument("--calib-mode", default="pooled", choices=["pooled", "perfam"],
                    help="pooled=全局校准(zero-shot 跨);perfam=逐家族自校准(few-shot 天花板)")
    ap.add_argument("--test-fams", default="synth210_s2007,synth150_s2005,synth180_s2006")
    ap.add_argument("--roles", default="上限,中档,下限")
    ap.add_argument("--out", default="runs/g_clabel_bandin/bandin.json")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfgs = [float(x) for x in a.cfgs.split(",")]
    modes = a.modes.split(",")
    fams = a.test_fams.split(",")
    roles = a.roles.split(",")

    if len(roles) < len(fams):
        roles = roles + [""] * (len(fams) - len(roles))

    calib = json.loads((REPO / a.calib_json).read_text(encoding="utf-8"))
    supp = calib["support"]
    S = load_stack(dev, a.denoiser)
    sbit = group_bit_index(Group.STATE)
    scols = np.where(COL_GROUP == sbit)[0]
    solver = PsaspSolver()
    workroot = REPO / "runs" / "g_clabel_bandin_work"
    print(f"[bandin] split={a.split} calib-mode={a.calib_mode} 家族{len(fams)} N={a.n} "
          f"modes={modes} CFG={cfgs}; calib={a.calib_json}")

    Fs = [prep_family(f, a.n, S, dev, a.split) for f in fams]
    report = {"split": a.split, "calib_mode": a.calib_mode, "writeback": a.writeback,
              "denoiser": a.denoiser,
              "test_fams": fams, "roles": roles, "n": a.n, "cfgs": cfgs, "modes": modes,
              "calib_json": a.calib_json, "results": {}, "macro": {}}
    for F, role in zip(Fs, roles):
        report["results"][F["fam"]] = {"role": role, "n_gen": F["true_labels"][0]["n_gen"],
                                       "nodes": F["n"], "cells": {}}
        for c in cfgs:
            for mode in modes:
                cp = resolve_cp(calib, c, F["fam"], a.calib_mode, supp)
                res = run_cell(F, c, mode, cp, S, dev, scols, solver, workroot, a.writeback)
                report["results"][F["fam"]]["cells"][f"{mode}_{c}"] = res
                print(f"[bandin] {F['fam']:16s}({role}) CFG{c} {mode:5s}: "
                      f"收敛 {res['converged']:>2}/{res['n']}({res['conv_rate']:.0%}) "
                      f"带内2s {res['band_2s']:>2}/{res['n']}({res['band_2s_rate']:.0%}) "
                      f"[旧1s {res['band_rate']:.0%} 越上限{res['over_vmax']}] | "
                      f"loadMAE {res['load_mae']:.3f} onlineMAE {res['online_mae']:.2f} | "
                      f"genLoad {res['gen_load_mean']:.3f} vcorr{res['vcorr_applied']:+.3f}",
                      flush=True)
    # [跨] 家族等权 macro(出口主口径,pooled 标 INFLATED 不用)
    for c in cfgs:
        for mode in modes:
            cel = [report["results"][f]["cells"][f"{mode}_{c}"] for f in fams]
            brs = [x["band_rate"] for x in cel]
            b2s = [x["band_2s_rate"] for x in cel]
            crs = [x["conv_rate"] for x in cel]
            # 误差棒用**家族级 SE**(家族是区组因子,占 VAL 方差 77~85%)
            se = (float(np.std(b2s, ddof=1)) / math.sqrt(len(b2s))) if len(b2s) > 1 else None
            report["macro"][f"{mode}_{c}"] = {
                "band_2s_rate_macro": round(float(np.mean(b2s)), 3),
                "band_2s_family_se": None if se is None else round(se, 3),
                "band_rate_macro_1s_LEGACY": round(float(np.mean(brs)), 3),
                "conv_rate_macro": round(float(np.mean(crs)), 3),
                "per_fam_band_2s": {f: x["band_2s_rate"] for f, x in zip(fams, cel)}}
            print(f"[MACRO] CFG{c} {mode:5s}: 带内2s macro {np.mean(b2s):.1%}"
                  f"{'' if se is None else f'±{se:.1%}'} "
                  f"[旧1s {np.mean(brs):.1%}] 收敛 macro {np.mean(crs):.1%}"
                  f"  (家族等权,n={len(fams)})", flush=True)
    shutil.rmtree(workroot, ignore_errors=True)
    outp = REPO / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[bandin] 写出 {outp}")


if __name__ == "__main__":
    main()
