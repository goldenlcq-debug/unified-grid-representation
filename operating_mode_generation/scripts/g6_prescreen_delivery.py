"""[G] G6 · 预筛式交付批产入口(2026-08-03;a 项工程落地,同为 b 项扩池实验工具)。

部署形态:每需求抽 M 个候选(GPU,--m-pool)→ 部署头打分(deploy_head.json,
json 手算不依赖 sklearn)→ 按分数降序 PSASP 试算到首个 band_2s 合格或预算
(--budget)用尽。与 g_bestofk_pilot(z 序试算)同判据同写回,仅顺序不同。

零烧机验证:--dry-run 只采样+打分+排序落盘,不进 PSASP。
烧机(非 dry)须先有预注册与用户放行——本脚本自身不设阈值判据,只出原始记录。

  python scripts/g6_prescreen_delivery.py --dry-run --only synth150_s2000
  python scripts/g6_prescreen_delivery.py --m-pool 16 --seed-base 1000 \\
      --only-unsolved --out runs/g6_ext_pool   # b 项定向扩池(预注册后)
"""
import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import gridfm  # noqa: E402
assert gridfm.__file__.startswith(str(REPO)), f"窜仓:{gridfm.__file__}"

from g_clabel_bandin import (DATA, MODE_FIELDS, VMAX_HI, VMIN_LO,  # noqa: E402
                             build_labels, load_stack, prep_family, resolve_cp)
from g1_baseline import writeback_and_trial  # noqa: E402
from gridfm.data import c_label as CL  # noqa: E402
from gridfm.fsutil import robust_rename, robust_rmtree  # noqa: E402
from gridfm.schema.fields import NODE_TABLE, Group  # noqa: E402
from gridfm.schema.sample import group_bit_index  # noqa: E402
from gridfm.solver.psasp import LicenseError, PsaspSolver  # noqa: E402
from gridfm.train.masking import COL_GROUP  # noqa: E402

CFG_SCALE = 1.5
VAL8 = ["synth210_s2007", "synth150_s2000", "synth210_s2002", "synth240_s2003",
        "synth270_s2004", "synth150_s2005", "synth180_s2001", "synth180_s2006"]
TRANSFER6 = ["synth400_s3000", "synth400_s3001", "synth400_s3002", "synth400_s3003",
             "synth700_s3004", "synth1000_s3008"]
LOAD_TOL, ONLINE_TOL = 0.05, 1          # c_label v1 容差(与 g_bestofk_pilot 同款)
NUM_FEATS = ["n_nodes", "req_load", "req_online", "req_ngen",
             "gen_load", "gen_online", "load_rel_dev", "online_dev",
             "pg_mean", "pg_std", "pg_max", "pg_neg", "ql_std", "pl_std",
             "vset_dev_mean", "vset_dev_std", "vset_dev_maxabs", "vset_hi",
             "z0_std", "z0_maxabs", "z0_clip_frac"]


class DeployHead:
    """deploy_head.json 的手算打分器(免 sklearn 版本耦合)。"""

    def __init__(self, path):
        h = json.loads(Path(path).read_text(encoding="utf-8"))
        assert h["kind"] == "logreg+standardscaler"
        self.names = h["feat_names"]
        self.mean = np.asarray(h["scaler_mean"])
        self.scale = np.asarray(h["scaler_scale"])
        self.coef = np.asarray(h["coef"])
        self.b = h["intercept"]
        self.fp = h["train_data_fp"]

    def score(self, feats: dict, model: str, fam: str) -> float:
        x = []
        for name in self.names:
            if name.startswith("model="):
                x.append(1.0 if name == f"model={model}" else 0.0)
            elif name.startswith("fam="):
                x.append(1.0 if name == f"fam={fam}" else 0.0)
            else:
                x.append(float(feats[name]))
        z = (np.asarray(x) - self.mean) / self.scale
        return float(1 / (1 + np.exp(-(z @ self.coef + self.b))))


def candidate_feats(pu, vset_gen, z0j, F, j, col, gen_ns):
    """与 g5_prescreen_replay 逐项同构的候选特征。"""
    g = copy.deepcopy(F["test_s"][j]); g.node_feat = pu
    gl = CL.recompute(g, F["base_pl"])
    tl = F["true_labels"][j]
    pg = pu[:, col["pg"]][gen_ns]
    ql = pu[:, col["ql"]]
    pl = pu[:, col["pl"]]
    dv = vset_gen[gen_ns] - F["base_feat"][:, col["v_set"]][gen_ns]
    ld = abs(gl["load_level"] - tl["load_level"]) / max(abs(tl["load_level"]), 1e-9)
    return {
        "n_nodes": F["n"], "req_load": round(float(tl["load_level"]), 4),
        "req_online": int(tl["online_count"]), "req_ngen": int(tl["n_gen"]),
        "gen_load": round(float(gl["load_level"]), 4),
        "gen_online": int(gl["online_count"]),
        "load_rel_dev": round(float(ld), 4),
        "online_dev": int(abs(gl["online_count"] - tl["online_count"])),
        "pg_mean": round(float(pg.mean()), 4), "pg_std": round(float(pg.std()), 4),
        "pg_max": round(float(pg.max()), 4), "pg_neg": int((pg < -1e-6).sum()),
        "ql_std": round(float(ql.std()), 4), "pl_std": round(float(pl.std()), 4),
        "vset_dev_mean": round(float(dv.mean()), 4),
        "vset_dev_std": round(float(dv.std()), 4),
        "vset_dev_maxabs": round(float(np.abs(dv).max()), 4),
        "vset_hi": int((vset_gen[gen_ns] > 1.06).sum()),
        "z0_std": round(float(z0j.std()), 4),
        "z0_maxabs": round(float(np.abs(z0j).max()), 4),
        "z0_clip_frac": round(float((np.abs(z0j) > 3.98).mean()), 5),
    }


def _balance_fix(pred, F, j, col, gen_ns, use_base_ql=False, blend=1.0):
    """[外协包 F,2026-08-26] 功率平衡校正 + 预校验(技术方案 6.1"生成后做功率平衡预校验,
    偏差超平衡机可调范围的样本直接打回重采样"——该步在 g6 里从未实现)。

    做法(只用需求本身给定的信息,不用任何留出真值):
      · 目标总负荷 = 需求 load_level x 基态 ΣPl;把生成的 pl/ql **同比缩放**到该总量;
      · 目标总出力 = 同一 load_level x 基态 Σ非平衡机 Pg;把生成的 pg 同比缩放;
        (与 sampler.sample_mode 的"负荷与出力按同因子缩放、平衡机吸收残差"同款口径)
      · 两步都只改标量总量,**逐元件的相对分布(扩散器真正生成的内容)原样保留**;
      · 缩放后若残余不平衡仍超平衡机 [Pmin,Pmax] 可调范围,标记 bal_reject=1(调用方判废)。
    先例:同管线的 vset_base 协议直接把生成的 v_set 整列换成基态常数,本步比它保守得多。
    """
    pred = pred.copy()
    i_pl, i_ql, i_pg = 0, 1, 2                      # MODE_FIELDS = (pl, ql, pg, v_set)
    tl = F["true_labels"][j]
    lf = float(tl["load_level"])                    # 需求的负荷水平(= c_label 所指定)
    bf = F["base_feat"]
    # col 只含 MODE_FIELDS,has_load 需从 NODE_TABLE 直接取偏移
    has_load = bf[:, NODE_TABLE.offset_of("has_load")] > 0.5
    pl_base = float(bf[has_load, col["pl"]].sum())
    pg_base = float(bf[gen_ns, col["pg"]].sum())

    pl_gen = float(pred[has_load, i_pl].sum())
    if abs(pl_gen) > 1e-6 and pl_base > 0:
        k = (lf * pl_base) / pl_gen
        pred[has_load, i_pl] *= k
        pred[has_load, i_ql] *= k
    if use_base_ql:
        # 无功挂基态功率因数:ql_i = lf x 基态 ql_i(空间分布取基态,总量随需求)
        pred[has_load, i_ql] = lf * bf[has_load, col["ql"]]
    if blend < 1.0:
        # 基态锚定混合(先混合,后总量缩放):向"比例调度"收缩,保留 α 份生成的空间分布
        pred[gen_ns, i_pg] = (blend * pred[gen_ns, i_pg]
                              + (1.0 - blend) * lf * bf[gen_ns, col["pg"]])
    pg_gen = float(pred[gen_ns, i_pg].sum())
    if abs(pg_gen) > 1e-6 and pg_base > 0:
        pred[gen_ns, i_pg] *= (lf * pg_base) / pg_gen

    resid = float(pred[gen_ns, i_pg].sum() - pred[has_load, i_pl].sum())
    return pred, {"bal_load_k": round(k if pl_gen else 1.0, 4),
                  "bal_resid_pu": round(resid, 3),
                  "bal_reject": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-pool", type=int, default=8)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--denoiser", default="runs/diffusion_g1_cgen_g4_s1/denoiser.pt")
    ap.add_argument("--calib-json", default="runs/g_clabel_calib/calib_s1.json")
    ap.add_argument("--head", default="runs/g5_prescreen/deploy_head.json")
    ap.add_argument("--out", default="runs/g6_prescreen_delivery")
    ap.add_argument("--balance-fix", action="store_true",
                    help="[外协包 F] 功率平衡校正 + 预校验闸(技术方案 6.1 明示但从未实现的一步):"
                         "把生成的 pl/ql 与 pg 各自按标量缩放到'需求 load_level x 基态总量',"
                         "保留扩散器给出的空间分布,只校准总量;残余不平衡仍超平衡机可调范围的"
                         "候选直接判废、不进 PSASP。缺省关 = 与原实现逐位一致。"
                         "背景:大网上 20%% 的相对负荷偏差 = 96 pu 绝对不平衡,平衡机能力的 10 倍,"
                         "必然发散(2000 档实测 480/480 全不收敛)。")
    ap.add_argument("--balance-fix-blend", type=float, default=1.0,
                    help="[外协包 F] 机组有功空间分布的基态锚定混合系数 α∈(0,1]:"
                         "pg_i = α·生成_i + (1-α)·(load_level x 基态 pg_i),再整体缩放到目标总量。"
                         "1.0(缺省)=纯生成;0.5=对半。动机:重载段(load>=0.98) 361 候选全不收敛,"
                         "且换基态功率因数无效(ql 变体同 40/60)——瓶颈是生成调度过于新颖"
                         "(实测到训练集最近邻 = 库内间距 32 倍),重载收敛域包不住;向可行调度收缩。")
    ap.add_argument("--balance-fix-ql", action="store_true",
                    help="[外协包 F] 在 --balance-fix 基础上,负荷无功改挂基态功率因数:"
                         "ql_i = load_level x 基态 ql_i(逐母线),弃用生成的 ql。"
                         "动机:2000 档重载段(load>=0.98)160 候选全不收敛——重载收敛对无功"
                         "极敏感,生成的 ql 空间分布误差在小网被电压裕度吸收,大网被放大。"
                         "与 vset_base 同哲学(不可观测量挂基态);生成的 pl/pg 空间分布保留。")
    ap.add_argument("--extra-plan", default=None,
                    help="[外协包 F] 追加评测计划,格式 split:fam:n_req[,...]。用于把大网族纳入"
                         "评测而**不动冻结的 val/transfer 划分**——例如 "
                         "holdout_f:f2000_ho:60。split 是 families/ 下的子目录名。")
    ap.add_argument("--only-extra", action="store_true",
                    help="[外协包 F] 只跑 --extra-plan,跳过 val8/transfer6(分层评测时省机时)")
    ap.add_argument("--set", dest="which", default="val8", choices=("val8", "full"),
                    help="full = VAL8(N=20)+ transfer6(N=15),交付批产口径")
    ap.add_argument("--only", default=None, help="只跑指定家族")
    ap.add_argument("--only-unsolved", action="store_true",
                    help="b 项定向:只跑 s1 k8 未解需求(名单取自 g3 交集报告)")
    ap.add_argument("--exhaust", action="store_true",
                    help="全试 budget 个不早停(实验口径:逐候选三态全落盘;"
                         "部署口径默认早停省仿真)")
    ap.add_argument("--dry-run", action="store_true", help="只采样+打分,不进 PSASP")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    head = DeployHead(REPO / a.head)
    only_sids = None
    if a.only_unsolved:
        assert a.which == "val8", "--only-unsolved 名单仅覆盖 val8 层"
        # s1 失败名单 = delivery unsolved ∧ ext still_unsolved(与 g3 交集诊断同源)
        only_sids = set()
        dv = json.loads((REPO / "runs/g2_delivery_v1/report.json").read_text(encoding="utf-8"))
        ux = json.loads((REPO / "runs/g2_union_ext/report.json").read_text(encoding="utf-8"))
        for f in VAL8:
            k5_uns = set(dv["families"][f]["unsolved_sids"])
            ext = ux["arms"]["s1_ext"]["families"].get(f)
            if ext:
                assert ext["pending_in"] == len(k5_uns), f      # m4 名单护栏(g3 同款)
                still = set(ext["still_unsolved"])
                assert still <= k5_uns, f
            else:
                still = k5_uns
            only_sids |= still
        print(f"[g6] 定向名单(s1 val8 k8 未解):{len(only_sids)} 需求", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = load_stack(dev, a.denoiser)
    calib = json.loads((REPO / a.calib_json).read_text(encoding="utf-8"))
    supp = calib["support"]
    o = NODE_TABLE.offset_of
    col = {f: o(f) for f in MODE_FIELDS}
    sbit = group_bit_index(Group.STATE)
    scols = np.where(COL_GROUP == sbit)[0]
    solver = None if a.dry_run else PsaspSolver()
    rp = out / "report.json"
    task_desc = (f"G6 预筛式交付(M={a.m_pool}, budget={a.budget}, "
                 f"seed_base={a.seed_base}, head_fp={head.fp[:8]}, "
                 f"denoiser={a.denoiser}, "
                 f"dry={bool(a.dry_run)}, set={a.which}, only={a.only}, "
                 f"only_unsolved={bool(a.only_unsolved)}, exhaust={bool(a.exhaust)})")
    if rp.exists():                          # 断点续族:已完成族跳过
        report = json.loads(rp.read_text(encoding="utf-8"))
        assert report["task"] == task_desc, \
            f"续跑参数与已有 report 不一致(M1 守卫):\n已有 {report['task']}\n本次 {task_desc}"
        report["partial"] = True
        print(f"[g6] 续跑:已有 {len(report['families'])} 族", flush=True)
    else:
        report = {"task": task_desc, "dry_run": bool(a.dry_run),
                  "partial": True, "families": {}}

    def flush():
        tmp = rp.with_suffix(".json.tmp")    # m2 原子写
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(rp)

    plan = [] if a.only_extra else [("val", f, 20) for f in VAL8]
    if a.which == "full" and not a.only_extra:
        plan += [("transfer", f, 15) for f in TRANSFER6]
    if a.extra_plan:                       # [外协包 F] 追加大网层,split 为 families/ 子目录名
        for tri in a.extra_plan.split(","):
            sp, fm, nn = tri.strip().split(":")
            plan.append((sp, fm, int(nn)))
    t00 = time.time()
    for split, fam, n_req_fam in plan:
        if a.only and fam != a.only:
            continue
        if fam in report["families"]:
            continue
        F = prep_family(fam, n_req_fam, S, dev, split)
        req_idx = [j for j in range(F["N"])
                   if only_sids is None or F["test_s"][j].sample_id in only_sids]
        if not req_idx:
            continue
        cp = resolve_cp(calib, CFG_SCALE, fam, "pooled", supp)
        lab_c = build_labels(F, "uncal", cp, S["lnorm"], dev)
        lab_u = torch.stack([torch.from_numpy(CL.encode(None, S["lnorm"])).to(dev)] * F["N"])
        n, Nf = F["n"], F["N"]
        net, dec, diff = S["net"], S["dec"], S["diff"]
        z_mean, z_std = S["z_mean"], S["z_std"]
        gen_ns = F["gen_ns"]

        @torch.no_grad()
        def eps_fn(z_t, t):
            tt = torch.full((Nf,), t, dtype=torch.long, device=dev)
            e_c = net(z_t, F["hc"], F["ec"], F["ei"], tt, lab_c, F["npg"])
            e_u = net(z_t, F["hc"], F["ec"], F["ei"], tt, lab_u, F["npg"])
            return e_u + CFG_SCALE * (e_c - e_u)

        base_vset = F["base_feat"][:, col["v_set"]]
        cand = defaultdict(list)          # j -> [(score, z_seed, pred)]
        for r in range(a.m_pool):
            z_seed = a.seed_base + r
            torch.manual_seed(z_seed)
            z_T = torch.randn(Nf * n, S["d"], device=dev)
            with torch.no_grad():
                z0 = diff.ddim_sample(eps_fn, z_T, steps=250, clip_x0=4.0)
                fields = dec(z0 * z_std + z_mean, F["ec"], F["ei"]).cpu().numpy()
            z0c = z0.cpu().numpy()
            for j in req_idx:
                full = F["normed"][j].node_feat.copy()
                full[:, scols] = fields[j * n:(j + 1) * n][:, scols]
                pu = S["norm"].inverse_transform_node(full)
                vset_gen = pu[:, col["v_set"]].copy()
                pu[:, col["v_set"]] = base_vset
                feats = candidate_feats(pu, vset_gen, z0c[j * n:(j + 1) * n],
                                        F, j, col, gen_ns)
                sc = head.score(feats, "s1", fam)
                pred = np.stack([pu[:, col[f]] for f in MODE_FIELDS], axis=1)
                if a.balance_fix:
                    pred, bfx = _balance_fix(pred, F, j, col, gen_ns,
                                             use_base_ql=a.balance_fix_ql,
                                             blend=a.balance_fix_blend)
                    # 记账修正(2026-08-26):特征原在校正**前**算,导致 load_rel_dev/label_ok
                    # 记的是校正前的旧值(实测报 0/20,而实际负荷偏差已是 0.0000)。
                    # 此处把校正后的量回写进 pu 再重算特征,使 label_ok 反映**真实交付内容**。
                    pu2 = pu.copy()
                    for _i, _f in enumerate(MODE_FIELDS):
                        pu2[:, col[_f]] = pred[:, _i]
                    feats = candidate_feats(pu2, vset_gen, z0c[j * n:(j + 1) * n],
                                            F, j, col, gen_ns)
                    feats.update(bfx)
                cand[j].append((sc, z_seed, pred, feats))
        famrec = {"split": split, "n_req": len(req_idx), "m_pool": a.m_pool,
                  "solved": {}, "unsolved_sids": [], "psasp_calls": 0,
                  "rank_tables": {}, "trials": {}}
        for j in req_idx:
            sid = F["test_s"][j].sample_id
            ranked = sorted(cand[j], key=lambda t: (-t[0], t[1]))
            famrec["rank_tables"][sid] = [
                {"z_seed": z, "score": round(s, 4)} for s, z, _p, _f in ranked]
            if a.dry_run:
                continue
            trials = []                     # F1:逐候选三态+特征全落盘
            first_ok = None                 # (pos, z_seed, work)
            for pos, (sc, z_seed, pred, feats) in enumerate(
                    ranked[:a.budget], start=1):
                work = out / "work" / fam / f"{sid}_z{z_seed}"
                conv, vmin, vmax = writeback_and_trial(
                    pred, F["famdir"] / "cases" / sid, work, solver)
                famrec["psasp_calls"] += 1
                ok = bool(conv and vmin is not None and vmin >= VMIN_LO
                          and vmax is not None and vmax <= VMAX_HI)
                lab = "ok" if ok else ("viol" if conv else "nonconv")
                trials.append({
                    "rank": pos, "z_seed": z_seed, "score": round(sc, 4),
                    "conv": bool(conv),
                    "vmin": None if vmin is None else round(float(vmin), 4),
                    "vmax": None if vmax is None else round(float(vmax), 4),
                    "label": lab, "feats": feats})
                if ok and first_ok is None:
                    first_ok = (pos, z_seed, work)
                    famrec["solved"][sid] = {
                        "tried": pos, "z_seed": z_seed, "score": round(sc, 4),
                        "vmin": round(float(vmin), 4), "vmax": round(float(vmax), 4),
                        # c_label 记账(交付协议 8.8 口径,取自 candidate_feats)
                        "gen_load": feats["gen_load"], "gen_online": feats["gen_online"],
                        "load_rel_dev": feats["load_rel_dev"],
                        "online_dev": feats["online_dev"],
                        "label_ok": bool(feats["load_rel_dev"] <= LOAD_TOL
                                         and feats["online_dev"] <= ONLINE_TOL)}
                    if not a.exhaust:
                        break
                    continue                # exhaust:留 work 到收尾再定去留
                if lab == "viol":           # 越限抽签留档 = 调整器重测输入(批产惯例)
                    dst = out / "violations" / fam / f"{sid}_z{z_seed}"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    robust_rmtree(dst)
                    if work.exists():
                        robust_rename(work, dst)      # [F 20260828] Windows 瞬态占用重试
                else:
                    robust_rmtree(work)
            famrec["trials"][sid] = trials
            # [外协包 F,2026-08-26 审计 P0-2] 逐需求增量落盘:单族计划下原实现只在族收尾
            # flush,中途被杀/异常 = 数小时 PSASP 全损(实证:首次 2000 交付测试被杀后零记录)。
            report["families"][fam] = famrec
            flush()
            if first_ok is None:
                famrec["unsolved_sids"].append(sid)
            else:
                _pos, _z, wk = first_ok
                dst = out / "delivered" / fam / sid
                dst.parent.mkdir(parents=True, exist_ok=True)
                robust_rmtree(dst)
                if wk.exists():
                    robust_rename(wk, dst)            # [F 20260828] Windows 瞬态占用重试
        report["families"][fam] = famrec
        ns = len(famrec["solved"])
        print(f"[g6] {fam}: {'dry' if a.dry_run else f'{ns}/{len(req_idx)} 解,'}"
              f"{famrec['psasp_calls']} 次 PSASP({time.time()-t00:.0f}s)", flush=True)
        flush()
        del F
        torch.cuda.empty_cache()
    planned = [f for _sp, f, _n in plan
               if (not a.only or f == a.only)
               and (only_sids is None
                    or any(s.rsplit("_", 1)[0] == f for s in only_sids))]
    if all(f in report["families"] for f in planned):   # M2:计划全跑完才敢自称完整
        report["partial"] = False
        if not a.dry_run:                   # macro 汇总(批产口径,分层禁并池)
            macro = {}
            for lname in ("val", "transfer"):
                fs = [F2 for F2 in report["families"].values()
                      if F2["split"] == lname]
                if not fs:
                    continue
                n_req = sum(F2["n_req"] for F2 in fs)
                n_sol = sum(len(F2["solved"]) for F2 in fs)
                calls = sum(F2["psasp_calls"] for F2 in fs)
                lab_ok = sum(1 for F2 in fs for v in F2["solved"].values()
                             if v.get("label_ok"))
                macro[lname] = {
                    "n_req": n_req, "delivered": n_sol,
                    "rate": round(n_sol / n_req, 4) if n_req else None,
                    "psasp_calls": calls,
                    "calls_per_delivered": round(calls / n_sol, 2) if n_sol else None,
                    "label_consistency": round(lab_ok / n_sol, 3) if n_sol else None}
            report["macro"] = macro
    flush()
    print(f"[g6] 完成 → {rp}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LicenseError:
        print("[g6] slm_init error:加密狗失效,部分已落盘,中止等用户", flush=True)
        raise SystemExit(2)
