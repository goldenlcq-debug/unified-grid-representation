# -*- coding: utf-8 -*-
r"""调整器效果提升线(2026-08-22):跨规模考卷模型臂跑分器(进程内、PSASP 唯一裁判、串行)。

对 libs/viol_<q|p>_<tier>/cases/* 逐例:备份 → 初判(1 次 PSASP)→ 档案化(与部署包 run_adjust 同口径:family "deploy"、schema v0)
→ 编码器(冻结 V1)+ 推理器 → 解码动作 → 写卡 → 复核(1 次 PSASP)。与 run_adjust.py 的差别全部显式:
  ① 进程内批处理(同拓扑 LaPE 只算一次;run_adjust 每例一个进程);② 可换检查点/变体(--ckpt v5|v06|<训练输出目录>);
  ③ 新变体的解码附加项:有功 v07 的平衡机裕度投影(--no-proj 可关)、无功 v6 的台数头 top-k、任意无功检查点的 --cap-k 截顶;
  ④ **模型引导的阶梯(--ladder,2026-08-23 新形态)**:首步失败 → 回退病态起点 → 按指针概率排序扩台数:
     无功 2/4/8 台 × 0.02/0.05(老师同款顺序、只换排序),有功 2/4/8/16/32 台 × 平衡机超额倍数(老师同款分摊,
     平衡机低于 Pmin 时零和对调);每阶 1 次 PSASP,每例总次数上限 --ladder-max-pf(缺省 8 = 初判 + 首步 + 6 阶);
     同一次跑同时记"首步"与"阶梯"两套数字和"成功数随 PSASP 次数"的曲线;
  ⑤ **动态列填法开关(--dyn-fill)**:auto = 按检查点配置(v5/v06 与老座 = zero);zero = 现状(常量伪值);
     mean = 动态组缺失的机组节点四列填归一化均值(adj2_common.apply_dyn_fill)。
规则臂不重跑:沿用 B 段 调整器跨规模验证-20260821/runs/<task>_<tier>/arms_results.json 的规则臂结果(同一病例、同一代码、确定性),
级联按"模型成功取模型,否则取规则"推导(与 run_adjust 级联逐位同构,B 段 118 档已实跑对账)。
报:成功率 + Wilson95 + 每例 PSASP 次 + 修后 vmin/vmax + 实际干预(改机数/Σ|Δ|/推满台数/投切)+ 合理性(有功:平衡机相对判据)
+ 重复运行动作是否逐位一致(--repeat)。产物 调整器效果提升-20260822/runs/eval_<name>/{arms_results.json,summary.json}。

  python adj2_eval.py --task reactive --tier 118 --ckpt v5 --out-name q118_v5_base --repeat 2
  python adj2_eval.py --task reactive --tier 3120 --ckpt ../调整器效果提升-20260822/runs/train_q_v6d_f1_s0 --out-name q3120_v6d_s0_lad --ladder
"""
from __future__ import annotations

import argparse
import copy as _copy
import json
import math
import shutil
import sys
import time
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import adj2_common as C  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from gridfm.data.graph import sample_to_data  # noqa: E402
from gridfm.data.normalize import Normalizer  # noqa: E402
from gridfm.data.synth import psasp_referee  # noqa: E402
from gridfm.fsutil import robust_copytree  # noqa: E402
from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.psasp_io.catalog import Catalog  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402
from gridfm.reason.adjust_task import AdjustReasoner, unit_span, viol_depth  # noqa: E402
from gridfm.reason.reactive_task import (  # noqa: E402
    DV_LEVELS, build_reactive_reasoner, decode_action, react_side, shunt_cand_mask, unit_span_vset)
from gridfm.reason.ts_task import gen_node_mask  # noqa: E402
from gridfm.schema.assemble import build_sample  # noqa: E402
from gridfm.solver.psasp import LicenseError, PsaspSolver  # noqa: E402
from gridfm.train.pretrain import TrainConfig, build_models  # noqa: E402
from scripts.adjust_labels import read_gens as read_gens_repo  # noqa: E402
from scripts.adjust_materialize import apply_fix_by_bus  # noqa: E402
from scripts.reactive_apply_eval import apply_mixed  # noqa: E402
from realized_actions import realized  # noqa: E402
from psasp_guard import guarded  # noqa: E402  2026-09-05：加密狗不响应守卫

BIG = {"25k", "70k"}
EXAM_TIERS = {"118", "300", "2000", "3120", "25k", "70k", "1354", "2869", "9241"}   # 陌生网泛化档 2026-08-27 加(300/1354/2869/9241)
LADDER_DEFAULT = {"reactive": ("2,4,8", "0.02,0.05"), "active": ("2,4,8,16,32", "1.0")}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def copy_files(src: Path, dst: Path) -> None:
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)


def restore_files(bak: Path, root: Path) -> None:
    for p in root.iterdir():
        if p.is_file():
            p.unlink()
    copy_files(bak, root)


def prune_big(work: Path) -> None:
    keep = {"LF.L5", "LF.L2", "LF.L6", "lfreport.lis", "LF.CAL", "LF.LP1"}
    for p in work.iterdir():
        if p.is_file() and p.name not in keep:
            p.unlink()
    bak = work / "_original_backup"
    if bak.is_dir():
        for p in bak.iterdir():
            if p.is_file() and not p.name.endswith(".json") and p.name not in keep:
                p.unlink()


class Counting:
    def __init__(self, inner):
        self._inner = inner
        self.n_pf = 0

    def run_powerflow(self, *a, **k):
        self.n_pf += 1
        return self._inner.run_powerflow(*a, **k)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)


def local_gen_mask(raw, case_dir: Path, n_bus: int, low: float, high: float, hops_max: int):
    """局部性预筛：返回"距越限母线 ≤hops_max 跳"的机组布尔掩码（1 基母线号 → 节点行）。
    越限母线取自该病例已解出的 LF.LP1（低压或高压皆算病灶）。找不到病灶则不限制（全 True）。"""
    import numpy as _np
    v = C.bus_voltage(case_dir, n_bus)
    live = v > 0.01
    bad = [int(i) + 1 for i in _np.where(live & ((v < low - 1e-4) | (v > high + 1e-4)))[0]]
    n = raw.n_nodes
    if not bad:
        return _np.ones(n, dtype=bool)
    from gridfm.psasp_io.case import PsaspCase as _PC
    hops = C.hops_from(C.bus_adjacency(_PC(case_dir)), bad, max_h=hops_max)
    m = _np.zeros(n, dtype=bool)
    for b in hops:
        if 1 <= b <= n:
            m[b - 1] = True
    return m


def load_reasoner(task: str, ckpt: str, dev, d: int):
    """返回 (reasoner, variant, rcfg, 路径)。v5/v06 = 部署包基线;其他 = adj2_train 输出目录。"""
    if ckpt in ("v5", "v06"):
        p = C.PKG / "runs" / ("reactive_reasoner_v5_s0" if ckpt == "v5" else "adjust_reasoner_v06_s0") / "ckpt" / "best.pt"
        b = torch.load(p, map_location=dev, weights_only=False)
        rcfg = b["config"]
        if task == "active":
            r = AdjustReasoner(d=rcfg["d"], n_queries=rcfg["queries"], n_layers=rcfg["layers"])
            variant = "v06"
        else:
            r = build_reactive_reasoner(d=rcfg["d"], n_queries=rcfg["queries"], n_layers=rcfg["layers"],
                                        with_shunt=bool(rcfg.get("with_shunt", False)), n_dv_levels=int(rcfg.get("n_dv_levels", 0)))
            variant = "v5"
    else:
        from adj2_train import build_model
        p = Path(ckpt) / "ckpt" / "best.pt"
        b = torch.load(p, map_location=dev, weights_only=False)
        rcfg = b["config"]
        variant = rcfg["variant"]
        r, _ = build_model(task, variant, rcfg["d"], rcfg["queries"], rcfg["layers"])
    r.load_state_dict(b["reasoner_state"])
    r.to(dev).eval()
    return r, variant, rcfg, str(p)


def _parse_list(s: str, typ):
    return [typ(x) for x in str(s).split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["active", "reactive"])
    ap.add_argument("--tier", required=True, choices=sorted(EXAM_TIERS))
    ap.add_argument("--ptr-local-hops", type=int, default=0,
                    help="指针候选局部性预筛：只保留距越限母线 ≤H 跳的机组（0=关闭，缺省关闭行为逐位不变）。"
                         "动因（2026-08-26 实测）：25k 上模型首选机组距病灶中位 45 跳、规则臂 5 跳，"
                         "指针在 2753 台候选里无区分度（超阈 314 台）——超大网失效的根因是候选集没约束。")
    ap.add_argument("--ckpt", required=True, help="v5 | v06 | adj2_train 输出目录")
    ap.add_argument("--out-name", required=True)
    ap.add_argument("--lib-name", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1, help="模型前向重复次数(动作逐位一致则不重复 PSASP)")
    ap.add_argument("--no-proj", action="store_true", help="有功 v07:关闭平衡机裕度投影")
    ap.add_argument("--cap-k", type=int, default=None, help="无功:指针过阈机组按概率截顶到 k 台(最小干预推理期变体)")
    ap.add_argument("--no-reuse-rule", action="store_true")
    ap.add_argument("--mock", action="store_true", help="无 PSASP 冒烟:裁判只读目录里已有的 LF.LP1(结论无意义,只验管线)")
    ap.add_argument("--sample-id-mode", default="case", choices=["case", "dirname"],
                    help="档案 sample_id(决定持久化随机特征列):case=病例 id(缺省);dirname=工作目录名 'model'"
                         "(= 部署包 run_adjust 的实际行为:B 段全部病例共用同一套随机列,对拍用)")
    ap.add_argument("--rf-seed", type=int, default=0,
                    help=">0 时随机特征列改用 seed(病例 id, rf_seed) 重抽(与持久化列不同),用于多次运行取区间")
    ap.add_argument("--dyn-fill", default="auto", choices=["auto"] + list(C.DYN_FILL_MODES),
                    help="动态列填法:auto=按检查点配置(缺省;v5/v06 与老座 = zero);zero=常量伪值现状;"
                         "mean=缺失动态组的机组节点四列填归一化均值;mean_all=所有缺失节点都填(对照项)")
    ap.add_argument("--ladder", action="store_true", help="模型引导的阶梯:首步失败后按指针排序扩台数(每阶 1 次 PSASP)")
    ap.add_argument("--ladder-ks", default=None, help="阶梯台数(逗号分隔;缺省 无功 2,4,8 / 有功 2,4,8,16,32)")
    ap.add_argument("--ladder-mags", default=None,
                    help="阶梯量级(逗号分隔;缺省 无功 Δv_set 0.02,0.05 / 有功 平衡机超额倍数 1.0;"
                         "有功无超额时 = 越限深度×Σspan×0.5×倍数)")
    ap.add_argument("--ladder-max-pf", type=int, default=8, help="每例 PSASP 总次数上限(含初判 + 首步;缺省 8)")
    ap.add_argument("--vset-cap", default="auto",
                    help="无功模型臂 v_set 上限守门(与部署包同语义):auto=电压上限 high;none=关;或数值。规则臂不受影响")
    a = ap.parse_args()

    disease = "p" if a.task == "active" else "q"
    lib = (Path(a.lib_name) if (a.lib_name and Path(a.lib_name).is_absolute()) else C.LINE / "libs" / (a.lib_name or "viol_%s_%s" % (disease, a.tier)))   # 2026-09-05: 允许绝对路径考卷
    summ = json.loads((lib / "eval_summary.json").read_text(encoding="utf-8"))
    low, high = summ["vio_band"]
    vcap = None
    if a.task == "reactive" and str(a.vset_cap).lower() != "none":
        vcap = float(high) if str(a.vset_cap).lower() == "auto" else float(a.vset_cap)
    cases = sorted(p for p in (lib / "cases").iterdir() if p.is_dir())
    if a.limit:
        cases = cases[:a.limit]
    out_dir = C.OUT / "runs" / ("eval_%s" % a.out_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "arms_results.json"
    results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.is_file() else {}
    rule_src = {}
    if not a.no_reuse_rule:
        rp = C.B_RUNS / ("%s_%s" % (a.task, a.tier)) / "arms_results.json"
        if rp.is_file():
            rule_src = json.loads(rp.read_text(encoding="utf-8"))

    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    v1 = C.REPO / "runs" / "encoder_v1"
    ck = torch.load(v1 / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in allowed})
    enc, _ = build_models(cfg)
    enc.load_state_dict(ck["model_state"])
    enc.to(dev).eval()
    norm = Normalizer.load(v1 / "normalizer.npz")
    reasoner, variant, rcfg, ckpt_path = load_reasoner(a.task, a.ckpt, dev, cfg.d)
    dyn_mode = (rcfg.get("dyn_fill") or "zero") if a.dyn_fill == "auto" else a.dyn_fill
    ks_s, mags_s = LADDER_DEFAULT[a.task]
    ladder_ks = _parse_list(a.ladder_ks or ks_s, int)
    ladder_mags = _parse_list(a.ladder_mags or mags_s, float)
    print("[adj2-eval] %s tier=%s 病例 %d 带=[%.2f,%.2f] ckpt=%s(%s) 设备=%s dyn_fill=%s 阶梯=%s 已有 %d 复用规则臂 %d"
          % (a.task, a.tier, len(cases), low, high, a.ckpt, variant, dev, dyn_mode,
             ("%s×%s ≤%d次" % (ladder_ks, ladder_mags, a.ladder_max_pf)) if a.ladder else "关",
             len(results), len(rule_src)), flush=True)
    solver = Counting(PsaspSolver(psasp_bin=str(C.PSASP_BIN)))
    def referee(w, s):                 # 2026-09-05：加密狗不响应 → 等待重试，重试用尽熔断；重试不计 PSASP 次数
        c_ = getattr(s, "n_pf", None)
        rf_ = guarded(lambda: psasp_referee(w, s), w, lambda r: bool(r.get("converged")), tag=Path(w).parent.name)
        if c_ is not None:
            s.n_pf = c_ + 1
        return rf_
    if a.mock:
        class _MockSolver:
            n_pf = 0
        solver = _MockSolver()

        def _mock_referee(w, _s):
            solver.n_pf += 1
            v = C.bus_voltage(Path(w), 10 ** 6)
            v = v[:max(1, int(np.max(np.nonzero(v)[0]) + 1))] if v.any() else v
            vv = v[v > 0.01]
            return {"converged": True, "vmin": float(vv.min()), "vmax": float(vv.max()), "v": v}
        referee = _mock_referee
    t_all = time.time()

    def judge_ok(rf):
        # 2026-08-24 A 项:由严格口径改为 eps 口径(vmin ≥ low−eps,无功再 vmax ≤ high+eps),
        # 与部署包 adjust_ext.judge_ok(--eps 5e-4 缺省)及用户裁决"模型臂统一 eps"对齐;
        # 动因:v_set 守门把机端钉在 cap=high,潮流解出 1.0800x,严格口径(≤ high)必判败——守门与严格口径天然矛盾。
        if not bool(rf.get("converged")):
            return False
        if a.task == "active":
            return rf.get("vmin", 0.0) >= low - C.EPS
        return rf.get("vmin", 0.0) >= low - C.EPS and rf.get("vmax", 9.9) <= high + C.EPS

    def forward_action(raw, l5_case: Path, vmin0=None):
        """档案 → (动作, 诊断, 辅助{prob, cand, ...})。动作:有功 {bus:ΔP} / 无功 {"dvset":{bus:Δv},"shunt":[bus]}。"""
        s = _copy.deepcopy(raw)
        norm.transform(s)
        n_filled = C.apply_dyn_fill(s, raw, dyn_mode)
        rf = brf = None
        if a.rf_seed > 0:
            from gridfm.schema.sample import stable_seed
            rng_rf = np.random.default_rng(stable_seed(raw.family_id, raw.sample_id, "rf%d" % a.rf_seed))
            rf = rng_rf.standard_normal((raw.n_nodes, cfg.random_dim)).astype(np.float32)
            brf = rng_rf.standard_normal((raw.n_branches, cfg.branch_random_dim)).astype(np.float32)
        data = sample_to_data(s, cfg.lape_k, cfg.rwse_steps, random_dim=cfg.random_dim, random_feat=rf,
                              branch_random_dim=cfg.branch_random_dim, branch_random_feat=brf)
        info = {"dyn_filled_nodes": n_filled}
        if a.task == "active":
            if variant == "v06":
                data.depth = torch.from_numpy(viol_depth(raw, low))
            else:
                from adj2_train import side_feats as _sf
                data.depth = torch.from_numpy(_sf("active", variant, raw, low, high))
            batch = Batch.from_data_list([data]).to(dev)
            k_head = None
            with torch.no_grad():
                h, g, e = enc(batch)
                if variant == "v07d":
                    ptr, dp, kl_ = reasoner(h, g, e, batch.batch, batch.batch[batch.edge_index[0]], batch.depth, n_graphs=1)
                    k_head = max(1, int(kl_.float().argmax(-1)[0]))
                else:
                    ptr, dp = reasoner(h, g, e, batch.batch, batch.batch[batch.edge_index[0]], batch.depth)
            l5 = Card.load(l5_case)
            _slack, gens = read_gens_repo(l5)
            adjustable = {int(x["bus"]) for x in gens.values()}
            gm = gen_node_mask(raw).copy()
            for i in range(len(gm)):
                if gm[i] and (i + 1) not in adjustable:
                    gm[i] = False
            if variant in ("v07", "v06re", "v07d"):
                # 新变体:固定出力机组(pmax<=pmin)不可调(与标签老师、训练掩码同口径);v06 基线保持部署包原口径以便对拍
                from gridfm.schema.fields import NODE_TABLE as _NT
                pmax_ = raw.node_feat[:, _NT.offset_of("pmax")]
                pmin_ = raw.node_feat[:, _NT.offset_of("pmin")]
                gm = gm & (pmax_ > pmin_ + 1e-9)
            if a.ptr_local_hops > 0:
                lm = local_gen_mask(raw, Path(l5_case).parent, 10 ** 6, low, high, a.ptr_local_hops)
                if (gm & lm).any():
                    info["ptr_local_kept"] = int((gm & lm).sum()); info["ptr_local_all"] = int(gm.sum())
                    gm = gm & lm
                else:
                    info["ptr_local_kept"] = 0; info["ptr_local_all"] = int(gm.sum())
            if not gm.any():
                return None, {"error": "no_adjustable"}, {}
            prob = torch.sigmoid(ptr.float()).cpu().numpy()
            span = unit_span(raw)
            if k_head is not None:
                rows_ = np.where(gm)[0]
                order_ = rows_[np.argsort(-prob[rows_], kind="stable")]
                sel = [int(i) for i in order_[:k_head]]
            else:
                sel = [int(i) for i in np.where(gm & (prob > 0.5))[0]]
            if not sel:
                sel = [int(np.where(gm)[0][int(np.argmax(prob[gm]))])]
            dpv = dp.float().cpu().numpy()
            action = {int(i) + 1: round(float(dpv[i]) * float(span[i]), 4) for i in sel}
            info.update({"n_sel": len(sel), "n_over_thr": int((gm & (prob > 0.5)).sum()), "net_raw": round(sum(action.values()), 4),
                         "k_head": k_head})
            if variant in ("v07", "v07d") and not a.no_proj:
                up, dn = C.slack_headroom_from_raw(raw)
                action, sc = C.project_action_to_slack(action, up, dn)
                action = {b: round(v, 4) for b, v in action.items()}
                info.update({"proj_scale": round(sc, 4), "slack_up": round(up, 3), "slack_dn": round(dn, 3)})
            return action, info, {"prob": prob, "cand": gm, "span": span}
        with_shunt = bool(rcfg.get("with_shunt", False))
        n_dv = int(rcfg.get("n_dv_levels", 0))
        if variant == "v5":
            data.depth = torch.from_numpy(react_side(raw, low, high, with_shunt))
        else:
            from adj2_train import side_feats as _sf
            data.depth = torch.from_numpy(_sf("reactive", variant, raw, low, high))
        batch = Batch.from_data_list([data]).to(dev)
        with torch.no_grad():
            h, g, e = enc(batch)
            o = reasoner(h, g, e, batch.batch, batch.batch[batch.edge_index[0]], batch.depth,
                         **({"n_graphs": 1} if variant in ("v6", "v6d", "v6e") else {}))
        ptr, dp = o[0], o[1]
        sh = o[2] if len(o) >= 3 else None
        gm = gen_node_mask(raw)
        if a.ptr_local_hops > 0:
            lm = local_gen_mask(raw, Path(l5_case).parent, 10 ** 6, low, high, a.ptr_local_hops)
            if (gm & lm).any():
                info["ptr_local_kept"] = int((gm & lm).sum()); info["ptr_local_all"] = int(gm.sum())
                gm = gm & lm
            else:
                info["ptr_local_kept"] = 0; info["ptr_local_all"] = int(gm.sum())
        prob = torch.sigmoid(ptr.float()).cpu().numpy()
        n_over = int((gm & (prob > 0.5)).sum())
        top_k = None
        if variant == "v6e":
            # v6e 首步 = 档位头(0=纯并补回退,1..12=(k,mag)):指针排序前 k 台 × 统一幅度,方向按越限侧
            rung = int(o[3].float().argmax(-1)[0])
            info["rung_head"] = rung
            if rung > 0:
                k, mag = C.RUNG_TABLE[rung - 1]
                is_low = (float(vmin0) < low - C.EPS) if vmin0 is not None else True
                sign = 1.0 if is_low else -1.0
                order = C.rank_by_prob(prob, gm)
                action = C.reactive_ladder_action(order, k, mag, sign)
                if vcap is not None:
                    action = C.clip_dvset_to_cap(action, raw, vcap)
                info.update({"n_sel": len(action["dvset"]), "n_over_thr": n_over, "n_shunt": 0,
                             "first_rung": (k, mag)})
                return action, info, {"prob": prob, "cand": gm}
        if variant in ("v6", "v6d", "v6e"):
            top_k = int(o[3].float().argmax(-1)[0])
            info["k_head"] = top_k
            if a.cap_k is not None:
                top_k = min(top_k, a.cap_k)
            if top_k == 0 and sh is None:
                top_k = 1
            if variant == "v6e":
                top_k = 1        # 档位 0(纯并补类)回退:1 台 + 并补,同 v6d 的 0 类处置
        elif a.cap_k is not None and n_over > a.cap_k:
            top_k = a.cap_k
        action = decode_action(gm, prob, dp.float().cpu().numpy(), unit_span_vset(raw),
                               shunt_cand_mask(raw) if sh is not None else None,
                               torch.sigmoid(sh.float()).cpu().numpy() if sh is not None else None,
                               dv_levels=(rcfg.get("dv_levels") or DV_LEVELS) if n_dv else None, top_k=top_k)
        if vcap is not None:
            action = C.clip_dvset_to_cap(action, raw, vcap)
        info.update({"n_sel": len(action["dvset"]), "n_over_thr": n_over, "n_shunt": len(action["shunt"]), "top_k": top_k})
        return action, info, {"prob": prob, "cand": gm}

    def apply_action(w: Path, act):
        if a.task == "active":
            l5 = Card.load(w / "LF.L5")
            apply_fix_by_bus(l5, act)
            l5.save(w / "LF.L5")
        else:
            apply_mixed(w, act)

    def run_ladder(w: Path, bak: Path, raw, aux, act_first, vmin0, vmax0, c0: int, sick_slack: dict):
        """首步失败后的模型引导阶梯。返回 (ladder 记录, 成功动作或 None, 成功时的裁判结果)。"""
        prob, cand = aux["prob"], aux["cand"].copy()
        steps = []
        if a.task == "active":
            up_room, dn_room, limited = C.gen_rooms_from_raw(raw)
            cand = cand & limited                       # 阶梯一律剔固定出力机组(物理口径)
            order = C.rank_by_prob(prob, cand)
            exc_up = max(0.0, (sick_slack.get("p_solved") or 0.0) - sick_slack.get("pmax", 0.0)) if sick_slack.get("limited") else 0.0
            exc_dn = max(0.0, sick_slack.get("pmin", 0.0) - (sick_slack.get("p_solved") or 0.0)) if sick_slack.get("limited") else 0.0
            depth = max(0.0, low - float(vmin0))
            zero_net = exc_dn > 1e-3
            skip = None
            meta = {"exc_up": round(exc_up, 3), "exc_dn": round(exc_dn, 3), "depth": round(depth, 4), "zero_net": zero_net,
                    "n_cand": int(cand.sum())}
        else:
            order = C.rank_by_prob(prob, cand)
            is_low = float(vmin0) < low - C.EPS
            sign = 1.0 if is_low else -1.0
            dvs = list((act_first or {}).get("dvset", {}).values())
            skip = None
            if dvs and len(set(round(abs(x), 4) for x in dvs)) == 1 and not (act_first or {}).get("shunt"):
                skip = (len(dvs), abs(dvs[0]))          # 首步是"k 台同幅度"时,阶梯不重复该档
            meta = {"sign": sign, "n_cand": int(cand.sum())}
        rungs = C.ladder_rungs(ladder_ks, ladder_mags, skip=skip, k_cap=len(order))
        meta["rungs"] = rungs
        fixed_act, fixed_rf, fixed_i = None, None, None
        for i, (k, mag) in enumerate(rungs):
            if solver.n_pf - c0 >= a.ladder_max_pf:
                meta["stopped"] = "max_pf"
                break
            restore_files(bak, w)
            if a.task == "active":
                sel = [int(r) for r in order if up_room[r] > 1e-3][:k]
                if exc_up > 1e-3:
                    amount = mag * exc_up
                else:
                    amount = depth * float(sum(aux["span"][r] for r in sel)) * 0.5 * mag
                act = C.active_ladder_action(order, k, amount, up_room, dn_room, zero_net=zero_net)
                if act and variant in ("v07", "v07d") and not a.no_proj:
                    up, dn = C.slack_headroom_from_raw(raw)
                    act, _sc = C.project_action_to_slack(act, up, dn)
                    act = {b: round(v, 4) for b, v in act.items()}
                n_app = len(act)
                sum_abs = round(sum(abs(v) for v in act.values()), 4)
            else:
                act = C.reactive_ladder_action(order, k, mag, sign)
                if vcap is not None:
                    act = C.clip_dvset_to_cap(act, raw, vcap)
                n_app = len(act["dvset"])
                sum_abs = round(sum(abs(v) for v in act["dvset"].values()), 4)
            if not n_app:
                steps.append({"rung": i, "k": k, "mag": mag, "skipped": "empty_action"})
                continue
            apply_action(w, act)
            rf = referee(w, solver)
            ok = judge_ok(rf)
            steps.append({"rung": i, "k": k, "mag": mag, "n_applied": n_app, "sum_abs": sum_abs, "converged": bool(rf.get("converged")),
                          "vmin": rf.get("vmin"), "vmax": rf.get("vmax"), "ok": ok, "pf_total": solver.n_pf - c0})
            if ok:
                fixed_act, fixed_rf, fixed_i = act, rf, i
                break
        rec = {"used": True, "steps": steps, "fixed_at_rung": fixed_i,
               "fixed_at_pf": (solver.n_pf - c0) if fixed_i is not None else None, **meta}
        return rec, fixed_act, fixed_rf

    fused = False
    try:
        for i, src in enumerate(cases):
            cid = src.name
            if cid in results and results[cid].get("complete"):
                continue
            w = out_dir / cid / "model"
            robust_copytree(src, w)
            bak = w / "_original_backup"
            bak.mkdir()
            copy_files(w, bak)
            c0 = solver.n_pf
            t0 = time.time()
            rf = referee(w, solver)
            conv = bool(rf.get("converged"))
            vmin0, vmax0 = rf.get("vmin"), rf.get("vmax")
            healthy = conv and vmin0 is not None and vmin0 >= low and (a.task == "active" or vmax0 <= high)
            rec = {"case_id": cid, "init_vmin": vmin0, "init_vmax": vmax0, "init_converged": conv}
            if not conv:                   # 2026-09-05：初判真不收敛（守卫已排除狗故障）记失败，不进阶梯
                rec["model"] = {"verdict": "init_diverged", "ok": False, "healthy_no_action": False, "psasp_calls": solver.n_pf - c0,
                                "final_vmin": None, "final_vmax": None, "action": None}
            elif healthy:
                rec["model"] = {"verdict": "healthy_no_action", "ok": True, "healthy_no_action": True, "psasp_calls": solver.n_pf - c0,
                                "final_vmin": vmin0, "final_vmax": vmax0, "action": None}
            else:
                sick_slack = C.slack_info(w) if a.task == "active" else {}
                pc = PsaspCase(bak)
                raw = build_sample(pc, Catalog(pc), sample_id=("model" if a.sample_id_mode == "dirname" else cid),
                                   family_id="deploy", schema="v0")
                t_enc = time.time()
                actions = []
                for r in range(max(1, a.repeat)):
                    act, info, aux = forward_action(raw, bak / "LF.L5", vmin0=vmin0)
                    actions.append((act, info, aux))
                t_enc = time.time() - t_enc
                act, info, aux = actions[0]
                if act is None:
                    rec["model"] = {"verdict": "no_adjustable", "ok": False, "psasp_calls": solver.n_pf - c0, "action": None, "info": info}
                else:
                    apply_action(w, act)
                    m_rf = referee(w, solver)
                    m_ok = judge_ok(m_rf)
                    same = [json.dumps(x[0], sort_keys=True) == json.dumps(act, sort_keys=True) for x in actions[1:]]
                    first = {"ok": m_ok, "final_converged": bool(m_rf.get("converged")), "final_vmin": m_rf.get("vmin"),
                             "final_vmax": m_rf.get("vmax"), "action": act, "psasp_calls": solver.n_pf - c0}
                    rec["model"] = {"verdict": "model_fixed" if m_ok else "model_failed", "ok": m_ok, "healthy_no_action": False,
                                    "psasp_calls": solver.n_pf - c0, "final_converged": bool(m_rf.get("converged")),
                                    "final_vmin": m_rf.get("vmin"), "final_vmax": m_rf.get("vmax"),
                                    "action": act, "info": info, "encode_secs": round(t_enc / max(1, a.repeat), 2),
                                    "repeat_same_action": (all(same) if same else None),
                                    "repeat_n_sel": [x[1].get("n_sel") for x in actions], "realized_state": "first_step"}
                    if a.ladder:
                        rec["model"]["first_step"] = first
                        if not m_ok:
                            lad, l_act, l_rf = run_ladder(w, bak, raw, aux, act, vmin0, vmax0, c0, sick_slack)
                            rec["model"]["ladder"] = lad
                            if l_act is not None:
                                rec["model"].update({"verdict": "ladder_fixed", "ok": True, "final_converged": True,
                                                     "final_vmin": l_rf.get("vmin"), "final_vmax": l_rf.get("vmax"), "action": l_act,
                                                     "realized_state": "ladder_rung_%d" % lad["fixed_at_rung"]})
                            else:
                                rec["model"]["verdict"] = "ladder_failed"
                                rec["model"]["realized_state"] = ("ladder_last_rung" if lad["steps"] else "first_step")
                            rec["model"]["psasp_calls"] = solver.n_pf - c0
                    try:
                        rec["realized_model"] = realized(w, a.task)
                        if a.task == "active":
                            # 固定出力机组(pmax<=pmin)上落地的 |ΔP| 占比:部署包 eff_bounds 把它当无限机组,B 段 3120 有功成功例 80~90% 的 ΔP 落在其上
                            _s0, g0 = read_gens_repo(Card.load(bak / "LF.L5"))
                            _s1, g1 = read_gens_repo(Card.load(w / "LF.L5"))
                            tot = onf = 0.0
                            nf = 0
                            for gi in g0:
                                if gi in g1:
                                    d_ = g1[gi]["p"] - g0[gi]["p"]
                                    if abs(d_) > 1e-6:
                                        tot += abs(d_)
                                        if g0[gi]["pmax"] <= g0[gi]["pmin"] + 1e-9:
                                            onf += abs(d_)
                                            nf += 1
                            rec["realized_model"]["sum_abs_dp_on_fixed"] = round(onf, 3)
                            rec["realized_model"]["frac_dp_on_fixed"] = (round(onf / tot, 3) if tot > 1e-9 else 0.0)
                            rec["realized_model"]["n_fixed_units_changed"] = nf
                    except Exception as e:  # noqa: BLE001
                        rec["realized_model"] = {"error": str(e)}
            rec["model"]["wall"] = round(time.time() - t0, 1)
            rs = rule_src.get(cid, {}).get("rule")
            if rs:
                rec["rule"] = dict(rs)
                rec["rule"]["source"] = "reused(B 段 %s_%s)" % (a.task, a.tier)
                m_, r_ = rec["model"], rec["rule"]
                rec["cascade"] = {"ok": m_["ok"] or r_["ok"],
                                  "verdict": "model_fixed" if m_["ok"] else ("rule_fixed" if r_["ok"] else "both_failed"),
                                  "psasp_calls": m_["psasp_calls"] + (r_["psasp_calls"] - 1 if not m_["ok"] else 0),
                                  "final_vmin": m_["final_vmin"] if m_["ok"] else r_["final_vmin"],
                                  "final_vmax": m_["final_vmax"] if m_["ok"] else r_["final_vmax"], "source": "derived(model→rule)"}
            if a.tier in BIG:
                prune_big(w)
            rec["complete"] = True
            results[cid] = rec
            res_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
            mi = rec["model"].get("info") or {}
            lad = rec["model"].get("ladder")
            print("[adj2-eval] %d/%d %s 模型=%s vmin %.4f→%s 选 %s 台%s%s (%d 次, %.0fs)" % (
                i + 1, len(cases), cid, rec["model"]["verdict"], vmin0 or 0,
                ("%.4f" % rec["model"]["final_vmin"]) if rec["model"].get("final_vmin") is not None else "—",
                mi.get("n_sel"), (" 投影×%.2f" % mi["proj_scale"]) if "proj_scale" in mi and mi["proj_scale"] < 1 else "",
                (" 阶梯%s/%d" % (("第%d阶" % (lad["fixed_at_rung"] + 1)) if lad.get("fixed_at_rung") is not None else "败", len(lad["steps"]))) if lad else "",
                rec["model"]["psasp_calls"], time.time() - t_all), flush=True)
    except LicenseError as e:
        fused = True
        print("[adj2-eval] 加密狗失效熔断: %s" % e, flush=True)
        res_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    recs = [results[c.name] for c in cases if c.name in results and results[c.name].get("complete")]
    n = len(recs)
    arms = {}

    def arm_stats(rr):
        ok = sum(1 for r in rr if r["ok"])
        calls = [r["psasp_calls"] for r in rr]
        vm = [r["final_vmin"] for r in rr if r["ok"] and r.get("final_vmin") is not None]
        vx = [r["final_vmax"] for r in rr if r["ok"] and r.get("final_vmax") is not None]
        return {"n": len(rr), "n_ok": ok, "rate": round(ok / len(rr), 3), "wilson95": wilson(ok, len(rr)),
                "psasp_calls_mean": round(sum(calls) / len(rr), 2), "psasp_calls_max": max(calls),
                "fixed_vmin_min_med_max": [round(min(vm), 4), round(sorted(vm)[len(vm) // 2], 4), round(max(vm), 4)] if vm else None,
                "fixed_vmax_max": round(max(vx), 4) if vx else None}
    for arm in ("model", "rule", "cascade"):
        rr = [r[arm] for r in recs if arm in r]
        if rr:
            arms[arm] = arm_stats(rr)
    ladder_sum = None
    if a.ladder and recs:
        # 首步臂 = 同一次跑里模型的第一个动作(健康例与 no_adjustable 例按模型臂原样计)
        rr = []
        ok_at = []
        for r in recs:
            m = r["model"]
            fs = m.get("first_step")
            if fs is not None:
                rr.append({"ok": fs["ok"], "psasp_calls": fs["psasp_calls"], "final_vmin": fs["final_vmin"], "final_vmax": fs["final_vmax"]})
            else:
                rr.append({"ok": m["ok"], "psasp_calls": m["psasp_calls"], "final_vmin": m.get("final_vmin"), "final_vmax": m.get("final_vmax")})
            if m.get("healthy_no_action"):
                ok_at.append(1)
            elif fs is not None and fs["ok"]:
                ok_at.append(fs["psasp_calls"])
            elif m.get("ladder") and m["ladder"].get("fixed_at_pf"):
                ok_at.append(m["ladder"]["fixed_at_pf"])
            elif fs is None and m["ok"]:
                ok_at.append(m["psasp_calls"])
            else:
                ok_at.append(None)
        arms["model_first"] = arm_stats(rr)
        lads = [r["model"]["ladder"] for r in recs if r["model"].get("ladder")]
        hist = {}
        for l_ in lads:
            if l_.get("fixed_at_rung") is not None:
                k_ = "%d×%g" % tuple(l_["rungs"][l_["fixed_at_rung"]])
                hist[k_] = hist.get(k_, 0) + 1
        max_pf = max([a.ladder_max_pf] + [x for x in ok_at if x])
        ladder_sum = {"ks": ladder_ks, "mags": ladder_mags, "max_pf": a.ladder_max_pf,
                      "n_first_fail": sum(1 for r in rr if not r["ok"]), "n_ladder_used": len(lads),
                      "n_ladder_fixed": sum(1 for l_ in lads if l_.get("fixed_at_rung") is not None),
                      "fixed_at_rung_hist": hist,
                      "ladder_steps_mean": (round(float(np.mean([len(l_["steps"]) for l_ in lads])), 2) if lads else None),
                      "ok_by_pf_curve": {str(p): sum(1 for x in ok_at if x is not None and x <= p) for p in range(1, max_pf + 1)},
                      "n_stopped_by_max_pf": sum(1 for l_ in lads if l_.get("stopped") == "max_pf")}
    # 实际干预 + 合理性(成功例中位)
    key = "sum_abs_dp" if a.task == "active" else "sum_abs_dv"
    okv = [r["realized_model"] for r in recs if r.get("realized_model") and r["model"]["ok"] and "error" not in r["realized_model"]]
    allv = [r["realized_model"] for r in recs if r.get("realized_model") and "error" not in r["realized_model"]]

    def med(lst, k):
        xs = [x[k] for x in lst if x.get(k) is not None]
        return (round(float(np.median(xs)), 3), round(min(xs), 3), round(max(xs), 3)) if xs else None
    inter = {"n_ok": len(okv), "n_all": len(allv),
             "ok_n_changed_med_min_max": med(okv, "n_changed"), "ok_sum_abs_med_min_max": med(okv, key),
             "all_n_changed_med_min_max": med(allv, "n_changed"), "all_sum_abs_med_min_max": med(allv, key)}
    if a.ladder:
        fv = [r["realized_model"] for r in recs if r.get("realized_model") and "error" not in r["realized_model"]
              and (r["model"].get("first_step") or {}).get("ok")]
        lv = [r["realized_model"] for r in recs if r.get("realized_model") and "error" not in r["realized_model"]
              and r["model"].get("verdict") == "ladder_fixed"]
        inter.update({"first_ok_n_changed_med_min_max": med(fv, "n_changed"), "first_ok_sum_abs_med_min_max": med(fv, key),
                      "ladder_ok_n_changed_med_min_max": med(lv, "n_changed"), "ladder_ok_sum_abs_med_min_max": med(lv, key)})
    if a.task == "active":
        inter.update({"ok_n_plausible": sum(1 for x in okv if x.get("plausible")),
                      "ok_frac_dp_on_fixed_med": med(okv, "frac_dp_on_fixed"),
                      "ok_n_with_fixed_units": sum(1 for x in okv if (x.get("n_fixed_units_changed") or 0) > 0),
                      "ok_n_pushed_to_pmax_med": med(okv, "n_pushed_to_pmax"), "ok_net_dp_med": med(okv, "net_dp"),
                      "ok_slack_dist_before_med": med(okv, "slack_dist_before"), "ok_slack_dist_after_med": med(okv, "slack_dist_after"),
                      "all_n_plausible": sum(1 for x in allv if x.get("plausible"))})
    else:
        inter.update({"ok_n_shunt_med": med(okv, "n_shunt_switched"), "ok_max_abs_dv_med": med(okv, "max_abs_dv")})
    rep_same = [r["model"].get("repeat_same_action") for r in recs if r["model"].get("repeat_same_action") is not None]
    summary = {"task": a.task, "tier": a.tier, "lib": lib.name, "vio_band": [low, high], "n_cases": n, "ckpt": a.ckpt,
               "ckpt_path": ckpt_path, "variant": variant, "device": str(dev), "cap_k": a.cap_k, "proj": (variant in ("v07", "v07d") and not a.no_proj),
               "dyn_fill": dyn_mode, "ladder": ladder_sum,
               "arms": arms, "intervention_model": inter,
               "model_only_ok_rule_fail": sum(1 for r in recs if "rule" in r and r["model"]["ok"] and not r["rule"]["ok"]),
               "rule_only_ok_model_fail": sum(1 for r in recs if "rule" in r and r["rule"]["ok"] and not r["model"]["ok"]),
               "both_fail": sum(1 for r in recs if "rule" in r and not r["model"]["ok"] and not r["rule"]["ok"]),
               "healthy_no_action": sum(1 for r in recs if r["model"].get("healthy_no_action")),
               "repeat": a.repeat, "repeat_checked": len(rep_same), "repeat_same_action": sum(1 for x in rep_same if x),
               "sample_id_mode": a.sample_id_mode, "rf_seed": a.rf_seed,
               "encode_secs_mean": round(float(np.mean([r["model"]["encode_secs"] for r in recs if r["model"].get("encode_secs") is not None])), 2)
               if any(r["model"].get("encode_secs") is not None for r in recs) else None,
               "wall_secs": round(time.time() - t_all, 1), "license_fused": fused,
               "note": "规则臂沿用 B 段同病例结果;级联推导;模型臂每例 2 次 PSASP(初判+复核)" + (";阶梯每阶 +1 次" if a.ladder else "") + ";分档禁并池"}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[adj2-eval] 汇总 %s/%s %s n=%d 模型 %s%s 规则 %s 级联 %s | 干预 %s | 重复一致 %d/%d%s" % (
        a.task, a.tier, variant, n, arms.get("model", {}).get("rate"),
        (" (首步 %s)" % arms["model_first"]["rate"]) if "model_first" in arms else "",
        arms.get("rule", {}).get("rate"), arms.get("cascade", {}).get("rate"), inter["ok_n_changed_med_min_max"],
        summary["repeat_same_action"], summary["repeat_checked"],
        (" | 阶梯 %s" % json.dumps(ladder_sum["ok_by_pf_curve"])) if ladder_sum else ""), flush=True)
    return 9 if fused else 0      # 9 = 加密狗熔断(与脚本崩溃的 1 区分)


if __name__ == "__main__":
    raise SystemExit(main())
