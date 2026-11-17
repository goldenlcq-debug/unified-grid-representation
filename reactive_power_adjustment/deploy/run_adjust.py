#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""电网大模型 · 调整器统一入口(Linux 内网部署版;2026-08-23 版,默认 CPU)。

对一个 PSASP 51 格式数据目录执行有功或无功调整(三段级联):
  初判(PSASP) → 健康即止 → ① 模型首步(V1 编码器 + 调整推理器,1 次 PSASP 复核)
  → 失败回退病态起点 → ② 模型引导的阶梯(按指针概率排序扩台数,每阶 1 次 PSASP,每例总次数上限 --ladder-max-pf)
  → 仍失败回退病态起点 → ③ 规则老师兜底 → 原地写回卡片 + 出报告。

2026-08-23 相对 2026-08-21 版的改动(评审单 `部署包修正评审单-20260823.md` §10 顺序):
  ① 有功解码/施加/规则臂一律剔除固定出力机组(MATPOWER 口径 PMAX==PMIN;CEPRI36 类全库 pmax=pmin=0 保留"未设限"约定;--fixed-units);
  ② 模型臂、阶梯、规则臂统一裁判 vmin ≥ low−eps(无功再 vmax ≤ high+eps),eps 缺省 5e-4(与规则臂 _is_healthy_2s、病态判定同一把尺子);
  ③ 档案 sample_id = 输入卡片内容哈希(--sample-id content;legacy = 目录名,对拍用),随机特征列不再随目录名变;--rf-seed 可重抽取区间;
  ④ 规则臂"就近机组"邻接含变压器;
  ⑤ 有功默认检查点 v0.7(侧通道 5 维 + 平衡机裕度投影),无功默认 v6d(v5 三头 + 台数头 + 越限邻近度);--legacy-ckpt 回到 v0.6 / v5;
  ⑥ 报告新增实际落地干预(改机数 / Σ|Δ| / 推满台数 / 投切数 / 平衡机修前后距限 / 落在固定机组上的 |ΔP|)、首步与阶梯逐阶记录。

目录约定不变:不在目标目录外创建任何文件夹;原始文件备份到 `_original_backup/`(已存在则据其恢复,重复运行幂等);
报告写在 `_original_backup/adjust_report.json`。

用法:
  python3 run_adjust.py --task reactive --case /data/case001 --psasp-bin /opt/psasp
  python3 run_adjust.py --task active   --case /data/case001 --psasp-bin /opt/psasp --card-encoding utf-8 --psasp-exe WMLFRTMsg
  python3 run_adjust.py --task reactive --case /data/case001 --psasp-bin /opt/psasp --legacy-ckpt --no-ladder --eps 0   # = 2026-08-21 版行为(sample_id 除外)
批量:scripts/run_batch.py(同一套 adjust() 代码路径,模型只载一次)。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG
DEFAULT_CKPT = {"active": "runs/adjust_reasoner_v07_s0", "reactive": "runs/reactive_reasoner_v6d_s0"}
LEGACY_CKPT = {"active": "runs/adjust_reasoner_v06_s0", "reactive": "runs/reactive_reasoner_v5_s0"}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="有功/无功调整推理器(V1 编码器 + 调整头 + 模型阶梯 + 规则兜底)")
    ap.add_argument("--task", required=True, choices=["active", "reactive"],
                    help="active=有功调整(修低电压) | reactive=无功调整(修电压越限)")
    ap.add_argument("--case", required=True, help="51 格式数据目录(原地调整;原始文件备份到其子目录 _original_backup)")
    ap.add_argument("--psasp-bin", required=True, help="PSASP 计算程序所在目录")
    ap.add_argument("--psasp-exe", default=None, help="潮流可执行文件名(默认 Linux=WMLFRTMsg / Windows=WMLFRTMsg.exe)")
    ap.add_argument("--card-encoding", default="gb18030", choices=["gb18030", "utf-8", "auto"],
                    help="51 数据卡片的文本编码(默认 gb18030;代码与报告文件一律 UTF-8)")
    ap.add_argument("--vio-low", type=float, default=0.93, help="电压下限(默认 0.93)")
    ap.add_argument("--vio-high", type=float, default=1.08, help="电压上限(仅 reactive 使用;默认 1.08)")
    ap.add_argument("--eps", type=float, default=5e-4,
                    help="裁判容差:修好 = vmin ≥ low−eps(无功再 vmax ≤ high+eps);模型臂/阶梯/规则臂同一把尺子(默认 5e-4;0 = 2026-08-21 版模型臂口径)")
    ap.add_argument("--no-fallback", action="store_true", help="不落规则兜底")
    ap.add_argument("--rule-only", action="store_true", help="只跑规则老师(不载模型;验收对照用)")
    ap.add_argument("--no-ladder", action="store_true", help="不做模型引导的阶梯(首步失败直接交规则)")
    ap.add_argument("--ladder-ks", default=None, help="阶梯台数(逗号分隔;缺省 无功 1,2,4,8 / 有功 2,4,8,16,32)")
    ap.add_argument("--ladder-mags", default=None, help="阶梯量级(逗号分隔;缺省 无功 Δv_set 0.02,0.05 / 有功 平衡机超额倍数 1.0)")
    ap.add_argument("--ladder-max-pf", type=int, default=8, help="每例 PSASP 总次数上限(含初判 + 首步 + 阶梯;缺省 8)")
    ap.add_argument("--fixed-units", default="auto", choices=["auto", "fixed", "unlimited"],
                    help="固定出力机组(pmax<=pmin)口径:auto=算例真填了限值则不可调、全库 0 限值走遗留'未设限'(缺省);fixed=一律不可调;unlimited=2026-08-21 版行为")
    ap.add_argument("--sample-id", default="content", choices=["content", "legacy"],
                    help="档案 sample_id(决定随机特征列):content=输入卡片内容哈希(缺省);legacy=目录名(2026-08-21 版行为,对拍用)")
    ap.add_argument("--rf-seed", type=int, default=0, help=">0 时随机特征列按 (sample_id, rf_seed) 重抽(多次运行取区间用)")
    ap.add_argument("--dyn-fill", default="auto", choices=["auto", "zero", "mean", "mean_all"],
                    help="动态列填法:auto=按检查点训练填法(缺省,现有检查点均为 zero)")
    ap.add_argument("--no-proj", action="store_true", help="有功 v0.7:关闭平衡机裕度投影")
    ap.add_argument("--vset-cap", type=float, default=None,
                    help="无功:模型首步/阶梯写入的 v_set 上限守门(缺省 = --vio-high;PV 母线电压 = v_set,当前 PSASP 不执行机组无功限制,"
                         "v_set 超过电压上限必判 vmax 越限;传 1.10 = 2026-08-21 版行为,规则臂不受此参数影响)")
    ap.add_argument("--ptr-local-hops", type=int, default=0,
                    help="局部性预筛(2026-08-27):把指针候选限制在距越限母线 ≤H 跳内再由模型排序。"
                         "0=关(缺省,行为与既往逐位相同)。**开不开、开多大由诊断定,不要拍脑袋**:"
                         "先算 Δ=模型指针首选台跳数−就近选台跳数,Δ≥8 才开;半径取'0台回退≤5% 且候选中位≥8'的最小 H。"
                         "详见 部署说明.md §六 红线 1。")
    ap.add_argument("--device", default="cpu", help="cpu(默认)或 cuda")
    ap.add_argument("--v1-dir", default=str(REPO / "runs" / "encoder_v1"))
    ap.add_argument("--ckpt", default=None, help="推理器检查点目录(默认按 --task 取包内 v0.7 / v6d)")
    ap.add_argument("--legacy-ckpt", action="store_true", help="用 2026-08-21 版检查点(有功 v0.6 / 无功 v5)")
    return ap


def setup_env(card_encoding: str, psasp_exe: str = None) -> str:
    """环境变量必须在导入 gridfm 之前设置(卡片编码/可执行名是模块级常量)。返回可执行名。"""
    os.environ["GRIDFM_CARD_ENCODING"] = card_encoding
    default_exe = "WMLFRTMsg.exe" if sys.platform == "win32" else "WMLFRTMsg"
    exe_name = psasp_exe or default_exe
    os.environ["GRIDFM_PSASP_LF_NAME"] = exe_name
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    return exe_name


def copy_files(src: Path, dst: Path) -> None:
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)


def restore_files(bak: Path, root: Path) -> None:
    for p in root.iterdir():
        if p.is_file():
            p.unlink()
    copy_files(bak, root)


def _parse_list(s, typ):
    return [typ(x) for x in str(s).split(",") if x.strip()]


def local_gen_mask(raw, case_dir: Path, low: float, high: float, hops_max: int):
    """局部性预筛(2026-08-27):返回"距越限母线 ≤hops_max 跳"的机组布尔掩码(1 基母线号 → 节点行)。

    越限母线取自该病例**已解出**的 LF.LP1(低压或高压皆算病灶);找不到病灶 → 全 True(不限制)。
    与本机跑分器 `adj2_eval.local_gen_mask` 同口径(交流线+变压器、只算 mark==1)。
    BFS 邻居按母线号排序 = 确定序,防哈希序飘(见 部署说明 §六 红线 1)。
    """
    import numpy as np
    from collections import deque
    from gridfm.psasp_io import cards
    from gridfm.psasp_io.case import PsaspCase
    from gridfm.psasp_io.lowlevel import Card
    from gridfm.data.repair_perturb import _adjacency

    n = raw.n_nodes
    lp1 = Path(case_dir) / "LF.LP1"
    if not lp1.is_file():
        return np.ones(n, dtype=bool)
    v = np.zeros(n + 1)                                     # 1 基母线号
    for r in cards.parse_lp1(Card.load(lp1)):
        if 1 <= r.bus_no <= n and np.isfinite(r.v_pu) and r.v_pu < 100:
            v[r.bus_no] = r.v_pu
    bad = [b for b in range(1, n + 1)
           if v[b] > 0.01 and (v[b] < low - 1e-4 or v[b] > high + 1e-4)]
    if not bad:
        return np.ones(n, dtype=bool)
    graph = _adjacency(PsaspCase(Path(case_dir)))
    dist = {b: 0 for b in bad}
    q = deque(bad)
    while q:
        u = q.popleft()
        if dist[u] >= hops_max:
            continue
        for w in sorted(graph.get(u, ())):
            if w not in dist:
                dist[w] = dist[u] + 1
                q.append(w)
    m = np.zeros(n, dtype=bool)
    for b in dist:
        if 1 <= b <= n:
            m[b - 1] = True
    return m


def load_models(a, dev=None) -> dict:
    """载 V1 编码器 + 归一化器 + 推理器(按检查点 config 建;变体见 adjust_ext.build_reasoner)。"""
    import torch
    from dataclasses import fields as dc_fields
    from gridfm.data.normalize import Normalizer
    from gridfm.train.pretrain import TrainConfig, build_models
    from adjust_ext import build_reasoner

    v1 = Path(a.v1_dir)
    ck = torch.load(v1 / "v1.pt", map_location="cpu", weights_only=False)
    allowed = {f.name for f in dc_fields(TrainConfig)}
    cfg = TrainConfig(**{k: v for k, v in ck["config"].items() if k in allowed})
    if dev is None:
        dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    enc, _ = build_models(cfg)
    enc.load_state_dict(ck["model_state"])
    enc.to(dev).eval()
    norm = Normalizer.load(v1 / "normalizer.npz")
    ckpt_dir = Path(a.ckpt) if a.ckpt else REPO / (LEGACY_CKPT if a.legacy_ckpt else DEFAULT_CKPT)[a.task]
    bck = torch.load(ckpt_dir / "ckpt" / "best.pt", map_location=dev, weights_only=False)
    rcfg = bck["config"]
    reasoner, variant = build_reasoner(a.task, rcfg)
    reasoner.load_state_dict(bck["reasoner_state"])
    reasoner.to(dev).eval()
    dyn_fill = (rcfg.get("dyn_fill") or "zero") if a.dyn_fill == "auto" else a.dyn_fill
    return {"enc": enc, "norm": norm, "cfg": cfg, "dev": dev, "reasoner": reasoner, "variant": variant, "rcfg": rcfg,
            "ckpt_dir": str(ckpt_dir), "dyn_fill": dyn_fill}


def adjust(case: Path, a, models: dict = None, solver=None) -> dict:
    """对一个算例目录做三段级联调整;返回报告(同时写 _original_backup/adjust_report.json)。PSASP 调用全部经 solver 计数。"""
    import copy as _copy
    import numpy as np
    import torch
    from torch_geometric.data import Batch

    from gridfm.data.graph import sample_to_data
    from gridfm.data.synth import psasp_referee
    from gridfm.psasp_io.case import PsaspCase
    from gridfm.psasp_io.catalog import Catalog
    from gridfm.psasp_io.lowlevel import Card
    from gridfm.reason.ts_task import gen_node_mask
    from gridfm.schema.assemble import build_sample
    from gridfm.schema.sample import stable_seed
    from gridfm.solver.psasp import PsaspSolver
    from adjust_apply_eval import CountingSolver
    import adjust_ext as X

    case = Path(case)
    if not (case / "LF.L0").is_file():
        raise FileNotFoundError("%s 不是 PSASP 数据目录(缺 LF.L0)" % case)
    bak = case / "_original_backup"
    if bak.exists():
        restore_files(bak, case)
    else:
        bak.mkdir()
        copy_files(case, bak)
    if solver is None:
        solver = CountingSolver(PsaspSolver(psasp_bin=str(a.psasp_bin)))
    c0 = solver.n_pf
    t0 = time.time()
    low, high, eps = a.vio_low, a.vio_high, a.eps
    task = a.task
    ladder_on = not a.no_ladder
    ks_d, mags_d = X.LADDER_DEFAULT[task]
    ladder_ks = _parse_list(a.ladder_ks, int) if a.ladder_ks else list(ks_d)
    ladder_mags = _parse_list(a.ladder_mags, float) if a.ladder_mags else list(mags_d)

    def judge(rf):
        return X.judge_ok(rf, task, low, high, eps)

    def write_report(rep):
        rep.update({"vio_low": low, "vio_high": high, "eps": eps, "fixed_units_mode": a.fixed_units,
                    "sample_id_mode": a.sample_id, "rf_seed": a.rf_seed, "ladder_enabled": ladder_on,
                    "ladder_ks": ladder_ks, "ladder_mags": ladder_mags, "ladder_max_pf": a.ladder_max_pf,
                    "card_encoding": a.card_encoding, "total_psasp_calls": solver.n_pf - c0,
                    "secs": round(time.time() - t0, 1), "adjusted_dir": str(case), "original_backup": str(bak),
                    "version": "run_adjust 2026-08-23"})
        (bak / "adjust_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        return rep

    # ---------- 0. 初判 ----------
    rf0 = psasp_referee(case, solver)
    conv = bool(rf0.get("converged"))
    vmin, vmax = rf0.get("vmin"), rf0.get("vmax")
    if judge(rf0):
        return write_report({"task": task, "verdict": "healthy_no_action", "init_vmin": vmin, "init_vmax": vmax,
                             "final_vmin": vmin, "final_vmax": vmax, "action": None})
    if not conv:
        print("[warn] 初判不收敛——本工具面向电压越限修复,不收敛病例应走调收敛流程", flush=True)
    slack_before = X.slack_info(case) if task == "active" else None

    if getattr(a, "rule_only", False):
        # ---------- 规则单干(验收对照) ----------
        c1 = solver.n_pf
        if task == "active":
            from adjust_apply_eval import rule_arm
            rule_action, r_rf = rule_arm(case, solver, low, fixed_mode=a.fixed_units, eps=eps, rf0=rf0, return_rf=True,
                                         matpower=X.is_matpower_case(bak))
        else:
            from reactive_apply_eval import reactive_rule_arm
            rule_action, r_rf = reactive_rule_arm(case, solver, low, high, use_shunt=True, rf0=rf0, eps=eps, return_rf=True)
        r_ok = rule_action is not None
        rep = {"task": task, "verdict": "rule_fixed" if r_ok else "both_failed", "action": rule_action if r_ok else None,
               "model_ok": None, "model_psasp_calls": 0, "first_step": None, "ladder": None, "rule_ok": r_ok,
               "rule_psasp_calls": solver.n_pf - c1, "rule_action": rule_action, "init_vmin": vmin, "init_vmax": vmax,
               "final_vmin": (r_rf or {}).get("vmin"), "final_vmax": (r_rf or {}).get("vmax"),
               "final_converged": bool((r_rf or {}).get("converged")),
               "realized": X.realized(case, bak, task, slack_before, a.fixed_units), "info": {"rule_only": True},
               "device": None, "ckpt": None, "variant": "rule_only"}
        return write_report(rep)

    # ---------- 1. 模型 ----------
    if models is None:
        models = load_models(a)
    enc, norm, cfg, dev = models["enc"], models["norm"], models["cfg"], models["dev"]
    reasoner, variant, rcfg = models["reasoner"], models["variant"], models["rcfg"]

    # ---------- 2. 档案化 + 前向 ----------
    pc = PsaspCase(bak)
    sample_id = X.sha256_cards(bak) if a.sample_id == "content" else case.name
    raw = build_sample(pc, Catalog(pc), sample_id=sample_id, family_id="deploy", schema="v0")
    s = _copy.deepcopy(raw)
    norm.transform(s)
    n_filled = X.apply_dyn_fill(s, raw, models["dyn_fill"])
    rfeat = brfeat = None
    if a.rf_seed > 0:
        rng_rf = np.random.default_rng(stable_seed(raw.family_id, raw.sample_id, "rf%d" % a.rf_seed))
        rfeat = rng_rf.standard_normal((raw.n_nodes, cfg.random_dim)).astype(np.float32)
        brfeat = rng_rf.standard_normal((raw.n_branches, cfg.branch_random_dim)).astype(np.float32)
    data = sample_to_data(s, cfg.lape_k, cfg.rwse_steps, random_dim=cfg.random_dim, random_feat=rfeat,
                          branch_random_dim=cfg.branch_random_dim, branch_random_feat=brfeat)
    with_shunt = bool(rcfg.get("with_shunt", False))
    data.depth = torch.from_numpy(X.side_feats(task, variant, raw, low, high, with_shunt=with_shunt))
    batch = Batch.from_data_list([data]).to(dev)
    with torch.no_grad():
        h, g, e = enc(batch)
        kw = {"n_graphs": 1} if variant in ("v6", "v6d") else {}
        o = reasoner(h, g, e, batch.batch, batch.batch[batch.edge_index[0]], batch.depth, **kw)
    info = {"variant": variant, "ckpt": models["ckpt_dir"], "sample_id_used": sample_id, "dyn_fill": models["dyn_fill"],
            "dyn_filled_nodes": n_filled}

    if task == "active":
        from gridfm.reason.adjust_task import unit_span
        from adjust_labels import read_gens
        from adjust_materialize import apply_fix_by_bus
        ptr, dp = o[0], o[1]
        l5 = Card.load(case / "LF.L5")
        _slack, gens = read_gens(l5)
        mp_case = X.is_matpower_case(bak)
        fixed_gis = X.fixed_unit_gis(gens, a.fixed_units, matpower=mp_case)
        adjustable = {int(x["bus"]) for gi, x in gens.items() if gi not in fixed_gis}
        gm = gen_node_mask(raw).copy()
        for i in range(len(gm)):
            if gm[i] and (i + 1) not in adjustable:
                gm[i] = False
        info["n_fixed_units"] = len(fixed_gis)
        info["matpower_case"] = mp_case
        if not gm.any():
            rep = {"task": task, "verdict": "no_adjustable", "init_vmin": vmin, "init_vmax": vmax, "final_vmin": vmin,
                   "final_vmax": vmax, "action": None, "info": info}
            print("[error] 本算例无可调机组(全为平衡机/停运/固定出力)", flush=True)
            return write_report(rep)
        prob = torch.sigmoid(ptr.float()).cpu().numpy()
        span = unit_span(raw)
        sel = [int(i) for i in np.where(gm & (prob > 0.5))[0]]
        if not sel:
            sel = [int(np.where(gm)[0][int(np.argmax(prob[gm]))])]
        dpv = dp.float().cpu().numpy()
        action = {int(i) + 1: round(float(dpv[i]) * float(span[i]), 4) for i in sel}
        info.update({"n_sel": len(sel), "n_over_thr": int((gm & (prob > 0.5)).sum()), "net_raw": round(sum(action.values()), 4)})
        if variant in ("v07", "v07d") and not a.no_proj:
            up, dn = X.slack_headroom_from_raw(raw)
            action, sc = X.project_action_to_slack(action, up, dn)
            action = {b: round(v, 4) for b, v in action.items()}
            info.update({"proj_scale": round(sc, 4), "slack_up": round(up, 3), "slack_dn": round(dn, 3)})
        aux = {"prob": prob, "cand": gm, "span": span}

        def apply_action(act):
            l5_ = Card.load(case / "LF.L5")
            got = apply_fix_by_bus(l5_, act, fixed_mode=a.fixed_units, matpower=mp_case)
            l5_.save(case / "LF.L5")
            return got
    else:
        from gridfm.reason.reactive_task import DV_LEVELS, decode_action, shunt_cand_mask, unit_span_vset
        from reactive_apply_eval import apply_mixed
        ptr, dp = o[0], o[1]
        sh = o[2] if len(o) >= 3 else None
        gm = gen_node_mask(raw)
        if getattr(a, "ptr_local_hops", 0) > 0:      # 缺省 0 → 整段不执行,行为逐位不变
            lm = local_gen_mask(raw, case, low, high, a.ptr_local_hops)
            if (gm & lm).any():                      # 空集不硬套(那一例回退成不限制),并记账
                info["ptr_local_all"] = int(gm.sum())
                gm = gm & lm
                info["ptr_local_kept"] = int(gm.sum())
            else:
                info["ptr_local_all"] = int(gm.sum())
                info["ptr_local_kept"] = 0
            info["ptr_local_hops"] = int(a.ptr_local_hops)
        prob = torch.sigmoid(ptr.float()).cpu().numpy()
        n_dv = int(rcfg.get("n_dv_levels", 0))
        top_k = None
        if variant in ("v6", "v6d"):
            top_k = int(o[3].float().argmax(-1)[0])
            info["k_head"] = top_k
            if top_k == 0 and sh is None:
                top_k = 1
        action = decode_action(gm, prob, dp.float().cpu().numpy(), unit_span_vset(raw),
                               shunt_cand_mask(raw) if sh is not None else None,
                               torch.sigmoid(sh.float()).cpu().numpy() if sh is not None else None,
                               dv_levels=(rcfg.get("dv_levels") or DV_LEVELS) if n_dv else None, top_k=top_k)
        info.update({"n_sel": len(action["dvset"]), "n_over_thr": int((gm & (prob > 0.5)).sum()), "n_shunt": len(action["shunt"])})
        aux = {"prob": prob, "cand": gm}

        vset_cap = a.vset_cap if a.vset_cap is not None else high
        info["vset_cap"] = vset_cap

        def apply_action(act):
            return apply_mixed(case, act, vset_hi=vset_cap)

    # ---------- 3. 模型首步(1 次 PSASP) ----------
    c_first = solver.n_pf
    applied_first = apply_action(action)
    m_rf = psasp_referee(case, solver)
    m_ok = judge(m_rf)
    first = {"ok": m_ok, "action": action, "applied": applied_first, "final_vmin": m_rf.get("vmin"), "final_vmax": m_rf.get("vmax"),
             "converged": bool(m_rf.get("converged")), "psasp_calls": solver.n_pf - c_first}
    verdict = "model_fixed" if m_ok else "model_failed"
    final_rf, final_action = m_rf, action

    # ---------- 4. 模型引导的阶梯 ----------
    ladder = None
    if not m_ok and ladder_on:
        steps = []
        c_lad = solver.n_pf
        if task == "active":
            up_room, dn_room, adj_mask = X.gen_rooms_from_raw(raw, a.fixed_units, matpower=X.is_matpower_case(bak))
            cand = aux["cand"] & adj_mask
            order = X.rank_by_prob(aux["prob"], cand)
            sb = slack_before or {}
            exc_up = max(0.0, (sb.get("p_solved") or 0.0) - sb.get("pmax", 0.0)) if sb.get("limited") else 0.0
            exc_dn = max(0.0, sb.get("pmin", 0.0) - (sb.get("p_solved") or 0.0)) if sb.get("limited") else 0.0
            depth = max(0.0, low - float(vmin if vmin is not None else low))
            zero_net = exc_dn > 1e-3
            rungs = X.ladder_rungs(ladder_ks, ladder_mags, first=None, k_cap=len(order))
            meta = {"exc_up": round(exc_up, 3), "exc_dn": round(exc_dn, 3), "depth": round(depth, 4), "zero_net": zero_net,
                    "n_cand": int(cand.sum())}
        else:
            order = X.rank_by_prob(aux["prob"], aux["cand"])
            is_low = vmin is not None and float(vmin) < low - eps
            sign = 1.0 if is_low else -1.0
            dvs = list(action.get("dvset", {}).values())
            first_rung = (len(dvs), abs(dvs[0])) if (dvs and len(set(round(abs(x), 4) for x in dvs)) == 1 and not action.get("shunt")) else None
            rungs = X.ladder_rungs(ladder_ks, ladder_mags, first=first_rung, k_cap=len(order))
            meta = {"sign": sign, "n_cand": int(aux["cand"].sum()), "first_rung": first_rung}
        fixed_i = None
        for i, (k, mag) in enumerate(rungs):
            if solver.n_pf - c0 >= a.ladder_max_pf:
                meta["stopped"] = "max_pf"
                break
            restore_files(bak, case)
            if task == "active":
                sel = [int(r) for r in order if up_room[r] > 1e-3][:k]
                amount = mag * exc_up if exc_up > 1e-3 else depth * float(sum(aux["span"][r] for r in sel)) * 0.5 * mag
                act = X.active_ladder_action(order, k, amount, up_room, dn_room, zero_net=zero_net)
                if act and variant in ("v07", "v07d") and not a.no_proj:
                    up, dn = X.slack_headroom_from_raw(raw)
                    act, _sc = X.project_action_to_slack(act, up, dn)
                    act = {b: round(v, 4) for b, v in act.items()}
                n_app = len(act)
                sum_abs = round(sum(abs(v) for v in act.values()), 4)
            else:
                act = X.reactive_ladder_action(order, k, mag, sign)
                n_app = len(act["dvset"])
                sum_abs = round(sum(abs(v) for v in act["dvset"].values()), 4)
            if not n_app:
                steps.append({"rung": i, "k": k, "mag": mag, "skipped": "empty_action"})
                continue
            apply_action(act)
            rf = psasp_referee(case, solver)
            ok = judge(rf)
            steps.append({"rung": i, "k": k, "mag": mag, "n_applied": n_app, "sum_abs": sum_abs, "converged": bool(rf.get("converged")),
                          "vmin": rf.get("vmin"), "vmax": rf.get("vmax"), "ok": ok, "pf_total": solver.n_pf - c0})
            if ok:
                fixed_i = i
                final_rf, final_action = rf, act
                break
        ladder = {"used": True, "rungs": rungs, "steps": steps, "fixed_at_rung": fixed_i,
                  "fixed_at_pf": (solver.n_pf - c0) if fixed_i is not None else None, "psasp_calls": solver.n_pf - c_lad, **meta}
        if fixed_i is not None:
            m_ok, verdict = True, "ladder_fixed"
        else:
            verdict = "ladder_failed"

    # ---------- 5. 规则兜底 ----------
    r_ok, r_calls, rule_action = None, 0, None
    if not m_ok and not a.no_fallback:
        restore_files(bak, case)
        c1 = solver.n_pf
        if task == "active":
            from adjust_apply_eval import rule_arm
            rule_action, r_rf = rule_arm(case, solver, low, fixed_mode=a.fixed_units, eps=eps, rf0=rf0, return_rf=True,
                                         matpower=X.is_matpower_case(bak))
        else:
            from reactive_apply_eval import reactive_rule_arm
            rule_action, r_rf = reactive_rule_arm(case, solver, low, high, use_shunt=True, rf0=rf0, eps=eps, return_rf=True)
        r_ok = rule_action is not None
        r_calls = solver.n_pf - c1
        verdict = "rule_fixed" if r_ok else "both_failed"
        if r_rf is not None:
            final_rf = r_rf
        if r_ok:
            final_action = rule_action
    realized = X.realized(case, bak, task, slack_before, a.fixed_units) if verdict != "both_failed" or True else None
    rep = {"task": task, "verdict": verdict, "action": final_action,
           "model_ok": bool(first["ok"]), "model_psasp_calls": first["psasp_calls"], "first_step": first,
           "ladder": ladder, "rule_ok": r_ok, "rule_psasp_calls": r_calls, "rule_action": rule_action,
           "init_vmin": vmin, "init_vmax": vmax, "final_vmin": final_rf.get("vmin"), "final_vmax": final_rf.get("vmax"),
           "final_converged": bool(final_rf.get("converged")), "realized": realized, "info": info,
           "device": str(dev), "ckpt": models["ckpt_dir"], "variant": variant}
    return write_report(rep)


def main() -> int:
    a = build_argparser().parse_args()
    exe_name = setup_env(a.card_encoding, a.psasp_exe)
    psasp_bin = Path(a.psasp_bin)
    if not (psasp_bin / exe_name).is_file():
        print(f"[error] 未找到潮流可执行文件: {psasp_bin / exe_name}\n        请检查 --psasp-bin 与 --psasp-exe", flush=True)
        return 3
    missing = [d for d in ("gridfm", "scripts", "runs") if not (REPO / d).is_dir()]
    if missing:
        print(f"[error] 部署包结构不完整,缺少目录: {missing}(gridfm/ scripts/ runs/ 必须与 run_adjust.py 同级)", flush=True)
        return 3
    case = Path(a.case)
    if not (case / "LF.L0").is_file():
        print(f"[error] {case} 不是 PSASP 数据目录(缺 LF.L0)", flush=True)
        return 3
    rep = adjust(case, a)
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    if rep["verdict"] == "no_adjustable":
        return 4
    return 0 if rep["verdict"] in ("healthy_no_action", "model_fixed", "ladder_fixed", "rule_fixed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
