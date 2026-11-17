"""[G] ① 预筛头定向补标(2026-08-03;预注册 = docs/预注册-预筛头补标-20260803.md)。

两段错时执行:
  --gen   GPU 段:重放名单涉及的 (族, z_seed) 采样,导出名单候选的 4 字段 pred
          数组落盘(零 PSASP);每 (族,种子) 组合首条候选与 features_s1.csv 已存
          特征逐位对齐(重放确定性护栏);
  --trial PSASP 段:纯 CPU 串行逐条 writeback_and_trial,断点续跑(跳已有结果),
          partial 先置真,每 10 条 flush;LicenseError 即停等用户。

  python scripts/g5_probe_label.py --stage 1 --gen
  python scripts/g5_probe_label.py --stage 1 --trial
"""
import argparse
import copy
import csv
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
from gridfm.fsutil import robust_rmtree  # noqa: E402
from gridfm.schema.fields import NODE_TABLE, Group  # noqa: E402
from gridfm.schema.sample import group_bit_index  # noqa: E402
from gridfm.solver.psasp import LicenseError, PsaspSolver  # noqa: E402
from gridfm.train.masking import COL_GROUP  # noqa: E402

CFG_SCALE = 1.5
DENOISER = "runs/diffusion_g1_cgen_g4_s1/denoiser.pt"     # 名单臂 = s1(val8)
CALIB = "runs/g_clabel_calib/calib_s1.json"


def load_list(stage):
    pl = json.load(open(REPO / "runs/g5_prescreen/probe_list.json", encoding="utf-8"))
    key = {1: "stage1_top2", 2: "stage2_top3to4"}[stage]
    return pl[key]


def known_labels():
    """当前已知标签键集(护栏:名单候选必须 unknown)。"""
    keys = set()
    with open(REPO / "runs/g5_prescreen/features_s1.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["label"] != "unknown":
                keys.add((r["fam"], int(r["z_seed"]), r["sid"]))
    return keys


def csv_row_index():
    """features_s1.csv 的 (fam,z,sid) → 行(重放确定性对齐用)。"""
    idx = {}
    with open(REPO / "runs/g5_prescreen/features_s1.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx[(r["fam"], int(r["z_seed"]), r["sid"])] = r
    return idx


def do_gen(stage, out):
    items = load_list(stage)
    print(f"[probe-gen] stage{stage}: {len(items)} 候选", flush=True)
    preds = _gen_items(items, out, stage_tag=stage)
    print(f"[probe-gen] {len(preds)} 条 pred 落盘(对齐护栏在 _gen_items 内逐组合过)",
          flush=True)


def do_trial(stage, out):
    items = load_list(stage)
    preds = np.load(out / f"preds_stage{stage}.npz", allow_pickle=False)
    missing = [it for it in items
               if f"{it['fam']}|{it['z_seed']}|{it['sid']}" not in preds]
    assert not missing, f"m5 预检:preds 缺 {len(missing)} 条(先跑 --gen):{missing[:3]}"
    if stage == 2:                          # m5 阶段闸:阶段 1 判定前不得烧阶段 2
        s1p = out / "labels_stage1.json"
        assert s1p.exists() and not json.loads(
            s1p.read_text(encoding="utf-8"))["partial"], \
            "阶段闸:labels_stage1 未完成,阶段 2 禁烧(预注册升级规则)"
    lab_path = out / f"labels_stage{stage}.json"
    rep = (json.loads(lab_path.read_text(encoding="utf-8"))
           if lab_path.exists() else {"task": f"补标 stage{stage}(预注册见 docs)",
                                      "partial": True, "rows": {}})
    solver = PsaspSolver()
    t0 = time.time()
    ndone = 0
    for it in items:
        key = f"{it['fam']}|{it['z_seed']}|{it['sid']}"
        if key in rep["rows"]:
            continue
        pred = preds[key]
        famdir = DATA / "families" / "val" / it["fam"]
        work = out / "work" / f"{it['sid']}_z{it['z_seed']}"
        conv, vmin, vmax = writeback_and_trial(
            pred, famdir / "cases" / it["sid"], work, solver)
        ok = bool(conv and vmin is not None and vmin >= VMIN_LO
                  and vmax is not None and vmax <= VMAX_HI)
        label = "ok" if ok else ("viol" if conv else "nonconv")
        if conv and not ok:
            dst = out / "violations" / f"{it['sid']}_z{it['z_seed']}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if work.exists():
                robust_rmtree(dst)          # m4:断点复跑防改名撞车
                work.rename(dst)
        else:
            robust_rmtree(work)
        rep["rows"][key] = {
            **it, "conv": bool(conv),
            "vmin": None if vmin is None else round(float(vmin), 4),
            "vmax": None if vmax is None else round(float(vmax), 4), "label": label}
        ndone += 1
        if ndone % 10 == 0:
            lab_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                encoding="utf-8")
            el = time.time() - t0
            print(f"[probe] {ndone} 条新标(总 {len(rep['rows'])}/{len(items)},"
                  f"{el:.0f}s,{el/ndone:.1f}s/条)", flush=True)
    cnt = defaultdict(int)
    for v in rep["rows"].values():
        cnt[v["label"]] += 1
    rep["summary"] = dict(cnt)
    rep["partial"] = len(rep["rows"]) < len(items)
    lab_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[probe] stage{stage} 完成 {len(rep['rows'])}/{len(items)}:{dict(cnt)}"
          f";partial={rep['partial']}", flush=True)


def do_control(out):
    """M1 跨工具链正控:5 条已知标签(2 ok + 2 nonconv + 1 viol,跨族)同通路复测,
    断言三态复现;只体检不入账。"""
    picked, seen_fam = [], defaultdict(set)
    with open(REPO / "runs/g5_prescreen/features_s1.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["split"] == "val" and r["z_seed"] != "0"
                and r["label"] in ("ok", "nonconv", "viol")]
    rows.sort(key=lambda r: (r["fam"], int(r["z_seed"]), r["sid"]))
    quota = {"ok": 2, "nonconv": 2, "viol": 1}
    for r in rows:                          # 每类跨族取首条(确定性选择)
        lab = r["label"]
        if quota[lab] > 0 and r["fam"] not in seen_fam[lab]:
            picked.append(r)
            seen_fam[lab].add(r["fam"])
            quota[lab] -= 1
        if not any(quota.values()):
            break
    assert not any(quota.values()), f"正控选样不足:{quota}"
    print("[control] 选样:", [(r["fam"], r["z_seed"], r["sid"], r["label"])
                               for r in picked], flush=True)
    # gen(复用 do_gen 的通路:临时名单)
    items = [{"fam": r["fam"], "z_seed": int(r["z_seed"]), "sid": r["sid"],
              "rank": 0, "score": 0.0} for r in picked]
    tmp = out / "control"
    tmp.mkdir(parents=True, exist_ok=True)
    _gen_items(items, tmp, skip_known_guard=True)
    preds = np.load(tmp / "preds_stage0.npz", allow_pickle=False)
    solver = PsaspSolver()
    bad = []
    for r in picked:
        key = f"{r['fam']}|{r['z_seed']}|{r['sid']}"
        work = tmp / "work" / f"{r['sid']}_z{r['z_seed']}"
        conv, vmin, vmax = writeback_and_trial(
            preds[key], DATA / "families" / "val" / r["fam"] / "cases" / r["sid"],
            work, solver)
        ok = bool(conv and vmin is not None and vmin >= VMIN_LO
                  and vmax is not None and vmax <= VMAX_HI)
        got = "ok" if ok else ("viol" if conv else "nonconv")
        robust_rmtree(work)
        status = "PASS" if got == r["label"] else "FAIL"
        print(f"[control] {key}: want {r['label']} got {got} "
              f"(conv={conv} vmin={vmin} vmax={vmax}) {status}", flush=True)
        if got != r["label"]:
            bad.append((key, r["label"], got))
    if bad:
        print(f"[control] 不复现 {len(bad)} 条 → 停,报用户:{bad}", flush=True)
        return 3
    print("[control] CONTROL-PASS:5/5 三态逐条复现,工具链体检过", flush=True)
    return 0


def _gen_items(items, out, skip_known_guard=False, stage_tag=0):
    """do_gen 的核心通路,供正控复用(名单直给)。"""
    if not skip_known_guard:
        known = known_labels()
        for it in items:
            assert (it["fam"], it["z_seed"], it["sid"]) not in known, \
                f"名单候选已有标签:{it}"
    rowidx = csv_row_index()
    by_fz = defaultdict(list)
    for it in items:
        by_fz[(it["fam"], it["z_seed"])].append(it["sid"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = load_stack(dev, DENOISER)
    calib = json.loads((REPO / CALIB).read_text(encoding="utf-8"))
    supp = calib["support"]
    o = NODE_TABLE.offset_of
    col = {f: o(f) for f in MODE_FIELDS}
    sbit = group_bit_index(Group.STATE)
    scols = np.where(COL_GROUP == sbit)[0]
    preds = {}
    fams = sorted({f for f, _z in by_fz})
    for fam in fams:
        F = prep_family(fam, 20, S, dev, "val")
        cp = resolve_cp(calib, CFG_SCALE, fam, "pooled", supp)
        lab_c = build_labels(F, "uncal", cp, S["lnorm"], dev)
        lab_u = torch.stack([torch.from_numpy(CL.encode(None, S["lnorm"])).to(dev)] * F["N"])
        n, Nf = F["n"], F["N"]
        net, dec, diff = S["net"], S["dec"], S["diff"]
        z_mean, z_std = S["z_mean"], S["z_std"]
        sid_idx = {F["test_s"][j].sample_id: j for j in range(Nf)}
        base_vset = F["base_feat"][:, col["v_set"]]

        @torch.no_grad()
        def eps_fn(z_t, t):
            tt = torch.full((Nf,), t, dtype=torch.long, device=dev)
            e_c = net(z_t, F["hc"], F["ec"], F["ei"], tt, lab_c, F["npg"])
            e_u = net(z_t, F["hc"], F["ec"], F["ei"], tt, lab_u, F["npg"])
            return e_u + CFG_SCALE * (e_c - e_u)

        for (fam2, z_seed), sids in sorted(by_fz.items()):
            if fam2 != fam:
                continue
            torch.manual_seed(z_seed)
            z_T = torch.randn(Nf * n, S["d"], device=dev)
            with torch.no_grad():
                z0 = diff.ddim_sample(eps_fn, z_T, steps=250, clip_x0=4.0)
                fields = dec(z0 * z_std + z_mean, F["ec"], F["ei"]).cpu().numpy()
            first = True
            for sid in sorted(sids):
                j = sid_idx[sid]
                full = F["normed"][j].node_feat.copy()
                full[:, scols] = fields[j * n:(j + 1) * n][:, scols]
                pu = S["norm"].inverse_transform_node(full)
                pu[:, col["v_set"]] = base_vset
                if first:
                    g = copy.deepcopy(F["test_s"][j]); g.node_feat = pu
                    gl = CL.recompute(g, F["base_pl"])
                    ref = rowidx[(fam, z_seed, sid)]
                    assert (round(float(gl["load_level"]), 4) == float(ref["gen_load"])
                            and int(gl["online_count"]) == int(ref["gen_online"])), \
                        f"重放对齐失败:{fam} z{z_seed} {sid}"
                    first = False
                pred = np.stack([pu[:, col[f]] for f in MODE_FIELDS], axis=1)
                preds[f"{fam}|{z_seed}|{sid}"] = pred.astype(np.float32)
        del F
        torch.cuda.empty_cache()
        print(f"[probe-gen] {fam} 完成(累计 {len(preds)})", flush=True)
    np.savez_compressed(out / f"preds_stage{stage_tag}.npz", **preds)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--control", action="store_true",
                    help="M1 跨工具链正控(5 条已知标签复测;点火第一步)")
    a = ap.parse_args()
    out = REPO / "runs/g5_probe"
    out.mkdir(parents=True, exist_ok=True)
    if a.control:
        try:
            return do_control(out)
        except LicenseError:
            print("[probe] slm_init error:加密狗失效,中止等用户", flush=True)
            return 2
    if a.gen:
        do_gen(a.stage, out)
    if a.trial:
        try:
            do_trial(a.stage, out)
        except LicenseError:
            print("[probe] slm_init error:加密狗失效,已标部分落盘,中止等用户", flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
