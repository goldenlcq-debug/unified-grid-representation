#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配对式 N-0 / N-1 / N-2 方式生产（2026-09-04，长卿；基于 tools_large/produce_modes.py 的逐次记账版）。

同一次抽样（sample_mode 一次）产出配对的三个变体：
  N0 = 只有负荷/出力/电压设定值扰动；N1 = N0 + 随机开断一条在运支路；N2 = N1 + 再开断一条。
开断前做连通性预检（排除桥支路）；三个变体各自送 PSASP 解算，收敛的入库，发散/低压逐条记账。
样本号在三个变体里相同（fid_NNNNN），便于配对比较；每个变体写自己的 manifest.csv（n1_branch 列记开断支路，N2 用 ";" 连接）。

用法：
  python produce_modes_nk.py --fid mp_case118 --base <_base_case> --out <OUT>/mp_case118 --work <WORK>/mp_case118 \
      --seed 9770 --id-start 3000 --n 60 --amp 0.5 [--variants 0,1,2] [--keep-p1d]
--keep-p1d：不清除底图的 LF.P1D 初值卡（70k 热启动需要，交付说明 §8.3）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("GRIDFM_PSASP_BIN",
                      r"D:\simulation\DISK1中国电力科学研究院有限公司7.95.07.10758冬滚正式版\PSASP")
REPO = Path(r"E:\统一计算大模型-迁移包\增量-20260902\N-k其他推理器实验-20260905\grid-foundation")
sys.path.insert(0, str(REPO))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.fsutil import robust_copytree  # noqa: E402
from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.psasp_io.catalog import Catalog  # noqa: E402
from gridfm.schema.assemble import build_sample  # noqa: E402
from gridfm.schema.fields import BRANCH_TABLE, NODE_TABLE, Group  # noqa: E402
from gridfm.schema.sample import group_bit_index  # noqa: E402
from gridfm.data.sampler import sample_mode, _min_valid_voltage  # noqa: E402
from gridfm.solver.psasp import LicenseError, PsaspSolver  # noqa: E402
from psasp_guard import guarded  # noqa: E402  2026-09-05：加密狗不响应守卫

RESULT_CARDS = ("LF.CAL", "LF.LP1", "LF.LP2", "LF.LP3", "LF.LP5", "LF.LP6", "LF.LP30", "LF.LP31", "LF.P1D",
                "LFERR.LIS", "ISWARN.TXT", "CHECKIS.TXT", "LFPORT.LIS", "lfreport.lis", "slack.lis",
                "dcreport.lis", "upfcreport.lis")
MANIFEST_COLS = ["sample_id", "family_id", "solvable", "load_factor", "decommit",
                 "theta_absmax_deg", "n_wrap_edges", "n1_branch"]
OFF_TH, OFF_MK = NODE_TABLE.offset_of("theta"), BRANCH_TABLE.offset_of("mark")
SBIT = group_bit_index(Group.STATE)


def rmtree_robust(path: Path, tries: int = 20, sleep: float = 0.5) -> None:
    for k in range(tries):
        try:
            if path.exists():
                shutil.rmtree(path)
            return
        except OSError:
            time.sleep(sleep * (1 + k))
    try:
        path.rename(path.with_name(f"{path.name}_stale_{int(time.time())}"))
    except OSError:
        pass


def strip_results(d: Path, keep_p1d: bool) -> None:
    for f in RESULT_CARDS:
        if keep_p1d and f == "LF.P1D":
            continue
        (d / f).unlink(missing_ok=True)


def make_clean_base(src: Path, dst: Path, keep_p1d: bool) -> None:
    rmtree_robust(dst)
    robust_copytree(src, dst)
    strip_results(dst, keep_p1d)


def theta_stats(gs):
    th = gs.node_feat[:, OFF_TH].astype(np.float64)
    ok = ~gs.node_missing[:, SBIT]
    d = np.abs(th[gs.edge_index[0]] - th[gs.edge_index[1]])
    amax = float(np.degrees(np.abs(th[ok]).max())) if ok.any() else float("nan")
    return round(amax, 3), int(((d > np.pi) & (gs.branch_feat[:, OFF_MK] > 0.5)).sum())


def _in_service_branches(case):
    ac, xf = case.aclines(), case.xfmrs()
    return ([("L", l) for l in ac if l.mark == 1 and l.i_no != l.j_no]
            + [("T", t) for t in xf if t.mark == 1])


def _connected_without(case, excluded) -> bool:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ac, xf, n = case.aclines(), case.xfmrs(), int(case.n_buses)
    e = [(a.i_no - 1, a.j_no - 1) for a in ac if a.mark == 1 and a.i_no != a.j_no and a not in excluded]
    e += [(abs(t.i_no) - 1, abs(t.j_no) - 1) for t in xf if t.mark == 1 and t not in excluded]
    if not e:
        return False
    ii, jj = zip(*e)
    A = coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n))
    return connected_components(A, directed=False)[0] == 1


def pick_branch(case, frng, excluded, tries: int = 30):
    """在 excluded 之外随机选一条在运支路，要求去掉 excluded ∪ {它} 后仍连通。返回 (kind, view) 或 None。"""
    cand = [(k, b) for k, b in _in_service_branches(case) if b not in excluded]
    if not cand:
        return None
    for _ in range(tries):
        kind, b = cand[int(frng.integers(0, len(cand)))]
        if _connected_without(case, list(excluded) + [b]):
            return kind, b
    return None


def branch_key(kind, b) -> str:
    return f"{kind}{int(b.i_no)}-{int(b.j_no)}-{int(b.id_no)}"


def open_branches(case, keys) -> None:
    ac, xf = case.aclines(), case.xfmrs()
    for key in keys:
        kind = key[0]
        i, j, idn = key[1:].split("-")
        views = ac if kind == "L" else xf
        hit = [b for b in views if int(b.i_no) == int(i) and int(b.j_no) == int(j) and int(b.id_no) == int(idn)]
        if not hit:
            raise KeyError(f"支路 {key} 在副本中未找到")
        hit[0].mark = 0


def write_rows(path: Path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fid", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--id-start", type=int, default=0)
    ap.add_argument("--n", type=int, required=True, help="目标抽样次数（每次抽样产 N0/N1/N2 三个变体）")
    ap.add_argument("--variants", default="0,1,2")
    ap.add_argument("--amp", type=float, default=0.5)
    ap.add_argument("--v-floor", type=float, default=0.85)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--keep-p1d", action="store_true")
    ap.add_argument("--save-pack", default=None,
                    help="目录：任一变体发散/低压/出错的抽样，把三个变体的 PSASP 工作目录与未解算扰动底图原样落盘（调收敛线用）")
    ap.add_argument("--save-all-cards", default=None,
                    help="目录：每个收敛入库(OK)的变体，把 PSASP 已解算的卡片目录整份落盘到 <DIR>/N{k}/<sid>/（2026-09-05 N-k 线：给调整器/生成器当源方式池）")
    a = ap.parse_args()
    variants = [int(x) for x in a.variants.split(",")]
    kmax = max(variants)

    out, work, base = Path(a.out), Path(a.work), Path(a.base)
    out.mkdir(parents=True, exist_ok=True)
    for k in variants:
        (out / f"N{k}").mkdir(parents=True, exist_ok=True)
    clean = work.parent / f"{work.name}_cleanbase"
    make_clean_base(base, clean, a.keep_p1d)
    rng, frng = np.random.default_rng(a.seed), np.random.default_rng(a.seed + 7_000_000)
    solver = PsaspSolver()
    rows = {k: [] for k in variants}
    lowv = {k: [] for k in variants}
    attempts_log = []
    counts = {k: dict(ok=0, div=0, lowv=0, nopick=0) for k in variants}
    t0 = time.time()
    print(f"[produce-nk] {a.fid} seed={a.seed} id_start={a.id_start} draws={a.n} variants={variants} "
          f"amp={a.amp} keep_p1d={a.keep_p1d} engine={solver.lf_exe}", flush=True)
    draw_dir = work.parent / f"{work.name}_draw"
    for i in range(a.n):
        sid = f"{a.fid}_{a.id_start + i:05d}"
        # ---- 一次抽样：负荷/出力/电压设定值扰动，落到 draw_dir（未解算）----
        rmtree_robust(draw_dir)
        robust_copytree(clean, draw_dir)
        strip_results(draw_dir, a.keep_p1d)
        case = PsaspCase(draw_dir)
        spec = sample_mode(case, rng, amp=a.amp)
        case.save_inplace()
        # ---- 选开断支路（内存里选，不落盘）：b1，再在去掉 b1 的网上选 b2 ----
        keys = []
        scratch = PsaspCase(draw_dir)
        excluded = []
        for k in range(1, kmax + 1):
            r = pick_branch(scratch, frng, excluded)
            if r is None:
                break
            kind, b = r
            keys.append(branch_key(kind, b))
            excluded.append(b)
        # ---- 逐变体解算 ----
        for k in variants:
            rec = dict(attempt=i, variant=f"N{k}", sample_id=sid, load_factor=round(spec.load_factor, 6),
                       n1_branch=";".join(keys[:k]), outcome="", vmin="", secs="")
            if k > len(keys):
                rec["outcome"] = "NO_BRANCH_PICK"; counts[k]["nopick"] += 1
                attempts_log.append(rec); continue
            wk = work.with_name(f"{work.name}_N{k}")
            rmtree_robust(wk)
            robust_copytree(draw_dir, wk)
            if k:
                ck = PsaspCase(wk)
                open_branches(ck, keys[:k])
                ck.save_inplace()
            t1 = time.time()
            try:
                pf = guarded(lambda: solver.run_powerflow(wk, timeout=a.timeout), wk, lambda p: bool(p.converged), tag=f"{sid}/N{k}")
            except LicenseError:
                rec["outcome"] = "LICENSE_ERROR"; attempts_log.append(rec)
                write_rows(out / "attempts.csv", attempts_log, list(rec.keys()))
                raise
            except Exception as e:  # noqa: BLE001
                rec["outcome"] = f"ERROR:{type(e).__name__}"; rec["secs"] = round(time.time() - t1, 1)
                attempts_log.append(rec); continue
            rec["secs"] = round(time.time() - t1, 1)
            if not pf.converged:
                rec["outcome"] = "DIVERGED"; counts[k]["div"] += 1
                attempts_log.append(rec); continue
            solved = PsaspCase(wk)
            gs = build_sample(solved, Catalog(solved), sample_id=sid, family_id=a.fid)
            vmin = _min_valid_voltage(gs)
            rec["vmin"] = round(float(vmin), 4) if np.isfinite(vmin) else ""
            if np.isfinite(vmin) and vmin < a.v_floor:
                q = out / f"N{k}" / "_lowv_quarantine"
                q.mkdir(exist_ok=True)
                gs.save_npz(q / f"{sid}.npz")
                lowv[k].append(dict(sample_id=sid, family_id=a.fid, solvable=1, load_factor=spec.load_factor,
                                    min_v_pu=float(vmin), n1_branch=";".join(keys[:k])))
                write_rows(q / "manifest.csv", lowv[k], list(lowv[k][0].keys()))
                rec["outcome"] = "LOWV_QUARANTINED"; counts[k]["lowv"] += 1
                attempts_log.append(rec); continue
            amax, nwrap = theta_stats(gs)
            gs.save_npz(out / f"N{k}" / f"{sid}.npz")
            rows[k].append(dict(sample_id=sid, family_id=a.fid, solvable=1, load_factor=spec.load_factor,
                                decommit=0.0, theta_absmax_deg=amax, n_wrap_edges=nwrap, n1_branch=";".join(keys[:k])))
            write_rows(out / f"N{k}" / "manifest.csv", rows[k], MANIFEST_COLS)
            counts[k]["ok"] += 1
            rec["outcome"] = "OK"; attempts_log.append(rec)
            if a.save_all_cards:
                dk = Path(a.save_all_cards) / f"N{k}" / sid
                rmtree_robust(dk)
                robust_copytree(wk, dk)
                (dk / "nk_meta.json").write_text(json.dumps(dict(
                    sample_id=sid, family_id=a.fid, variant=f"N{k}", seed=a.seed, attempt=i,
                    load_factor=round(spec.load_factor, 6), n1_branch=";".join(keys[:k]), opened_branches=keys[:k],
                    vmin=(round(float(vmin), 4) if np.isfinite(vmin) else None), theta_absmax_deg=amax,
                    recipe=dict(amp=a.amp, v_floor=a.v_floor, keep_p1d=a.keep_p1d)), ensure_ascii=False, indent=1), encoding="utf-8")
        write_rows(out / "attempts.csv", attempts_log, ["attempt", "variant", "sample_id", "load_factor", "n1_branch", "outcome", "vmin", "secs"])
        if a.save_pack:
            recs = [r for r in attempts_log if r["sample_id"] == sid]
            if any(r["outcome"] != "OK" for r in recs):
                pk = Path(a.save_pack) / sid
                rmtree_robust(pk)
                pk.mkdir(parents=True, exist_ok=True)
                robust_copytree(draw_dir, pk / "draw_unsolved")
                for k in variants:
                    wk = work.with_name(f"{work.name}_N{k}")
                    if wk.exists():
                        robust_copytree(wk, pk / f"N{k}")
                (pk / "draw.json").write_text(json.dumps(dict(
                    sample_id=sid, family_id=a.fid, seed=a.seed, attempt=i, load_factor=round(spec.load_factor, 6),
                    opened_branches=keys, variants={r["variant"]: dict(outcome=r["outcome"], vmin=r["vmin"], secs=r["secs"]) for r in recs},
                    category=("topology_induced" if (recs and recs[0]["variant"] == "N0" and recs[0]["outcome"] == "OK") else "load_induced"),
                    recipe=dict(amp=a.amp, v_floor=a.v_floor, keep_p1d=a.keep_p1d)), ensure_ascii=False, indent=1), encoding="utf-8")
        if (i + 1) % 10 == 0:
            print(f"[produce-nk] draws {i + 1}/{a.n} " + " ".join(f"N{k}:ok{c['ok']}/div{c['div']}/lowv{c['lowv']}" for k, c in counts.items())
                  + f" {time.time() - t0:.0f}s", flush=True)
    stats = dict(family_id=a.fid, seed=a.seed, id_start=a.id_start, n_draws=a.n, variants=variants, counts=counts,
                 secs=round(time.time() - t0, 1),
                 recipe=dict(amp=a.amp, v_floor=a.v_floor, dispatch="proportional", p_n1=0.0, p_dec=0.0,
                             paired_variants=True, keep_p1d=a.keep_p1d, timeout=a.timeout),
                 base_case=str(base), engine=str(solver.lf_exe))
    (out / "batch_report.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[produce-nk] done " + json.dumps(stats, ensure_ascii=False), flush=True)
    for d in (draw_dir,) + tuple(work.with_name(f"{work.name}_N{k}") for k in variants):
        rmtree_robust(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
