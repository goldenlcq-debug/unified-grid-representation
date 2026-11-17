"""大规模合成电网族批产(2026-08-18 主线接手后新增;独立目录,不触碰任何冻结集)。

配方沿用 2k 标定(变更记录 2026-07-11):SYNTH_KW load_level=0.55/gen_frac=0.30/q_comp=0.25,
amp=0.5,方式验收 v_floor=0.85,底图验收 base_v_floor=0.87,纯收敛库(keep_divergent_frac=0)。
输出 <out_root>/<fid>/:_base_case/(底图 PSASP 卡片)+ 方式 npz + manifest.csv + cases/(可选,
keep_cases=True 时保留每个方式的卡片目录,供 A 线电池/潮流头评估作 donor)。
种子段 s8xxx,与 train s1xxx / val s2xxx / transfer s3xxx / large400 s4xxx / 2k 冻结 s6000-6005 /
700 s7xxx 不撞。PSASP 唯一裁判;家族级并行(并发潮流已实测与串行逐位一致)。

  python gen_synth_large.py --nodes 2000 --seeds 8000-8002 --n-modes 50 --workers 2
  python gen_synth_large.py --nodes 3000 --seeds 8100 --n-modes 20 --workers 1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("GRIDFM_DATA_ROOT", str(REPO.parent / "gridfm-data")))
sys.path.insert(0, str(REPO))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

DONOR = REPO / "tests" / "fixtures" / "ieee118_golden"
SYNTH_KW = dict(load_level=0.55, gen_frac=0.30, q_comp=0.25)


def _worker(args):
    nodes, seed, out_root, work_root, n_modes, amp, keep_cases = args
    sys.path.insert(0, str(REPO))
    from gridfm.data.sampler import generate_library
    from gridfm.data.synth import synth_family
    from gridfm.fsutil import robust_copytree
    from gridfm.solver.psasp import LicenseError, PsaspSolver

    fid = f"synth{nodes}_s{seed}"
    solver = PsaspSolver()
    case = Path(work_root) / f"case_{fid}"
    work = Path(work_root) / f"mode_{fid}"
    lib = Path(out_root) / fid
    t0 = time.time()
    try:
        if (lib / "manifest.csv").exists():
            return {"fid": fid, "ok": True, "skipped": True, "reason": "已存在(resume)"}
        t_base = time.time()
        info = synth_family(nodes, DONOR, case, solver, seed=seed, **SYNTH_KW)
        base_secs = round(time.time() - t_base, 1)
        if not info.get("converged"):
            return {"fid": fid, "ok": False, "stage": "base", "base_secs": base_secs,
                    "seed_tries": info.get("seed_tries"), "low_v_rejected": info.get("low_v_rejected"),
                    "reason": f"合成底图不收敛/电压病态(试 {info.get('seed_tries')} 种子)"}
        lib.mkdir(parents=True, exist_ok=True)
        robust_copytree(case, lib / "_base_case")
        st = generate_library(case, solver, n_modes, lib, work, seed=seed, family_id=fid,
                              keep_divergent_frac=0.0, v_floor=0.85, max_attempts=3 * n_modes,
                              amp=amp, keep_cases=keep_cases)
        rep = {"fid": fid, "nodes": nodes, "ok": st["n_converged"] >= n_modes,
               "seed_used": info.get("seed_used"), "seed_tries": info.get("seed_tries"),
               "base_vmin": info.get("vmin"), "base_secs": base_secs,
               "n_converged": st["n_converged"], "n_attempts": st["n_attempts"],
               "n_lowv": st.get("n_lowv_quarantined"),
               "yield": round(st["n_converged"] / max(st["n_attempts"], 1), 3),
               "amp": amp, "secs": round(time.time() - t0, 1)}
        (lib / "gen_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        return rep
    except LicenseError as e:
        return {"fid": fid, "ok": False, "fatal": True, "reason": f"加密狗失效:{e}"}
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"fid": fid, "ok": False, "reason": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    finally:
        shutil.rmtree(case, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)


def _parse_seeds(s: str):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--n-modes", type=int, default=50)
    ap.add_argument("--amp", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-root", default=str(DATA / "families" / "synth_large_dev"))
    ap.add_argument("--work-root", default=str(DATA / "_work_large_dev"))
    ap.add_argument("--no-keep-cases", action="store_true")
    a = ap.parse_args()
    out_root = Path(a.out_root); out_root.mkdir(parents=True, exist_ok=True)
    work_root = Path(a.work_root); work_root.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(a.seeds)
    jobs = [(a.nodes, s, str(out_root), str(work_root), a.n_modes, a.amp, not a.no_keep_cases) for s in seeds]
    t0 = time.time()
    results = []
    print(f"[large] nodes={a.nodes} seeds={seeds} n_modes={a.n_modes} amp={a.amp} workers={a.workers} out={out_root}", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_worker, j): j[1] for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"[large] {r['fid']}: {'OK' if r.get('ok') else 'FAIL ' + str(r.get('reason',''))} "
                  f"base_vmin={r.get('base_vmin')} base={r.get('base_secs')}s conv={r.get('n_converged')}/{r.get('n_attempts')} "
                  f"yield={r.get('yield')} lowv={r.get('n_lowv')} total={r.get('secs')}s", flush=True)
            if r.get("trace"):
                print(r["trace"], flush=True)
            if r.get("fatal"):
                print("[large] 加密狗失效,熔断整批", flush=True)
                return 1
    rep = {"nodes": a.nodes, "amp": a.amp, "n_modes": a.n_modes, "families": results,
           "wall_min": round((time.time() - t0) / 60, 1)}
    (out_root / f"build_report_n{a.nodes}_{int(time.time())}.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[large] 完成 {sum(1 for r in results if r.get('ok'))}/{len(seeds)} 家族, wall {rep['wall_min']} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
