"""调整标签病态方式落料(R3 训练样本原料;方向二)。

动因:adjust_labels 生产时病态方式在临时目录、随产随弃——标签行只存了扰动量与母线号。
R3 训练需要每个病态方式的**已解档案**(病态方式收敛但越限,其解后状态是合法输入,
与判稳任务同口径:输入 = 已解方式标准档案;泄漏红线 = 修复标签不得入输入)。

本脚本确定性重建:标签行 (sid, applied 母线→ΔP) → 拷基态方式 → 写入 ΔP(GenView)→
PSASP 重解(同一扰动同一解)→ 校验 vmin 与标签行逐位吻合(不吻合即 raise,防错位落料)
→ 保存已解方式目录 + 装配 npz 档案(build_sample,v0.3 全 schema)。

  python scripts/adjust_materialize.py --lib <数据根>/lib_cepri36 \
      --labels runs/adjust_labels_cepri36_r2_s1 [more...] --out runs/adjust_cases_cepri36
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

from gridfm.psasp_io.case import PsaspCase  # noqa: E402
from gridfm.psasp_io.catalog import Catalog  # noqa: E402
from gridfm.psasp_io.lowlevel import Card  # noqa: E402
from gridfm.schema.assemble import build_sample  # noqa: E402
from scripts.adjust_labels import _referee, eff_bounds, read_gens, share_by_headroom  # noqa: E402


def apply_fix_by_bus(l5: Card, deltas_by_bus: dict, fixed_mode: str = "unlimited", matpower: bool = False) -> dict:
    """按母线号把 ΔP 加到 LF.L5(GenView 版式自适应)。2026-08-23 改:
    ① 固定出力机组(按 fixed_mode 判定,见 adjust_ext.fixed_unit_gis;缺省 "unlimited" = 2026-08-21 版行为,run_adjust 传 auto)
       的 ΔP 直接丢弃并计数,不再施加(B 段 3120 有功"成功"80~90% 的 ΔP 落在固定机组上);
    ② 一母线多机记录按该母线各机在 ΔP 方向上的余量比例分摊(原来 raise ValueError 整例崩);
    ③ 返回 {"applied": {bus: 实际ΔP(裁剪后)}, "dropped_fixed": {bus: ΔP}, "missing": [bus], "multi_gen_buses": [bus]}。"""
    try:
        from scripts.adjust_ext import fixed_unit_gis
    except ImportError:                      # scripts 目录直接在 sys.path 上(run_adjust)
        from adjust_ext import fixed_unit_gis
    _slack, gens = read_gens(l5)
    fixed = fixed_unit_gis(gens, fixed_mode, matpower)
    by_bus: dict = {}
    for gi, g in gens.items():
        by_bus.setdefault(g["bus"], []).append(gi)
    out = {"applied": {}, "dropped_fixed": {}, "missing": [], "multi_gen_buses": []}
    for bus_s, dp in deltas_by_bus.items():
        bus = int(bus_s)
        dp = float(dp)
        gis = by_bus.get(bus)
        if not gis:
            out["missing"].append(bus)
            continue
        pool = [gi for gi in gis if gi not in fixed]
        if not pool:
            out["dropped_fixed"][bus] = round(dp, 4)
            continue
        if len(pool) > 1:
            out["multi_gen_buses"].append(bus)
            states = {gi: {"p": gens[gi]["view"].pg, "pmin": gens[gi]["pmin"], "pmax": gens[gi]["pmax"]} for gi in pool}
            share = share_by_headroom(states, pool, dp) or {pool[0]: dp}
        else:
            share = {pool[0]: dp}
        got = 0.0
        for gi, d in share.items():
            g = gens[gi]
            lo, hi = eff_bounds(g["pmin"], g["pmax"], g["view"].pg)
            p1 = float(np.clip(g["view"].pg + float(d), lo, hi))
            got += p1 - g["view"].pg
            g["view"].pg = p1
        out["applied"][bus] = round(got, 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--labels", nargs="+", required=True,
                    help="adjust_labels 输出目录(可多个批)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vmin-tol", type=float, default=5e-4,
                    help="重解 vmin 与标签行吻合容差(超差 = 落料错位,raise)")
    a = ap.parse_args()

    from gridfm.solver.psasp import PsaspSolver
    solver = PsaspSolver()
    lib = Path(a.lib)
    out = Path(a.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    manifest = []
    n_ok = n_skip = 0
    for lab_dir in a.labels:
        lab = Path(lab_dir)
        rows = [json.loads(ln) for ln in
                (lab / "adjust_labels.jsonl").read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        batch = lab.name
        for i, r in enumerate(rows):
            if not r.get("feasible"):
                continue
            tag = f"{batch}__{r['sid']}__{i:03d}"
            dst = out / "cases" / tag
            if dst.exists() and (dst / "_materialized.json").exists():
                n_skip += 1
                continue                       # 断点续跑
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(lib / "cases" / r["sid"], dst)
            # 病态 = 基态 + 生产时施加的扰动 = 反向修法#1 的负向
            rev = next(f for f in r["fixes"] if f["source"] == "reverse_perturb")
            applied = {b: -dp for b, dp in rev["deltas"].items()}
            l5 = Card.load(dst / "LF.L5")
            apply_fix_by_bus(l5, applied)
            l5.save(dst / "LF.L5")
            rf = _referee(dst, solver)
            if not rf.get("converged"):
                raise RuntimeError(f"{tag}: 病态方式重解不收敛(生产时收敛)——落料错位")
            if abs(rf["vmin"] - r["violation"]["vmin"]) > a.vmin_tol:
                raise RuntimeError(f"{tag}: vmin 重解 {rf['vmin']:.4f} vs 标签 "
                                   f"{r['violation']['vmin']}(容差 {a.vmin_tol})——落料错位")
            case = PsaspCase(dst)
            s = build_sample(case, Catalog(case), sample_id=tag,
                             family_id=r["battery"], schema="v0.3")
            s.save_npz(out / "cases" / f"{tag}.npz")
            (dst / "_materialized.json").write_text(json.dumps(
                {"tag": tag, "battery": r["battery"], "sid": r["sid"],
                 "vmin": rf["vmin"], "vmax": rf["vmax"],
                 "n_fixes": r["n_fixes"]}, ensure_ascii=False), encoding="utf-8")
            manifest.append({"tag": tag, "battery": r["battery"], "sid": r["sid"],
                             "batch": batch, "row_index": i,
                             "vmin": round(rf["vmin"], 4)})
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"[mat] 已落料 {n_ok}", flush=True)
    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in manifest) + "\n",
        encoding="utf-8")
    print(f"[mat] 收尾:落料 {n_ok}(续跑跳过 {n_skip});{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
