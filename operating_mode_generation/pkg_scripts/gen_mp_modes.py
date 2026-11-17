# -*- coding: utf-8 -*-
"""大算例样本线(2026-08-20):在 MATPOWER→PSASP 转换底图上批产运行方式库。

口径与 B 包契约 §B3.2 /《02》§5 逐位对齐:keep_divergent_frac=0.0、v_floor=0.85、
max_attempts=3*n_modes、amp=0.5、keep_cases=True、schema 默认 v0、diverse=False
(基础 sample_mode;dispatch 只在 diverse=True 时生效,此处显式传 proportional 作防呆)。
种子段 s91xx(本线专用,避开 B 包示例的 9000 起与主线 1xxx~8xxx)。
纪律:PSASP 串行唯一裁判;LicenseError 立即熔断不重试。

  python gen_mp_modes.py --case ACTIVSg2000 --fid mp_activsg2000 --seed 9101 --n-modes 30
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from pathlib import Path as _P
_PKG = _P(__file__).resolve().parent.parent   # 包根(本文件在 scripts/ 下)
REPO = _PKG / "grid-foundation"
LINE = _PKG
CASES = _PKG / "cases"
MP = _PKG / "matpower_cases"

# [外协包内版] 路径由文件头 _PKG 接管: REPO
# [外协包内版] 路径由文件头 _PKG 接管: CASES
# [外协包内版] 路径由文件头 _PKG 接管: LINE
sys.path.insert(0, str(REPO))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def _dir_mb(p: Path) -> float:
    tot = 0
    for r, _d, fs in os.walk(p):
        for f in fs:
            try:
                tot += (Path(r) / f).stat().st_size
            except OSError:
                pass
    return round(tot / 1024 / 1024, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)          # cases/ 下的底图目录名
    ap.add_argument("--fid", required=True)           # family_id
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-modes", type=int, default=30)
    ap.add_argument("--psasp-bin", default=None, help="PSASP 安装目录(外协机器必填)")
    ap.add_argument("--psasp-exe", default=None, help="潮流可执行名(缺省 WMLFRTMsg.exe;Linux 版不带 .exe 时指定)")
    ap.add_argument("--amp", type=float, default=0.5)
    ap.add_argument("--v-floor", type=float, default=0.85)
    ap.add_argument("--schema", default="v0")
    ap.add_argument("--no-keep-cases", action="store_true")
    ap.add_argument("--id-start", type=int, default=0)   # 批次号段隔离(b2=1000 起)
    ap.add_argument("--out-name", default=None)          # 库目录名,默认=fid
    ap.add_argument("--keep-divergent-frac", type=float, default=0.0)  # 负类考卷重放用(1.0=发散全留)
    # 电压带闭环校正(外协包 E,2026-08-25;详见 sampler.generate_library docstring)
    ap.add_argument("--mp-file", default=None,
                    help="逐母线电压带来源 .m 文件名(缺省按 --case 去 _vNN 后缀自动推断;找不到则退回历史行为)")
    ap.add_argument("--no-vfix", action="store_true", help="关闭电压带校正,退回历史采样行为")
    ap.add_argument("--vfix-iters", type=int, default=2, help="解算后闭环修正轮数上限(每轮 1 次重解)")
    ap.add_argument("--no-vfix-prior", action="store_true", help="关闭底图静态先验(仅留闭环)")
    a = ap.parse_args()

    from gridfm.data.sampler import generate_library
    from gridfm.fsutil import robust_copytree
    from gridfm.solver.psasp import LicenseError, PsaspSolver

    base = CASES / a.case
    if not (base / "LF.L0").exists():
        print(f"[mp] 底图不存在或不是 PSASP 数据目录: {base}", flush=True)
        return 2
    lib = LINE / "libs" / (a.out_name or a.fid)
    work = LINE / "_work" / (a.out_name or a.fid)
    if (lib / "manifest.csv").exists():
        print(f"[mp] {a.fid} 已存在 manifest.csv,拒绝覆盖(新产物落新目录纪律)", flush=True)
        return 3
    lib.mkdir(parents=True, exist_ok=True)
    work.parent.mkdir(parents=True, exist_ok=True)

    # 底图卡片随库留档(_base_case),与 gen_synth_large 一致
    robust_copytree(base, lib / "_base_case")
    for extra in ("convert_info.json", "validate.json"):
        if (base / extra).exists():
            shutil.copy2(base / extra, lib / extra)

    # 电压带闭环校正:限值与验收计分器同源(.m 的 VMIN/VMAX 列)。找不到 .m 时退回
    # 历史行为并明示——宁可少改进,不可静默用错带。
    v_band = None
    mp_name = None
    if not a.no_vfix:
        import re as _re
        sys.path.insert(0, str(_PKG / "scripts"))
        mp_name = a.mp_file or (_re.sub(r"_v\d+$", "", a.case) + ".m")
        mp_path = MP / mp_name
        if mp_path.is_file():
            from audit_static_feasibility import read_m
            _mbase, _mbus, _mbr = read_m(mp_path)
            v_band = (_mbus[:, 12].copy(), _mbus[:, 11].copy())
            print(f"[mp] 电压带校正开: {mp_name} ({len(_mbus)} 母线) "
                  f"iters={a.vfix_iters} prior={not a.no_vfix_prior}", flush=True)
        else:
            print(f"[mp] 警告: 找不到 {mp_path},电压带校正关(退回历史采样行为)", flush=True)

    keep_cases = not a.no_keep_cases
    cfg = dict(case=a.case, family_id=a.fid, seed=a.seed, id_start=a.id_start, n_modes=a.n_modes, amp=a.amp,
               v_floor=a.v_floor, keep_divergent_frac=a.keep_divergent_frac, max_attempts=3 * a.n_modes,
               keep_cases=keep_cases, schema=a.schema, diverse=False,
               diverse_dispatch="proportional",
               vfix=(v_band is not None), vfix_mp_file=(mp_name if v_band is not None else None),
               vfix_iters=(a.vfix_iters if v_band is not None else 0),
               vfix_prior=(not a.no_vfix_prior) if v_band is not None else False)
    print(f"[mp] 开跑 {json.dumps(cfg, ensure_ascii=False)}", flush=True)

    solver = PsaspSolver(psasp_bin=a.psasp_bin) if a.psasp_bin else PsaspSolver()
    if a.psasp_exe:
        solver.lf_exe = Path(solver.bin) / a.psasp_exe
    t0 = time.time()
    try:
        st = generate_library(
            base, solver, a.n_modes, lib, work,
            seed=a.seed, family_id=a.fid,
            keep_divergent_frac=a.keep_divergent_frac, v_floor=a.v_floor,
            max_attempts=3 * a.n_modes, amp=a.amp,
            keep_cases=keep_cases, schema=a.schema, id_start=a.id_start,
            diverse=False, diverse_dispatch="proportional",
            v_band=v_band, vfix_iters=a.vfix_iters,
            vfix_prior=not a.no_vfix_prior)
    except LicenseError as e:
        print(f"[mp] 加密狗失效熔断,立即停止不重试: {e}", flush=True)
        return 1
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"[mp] 失败 {type(e).__name__}: {e}\n{traceback.format_exc()[-2500:]}", flush=True)
        return 4
    finally:
        shutil.rmtree(work, ignore_errors=True)

    secs = round(time.time() - t0, 1)
    att = max(st["n_attempts"], 1)
    rep = dict(cfg)
    rep.update({
        "n_converged": st["n_converged"], "n_attempts": st["n_attempts"],
        "n_lowv_quarantined": st["n_lowv_quarantined"],
        "n_divergent_kept": st["n_divergent_kept"], "n_total": st["n_total"],
        "n_fix_solves": st.get("n_fix_solves", 0),
        "n_fix_reverts": st.get("n_fix_reverts", 0),
        "n_modes_vfixed": st.get("n_modes_vfixed", 0),
        "solves_per_attempt": round((att + st.get("n_fix_solves", 0)) / att, 2),
        "yield_converged_per_attempt": round(st["n_converged"] / att, 4),
        "lowv_rate": round(st["n_lowv_quarantined"] / att, 4),
        "divergent_rate": round(1 - (st["n_converged"] + st["n_lowv_quarantined"]) / att, 4),
        "hit_cap": st["n_converged"] < a.n_modes,
        "wall_secs": secs, "secs_per_attempt": round(secs / att, 2),
        "lib_mb": _dir_mb(lib),
        "npz_mb": round(sum(f.stat().st_size for f in lib.glob("*.npz")) / 1024 / 1024, 2),
        "cases_mb": _dir_mb(lib / "cases") if (lib / "cases").exists() else 0.0,
    })
    (lib / "gen_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[mp] {a.fid}: conv={rep['n_converged']}/{rep['n_attempts']} "
          f"yield={rep['yield_converged_per_attempt']} lowv={rep['n_lowv_quarantined']} "
          f"发散率={rep['divergent_rate']} 封顶={rep['hit_cap']} "
          f"修正解算={rep['n_fix_solves']}(回滚 {rep['n_fix_reverts']},涉 {rep['n_modes_vfixed']} 方式,"
          f"{rep['solves_per_attempt']} 解/抽) "
          f"wall={secs}s ({rep['secs_per_attempt']}s/次) 体积={rep['lib_mb']}MB "
          f"(npz {rep['npz_mb']}MB + cases {rep['cases_mb']}MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
