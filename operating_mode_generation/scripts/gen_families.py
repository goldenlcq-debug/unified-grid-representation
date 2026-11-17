"""多家族合成库批产(方案 7.2 / 7.5 / 8.4 / 8.9;交接清单第 ③ 项)。

规模档设计**严格按方案 8.4**:迁移门禁的分子 = 合成大网(300~1000 节点)**冻结测试集**,
分母 = 同分布留出集(**训练规模档**、留出家族)。故训练规模档必须落在 300 以下,
否则 300~1000 不是"留出规模档",跨规模泛化(D3 核心主张)就无从检验。

    train    : 合成 150~280 节点 × N_TRAIN 家族 × 100 方式   (+ 已有 lib118 真实家族)
    val      : 合成 150~280 节点 × N_VAL   家族 × 50 方式    (同分布留出家族,8.4 分母)
    transfer : 合成 400/700/1000 节点 × N_XFER 家族 × 50 方式(留出规模档,8.4 分子,**冻结**)

家族 = 一个合成拓扑(D14:train/val/test 无共同家族)。冻结测试集算内容哈希入报告(8.9)。

并行(D16 探索级):**家族级** ProcessPoolExecutor;单个家族内部串行(7.0 纪律 7)。
实测本机 PSASP 允许并发潮流(6 进程结果与串行逐位一致,吞吐 ~3.9×)。

  python scripts/gen_families.py --out E:/AI-2/统一计算大模型/gridfm-data --workers 6
  python scripts/gen_families.py --out ... --resume          # 跳过已完成家族
  python scripts/gen_families.py --out ... --dry-run         # 只算账不生产

**G4 修正配方(2026-07-18 呈批批准 + 首批实测回退,单测锁定)**:
  - diverse=True 仅 train/val:交接 §2-2 规格"每家族多方式(lf 多档)"(裁决④采样器,
    lf 三档 + 开机组合多样 = 启停两阶段 6.3② 的数据前提);**transfer 档 diverse=False**
    (首批 1000/700 实测 0/50 全发散证伪外推;回退 §8.4 原语义 sample_mode 纯收敛库,
    详见 _family_params docstring 与变更记录);
  - amp:train/val=1.0(贴 v1 编码器预训练分布);transfer 按 07-11"大网按规模收紧"
    裁决 + transfer2k(0.5@2000)先例:400→0.7,700/1000→0.5;
  - keep_cases:val/transfer 开(评估协议"双底版"天花板臂;train 省盘不开,
    [内] 试算臂沿用 lib118_diverse 保持与 v0.2 直接可比);
  - 家族基态算例全 split 留档 `<split>/<fid>/_base_case/`(诚实底版 + 后续资产;
    "同种子重放≠忠实复现"教训——生产管线演进后基态不可重建,删了就真丢);
  - --resume 完成判据 = manifest 内 solvable=1 行数 ≥ 目标方式数(仅"文件存在"会把
    FAIL 家族的表头 manifest 当完成,静默漏产)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
DONOR = REPO / "tests" / "fixtures" / "ieee118_golden"

# 训练规模档(< 300,方案 8.4)
TRAIN_SIZES = [150, 180, 210, 240, 270]
# 留出规模档(300~1000,方案 8.4 分子)
XFER_SIZES = [400, 700, 1000]

SYNTH_KW = dict(load_level=0.55, gen_frac=0.30, q_comp=0.25)   # 9/9 收敛配方(变更记录)


def _family_params(split: str, n_buses: int) -> dict:
    """G4 生产配置(dict 单一出处;纯函数,单测锁定)。

    **transfer 档 diverse=False(2026-07-18 实测证伪后回退)**:首版把"lf 多档"从
    训练档外推到 transfer,首批 1000/700 节点家族方式库 0/50 全发散(3n 尝试烧尽,
    无一进入低压封存 = 纯收敛崩溃)。回退到 §8.4 原语义 = sample_mode 纯收敛库
    (transfer2k 先例:amp=0.5@2000 裸收敛 96%+)。
    **train/val 档 = diverse-lite + 停机 0.15(2026-07-18 两次标定后用户批准)**:
    三臂控制变量@270 合成基态定凶——sample_mode 20/20、diverse-lite 停机0 20/20、
    停机0.15 12/20、停机0.30 ~3/30:**真凶 = 停机重分配,剂量依赖**;调度机制
    (dirichlet vs proportional)非主因,但 proportional(基态比例 × 逐机 ±15% 抖动
    → 归一档位目标)保留为可行性裕度。lf 三档 + 开机组合多样(0.15×~80 机 ≈
    每方式 12 台停机)两条规格轴保留;lf 档条件化评估由 val 留出家族承担。
    """
    if split == "transfer":
        return {"amp": 0.7 if n_buses <= 400 else 0.5, "keep_cases": True,
                "diverse": False, "dispatch": None, "decommit": None}
    # S2 自适应停机阶梯(2026-07-18 用户批准):0.15 起步,家族"体质"不足自动降档,
    # 全记录(样本级 manifest.decommit + 家族级 decommit_final/steps 入 build_report)。
    # 背景:固定 0.15 实测 270 档 1/13 达标(收得率 2.3%~60% 家族方差)。
    return {"amp": 1.0, "keep_cases": split == "val", "diverse": True,
            "dispatch": "proportional", "decommit": (0.15, 0.08, 0.0)}


def audit_families_disjoint(families_root: Path) -> dict:
    """D14 数据级审计:三 split 的 family_id 集合两两不相交(转正强制单测配套)。

    以磁盘 manifest.csv 为准(而非计划函数)——resume 合并/人工改动后仍受检。
    返回 {split: set(family_id)};有交集即抛 AssertionError。
    """
    import csv
    fams = {}
    for split in ("train", "val", "transfer"):
        s = set()
        d = Path(families_root) / split
        if d.exists():
            for mf in sorted(d.glob("*/manifest.csv")):
                with open(mf, encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        s.add(row["family_id"])
        fams[split] = s
    for a, b in (("train", "val"), ("train", "transfer"), ("val", "transfer")):
        inter = fams[a] & fams[b]
        assert not inter, f"D14 违规:{a}∩{b} 家族非空 {sorted(inter)[:5]}"
    return fams


def _plan(n_train: int, n_val: int, n_xfer_per_size: int, modes_train: int, modes_eval: int):
    """返回 [(split, family_id, n_buses, seed, n_modes, keep_div_frac)]。"""
    jobs = []
    for i in range(n_train):
        n = TRAIN_SIZES[i % len(TRAIN_SIZES)]
        jobs.append(("train", f"synth{n}_s{1000+i}", n, 1000 + i, modes_train, 0.08))
    for i in range(n_val):
        n = TRAIN_SIZES[i % len(TRAIN_SIZES)]
        jobs.append(("val", f"synth{n}_s{2000+i}", n, 2000 + i, modes_eval, 0.08))
    k = 0
    for n in XFER_SIZES:
        for i in range(n_xfer_per_size):
            # 冻结迁移测试集:纯收敛库(keep_div=0)——8.4 度量的是状态通道重构 MAE
            jobs.append(("transfer", f"synth{n}_s{3000+k}", n, 3000 + k, modes_eval, 0.0))
            k += 1
    # 大家族优先(最长作业优先调度):既缩短总 wall,又让高风险的 1000 节点档尽早暴露问题。
    # 实测:合成耗时随 N 与种子剧烈波动(N=1000 seed0 用 6.8s,seed2 用 63s)。
    jobs.sort(key=lambda j: -j[2])
    return jobs


def _one_family(args) -> dict:
    """一个 worker = 一个家族(合成 → 方式库),内部全串行。"""
    (split, fid, n_buses, seed, n_modes, keep_div, v_floor, attempt_mult,
     out_root, work_root) = args
    sys.path.insert(0, str(REPO))
    from gridfm.data.sampler import generate_library
    from gridfm.data.synth import synth_family
    from gridfm.fsutil import robust_copytree
    from gridfm.solver.psasp import LicenseError, PsaspSolver

    t0 = time.time()
    solver = PsaspSolver()
    case = Path(work_root) / f"case_{fid}"
    work = Path(work_root) / f"mode_{fid}"
    lib = Path(out_root) / split / fid
    p = _family_params(split, n_buses)
    try:
        info = synth_family(n_buses, DONOR, case, solver, seed=seed, **SYNTH_KW)
        if not info.get("converged"):
            return {"family_id": fid, "split": split, "ok": False,
                    "reason": f"合成不收敛(换了 {info['seed_tries']} 个种子)"}
        # G4:基态算例留档(诚实底版/后续资产;管线演进后不可重建,先于方式库落盘)
        robust_copytree(case, lib / "_base_case")
        st = generate_library(case, solver, n_modes, lib, work, seed=seed,
                              family_id=fid, keep_divergent_frac=keep_div, v_floor=v_floor,
                              max_attempts=attempt_mult * n_modes,
                              amp=p["amp"], keep_cases=p["keep_cases"],
                              diverse=p["diverse"],
                              diverse_dispatch=p["dispatch"] or "dirichlet",
                              diverse_decommit=(0.3 if p["decommit"] is None
                                                else p["decommit"]))
        ok = st["n_converged"] >= n_modes
        return {"family_id": fid, "split": split, "ok": ok, "n_buses": n_buses,
                "seed_used": info["seed_used"], "vmin": info["vmin"], "vmax": info["vmax"],
                "n_converged": st["n_converged"], "n_divergent_kept": st["n_divergent_kept"],
                "n_lowv": st["n_lowv_quarantined"], "amp": p["amp"],
                "decommit_final": st["decommit_final"],
                "decommit_steps": st["decommit_steps"],
                "n_attempts": st["n_attempts"], "secs": round(time.time() - t0, 1),
                "reason": "" if ok else f"方式库只凑到 {st['n_converged']}/{n_modes}"}
    except LicenseError as e:
        return {"family_id": fid, "split": split, "ok": False, "fatal": True,
                "reason": f"加密狗失效:{e}"}
    except Exception as e:  # noqa: BLE001
        return {"family_id": fid, "split": split, "ok": False,
                "reason": f"{type(e).__name__}: {e}"}
    finally:
        shutil.rmtree(case, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)


def _content_hash(data_root: Path) -> str:
    """冻结评估集内容哈希(8.9①)。

    **唯一实现** = `gridfm.eval.transfer.content_hash` + `dataset.family_files`(manifest 驱动)。
    此前这里另写了一套(relpath+bytes、裸 glob),与评估侧(name+bytes、manifest 驱动)算出不同值,
    "冻结哈希"有两个真相 —— 2026-07-09 实测撞上,已合并。
    """
    from gridfm.data.dataset import family_files
    from gridfm.eval.transfer import content_hash
    return content_hash(family_files(data_root, "transfer"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="数据根(GRIDFM_DATA_ROOT)")
    ap.add_argument("--work", default=None, help="临时工作根(默认 out/_work)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--n-xfer-per-size", type=int, default=4)
    ap.add_argument("--modes-train", type=int, default=100)
    ap.add_argument("--modes-eval", type=int, default=50)
    ap.add_argument("--resume", action="store_true", help="跳过已有 manifest 的家族")
    ap.add_argument("--attempt-mult", type=int, default=3,
                    help="每家族尝试上限 = mult×目标方式数(默认 3;收得率长尾家族"
                         "补产时调大——同种子尝试流确定,扩预算只是接着往后走,"
                         "不改已产样本;凑满即停,成功家族不受影响)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--v-floor", default="0.85",
                    help="'none' = 关闭电压带验收(复现 07-11 前生产的家族树必须关)")
    a = ap.parse_args()
    v_floor = None if str(a.v_floor).lower() == "none" else float(a.v_floor)

    out_root = Path(a.out) / "families"
    work_root = Path(a.work) if a.work else Path(a.out) / "_work"
    jobs = _plan(a.n_train, a.n_val, a.n_xfer_per_size, a.modes_train, a.modes_eval)

    n_samp = sum(j[4] for j in jobs)
    print(f"[plan] 家族 {len(jobs)}(train {a.n_train} / val {a.n_val} / "
          f"transfer {a.n_xfer_per_size * len(XFER_SIZES)});目标收敛样本 {n_samp}")
    print(f"[plan] 训练规模档 {TRAIN_SIZES};留出规模档 {XFER_SIZES}(方案 8.4)")
    print("[plan] G4 配方:train/val=diverse-lite(lf 三档,dispatch=proportional,"
          "停机自适应阶梯 0.15→0.08→0.0 [S2,查点每40尝试 keep<0.25 降档,全记录]);"
          "transfer=sample_mode 纯收敛(diverse 大网 0/50 实测回退);"
          "amp train/val=1.0,transfer 400=0.7/700+=0.5;"
          "keep_cases=val/transfer;基态算例留档 _base_case/")
    if a.dry_run:
        # 修正成本模型(2026-07-18 审查):单解耗时随规模增长(锚 118≈1.7s/解 → 1.2+0.006N);
        # 尝试/入库放大 = 1/收得率(diverse-lite 停机 0.15 标定 60%@270,小规模更好,
        # 取 1.55;transfer sample_mode 收紧档按 transfer2k 先例取 1.3)。
        est = 0.0
        for _split, _fid, n, _seed, modes, _kd in jobs:
            p = _family_params(_split, n)
            infl = 1.55 if p["diverse"] else 1.3
            est += 0.03 * n + modes * (1.2 + 0.006 * n) * infl
        print(f"[plan] 修正串行估计 {est/3600:.1f}h;{a.workers} 并行(实测 ~3.9×)"
              f"≈ {est/3.9/3600:.1f}h(如实区间 ±30%)")
        return 0

    if a.resume:
        import csv as _csv

        def _complete(j) -> bool:
            """完成 = manifest 内 solvable=1 行数 ≥ 目标方式数(FAIL 家族的表头
            manifest 不算完成;"文件存在"判据会把 0/50 家族静默当完成漏产)。"""
            mf = out_root / j[0] / j[1] / "manifest.csv"
            if not mf.exists():
                return False
            with open(mf, encoding="utf-8") as f:
                n_ok = sum(1 for r in _csv.DictReader(f) if r["solvable"] == "1")
            return n_ok >= j[4]

        jobs = [j for j in jobs if not _complete(j)]
        print(f"[resume] 剩余 {len(jobs)} 个家族")

    out_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    payload = [(s, f, n, sd, m, kd, v_floor, a.attempt_mult,
                str(out_root), str(work_root))
               for s, f, n, sd, m, kd in jobs]

    t0 = time.time()
    results, done = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_one_family, p): p[1] for p in payload}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            flag = "OK " if r["ok"] else "FAIL"
            print(f"[{done}/{len(payload)}] {flag} {r['family_id']:<18} "
                  f"{r.get('secs', 0):>6.1f}s  {r.get('reason', '')}", flush=True)
            if r.get("fatal"):
                print("[fatal] 加密狗失效 —— 熔断整批(7.0 纪律 8:继续=制造假结果)")
                for f in futs:
                    f.cancel()
                break

    wall = time.time() - t0
    ok = [r for r in results if r["ok"]]
    print(f"\n[done] {len(ok)}/{len(results)} 家族成功,wall {wall/60:.1f} min")

    # D14 数据级审计(转正:三 split 家族两两不相交,以磁盘 manifest 为准)
    fams = audit_families_disjoint(out_root)
    print(f"[audit] D14 家族不相交:train {len(fams['train'])} / "
          f"val {len(fams['val'])} / transfer {len(fams['transfer'])}")

    # 冻结迁移测试集内容哈希(8.9)
    from gridfm.data.sampler import DIVERSE_LF_LEVELS
    report = {"wall_min": round(wall / 60, 1), "families": results,
              "train_sizes": TRAIN_SIZES, "xfer_sizes": XFER_SIZES,
              "recipe": {"diverse": {"train": True, "val": True,
                                     "transfer": False},   # transfer 实测回退 sample_mode
                         "diverse_dispatch": "proportional",   # diverse-lite(2026-07-18)
                         # S2 自适应阶梯(查点每40尝试 keep<0.25 降档;逐家族终值见 families)
                         "diverse_decommit": [0.15, 0.08, 0.0],
                         "lf_levels_train_val": list(DIVERSE_LF_LEVELS),
                         "amp": {"train": 1.0, "val": 1.0,
                                 "transfer<=400": 0.7, "transfer>=700": 0.5},
                         "keep_cases": ["val", "transfer"], "v_floor": v_floor,
                         "base_case_kept": True},
              "d14_families": {k: sorted(v) for k, v in fams.items()}}
    if (out_root / "transfer").exists():
        from gridfm.data.dataset import family_files
        report["transfer_content_hash"] = _content_hash(Path(a.out))
        report["transfer_n_npz"] = len(family_files(Path(a.out), "transfer"))
        print(f"[frozen] 冻结迁移测试集 {report['transfer_n_npz']} 个样本  "
              f"内容哈希 {report['transfer_content_hash']}")
    (out_root / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] 报告 {out_root / 'build_report.json'}")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
