"""方式库生成入口(方案 T3.1)。用真实 PSASP 求解器批量产收敛方式样本。

  GRIDFM_PSASP_BIN=... python scripts/gen_data.py --base <case_dir> --n 100 --out <lib_dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gridfm.data.sampler import generate_library  # noqa: E402
from gridfm.solver.psasp import PsaspSolver  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基准算例目录(含全套 51 卡)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True, help="方式库输出目录")
    ap.add_argument("--work", default=None, help="临时工作目录")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", default="ieee118")
    ap.add_argument("--keep-cases", action="store_true",
                    help="留存每方式已解算例目录到 out/cases/<sid>/(T4.2 判稳需要)")
    ap.add_argument("--v-floor", default="0.85",
                    help="电压带验收阈值;'none' = 关闭(复现 07-11 之前生产的库必须关,"
                         "否则低压样本被封存改变 RNG 序列,复现失真)")
    ap.add_argument("--id-start", type=int, default=0,
                    help="sample_id 号段起点(外协扩产护栏:按任务书指派,如包 C=20000;"
                         "默认 0 = 主线段,命名与旧行为逐位一致,不消耗 RNG)")
    ap.add_argument("--diverse", action="store_true",
                    help="多模态方式采样(裁决④ 2026-07-14:离散负荷档 × 随机开机组合/"
                         "Dirichlet 参与度;G1 平均化失败演示的数据前提)")
    a = ap.parse_args()
    v_floor = None if str(a.v_floor).lower() == "none" else float(a.v_floor)

    work = Path(a.work) if a.work else Path(a.out).parent / "_gen_work"
    stats = generate_library(
        base_case_dir=Path(a.base), solver=PsaspSolver(), n=a.n,
        out_dir=Path(a.out), work_dir=work, seed=a.seed, family_id=a.family,
        keep_cases=a.keep_cases, v_floor=v_floor, id_start=a.id_start,
        diverse=a.diverse,
    )
    print(f"[gen] {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
