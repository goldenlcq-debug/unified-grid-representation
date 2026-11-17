"""运行 A4 冻结受控扰动电池。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridfm.data.repair_battery import default_battery_spec
from gridfm.data.repair_perturb import run_controlled_battery
from gridfm.solver.psasp import PsaspSolver


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("runs/pf_repair_controlled_v1"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--psasp-bin", default=os.environ.get("GRIDFM_PSASP_BIN", ""))
    args = parser.parse_args()
    if not args.psasp_bin:
        raise SystemExit("请通过 --psasp-bin 或 GRIDFM_PSASP_BIN 指定 PSASP 目录")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit 不得为负")
    summary = run_controlled_battery(
        default_battery_spec(), args.out,
        Path(__file__).resolve().parents[1],
        PsaspSolver(args.psasp_bin),
        resume=not args.no_resume,
        limit=args.limit,
        case_ids=set(args.case_id) if args.case_id else None)
    print(summary)
    return 0 if (
        summary["counts"]["bad_input_data"] == 0
        and summary["counts"]["engine_error"] == 0
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
