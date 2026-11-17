"""A3 规则潮流修复统一评测入口。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gridfm.data.repair import DEFAULT_REPAIR_CONFIG
from gridfm.data.repair_eval import (
    CaseSpec,
    discover_library_cases,
    evaluate_cases,
)
from gridfm.solver.psasp import PsaspSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(
        os.environ.get("GRIDFM_DATA_ROOT", "../gridfm-data")))
    parser.add_argument("--libraries", nargs="+", default=["lib_cepri36", "lib118"])
    parser.add_argument("--suite", default="a_v1")
    parser.add_argument("--out", type=Path, default=Path("runs/pf_repair_a_v1"))
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--family", default="single")
    parser.add_argument("--protected-bus", type=int, action="append", default=[])
    parser.add_argument("--include-unsolvable", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--psasp-bin", default=os.environ.get("GRIDFM_PSASP_BIN", ""))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.psasp_bin:
        raise SystemExit("请通过 --psasp-bin 或 GRIDFM_PSASP_BIN 指定 PSASP 目录")
    if args.case_dir:
        sample_id = args.sample_id or args.case_dir.name
        cases = [CaseSpec(
            sample_id, args.case_dir, args.family, tuple(args.protected_bus))]
    else:
        if args.sample_id:
            raise SystemExit("--sample-id 仅能与 --case-dir 同时使用")
        cases = discover_library_cases(
            args.root, args.libraries,
            solvable_only=not args.include_unsolvable)
    if args.limit is not None:
        if args.limit < 0:
            raise SystemExit("--limit 不得为负")
        cases = cases[:args.limit]
    solver = PsaspSolver(args.psasp_bin)
    summary = evaluate_cases(
        cases, args.out, solver, DEFAULT_REPAIR_CONFIG,
        resume=not args.no_resume, suite=args.suite)
    print(f"suite={args.suite} total={summary['total']} counts={summary['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
