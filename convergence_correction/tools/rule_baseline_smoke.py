"""Small PSASP baseline: greedily energize legal off-SH candidates by key order."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_poweradj.case_adapter import CaseAdapter
from rl_poweradj.data_preflight import input_card_hashes
from rl_poweradj.foundation import UnifiedFrozenEncoder
from rl_poweradj.graph_environment import GraphPowerAdjustmentEnv
from rl_poweradj.psasp import PSASPRunner, solver_processes_present
from rl_poweradj.run_store import atomic_json

SOLVER = ROOT / "PSASP/PSASP/WMLFRTMsg.exe"
OUTPUT = ROOT / "artifacts/results/multifamily-rule-smoke-20260923.json"
CASES = {
    "grid_2000": ROOT / "lfsamples/lfsample-bad/sample_1_1",
    "grid_3120": ROOT / "跨网训练数据/case3120sp/g011/negatives/neg_001/input",
    "grid_300": ROOT / "跨网训练数据/case300/g001/negatives/neg_001/input",
    "grid_118": ROOT / "跨网训练数据/case118/g001/negatives/neg_001/input",
}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if solver_processes_present():
        raise RuntimeError("Another PSASP solver is active")
    encoder = UnifiedFrozenEncoder.from_project(ROOT, "cpu")
    runner = PSASPRunner(SOLVER, audit_dir=ROOT / "artifacts/solver-audit/multifamily-rule-smoke-20260923")
    rows = []
    for family, source in CASES.items():
        before = input_card_hashes(source)
        policy = CaseAdapter(source).proposal()
        policy["approved"] = True
        with GraphPowerAdjustmentEnv(encoder, runner, 150) as env:
            state, info = env.reset(source, policy, expected_inputs=before)
            actions = []
            while not env.done:
                valid = np.flatnonzero(state.mask & (state.targets == 1)
                                       & (state.edges[state.candidate_edges, 0] == 0))
                if not len(valid):
                    break
                action = int(valid[0])
                chosen = (state.device_keys[action], int(state.targets[action]))
                state, _, _, _, info = env.step(action)
                actions.append(chosen)
            row = {"family": family, "case": str(source), "initial_converged": not actions and info["converged"],
                   "converged": info["converged"], "reason": info["reason"],
                   "steps": info["steps"], "actions": actions}
        if input_card_hashes(source) != before:
            raise RuntimeError(f"Original inputs changed: {source}")
        rows.append(row)
        print(f"{family}: converged={row['converged']}, steps={row['steps']}", flush=True)
    encoder.verify_frozen()
    atomic_json(OUTPUT, {"status": "complete", "strategy": "first_legal_off_shunt_to_on_by_case_key",
                         "looked_at_sample_perturbation_list": False, "model_updates": 0,
                         "solver_sha256": runner.executable_sha256, "cases": rows})
    print(json.dumps({"output": str(OUTPUT), "cases": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
