"""Read-only frozen-token and physical-map smoke test across four grid sizes."""
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
from rl_poweradj.run_store import atomic_json

CASES = {
    "grid_2000": ROOT / "lfsamples/lfsample-bad/sample_1_1",
    "grid_3120": ROOT / "跨网训练数据/case3120sp/g011/negatives/neg_001/input",
    "grid_300": ROOT / "跨网训练数据/case300/g001/negatives/neg_001/input",
    "grid_118": ROOT / "跨网训练数据/case118/g001/negatives/neg_001/input",
}
OUTPUT = ROOT / "artifacts/results/multifamily-graph-interface-20260923.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    encoder = UnifiedFrozenEncoder.from_project(ROOT, "cpu")
    rows = []
    for family, case in CASES.items():
        before = input_card_hashes(case)
        adapter = CaseAdapter(case)
        proposal = adapter.proposal()
        proposal["approved"] = True  # This smoke test executes no action.
        candidates = adapter.authorize(proposal)
        graph = encoder.encode_graph(case)
        expected = np.array([[abs(int(row["I_No"])) - 1, abs(int(row["J_No"])) - 1]
                             for row in adapter.rows], dtype=np.int64).reshape(-1, 2).T
        if not np.array_equal(graph.edge_index, expected):
            raise ValueError(f"{family}: frozen/physical edge mapping differs")
        if len(graph.nodes) != adapter.n or len(graph.edges) != len(adapter.rows):
            raise ValueError(f"{family}: graph shape differs from PSASP cards")
        if input_card_hashes(case) != before:
            raise RuntimeError(f"{family}: original inputs changed during inspection")
        rows.append({"family": family, "case": str(case), "nodes": adapter.n,
                     "physical_edges": len(adapter.rows), "authorized_devices": len(adapter.devices),
                     "candidate_targets": len(candidates), "frozen_graph_key": graph.key,
                     "inputs_unchanged": True})
        print(f"verified {family}: N={adapter.n}, M={len(adapter.rows)}, A={len(candidates)}", flush=True)
    encoder.verify_frozen()
    atomic_json(OUTPUT, {"status": "passed", "solver_invoked": False,
                         "encoder_frozen": True, "cases": rows})
    print(json.dumps({"output": str(OUTPUT), "cases": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
