"""Compare real CUDA/PSASP continuous and episode-boundary resumed runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_poweradj.foundation import UnifiedFrozenEncoder
from rl_poweradj.graph_dqn import GraphDQN
from rl_poweradj.graph_evaluation import state_hash
from rl_poweradj.run_store import atomic_json, file_hash

CONTINUOUS = ROOT / "artifacts/runs/multifamily-lifecycle-acceptance-20260923"
RESUMED = ROOT / "artifacts/runs/multifamily-lifecycle-pause-parity-20260923"
OUTPUT = ROOT / "artifacts/results/multifamily-resume-parity-20260923.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", type=Path, default=CONTINUOUS)
    parser.add_argument("--resumed", type=Path, default=RESUMED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    continuous, resumed, output = args.continuous.resolve(), args.resumed.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    encoder = UnifiedFrozenEncoder.from_project(ROOT, "cpu")
    summaries = [load(run / "summary.json") for run in (continuous, resumed)]
    selected = [load(run / "selected.json") for run in (continuous, resumed)]
    models = []
    targets = []
    for item in selected:
        model, _, _, _ = GraphDQN.load(item["model"], encoder, device="cpu")
        models.append(state_hash(model.online))
        targets.append(state_hash(model.target))
    progress_keys = ("episodes", "transitions", "updates", "successes", "family_successes", "lengths",
                     "losses", "batch_sizes", "validation_history", "best_rank", "sampler")
    mismatches = [key for key in progress_keys
                  if summaries[0]["progress"][key] != summaries[1]["progress"][key]]
    if models[0] != models[1] or targets[0] != targets[1] or mismatches:
        raise RuntimeError(f"CUDA/PSASP resume parity differs: online={models}, target={targets}, "
                           f"progress={mismatches}")
    encoder.verify_frozen()
    payload = {"status": "passed", "continuous_model_sha256": file_hash(selected[0]["model"]),
               "resumed_model_sha256": file_hash(selected[1]["model"]),
               "online_state_hash": models[0], "target_state_hash": targets[0],
               "episodes": summaries[0]["progress"]["episodes"],
               "transitions": summaries[0]["progress"]["transitions"],
               "updates": summaries[0]["progress"]["updates"],
               "compared_progress_fields": progress_keys, "encoder_frozen": True}
    atomic_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
