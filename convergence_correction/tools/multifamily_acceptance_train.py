"""Bounded real-PSASP/CUDA acceptance of the four-family training path."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from rl_poweradj.foundation import UnifiedFrozenEncoder
from rl_poweradj.graph_dqn import GraphDQN, GraphTransition
from rl_poweradj.graph_environment import GraphPowerAdjustmentEnv
from rl_poweradj.graph_evaluation import state_hash
from rl_poweradj.lifecycle import run_lock
from rl_poweradj.multifamily_manifest import audit_multifamily_manifest
from rl_poweradj.multifamily_replay import MultiFamilyReplay, sample_balanced
from rl_poweradj.multifamily_sampler import TRAIN_FAMILIES
from rl_poweradj.psasp import PSASPRunner, solver_processes_present
from rl_poweradj.readiness import code_manifest, runtime_identity
from rl_poweradj.run_store import atomic_json, file_hash, prepare_run

MANIFEST = ROOT / "artifacts/datasets/multifamily-joint-v1/train-validation-manifest.json"
SOLVER = ROOT / "PSASP/PSASP/WMLFRTMsg.exe"
RUN = ROOT / "artifacts/runs/multifamily-acceptance-20260923"
HORIZON = 5


def _run() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this four-family acceptance run")
    if solver_processes_present():
        raise RuntimeError("Another PSASP solver is active")
    audited = audit_multifamily_manifest(MANIFEST, verify_files=False)
    records = audited["cases"]
    chosen = {family: next(row for row in records
                           if row["grid_family_id"] == family and row["split"] == "train")
              for family in TRAIN_FAMILIES}
    contract = {"purpose": "multifamily_acceptance_only", "protocol": audited["split_protocol"],
                "manifest_sha256": file_hash(MANIFEST), "solver_sha256": file_hash(SOLVER),
                "families": list(TRAIN_FAMILIES), "horizon": HORIZON,
                "code": code_manifest(ROOT), "runtime": runtime_identity()}
    prepare_run(RUN, contract)
    encoder = UnifiedFrozenEncoder.from_project(ROOT, "cuda")
    agent = GraphDQN(encoder, device="cuda", width=128, layers=2, seed=20260923)
    agent.training_cases = {(row["grid_family_id"], row["case_id"])
                            for row in records if row["split"] == "train"}
    before = state_hash(agent.online)
    replay = MultiFamilyReplay(64)
    runner = PSASPRunner(SOLVER, audit_dir=RUN / "solver")
    outcomes = []
    transitions = 0
    with GraphPowerAdjustmentEnv(encoder, runner, HORIZON) as env:
        for family in TRAIN_FAMILIES:
            record = chosen[family]
            state, info = env.reset(record["source"], record["policy"],
                                    expected_inputs=record["input_cards"])
            while not env.done:
                action = agent.choose(state, epsilon=0.5)
                if action is None:
                    raise RuntimeError("Nonterminal environment has no legal action")
                next_state, reward, terminated, truncated, info = env.step(action)
                replay.add(GraphTransition(state, action, reward, next_state,
                                           terminated or truncated, family, record["case_id"]))
                transitions += 1
                if len(replay.items) >= 4:
                    batch = sample_balanced(replay, agent.rng, 4, node_budget=100000,
                                            edge_budget=200000, action_budget=100000)
                    if not batch or agent.learn(batch) is None:
                        raise RuntimeError("Four-family acceptance yielded an empty learning batch")
                state = next_state
            outcomes.append({"family": family, "case_id": record["case_id"],
                             "converged": info["converged"], "steps": info["steps"],
                             "reason": info["reason"]})
            print(f"{family}: steps={info['steps']}, reason={info['reason']}", flush=True)
    if transitions < 4 or agent.updates < 1 or state_hash(agent.online) == before:
        raise RuntimeError("Task head did not train in acceptance")
    encoder.verify_frozen()
    progress = {"episodes": len(outcomes), "transitions": transitions,
                "updates": agent.updates, "outcomes": outcomes}
    checkpoint = RUN / "checkpoints/acceptance.pt"
    agent.save(checkpoint, replay, progress, contract)
    restored, saved_replay, saved_progress, _ = GraphDQN.load(
        checkpoint, encoder, device="cuda", expected_run_contract=contract)
    if (state_hash(restored.online) != state_hash(agent.online)
            or len(saved_replay.items) != len(replay.items) or saved_progress != progress):
        raise RuntimeError("Four-family checkpoint round-trip differs")
    atomic_json(RUN / "summary.json", {
        "status": "passed", "purpose": "acceptance_only", "formal_training_started": False,
        "families": list(TRAIN_FAMILIES), "outcomes": outcomes,
        "transitions": transitions, "updates": agent.updates,
        "task_parameters_changed": True, "encoder_frozen": True,
        "checkpoint": str(checkpoint), "solver_sha256": runner.executable_sha256,
    })
    print(f"acceptance passed: transitions={transitions}, updates={agent.updates}", flush=True)


def main() -> None:
    with run_lock(RUN):
        _run()


if __name__ == "__main__":
    main()
