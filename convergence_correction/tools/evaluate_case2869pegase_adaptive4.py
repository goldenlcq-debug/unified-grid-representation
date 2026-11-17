"""Frozen-model Pegase evaluation with a four-case, per-parent skip rule.

Each parent has 100 ordered negatives. Test the first four; when at most one
converges, record the other 96 as skipped without invoking PSASP. The earlier
full-protocol attempt is an immutable bootstrap source, not this run's result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from rl_poweradj.foundation import UnifiedFrozenEncoder
from rl_poweradj.graph_dqn import GraphDQN
from rl_poweradj.graph_environment import GraphPowerAdjustmentEnv
from rl_poweradj.graph_evaluation import state_hash
from rl_poweradj.multifamily_external import audit_external_manifest
from rl_poweradj.multifamily_sampler import TRAIN_FAMILIES
from rl_poweradj.psasp import PSASPRunner, solver_processes_present
from rl_poweradj.run_store import atomic_json, file_hash


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/datasets/external-case2869pegase-20260926/manifest.json"
PUBLICATION = ROOT / "artifacts/runs/multifamily-joint-v1-seed0-20260923/selected.json"
SOLVER = ROOT / "PSASP/PSASP/WMLFRTMsg.exe"
SOURCE_SNAPSHOT = (ROOT / "artifacts/pauses/case2869pegase-strategy-change-20260927"
                   / "original-full-protocol-at-6.json")
SOLVER_AUDIT = ROOT / "artifacts/solver-audit/case2869pegase-adaptive4-20260927"
EXPECTED_MANIFEST_SHA256 = "24d7be3e2325564647acb36e13290820e6dc89cdc5a165832123b6c6a9bebae9"
PARENTS = 5
PER_PARENT = 100
TOTAL = PARENTS * PER_PARENT
RULE = {"evaluate_first": 4, "skip_if_successes_at_most": 1,
        "parent_case_count": PER_PARENT, "skipped_cases_are_not_failures": True}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audited_records() -> tuple[dict, list[dict], dict]:
    selected = read_json(PUBLICATION)
    require(selected.get("purpose") == "formal_multifamily_training"
            and selected.get("protocol") == "multi_family_joint_training_v1"
            and selected.get("selection") == "family_macro_validation_only"
            and tuple(selected.get("training_family_ids", ())) == TRAIN_FAMILIES,
            "Wrong selected model")
    model = Path(selected["model"]).resolve()
    require(model.is_relative_to(PUBLICATION.parent / "models")
            and file_hash(model) == selected["model_sha256"]
            and file_hash(SOLVER) == selected["solver_sha256"],
            "Model or solver identity changed")
    require(file_hash(MANIFEST) == EXPECTED_MANIFEST_SHA256,
            "Pegase manifest file changed")
    audit = audit_external_manifest(
        MANIFEST, training_families=TRAIN_FAMILIES,
        training_bus_counts=selected["training_bus_counts"])
    records = audit["cases"]
    require(len(records) == TOTAL, "Expected 500 Pegase cases")
    for start in range(0, TOTAL, PER_PARENT):
        group = records[start:start + PER_PARENT]
        require(len(group) == PER_PARENT
                and len({row["parent_case_id"] for row in group}) == 1
                and all(row["grid_family_id"] == "grid_2869_pegase_never_trained"
                        and row["bus_count"] == 2869 for row in group),
                f"Invalid Pegase parent group at {start}")
    return selected, records, audit


def protocol(selected: dict, audit: dict) -> dict:
    return {"phase": "external_test_adaptive_parent_skip_pegase_v1",
            "publication_sha256": file_hash(PUBLICATION),
            "model_sha256": selected["model_sha256"],
            "solver_sha256": selected["solver_sha256"],
            "manifest_digest": audit["digest"],
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "source_full_protocol_snapshot_sha256": file_hash(SOURCE_SNAPSHOT),
            "device": "cuda", "horizon": selected["recipe"]["horizon"],
            "zero_updates": True, "rule": RULE}


def tested(row: dict, parent: str, origin: str) -> dict:
    require(row["initial_converged"] is False,
            f"Initially converged: {row['case_id']}")
    return {**row, "parent_case_id": parent, "disposition": "tested", "origin": origin}


def skipped(case_id: str, parent: str, prior_executed: bool) -> dict:
    return {"case_id": case_id, "parent_case_id": parent,
            "disposition": "skipped", "skip_reason": "first4_successes_at_most_1",
            "physically_tested_in_prior_protocol": prior_executed}


def validate_prefix(entries: list[dict], records: list[dict]) -> None:
    require(len(entries) <= TOTAL, "Adaptive result exceeds manifest")
    require([row["case_id"] for row in entries]
            == [row["case_id"] for row in records[:len(entries)]],
            "Adaptive result is not a manifest prefix")
    for start in range(0, len(entries), PER_PARENT):
        group = entries[start:start + PER_PARENT]
        parent = records[start]["parent_case_id"]
        require(all(row["parent_case_id"] == parent for row in group),
                "Parent identity changed")
        if len(group) < 4:
            require(all(row["disposition"] == "tested" for row in group),
                    "Skip before four evaluated cases")
            continue
        require(all(row["disposition"] == "tested" for row in group[:4]),
                "First four cases must be tested")
        first_successes = sum(row["converged"] for row in group[:4])
        expected = "skipped" if first_successes <= 1 else "tested"
        require(all(row["disposition"] == expected for row in group[4:]),
                "Saved skip decision differs from first four outcomes")


def initial_prefix(records: list[dict]) -> list[dict]:
    source = read_json(SOURCE_SNAPSHOT)
    old = source.get("cases", [])
    require(source.get("status") == "paused" and len(old) == 6,
            "Wrong original Pegase pause snapshot")
    require([row["case_id"] for row in old]
            == [row["case_id"] for row in records[:len(old)]],
            "Original Pegase result is not a manifest prefix")
    entries: list[dict] = []
    for start in range(0, TOTAL, PER_PARENT):
        previous = old[start:min(start + PER_PARENT, len(old))]
        if not previous:
            break
        parent = records[start]["parent_case_id"]
        if len(previous) < 4:
            entries.extend(tested(row, parent, "reused_prior_full_protocol")
                           for row in previous)
            break
        first_successes = sum(row["converged"] for row in previous[:4])
        if first_successes <= 1:
            entries.extend(tested(row, parent, "reused_prior_full_protocol")
                           for row in previous[:4])
            entries.extend(skipped(records[index]["case_id"], parent, index < len(old))
                           for index in range(start + 4, start + PER_PARENT))
        else:
            entries.extend(tested(row, parent, "reused_prior_full_protocol")
                           for row in previous)
            if len(previous) < PER_PARENT:
                break
    validate_prefix(entries, records)
    return entries


def prepare(output: Path) -> None:
    require(not output.exists(), "Adaptive Pegase output already exists")
    selected, records, audit = audited_records()
    entries = initial_prefix(records)
    payload = {"status": "prepared", "protocol": protocol(selected, audit),
               "cases": entries}
    atomic_json(output, payload)
    print(json.dumps({"status": "prepared", "processed": len(entries),
                      "tested": sum(row["disposition"] == "tested" for row in entries),
                      "skipped": sum(row["disposition"] == "skipped" for row in entries)},
                     ensure_ascii=False))


def evaluate(output: Path, pause_request: Path | None) -> None:
    selected, records, audit = audited_records()
    require(output.is_file(), "Run prepare before evaluate")
    payload = read_json(output)
    require(payload["protocol"] == protocol(selected, audit),
            "Adaptive Pegase protocol changed")
    require(payload["status"] in ("prepared", "paused", "running", "failed"),
            "Adaptive Pegase run is already complete or has wrong status")
    entries = payload["cases"]
    validate_prefix(entries, records)
    require(not solver_processes_present(), "Another PSASP process is active")
    encoder = UnifiedFrozenEncoder.from_project(ROOT, "cuda")
    agent, _, _, _ = GraphDQN.load(selected["model"], encoder, device="cuda")
    agent.online.eval()
    agent.target.eval()
    before, updates = state_hash(agent.online), agent.updates
    runner = PSASPRunner(SOLVER, audit_dir=SOLVER_AUDIT)
    payload.pop("error_type", None)
    payload.pop("error", None)
    payload["status"] = "running"
    atomic_json(output, payload)
    try:
        with GraphPowerAdjustmentEnv(encoder, runner, payload["protocol"]["horizon"]) as env:
            while len(entries) < TOTAL:
                if pause_request is not None and pause_request.exists():
                    payload["status"] = "paused"
                    atomic_json(output, payload)
                    return
                index = len(entries)
                start = (index // PER_PARENT) * PER_PARENT
                parent = records[start]["parent_case_id"]
                if index - start >= 4:
                    first_successes = sum(row["converged"] for row in entries[start:start + 4])
                    if first_successes <= 1:
                        entries.append(skipped(records[index]["case_id"], parent, False))
                        atomic_json(output, payload)
                        continue
                record = records[index]
                state, info = env.reset(record["source"], record["policy"],
                                        expected_inputs=record.get("input_cards"))
                initial = info["converged"]
                require(not initial, f"Unexpected converged negative: {record['case_id']}")
                while not env.done:
                    action = agent.choose(state)
                    if action is None:
                        break
                    state, _, _, _, info = env.step(action)
                row = {"case_id": record["case_id"], "family": record["grid_family_id"],
                       "initial_converged": initial, **info}
                entries.append(tested(row, parent, "adaptive_new_evaluation"))
                atomic_json(output, payload)
        require(state_hash(agent.online) == before and agent.updates == updates,
                "Evaluation changed task network")
        encoder.verify_frozen()
        validate_prefix(entries, records)
        require(len(entries) == TOTAL, "Adaptive Pegase result has wrong coverage")
        groups = {}
        for start in range(0, TOTAL, PER_PARENT):
            group = entries[start:start + PER_PARENT]
            parent = records[start]["parent_case_id"]
            tested_rows = [row for row in group if row["disposition"] == "tested"]
            skipped_rows = [row for row in group if row["disposition"] == "skipped"]
            first_successes = sum(row["converged"] for row in group[:4])
            require((len(tested_rows), len(skipped_rows)) in ((4, 96), (100, 0)),
                    f"Wrong Pegase parent coverage: {parent}")
            require(bool(skipped_rows) == (first_successes <= 1),
                    f"Wrong Pegase skip decision: {parent}")
            successes = sum(row["converged"] for row in tested_rows)
            groups[parent] = {"first4_successes": first_successes,
                              "skip_decision": bool(skipped_rows),
                              "tested": len(tested_rows), "skipped": len(skipped_rows),
                              "successes": successes,
                              "nonconverged": len(tested_rows) - successes,
                              "prior_protocol_excluded": sum(
                                  row.get("physically_tested_in_prior_protocol", False)
                                  for row in skipped_rows)}
        tested_rows = [row for row in entries if row["disposition"] == "tested"]
        successes = sum(row["converged"] for row in tested_rows)
        payload.update(status="complete", processed_cases=TOTAL,
                       tested_cases=len(tested_rows),
                       skipped_cases=TOTAL - len(tested_rows),
                       successes=successes,
                       nonconverged=len(tested_rows) - successes,
                       success_rate_among_tested=successes / len(tested_rows),
                       groups=groups,
                       failure_reasons=dict(Counter(
                           row["reason"] for row in tested_rows if not row["converged"])),
                       task_parameters_unchanged=True, encoder_frozen=True)
    except BaseException as exc:
        payload.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        atomic_json(output, payload)
        raise
    atomic_json(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pause-request", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == "prepare":
        prepare(output)
    else:
        evaluate(output, args.pause_request.resolve() if args.pause_request else None)


if __name__ == "__main__":
    main()
