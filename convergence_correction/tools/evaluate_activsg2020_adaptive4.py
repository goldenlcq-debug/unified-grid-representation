"""Zero-update external evaluation with a four-case parent skip rule.

For each ten-case parent, test four cases. If at most one converges, mark the
remaining six as skipped (not failures). This protocol is distinct from the
previous full-600 attempt and keeps that attempt immutable.
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
from rl_poweradj.psasp import PSASPRunner, solver_processes_present
from rl_poweradj.run_store import atomic_json, file_hash


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/datasets/external-activsg2020-matched-v3-20260926/manifest.json"
PUBLICATION = ROOT / "artifacts/runs/multifamily-joint-v1-seed0-20260923/selected.json"
SOLVER = ROOT / "PSASP/PSASP/WMLFRTMsg.exe"
SOURCE_SNAPSHOT = (ROOT / "artifacts/pauses/activsg2020-matched-v3-strategy-change-20260926"
                   / "original-full-protocol-at-199.json")
SOLVER_AUDIT = ROOT / "artifacts/solver-audit/activsg2020-matched-v3-adaptive4-20260926"
RULE = {"evaluate_first": 4, "skip_if_successes_at_most": 1,
        "parent_case_count": 10, "skipped_cases_are_not_failures": True}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audited_records() -> tuple[dict, list[dict], dict]:
    selected = read_json(PUBLICATION)
    require(selected.get("purpose") == "formal_multifamily_training"
            and selected.get("selection") == "family_macro_validation_only",
            "Wrong selected model")
    require(file_hash(selected["model"]) == selected["model_sha256"]
            and file_hash(SOLVER) == selected["solver_sha256"],
            "Model or solver identity changed")
    audit = audit_external_manifest(
        MANIFEST, training_families=selected["training_family_ids"],
        training_bus_counts=selected["training_bus_counts"])
    records = audit["cases"]
    require(len(records) == 600, "Expected 600 manifest cases")
    for index in range(0, 600, 10):
        group = records[index:index + 10]
        require(len(group) == 10 and len({row["parent_case_id"] for row in group}) == 1,
                f"Parent cases are not ten contiguous records at {index}")
    return selected, records, audit


def protocol(selected: dict, audit: dict) -> dict:
    return {"phase": "external_test_adaptive_parent_skip_v1",
            "publication_sha256": file_hash(PUBLICATION),
            "model_sha256": selected["model_sha256"],
            "solver_sha256": selected["solver_sha256"],
            "manifest_digest": audit["digest"],
            "source_full_protocol_snapshot_sha256": file_hash(SOURCE_SNAPSHOT),
            "device": "cuda", "horizon": selected["recipe"]["horizon"],
            "zero_updates": True, "rule": RULE}


def tested(row: dict, parent: str, origin: str) -> dict:
    require(row["initial_converged"] is False, f"Initially converged: {row['case_id']}")
    return {**row, "parent_case_id": parent, "disposition": "tested", "origin": origin}


def skipped(case_id: str, parent: str, prior_executed: bool) -> dict:
    return {"case_id": case_id, "parent_case_id": parent,
            "disposition": "skipped", "skip_reason": "first4_successes_at_most_1",
            "physically_tested_in_prior_protocol": prior_executed}


def initial_prefix(records: list[dict]) -> list[dict]:
    source = read_json(SOURCE_SNAPSHOT)
    old = source.get("cases", [])
    require(source.get("status") == "paused" and len(old) == 199,
            "Wrong original full-protocol pause snapshot")
    require([item["case_id"] for item in old]
            == [item["case_id"] for item in records[:len(old)]],
            "Original full-protocol result is not a manifest prefix")
    entries: list[dict] = []
    for start in range(0, len(records), 10):
        parent_records = records[start:start + 10]
        previous = old[start:min(start + 10, len(old))]
        if not previous:
            break
        parent = parent_records[0]["parent_case_id"]
        if len(previous) < 4:
            entries.extend(tested(item, parent, "reused_prior_full_protocol") for item in previous)
            break
        first_four = previous[:4]
        first_successes = sum(item["converged"] for item in first_four)
        if first_successes <= 1:
            entries.extend(tested(item, parent, "reused_prior_full_protocol")
                           for item in first_four)
            entries.extend(skipped(item["case_id"], parent,
                                   (start + offset) < len(old))
                           for offset, item in enumerate(parent_records[4:], start=4))
        else:
            entries.extend(tested(item, parent, "reused_prior_full_protocol")
                           for item in previous)
            if len(previous) < 10:
                break
    require([item["case_id"] for item in entries]
            == [item["case_id"] for item in records[:len(entries)]],
            "Adaptive bootstrap is not a manifest prefix")
    return entries


def validate_prefix(entries: list[dict], records: list[dict]) -> None:
    require(len(entries) <= len(records), "Adaptive result exceeds manifest")
    require([item["case_id"] for item in entries]
            == [item["case_id"] for item in records[:len(entries)]],
            "Adaptive result is not a manifest prefix")
    for start in range(0, len(entries), 10):
        group = entries[start:start + 10]
        if len(group) < 4:
            require(all(item["disposition"] == "tested" for item in group),
                    "Skip before four evaluated cases")
            continue
        require(all(item["disposition"] == "tested" for item in group[:4]),
                "First four cases must be tested")
        first_successes = sum(item["converged"] for item in group[:4])
        expected = "skipped" if first_successes <= 1 else "tested"
        require(all(item["disposition"] == expected for item in group[4:]),
                "Saved skip decision differs from first four outcomes")


def prepare(output: Path) -> None:
    require(not output.exists(), "Adaptive output already exists")
    selected, records, audit = audited_records()
    entries = initial_prefix(records)
    validate_prefix(entries, records)
    payload = {"status": "prepared", "protocol": protocol(selected, audit),
               "cases": entries}
    atomic_json(output, payload)
    print(json.dumps({"status": "prepared", "processed": len(entries),
                      "tested": sum(row["disposition"] == "tested" for row in entries),
                      "skipped": sum(row["disposition"] == "skipped" for row in entries)},
                     ensure_ascii=False))


def evaluate(output: Path, pause_request: Path | None,
             stop_after: int | None = None) -> None:
    require(stop_after is None or (10 <= stop_after <= 600 and stop_after % 10 == 0),
            "Stop boundary must be a whole parent group within the manifest")
    selected, records, audit = audited_records()
    require(output.is_file(), "Run prepare before evaluate")
    payload = read_json(output)
    require(payload["protocol"] == protocol(selected, audit), "Adaptive protocol changed")
    require(payload["status"] in ("prepared", "paused", "running", "failed"),
            "Adaptive run is already complete or has wrong status")
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
            while len(entries) < len(records):
                if stop_after is not None and len(entries) >= stop_after:
                    require(len(entries) == stop_after, "Adaptive stop boundary overshot")
                    require(state_hash(agent.online) == before and agent.updates == updates,
                            "Evaluation changed task network")
                    encoder.verify_frozen()
                    validate_prefix(entries, records)
                    tested_rows = [item for item in entries
                                   if item["disposition"] == "tested"]
                    successes = sum(item["converged"] for item in tested_rows)
                    payload.update(status="paused", processed_cases=len(entries),
                                   tested_cases=len(tested_rows),
                                   skipped_cases=len(entries) - len(tested_rows),
                                   successes=successes,
                                   nonconverged=len(tested_rows) - successes,
                                   success_rate_among_tested=successes / len(tested_rows),
                                   user_requested_stop_after=stop_after,
                                   task_parameters_unchanged=True,
                                   encoder_frozen=True)
                    atomic_json(output, payload)
                    return
                if pause_request is not None and pause_request.exists():
                    payload["status"] = "paused"
                    atomic_json(output, payload)
                    return
                index = len(entries)
                start = (index // 10) * 10
                parent = records[start]["parent_case_id"]
                position = index - start
                if position >= 4:
                    first_successes = sum(item["converged"] for item in entries[start:start + 4])
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
        require(len(entries) == 600, "Adaptive result has wrong coverage")
        groups = {}
        for start in range(0, 600, 10):
            group = entries[start:start + 10]
            parent = records[start]["parent_case_id"]
            tested_rows = [item for item in group if item["disposition"] == "tested"]
            skipped_rows = [item for item in group if item["disposition"] == "skipped"]
            successes = sum(item["converged"] for item in tested_rows)
            groups[parent] = {"first4_successes": sum(item["converged"] for item in group[:4]),
                              "skip_decision": bool(skipped_rows),
                              "tested": len(tested_rows), "skipped": len(skipped_rows),
                              "successes": successes,
                              "nonconverged": len(tested_rows) - successes,
                              "prior_protocol_excluded": sum(
                                  item.get("physically_tested_in_prior_protocol", False)
                                  for item in skipped_rows)}
            require((len(tested_rows), len(skipped_rows)) in ((4, 6), (10, 0)),
                    f"Wrong parent coverage: {parent}")
        tested_rows = [item for item in entries if item["disposition"] == "tested"]
        successes = sum(item["converged"] for item in tested_rows)
        failures = len(tested_rows) - successes
        payload.update(status="complete", processed_cases=600,
                       tested_cases=len(tested_rows), skipped_cases=600 - len(tested_rows),
                       successes=successes, nonconverged=failures,
                       success_rate_among_tested=successes / len(tested_rows),
                       groups=groups, failure_reasons=dict(Counter(
                           item["reason"] for item in tested_rows if not item["converged"])),
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
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.mode == "prepare":
        prepare(output)
    else:
        evaluate(output, args.pause_request.resolve() if args.pause_request else None,
                 args.stop_after)


if __name__ == "__main__":
    main()
