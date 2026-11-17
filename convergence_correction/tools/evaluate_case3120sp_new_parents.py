"""Audited, zero-update supplemental test on six held-out case3120sp parents.

This is a same-grid unseen-parent test, NOT the different-size external_test.
The historical sealed train/validation/test manifests remain untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from rl_poweradj.case_adapter import CaseAdapter
from rl_poweradj.data_preflight import input_card_hashes
from rl_poweradj.foundation import UnifiedFrozenEncoder
from rl_poweradj.graph_dqn import GraphDQN
from rl_poweradj.graph_evaluation import state_hash
from rl_poweradj.graph_training import GraphEvaluationPaused, evaluate_records
from rl_poweradj.psasp import PSASPRunner, solver_processes_present
from rl_poweradj.run_store import atomic_json, file_hash


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "跨网训练数据" / "case3120sp"
GROUPS = tuple(f"g{i:03d}" for i in range(5, 11))
TRAIN_MANIFEST = ROOT / "artifacts/datasets/multifamily-joint-v1/train-validation-manifest.json"
SEALED_TEST = ROOT / "artifacts/datasets/multifamily-joint-v1/seen-family-test-sealed.json"
PUBLICATION = ROOT / "artifacts/runs/multifamily-joint-v1-seed0-20260923/selected.json"
SOLVER = ROOT / "PSASP/PSASP/WMLFRTMsg.exe"
SOLVER_AUDIT = ROOT / "artifacts/solver-audit/case3120sp-newparents-g005-g010-20260925"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fingerprint(cards: dict) -> str:
    return hashlib.sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest()


def build_audit() -> dict:
    selected = read_json(PUBLICATION)
    require(selected.get("purpose") == "formal_multifamily_training", "Wrong publication purpose")
    require(selected.get("selection") == "family_macro_validation_only", "Wrong model selection")
    require(selected.get("training_bus_counts") == [118, 300, 2000, 3120], "Wrong training sizes")
    require(file_hash(selected["model"]) == selected["model_sha256"], "Selected model changed")
    require(file_hash(SOLVER) == selected["solver_sha256"], "Solver changed")
    existing_ids: set[str] = set()
    existing_parents: set[str] = set()
    existing_inputs: set[str] = set()
    for path in (TRAIN_MANIFEST, SEALED_TEST):
        for row in read_json(path)["cases"]:
            existing_ids.add(row["case_id"])
            existing_parents.add(row["parent_case_id"])
            existing_inputs.add(fingerprint(row["input_cards"]))

    records: list[dict] = []
    policies: dict[str, dict] = {}
    new_inputs: set[str] = set()
    group_evidence: dict[str, dict] = {}
    for group in GROUPS:
        parent = f"case3120sp_{group}"
        require(parent not in existing_parents, f"Parent was used before: {parent}")
        group_dir = DATA / group
        group_manifest_path = group_dir / "group_manifest.json"
        group_manifest = read_json(group_manifest_path)
        require(group_manifest.get("group_id") == parent, f"Group identity mismatch: {group}")
        require(group_manifest.get("status") == "COMPLETE", f"Incomplete group: {group}")
        baseline_path = group_dir / "baseline/result/run_result.json"
        baseline = read_json(baseline_path)
        require(baseline.get("solver_status") == "CONVERGED" and baseline.get("cal_flag") == 1,
                f"Baseline not converged: {group}")
        require(baseline.get("executable_sha256") == selected["solver_sha256"],
                f"Baseline used different solver: {group}")
        expected_names = [f"neg_{i:03d}" for i in range(1, 21)]
        negative_dir = group_dir / "negatives"
        actual_names = sorted(item.name for item in negative_dir.iterdir() if item.is_dir())
        require(actual_names == expected_names, f"Wrong negative set: {group}")
        indexed = {item["sample_id"]: item for item in group_manifest["negative_samples"]}
        require(len(indexed) == 20 and len(group_manifest["negative_samples"]) == 20,
                f"Wrong group manifest count: {group}")
        group_evidence[group] = {"group_manifest_sha256": file_hash(group_manifest_path),
                                 "baseline_result_sha256": file_hash(baseline_path)}
        for name in expected_names:
            case_id = f"{parent}_{name}"
            require(case_id not in existing_ids, f"Case used before: {case_id}")
            sample_dir = negative_dir / name
            input_dir = sample_dir / "input"
            sample_path = sample_dir / "sample_manifest.json"
            diff_path = sample_dir / "input_diff.json"
            result_path = sample_dir / "result/run_result.json"
            sample = read_json(sample_path)
            result = read_json(result_path)
            changes = read_json(diff_path)
            require(sample.get("sample_id") == case_id and sample.get("group_id") == parent,
                    f"Sample identity mismatch: {case_id}")
            require(sample.get("accepted") is True and sample.get("reason") == "ACCEPTED",
                    f"Sample not accepted: {case_id}")
            require(sample.get("solver_status") == "NONCONVERGED"
                    and result.get("solver_status") == "NONCONVERGED"
                    and result.get("cal_flag") == -1 and result.get("returncode") == 0,
                    f"Not a verified nonconvergent sample: {case_id}")
            require(result.get("executable_sha256") == selected["solver_sha256"]
                    and result.get("input_mutations") == [],
                    f"Solver identity or input mutation differs: {case_id}")
            checks = sample.get("checks", {})
            require(all(checks.get(key) is True for key in
                        ("mark_only", "other_inputs_identical", "restore_input_identical")),
                    f"SH-only disturbance proof missing: {case_id}")
            require(changes == sample.get("changes") and changes
                    and all(item["id_name"].startswith("SH")
                            and item["before"] == 1 and item["after"] == 0 for item in changes),
                    f"Wrong disturbance direction: {case_id}")
            require(indexed[case_id]["input_hash"] == sample["input_hash"],
                    f"Group sample hash differs: {case_id}")
            actual_files = {item.name: file_hash(item) for item in input_dir.iterdir() if item.is_file()}
            require(actual_files == result.get("input_hashes"), f"Input files changed: {case_id}")
            cards = input_card_hashes(input_dir)
            fp = fingerprint(cards)
            require(fp not in existing_inputs and fp not in new_inputs,
                    f"Duplicate/leaked electrical input: {case_id}")
            new_inputs.add(fp)
            adapter = CaseAdapter(input_dir)
            require(adapter.n == 3120, f"Wrong bus count: {case_id}")
            if group not in policies:
                proposal = adapter.proposal()
                require(proposal["devices"], f"No SH candidates: {group}")
                policies[group] = {**proposal, "approved": True}
            adapter.authorize(policies[group])
            records.append({"case_id": case_id, "grid_family_id": "grid_3120",
                            "parent_case_id": parent, "group": group,
                            "source": str(input_dir.resolve()), "input_cards": cards,
                            "sample_manifest_sha256": file_hash(sample_path),
                            "input_diff_sha256": file_hash(diff_path),
                            "label_evidence_sha256": file_hash(result_path)})
    require(len(records) == 120, "Expected exactly 120 held-out-parent cases")
    return {"schema": 1, "phase": "supplemental_unseen_parent_same_grid_test",
            "not_external_test": True, "purpose": "case3120sp_SH_adjustment",
            "publication": str(PUBLICATION), "publication_sha256": file_hash(PUBLICATION),
            "model_sha256": selected["model_sha256"],
            "solver_sha256": selected["solver_sha256"],
            "train_manifest_sha256": file_hash(TRAIN_MANIFEST),
            "sealed_seen_test_sha256": file_hash(SEALED_TEST),
            "groups": list(GROUPS), "group_evidence": group_evidence,
            "policies": policies, "cases": records}


def evaluate(audit_path: Path, output: Path, *, resume: bool, pause_request: Path | None) -> None:
    audit_path, output = audit_path.resolve(), output.resolve()
    require(output.exists() == resume, "Result existence and resume mode differ")
    audited = read_json(audit_path)
    require(audited == build_audit(), "Audit or source files changed since preflight")
    selected = read_json(PUBLICATION)
    protocol = {"phase": "supplemental_unseen_parent_same_grid_test",
                "audit_sha256": file_hash(audit_path),
                "publication_sha256": file_hash(PUBLICATION),
                "model_sha256": selected["model_sha256"],
                "solver_sha256": selected["solver_sha256"],
                "horizon": selected["recipe"]["horizon"],
                "device": "cuda", "zero_updates": True}
    completed = []
    if resume:
        old = read_json(output)
        require(old["protocol"] == protocol, "Resume protocol changed")
        completed = old["cases"]
        require([r["case_id"] for r in completed]
                == [r["case_id"] for r in audited["cases"][:len(completed)]],
                "Completed results are not a manifest prefix")
    require(not solver_processes_present(), "Another PSASP process is active")
    encoder = UnifiedFrozenEncoder.from_project(ROOT, "cuda")
    agent, _, _, _ = GraphDQN.load(selected["model"], encoder, device="cuda")
    agent.online.eval()
    agent.target.eval()
    before, updates = state_hash(agent.online), agent.updates
    payload = {"status": "running", "protocol": protocol, "cases": completed}
    records = [{**row, "policy": audited["policies"][row["group"]]}
               for row in audited["cases"]]

    def record(row: dict) -> None:
        completed.append(row)
        atomic_json(output, payload)

    atomic_json(output, payload)
    try:
        evaluate_records(agent, encoder, PSASPRunner(SOLVER, audit_dir=SOLVER_AUDIT),
                         records[len(completed):], protocol["horizon"], on_case=record,
                         pause_requested=(lambda: pause_request.exists()) if pause_request else None)
        require(state_hash(agent.online) == before and agent.updates == updates,
                "Evaluation changed task-network parameters")
        encoder.verify_frozen()
        require(build_audit() == audited, "Source files changed during evaluation")
        summary = {}
        for group in GROUPS:
            cases = [row for row in completed if row["case_id"].startswith(f"case3120sp_{group}_")]
            required = [row for row in cases if not row["initial_converged"]]
            require(len(cases) == 20 and len(required) == 20,
                    f"Incomplete or initially converged group: {group}")
            successes = sum(row["converged"] for row in required)
            summary[group] = {"action_required": 20, "successes": successes,
                              "success_rate": successes / 20,
                              "mean_steps": sum(row["steps"] for row in required) / 20,
                              "reasons": dict(Counter(row["reason"] for row in required))}
        payload.update(status="complete", groups=summary, action_required=120,
                       successes=sum(row["successes"] for row in summary.values()),
                       task_parameters_unchanged=True, encoder_frozen=True)
    except GraphEvaluationPaused:
        payload["status"] = "paused"
    except BaseException as exc:
        payload.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        atomic_json(output, payload)
        raise
    atomic_json(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "evaluate"))
    parser.add_argument("--audit-file", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-request", type=Path)
    args = parser.parse_args()
    if args.mode == "audit":
        require(not args.audit_file.exists(), "Audit file already exists")
        atomic_json(args.audit_file, build_audit())
    else:
        require(args.output is not None, "Evaluation output is required")
        evaluate(args.audit_file, args.output, resume=args.resume,
                 pause_request=args.pause_request)


if __name__ == "__main__":
    main()
