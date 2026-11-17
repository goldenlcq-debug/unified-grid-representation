"""Zero-update evaluation of sealed joint-grid models on seen or unseen grids."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from .foundation import UnifiedFrozenEncoder
from .graph_dqn import GraphDQN
from .graph_evaluation import state_hash
from .graph_training import GraphEvaluationPaused, evaluate_records
from .multifamily_external import audit_external_manifest
from .multifamily_manifest import audit_multifamily_manifest
from .multifamily_sampler import TRAIN_FAMILIES
from .psasp import PSASPRunner
from .run_store import atomic_json, file_hash
from .evaluation_metrics import summarize_action_steps, STEP_METRIC_DEFINITIONS


def evaluate_multifamily_published(root, publication, manifest, solver, output, *,
                                   phase: str, device: str = "cuda", resume: bool = False,
                                   pause_request=None) -> dict:
    if phase not in ("seen_family_test", "external_test"):
        raise ValueError("Only sealed final test phases are allowed")
    root = Path(root).resolve()
    publication, manifest, solver, output = map(lambda x: Path(x).resolve(),
                                                 (publication, manifest, solver, output))
    if output.exists() != resume:
        raise ValueError("Result existence and resume mode differ")
    selected = json.loads(publication.read_text(encoding="utf-8"))
    if (selected.get("purpose") != "formal_multifamily_training"
            or selected.get("protocol") != "multi_family_joint_training_v1"
            or selected.get("selection") != "family_macro_validation_only"
            or tuple(selected.get("training_family_ids", ())) != TRAIN_FAMILIES):
        raise ValueError("Not a sealed four-family model")
    model = Path(selected["model"]).resolve()
    if (not model.is_relative_to(publication.parent / "models")
            or file_hash(model) != selected["model_sha256"]):
        raise ValueError("Published model path or hash differs")
    if file_hash(solver) != selected["solver_sha256"]:
        raise ValueError("Solver differs from sealed model identity")
    training_sizes = selected.get("training_bus_counts")
    if sorted(training_sizes or []) != [118, 300, 2000, 3120]:
        raise ValueError("Published training node sizes are incomplete")
    if phase == "seen_family_test":
        audited = audit_multifamily_manifest(manifest, verify_files=True)
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        if raw["status"] != "sealed_seen_family_test":
            raise ValueError("Wrong seen-family test manifest")
    else:
        audited = audit_external_manifest(
            manifest, training_families=TRAIN_FAMILIES, training_bus_counts=training_sizes)
    records = audited["cases"]
    protocol = {"publication_sha256": file_hash(publication),
                "model_sha256": file_hash(model), "manifest_digest": audited["digest"],
                "solver_sha256": file_hash(solver), "phase": phase, "device": device,
                "horizon": selected["recipe"]["horizon"], "zero_updates": True}
    completed = []
    if resume:
        old = json.loads(output.read_text(encoding="utf-8"))
        if old["protocol"] != protocol:
            raise ValueError("Evaluation resume protocol changed")
        completed = old["cases"]
        if [row["case_id"] for row in records[:len(completed)]] != [row["case_id"] for row in completed]:
            raise ValueError("Saved cases are not an exact manifest prefix")
    encoder = UnifiedFrozenEncoder.from_project(root, device)
    agent, _, _, _ = GraphDQN.load(model, encoder, device=device)
    agent.online.eval()
    agent.target.eval()
    before, updates = state_hash(agent.online), agent.updates
    payload = {"status": "running", "protocol": protocol, "cases": completed}

    def record(row):
        completed.append(row)
        atomic_json(output, payload)

    atomic_json(output, payload)
    try:
        evaluate_records(agent, encoder, PSASPRunner(solver), records[len(completed):],
                         protocol["horizon"], on_case=record,
                         pause_requested=(lambda: Path(pause_request).exists()) if pause_request else None)
        if state_hash(agent.online) != before or agent.updates != updates:
            raise RuntimeError("Evaluation updated the task network")
        encoder.verify_frozen()
        required = [row for row in completed if not row["initial_converged"]]
        families = {}
        for family in sorted({row["family"] for row in completed}):
            group = [row for row in required if row["family"] == family]
            successes = sum(row["converged"] for row in group)
            families[family] = {"action_required": len(group), "successes": successes,
                                "success_rate": successes / len(group) if group else None,
                                "reasons": dict(Counter(row["reason"] for row in group)),
                                **summarize_action_steps([row for row in completed if row["family"] == family])}
        rates = [item["success_rate"] for item in families.values() if item["success_rate"] is not None]
        payload.update(status="complete", action_required=len(required), families=families,
                       micro_success_rate=sum(row["converged"] for row in required) / len(required)
                       if required else None,
                       macro_success_rate=sum(rates) / len(rates) if rates else None,
                       task_parameters_unchanged=True, encoder_frozen=True)
        # Use all saved cases, including the prefix from an earlier invocation.
        payload.update(summarize_action_steps(completed))
        payload["step_metric_definitions"] = dict(STEP_METRIC_DEFINITIONS)
    except GraphEvaluationPaused:
        payload["status"] = "paused"
    except BaseException as exc:
        payload.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        atomic_json(output, payload)
        raise
    atomic_json(output, payload)
    return payload
