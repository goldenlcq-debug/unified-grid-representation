"""Build a content-bound release after the four-family engineering gates pass."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_poweradj.graph_recipe import GraphRecipe
from rl_poweradj.multifamily_manifest import audit_multifamily_manifest
from rl_poweradj.multifamily_release import release_identity, verify_release
from rl_poweradj.run_store import atomic_json, file_hash

MANIFEST = ROOT / "artifacts/datasets/multifamily-joint-v1/train-validation-manifest.json"
SOLVER = ROOT / "PSASP/PSASP/WMLFRTMsg.exe"
CONFIG = ROOT / "config/multifamily-joint-v1-proposed.json"
OUTPUT = ROOT / "artifacts/readiness/multifamily-joint-v1-20260923/ready.json"
AUTHORIZATION = ROOT / "artifacts/decisions/multifamily-formal-run-authorization-20260923.json"
PILOT = ROOT / "artifacts/runs/multifamily-pilot20-20260923/summary.json"
EVIDENCE = {
    "tests": "artifacts/results/multifamily-test-scope-20260923.json",
    "data_audit": "artifacts/results/multifamily-data-audit-20260923.json",
    "lifecycle": "artifacts/runs/multifamily-final-acceptance-20260923/summary.json",
    "parity": "artifacts/results/multifamily-final-resume-parity-20260923.json",
    "prerun": "artifacts/results/multifamily-graph-interface-20260923.json",
    "resources": "artifacts/results/multifamily-resource-audit-20260923.json",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    recipe = GraphRecipe.load(CONFIG)
    authorization = read(AUTHORIZATION)
    if (authorization.get("authority") != "explicit_user_choice_in_current_task"
            or authorization.get("choice") != "directly_start_full_recipe_after_engineering_release"
            or authorization.get("episodes") != recipe.total_episodes
            or authorization.get("horizon") != recipe.horizon):
        raise ValueError("Full-run budget was not explicitly authorized")
    audit = audit_multifamily_manifest(MANIFEST, verify_files=True)
    if len(audit["cases"]) != 12934:
        raise ValueError("Training data coverage changed")
    data_evidence = read(ROOT / EVIDENCE["data_audit"])
    if (data_evidence.get("status") != "passed"
            or data_evidence.get("training_manifest_sha256") != file_hash(MANIFEST)
            or data_evidence.get("seen_family_test_sha256") != read(MANIFEST)["seen_test_sha256"]):
        raise ValueError("Data audit evidence differs from sealed inputs")
    tests = read(ROOT / EVIDENCE["tests"])
    if (tests["applicable_suite"]["tests"] < 120
            or tests["applicable_suite"]["failures"] != 0
            or tests["applicable_suite"]["errors"] != 0
            or tests["full_suite"]["failures"] != tests["applicable_suite"]["known_missing_fixture_tests_deselected"]):
        raise ValueError("Applicable code regression was not passed")
    for suite in ("applicable_suite", "full_suite"):
        item = tests[suite]
        if file_hash(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Changed {suite} XML result")
    lifecycle = read(ROOT / EVIDENCE["lifecycle"])
    if (lifecycle.get("status") != "passed" or lifecycle.get("encoder_frozen") is not True
            or lifecycle.get("formal_training_started") is not False
            or lifecycle["progress"]["updates"] < 1):
        raise ValueError("Real four-family lifecycle acceptance differs")
    parity = read(ROOT / EVIDENCE["parity"])
    if parity.get("status") != "passed" or parity.get("encoder_frozen") is not True:
        raise ValueError("Real CUDA resume parity differs")
    prerun = read(ROOT / EVIDENCE["prerun"])
    if (prerun.get("status") != "passed" or prerun.get("encoder_frozen") is not True
            or {row["family"] for row in prerun["cases"]}
            != {"grid_2000", "grid_3120", "grid_300", "grid_118"}):
        raise ValueError("Variable-grid interface evidence differs")
    resources = read(ROOT / EVIDENCE["resources"])
    if resources.get("guided_acceptance", {}).get("solver_calls", 0) < 1:
        raise ValueError("PSASP/CUDA resource measurement absent")
    pilot = read(PILOT)
    if pilot.get("status") != "passed" or pilot.get("formal_training_started") is not False:
        raise ValueError("Bounded development pilot did not finish")
    evidence = {kind: {"path": path, "sha256": file_hash(ROOT / path)}
                for kind, path in EVIDENCE.items()}
    release = {
        "schema": 1,
        "kind": "multifamily_joint_release_v1",
        "status": "ready_for_training",
        "identity": release_identity(ROOT, MANIFEST, SOLVER, recipe),
        "evidence": evidence,
        "pilot": {"path": str(PILOT.relative_to(ROOT)), "sha256": file_hash(PILOT),
                  "validation_macro_success_rate": read(PILOT.parent / "validation-summary.json")
                  ["macro_success_rate"]},
        "authorization": {"path": str(AUTHORIZATION.relative_to(ROOT)),
                          "sha256": file_hash(AUTHORIZATION)},
        "limitations": [
            "The 20-episode pilot validation macro success rate is not a formal model result.",
            "Seven legacy tests require a removed 600-node fixture; 127 applicable tests passed.",
            "Full training and validation may take tens to hundreds of hours.",
        ],
    }
    atomic_json(OUTPUT, release)
    verify_release(ROOT, OUTPUT, MANIFEST, SOLVER, recipe)
    print(json.dumps({"status": release["status"], "path": str(OUTPUT),
                      "manifest_sha256": file_hash(MANIFEST),
                      "recipe_episodes": recipe.total_episodes}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
