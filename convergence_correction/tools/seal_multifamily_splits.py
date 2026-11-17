"""Separate train/validation inputs from the committed seen-family test set."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_poweradj.run_store import atomic_json, file_hash

SOURCE = ROOT / "artifacts/datasets/multifamily-joint-v1/development-manifest-20260923.json"
TRAIN = ROOT / "artifacts/datasets/multifamily-joint-v1/train-validation-manifest.json"
TEST = ROOT / "artifacts/datasets/multifamily-joint-v1/seen-family-test-sealed.json"


def build_parts(draft: dict) -> tuple[dict, dict]:
    if draft.get("status") != "development_data_verified_not_training_release":
        raise ValueError("Development data audit is incomplete")
    cases = draft["cases"]
    training_cases = [row for row in cases if row["split"] in ("train", "validation")]
    test_cases = [row for row in cases if row["split"] == "test"]
    if len(training_cases) != 12934 or len(test_cases) != 2518:
        raise ValueError("Wrong train/validation/test coverage")
    if len({(row["grid_family_id"], row["parent_case_id"]) for row in training_cases}
           & {(row["grid_family_id"], row["parent_case_id"]) for row in test_cases}):
        raise ValueError("Parent group leaked into sealed test")
    common = {key: draft[key] for key in (
        "schema", "split_protocol", "training_family_ids", "excluded_family_ids",
        "excluded_group_ids", "source_manifest_sha256", "new_grid_inventory_sha256",
        "control_authority_sha256", "split_seed", "family_parent_counts")}

    def part(rows: list[dict], status: str) -> dict:
        used = {row["policy_id"] for row in rows}
        return {**common, "status": status, "cases": rows,
                "policies": {identity: draft["policies"][identity] for identity in sorted(used)},
                "family_case_counts": dict(Counter(row["grid_family_id"] for row in rows)),
                "family_split_counts": {
                    family: dict(Counter(row["split"] for row in rows if row["grid_family_id"] == family))
                    for family in common["training_family_ids"]},
                "development_manifest_sha256": file_hash(SOURCE)}

    test = part(test_cases, "sealed_seen_family_test")
    train = part(training_cases, "sealed_train_validation")
    return train, test


if __name__ == "__main__":
    if TRAIN.exists() or TEST.exists():
        raise FileExistsError("Sealed split output already exists")
    draft = json.loads(SOURCE.read_text(encoding="utf-8"))
    train, test = build_parts(draft)
    atomic_json(TEST, test)
    train["seen_test_sha256"] = file_hash(TEST)
    train["seen_test_case_count"] = len(test["cases"])
    atomic_json(TRAIN, train)
    print(json.dumps({"training_cases": len(train["cases"]), "sealed_test_cases": len(test["cases"]),
                      "training_manifest": str(TRAIN), "test_commitment_sha256": train["seen_test_sha256"]},
                     ensure_ascii=False), flush=True)
