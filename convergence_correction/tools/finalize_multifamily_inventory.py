"""Apply the recorded parent-level split policy to a verified inventory."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_poweradj.run_store import atomic_json, file_hash
from tools.build_multifamily_inventory import FAMILIES, SEED, group_splits

SOURCE = ROOT / "artifacts/datasets/multifamily-joint-v1/development-inventory-20260923.json"
OUTPUT = ROOT / "artifacts/datasets/multifamily-joint-v1/development-inventory-20260923-final-v2.json"
PINNED = {
    "grid_3120": frozenset(f"case3120sp_g{i:03d}" for i in range(16, 21)),
    "grid_300": frozenset(),
    "grid_118": frozenset({"case118_g001"}),
}


def finalize(data: dict) -> dict:
    if data.get("status") != "development_inventory_verified_requires_control_authorization_and_training_release":
        raise ValueError("Inventory was not fully verified")
    if len(data.get("cases", [])) != 12000:
        raise ValueError("Inventory does not cover 12000 new-grid samples")
    if set(data.get("groups", {})) != set(FAMILIES):
        raise ValueError("Wrong family coverage")
    mapping = {}
    baseline_fingerprints = {}
    for family, (_, _, per_group, quotas, _) in FAMILIES.items():
        groups = data["groups"][family]
        expected_cases = sum(quotas) * per_group
        if len(groups) != sum(quotas):
            raise ValueError(f"{family}: wrong group count")
        for group in groups:
            fingerprint = hashlib.sha256(json.dumps(group["baseline_input_cards"],
                                                    sort_keys=True).encode()).hexdigest()
            if fingerprint in baseline_fingerprints:
                raise ValueError(f"Duplicate baseline groups: {group['group_id']} and "
                                 f"{baseline_fingerprints[fingerprint]}")
            baseline_fingerprints[fingerprint] = group["group_id"]
        forced = PINNED[family]
        splits = group_splits(family, [
            {"group_id": row["group_id"], "baseline_hash": row["baseline_hash_declared"]}
            for row in groups], quotas, forced)
        for row in groups:
            row["split"] = splits[row["group_id"]]
        mapping[family] = splits
        if sum(case["grid_family_id"] == family for case in data["cases"]) != expected_cases:
            raise ValueError(f"{family}: wrong sample count")
    for case in data["cases"]:
        case["split"] = mapping[case["grid_family_id"]][case["parent_case_id"]]
    data["training_family_ids"] = ["grid_2000", *FAMILIES]
    data["split_seed"] = SEED
    data["split_policy"] = {
        "forced_train_parent_ids": {family: sorted(parents) for family, parents in PINNED.items()},
        "reason": "3120 g016-g020 had prior rejected-test exposure; case118 g001 was used in a development rule smoke test before split sealing",
        "selection": "hash_ranked_remaining_parents",
    }
    counts = {family: dict(Counter(row["split"] for row in data["cases"]
                                    if row["grid_family_id"] == family)) for family in FAMILIES}
    if counts != {"grid_3120": {"train": 2400, "validation": 800, "test": 800},
                   "grid_300": {"train": 2800, "validation": 600, "test": 600},
                   "grid_118": {"train": 2800, "validation": 600, "test": 600}}:
        raise ValueError(f"Wrong case splits: {counts}")
    return data


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    payload = finalize(json.loads(SOURCE.read_text(encoding="utf-8")))
    payload["pre_split_inventory_sha256"] = file_hash(SOURCE)
    atomic_json(OUTPUT, payload)
    print(json.dumps({"output": str(OUTPUT), "cases": len(payload["cases"]),
                      "source_sha256": payload["pre_split_inventory_sha256"]}, ensure_ascii=False))
