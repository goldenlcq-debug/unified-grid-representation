"""Schema-2 joint-grid manifest audit. Historical single-source parser stays intact."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .case_adapter import CaseAdapter
from .data_preflight import input_card_hashes
from .multifamily_sampler import TRAIN_FAMILIES
from .run_store import file_hash

EXPECTED_CASES = {"grid_2000": 3452, "grid_3120": 4000, "grid_300": 4000, "grid_118": 4000}
EXPECTED_PARENTS = {
    "grid_2000": {"train": 126, "validation": 29, "test": 28},
    "grid_3120": {"train": 6, "validation": 2, "test": 2},
    "grid_300": {"train": 14, "validation": 3, "test": 3},
    "grid_118": {"train": 14, "validation": 3, "test": 3},
}
SEAL_EXPECTATIONS = {
    "development_data_verified_not_training_release":
        (EXPECTED_CASES, EXPECTED_PARENTS, {"train", "validation", "test"}),
    "sealed_train_validation":
        ({"grid_2000": 2934, "grid_3120": 3200, "grid_300": 3400, "grid_118": 3400},
         {"grid_2000": {"train": 126, "validation": 29},
          "grid_3120": {"train": 6, "validation": 2},
          "grid_300": {"train": 14, "validation": 3},
          "grid_118": {"train": 14, "validation": 3}},
         {"train", "validation"}),
    "sealed_seen_family_test":
        ({"grid_2000": 518, "grid_3120": 800, "grid_300": 600, "grid_118": 600},
         {"grid_2000": {"test": 28}, "grid_3120": {"test": 2},
          "grid_300": {"test": 3}, "grid_118": {"test": 3}},
         {"test"}),
}


def audit_multifamily_manifest(path, *, verify_files: bool = True) -> dict:
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    status = data.get("status")
    if (data.get("schema") != 2
            or status not in SEAL_EXPECTATIONS
            or data.get("split_protocol") != "multi_family_joint_training_v1"
            or tuple(data.get("training_family_ids", ())) != TRAIN_FAMILIES
            or data.get("excluded_family_ids") != ["grid_2869"]
            or data.get("excluded_group_ids") != ["case3120sp_g010"]):
        raise ValueError("Wrong multi-family manifest identity")
    policies = data.get("policies")
    if not isinstance(policies, dict) or not policies:
        raise ValueError("Missing approved physical control policies")
    cases = data.get("cases")
    expected_cases, expected_parents, allowed_splits = SEAL_EXPECTATIONS[status]
    if not isinstance(cases, list) or len(cases) != sum(expected_cases.values()):
        raise ValueError("Wrong multi-family case coverage")
    if verify_files:
        root = path.parents[3]
        dependencies = {
            "source_manifest_sha256": root / "artifacts/datasets/single-source-20260920/manifest.json",
            "new_grid_inventory_sha256": root / "artifacts/datasets/multifamily-joint-v1/development-inventory-20260923-final-v2.json",
            "control_authority_sha256": root / "artifacts/decisions/multifamily-shunt-control-20260923.json",
        }
        if any(file_hash(dep) != data.get(field) for field, dep in dependencies.items()):
            raise ValueError("Multi-family manifest dependency changed")
        if status != "development_data_verified_not_training_release":
            development = root / "artifacts/datasets/multifamily-joint-v1/development-manifest-20260923.json"
            if file_hash(development) != data.get("development_manifest_sha256"):
                raise ValueError("Sealed manifest development source changed")
        if status == "sealed_train_validation":
            seen_test = root / "artifacts/datasets/multifamily-joint-v1/seen-family-test-sealed.json"
            if file_hash(seen_test) != data.get("seen_test_sha256") or data.get("seen_test_case_count") != 2518:
                raise ValueError("Sealed seen-family test commitment changed")
    seen = set()
    fingerprints = set()
    parent_split = {}
    counts = Counter()
    parents = defaultdict(lambda: defaultdict(set))
    checked_policies = set()
    records = []
    for row in cases:
        case_id, family, parent, split = (row[key] for key in
                                          ("case_id", "grid_family_id", "parent_case_id", "split"))
        if (case_id in seen or family not in TRAIN_FAMILIES
                or parent == "case3120sp_g010" or split not in allowed_splits):
            raise ValueError(f"Invalid/repeated training identity: {case_id}")
        seen.add(case_id)
        fingerprint = hashlib.sha256(json.dumps(row["input_cards"], sort_keys=True).encode()).hexdigest()
        if fingerprint in fingerprints:
            raise ValueError(f"Duplicate electrical input: {case_id}")
        fingerprints.add(fingerprint)
        key = (family, parent)
        if key in parent_split and parent_split[key] != split:
            raise ValueError(f"Parent leakage: {key}")
        parent_split[key] = split
        policy_id = row["policy_id"]
        policy = policies.get(policy_id)
        if policy is None or policy.get("approved") is not True:
            raise ValueError(f"Missing approved policy: {case_id}")
        case = Path(row["source"])
        if verify_files:
            if input_card_hashes(case) != row["input_cards"]:
                raise ValueError(f"Changed input: {case_id}")
            evidence = row["label_evidence"]
            if file_hash(evidence["path"]) != evidence["sha256"]:
                raise ValueError(f"Changed label evidence: {case_id}")
            # The inventory proves Mark-only changes for every new-grid case.
            # A case-local adapter independently validates each parent policy
            # once; the input-card hash is checked for every child above.
            if key not in checked_policies:
                adapter = CaseAdapter(case)
                adapter.authorize(policy)
                checked_policies.add(key)
        counts[family] += 1
        parents[family][split].add(parent)
        records.append({**row, "policy": policy})
    if dict(counts) != expected_cases:
        raise ValueError(f"Wrong family case counts: {dict(counts)}")
    actual_parents = {family: {split: len(items) for split, items in splits.items()}
                      for family, splits in parents.items()}
    if actual_parents != expected_parents:
        raise ValueError(f"Wrong family parent counts: {actual_parents}")
    return {"cases": records, "digest": file_hash(path),
            "families": dict(counts), "parent_counts": actual_parents,
            "split_protocol": data["split_protocol"],
            "training_family_ids": list(TRAIN_FAMILIES)}
