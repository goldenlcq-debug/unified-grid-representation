"""Combine verified new-grid inventory with the prior 2000-bus split.

The resulting schema-2 manifest remains a development artifact. Formal training
requires a separate multi-family auditor, release gate, and runtime.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_poweradj.data_preflight import input_card_hashes
from rl_poweradj.run_store import atomic_json, file_hash

INVENTORY = ROOT / "artifacts/datasets/multifamily-joint-v1/development-inventory-20260923-final-v2.json"
SOURCE = ROOT / "artifacts/datasets/single-source-20260920/manifest.json"
AUTHORITY = ROOT / "artifacts/decisions/multifamily-shunt-control-20260923.json"
OUTPUT = ROOT / "artifacts/datasets/multifamily-joint-v1/development-manifest-20260923.json"
EXPECTED = {"grid_2000": 3452, "grid_3120": 4000, "grid_300": 4000, "grid_118": 4000}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def policy_id(policy: dict) -> str:
    return hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build() -> dict:
    inventory = load(INVENTORY)
    source = load(SOURCE)
    authority = load(AUTHORITY)
    if (inventory.get("schema") != 2
            or inventory.get("status") != "development_inventory_verified_requires_control_authorization_and_training_release"
            or inventory.get("split_protocol") != "multi_family_joint_training_v1"
            or len(inventory.get("cases", [])) != 12000
            or inventory.get("excluded_group_ids") != ["case3120sp_g010"]
            or set(inventory.get("excluded_family_ids", [])) != {"grid_2869"}
            or set(inventory.get("split_policy", {}).get("forced_train_parent_ids", {}).get("grid_3120", []))
               != {f"case3120sp_g{i:03d}" for i in range(16, 21)}):
        raise ValueError("New-grid inventory is not complete or has wrong exclusions")
    if inventory["split_policy"]["forced_train_parent_ids"].get("grid_118") != ["case118_g001"]:
        raise ValueError("Development-exposed case118 parent was not isolated in train")
    if (source.get("schema") != 1 or source.get("split_protocol") != "single_source_zero_shot"
            or source.get("source_family_id") != "source_2000" or len(source.get("cases", [])) != 3452):
        raise ValueError("2000-bus source manifest changed protocol or coverage")
    if (authority.get("authority") != "explicit_user_instruction"
            or set(authority.get("grid_family_ids", [])) != {"grid_118", "grid_300", "grid_3120"}
            or authority.get("authorized_operation") != "Mark_0_or_1"):
        raise ValueError("Missing explicit new-grid control authorization")
    policies: dict[str, dict] = {}
    group_policy: dict[str, str] = {}
    for family, groups in inventory["groups"].items():
        for group in groups:
            proposal = group["policy_proposal"]
            if proposal.get("approved") is not False or not proposal.get("devices"):
                raise ValueError(f"{group['group_id']}: invalid discovery proposal")
            approved = {**proposal, "approved": True,
                        "authorization": "User 2026-09-23 approved all case-local same-bus SH Mark operations; "
                                         "see artifacts/decisions/multifamily-shunt-control-20260923.json"}
            identity = policy_id(approved)
            policies.setdefault(identity, approved)
            group_policy[group["group_id"]] = identity
    cases: list[dict] = []
    seen_ids: set[str] = set()
    parent_splits: dict[tuple[str, str], str] = {}
    fingerprints: dict[str, str] = {}

    def add(row: dict) -> None:
        case_id = row["case_id"]
        family = row["grid_family_id"]
        parent = row["parent_case_id"]
        split = row["split"]
        if case_id in seen_ids or family not in EXPECTED or split not in ("train", "validation", "test"):
            raise ValueError(f"Invalid or duplicate case: {case_id}")
        key = (family, parent)
        if key in parent_splits and parent_splits[key] != split:
            raise ValueError(f"Parent leakage: {key}")
        fp = hashlib.sha256(json.dumps(row["input_cards"], sort_keys=True).encode()).hexdigest()
        if fp in fingerprints:
            raise ValueError(f"Duplicate input: {case_id} and {fingerprints[fp]}")
        fingerprints[fp] = case_id
        parent_splits[key] = split
        seen_ids.add(case_id)
        cases.append(row)

    for source_index, old in enumerate(source["cases"], 1):
        if old["grid_family_id"] != "source_2000":
            raise ValueError("Old source contains an unexpected family")
        case = Path(old["source"])
        if input_card_hashes(case) != old["input_cards"]:
            raise ValueError(f"Changed 2000-bus input: {case}")
        label = old["label_evidence"]
        label_path = SOURCE.parent / label["path"]
        if file_hash(label_path) != label["sha256"]:
            raise ValueError(f"Changed 2000-bus label: {label_path}")
        policy = old["policy"]
        if policy.get("approved") is not True:
            raise ValueError(f"2000-bus case has no approved policy: {case}")
        identity = policy_id(policy)
        policies.setdefault(identity, policy)
        add({
            "case_id": old["case_id"], "grid_family_id": "grid_2000",
            "parent_case_id": old["parent_case_id"], "split": old["split"],
            "source": str(case.resolve()), "input_cards": old["input_cards"],
            "policy_id": identity, "flow_label": old["flow_label"],
            "label_evidence": {"path": str(label_path.resolve()), "sha256": label["sha256"],
                               "origin": "single_source_20260920"},
        })
        if source_index % 500 == 0:
            print(f"verified 2000-bus source {source_index}/{len(source['cases'])}", flush=True)
    for row in inventory["cases"]:
        if row["grid_family_id"] == "grid_2869" or row["parent_case_id"] == "case3120sp_g010":
            raise ValueError("Rejected data entered the new inventory")
        group_id = row["policy_group_id"]
        if group_id not in group_policy:
            raise ValueError(f"Missing parent policy for {row['case_id']}")
        case = Path(row["source"])
        result_path = case.parent / "result" / "run_result.json"
        # The content-bound inventory has already rehashed every new-grid
        # input and solver record. Formal readiness rechecks them once more.
        add({
            "case_id": row["case_id"], "grid_family_id": row["grid_family_id"],
            "parent_case_id": row["parent_case_id"], "split": row["split"],
            "source": row["source"], "input_cards": row["input_cards"],
            "policy_id": group_policy[group_id], "flow_label": "nonconverged",
            "changed_devices": row["changed_devices"],
            "label_evidence": {"path": str(result_path.resolve()),
                               "sha256": row["solver_record_sha256"],
                               "origin": "new_grid_generator_psasp_record"},
        })
    totals = Counter(row["grid_family_id"] for row in cases)
    if dict(totals) != EXPECTED:
        raise ValueError(f"Wrong family coverage: {dict(totals)}")
    by_split = {family: dict(Counter(row["split"] for row in cases if row["grid_family_id"] == family))
                for family in EXPECTED}
    by_parents: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in cases:
        by_parents[row["grid_family_id"]][row["split"]].add(row["parent_case_id"])
    parents = {family: {split: len(items) for split, items in splits.items()}
               for family, splits in by_parents.items()}
    if parents != {"grid_2000": {"train": 126, "validation": 29, "test": 28},
                   "grid_3120": {"train": 6, "validation": 2, "test": 2},
                   "grid_300": {"train": 14, "validation": 3, "test": 3},
                   "grid_118": {"train": 14, "validation": 3, "test": 3}}:
        raise ValueError(f"Wrong parent coverage: {parents}")
    return {
        "schema": 2,
        "status": "development_data_verified_not_training_release",
        "split_protocol": "multi_family_joint_training_v1",
        "training_family_ids": list(EXPECTED),
        "excluded_family_ids": ["grid_2869"],
        "excluded_group_ids": ["case3120sp_g010"],
        "source_manifest_sha256": file_hash(SOURCE),
        "new_grid_inventory_sha256": file_hash(INVENTORY),
        "control_authority_sha256": file_hash(AUTHORITY),
        "split_seed": inventory["split_seed"],
        "family_case_counts": dict(totals),
        "family_split_counts": by_split,
        "family_parent_counts": parents,
        "policies": policies,
        "cases": cases,
    }


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    draft = build()
    atomic_json(OUTPUT, draft)
    print(json.dumps({"output": str(OUTPUT), "cases": len(draft["cases"]),
                      "parents": draft["family_parent_counts"],
                      "splits": draft["family_split_counts"]}, ensure_ascii=False), flush=True)
