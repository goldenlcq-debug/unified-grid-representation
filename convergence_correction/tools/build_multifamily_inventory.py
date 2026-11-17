"""Audit delivered multi-grid samples and build a non-executable split inventory.

This tool reads original samples only. It neither invokes PSASP nor grants control
authorization. The output is a development artifact, not a formal training manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_poweradj.case_adapter import CaseAdapter
from rl_poweradj.data_preflight import input_card_hashes
from rl_poweradj.run_store import atomic_json, file_hash

SOLVER_SHA256 = "02814ae90a8bd2770aba53fc1329ab6d84748a5d8d32fad2736e5f67c2e703e2"
SEED = "multifamily-joint-v1-20260923"
FAMILIES = {
    "grid_3120": ("case3120sp", tuple(range(11, 21)), 400, (6, 2, 2), 3120),
    "grid_300": ("case300", tuple(range(1, 21)), 200, (14, 3, 3), 300),
    "grid_118": ("case118", tuple(range(1, 21)), 200, (14, 3, 3), 118),
}
SPLITS = ("train", "validation", "test")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def difference(base: bytes, altered: bytes) -> list[dict]:
    """Return exact one-based L2 row/Mark changes; reject every other edit."""
    before = base.splitlines()
    after = altered.splitlines()
    if len(before) != len(after):
        raise ValueError("LF.L2 row count changed")
    changes = []
    for line_number, (left, right) in enumerate(zip(before, after), 1):
        if left == right:
            continue
        old = left.split(b",")
        new = right.split(b",")
        if len(old) < 4 or len(old) != len(new) or old[1:] != new[1:]:
            raise ValueError(f"LF.L2 line {line_number} changed outside Mark")
        mark_before, mark_after = old[0].strip(), new[0].strip()
        if mark_before not in (b"0", b"1") or mark_after not in (b"0", b"1") or mark_before == mark_after:
            raise ValueError(f"LF.L2 line {line_number} has invalid Mark change")
        if abs(int(old[1])) != abs(int(old[2])):
            raise ValueError(f"LF.L2 line {line_number} is not a same-bus branch")
        changes.append({"line": line_number, "before": int(mark_before), "after": int(mark_after)})
    if not changes:
        raise ValueError("Sample does not differ from baseline")
    return changes


def group_splits(family: str, groups: list[dict], quotas: tuple[int, int, int],
                 forced_train: frozenset[str] = frozenset()) -> dict[str, str]:
    if len(groups) != sum(quotas):
        raise ValueError(f"{family}: expected {sum(quotas)} parent groups, found {len(groups)}")
    ids = {group["group_id"] for group in groups}
    if not forced_train <= ids or len(forced_train) > quotas[0]:
        raise ValueError(f"{family}: invalid forced training parents")
    ranked = sorted(groups, key=lambda group: hashlib.sha256(
        f"{SEED}|{family}|{group['group_id']}|{group['baseline_hash']}".encode()
    ).hexdigest())
    ranked = [group for group in ranked if group["group_id"] not in forced_train]
    result = {group_id: "train" for group_id in forced_train}
    start = 0
    for split, quota in zip(SPLITS, (quotas[0] - len(forced_train), *quotas[1:])):
        for group in ranked[start:start + quota]:
            result[group["group_id"]] = split
        start += quota
    return result


def scan(root: Path, *, progress: bool = True) -> dict:
    root = root.resolve()
    selected_groups: dict[str, list[dict]] = {}
    selected_cases: list[dict] = []
    all_fingerprints: dict[str, str] = {}
    seen_ids: set[str] = set()
    for family, (directory, numbers, per_group, quotas, expected_buses) in FAMILIES.items():
        family_root = root / "跨网训练数据" / directory
        if not family_root.is_dir():
            raise FileNotFoundError(family_root)
        expected_names = {f"g{number:03d}" for number in numbers}
        actual_names = {p.name for p in family_root.iterdir() if p.is_dir()}
        # g010 is an excluded historical group. It may have been archived or
        # may still be present; in neither case can it enter this inventory.
        allowed_names = expected_names | ({"g010"} if family == "grid_3120" else set())
        if not expected_names <= actual_names or not actual_names <= allowed_names:
            raise ValueError(f"{family}: unexpected group directories: {sorted(actual_names ^ expected_names)}")
        group_rows: list[dict] = []
        for number in numbers:
            parent = family_root / f"g{number:03d}"
            group_path = parent / "group_manifest.json"
            group = load_json(group_path)
            group_id = f"{directory}_g{number:03d}"
            if (group.get("group_id") != group_id or group.get("case") != directory
                    or group.get("status") != "COMPLETE" or group.get("accepted_count") != per_group
                    or group.get("target_negatives") != per_group):
                raise ValueError(f"{group_path}: incomplete or unexpected group identity")
            members = group.get("negative_samples")
            if not isinstance(members, list) or len(members) != per_group:
                raise ValueError(f"{group_path}: wrong negative_samples coverage")
            baseline = parent / "baseline" / "input"
            baseline_cards = input_card_hashes(baseline)
            if not baseline_cards or "LF.L2" not in baseline_cards:
                raise ValueError(f"{baseline}: missing baseline input cards")
            adapter = CaseAdapter(baseline)
            if adapter.n != expected_buses:
                raise ValueError(f"{baseline}: bus count differs")
            baseline_l2 = (baseline / "LF.L2").read_bytes()
            policy = adapter.proposal()
            group_row = {
                "grid_family_id": family,
                "group_id": group_id,
                "baseline_hash_declared": group["baseline_hash"],
                "baseline_input_cards": baseline_cards,
                "group_manifest_sha256": file_hash(group_path),
                "bus_count": adapter.n,
                "physical_edge_count": len(adapter.rows),
                "candidate_device_count": len(policy["devices"]),
                "policy_proposal": policy,
                "case_count": per_group,
            }
            group_rows.append(group_row)
            listed = {item["sample_id"]: item for item in members}
            expected_ids = {f"{group_id}_neg_{i:03d}" for i in range(1, per_group + 1)}
            if set(listed) != expected_ids:
                raise ValueError(f"{group_path}: missing/duplicate sample IDs")
            dirs = {p.name for p in (parent / "negatives").iterdir() if p.is_dir()}
            expected_dirs = {f"neg_{i:03d}" for i in range(1, per_group + 1)}
            if dirs != expected_dirs:
                raise ValueError(f"{parent}: missing/extra sample directories")
            for index in range(1, per_group + 1):
                sample_id = f"{group_id}_neg_{index:03d}"
                sample = parent / "negatives" / f"neg_{index:03d}"
                manifest_path = sample / "sample_manifest.json"
                manifest = load_json(manifest_path)
                listing = listed[sample_id]
                if (manifest.get("sample_id") != sample_id or manifest.get("group_id") != group_id
                        or manifest.get("baseline_hash") != group["baseline_hash"]
                        or manifest.get("input_hash") != listing.get("input_hash")
                        or manifest.get("accepted") is not True
                        or manifest.get("solver_status") != "NONCONVERGED"):
                    raise ValueError(f"{manifest_path}: identity or label mismatch")
                checks = manifest.get("checks", {})
                if checks.get("mark_only") is not True:
                    raise ValueError(f"{manifest_path}: failed mark_only")
                # g016-g020 include a historical expansion whose equivalent
                # restoration assertion was named restore_exact. The actual
                # non-L2 card equality is independently recomputed below.
                if checks.get("other_inputs_identical") is False:
                    raise ValueError(f"{manifest_path}: failed other_inputs_identical")
                if (checks.get("restore_input_identical") is not True
                        and checks.get("restore_exact") is not True):
                    raise ValueError(f"{manifest_path}: missing successful restoration evidence")
                result_path = sample / "result" / "run_result.json"
                result = load_json(result_path)
                if (result.get("solver_status") != "NONCONVERGED" or result.get("cal_flag") != -1
                        or result.get("returncode") != 0 or result.get("diagnostic_only") is not False
                        or result.get("executable_sha256") != SOLVER_SHA256
                        or result.get("input_mutations") != []):
                    raise ValueError(f"{result_path}: invalid PSASP label evidence")
                case = sample / "input"
                cards = input_card_hashes(case)
                reported = result.get("input_hashes", {})
                if not cards or any(reported.get(key) != value for key, value in cards.items()):
                    raise ValueError(f"{case}: input differs from solver record")
                if set(reported) != {p.name for p in case.iterdir() if p.is_file()}:
                    raise ValueError(f"{case}: solver record misses an input file")
                # input_card_hashes covers the model/solver contract; verify
                # remaining files against the run record too.
                for key in reported.keys() - cards.keys():
                    if file_hash(case / key) != reported[key]:
                        raise ValueError(f"{case / key}: input differs from solver record")
                if {key: value for key, value in cards.items() if key != "LF.L2"} != {
                        key: value for key, value in baseline_cards.items() if key != "LF.L2"}:
                    raise ValueError(f"{case}: non-L2 input differs from baseline")
                actual_changes = difference(baseline_l2, (case / "LF.L2").read_bytes())
                claimed_changes = manifest.get("changes")
                diff_path = sample / "input_diff.json"
                if claimed_changes != load_json(diff_path):
                    raise ValueError(f"{diff_path}: differs from sample manifest")
                if sorted((x["line"], x["before"], x["after"]) for x in claimed_changes) != sorted(
                        (x["line"], x["before"], x["after"]) for x in actual_changes):
                    raise ValueError(f"{case}: L2 changes do not match manifest")
                fingerprint = hashlib.sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest()
                if fingerprint in all_fingerprints:
                    raise ValueError(f"{case}: duplicate input with {all_fingerprints[fingerprint]}")
                all_fingerprints[fingerprint] = sample_id
                if sample_id in seen_ids:
                    raise ValueError(f"{case}: duplicate sample ID")
                seen_ids.add(sample_id)
                selected_cases.append({
                    "case_id": sample_id,
                    "grid_family_id": family,
                    "parent_case_id": group_id,
                    "source": str(case.resolve()),
                    "input_cards": cards,
                    "input_fingerprint": fingerprint,
                    "sample_manifest_sha256": file_hash(manifest_path),
                    "solver_record_sha256": file_hash(result_path),
                    "baseline_hash_declared": group["baseline_hash"],
                    "changed_devices": len(actual_changes),
                    "flow_label": "nonconverged",
                    "label_basis": "generator_record_verified_against_current_inputs",
                    "policy_group_id": group_id,
                })
            if progress:
                print(f"audited {family}/{group_id}: {per_group} samples", flush=True)
        forced_train = (frozenset(f"case3120sp_g{i:03d}" for i in range(16, 21))
                        if family == "grid_3120" else
                        frozenset({"case118_g001"}) if family == "grid_118" else frozenset())
        split_map = group_splits(family, [
            {"group_id": row["group_id"], "baseline_hash": row["baseline_hash_declared"]}
            for row in group_rows], quotas, forced_train)
        for row in group_rows:
            row["split"] = split_map[row["group_id"]]
        for row in selected_cases:
            if row["grid_family_id"] == family:
                row["split"] = split_map[row["parent_case_id"]]
        selected_groups[family] = group_rows
    return {
        "schema": 2,
        "status": "development_inventory_verified_requires_control_authorization_and_training_release",
        "split_protocol": "multi_family_joint_training_v1",
        "split_seed": SEED,
        "training_family_ids": ["grid_2000", *FAMILIES],
        "excluded_group_ids": ["case3120sp_g010"],
        "excluded_family_ids": ["grid_2869"],
        "solver_sha256": SOLVER_SHA256,
        "groups": selected_groups,
        "cases": selected_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = scan(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "cases": len(payload["cases"]),
        "groups": {family: len(groups) for family, groups in payload["groups"].items()},
        "status": payload["status"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
