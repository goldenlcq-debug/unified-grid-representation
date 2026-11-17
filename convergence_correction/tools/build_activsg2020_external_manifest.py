"""Seal and audit the 600-case ACTIVSg2020 derived-grid external test set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rl_poweradj.case_adapter import CaseAdapter
from rl_poweradj.data_preflight import input_card_hashes
from rl_poweradj.multifamily_external import audit_external_manifest
from rl_poweradj.run_store import atomic_json, file_hash


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "跨网测试数据" / "ACTIVSg2020"
PUBLICATION = ROOT / "artifacts/runs/multifamily-joint-v1-seed0-20260923/selected.json"
SOLVER = ROOT / "PSASP/PSASP/WMLFRTMsg.exe"
TRAIN_MANIFEST = ROOT / "artifacts/datasets/multifamily-joint-v1/train-validation-manifest.json"
SEEN_TEST = ROOT / "artifacts/datasets/multifamily-joint-v1/seen-family-test-sealed.json"
FAMILY = "grid_2020_activsg2000_derived"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def hash_all_input_files(directory: Path) -> dict[str, str]:
    return {path.name: file_hash(path) for path in directory.iterdir() if path.is_file()}


def fingerprint(cards: dict) -> str:
    return hashlib.sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest()


def build_manifest() -> dict:
    selected = read_json(PUBLICATION)
    require(selected.get("purpose") == "formal_multifamily_training", "Wrong model purpose")
    require(selected.get("selection") == "family_macro_validation_only", "Wrong model selection")
    require(selected.get("training_bus_counts") == [118, 300, 2000, 3120],
            "Wrong published training bus counts")
    require(file_hash(selected["model"]) == selected["model_sha256"], "Selected model changed")
    require(file_hash(SOLVER) == selected["solver_sha256"], "Solver changed")
    old_ids: set[str] = set()
    old_fingerprints: set[str] = set()
    for path in (TRAIN_MANIFEST, SEEN_TEST):
        for row in read_json(path)["cases"]:
            old_ids.add(row["case_id"])
            old_fingerprints.add(fingerprint(row["input_cards"]))

    groups = sorted((item for item in DATA.iterdir() if item.is_dir()),
                    key=lambda path: int(path.name.removeprefix("sample_")))
    require(len(groups) == 60 and len({item.name for item in groups}) == 60,
            "Expected exactly 60 parent groups")
    require(all(item.name.startswith("sample_") and item.name[7:].isdigit() for item in groups),
            "Unrecognized parent directory")
    cases: list[dict] = []
    group_evidence: dict[str, dict] = {}
    fingerprints: set[str] = set()
    for group in groups:
        group_path = group / "group_manifest.json"
        info = read_json(group_path)
        require(info.get("parent") == group.name and info.get("accepted_count") == 10,
                f"Group manifest identity/count differs: {group.name}")
        require(info.get("baseline_solver_status") == "CONVERGED"
                and info.get("baseline_solver_executable_sha256") == selected["solver_sha256"],
                f"Baseline evidence differs: {group.name}")
        baseline_input = group / "baseline/input"
        baseline_result_path = group / "baseline/result/solver_run_result.json"
        baseline_result = read_json(baseline_result_path)
        require(baseline_result.get("solver_status") == "CONVERGED"
                and baseline_result.get("cal_flag") == 1
                and baseline_result.get("returncode") == 0
                and baseline_result.get("executable_sha256") == selected["solver_sha256"],
                f"Baseline not independently converged: {group.name}")
        baseline_hashes = hash_all_input_files(baseline_input)
        require(baseline_hashes == info.get("baseline_input_hashes")
                and baseline_hashes == baseline_result.get("input_hashes"),
                f"Baseline input changed: {group.name}")
        baseline_adapter = CaseAdapter(baseline_input)
        require(baseline_adapter.n == 2020, f"Wrong baseline bus count: {group.name}")
        proposal = baseline_adapter.proposal()
        require(proposal["devices"], f"No controllable shunts: {group.name}")
        names = {baseline_adapter.devices[item["key"]].name for item in proposal["devices"]}
        whitelist = set(info.get("mark_only_whitelist", []))
        require(names == whitelist and len(names) == len(proposal["devices"]),
                f"SH authorization differs from group whitelist: {group.name}")
        policy = {**proposal, "approved": True}
        baseline_adapter.authorize(policy)
        children = {item["sample_id"]: item for item in info.get("children", [])}
        require(len(children) == 10 and len(info["children"]) == 10,
                f"Wrong child manifest count: {group.name}")
        negative_dir = group / "negatives"
        expected = [f"neg_{i:03d}" for i in range(1, 11)]
        actual = sorted(item.name for item in negative_dir.iterdir() if item.is_dir())
        require(actual == expected, f"Wrong negative directory set: {group.name}")
        group_evidence[group.name] = {"group_manifest_sha256": file_hash(group_path),
                                      "baseline_run_sha256": file_hash(baseline_result_path),
                                      "shunt_candidates": len(names)}
        for name in expected:
            case_id = f"{group.name}_{name}"
            require(case_id not in old_ids, f"Case ID leaked from training or seen test: {case_id}")
            child = negative_dir / name
            sample_path = child / "sample_manifest.json"
            result_path = child / "result/run_result.json"
            sample = read_json(sample_path)
            result = read_json(result_path)
            require(sample.get("accepted") is True and sample.get("main_cal_flag") == -1
                    and sample.get("repeat_cal_flag") == -1
                    and sample.get("validation_protocol")
                    == "independent_main_and_repeat_nonconvergence_v1",
                    f"Nonconvergence confirmation missing: {case_id}")
            require(result.get("solver_status") == "NONCONVERGED"
                    and result.get("cal_flag") == -1 and result.get("returncode") == 0
                    and result.get("executable_sha256") == selected["solver_sha256"]
                    and result.get("input_mutations") == [],
                    f"Primary solver label differs: {case_id}")
            changes = sample.get("changes", [])
            off_ids = sample.get("off_ids", [])
            require(changes and len(changes) == len(off_ids)
                    and {item["id_name"] for item in changes} == set(off_ids)
                    and set(off_ids).issubset(whitelist)
                    and all(item["before"] == 1 and item["after"] == 0 for item in changes),
                    f"Disturbance is not SH Mark 1-to-0 only: {case_id}")
            require(children[case_id]["input_hash"] == sample.get("input_hash"),
                    f"Child commitment differs: {case_id}")
            source = child / "input"
            actual_hashes = hash_all_input_files(source)
            require(actual_hashes == sample.get("input_hashes")
                    and actual_hashes == result.get("input_hashes"),
                    f"Input content changed: {case_id}")
            cards = input_card_hashes(source)
            fp = fingerprint(cards)
            require(fp not in old_fingerprints and fp not in fingerprints,
                    f"Duplicate or previously used electrical input: {case_id}")
            fingerprints.add(fp)
            adapter = CaseAdapter(source)
            require(adapter.n == 2020, f"Wrong sample bus count: {case_id}")
            adapter.authorize(policy)
            cases.append({"case_id": case_id, "grid_family_id": FAMILY,
                          "parent_case_id": group.name, "bus_count": 2020,
                          "split": "external_test", "source": str(source.resolve()),
                          "input_cards": cards, "policy": policy,
                          "sample_manifest_sha256": file_hash(sample_path),
                          "label_evidence_sha256": file_hash(result_path),
                          "changed_shunt_count": len(changes)})
    require(len(cases) == 600, "Expected exactly 600 negative samples")
    return {"schema": 2, "split_protocol": "unseen_grid_external_test_v1",
            "phase": "external_test", "dataset_kind": "ACTIVSg2000_derived_2020_bus_grid",
            "interpretation_boundary": "derived_grid_same_lineage_not_unrelated_topology",
            "selected_publication_sha256": file_hash(PUBLICATION),
            "selected_model_sha256": selected["model_sha256"],
            "solver_sha256": selected["solver_sha256"],
            "train_manifest_sha256": file_hash(TRAIN_MANIFEST),
            "sealed_seen_test_sha256": file_hash(SEEN_TEST),
            "group_evidence": group_evidence, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "External manifest already exists")
    manifest = build_manifest()
    atomic_json(output, manifest)
    selected = read_json(PUBLICATION)
    audited = audit_external_manifest(
        output, training_families=selected["training_family_ids"],
        training_bus_counts=selected["training_bus_counts"])
    print(json.dumps({"manifest": str(output), "cases": len(audited["cases"]),
                      "families": audited["families"], "bus_counts": audited["bus_counts"],
                      "manifest_sha256": file_hash(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
