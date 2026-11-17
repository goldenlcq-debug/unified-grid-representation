"""Independently audit and seal the ACTIVSg2020 matched-v3 external test."""
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
DATA = ROOT / "跨网测试数据/ACTIVSg2020_matched_v3"
PUBLICATION = ROOT / "artifacts/runs/multifamily-joint-v1-seed0-20260923/selected.json"
SOLVER = ROOT / "PSASP/PSASP/WMLFRTMsg.exe"
TRAIN = ROOT / "artifacts/datasets/multifamily-joint-v1/train-validation-manifest.json"
SEEN = ROOT / "artifacts/datasets/multifamily-joint-v1/seen-family-test-sealed.json"
INTENSITY = ROOT / "artifacts/results/activsg2020-matched-v3-independent-intensity-20260926.json"
SUPPLIED_INTENSITY = ROOT / "artifacts/results/activsg2020-matched-v3-intensity.json"
SCREEN_PLAN = ROOT / "artifacts/activsg2020_matched_v3_selection_v2/selection_screen_plan.json"
SCREEN_RESULTS = ROOT / "artifacts/activsg2020_matched_v3_selection_v2/selection_screen_results.json"
FAMILY = "grid_2020_activsg2000_derived"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def all_input_hashes(directory: Path) -> dict[str, str]:
    return {item.name: file_hash(item) for item in directory.iterdir() if item.is_file()}


def fingerprint(cards: dict) -> str:
    return hashlib.sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest()


def check_run(path: Path, expected_flag: int, solver_sha: str,
              expected_hashes: dict[str, str], expected_id: str | None = None) -> str:
    record = read_json(path)
    require(record.get("cal_flag") == expected_flag
            and record.get("return_code") == 0
            and record.get("solver_sha256") == solver_sha
            and record.get("input_unchanged") is True
            and record.get("input_hashes") == expected_hashes,
            f"Solver evidence differs: {path}")
    if expected_id is not None:
        require(record.get("run_id") == expected_id, f"Run identity differs: {path}")
    require(Path(record["fresh_output_audit"]).is_file(), f"Fresh-output audit missing: {path}")
    return file_hash(path)


def build() -> dict:
    selected = read_json(PUBLICATION)
    require(selected.get("purpose") == "formal_multifamily_training"
            and selected.get("selection") == "family_macro_validation_only",
            "Wrong selected model")
    solver_sha = file_hash(SOLVER)
    require(file_hash(selected["model"]) == selected["model_sha256"]
            and solver_sha == selected["solver_sha256"],
            "Model or solver identity changed")
    plan_path = DATA / "generation_plan.json"
    quality_path = DATA / "quality_gate.json"
    verification_path = DATA / "verification.json"
    plan, quality, verification = map(read_json, (plan_path, quality_path, verification_path))
    plan_sha = file_hash(plan_path)
    require(plan.get("target_parent_groups") == 60
            and plan.get("slots_per_parent") == 10
            and plan.get("target_negative_count") == 600
            and Path(plan["new_test_root"]).resolve() == DATA.resolve(),
            "Wrong generation plan coverage/root")
    require(plan.get("solver_sha256") == solver_sha
            and plan.get("training_manifest_sha256") == file_hash(TRAIN)
            and plan.get("selection_screen_plan_sha256") == file_hash(SCREEN_PLAN)
            and plan.get("selection_screen_results_sha256") == file_hash(SCREEN_RESULTS),
            "Generation plan dependency changed")
    require(quality.get("status") == "PASS" and verification.get("status") == "PASS"
            and quality.get("parent_groups") == 60 and quality.get("negative_cases") == 600
            and verification.get("parent_groups") == 60
            and verification.get("negative_cases") == 600
            and quality.get("plan_sha256") == plan_sha
            and verification.get("plan_sha256") == plan_sha
            and quality.get("verification_sha256") == file_hash(verification_path)
            and quality.get("intensity_report_sha256") == file_hash(SUPPLIED_INTENSITY),
            "Quality gate dependency or status differs")
    require(verification.get("only_authorized_shunt_mark_1_to_0") is True
            and verification.get("paired_intensity_targets_met") is True
            and verification.get("fresh_baseline_main_repeat_restore") is True
            and quality.get("exact_old_input_overlap") == 0
            and quality.get("new_full_inputs_unique") is True
            and quality.get("inductive_removed_category_pairwise_matched") is True,
            "Supplied quality checks are incomplete")
    independently_measured = read_json(INTENSITY)
    supplied = read_json(SUPPLIED_INTENSITY)
    require(independently_measured["training_grid2000"]["summary"]
            == supplied["training_grid2000"]["summary"]
            and independently_measured["candidate_2020"]["summary"]
            == supplied["candidate_2020"]["summary"],
            "Independent intensity summary differs")
    measured_by_id = {item["case_id"]: item
                      for item in independently_measured["candidate_2020"]["cases"]}
    require(len(measured_by_id) == 600, "Wrong independent intensity coverage")
    old_ids: set[str] = set()
    old_inputs: set[str] = set()
    for path in (TRAIN, SEEN):
        for item in read_json(path)["cases"]:
            old_ids.add(item["case_id"])
            old_inputs.add(fingerprint(item["input_cards"]))
    slots = {item["slot_id"]: item for item in plan["slots"]}
    require(len(slots) == 600 and len(plan["slots"]) == 600,
            "Duplicate or missing plan slots")
    groups = sorted((item for item in DATA.iterdir() if item.is_dir()),
                    key=lambda item: int(item.name.removeprefix("sample_")))
    require(len(groups) == 60 and {item.name for item in groups} == set(plan["parents"]),
            "Parent set differs from plan")
    cases: list[dict] = []
    new_inputs: set[str] = set()
    group_evidence: dict[str, dict] = {}
    for group in groups:
        group_manifest_path = group / "group_manifest.json"
        group_manifest = read_json(group_manifest_path)
        require(group_manifest.get("parent") == group.name
                and group_manifest.get("plan_sha256") == plan_sha
                and group_manifest.get("solver_sha256") == solver_sha,
                f"Group identity/dependencies differ: {group.name}")
        expected_ids = [f"{group.name}_neg_{number:03d}" for number in range(1, 11)]
        require(group_manifest.get("sample_ids") == expected_ids,
                f"Group case IDs differ: {group.name}")
        baseline = group / "baseline/input"
        baseline_hashes = all_input_hashes(baseline)
        require(baseline_hashes == group_manifest.get("baseline_input_hashes"),
                f"Baseline input changed: {group.name}")
        baseline_result = group / "baseline/result/run_result.json"
        baseline_result_sha = check_run(baseline_result, 1, solver_sha, baseline_hashes)
        baseline_adapter = CaseAdapter(baseline)
        require(baseline_adapter.n == 2020, f"Baseline bus count differs: {group.name}")
        proposal = baseline_adapter.proposal()
        names = {baseline_adapter.devices[item["key"]].name for item in proposal["devices"]}
        require(len(names) == 171 and names == set(plan["authorized_shunt_names"]),
                f"Authorized SH set differs: {group.name}")
        policy = {**proposal, "approved": True}
        baseline_adapter.authorize(policy)
        negatives = group / "negatives"
        expected_dirs = [f"neg_{number:03d}" for number in range(1, 11)]
        require(sorted(item.name for item in negatives.iterdir() if item.is_dir()) == expected_dirs,
                f"Wrong negative directories: {group.name}")
        group_evidence[group.name] = {"group_manifest_sha256": file_hash(group_manifest_path),
                                      "baseline_result_sha256": baseline_result_sha,
                                      "authorized_shunts": len(names)}
        for number, directory_name in enumerate(expected_dirs, start=1):
            case_id = expected_ids[number - 1]
            slot_id = f"{group.name}_slot_{number:02d}"
            child = negatives / directory_name
            sample_path = child / "sample_manifest.json"
            sample = read_json(sample_path)
            target = slots[slot_id]
            require(case_id not in old_ids and sample.get("accepted") is True
                    and sample.get("slot_id") == slot_id
                    and sample.get("target") == target,
                    f"Sample plan/identity differs: {case_id}")
            measured = measured_by_id[case_id]
            realized = sample["realized_intensity"]
            require(measured["off_count"] == realized["N_off"]
                    and abs(measured["withdrawn_cap_fraction"] - realized["F_cap_off"]) < 1e-8
                    and abs(measured["withdrawn_cap_proxy"] - realized["B_cap_off"]) < 1e-8
                    and abs(measured["baseline_cap_proxy"] - realized["B_cap_base"]) < 1e-8,
                    f"Independent intensity differs: {case_id}")
            require(abs(realized["N_off"] - target["target_N_off"])
                    <= target["tolerance_N_off"]
                    and abs(realized["F_cap_off"] - target["target_F_cap_off"])
                    <= target["tolerance_F_cap_off"] + 1e-10
                    and realized["N_off"] <= 147
                    and (realized["N_inductive_off"] > 0)
                    == target["target_inductive_removed"],
                    f"Matched strength target not met: {case_id}")
            source = child / "input"
            input_hashes = all_input_hashes(source)
            require(input_hashes.get("LF.L2") == sample.get("input_l2_sha256"),
                    f"Sample L2 hash changed: {case_id}")
            adapter = CaseAdapter(source)
            require(adapter.n == 2020, f"Wrong sample bus count: {case_id}")
            adapter.authorize(policy)
            actual_off = {base.identifier for base, changed in
                          zip(baseline_adapter.lines, adapter.lines)
                          if int(base["Mark"]) == 1 and int(changed["Mark"]) == 0}
            require(actual_off == set(sample["off_ids"])
                    and len(actual_off) == realized["N_off"]
                    and actual_off.issubset(names),
                    f"Actual SH withdrawal differs from sample record: {case_id}")
            main_path = child / "result/run_result.json"
            repeat_path = child / "diagnostics/repeat/run_result.json"
            restore_path = child / "diagnostics/restore/run_result.json"
            main_sha = check_run(main_path, -1, solver_sha, input_hashes,
                                 sample["main_run_id"])
            repeat_sha = check_run(repeat_path, -1, solver_sha, input_hashes,
                                   sample["repeat_run_id"])
            restore_sha = check_run(restore_path, 1, solver_sha, baseline_hashes,
                                    sample["restore_run_id"])
            cards = input_card_hashes(source)
            fp = fingerprint(cards)
            require(fp not in old_inputs and fp not in new_inputs,
                    f"Duplicate or previously used electrical input: {case_id}")
            new_inputs.add(fp)
            cases.append({"case_id": case_id, "grid_family_id": FAMILY,
                          "parent_case_id": group.name, "bus_count": 2020,
                          "split": "external_test", "source": str(source.resolve()),
                          "input_cards": cards, "policy": policy,
                          "changed_shunt_count": realized["N_off"],
                          "withdrawn_cap_fraction": realized["F_cap_off"],
                          "reference_case_id": target["reference_case_id"],
                          "sample_manifest_sha256": file_hash(sample_path),
                          "main_evidence_sha256": main_sha,
                          "repeat_evidence_sha256": repeat_sha,
                          "restore_evidence_sha256": restore_sha})
    require(len(cases) == 600, "Expected exactly 600 fully audited cases")
    return {"schema": 2, "split_protocol": "unseen_grid_external_test_v1",
            "phase": "external_test", "dataset_kind": "ACTIVSg2000_derived_2020_bus_matched_v3",
            "interpretation_boundary": "post_hoc_intensity_matched_derived_grid_not_unrelated_topology",
            "parent_selection_note": "60 of 188 candidate parents selected by precommitted vulnerability screen; not a random parent sample",
            "selected_publication_sha256": file_hash(PUBLICATION),
            "selected_model_sha256": selected["model_sha256"],
            "solver_sha256": solver_sha,
            "training_manifest_sha256": file_hash(TRAIN),
            "sealed_seen_test_sha256": file_hash(SEEN),
            "generation_plan_sha256": plan_sha,
            "quality_gate_sha256": file_hash(quality_path),
            "verification_sha256": file_hash(verification_path),
            "independent_intensity_sha256": file_hash(INTENSITY),
            "group_evidence": group_evidence, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "Matched-v3 external manifest already exists")
    manifest = build()
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
