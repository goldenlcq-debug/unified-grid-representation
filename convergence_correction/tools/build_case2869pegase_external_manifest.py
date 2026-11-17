"""Independently audit and seal the never-trained case2869pegase external test."""
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
DATA = ROOT / "跨网测试数据/case2869pegase"
PUBLICATION = ROOT / "artifacts/runs/multifamily-joint-v1-seed0-20260923/selected.json"
SOLVER = ROOT / "PSASP/PSASP/WMLFRTMsg.exe"
TRAIN = ROOT / "artifacts/datasets/multifamily-joint-v1/train-validation-manifest.json"
SEEN = ROOT / "artifacts/datasets/multifamily-joint-v1/seen-family-test-sealed.json"
FAMILY = "grid_2869_pegase_never_trained"
GROUPS = tuple(f"g{i:03d}" for i in range(16, 21))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def input_hashes(directory: Path) -> dict[str, str]:
    return {item.name: file_hash(item) for item in directory.iterdir() if item.is_file()}


def fingerprint(cards: dict) -> str:
    return hashlib.sha256(json.dumps(cards, sort_keys=True).encode()).hexdigest()


def check_run(path: Path, *, converged: bool, solver_sha: str,
              expected_inputs: dict[str, str], run_id: str | None = None) -> str:
    record = read_json(path)
    require(record.get("solver_status") == ("CONVERGED" if converged else "NONCONVERGED")
            and record.get("cal_flag") == (1 if converged else -1)
            and record.get("returncode") == 0
            and record.get("executable_sha256") == solver_sha
            and record.get("input_mutations") == []
            and record.get("input_hashes") == expected_inputs,
            f"Solver evidence differs: {path}")
    if run_id is not None:
        require(record.get("run_id") == run_id, f"Run identity differs: {path}")
    for filename in ("LF.CAL", "lfreport.lis"):
        info = record.get("output_files", {}).get(filename)
        stored = path.parent / filename
        require(info is not None and stored.is_file()
                and file_hash(stored) == info.get("sha256")
                and stored.stat().st_size == info.get("size"),
                f"Stored solver output differs: {stored}")
    return file_hash(path)


def intensity(baseline: CaseAdapter, case: CaseAdapter) -> tuple[set[str], dict]:
    require(len(baseline.lines) == len(case.lines), "LF.L2 row count changed")
    off: set[str] = set()
    cap_base = 0.0
    cap_off = 0.0
    inductive_off = 0.0
    inductive_count = 0
    for old, new in zip(baseline.lines, case.lines):
        require(old.values[1:] == new.values[1:], "Non-Mark LF.L2 field changed")
        prior, current = int(old["Mark"]), int(new["Mark"])
        require(prior in (0, 1) and current in (0, 1)
                and not (prior == 0 and current == 1), "Wrong Mark direction")
        is_shunt = abs(int(old["I_No"])) == abs(int(old["J_No"]))
        require(prior == current or is_shunt, "Non-shunt Mark changed")
        if not is_shunt:
            continue
        x = float(old["X"])
        require(abs(x) > 1e-12, "Zero shunt X")
        b = -1.0 / x
        if prior == 1 and b > 0:
            cap_base += b
        if prior == 1 and current == 0:
            off.add(old.identifier)
            if b > 0:
                cap_off += b
            else:
                inductive_off += -b
                inductive_count += 1
    require(cap_base > 0, "No baseline capacitive support")
    return off, {"N_off": len(off), "B_cap_base": cap_base,
                 "B_cap_off": cap_off, "F_cap_off": cap_off / cap_base,
                 "N_inductive_off": inductive_count,
                 "B_inductive_off": inductive_off,
                 "inductive_removed": inductive_count > 0}


def build() -> dict:
    selected = read_json(PUBLICATION)
    require(selected.get("purpose") == "formal_multifamily_training"
            and selected.get("selection") == "family_macro_validation_only",
            "Wrong selected model")
    solver_sha = file_hash(SOLVER)
    require(file_hash(selected["model"]) == selected["model_sha256"]
            and solver_sha == selected["solver_sha256"],
            "Model or solver identity changed")
    old_ids: set[str] = set()
    old_fingerprints: set[str] = set()
    for path in (TRAIN, SEEN):
        for row in read_json(path)["cases"]:
            old_ids.add(row["case_id"])
            old_fingerprints.add(fingerprint(row["input_cards"]))
    require({item.name for item in DATA.iterdir() if item.is_dir()} == set(GROUPS),
            "Wrong Pegase parent group set")
    cases: list[dict] = []
    new_fingerprints: set[str] = set()
    group_evidence: dict[str, dict] = {}
    for group in GROUPS:
        parent = DATA / group
        group_manifest_path = parent / "group_manifest.json"
        group_manifest = read_json(group_manifest_path)
        require(group_manifest.get("group") == group
                and group_manifest.get("negative_count") == 100
                and group_manifest.get("validation_protocol") == "main_repeat_restore_v2"
                and group_manifest.get("self_loop_rows") == 80
                and group_manifest.get("LF_L6_unchanged") is True,
                f"Group identity or generation protocol differs: {group}")
        baseline_path = parent / "baseline/input"
        baseline_hashes = input_hashes(baseline_path)
        baseline_result_path = parent / "baseline/result/run_result.json"
        baseline_sha = check_run(baseline_result_path, converged=True,
                                 solver_sha=solver_sha, expected_inputs=baseline_hashes,
                                 run_id=group_manifest["baseline_run_id"])
        baseline = CaseAdapter(baseline_path)
        require(baseline.n == 2869, f"Wrong baseline bus count: {group}")
        proposal = baseline.proposal()
        names = {baseline.devices[item["key"]].name for item in proposal["devices"]}
        require(len(proposal["devices"]) == 80
                and names == set(group_manifest["same_80_ids"]),
                f"Physical SH whitelist differs: {group}")
        policy = {**proposal, "approved": True}
        baseline.authorize(policy)
        child_rows = group_manifest.get("children", [])
        children = {row["sample_id"]: row for row in child_rows}
        require(len(child_rows) == 100 and len(children) == 100,
                f"Wrong child commitment count: {group}")
        negative_root = parent / "negatives"
        expected_dirs = [f"neg_{i:03d}" for i in range(1, 101)]
        require(sorted(item.name for item in negative_root.iterdir() if item.is_dir())
                == expected_dirs, f"Wrong negative directories: {group}")
        group_evidence[group] = {"group_manifest_sha256": file_hash(group_manifest_path),
                                 "baseline_result_sha256": baseline_sha,
                                 "authorized_shunts": len(names)}
        for number, dirname in enumerate(expected_dirs, start=1):
            case_id = f"pegase80matched5_{group}_{dirname}"
            child = negative_root / dirname
            sample_path = child / "sample_manifest.json"
            sample = read_json(sample_path)
            require(sample.get("sample_id") == case_id
                    and sample.get("parent_group") == group
                    and sample.get("slot_id") == f"{group}_slot_{number:03d}"
                    and sample.get("accepted") is True
                    and sample.get("no_engineering_limit_gate") is True
                    and case_id not in old_ids,
                    f"Sample identity/approval differs: {case_id}")
            require(children[case_id]["input_hash"] == sample.get("input_hash"),
                    f"Group input commitment differs: {case_id}")
            source = child / "input"
            actual_hashes = input_hashes(source)
            case = CaseAdapter(source)
            require(case.n == 2869, f"Wrong sample bus count: {case_id}")
            case.authorize(policy)
            actual_off, measured = intensity(baseline, case)
            require(actual_off == set(sample["off_ids"])
                    and actual_off.issubset(names)
                    and {change["id_name"] for change in sample["changes"]} == actual_off
                    and len(sample["changes"]) == len(actual_off)
                    and all(change["before"] == 1 and change["after"] == 0
                            for change in sample["changes"]),
                    f"Actual SH Mark withdrawal differs: {case_id}")
            realized = sample["realized_intensity"]
            require(all(abs(measured[key] - realized[key]) < 1e-8 for key in
                        ("N_off", "B_cap_base", "B_cap_off", "F_cap_off",
                         "N_inductive_off", "B_inductive_off"))
                    and measured["inductive_removed"] == realized["inductive_removed"],
                    f"Independent intensity differs: {case_id}")
            target = sample["target"]
            require(target["group"] == group
                    and target["slot_id"] == sample["slot_id"]
                    and abs(measured["N_off"] - target["target_N_off"])
                    <= target["tolerance_N_off"]
                    and abs(measured["F_cap_off"] - target["target_F_cap_off"])
                    <= target["tolerance_F_cap_off"] + 1e-10
                    and measured["inductive_removed"] == target["target_inductive_removed"],
                    f"Matched target not met: {case_id}")
            main = check_run(child / "result/run_result.json", converged=False,
                             solver_sha=solver_sha, expected_inputs=actual_hashes,
                             run_id=sample["main_run_id"])
            repeat = check_run(child / "diagnostics/repeat/run_result.json", converged=False,
                               solver_sha=solver_sha, expected_inputs=actual_hashes,
                               run_id=sample["repeat_run_id"])
            restore = check_run(child / "diagnostics/restore/run_result.json", converged=True,
                                solver_sha=solver_sha, expected_inputs=baseline_hashes,
                                run_id=sample["restore_run_id"])
            cards = input_card_hashes(source)
            fp = fingerprint(cards)
            require(fp not in old_fingerprints and fp not in new_fingerprints,
                    f"Duplicate or leaked electrical input: {case_id}")
            new_fingerprints.add(fp)
            cases.append({"case_id": case_id, "grid_family_id": FAMILY,
                          "parent_case_id": group, "bus_count": 2869,
                          "split": "external_test", "source": str(source.resolve()),
                          "input_cards": cards, "policy": policy,
                          "changed_shunt_count": measured["N_off"],
                          "withdrawn_cap_fraction": measured["F_cap_off"],
                          "sample_manifest_sha256": file_hash(sample_path),
                          "main_evidence_sha256": main,
                          "repeat_evidence_sha256": repeat,
                          "restore_evidence_sha256": restore})
    require(len(cases) == 500, "Expected exactly 500 fully audited Pegase cases")
    return {"schema": 2, "split_protocol": "unseen_grid_external_test_v1",
            "phase": "external_test", "dataset_kind": "case2869pegase_never_trained_grid",
            "interpretation_boundary": "independent_never_trained_2869_bus_grid_not_ACTIVSg2020_derived",
            "selected_publication_sha256": file_hash(PUBLICATION),
            "selected_model_sha256": selected["model_sha256"],
            "solver_sha256": solver_sha,
            "training_manifest_sha256": file_hash(TRAIN),
            "sealed_seen_test_sha256": file_hash(SEEN),
            "group_evidence": group_evidence, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "Pegase external manifest already exists")
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
