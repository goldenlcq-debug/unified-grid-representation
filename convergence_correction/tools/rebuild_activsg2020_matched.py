"""Frozen intensity targets and staged PSASP pilot for ACTIVSg2020 matched_v1.

This is data preparation only. It never loads or queries the trained policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from rl_poweradj.case_adapter import CaseAdapter
from rl_poweradj.psasp import PSASPRunner, read_records
from rl_poweradj.run_store import atomic_json, file_hash

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "artifacts/datasets/multifamily-joint-v1/train-validation-manifest.json"
BASELINE_REPORT = ROOT / "artifacts/results/activsg2020-intensity-baseline-20260925.json"
PAUSED_RESULT = ROOT / "artifacts/results/external-activsg2020-seed0-20260925.json"
TEST_ROOT = ROOT / "跨网测试数据"
OLD = TEST_ROOT / "ACTIVSg2020"
SOURCE188 = Path(r"D:\Code\样本\ACTIVSg2020\lfsample-upgraded")
VERSION = os.environ.get("ACTIVSG2020_MATCHED_VERSION", "ACTIVSg2020_matched_v1")
assert VERSION in ("ACTIVSg2020_matched_v1", "ACTIVSg2020_matched_v2", "ACTIVSg2020_matched_v3")
NEW = TEST_ROOT / VERSION
BUILD = ROOT / "artifacts" / (VERSION.lower() + "_build")
SELECTION_VERSION = os.environ.get("ACTIVSG2020_SELECTION_VERSION", "v1")
assert SELECTION_VERSION in ("v1", "v2")
SELECTION_BUILD = ROOT / "artifacts" / (
    "activsg2020_matched_v3_selection" if SELECTION_VERSION == "v1"
    else "activsg2020_matched_v3_selection_v2")
SOLVER = ROOT / "PSASP/PSASP/WMLFRTMsg.exe"
SEED = 20260925
INPUT_FILES = frozenset(("DATALIB.DAT", "PHYBUS.LFI", "PSASPDATA.INI", "LF.AREA", "LF.BUS",
                         "LF.L0", "LF.L1", "LF.L2", "LF.L20", "LF.L3", "LF.L5", "LF.L6",
                         "LF.L7", "LF.ML4", "LF.PSTATION", "LF.UDP", "LF.ZONE"))


def old_tree_digest(path: Path) -> tuple[str, int, int]:
    state = hashlib.sha256()
    count = size = 0
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(path).as_posix()
        digest = file_hash(item)
        state.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
        count += 1
        size += item.stat().st_size
    return state.hexdigest(), count, size


def parent_paths() -> list[Path]:
    if VERSION.endswith("v1"):
        paths = sorted((p for p in OLD.iterdir() if p.is_dir()),
                       key=lambda p: int(p.name.removeprefix("sample_")))
    elif VERSION.endswith("v2"):
        available = sorted((p for p in SOURCE188.iterdir() if p.is_dir()),
                           key=lambda p: int(p.name.removeprefix("sample_")))
        assert len(available) == 188
        ranked = sorted((p for p in available if p.name != "sample_1"),
                        key=lambda p: (sum(int(row[0]) == 1 and int(row[3]) == -1
                                           for row in read_records(p / "LF.L5")),
                                       int(p.name.removeprefix("sample_"))))
        selected = [SOURCE188 / "sample_1", *ranked[:59]]
        paths = sorted(selected, key=lambda p: int(p.name.removeprefix("sample_")))
    else:
        assert SELECTION_VERSION == "v2"
        screen_plan = json.loads((SELECTION_BUILD / "selection_screen_plan.json").read_text(encoding="utf-8"))
        screen = json.loads((SELECTION_BUILD / "selection_screen_results.json").read_text(encoding="utf-8"))
        assert len(screen["parents"]) == 188
        assert screen["plan_sha256"] == file_hash(SELECTION_BUILD / "selection_screen_plan.json")
        ids = [x["pattern_id"] for x in screen_plan["patterns"]]
        assert len(ids) == 11 and all(set(x) == set(ids) for x in screen["parents"].values())
        ranked = []
        for name, outcomes in screen["parents"].items():
            if name == "sample_1":
                continue
            positives = sum(outcomes[k]["cal_flag"] == -1 for k in ids)
            if positives == 0:
                continue
            parent = SOURCE188 / name
            pv = sum(int(row[0]) == 1 and int(row[3]) == -1
                     for row in read_records(parent / "LF.L5"))
            ranked.append((name, positives, outcomes[ids[0]]["cal_flag"] == -1, pv))
        ranked.sort(key=lambda x: (-x[1], -int(x[2]), x[3], int(x[0].removeprefix("sample_"))))
        assert len(ranked) >= 59
        chosen = [SOURCE188 / "sample_1", *(SOURCE188 / x[0] for x in ranked[:59])]
        paths = sorted(chosen, key=lambda p: int(p.name.removeprefix("sample_")))
    assert len(paths) == 60
    assert all(p.name.startswith("sample_") and p.name[7:].isdigit() for p in paths)
    if VERSION.endswith("v1"):
        assert not {"sample_7", "sample_8", "sample_9"}.intersection(p.name for p in paths)
        for p in paths:
            assert (p / "baseline/input/LF.L2").is_file()
            assert len([q for q in (p / "negatives").iterdir() if q.is_dir()]) == 10
    else:
        for p in paths:
            assert (p / "LF.L2").is_file()
            record = json.loads((p / "solver_run_result.json").read_text(encoding="utf-8"))
            assert record["solver_status"] == "CONVERGED" and record["cal_flag"] == 1
    return paths


def parent_input_dir(parent_id: str) -> Path:
    return ((OLD / parent_id / "baseline/input") if VERSION.endswith("v1")
            else (SOURCE188 / parent_id))


def previous_l2_hashes(parent_id: str) -> set[str]:
    group = OLD / parent_id / "negatives"
    return ({file_hash(p / "input/LF.L2") for p in group.iterdir() if p.is_dir()}
            if group.is_dir() else set())


def plan() -> dict:
    if NEW.exists():
        existing = NEW / "generation_plan.json"
        if not existing.is_file():
            raise FileExistsError(f"Destination exists without frozen plan: {NEW}")
        value = json.loads(existing.read_text(encoding="utf-8"))
        assert value["version"] == VERSION
        return value
    paths = parent_paths()
    manifest_hash = file_hash(TRAIN)
    formal = json.loads(TRAIN.read_text(encoding="utf-8"))
    references = {r["case_id"]: r for r in formal["cases"]
                  if r["grid_family_id"] == "grid_2000" and r["split"] == "train"}
    assert len(references) == 2416
    baseline_report = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    assert baseline_report["training_manifest_sha256"] == manifest_hash
    training = baseline_report["training_grid2000"]["cases"]
    assert len(training) == 2416
    assert {r["case_id"] for r in training} == set(references)
    ranked = sorted(training, key=lambda r: (r["withdrawn_cap_fraction"], r["case_id"]))
    rng = random.Random(SEED)
    layers: list[dict[str, dict]] = []
    predecessor_hash = None
    if VERSION.endswith("v1"):
        for layer in range(10):
            cohort = ranked[(layer * len(ranked)) // 10:((layer + 1) * len(ranked)) // 10]
            # Two full-capacitor-off outliers need not be selected to reproduce
            # 2,416-case quantiles in a 600-case comparison.
            eligible = [r for r in cohort if r["withdrawn_cap_fraction"] < 1 - 1e-10]
            assert len(eligible) >= 60
            selected = rng.sample(eligible, 60)
            rng.shuffle(selected)
            layers.append({p.name: r for p, r in zip(paths, selected)})
    else:
        predecessor = TEST_ROOT / "ACTIVSg2020_matched_v1/generation_plan.json"
        predecessor_hash = file_hash(predecessor)
        earlier = json.loads(predecessor.read_text(encoding="utf-8"))
        assert earlier["version"] == "ACTIVSg2020_matched_v1"
        by_id = {r["case_id"]: r for r in training}
        if VERSION.endswith("v2"):
            others = sorted((p for p in paths if p.name != "sample_1"),
                            key=lambda p: (sum(int(row[0]) == 1 and int(row[3]) == -1
                                               for row in read_records(p / "LF.L5")),
                                           int(p.name.removeprefix("sample_"))))
        else:
            results = json.loads((SELECTION_BUILD / "selection_screen_results.json").read_text(encoding="utf-8"))["parents"]
            ids = [x["pattern_id"] for x in json.loads(
                (SELECTION_BUILD / "selection_screen_plan.json").read_text(encoding="utf-8"))["patterns"]]
            others = sorted((p for p in paths if p.name != "sample_1"),
                            key=lambda p: (-sum(results[p.name][k]["cal_flag"] == -1 for k in ids),
                                           -int(results[p.name][ids[0]]["cal_flag"] == -1),
                                           sum(int(row[0]) == 1 and int(row[3]) == -1
                                               for row in read_records(p / "LF.L5")),
                                           int(p.name.removeprefix("sample_"))))
        for layer in range(1, 11):
            old_layer = [s for s in earlier["slots"] if s["strength_layer"] == layer]
            assert len(old_layer) == 60
            pilot = next(s for s in old_layer if s["parent_id"] == "sample_1")
            remaining = sorted((by_id[s["reference_case_id"]] for s in old_layer
                                if s["parent_id"] != "sample_1"),
                               key=lambda r: (r["withdrawn_cap_fraction"], r["case_id"]))
            layers.append({"sample_1": by_id[pilot["reference_case_id"]],
                           **{p.name: r for p, r in zip(others, remaining)}})
    slots = []
    for parent_index, parent in enumerate(paths):
        for layer, selected in enumerate(layers):
            reference = selected[parent.name]
            source = references[reference["case_id"]]
            n_off = int(reference["off_count"])
            fraction = float(reference["withdrawn_cap_fraction"])
            assert 61 <= n_off <= 147 and 0 < fraction < 1
            slots.append({
                "slot_id": f"{parent.name}_slot_{layer + 1:02d}",
                "parent_id": parent.name,
                "slot_number": layer + 1,
                "strength_layer": layer + 1,
                "reference_case_id": reference["case_id"],
                "reference_parent_case_id": source["parent_case_id"],
                "reference_input_dir": source["source"],
                "target_N_off": n_off,
                "target_F_cap_off": fraction,
                "target_removed_inductive_proxy": float(reference["removed_inductive_proxy"]),
                "target_inductive_removed": reference["removed_inductive_proxy"] > 1e-12,
                "tolerance_N_off": 2,
                "tolerance_F_cap_off": 0.02,
                "seed": SEED + parent_index * 1000 + layer + 1,
            })
    assert len(slots) == 600 and len({x["reference_case_id"] for x in slots}) == 600
    adapter = CaseAdapter(parent_input_dir(paths[0].name))
    baseline_info = []
    first_l2_hash = file_hash(parent_input_dir(paths[0].name) / "LF.L2")
    for parent in paths:
        parent_input = parent_input_dir(parent.name)
        items = sorted(p for p in parent_input.iterdir() if p.is_file() and p.name in INPUT_FILES)
        assert len(items) == 17
        if VERSION.endswith("v1"):
            source_manifest = json.loads((parent / "group_manifest.json").read_text(encoding="utf-8"))
            assert source_manifest["parent"] == parent.name
            assert set(source_manifest["mark_only_whitelist"]) == {d.name for d in adapter.devices.values()}
        else:
            source_result = json.loads((parent / "solver_run_result.json").read_text(encoding="utf-8"))
            assert source_result["input_hashes"] == {p.name: file_hash(p) for p in items}
        assert file_hash(parent_input / "LF.L2") == first_l2_hash
        baseline_info.append({"parent": parent.name, "input_hashes":
                              {p.name: file_hash(p) for p in items}})
    old_hash, old_files, old_bytes = old_tree_digest(OLD)
    payload = {
        "version": VERSION, "created_for": "ACTIVSg2000-derived 2020-bus intensity-matched test",
        "seed": SEED, "target_parent_groups": 60, "slots_per_parent": 10,
        "target_negative_count": 600, "original_test_root": str(OLD),
        "new_test_root": str(NEW), "training_manifest": str(TRAIN),
        "parent_source_root": str(OLD if VERSION.endswith("v1") else SOURCE188),
        "predecessor_plan_sha256": predecessor_hash,
        "selection_screen_plan_sha256": (file_hash(SELECTION_BUILD / "selection_screen_plan.json")
                                          if VERSION.endswith("v3") else None),
        "selection_screen_results_sha256": (file_hash(SELECTION_BUILD / "selection_screen_results.json")
                                             if VERSION.endswith("v3") else None),
        "training_manifest_sha256": manifest_hash,
        "training_intensity_report_sha256": file_hash(BASELINE_REPORT),
        "solver": str(SOLVER), "solver_sha256": file_hash(SOLVER),
        "old_tree_sha256": old_hash, "old_file_count": old_files,
        "old_total_bytes": old_bytes,
        "paused_test_result_sha256": file_hash(PAUSED_RESULT),
        "parents": [p.name for p in paths], "baseline_inputs": baseline_info,
        "source_l2_sha256": first_l2_hash,
        "authorized_shunt_names": sorted(d.name for d in adapter.devices.values()),
        "authorized_shunt_keys": sorted(adapter.devices),
        "target_selection": {
            "population": "grid_2000 train only", "reference_cases": 2416,
            "sort": "F_cap_off then case_id", "layers": 10,
            "selection": ("60 unique references per layer, fixed seed, shuffled one-to-one with 60 parents"
                          if VERSION.endswith("v1") else
                          "Reuse v1's 600 unique references; retain sample_1; assign each layer's "
                          "remaining references by ascending F to vulnerability-ranked parents"),
            "parent_selection": (
                "current 60 old test groups" if VERSION.endswith("v1") else
                "sample_1 plus 59 lowest online PV counts among other 187; numeric-name tie break"
                if VERSION.endswith("v2") else
                "sample_1 plus 59 of 188 selected by the frozen v3 screening rule; no model output"),
            "exclude_full_cap_off_references": True,
            "paired_targets": ["N_off", "F_cap_off", "inductive_removed_category"],
            "N_off_absolute_tolerance": 2, "F_cap_off_absolute_tolerance": 0.02,
            "max_N_off": 147, "max_full_cap_off_cases": 1,
            "no_model_inference_or_test_result_selection": True,
        },
        "slots": slots,
    }
    staging = Path(tempfile.mkdtemp(prefix=VERSION + "_plan_", dir=TEST_ROOT))
    atomic_json(staging / "generation_plan.json", payload)
    os.rename(staging, NEW)
    print(json.dumps({"plan": str(NEW / "generation_plan.json"),
                      "slots": 600, "old_tree_sha256": old_hash,
                      "training_manifest_sha256": manifest_hash}, ensure_ascii=False), flush=True)
    return payload


def pilot_plan() -> dict:
    value = json.loads((NEW / "generation_plan.json").read_text(encoding="utf-8"))
    assert value["version"] == VERSION
    assert file_hash(TRAIN) == value["training_manifest_sha256"]
    assert file_hash(SOLVER) == value["solver_sha256"]
    assert file_hash(BASELINE_REPORT) == value["training_intensity_report_sha256"]
    assert file_hash(PAUSED_RESULT) == value["paused_test_result_sha256"]
    if VERSION.endswith("v3"):
        assert file_hash(SELECTION_BUILD / "selection_screen_plan.json") == value["selection_screen_plan_sha256"]
        assert file_hash(SELECTION_BUILD / "selection_screen_results.json") == value["selection_screen_results_sha256"]
    assert value["parents"][0] == "sample_1"
    return value


def input_bytes(folder: Path) -> dict[str, bytes]:
    result = {p.name: p.read_bytes() for p in folder.iterdir()
              if p.is_file() and p.name in INPUT_FILES}
    assert set(result) == INPUT_FILES
    return result


def byte_hashes(data: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(data.items())}


def shunt_catalog(parent_input: Path, whitelist: set[str]) -> list[dict]:
    adapter = CaseAdapter(parent_input)
    assert adapter.n == 2020
    assert {d.name for d in adapter.devices.values()} == whitelist
    rows = read_records(parent_input / "LF.L2")
    catalog = []
    for device in adapter.devices.values():
        row = rows[device.row]
        assert int(row[0]) == 1 and float(row[4]) == 0
        b = -1.0 / float(row[5])
        catalog.append({"id": device.name, "key": device.key,
                        "row": device.row, "b_proxy": b,
                        "kind": "capacitor" if b > 0 else "reactor"})
    assert len(catalog) == 171
    assert sum(x["kind"] == "capacitor" for x in catalog) == 157
    assert sum(x["kind"] == "reactor" for x in catalog) == 14
    return sorted(catalog, key=lambda x: x["id"])


def intensity(catalog: list[dict], off_ids: set[str]) -> dict:
    known = {x["id"] for x in catalog}
    assert off_ids <= known
    total_cap = sum(max(x["b_proxy"], 0.0) for x in catalog)
    removed_cap = sum(max(x["b_proxy"], 0.0) for x in catalog if x["id"] in off_ids)
    removed_reactor = sum(max(-x["b_proxy"], 0.0) for x in catalog if x["id"] in off_ids)
    n_reactor = sum(x["id"] in off_ids and x["kind"] == "reactor" for x in catalog)
    assert total_cap > 0
    return {"N_off": len(off_ids), "B_cap_base": total_cap,
            "B_cap_off": removed_cap, "F_cap_off": removed_cap / total_cap,
            "N_inductive_off": n_reactor,
            "B_inductive_off": removed_reactor,
            "inductive_removed": n_reactor > 0}


def within_target(actual: dict, slot: dict) -> bool:
    return (abs(actual["N_off"] - slot["target_N_off"]) <= slot["tolerance_N_off"]
            and abs(actual["F_cap_off"] - slot["target_F_cap_off"]) <= slot["tolerance_F_cap_off"] + 1e-10
            and actual["inductive_removed"] == slot["target_inductive_removed"]
            and actual["N_off"] <= 147 and actual["F_cap_off"] < 1 - 1e-10)


def candidate_ids(catalog: list[dict], slot: dict, attempt: int) -> set[str] | None:
    n = len(catalog)
    caps = np.array([max(x["b_proxy"], 0.0) for x in catalog], dtype=float)
    reactor = np.array([float(x["kind"] == "reactor") for x in catalog])
    cap_count = np.array([float(x["kind"] == "capacitor") for x in catalog])
    total = float(caps.sum())
    target_n = slot["target_N_off"]
    # First seek the exact paired count, then use only the documented ±2 slack.
    shift = (0, -1, 1, -2, 2)[(attempt - 1) % 5]
    count = target_n + shift
    target_f = slot["target_F_cap_off"]
    low_f = max(0.0, target_f - slot["tolerance_F_cap_off"]) * total
    high_f = min(1.0, target_f + slot["tolerance_F_cap_off"]) * total
    matrix = np.vstack((np.ones(n), caps, reactor, cap_count))
    low = [count, low_f, 1 if slot["target_inductive_removed"] else 0, 0]
    high = [count, high_f, 14 if slot["target_inductive_removed"] else 0, 156]
    rng = random.Random(slot["seed"] * 1000003 + attempt)
    if attempt <= 64:
        costs = np.array([rng.uniform(-1, 1) for _ in range(n)])
    else:
        # Search the high end of the *same frozen tolerance window*, not a
        # stronger target. Random tie-breaking changes the spatial subset.
        scale = float(max(caps))
        costs = -caps / scale + np.array([rng.uniform(-0.025, 0.025) for _ in range(n)])
    solved = milp(costs, integrality=np.ones(n),
                  bounds=Bounds(np.zeros(n), np.ones(n)),
                  constraints=LinearConstraint(matrix, low, high),
                  options={"time_limit": 5.0, "presolve": True})
    if solved.x is None:
        return None
    ids = {catalog[i]["id"] for i, x in enumerate(solved.x) if x >= 0.5}
    return ids if within_target(intensity(catalog, ids), slot) else None


def candidate_near_seed(catalog: list[dict], slot: dict, seed_ids: set[str],
                        variation: int) -> set[str] | None:
    """Minimum-Mark-distance intensity match around a screened failure."""
    n = len(catalog)
    caps = np.array([max(x["b_proxy"], 0.0) for x in catalog], dtype=float)
    reactor = np.array([float(x["kind"] == "reactor") for x in catalog])
    cap_count = np.array([float(x["kind"] == "capacitor") for x in catalog])
    total = float(caps.sum())
    target_f = slot["target_F_cap_off"]
    matrix = np.vstack((np.ones(n), caps, reactor, cap_count))
    low = [slot["target_N_off"] - 2,
           max(0.0, target_f - slot["tolerance_F_cap_off"]) * total,
           1 if slot["target_inductive_removed"] else 0, 0]
    high = [slot["target_N_off"] + 2,
            min(1.0, target_f + slot["tolerance_F_cap_off"]) * total,
            14 if slot["target_inductive_removed"] else 0, 156]
    rng = random.Random(slot["seed"] * 1000033 + variation + len(seed_ids))
    costs = np.array([(-1 if x["id"] in seed_ids else 1) + rng.uniform(-0.02, 0.02)
                      for x in catalog], dtype=float)
    solved = milp(costs, integrality=np.ones(n),
                  bounds=Bounds(np.zeros(n), np.ones(n)),
                  constraints=LinearConstraint(matrix, low, high),
                  options={"time_limit": 5.0, "presolve": True})
    if solved.x is None:
        return None
    ids = {catalog[i]["id"] for i, value in enumerate(solved.x) if value >= 0.5}
    return ids if within_target(intensity(catalog, ids), slot) else None


def reference_off_ids(slot: dict) -> set[str]:
    """Read the frozen target case's actual SH Mark vector, never model output."""
    source_parent = ROOT / "lfsamples/lfsample-ok" / slot["reference_parent_case_id"]
    before = read_records(source_parent / "LF.L2")
    after = read_records(Path(slot["reference_input_dir"]) / "LF.L2")
    assert len(before) == len(after)
    off = set()
    for left, right in zip(before, after):
        assert left[1:] == right[1:]
        if left[0] == right[0]:
            continue
        assert int(left[0]) == 1 and int(right[0]) == 0
        assert abs(int(left[1])) == abs(int(left[2]))
        off.add(str(left[17]).strip())
    assert len(off) == slot["target_N_off"]
    return off


def probe_transfer(parent_id: str, slot_number: int) -> None:
    frozen = pilot_plan()
    slot = next(s for s in frozen["slots"]
                if s["parent_id"] == parent_id and s["slot_number"] == slot_number)
    source = parent_input_dir(parent_id)
    baseline = input_bytes(source)
    catalog = shunt_catalog(source, set(frozen["authorized_shunt_names"]))
    off = reference_off_ids(slot)
    actual = intensity(catalog, off)
    assert within_target(actual, slot)
    candidate = {**baseline, "LF.L2": switch_only_l2_mark(
        baseline["LF.L2"], off, set(frozen["authorized_shunt_names"]))}
    run_id = f"transfer_{VERSION}_{parent_id}_slot{slot_number:02d}"
    record, _ = solver_run(candidate, run_id, SELECTION_BUILD / "transfer_evidence",
                           SELECTION_BUILD / "_work")
    destination = SELECTION_BUILD / "transfer_probes"
    destination.mkdir(exist_ok=True)
    atomic_json(destination / f"{run_id}.json", {"parent": parent_id, "slot": slot,
                                                  "realized": actual, "off_ids": sorted(off),
                                                  "run": record})
    print({"TRANSFER_PROBE": run_id, "cal_flag": record["cal_flag"],
           "N_off": actual["N_off"], "F_cap_off": actual["F_cap_off"]}, flush=True)


def switch_only_l2_mark(original: bytes, off_ids: set[str], whitelist: set[str]) -> bytes:
    assert off_ids <= whitelist
    changed = []
    found = set()
    for line in original.splitlines(keepends=True):
        fields = line.split(b",")
        if len(fields) >= 18:
            identifier = fields[17].strip().strip(b"'").strip().decode("gbk")
            if identifier in off_ids:
                assert identifier not in found
                assert fields[0].strip() == b"1"
                assert fields[1].strip() == fields[2].strip()
                assert float(fields[4]) == 0
                replacement, count = re.subn(rb"^([ \t]*)1([ \t]*)$", rb"\g<1>0\g<2>", fields[0])
                assert count == 1
                fields[0] = replacement
                line = b",".join(fields)
                found.add(identifier)
        changed.append(line)
    assert found == off_ids
    output = b"".join(changed)
    before_lines, after_lines = original.splitlines(keepends=True), output.splitlines(keepends=True)
    assert len(before_lines) == len(after_lines)
    for before, after in zip(before_lines, after_lines):
        if before != after:
            left, right = before.split(b","), after.split(b",")
            assert len(left) == len(right) and left[1:] == right[1:]
            assert left[0].strip() == b"1" and right[0].strip() == b"0"
    return output


def solver_run(data: dict[str, bytes], run_id: str, output_root: Path,
               work_root: Path | None = None) -> tuple[dict, dict[str, bytes]]:
    scratch_root = work_root or (BUILD / "_work")
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=run_id + "_", dir=scratch_root) as temporary:
        work = Path(temporary)
        for name, raw in data.items():
            (work / name).write_bytes(raw)
        expected = byte_hashes(data)
        history = output_root / run_id
        history.mkdir(parents=True, exist_ok=True)
        previous = [p for p in history.iterdir() if p.is_dir() and p.name.startswith("call_")]
        audit = history / f"call_{len(previous) + 1:03d}"
        audit.mkdir(exist_ok=False)
        runner = PSASPRunner(SOLVER, timeout=60, audit_dir=audit, retries=0)
        while True:
            try:
                started = time.monotonic()
                result = runner.run(work)
                seconds = time.monotonic() - started
                break
            except RuntimeError as exc:
                if "license is busy" not in str(exc) and "solver process already exists" not in str(exc):
                    raise
                print({"waiting_for_other_psasp": str(exc), "run_id": run_id}, flush=True)
                time.sleep(15)
        after = {name: file_hash(work / name) for name in data}
        assert expected == after, f"PSASP modified an input card: {run_id}"
        cal = (work / "LF.CAL").read_bytes()
        flag = int(cal.split(b",")[0].strip())
        assert flag in (-1, 0, 1) and result.return_code == 0
        assert result.converged == (flag == 1)
        outputs = {p.name: p.read_bytes() for p in work.iterdir()
                   if p.is_file() and p.name not in data}
        assert "LF.CAL" in outputs and "lfreport.lis" in outputs
        audit_runs = [p for p in audit.iterdir() if p.is_dir()]
        assert len(audit_runs) == 1
        record = {"run_id": run_id, "solver": str(SOLVER),
                  "solver_sha256": runner.executable_sha256,
                  "return_code": result.return_code, "cal_flag": flag,
                  "report_final_error": result.error,
                  "seconds": seconds, "input_hashes": expected,
                  "input_unchanged": True, "fresh_output_audit": str(audit_runs[0] / "run.json"),
                  "output_hashes": byte_hashes(outputs)}
        return record, outputs


def materialize_result(destination: Path, record: dict, outputs: dict[str, bytes]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in outputs.items():
        (destination / name).write_bytes(raw)
    atomic_json(destination / "run_result.json", record)


def pilot(max_attempts: int) -> None:
    assert 1 <= max_attempts <= 500
    frozen = pilot_plan()
    slots = [s for s in frozen["slots"] if s["parent_id"] == "sample_1"]
    assert len(slots) == 10
    source = parent_input_dir("sample_1")
    parent = input_bytes(source)
    expected = frozen["baseline_inputs"][0]["input_hashes"]
    assert byte_hashes(parent) == expected
    catalog = shunt_catalog(source, set(frozen["authorized_shunt_names"]))
    whitelist = {x["id"] for x in catalog}
    pilot_root = BUILD / "pilot"
    pilot_root.mkdir(parents=True, exist_ok=True)
    group = pilot_root / "sample_1"
    (group / "baseline/input").mkdir(parents=True, exist_ok=True)
    for name, raw in parent.items():
        path = group / "baseline/input" / name
        if path.exists():
            assert path.read_bytes() == raw
        else:
            path.write_bytes(raw)
    if not (group / "baseline/result/run_result.json").is_file():
        baseline_run, baseline_outputs = solver_run(parent, "pilot_sample_1_baseline",
                                                    BUILD / "evidence")
        assert baseline_run["cal_flag"] == 1, "Fresh parent calculation did not converge"
        materialize_result(group / "baseline/result", baseline_run, baseline_outputs)
    old_l2_hashes = previous_l2_hashes("sample_1")
    seen = set()
    failures = []
    for slot in slots:
        number = slot["slot_number"]
        accepted_path = group / "negatives" / f"neg_{number:03d}"
        if accepted_path.exists():
            manifest = json.loads((accepted_path / "sample_manifest.json").read_text(encoding="utf-8"))
            assert manifest["slot_id"] == slot["slot_id"] and manifest["accepted"]
            seen.add(manifest["input_l2_sha256"])
            continue
        rejection_path = pilot_root / "rejections" / f"{slot['slot_id']}.jsonl"
        rejection_path.parent.mkdir(parents=True, exist_ok=True)
        tried = set()
        for attempt in range(1, max_attempts + 1):
            off_ids = candidate_ids(catalog, slot, attempt)
            if off_ids is None:
                reason = "INTENSITY_INFEASIBLE_OR_MILP_TIMEOUT"
                with rejection_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"attempt": attempt, "reason": reason}) + "\n")
                continue
            actual = intensity(catalog, off_ids)
            assert within_target(actual, slot)
            new_l2 = switch_only_l2_mark(parent["LF.L2"], off_ids, whitelist)
            digest = hashlib.sha256(new_l2).hexdigest()
            if digest in seen or digest in old_l2_hashes or digest in tried:
                continue
            tried.add(digest)
            candidate = {**parent, "LF.L2": new_l2}
            run_id = f"pilot_s1_slot{number:02d}_try{attempt:03d}"
            main, main_outputs = solver_run(candidate, run_id + "_main", BUILD / "evidence")
            if main["cal_flag"] != -1:
                with rejection_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"attempt": attempt, "reason": "MAIN_NOT_NONCONVERGED",
                                          "cal_flag": main["cal_flag"], "intensity": actual,
                                          "input_l2_sha256": digest, "run_id": main["run_id"]}) + "\n")
                if attempt % 10 == 0:
                    print({"pilot_slot": number, "attempt": attempt,
                           "main_not_nonconverged": True}, flush=True)
                continue
            repeat, repeat_outputs = solver_run(candidate, run_id + "_repeat", BUILD / "evidence")
            if repeat["cal_flag"] != -1:
                with rejection_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"attempt": attempt, "reason": "REPEAT_NOT_NONCONVERGED",
                                          "cal_flag": repeat["cal_flag"], "intensity": actual,
                                          "input_l2_sha256": digest, "run_id": repeat["run_id"]}) + "\n")
                continue
            # Reverse exactly the authorized first-field bytes and compare all input cards.
            lines = new_l2.splitlines(keepends=True)
            restored = []
            for line in lines:
                fields = line.split(b",")
                if len(fields) >= 18:
                    identifier = fields[17].strip().strip(b"'").strip().decode("gbk")
                    if identifier in off_ids:
                        fields[0] = re.sub(rb"^([ \t]*)0([ \t]*)$", rb"\g<1>1\g<2>", fields[0])
                        line = b",".join(fields)
                restored.append(line)
            restored_l2 = b"".join(restored)
            assert restored_l2 == parent["LF.L2"]
            restored_data = {**candidate, "LF.L2": restored_l2}
            assert byte_hashes(restored_data) == expected
            recovery, recovery_outputs = solver_run(restored_data, run_id + "_restore", BUILD / "evidence")
            if recovery["cal_flag"] != 1:
                with rejection_path.open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"attempt": attempt, "reason": "RESTORE_NOT_CONVERGED",
                                          "cal_flag": recovery["cal_flag"], "intensity": actual,
                                          "input_l2_sha256": digest}) + "\n")
                continue
            (accepted_path / "input").mkdir(parents=True, exist_ok=False)
            for name, raw in candidate.items():
                (accepted_path / "input" / name).write_bytes(raw)
            materialize_result(accepted_path / "result", main, main_outputs)
            materialize_result(accepted_path / "diagnostics/repeat", repeat, repeat_outputs)
            materialize_result(accepted_path / "diagnostics/restore", recovery, recovery_outputs)
            record = {"accepted": True, "slot_id": slot["slot_id"],
                      "reference_case_id": slot["reference_case_id"],
                      "target": slot, "realized_intensity": actual,
                      "off_ids": sorted(off_ids), "input_l2_sha256": digest,
                      "main_run_id": main["run_id"], "repeat_run_id": repeat["run_id"],
                      "restore_run_id": recovery["run_id"],
                      "validation": "fresh_baseline_main_repeat_restore"}
            atomic_json(accepted_path / "sample_manifest.json", record)
            seen.add(digest)
            print({"pilot_slot_accepted": number, "attempt": attempt,
                   "N_off": actual["N_off"], "F_cap_off": actual["F_cap_off"]}, flush=True)
            break
        if not accepted_path.exists():
            failures.append({"slot_id": slot["slot_id"], "attempts": max_attempts,
                             "target_N_off": slot["target_N_off"],
                             "target_F_cap_off": slot["target_F_cap_off"]})
            atomic_json(BUILD / "pilot_status.json",
                        {"status": "PILOT_TARGET_UNREACHED", "accepted_slots": len(seen),
                         "failures": failures, "no_formal_samples_published": True})
            print({"PILOT_STOPPED": failures[-1]}, flush=True)
            return
    assert len(seen) == 10
    atomic_json(BUILD / "pilot_status.json",
                {"status": "PILOT_PASS", "accepted_slots": 10,
                 "no_formal_samples_published": True})
    print({"PILOT_PASS": "sample_1", "children": 10}, flush=True)


def rebase_audit_records(stage: Path) -> None:
    """The pilot evidence moved out of the compare tool's candidate root."""
    for path in stage.rglob("run_result.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        audit = Path(record["fresh_output_audit"])
        if audit.is_file():
            continue
        candidates = list((BUILD / "evidence" / record["run_id"]).glob("call_*/*/run.json"))
        assert len(candidates) == 1, (record["run_id"], candidates)
        record["fresh_output_audit"] = str(candidates[0])
        atomic_json(path, record)


def restored_l2(candidate: bytes, off_ids: set[str]) -> bytes:
    restored = []
    found = set()
    for line in candidate.splitlines(keepends=True):
        fields = line.split(b",")
        if len(fields) >= 18:
            identifier = fields[17].strip().strip(b"'").strip().decode("gbk")
            if identifier in off_ids:
                assert identifier not in found
                assert fields[0].strip() == b"0"
                replacement, count = re.subn(rb"^([ \t]*)0([ \t]*)$",
                                             rb"\g<1>1\g<2>", fields[0])
                assert count == 1
                fields[0] = replacement
                line = b",".join(fields)
                found.add(identifier)
        restored.append(line)
    assert found == off_ids
    return b"".join(restored)


def verify_result_dir(path: Path, input_hashes: dict[str, str], flag: int,
                      solver_sha256: str) -> dict:
    record = json.loads((path / "run_result.json").read_text(encoding="utf-8"))
    assert record["cal_flag"] == flag and record["return_code"] == 0
    assert record["input_hashes"] == input_hashes and record["input_unchanged"]
    assert record["solver_sha256"] == solver_sha256
    assert Path(record["fresh_output_audit"]).is_file()
    assert (path / "LF.CAL").read_bytes().split(b",")[0].strip() == str(flag).encode()
    for name, digest in record["output_hashes"].items():
        assert file_hash(path / name) == digest
    return record


def verify_group(group: Path, frozen: dict) -> list[dict]:
    parent_id = group.name
    slots = [s for s in frozen["slots"] if s["parent_id"] == parent_id]
    assert len(slots) == 10
    source = parent_input_dir(parent_id)
    baseline = input_bytes(group / "baseline/input")
    assert baseline == input_bytes(source)
    expected = next(x["input_hashes"] for x in frozen["baseline_inputs"]
                    if x["parent"] == parent_id)
    assert byte_hashes(baseline) == expected
    verify_result_dir(group / "baseline/result", expected, 1, frozen["solver_sha256"])
    catalog = shunt_catalog(source, set(frozen["authorized_shunt_names"]))
    white = {x["id"] for x in catalog}
    original_l2 = previous_l2_hashes(parent_id)
    members = sorted((group / "negatives").iterdir())
    assert len(members) == 10
    seen = set()
    realized = []
    for slot in slots:
        child = group / "negatives" / f"neg_{slot['slot_number']:03d}"
        assert child in members
        record = json.loads((child / "sample_manifest.json").read_text(encoding="utf-8"))
        assert record["accepted"] and record["slot_id"] == slot["slot_id"]
        assert record["reference_case_id"] == slot["reference_case_id"]
        data = input_bytes(child / "input")
        assert set(data) == set(baseline)
        assert all(data[name] == baseline[name] for name in baseline if name != "LF.L2")
        digest = hashlib.sha256(data["LF.L2"]).hexdigest()
        assert digest == record["input_l2_sha256"]
        assert digest not in seen and digest not in original_l2
        seen.add(digest)
        off = set(record["off_ids"])
        assert off <= white and data["LF.L2"] == switch_only_l2_mark(baseline["LF.L2"], off, white)
        assert restored_l2(data["LF.L2"], off) == baseline["LF.L2"]
        value = intensity(catalog, off)
        assert within_target(value, slot)
        assert value == record["realized_intensity"]
        child_hashes = byte_hashes(data)
        assert child_hashes != expected
        verify_result_dir(child / "result", child_hashes, -1, frozen["solver_sha256"])
        verify_result_dir(child / "diagnostics/repeat", child_hashes, -1,
                          frozen["solver_sha256"])
        verify_result_dir(child / "diagnostics/restore", expected, 1,
                          frozen["solver_sha256"])
        realized.append({"sample_id": f"{parent_id}_neg_{slot['slot_number']:03d}",
                         "parent_id": parent_id, "slot_id": slot["slot_id"],
                         "reference_case_id": slot["reference_case_id"],
                         "input_dir": str(child / "input"),
                         "result_dir": str(child / "result"), **value})
    return realized


def generate_group(parent_id: str, frozen: dict, max_attempts: int) -> None:
    assert 1 <= max_attempts <= 500
    destination = NEW / parent_id
    if destination.exists():
        verify_group(destination, frozen)
        print({"group_verified_existing": parent_id}, flush=True)
        return
    stage = BUILD / "staging" / parent_id
    if parent_id == "sample_1":
        if not stage.exists():
            shutil.copytree(BUILD / "pilot" / parent_id, stage)
            rebase_audit_records(stage)
    else:
        (stage / "baseline/input").mkdir(parents=True, exist_ok=True)
        source = parent_input_dir(parent_id)
        baseline = input_bytes(source)
        expected = next(x["input_hashes"] for x in frozen["baseline_inputs"]
                        if x["parent"] == parent_id)
        assert byte_hashes(baseline) == expected
        for name, raw in baseline.items():
            target = stage / "baseline/input" / name
            if target.exists():
                assert target.read_bytes() == raw
            else:
                target.write_bytes(raw)
        if not (stage / "baseline/result/run_result.json").is_file():
            run_id = f"matched_{parent_id}_baseline"
            record, outputs = solver_run(baseline, run_id, BUILD / "evidence")
            if record["cal_flag"] != 1:
                raise RuntimeError(f"Fresh parent failed to converge: {parent_id}")
            materialize_result(stage / "baseline/result", record, outputs)
    source = parent_input_dir(parent_id)
    baseline = input_bytes(stage / "baseline/input")
    expected = byte_hashes(baseline)
    catalog = shunt_catalog(source, set(frozen["authorized_shunt_names"]))
    white = {x["id"] for x in catalog}
    old_l2_hashes = previous_l2_hashes(parent_id)
    slots = [s for s in frozen["slots"] if s["parent_id"] == parent_id]
    seen = set()
    screen_plan = screen_results = None
    if VERSION.endswith("v3"):
        assert SELECTION_VERSION == "v2"
        screen_plan = json.loads((SELECTION_BUILD / "selection_screen_plan.json").read_text(encoding="utf-8"))
        screen_results = json.loads((SELECTION_BUILD / "selection_screen_results.json").read_text(encoding="utf-8"))["parents"]
    for slot in slots:
        number = slot["slot_number"]
        child = stage / "negatives" / f"neg_{number:03d}"
        if child.exists():
            manifest = json.loads((child / "sample_manifest.json").read_text(encoding="utf-8"))
            assert manifest["accepted"] and manifest["slot_id"] == slot["slot_id"]
            seen.add(manifest["input_l2_sha256"])
            continue
        rejection_file = BUILD / "rejections" / parent_id / f"slot_{number:02d}.jsonl"
        rejection_file.parent.mkdir(parents=True, exist_ok=True)
        prior = []
        if rejection_file.is_file():
            prior = [json.loads(line) for line in rejection_file.read_text(encoding="utf-8").splitlines()]
        start = max((x["attempt"] for x in prior), default=0) + 1
        tried = {x["input_l2_sha256"] for x in prior if "input_l2_sha256" in x}
        preferred = []
        if VERSION.endswith("v3"):
            reference = reference_off_ids(slot)
            if within_target(intensity(catalog, reference), slot):
                preferred.append(reference)
            assert screen_plan is not None and screen_results is not None
            for pattern in screen_plan["patterns"]:
                if screen_results[parent_id][pattern["pattern_id"]]["cal_flag"] != -1:
                    continue
                seed = set(pattern["off_ids"])
                if within_target(intensity(catalog, seed), slot):
                    preferred.append(seed)
                for variation in range(1, 4):
                    local = candidate_near_seed(catalog, slot, seed, variation)
                    if local is not None:
                        preferred.append(local)
            preferred = list({tuple(sorted(ids)): ids for ids in preferred}.values())
        for attempt in range(start, max_attempts + 1):
            ids = (preferred[attempt - 1] if attempt <= len(preferred) else
                   candidate_ids(catalog, slot, attempt - len(preferred)))
            if ids is None:
                rejection = {"attempt": attempt, "reason": "INTENSITY_INFEASIBLE_OR_MILP_TIMEOUT"}
            else:
                actual = intensity(catalog, ids)
                assert within_target(actual, slot)
                l2 = switch_only_l2_mark(baseline["LF.L2"], ids, white)
                digest = hashlib.sha256(l2).hexdigest()
                if digest in seen or digest in tried or digest in old_l2_hashes:
                    rejection = {"attempt": attempt, "reason": "DUPLICATE_INPUT",
                                 "input_l2_sha256": digest}
                else:
                    candidate = {**baseline, "LF.L2": l2}
                    run_id = f"matched_{parent_id}_slot{number:02d}_try{attempt:03d}"
                    main, main_outputs = solver_run(candidate, run_id + "_main", BUILD / "evidence")
                    if main["cal_flag"] != -1:
                        rejection = {"attempt": attempt, "reason": "MAIN_NOT_NONCONVERGED",
                                     "cal_flag": main["cal_flag"], "input_l2_sha256": digest,
                                     "intensity": actual, "run_id": main["run_id"]}
                    else:
                        repeat, repeat_outputs = solver_run(candidate, run_id + "_repeat",
                                                            BUILD / "evidence")
                        if repeat["cal_flag"] != -1:
                            rejection = {"attempt": attempt, "reason": "REPEAT_NOT_NONCONVERGED",
                                         "cal_flag": repeat["cal_flag"], "input_l2_sha256": digest,
                                         "intensity": actual, "run_id": repeat["run_id"]}
                        else:
                            restored = restored_l2(l2, ids)
                            assert restored == baseline["LF.L2"]
                            restored_data = {**candidate, "LF.L2": restored}
                            assert byte_hashes(restored_data) == expected
                            recovery, recovery_outputs = solver_run(
                                restored_data, run_id + "_restore", BUILD / "evidence")
                            if recovery["cal_flag"] != 1:
                                rejection = {"attempt": attempt, "reason": "RESTORE_NOT_CONVERGED",
                                             "cal_flag": recovery["cal_flag"],
                                             "input_l2_sha256": digest, "intensity": actual}
                            else:
                                staging_root = BUILD / "child_staging"
                                staging_root.mkdir(exist_ok=True)
                                child_stage = Path(tempfile.mkdtemp(
                                    prefix=f"{parent_id}_neg_{number:03d}_", dir=staging_root))
                                (child_stage / "input").mkdir()
                                for name, raw in candidate.items():
                                    (child_stage / "input" / name).write_bytes(raw)
                                materialize_result(child_stage / "result", main, main_outputs)
                                materialize_result(child_stage / "diagnostics/repeat", repeat, repeat_outputs)
                                materialize_result(child_stage / "diagnostics/restore", recovery, recovery_outputs)
                                atomic_json(child_stage / "sample_manifest.json", {
                                    "accepted": True, "slot_id": slot["slot_id"],
                                    "reference_case_id": slot["reference_case_id"],
                                    "target": slot, "realized_intensity": actual,
                                    "off_ids": sorted(ids), "input_l2_sha256": digest,
                                    "main_run_id": main["run_id"],
                                    "repeat_run_id": repeat["run_id"],
                                    "restore_run_id": recovery["run_id"],
                                    "validation": "fresh_baseline_main_repeat_restore"})
                                child.parent.mkdir(parents=True, exist_ok=True)
                                assert not child.exists()
                                os.rename(child_stage, child)
                                seen.add(digest)
                                print({"accepted": f"{parent_id}_neg_{number:03d}",
                                       "attempt": attempt, "N_off": actual["N_off"],
                                       "F_cap_off": round(actual["F_cap_off"], 5)}, flush=True)
                                break
            with rejection_file.open("a", encoding="utf-8") as log:
                log.write(json.dumps(rejection, ensure_ascii=False) + "\n")
        if not child.exists():
            atomic_json(BUILD / "status.json", {
                "status": "TARGET_SLOT_UNREACHED", "parent": parent_id,
                "slot_id": slot["slot_id"], "attempts": max_attempts,
                "formal_groups_published": len([p for p in NEW.iterdir() if p.is_dir()])})
            print({"STOPPED_UNREACHED_SLOT": slot["slot_id"], "attempts": max_attempts}, flush=True)
            return
    assert len(seen) == 10
    atomic_json(stage / "group_manifest.json", {
        "parent": parent_id, "source_baseline": str(source),
        "baseline_input_hashes": expected,
        "plan_sha256": file_hash(NEW / "generation_plan.json"),
        "solver_sha256": frozen["solver_sha256"],
        "sample_ids": [f"{parent_id}_neg_{i:03d}" for i in range(1, 11)]})
    verify_group(stage, frozen)
    assert not destination.exists()
    os.rename(stage, destination)
    print({"GROUP_COMMITTED": parent_id, "children": 10}, flush=True)


def generate(max_attempts: int) -> None:
    frozen = pilot_plan()
    pilot_status = json.loads((BUILD / "pilot_status.json").read_text(encoding="utf-8"))
    assert pilot_status["status"] == "PILOT_PASS" and pilot_status["accepted_slots"] == 10
    for position, parent_id in enumerate(frozen["parents"], 1):
        atomic_json(BUILD / "status.json", {
            "status": "GENERATING", "parent": parent_id,
            "parent_position": position, "target_parent_groups": 60})
        generate_group(parent_id, frozen, max_attempts)
        if not (NEW / parent_id).is_dir():
            return
    print({"ALL_GROUPS_COMMITTED": 60}, flush=True)


def verify_all() -> dict:
    frozen = pilot_plan()
    found = sorted((p for p in NEW.iterdir() if p.is_dir()),
                   key=lambda p: int(p.name.removeprefix("sample_")))
    assert len(found) == 60 and [p.name for p in found] == frozen["parents"]
    records = []
    for position, group in enumerate(found, 1):
        records.extend(verify_group(group, frozen))
        if position % 10 == 0:
            print({"verified_groups": position, "verified_children": len(records)}, flush=True)
    assert len(records) == 600
    assert len({r["input_dir"] for r in records}) == 600
    assert max(r["N_off"] for r in records) <= 147
    assert sum(r["F_cap_off"] >= 1 - 1e-10 for r in records) <= 1
    current_tree, current_files, current_bytes = old_tree_digest(OLD)
    assert (current_tree, current_files, current_bytes) == (
        frozen["old_tree_sha256"], frozen["old_file_count"], frozen["old_total_bytes"])
    assert file_hash(BASELINE_REPORT) == frozen["training_intensity_report_sha256"]
    assert file_hash(PAUSED_RESULT) == frozen["paused_test_result_sha256"]
    result = {"status": "PASS", "version": frozen["version"],
              "parent_groups": len(found), "negative_cases": len(records),
              "only_authorized_shunt_mark_1_to_0": True,
              "paired_intensity_targets_met": True,
              "fresh_baseline_main_repeat_restore": True,
              "old_test_root_sha256_unchanged": current_tree,
              "old_result_hashes_unchanged": True,
              "plan_sha256": file_hash(NEW / "generation_plan.json"),
              "solver_sha256": frozen["solver_sha256"], "cases": records}
    atomic_json(NEW / "verification.json", result)
    atomic_json(BUILD / "status.json", {"status": "ALL_VERIFIED",
                                        "parent_groups": 60, "negative_cases": 600})
    print({"ALL_VERIFIED": 600}, flush=True)
    return result


def selection_screen_plan() -> dict:
    path = SELECTION_BUILD / "selection_screen_plan.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    first_plan = TEST_ROOT / "ACTIVSg2020_matched_v1/generation_plan.json"
    frozen = json.loads(first_plan.read_text(encoding="utf-8"))
    assert frozen["version"] == "ACTIVSg2020_matched_v1"
    first_layer = sorted((s for s in frozen["slots"] if s["strength_layer"] == 1),
                         key=lambda s: (s["target_F_cap_off"], s["reference_case_id"]))
    assert len(first_layer) == 60
    references = [first_layer[i] for i in (0, 29, 59)] if SELECTION_VERSION == "v1" else []
    if SELECTION_VERSION == "v2":
        used_reference_parents = set()
        for bin_number in range(10):
            cohort = first_layer[bin_number * 6:(bin_number + 1) * 6]
            assert len(cohort) == 6
            options = [s for s in cohort if s["reference_parent_case_id"]
                       not in used_reference_parents]
            chosen = options[len(options) // 2] if options else cohort[len(cohort) // 2]
            references.append(chosen)
            used_reference_parents.add(chosen["reference_parent_case_id"])
    source = SOURCE188 / "sample_1"
    catalog = shunt_catalog(source, set(frozen["authorized_shunt_names"]))
    patterns = []
    for index, slot in enumerate(references, 1):
        ids = (candidate_ids(catalog, slot, 65) if SELECTION_VERSION == "v1"
               else reference_off_ids(slot))
        assert ids is not None
        value = intensity(catalog, ids)
        assert within_target(value, slot)
        patterns.append({"pattern_id": f"low_strength_{index}",
                         "reference_case_id": slot["reference_case_id"],
                         "target_N_off": slot["target_N_off"],
                         "target_F_cap_off": slot["target_F_cap_off"],
                         "target_inductive_removed": slot["target_inductive_removed"],
                         "off_ids": sorted(ids), "realized": value})
    if SELECTION_VERSION == "v2":
        pilot_record = json.loads((ROOT / "artifacts/activsg2020_matched_v1_build/pilot/sample_1/negatives/neg_001/sample_manifest.json").read_text(encoding="utf-8"))
        pilot_slot = next(s for s in first_layer if s["slot_id"] == pilot_record["slot_id"])
        ids = set(pilot_record["off_ids"])
        value = intensity(catalog, ids)
        assert within_target(value, pilot_slot)
        patterns.append({"pattern_id": "sample_1_pilot_success",
                         "reference_case_id": pilot_slot["reference_case_id"],
                         "target_N_off": pilot_slot["target_N_off"],
                         "target_F_cap_off": pilot_slot["target_F_cap_off"],
                         "target_inductive_removed": pilot_slot["target_inductive_removed"],
                         "off_ids": sorted(ids), "realized": value})
    parent_list = sorted((p.name for p in SOURCE188.iterdir() if p.is_dir()),
                         key=lambda n: int(n.removeprefix("sample_")))
    assert len(parent_list) == 188
    payload = {"version": f"ACTIVSg2020_matched_v3_selection_{SELECTION_VERSION}",
               "source_parent_root": str(SOURCE188),
               "source_final_verification_sha256": file_hash(SOURCE188 / "final_verification.json"),
               "predecessor_v1_plan_sha256": file_hash(first_plan),
               "training_manifest_sha256": file_hash(TRAIN),
               "solver": str(SOLVER), "solver_sha256": file_hash(SOLVER),
               "parent_ids": parent_list, "patterns": patterns,
               "selection_rule": (
                   "Retain previously validated sample_1. Rank other 187 parents by count of "
                   "fixed low-strength screens with fresh LF.CAL=-1 (descending), then "
                   "lowest-strength screen failure first, then online PV count ascending, "
                   "then numeric sample ID. Select first 59 with at least one failed screen. "
                   "No model output is read."),
               "same_single_psasp_solver_workflow": True}
    SELECTION_BUILD.mkdir(parents=True, exist_ok=True)
    atomic_json(path, payload)
    print({"SCREEN_PLAN_FROZEN": str(path),
           "parents": 188, "patterns": len(patterns)}, flush=True)
    return payload


def screen_parents() -> dict:
    frozen = selection_screen_plan()
    assert frozen["solver_sha256"] == file_hash(SOLVER)
    assert frozen["source_final_verification_sha256"] == file_hash(
        SOURCE188 / "final_verification.json")
    result_path = SELECTION_BUILD / "selection_screen_results.json"
    prior = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {"parents": {}}
    rows = prior["parents"]
    reference_plan = json.loads((TEST_ROOT / "ACTIVSg2020_matched_v1/generation_plan.json").read_text(encoding="utf-8"))
    white = set(reference_plan["authorized_shunt_names"])
    for position, parent_id in enumerate(frozen["parent_ids"], 1):
        source = SOURCE188 / parent_id
        data = input_bytes(source)
        source_meta = json.loads((source / "solver_run_result.json").read_text(encoding="utf-8"))
        assert source_meta["solver_status"] == "CONVERGED"
        assert byte_hashes(data) == source_meta["input_hashes"]
        entry = rows.setdefault(parent_id, {})
        for pattern in frozen["patterns"]:
            pattern_id = pattern["pattern_id"]
            if pattern_id in entry:
                continue
            changed = switch_only_l2_mark(data["LF.L2"], set(pattern["off_ids"]), white)
            candidate = {**data, "LF.L2": changed}
            run_id = f"v3screen_{parent_id}_{pattern_id}"
            run_record, _ = solver_run(candidate, run_id,
                                       SELECTION_BUILD / "evidence",
                                       SELECTION_BUILD / "_work")
            entry[pattern_id] = {"cal_flag": run_record["cal_flag"],
                                 "run_id": run_id, "input_hashes": run_record["input_hashes"],
                                 "solver_sha256": run_record["solver_sha256"],
                                 "fresh_output_audit": run_record["fresh_output_audit"]}
            atomic_json(result_path, {"plan_sha256": file_hash(SELECTION_BUILD / "selection_screen_plan.json"),
                                      "parents": rows})
        if position % 10 == 0 or position == len(frozen["parent_ids"]):
            all_patterns = sum(all(rows[p][x["pattern_id"]]["cal_flag"] == -1
                                   for x in frozen["patterns"])
                               for p in list(rows) if len(rows[p]) == len(frozen["patterns"]))
            any_one = sum(any(x["cal_flag"] == -1 for x in rows[p].values())
                          for p in rows if len(rows[p]) == len(frozen["patterns"]))
            print({"SCREENED": position, "all_patterns_nonconvergent": all_patterns,
                   "at_least_one_nonconvergent": any_one}, flush=True)
    return {"plan": frozen, "results": rows}


def full_input_fingerprint(directory: Path) -> str:
    data = input_bytes(directory)
    state = hashlib.sha256()
    for name, digest in sorted(byte_hashes(data).items()):
        state.update(name.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return state.hexdigest()


def finalize() -> dict:
    frozen = pilot_plan()
    verified_path = NEW / "verification.json"
    verified = json.loads(verified_path.read_text(encoding="utf-8"))
    assert verified["status"] == "PASS"
    assert verified["parent_groups"] == 60 and verified["negative_cases"] == 600
    intensity_path = ROOT / "artifacts/results/activsg2020-matched-v3-intensity.json"
    assert VERSION.endswith("v3") and intensity_path.is_file()
    comparison = json.loads(intensity_path.read_text(encoding="utf-8"))
    assert comparison["training_manifest_sha256"] == frozen["training_manifest_sha256"]
    assert Path(comparison["candidate_root"]).resolve() == NEW.resolve()
    training = comparison["training_grid2000"]
    candidate = comparison["candidate_2020"]
    assert training["summary"]["cases"] == 2416 and candidate["summary"]["cases"] == 600
    by_id = {row["case_id"]: row for row in candidate["cases"]}
    assert len(by_id) == 600
    target_deltas = []
    for slot in frozen["slots"]:
        name = f"{slot['parent_id']}_neg_{slot['slot_number']:03d}"
        actual = by_id[name]
        count_error = actual["off_count"] - slot["target_N_off"]
        fraction_error = actual["withdrawn_cap_fraction"] - slot["target_F_cap_off"]
        category = actual["removed_inductive_proxy"] > 1e-12
        assert abs(count_error) <= 2 and abs(fraction_error) <= 0.02 + 1e-10
        assert category == slot["target_inductive_removed"]
        target_deltas.append({"sample_id": name, "reference_case_id": slot["reference_case_id"],
                              "N_off_delta": count_error,
                              "F_cap_off_delta": fraction_error,
                              "inductive_category_matched": True})
    quantile_deltas = {}
    for field in ("off_count", "withdrawn_cap_fraction"):
        limit = 2 if field == "off_count" else 0.02
        quantile_deltas[field] = {}
        for percentile in ("p10", "p25", "median", "p75", "p90"):
            original = training["summary"][field][percentile]
            current = candidate["summary"][field][percentile]
            delta = current - original
            assert abs(delta) <= limit + 1e-10
            quantile_deltas[field][percentile] = {"training": original,
                                                  "matched": current, "delta": delta}
    full_cap_count = sum(abs(row["withdrawn_cap_fraction"] - 1) <= 1e-10
                         for row in candidate["cases"])
    assert full_cap_count <= 1
    assert max(row["off_count"] for row in candidate["cases"]) <= 147
    old_fingerprints = set()
    for group in OLD.iterdir():
        if not group.is_dir():
            continue
        for negative in (group / "negatives").iterdir():
            if negative.is_dir():
                old_fingerprints.add(full_input_fingerprint(negative / "input"))
    new_fingerprints = [full_input_fingerprint(
        NEW / slot["parent_id"] / "negatives" / f"neg_{slot['slot_number']:03d}" / "input")
        for slot in frozen["slots"]]
    assert len(old_fingerprints) == len(new_fingerprints) == len(set(new_fingerprints)) == 600
    assert not old_fingerprints.intersection(new_fingerprints)
    old_hash, old_count, old_bytes = old_tree_digest(OLD)
    assert (old_hash, old_count, old_bytes) == (
        frozen["old_tree_sha256"], frozen["old_file_count"], frozen["old_total_bytes"])
    assert file_hash(PAUSED_RESULT) == frozen["paused_test_result_sha256"]
    assert file_hash(BASELINE_REPORT) == frozen["training_intensity_report_sha256"]
    report = {"status": "PASS", "version": frozen["version"],
              "parent_groups": 60, "negative_cases": 600,
              "training_cases": 2416,
              "plan_sha256": file_hash(NEW / "generation_plan.json"),
              "verification_sha256": file_hash(verified_path),
              "intensity_report_sha256": file_hash(intensity_path),
              "old_tree_sha256_unchanged": old_hash,
              "old_results_sha256_unchanged": True,
              "matched_pair_tolerance": {"N_off_abs_max": 2, "F_cap_off_abs_max": 0.02,
                                          "actual_N_off_abs_max": max(abs(x["N_off_delta"]) for x in target_deltas),
                                          "actual_F_cap_off_abs_max": max(abs(x["F_cap_off_delta"]) for x in target_deltas)},
              "quantile_deltas": quantile_deltas,
              "full_capacitive_withdrawal_count": full_cap_count,
              "max_N_off": max(row["off_count"] for row in candidate["cases"]),
              "exact_old_input_overlap": 0,
              "new_full_inputs_unique": True,
              "inductive_removed_category_pairwise_matched": True,
              "inductive_any_training": sum(x["removed_inductive_proxy"] > 1e-12
                                            for x in training["cases"]),
              "inductive_any_matched": sum(x["removed_inductive_proxy"] > 1e-12
                                           for x in candidate["cases"]),
              "caution": ("Post-hoc independent benchmark from ACTIVSg2000, extended by 20 buses; "
                          "not a completely unrelated topology and not a mathematical no-solution proof."),
              "target_deltas": target_deltas}
    atomic_json(NEW / "quality_gate.json", report)
    readme = f"""# ACTIVSg2020 强度匹配版（v3）

已验收 60 个不同的 2020 节点收敛父组、每父 10 个 SH Mark-only 不收敛子样本，共 600 例。版本标注：**ACTIVSg2000 衍生、2020 节点、强度匹配版**。

`generation_plan.json` 是事先冻结的只读 600 槽位目标表；`sample_N/baseline/input/` 是父基态，`sample_N/negatives/neg_XXX/input/` 是负样本输入，旁边 `result/`、`diagnostics/repeat/`、`diagnostics/restore/` 保存新鲜 PSASP 输出。`verification.json` 是逐例输入与标签审计，`quality_gate.json` 是强度和去重门禁，项目 `artifacts/results/activsg2020-matched-v3-intensity.json` 是指定对比工具生成的报告。拒绝记录与完整求解审计位于 `artifacts/activsg2020_matched_v3_build/`。

每例仅将 CaseAdapter 识别且在父组白名单内的 LF.L2 同母线 SH Mark 从 1 改为 0，不改其他输入字节。父基态新鲜收敛，负样本主算与独立重复均由指定 WMLFRTMsg.exe 返回 LF.CAL=-1，恢复原 Mark 后 LF.CAL=1。许可证或超时不当作负样本。全部 600 例的成对数量目标误差不超过 2、成对电容性电纳代理比例误差不超过 0.02，电抗退出有无类别一致。

训练集与本版的退出数量 P10/P25/中位数/P75/P90 分别为 84/127/135/140/143 和 83/127/135/140/143；电容性电纳代理损失比例分别为 57.5%/86.5%/91.7%/95.7%/97.7% 和 55.7%/85.9%/91.5%/95.3%/97.4%。本版退出数最大 {report['max_N_off']}，全部电容退出 {full_cap_count} 例，与旧测试输入精确重复 0 例。

这是一套在观察到旧版早期测试结果后设计的事后独立基准。生成及父组筛选不使用已冻结模型的推理成绩；不得把其结果解释为完全无关网架的零样本泛化，也不能把数值不收敛解释为数学无解或动态失稳。旧 600 例及已暂停测试结果均未覆盖。
"""
    (NEW / "README_强度匹配版.md").write_text(readme, encoding="utf-8")
    atomic_json(BUILD / "status.json", {"status": "QUALITY_GATE_PASS",
                                        "parent_groups": 60, "negative_cases": 600,
                                        "quality_gate_sha256": file_hash(NEW / "quality_gate.json")})
    print({"QUALITY_GATE": "PASS", "parents": 60, "negatives": 600,
           "full_cap_off": full_cap_count, "old_overlap": 0}, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("plan", "pilot", "generate", "verify",
                                            "screen-plan", "screen", "probe-transfer",
                                            "finalize"))
    parser.add_argument("--max-attempts", type=int, default=64)
    parser.add_argument("--parent")
    parser.add_argument("--slot-number", type=int)
    args = parser.parse_args()
    if args.command == "plan":
        plan()
    elif args.command == "pilot":
        pilot(args.max_attempts)
    elif args.command == "generate":
        generate(args.max_attempts)
    elif args.command == "verify":
        verify_all()
    elif args.command == "screen-plan":
        selection_screen_plan()
    elif args.command == "screen":
        screen_parents()
    elif args.command == "probe-transfer":
        assert args.parent and args.slot_number and 1 <= args.slot_number <= 10
        probe_transfer(args.parent, args.slot_number)
    elif args.command == "finalize":
        finalize()


if __name__ == "__main__":
    main()
