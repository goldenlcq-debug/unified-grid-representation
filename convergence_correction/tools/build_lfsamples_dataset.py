from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from rl_poweradj.input_validation import InputCaseValidator
from rl_poweradj.model import (
    L2_FIELDS, L3_FIELDS, L6_FIELDS, is_shunt, read_generator_table,
    read_table, transformer_identifier,
)
from rl_poweradj.paths import case_input_fingerprint
from rl_poweradj.psasp import convergence_error, convergence_status, read_records


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "lfsamples"
OK_ROOT = SAMPLE_ROOT / "lfsample-ok"
BAD_ROOT = SAMPLE_ROOT / "lfsample-bad"
BAD_PATTERN = re.compile(r"^sample_(\d+)_(\d+)$")
LABEL_AUDIT = SAMPLE_ROOT / "label-audit-20260902.json"
ACTIVE_FILES = ("LF.L0", "LF.L1", "LF.L3", "LF.L5", "LF.L6", "LF.NL4")


def parent_id(path: Path) -> int:
    match = BAD_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Unexpected bad-case name: {path.name}")
    return int(match.group(1))


def active_fingerprint(case: Path) -> str:
    digest = hashlib.sha256()
    for name in ACTIVE_FILES:
        path = case / name
        if path.is_file():
            digest.update(name.encode("ascii"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_l2_only(parent: Path, child: Path) -> None:
    base = read_records(parent / "LF.L2")
    changed = read_records(child / "LF.L2")
    if len(base) != len(changed):
        raise ValueError(f"LF.L2 row count differs: {parent.name} -> {child.name}")
    differences = 0
    for index, (left, right) in enumerate(zip(base, changed)):
        if left == right:
            continue
        differences += 1
        if len(left) != len(right) or left[1:] != right[1:]:
            raise ValueError(f"Non-Mark LF.L2 change at {child.name} row {index}")
        if abs(int(left[1])) != abs(int(left[2])):
            raise ValueError(f"Non-shunt Mark changed at {child.name} row {index}")
        if int(left[0]) not in (0, 1) or int(right[0]) not in (0, 1):
            raise ValueError(f"Unsupported shunt Mark change at {child.name} row {index}")
    if differences == 0:
        raise ValueError(f"Bad case has no LF.L2 Mark change: {child.name}")


def identity_signature(case: Path) -> tuple:
    l2 = read_table(case / "LF.L2", L2_FIELDS)
    l3 = read_table(case / "LF.L3", L3_FIELDS)
    l5 = read_generator_table(case / "LF.L5")
    l6 = read_table(case / "LF.L6", L6_FIELDS)
    return (
        tuple(sorted(row.identifier for row in l5)),
        tuple(sorted(row.identifier for row in l2 if is_shunt(row))),
        tuple(sorted(row.identifier for row in l2 if not is_shunt(row))),
        tuple(sorted(row.identifier for row in l6)),
        tuple(sorted(transformer_identifier(row) for row in l3)),
    )


def assign_groups(groups: list[dict]) -> dict[str, list[dict]]:
    total = sum(item["size"] for item in groups)
    targets = {
        "train": round(total * 0.70),
        "validation": round(total * 0.15),
    }
    targets["test"] = total - targets["train"] - targets["validation"]
    result = {name: [] for name in targets}
    counts = {name: 0 for name in targets}
    # Largest groups first; fingerprint breaks ties independently of directory order.
    ordered = sorted(groups, key=lambda item: (-item["size"], item["fingerprint"]))
    for group in ordered:
        fitting = [
            name for name in targets
            if counts[name] + group["size"] <= targets[name]
        ]
        candidates = fitting or list(targets)
        destination = min(
            candidates,
            key=lambda name: (
                (counts[name] + group["size"]) / max(targets[name], 1),
                name,
            ),
        )
        result[destination].append(group)
        counts[destination] += group["size"]
    return result


def write_catalog(path: Path, cases: list[Path]) -> None:
    path.write_text(
        "".join(f"{case.name}\n" for case in sorted(cases, key=lambda p: p.name.casefold())),
        encoding="utf-8",
    )


def main() -> None:
    ok_cases = sorted((path for path in OK_ROOT.iterdir() if path.is_dir()), key=lambda p: p.name)
    all_bad_cases = sorted((path for path in BAD_ROOT.iterdir() if path.is_dir()), key=lambda p: p.name)
    if len(ok_cases) != 188 or len(all_bad_cases) != 3471:
        raise ValueError(f"Expected 188 ok / 3471 bad, got {len(ok_cases)} / {len(all_bad_cases)}")
    label_audit = json.loads(LABEL_AUDIT.read_text(encoding="utf-8"))
    excluded_converged = set(label_audit["recomputed_converged"])
    recomputed_nonconverged = {
        str(name): float(error)
        for name, error in label_audit["recomputed_nonconverged"].items()
    }
    bad_cases = [case for case in all_bad_cases if case.name not in excluded_converged]

    ok_by_id = {int(path.name.split("_", 1)[1]): path for path in ok_cases}
    children: dict[int, list[Path]] = {identifier: [] for identifier in ok_by_id}
    for child in all_bad_cases:
        identifier = parent_id(child)
        if identifier not in children:
            raise ValueError(f"Bad case has no converged parent: {child.name}")
        children[identifier].append(child)
    if any(not values for values in children.values()):
        raise ValueError("At least one converged parent has no bad child")

    reviewed_names = excluded_converged | set(recomputed_nonconverged)
    expected_identity = identity_signature(ok_cases[0])

    def verify_group(identifier: int):
        parent = ok_by_id[identifier]
        if identity_signature(parent) != expected_identity:
            raise ValueError(f"Parent changes the fixed device identity contract: {parent.name}")
        validation = InputCaseValidator().validate(parent)
        if not validation.active_power_valid:
            raise ValueError(f"Converged parent fails active contract: {parent.name}: {validation.errors}")
        if not convergence_status(parent):
            raise ValueError(f"Converged parent LF.CAL is not converged: {parent.name}")
        fingerprint = active_fingerprint(parent)
        for child in children[identifier]:
            if active_fingerprint(child) != fingerprint:
                raise ValueError(f"Non-LF.L2 input differs from parent: {child.name}")
            verify_l2_only(parent, child)
            if convergence_status(child) and child.name not in reviewed_names:
                raise ValueError(f"Unreviewed bad child LF.CAL says converged: {child.name}")
        usable_children = [
            child for child in children[identifier]
            if child.name not in excluded_converged
        ]
        return validation, {
            "parent_id": identifier,
            "parent": parent,
            "children": usable_children,
            "size": len(usable_children),
            "fingerprint": fingerprint,
        }

    identifiers = sorted(ok_by_id)
    groups = []
    parent_results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (identifier, result) in enumerate(
            zip(identifiers, pool.map(verify_group, identifiers)), 1
        ):
            validation, group = result
            parent_results[identifier] = validation
            groups.append(group)
            if index % 20 == 0:
                print(f"verified parent groups {index}/{len(ok_by_id)}", flush=True)

    partitions = assign_groups(groups)
    assignment = {}
    split_bad_cases = {}
    split_ok_cases = {}
    for split, split_groups in partitions.items():
        split_bad_cases[split] = [case for group in split_groups for case in group["children"]]
        split_ok_cases[split] = [group["parent"] for group in split_groups]
        for group in split_groups:
            assignment[str(group["parent_id"])] = split

    write_catalog(SAMPLE_ROOT / "train-cases.txt", split_bad_cases["train"])
    write_catalog(SAMPLE_ROOT / "validation-cases.txt", split_bad_cases["validation"])
    write_catalog(SAMPLE_ROOT / "test-cases.txt", split_bad_cases["test"])
    write_catalog(SAMPLE_ROOT / "converged-cases.txt", ok_cases)
    for split in partitions:
        write_catalog(SAMPLE_ROOT / f"{split}-parents.txt", split_ok_cases[split])

    records = []
    for case in bad_cases:
        identifier = parent_id(case)
        validation = parent_results[identifier]
        error = recomputed_nonconverged.get(case.name, convergence_error(case))
        records.append({
            "case": case.name,
            "parent_case": ok_by_id[identifier].name,
            "split": assignment[str(identifier)],
            "input_fingerprint": case_input_fingerprint(case).hex(),
            "active_power_valid": True,
            "active_power_errors": [],
            "first_ac_converged": False,
            "first_ac_error": error,
            "category": "requires_reactive_recovery",
            "island_count": len(validation.islands),
        })
    for case in ok_cases:
        identifier = int(case.name.split("_", 1)[1])
        validation = parent_results[identifier]
        records.append({
            "case": case.name,
            "parent_case": case.name,
            "split": assignment[str(identifier)],
            "input_fingerprint": case_input_fingerprint(case).hex(),
            "active_power_valid": True,
            "active_power_errors": [],
            "first_ac_converged": True,
            "first_ac_error": convergence_error(case),
            "category": "raw_converged",
            "island_count": len(validation.islands),
        })
    manifest = {
        "schema_version": 2,
        "dataset": "lfsamples",
        "provenance": "stored PSASP outputs; every bad child verified to differ from its parent only in LF.L2 shunt Mark",
        "grouping": "all children of one converged parent stay in one split",
        "records": records,
    }
    (SAMPLE_ROOT / "formal-flow-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "ok_cases": len(ok_cases),
        "bad_cases": len(bad_cases),
        "bad_directories": len(all_bad_cases),
        "excluded_recomputed_converged": len(excluded_converged),
        "parent_groups": len(groups),
        "variant_count_distribution": dict(sorted(Counter(item["size"] for item in groups).items())),
        "splits": {
            split: {
                "parent_groups": len(partitions[split]),
                "bad_cases": len(split_bad_cases[split]),
                "ok_parents": len(split_ok_cases[split]),
            }
            for split in ("train", "validation", "test")
        },
        "parent_assignment": assignment,
    }
    (SAMPLE_ROOT / "split-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary | {"parent_assignment": "see split-summary.json"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
