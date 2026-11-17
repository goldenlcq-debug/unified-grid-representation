"""Compare SH Mark-withdrawal intensity with formal 2000-bus training cases.

Read-only on sample inputs. The output is an audit artifact, not a PSASP label.
For self-loop LF.L2 devices, -1/X is a signed susceptance proxy; positive
values correspond to capacitive support at nominal voltage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median

from rl_poweradj.psasp import read_records
from rl_poweradj.run_store import atomic_json, file_hash


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "artifacts/datasets/multifamily-joint-v1/train-validation-manifest.json"
SOURCE_BASELINES = ROOT / "lfsamples/lfsample-ok"


def rows(path: Path) -> list[list]:
    return read_records(path / "LF.L2")


def metric(parent: list[list], child: list[list], name: str) -> dict:
    if len(parent) != len(child):
        raise ValueError(f"LF.L2 row count changed: {name}")
    available = 0.0
    cap_lost = 0.0
    inductive_removed = 0.0
    off_count = 0
    off_cap_count = 0
    for before, after in zip(parent, child):
        if before[1:] != after[1:]:
            raise ValueError(f"Non-Mark LF.L2 field changed: {name}")
        first, second = int(before[0]), int(after[0])
        if first not in (0, 1) or second not in (0, 1):
            raise ValueError(f"Non-binary Mark: {name}")
        if first == 0 and second == 1:
            raise ValueError(f"This comparison expects only 1-to-0 withdrawals: {name}")
        shunt = abs(int(before[1])) == abs(int(before[2]))
        if first != second and not shunt:
            raise ValueError(f"Non-shunt Mark changed: {name}")
        if not shunt:
            continue
        x = float(before[5])
        if abs(x) < 1e-12:
            raise ValueError(f"Zero shunt X: {name}")
        proxy = -1.0 / x
        if first == 1 and proxy > 0:
            available += proxy
        if first == 1 and second == 0:
            off_count += 1
            if proxy > 0:
                off_cap_count += 1
                cap_lost += proxy
            else:
                inductive_removed += -proxy
    if available <= 0:
        raise ValueError(f"No online capacitive support in baseline: {name}")
    return {"case_id": name, "off_count": off_count,
            "off_cap_count": off_cap_count,
            "withdrawn_cap_proxy": cap_lost,
            "baseline_cap_proxy": available,
            "withdrawn_cap_fraction": cap_lost / available,
            "removed_inductive_proxy": inductive_removed}


def describe(values: list[dict]) -> dict:
    fields = ("off_count", "off_cap_count", "withdrawn_cap_proxy",
              "withdrawn_cap_fraction", "removed_inductive_proxy")
    summary = {"cases": len(values)}
    for field in fields:
        ordered = sorted(row[field] for row in values)
        if not ordered:
            raise ValueError("Empty intensity cohort")
        summary[field] = {
            "min": ordered[0], "p10": ordered[math.ceil(len(ordered) * 0.1) - 1],
            "p25": ordered[math.ceil(len(ordered) * 0.25) - 1],
            "median": median(ordered), "mean": mean(ordered),
            "p75": ordered[math.ceil(len(ordered) * 0.75) - 1],
            "p90": ordered[math.ceil(len(ordered) * 0.9) - 1],
            "max": ordered[-1],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_root = args.candidate_root.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    train_manifest = json.loads(TRAIN.read_text(encoding="utf-8"))
    train_cases = [item for item in train_manifest["cases"]
                   if item["grid_family_id"] == "grid_2000" and item["split"] == "train"]
    parent_cache = {name: rows(SOURCE_BASELINES / name)
                    for name in {item["parent_case_id"] for item in train_cases}}
    training = [metric(parent_cache[item["parent_case_id"]], rows(Path(item["source"])),
                       item["case_id"]) for item in train_cases]
    groups = sorted((item for item in candidate_root.iterdir() if item.is_dir()),
                    key=lambda item: int(item.name.removeprefix("sample_")))
    candidate: list[dict] = []
    for group in groups:
        parent = rows(group / "baseline/input")
        negatives = sorted((item for item in (group / "negatives").iterdir() if item.is_dir()),
                           key=lambda item: item.name)
        for negative in negatives:
            candidate.append(metric(parent, rows(negative / "input"),
                                    f"{group.name}_{negative.name}"))
    payload = {
        "schema": 1,
        "definition": {
            "off_count": "Same-bus LF.L2 Mark 1-to-0 count relative to converged parent",
            "withdrawn_cap_proxy": "Sum of positive -1/X for withdrawn capacitive self-loops; not actual solved Mvar",
            "withdrawn_cap_fraction": "Withdrawn capacitive proxy divided by all online baseline capacitive proxy",
            "removed_inductive_proxy": "Absolute -1/X of withdrawn inductive self-loops, reported separately",
        },
        "training_manifest_sha256": file_hash(TRAIN),
        "candidate_root": str(candidate_root),
        "training_grid2000": {"summary": describe(training), "cases": training},
        "candidate_2020": {"summary": describe(candidate), "cases": candidate},
    }
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output),
                      "training_cases": len(training), "candidate_cases": len(candidate)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
