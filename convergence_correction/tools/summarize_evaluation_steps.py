"""Read saved graph results and write a separate R04 action-statistics report.

Run: python -m tools.summarize_evaluation_steps --input RESULT.json --output NEW.json
No solver, model loading, input data, or changes to the original result are needed.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from rl_poweradj.evaluation_metrics import STEP_METRIC_DEFINITIONS, summarize_action_steps


def summarize_result(payload):
    rows = payload["cases"]
    result = {"source_status": payload.get("status"),
              "scope": "recorded completed cases only; no unrecorded cases or calls inferred",
              "recorded_slots": len(rows),
              "step_metric_definitions": dict(STEP_METRIC_DEFINITIONS),
              **summarize_action_steps(rows)}
    for field, output in (("family", "families"), ("parent_case_id", "parents")):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row.get(field, "unspecified"))].append(row)
        result[output] = {key: summarize_action_steps(group) for key, group in sorted(groups.items())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.output.exists():
        parser.error("Output must be a new file, never the original result")
    raw = args.input.read_bytes()
    report = summarize_result(json.loads(raw.decode("utf-8-sig")))
    report["source"] = str(args.input)
    report["source_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
