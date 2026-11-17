"""Write tested-but-still-nonconverged directories from an adaptive run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

from rl_poweradj.grid_types import digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix-slots", type=int,
                        help="Verified user-requested whole-parent pause boundary")
    args = parser.parse_args()
    result_path, manifest_path, output = (path.resolve() for path in
                                          (args.result, args.manifest, args.output))
    if output.exists():
        raise FileExistsError(output)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = result.get("cases", [])
    source = manifest.get("cases", [])
    protocol = result.get("protocol", {})
    prefix_slots = args.prefix_slots
    if prefix_slots is None:
        coverage_ok = (result.get("status") == "complete"
                       and len(entries) == 600
                       and result.get("tested_cases", 0)
                       + result.get("skipped_cases", 0) == 600)
    else:
        coverage_ok = (prefix_slots in range(10, 601, 10)
                       and result.get("status") == "paused"
                       and result.get("user_requested_stop_after") == prefix_slots
                       and len(entries) == prefix_slots
                       and result.get("tested_cases", 0)
                       + result.get("skipped_cases", 0) == prefix_slots)
    if (not coverage_ok
            or protocol.get("phase") != "external_test_adaptive_parent_skip_v1"
            or protocol.get("manifest_digest") != digest(manifest)
            or protocol.get("zero_updates") is not True
            or result.get("task_parameters_unchanged") is not True
            or result.get("encoder_frozen") is not True
            or len(source) != 600):
        raise ValueError("Adaptive evaluation is not a verified result")
    if [entry["case_id"] for entry in entries] != [item["case_id"] for item in source[:len(entries)]]:
        raise ValueError("Adaptive result differs from manifest order")
    for start in range(0, len(entries), 10):
        group = entries[start:start + 10]
        tested_rows = [item for item in group if item["disposition"] == "tested"]
        skipped_rows = [item for item in group if item["disposition"] == "skipped"]
        if (len(tested_rows), len(skipped_rows)) not in ((4, 6), (10, 0)):
            raise ValueError("Incomplete or invalid adaptive parent group")
        first_four_successes = sum(item["converged"] for item in group[:4])
        if bool(skipped_rows) != (first_four_successes <= 1):
            raise ValueError("Adaptive skip rule differs from saved outcomes")
    failed = [(entry, item) for entry, item in zip(entries, source)
              if entry["disposition"] == "tested" and not entry["converged"]]
    if len(failed) != result.get("nonconverged"):
        raise ValueError("Failed path count differs from adaptive summary")
    directories = [str(Path(item["source"]).resolve().parent) for _, item in failed]
    if len(directories) != len(set(directories)):
        raise ValueError("Duplicate nonconverged directories")
    for directory in directories:
        if not (Path(directory) / "sample_manifest.json").is_file():
            raise FileNotFoundError(directory)
    scope = (f"前 {prefix_slots} 个清单槽位" if prefix_slots is not None
             else "完整 600 个清单槽位")
    lines = ["# ACTIVSg2020_matched_v3 自适应测试后仍未收敛的样本目录", "",
             f"范围：{scope}；其余案例未纳入本次报告。", "",
             f"实际测试 {result['tested_cases']} 例，其中 {len(failed)} 例未收敛；"
             f"另有 {result['skipped_cases']} 例按父组规则跳过，未计为失败。", "",
             "以下每行是一个实际测试后仍未收敛的负样本目录（不是 input 子目录）：", ""]
    lines.extend(f"- `{directory}`" for directory in directories)
    if not directories:
        lines.append("（无）")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    print(json.dumps({"markdown": str(output), "nonconverged": len(failed),
                      "tested": result["tested_cases"], "skipped": result["skipped_cases"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
