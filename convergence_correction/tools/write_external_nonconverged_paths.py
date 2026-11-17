"""Write only the final nonconverged sample directories to a standalone MD."""
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
    parser.add_argument("--expected-cases", type=int, default=600)
    args = parser.parse_args()
    result_path, manifest_path, output = (path.resolve() for path in
                                          (args.result, args.manifest, args.output))
    if output.exists():
        raise FileExistsError(output)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = result.get("cases", [])
    entries = manifest.get("cases", [])
    if (result.get("status") != "complete"
            or result.get("protocol", {}).get("phase") != "external_test"
            or result["protocol"].get("manifest_digest") != digest(manifest)
            or result["protocol"].get("zero_updates") is not True
            or result.get("task_parameters_unchanged") is not True
            or result.get("encoder_frozen") is not True
            or len(cases) != args.expected_cases or len(entries) != args.expected_cases):
        raise ValueError("External test is not a verified complete zero-update run")
    if [case["case_id"] for case in cases] != [entry["case_id"] for entry in entries]:
        raise ValueError("Result order/identity differs from manifest")
    if any(case["initial_converged"] for case in cases):
        raise ValueError("An external negative case was initially converged")
    failed = [(case, entry) for case, entry in zip(cases, entries) if not case["converged"]]
    directories = [str(Path(entry["source"]).resolve().parent) for _, entry in failed]
    if len(directories) != len(set(directories)):
        raise ValueError("Duplicate failed sample directories")
    for directory in directories:
        if not (Path(directory) / "sample_manifest.json").is_file():
            raise FileNotFoundError(f"Sample directory unavailable: {directory}")
    lines = ["# ACTIVSg2020_matched_v3 未收敛潮流样本目录", "",
             f"共 {len(directories)} / {len(cases)} 套在模型调节后仍未收敛。", "",
             "以下每行是一个负样本目录（不是其 input 子目录）：", ""]
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
    print(json.dumps({"markdown": str(output), "nonconverged": len(directories),
                      "tested": len(cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
