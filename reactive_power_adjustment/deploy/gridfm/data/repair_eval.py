"""A3 规则潮流修复的单例/批量评测器。

原始方式只读；每个样本先复制到独立产物目录，再运行统一 repair_pipeline。结果按
输入、配置和代码三类哈希断点续跑，并在每个样本结束后立即重建汇总文件。
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..fsutil import robust_copytree
from .repair import DEFAULT_REPAIR_CONFIG, RepairTrace, RuleRepairConfig
from .synth import repair_case_convergent


EVAL_SCHEMA_VERSION = "a_repair_eval_v1"
RESULT_FILE = "result.json"


@dataclass(frozen=True)
class CaseSpec:
    sample_id: str
    case_dir: Path
    family: str = ""
    protected_buses: tuple[int, ...] = ()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(content)
    temp.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8"))


def hash_case_tree(case_dir: Path) -> Dict[str, Any]:
    """逐文件及整树哈希；相对路径参与整树摘要，目录移动不改变结果。"""
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        raise FileNotFoundError(f"方式目录不存在:{case_dir}")
    files: Dict[str, str] = {}
    aggregate = hashlib.sha256()
    for path in sorted(p for p in case_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(case_dir).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[rel] = digest
        aggregate.update(rel.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    if not files:
        raise ValueError(f"方式目录为空:{case_dir}")
    return {"tree_sha256": aggregate.hexdigest(), "files": files}


def code_fingerprint(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """对 A3 运行语义相关源码做内容指纹，同时记录 Git HEAD。"""
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    relative = (
        "gridfm/data/repair.py",
        "gridfm/data/synth.py",
        "gridfm/data/repair_eval.py",
        "scripts/eval_pf_repair.py",
    )
    files = {}
    aggregate = hashlib.sha256()
    for name in relative:
        path = root / name
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[name] = digest
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        head = ""
    return {"sha256": aggregate.hexdigest(), "git_head": head, "files": files}


def environment_record() -> Dict[str, Any]:
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "psasp_bin": os.environ.get("GRIDFM_PSASP_BIN", ""),
        "data_root": os.environ.get("GRIDFM_DATA_ROOT", ""),
    }


def discover_library_cases(
    root: Path,
    libraries: Sequence[str] = ("lib_cepri36", "lib118"),
    solvable_only: bool = True,
) -> List[CaseSpec]:
    """按 manifest 顺序发现真实卡片目录，不从裸目录 glob 静默增样。"""
    root = Path(root)
    cases: List[CaseSpec] = []
    seen = set()
    for library in libraries:
        lib_dir = root / library
        manifest = lib_dir / "manifest.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"缺少 manifest:{manifest}")
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if solvable_only and row.get("solvable", "1").strip() == "0":
                continue
            sample_id = row.get("sample_id", "").strip()
            if not sample_id:
                raise ValueError(f"{manifest} 存在空 sample_id")
            if sample_id in seen:
                raise ValueError(f"sample_id 重复:{sample_id}")
            case_dir = lib_dir / "cases" / sample_id
            if not case_dir.is_dir():
                raise FileNotFoundError(f"样本卡片目录不存在:{case_dir}")
            seen.add(sample_id)
            family = row.get("family_id", "").strip() or library
            protected = (119,) if library == "lib118" else ()
            cases.append(CaseSpec(sample_id, case_dir, family, protected))
    return cases


def _run_key(input_hash: str, config_hash: str, code_hash: str) -> str:
    return hashlib.sha256(_json_bytes({
        "input": input_hash, "config": config_hash, "code": code_hash
    })).hexdigest()


def _archive_stale(artifact_dir: Path, out_root: Path, old_key: str) -> Path:
    stale_root = out_root / "_stale"
    stale_root.mkdir(parents=True, exist_ok=True)
    stem = f"{artifact_dir.name}-{(old_key or 'unknown')[:12]}"
    target = stale_root / stem
    suffix = 1
    while target.exists():
        target = stale_root / f"{stem}-{suffix}"
        suffix += 1
    shutil.move(str(artifact_dir), str(target))
    return target


def _diagnosis_from_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    observations = trace.get("observations", [])
    return {
        "termination_reason": trace.get("termination_reason", ""),
        "final_converged": trace.get("final_converged"),
        "nr_observations": [
            item for item in observations if item.get("source") == "nrpf"
        ],
        "psasp_observations": [
            item for item in observations if item.get("source") == "psasp"
        ],
    }


def evaluate_case(
    spec: CaseSpec,
    out_root: Path,
    solver: Any,
    config: RuleRepairConfig = DEFAULT_REPAIR_CONFIG,
    *,
    resume: bool = True,
    code_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """运行一个样本并立即落盘；算法失败是正常结果，引擎异常单独计数。"""
    out_root = Path(out_root)
    artifact_dir = out_root / "artifacts" / spec.sample_id
    input_hashes = hash_case_tree(spec.case_dir)
    effective = config.with_protected(spec.protected_buses)
    code_info = code_info or code_fingerprint()
    run_key = _run_key(
        input_hashes["tree_sha256"], effective.content_hash(),
        code_info["sha256"])

    old_result_path = artifact_dir / RESULT_FILE
    if old_result_path.is_file():
        old_result = json.loads(old_result_path.read_text(encoding="utf-8"))
        if resume and old_result.get("run_key") == run_key:
            return {**old_result, "resume_skipped": True}
        _archive_stale(artifact_dir, out_root, old_result.get("run_key", ""))

    final_cards = artifact_dir / "final_cards"
    robust_copytree(spec.case_dir, final_cards)
    _write_json(artifact_dir / "input_hashes.json", input_hashes)
    trace = RepairTrace.for_config(effective)
    started = time.perf_counter()
    status = "engine_error"
    info: Dict[str, Any] = {}
    error: Optional[Dict[str, str]] = None
    try:
        info = repair_case_convergent(
            final_cards, solver, protected_buses=spec.protected_buses,
            config=config, trace=trace)
        status = "success" if bool(info.get("nr_ok")) else "algorithm_failure"
    except (FileNotFoundError, ValueError) as exc:
        status = "bad_input_data"
        error = {"type": type(exc).__name__, "message": str(exc)}
        (out_root / "logs").mkdir(parents=True, exist_ok=True)
        _atomic_write(
            out_root / "logs" / f"{spec.sample_id}.log",
            traceback.format_exc().encode("utf-8"))
    except Exception as exc:
        status = "engine_error"
        error = {"type": type(exc).__name__, "message": str(exc)}
        (out_root / "logs").mkdir(parents=True, exist_ok=True)
        _atomic_write(
            out_root / "logs" / f"{spec.sample_id}.log",
            traceback.format_exc().encode("utf-8"))

    elapsed = time.perf_counter() - started
    output_hashes = hash_case_tree(final_cards)
    donor_after = hash_case_tree(spec.case_dir)
    donor_unchanged = donor_after == input_hashes
    if not donor_unchanged:
        status = "engine_error"
        error = {
            "type": "DonorMutationError",
            "message": "原始 donor 在评测期间发生变化",
        }

    trace_dict = info.get("trace", trace.to_dict())
    result = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "sample_id": spec.sample_id,
        "family": spec.family,
        "status": status,
        "run_key": run_key,
        "input_tree_sha256": input_hashes["tree_sha256"],
        "output_tree_sha256": output_hashes["tree_sha256"],
        "config_hash": effective.content_hash(),
        "code_hash": code_info["sha256"],
        "protected_buses": list(effective.protected_buses),
        "donor_unchanged": donor_unchanged,
        "elapsed_seconds": elapsed,
        "error": error,
        "repair": {k: v for k, v in info.items() if k != "trace"},
        "trace": trace_dict,
    }
    _write_json(artifact_dir / "output_hashes.json", output_hashes)
    _write_json(artifact_dir / "diagnosis.json", _diagnosis_from_trace(trace_dict))
    _write_json(artifact_dir / "provenance.json", {
        "result": result,
        "input_case": str(spec.case_dir.resolve()),
        "code": code_info,
        "config": effective.to_dict(),
    })
    _write_json(artifact_dir / RESULT_FILE, result)
    _write_json(out_root / "logs" / f"{spec.sample_id}.json", {
        "sample_id": spec.sample_id,
        "status": status,
        "elapsed_seconds": elapsed,
        "termination_reason": trace_dict.get("termination_reason", ""),
        "error": error,
    })
    return result


def write_aggregates(out_root: Path) -> Dict[str, Any]:
    """只从逐例 result.json 重建汇总，保证续跑不重复计数。"""
    out_root = Path(out_root)
    results = []
    for path in sorted((out_root / "artifacts").glob(f"*/{RESULT_FILE}")):
        results.append(json.loads(path.read_text(encoding="utf-8")))

    case_lines = [
        json.dumps({k: v for k, v in item.items() if k != "trace"},
                   ensure_ascii=False, sort_keys=True)
        for item in results
    ]
    _atomic_write(
        out_root / "cases.jsonl",
        (("\n".join(case_lines) + "\n") if case_lines else "").encode("utf-8"))

    action_lines = []
    for item in results:
        for index, action in enumerate(item.get("trace", {}).get("actions", [])):
            action_lines.append(json.dumps({
                "sample_id": item["sample_id"], "family": item["family"],
                "action_index": index, **action,
            }, ensure_ascii=False, sort_keys=True))
    _atomic_write(
        out_root / "actions.jsonl",
        (("\n".join(action_lines) + "\n") if action_lines else "").encode("utf-8"))

    failures = [item for item in results if item["status"] != "success"]
    failure_rows = ["sample_id,family,status,error_type,error_message"]
    for item in failures:
        err = item.get("error") or {}
        values = [
            item["sample_id"], item["family"], item["status"],
            err.get("type", ""), err.get("message", ""),
        ]
        failure_rows.append(",".join(
            '"' + str(value).replace('"', '""') + '"' for value in values))
    _atomic_write(
        out_root / "failures.csv",
        ("\n".join(failure_rows) + "\n").encode("utf-8"))

    counts = {
        name: sum(item["status"] == name for item in results)
        for name in ("success", "algorithm_failure", "bad_input_data", "engine_error")
    }
    summary = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "total": len(results),
        "counts": counts,
        "families": {
            family: {
                name: sum(
                    item["family"] == family and item["status"] == name
                    for item in results)
                for name in counts
            }
            for family in sorted({item["family"] for item in results})
        },
    }
    _write_json(out_root / "summary.json", summary)
    return summary


def evaluate_cases(
    cases: Iterable[CaseSpec],
    out_root: Path,
    solver: Any,
    config: RuleRepairConfig = DEFAULT_REPAIR_CONFIG,
    *,
    resume: bool = True,
    code_info: Optional[Dict[str, Any]] = None,
    suite: str = "a_v1",
) -> Dict[str, Any]:
    """顺序批量运行；每例完成即刷新汇总，因此进程中断后可从已落盘结果续跑。"""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    code_info = code_info or code_fingerprint()
    _write_json(out_root / "config.json", {
        "suite": suite,
        "config": config.to_dict(),
        "config_hash": config.content_hash(),
        "code": code_info,
    })
    if not (out_root / "environment.json").exists():
        _write_json(out_root / "environment.json", environment_record())
    for spec in cases:
        evaluate_case(
            spec, out_root, solver, config, resume=resume, code_info=code_info)
        write_aggregates(out_root)
    return write_aggregates(out_root)
