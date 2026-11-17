from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rl_poweradj.dqn import DQNAgent, evaluate
from rl_poweradj.environment import PowerAdjustmentEnv
from rl_poweradj.foundation import FoundationPowerAdjustmentEnv, UnifiedFrozenEncoder
from rl_poweradj.input_validation import InputCaseValidator
from rl_poweradj.model import DeviceRegistry, L2_FIELDS, is_shunt, read_table
from rl_poweradj.paths import ProjectPaths, case_input_fingerprint, discover_cases


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def parent_group(case_name: str) -> str:
    match = re.fullmatch(r"sample_(.+)_\d+", case_name)
    return match.group(1) if match else case_name


def reference_fingerprints(catalogs: list[Path], case_root: Path) -> set[str]:
    fingerprints: set[str] = set()
    for catalog in catalogs:
        for raw in catalog.read_text(encoding="utf-8-sig").splitlines():
            name = raw.strip()
            if name:
                fingerprints.add(case_input_fingerprint(case_root / name).hex())
    return fingerprints


def preflight(
    cases: list[Path], registry: DeviceRegistry, reference: set[str]
) -> dict:
    validator = InputCaseValidator()
    controlled = {identifier for identifier, _target in registry.action_targets}
    fingerprints: dict[str, list[str]] = {}
    invalid_active_power: list[dict] = []
    missing_controlled_devices: list[dict] = []

    for case in cases:
        fingerprint = case_input_fingerprint(case).hex()
        fingerprints.setdefault(fingerprint, []).append(case.name)

        validation = validator.validate(case)
        if not validation.active_power_valid:
            invalid_active_power.append(
                {"case": case.name, "errors": list(validation.errors)}
            )

        rows = read_table(case / "LF.L2", L2_FIELDS)
        available = {row.identifier for row in rows if is_shunt(row)}
        missing = sorted(controlled - available, key=str.casefold)
        if missing:
            missing_controlled_devices.append({"case": case.name, "missing": missing})

    duplicates = [
        {"fingerprint": fingerprint, "cases": names}
        for fingerprint, names in sorted(fingerprints.items())
        if len(names) > 1
    ]
    overlap = sorted(
        name
        for fingerprint, names in fingerprints.items()
        if fingerprint in reference
        for name in names
    )
    dataset_digest = hashlib.sha256(
        "\n".join(
            f"{name}:{fingerprint}"
            for fingerprint, names in sorted(fingerprints.items())
            for name in sorted(names, key=str.casefold)
        ).encode("utf-8")
    ).hexdigest()

    result = {
        "case_count": len(cases),
        "unique_input_fingerprints": len(fingerprints),
        "duplicate_input_group_count": len(duplicates),
        "duplicate_case_count": sum(len(item["cases"]) - 1 for item in duplicates),
        "duplicate_input_groups": duplicates,
        "reference_fingerprint_overlap_count": len(overlap),
        "reference_fingerprint_overlap_cases": overlap,
        "parent_group_count": len({parent_group(case.name) for case in cases}),
        "parent_groups": sorted({parent_group(case.name) for case in cases}),
        "active_power_invalid_count": len(invalid_active_power),
        "active_power_invalid_cases": invalid_active_power,
        "controlled_device_count": len(controlled),
        "cases_missing_controlled_devices_count": len(missing_controlled_devices),
        "cases_missing_controlled_devices": missing_controlled_devices,
        "dataset_digest": dataset_digest,
    }
    if invalid_active_power or missing_controlled_devices:
        raise RuntimeError(f"External dataset preflight failed: {result}")
    return result


def summarize(cases: list[dict], horizon: int = 150) -> dict:
    rewards = [float(case["reward"]) for case in cases]
    steps = [int(case["steps"]) for case in cases]
    finite_errors = [
        float(case["final_error"])
        for case in cases
        if case.get("final_error") is not None
        and math.isfinite(float(case["final_error"]))
    ]
    successes = sum(bool(case["success"]) for case in cases)
    initial_converged = sum(
        bool(case["success"]) and int(case["steps"]) == 0 for case in cases
    )
    action_required = len(cases) - initial_converged
    action_required_successes = sum(
        bool(case["success"]) and int(case["steps"]) > 0 for case in cases
    )
    return {
        "episodes": len(cases),
        "successes": successes,
        "success_rate": successes / max(len(cases), 1),
        "initial_converged": initial_converged,
        "action_required": action_required,
        "action_required_successes": action_required_successes,
        "action_required_success_rate": (
            action_required_successes / max(action_required, 1)
        ),
        "mean_reward": float(np.mean(rewards)) if rewards else None,
        "mean_steps": float(np.mean(steps)) if steps else None,
        "max_steps": max(steps, default=0),
        "step_percentiles": {
            key: float(np.percentile(steps, percentile)) if steps else None
            for key, percentile in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))
        },
        "mean_final_error": float(np.mean(finite_errors)) if finite_errors else None,
        "worst_final_error": max(finite_errors, default=None),
        "failed_case_count": len(cases) - successes,
        "full_length_case_count": sum(step >= horizon for step in steps),
        "horizon_success_count": sum(c['success'] and c['steps'] >= horizon for c in cases),
        "horizon_failure_count": sum(not c['success'] and c['steps'] >= horizon for c in cases),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL_POWERADJ model on an external PSASP case directory."
    )
    parser.add_argument("--cases-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference-case-root", type=Path)
    parser.add_argument("--reference-catalog", type=Path, action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=10)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    cases_dir = args.cases_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.resume:
        raise FileExistsError('Existing evaluation requires explicit --resume or a new output path')
    if args.resume and not output_path.is_file():
        raise FileNotFoundError('Resume result file does not exist')
    paths = ProjectPaths.from_root(root)
    solver_hash = sha256_file(paths.solver)
    expected_contract = {'max_steps': args.max_steps, 'device': args.device,
                         'psasp_convergence_gold_standard': True, 'qvio_reward_enabled': False,
                         'solver_sha256': solver_hash}
    cases = discover_cases(cases_dir)
    model_hash = sha256_file(model_path)
    agent = DQNAgent.load(model_path, args.device)
    foundation = UnifiedFrozenEncoder.from_project(root, device=args.device)
    if agent.metadata.get('encoder') != foundation.contract:
        raise ValueError('Model does not match actual unified encoder contract')
    registry_data = agent.metadata.get("registry")
    if registry_data is None:
        raise ValueError("Model checkpoint does not contain a device registry")
    registry = DeviceRegistry.from_dict(registry_data)

    reference: set[str] = set()
    if args.reference_catalog:
        if args.reference_case_root is None:
            raise ValueError("--reference-case-root is required with --reference-catalog")
        reference = reference_fingerprints(
            [path.expanduser().resolve() for path in args.reference_catalog],
            args.reference_case_root.expanduser().resolve(),
        )
    dataset = preflight(cases, registry, reference)

    completed: list[dict] = []
    started_at = utc_now()
    if args.resume and output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        if previous.get('contract') != expected_contract:
            raise ValueError('Evaluation protocol/solver mismatch; use a new output path')
        if previous.get('contract', {}).get('max_steps') != args.max_steps:
            raise ValueError('Existing result uses a different horizon')
        if previous.get('model', {}).get('metadata', {}).get('encoder') != foundation.contract:
            raise ValueError('Existing result uses a different encoder contract')
        if previous.get("model", {}).get("sha256") != model_hash:
            raise ValueError("Existing result uses a different model")
        if previous.get("dataset", {}).get("dataset_digest") != dataset["dataset_digest"]:
            raise ValueError("Existing result uses a different dataset")
        completed = list(previous.get("evaluation", {}).get("cases", []))
        expected = [case.name for case in cases[: len(completed)]]
        actual = [Path(case["source_case"]).name for case in completed]
        if actual != expected:
            raise ValueError("Existing result is not a valid prefix of the current dataset")
        started_at = previous.get("started_at", started_at)

    payload = {
        "schema_version": 1,
        "status": "running",
        "started_at": started_at,
        "updated_at": utc_now(),
        "completed_at": None,
        "project_root": str(root),
        "model": {
            "path": str(model_path),
            "sha256": model_hash,
            "observation_size": agent.observation_size,
            "action_size": agent.action_size,
            "metadata": agent.metadata,
        },
        "contract": expected_contract,
        "dataset": {"path": str(cases_dir), **dataset},
        "completed_cases": len(completed),
        "evaluation": summarize(completed, args.max_steps),
        "error": None,
    }
    atomic_write_json(output_path, payload)

    paths = ProjectPaths.from_root(root)
    try:
        with FoundationPowerAdjustmentEnv(
            paths=paths,
            cases=cases,
            registry=registry,
            max_steps=args.max_steps,
            enable_qvio_reward=False,
            foundation_encoder=foundation,
        ) as env:
            for index in range(len(completed), len(cases)):
                started = time.monotonic()
                result = evaluate(agent, env, 1, args.max_steps, start_case=index)
                case_result = result.cases[0]
                completed.append(
                    {
                        "case_index": case_result.case_index,
                        "source_case": case_result.source_case,
                        "success": case_result.success,
                        "reward": case_result.reward,
                        "steps": case_result.steps,
                        "final_error": case_result.final_error,
                        "duration_seconds": time.monotonic() - started,
                    }
                )
                payload.update(
                    {
                        "updated_at": utc_now(),
                        "completed_cases": len(completed),
                        "evaluation": summarize(completed, args.max_steps),
                    }
                )
                atomic_write_json(output_path, payload)
                if (
                    len(completed) % max(args.progress_interval, 1) == 0
                    or len(completed) == len(cases)
                ):
                    print(
                        json.dumps(
                            {
                                "progress": f"{len(completed)}/{len(cases)}",
                                "successes": payload["evaluation"]["successes"],
                                "case": cases[index].name,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    except Exception as exc:
        payload.update(
            {
                "status": "failed",
                "updated_at": utc_now(),
                "completed_cases": len(completed),
                "evaluation": summarize(completed, args.max_steps),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        atomic_write_json(output_path, payload)
        raise

    payload.update(
        {
            "status": "complete",
            "updated_at": utc_now(),
            "completed_at": utc_now(),
            "completed_cases": len(completed),
            "evaluation": summarize(completed, args.max_steps),
            "error": None,
        }
    )
    atomic_write_json(output_path, payload)
    print(json.dumps({
        "output": str(output_path),
        "status": payload["status"],
        **{key: payload["evaluation"][key] for key in (
            "episodes", "successes", "success_rate", "mean_steps", "max_steps"
        )},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
