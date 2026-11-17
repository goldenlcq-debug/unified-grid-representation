from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rl_poweradj.dqn import DQNAgent
from rl_poweradj.foundation import FoundationPowerAdjustmentEnv, UnifiedFrozenEncoder
from rl_poweradj.model import DeviceRegistry
from rl_poweradj.paths import ProjectPaths


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def q_diagnostics(agent: DQNAgent, state: np.ndarray, mask: np.ndarray) -> dict:
    with torch.no_grad():
        q = agent.online(torch.as_tensor(state, device=agent.device).unsqueeze(0)).squeeze(0)
        q = q.detach().cpu().numpy()
    valid = np.flatnonzero(mask)
    order = valid[np.argsort(q[valid])[::-1]]
    selected = int(order[0])
    second = float(q[order[1]]) if len(order) > 1 else float("nan")
    return {
        "selected_action": selected,
        "selected_q": float(q[selected]),
        "top2_margin": float(q[selected] - second),
        "valid_q_mean": float(np.mean(q[valid])),
        "valid_q_std": float(np.std(q[valid])),
        "valid_action_count": int(len(valid)),
    }


def active_count(env: PowerAdjustmentEnv) -> int:
    return sum(int(row["Mark"]) for row in env._l2_by_id().values())


def summarize(trace: list[dict], initial_error: float | None, success: bool) -> dict:
    actions = [(str(row["device_id"]), int(row["target_status"])) for row in trace]
    devices = Counter(item[0] for item in actions)
    finite = [
        (int(row["step"]), abs(float(row["error_after"])))
        for row in trace
        if row["error_after"] is not None and math.isfinite(float(row["error_after"]))
    ]
    comparisons = [
        row for row in trace
        if row["error_before"] is not None and row["error_after"] is not None
    ]
    best = min(finite, key=lambda item: item[1]) if finite else (None, None)
    return {
        "success": success,
        "steps": len(trace),
        "initial_error": initial_error,
        "final_error": trace[-1]["error_after"] if trace else initial_error,
        "minimum_absolute_error": best[1],
        "minimum_error_step": best[0],
        "unique_devices_acted": len(devices),
        "repeated_device_count": sum(count > 1 for count in devices.values()),
        "maximum_actions_on_one_device": max(devices.values(), default=0),
        "target_on_actions": sum(target == 1 for _device, target in actions),
        "target_off_actions": sum(target == 0 for _device, target in actions),
        "absolute_error_improved_steps": sum(
            abs(float(row["error_after"])) < abs(float(row["error_before"])) for row in comparisons
        ),
        "absolute_error_worsened_steps": sum(
            abs(float(row["error_after"])) > abs(float(row["error_before"])) for row in comparisons
        ),
        "device_action_counts": devices.most_common(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace greedy DQN actions on selected external cases")
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--cases-dir", type=Path, default=Path("testdata2000/samples"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/dqn-best-foundation.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/testdata2000-policy-traces.json"))
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    paths = ProjectPaths.from_root(root)
    agent = DQNAgent.load(args.model, args.device)
    foundation = UnifiedFrozenEncoder.from_project(root, device=args.device)
    registry = DeviceRegistry.from_dict(agent.metadata["registry"])
    case_paths = [(args.cases_dir / name).resolve() for name in args.case]
    if any(not case.is_dir() for case in case_paths):
        missing = [str(case) for case in case_paths if not case.is_dir()]
        raise FileNotFoundError(f"Missing cases: {missing}")

    payload = {
        "schema_version": 1,
        "status": "running",
        "started_at": now(),
        "completed_at": None,
        "model": str(args.model.resolve()),
        "max_steps": args.max_steps,
        "cases": [],
    }
    write_json(args.output, payload)
    try:
        with FoundationPowerAdjustmentEnv(
            paths=paths,
            cases=case_paths,
            registry=registry,
            max_steps=args.max_steps,
            enable_qvio_reward=False,
            foundation_encoder=foundation,
        ) as env:
            for index, case in enumerate(case_paths):
                started = time.monotonic()
                state, info = env.reset(case_index=index)
                initial_error = info.get("error")
                initial_active = active_count(env)
                trace: list[dict] = []
                success = bool(info["converged"])
                while not success and env.steps < args.max_steps and info["action_mask"].any():
                    qinfo = q_diagnostics(agent, state, np.asarray(info["action_mask"], dtype=bool))
                    action = int(qinfo["selected_action"])
                    before = info.get("error")
                    state, reward, terminated, truncated, info = env.step(action)
                    trace.append(
                        {
                            "step": env.steps,
                            "action": action,
                            "device_id": info.get("device_id"),
                            "target_status": info.get("target_status"),
                            "error_before": before,
                            "error_after": info.get("error"),
                            "reward": reward,
                            "active_shunts_after": active_count(env),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            **qinfo,
                        }
                    )
                    success = bool(terminated)
                    if terminated or truncated:
                        break
                payload["cases"].append(
                    {
                        "case": case.name,
                        "duration_seconds": time.monotonic() - started,
                        "initial_active_shunts": initial_active,
                        "final_active_shunts": active_count(env),
                        "summary": summarize(trace, initial_error, success),
                        "trace": trace,
                    }
                )
                write_json(args.output, payload)
                print(json.dumps({"case": case.name, **payload["cases"][-1]["summary"]}, ensure_ascii=False), flush=True)
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(args.output, payload)
        raise
    payload["status"] = "complete"
    payload["completed_at"] = now()
    payload["error"] = None
    write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
