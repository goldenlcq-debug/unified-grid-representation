from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rl_poweradj.dqn import DQNAgent
from rl_poweradj.baseline import greedy_search
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Exhaustively evaluate all valid one-step shunt actions")
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--cases-dir", type=Path, default=Path("testdata2000/samples"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/models/dqn-best-foundation.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/testdata2000-one-step-sweep.json"))
    parser.add_argument("--greedy-steps", type=int, default=0)
    args = parser.parse_args()

    root = Path.cwd().resolve()
    agent = DQNAgent.load(args.model, "cpu")
    foundation = UnifiedFrozenEncoder.from_project(root, device="cpu")
    registry = DeviceRegistry.from_dict(agent.metadata["registry"])
    cases = [(args.cases_dir / name).resolve() for name in args.case]
    payload = {
        "schema_version": 1,
        "status": "running",
        "started_at": now(),
        "completed_at": None,
        "cases": [],
    }
    write_json(args.output, payload)
    try:
        with FoundationPowerAdjustmentEnv(
            paths=ProjectPaths.from_root(root),
            cases=cases,
            registry=registry,
            max_steps=1,
            enable_qvio_reward=False,
            foundation_encoder=foundation,
        ) as env:
            for case_index, case in enumerate(cases):
                _state, info = env.reset(case_index=case_index)
                initial_error = info.get("error")
                rows = []
                for action in np.flatnonzero(info["action_mask"]):
                    result = env.preview_action(int(action))
                    identifier, target = registry.action_targets[int(action)]
                    rows.append(
                        {
                            "action": int(action),
                            "device_id": identifier,
                            "target_status": int(target),
                            "converged": bool(result.converged),
                            "error": float(result.error),
                        }
                    )
                rows.sort(key=lambda row: (not row["converged"], abs(row["error"]), row["action"]))
                payload["cases"].append(
                    {
                        "case": case.name,
                        "initial_error": initial_error,
                        "valid_actions": len(rows),
                        "converged_action_count": sum(row["converged"] for row in rows),
                        "best_actions": rows[:20],
                        "actions": sorted(rows, key=lambda row: row["action"]),
                    }
                )
                write_json(args.output, payload)
                print(json.dumps({key: payload["cases"][-1][key] for key in ("case", "initial_error", "valid_actions", "converged_action_count", "best_actions")}, ensure_ascii=False), flush=True)
        if args.greedy_steps > 0:
            payload["residual_greedy"] = []
            for case in cases:
                def factory(case_path=case):
                    return FoundationPowerAdjustmentEnv(
                        paths=ProjectPaths.from_root(root),
                        cases=[case_path],
                        registry=registry,
                        max_steps=args.greedy_steps,
                        enable_qvio_reward=False,
                        foundation_encoder=foundation,
                    )

                result = greedy_search(factory, case_index=0, max_steps=args.greedy_steps)
                payload["residual_greedy"].append({"case": case.name, **asdict(result)})
                write_json(args.output, payload)
                print(json.dumps(payload["residual_greedy"][-1], ensure_ascii=False), flush=True)
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
