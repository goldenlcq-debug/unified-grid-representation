"""Verify the zero-weight foundation bridge on real PSASP input observations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rl_poweradj.dqn import DQNAgent
from rl_poweradj.foundation import UnifiedFrozenEncoder
from rl_poweradj.model import (
    DeviceRegistry,
    L2_FIELDS,
    L3_FIELDS,
    L6_FIELDS,
    encode_observation,
    is_shunt,
    read_dc_plans,
    read_generator_table,
    read_table,
)
from rl_poweradj.paths import ProjectPaths
from rl_poweradj.psasp import convergence_error


def base_observation(case: Path, registry: DeviceRegistry) -> tuple[np.ndarray, np.ndarray]:
    l2 = read_table(case / "LF.L2", L2_FIELDS)
    l3_path, l6_path = case / "LF.L3", case / "LF.L6"
    state = encode_observation(
        registry,
        read_generator_table(case / "LF.L5"),
        l2,
        convergence_error(case),
        read_table(l6_path, L6_FIELDS) if l6_path.is_file() else (),
        read_table(l3_path, L3_FIELDS) if l3_path.is_file() else (),
        read_dc_plans(case),
    )
    by_id = {row.identifier: row for row in l2 if is_shunt(row)}
    mask = np.asarray(
        [int(by_id[identifier]["Mark"]) != target for identifier, target in registry.action_targets],
        dtype=bool,
    )
    return state, mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("artifacts/models/dqn-best.pt"))
    parser.add_argument("--bridge", type=Path, default=Path("artifacts/models/dqn-best-foundation.pt"))
    parser.add_argument("--per-split", type=int, default=5)
    parser.add_argument("--external", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/foundation-bridge-equivalence.json"))
    args = parser.parse_args()
    root = Path.cwd().resolve()
    paths = ProjectPaths.from_root(root)
    source = DQNAgent.load(args.source, "cpu")
    bridge = DQNAgent.load(args.bridge, "cpu")
    encoder = UnifiedFrozenEncoder.from_project(root, "cpu")
    registry = DeviceRegistry.from_dict(source.metadata["registry"])
    cases = []
    for split in ("train", "validation", "test"):
        cases.extend(paths.cases(split)[: args.per_split])
    external_root = root / "testdata2000" / "samples"
    if external_root.is_dir():
        cases.extend(sorted(external_root.iterdir(), key=lambda path: path.name.casefold())[: args.external])

    records = []
    with torch.no_grad():
        for case in cases:
            base, mask = base_observation(case, registry)
            tokens = encoder.encode(case)
            old_q = source.online(torch.as_tensor(base).unsqueeze(0)).squeeze(0)
            new_q = bridge.online(torch.as_tensor(np.concatenate((base, tokens))).unsqueeze(0)).squeeze(0)
            valid = torch.as_tensor(mask)
            old_action = int(old_q.masked_fill(~valid, -torch.inf).argmax().item())
            new_action = int(new_q.masked_fill(~valid, -torch.inf).argmax().item())
            records.append({
                "case": str(case),
                "q_max_abs": float((old_q - new_q).abs().max().item()),
                "old_action": old_action,
                "bridge_action": new_action,
                "action_equal": old_action == new_action,
                "token_norm": float(np.linalg.norm(tokens)),
            })
    payload = {
        "schema_version": 1,
        "case_count": len(records),
        "max_q_abs": max(row["q_max_abs"] for row in records),
        "action_mismatch_count": sum(not row["action_equal"] for row in records),
        "all_tokens_nonzero": all(row["token_norm"] > 0 for row in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("case_count", "max_q_abs", "action_mismatch_count", "all_tokens_nonzero")}, ensure_ascii=False))
    return 0 if payload["action_mismatch_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
