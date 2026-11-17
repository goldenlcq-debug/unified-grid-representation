"""Create an inference-only unified-encoder bridge without retraining the DQN."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rl_poweradj.dqn import DQNAgent
from rl_poweradj.foundation import FOUNDATION_FEATURE_SIZE, UnifiedFrozenEncoder


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_state_dict(state: dict[str, torch.Tensor], base_size: int) -> dict[str, torch.Tensor]:
    result = {key: value.detach().cpu().clone() for key, value in state.items()}
    key = "layers.0.weight"
    old = result[key]
    if tuple(old.shape) != (256, base_size):
        raise ValueError(f"Unexpected source first-layer shape: {tuple(old.shape)}")
    expanded = old.new_zeros((old.shape[0], base_size + FOUNDATION_FEATURE_SIZE))
    expanded[:, :base_size] = old
    result[key] = expanded
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate a 17096-D checkpoint to the unified frozen-encoder bridge")
    parser.add_argument("--input", type=Path, default=Path("artifacts/models/dqn-best.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/dqn-best-foundation.pt"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("Bridge migration must not overwrite the source checkpoint")
    if output.exists():
        raise FileExistsError(f"Bridge output already exists: {output}")

    old, _replay, metrics, episode = DQNAgent.load_checkpoint(source, "cpu")
    if old.observation_size != 17096 or old.action_size != 302:
        raise ValueError("Source checkpoint is not the audited 17096x302 model")
    encoder = UnifiedFrozenEncoder.from_project(Path(__file__).resolve().parents[1], device=args.device)
    metadata = dict(old.metadata)
    metadata["encoder"] = dict(encoder.contract)
    metadata["foundation_usage"] = {
        "mode": "zero_weight_bridge",
        "foundation_effective": False,
        "inference_only": True,
        "source_checkpoint_sha256": sha256_file(source),
        "statement": "Global tokens are computed and validated but have zero first-layer weights.",
    }
    migrated = DQNAgent(
        old.observation_size + FOUNDATION_FEATURE_SIZE,
        old.action_size,
        gamma=old.gamma,
        target_update_interval=old.target_update_interval,
        device="cpu",
        metadata=metadata,
    )
    migrated.online.load_state_dict(expand_state_dict(old.online.state_dict(), old.observation_size))
    migrated.target.load_state_dict(expand_state_dict(old.target.state_dict(), old.observation_size))
    migrated.updates = old.updates
    migrated.save(output, replay=None, metrics=metrics, episode=episode)

    with torch.no_grad():
        base = torch.randn(8, old.observation_size)
        tokens = torch.randn(8, FOUNDATION_FEATURE_SIZE)
        expected = old.online(base)
        actual = migrated.online(torch.cat((base, tokens), dim=1))
    max_abs = float((expected - actual).abs().max().item())
    argmax_equal = bool(torch.equal(expected.argmax(1), actual.argmax(1)))
    if max_abs > 1e-5 or not argmax_equal:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"Bridge equivalence check failed: max_abs={max_abs}")
    encoder.verify_frozen()
    manifest = {
        "schema_version": 1,
        "checkpoint": str(output),
        "checkpoint_sha256": sha256_file(output),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256_file(source),
        "observation_size": migrated.observation_size,
        "action_size": migrated.action_size,
        "encoder": encoder.contract,
        "foundation_effective": False,
        "inference_only": True,
        "random_probe_q_max_abs": max_abs,
        "equivalence_tolerance": 1e-5,
        "argmax_equal": argmax_equal,
        "retrained": False,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
