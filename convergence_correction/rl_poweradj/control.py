from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ControlAction:
    device_id: str
    target_status: int

    def __post_init__(self) -> None:
        if self.target_status not in (0, 1):
            raise ValueError("Control target_status must be zero or one")


@dataclass(frozen=True)
class ControlPolicy:
    """Versioned, explicit authorization for reactive-control actions."""

    schema_version: int
    actions: tuple[ControlAction, ...]
    source: str | None = None

    @classmethod
    def from_dict(cls, value: dict, source: str | None = None) -> "ControlPolicy":
        version = int(value.get("schema_version", 1))
        if version != 1:
            raise ValueError(f"Unsupported control-policy schema version: {version}")
        devices: dict[str, tuple[int, ...]] = {}
        for item in value.get("devices", []):
            identifier = str(item["device_id"]).strip()
            devices[identifier] = tuple(int(x) for x in item.get("allowed_targets", (0, 1)))
        for group in value.get("ranges", []):
            prefix = str(group["prefix"])
            targets = tuple(int(x) for x in group.get("allowed_targets", (0, 1)))
            for index in range(int(group["start"]), int(group["end"]) + 1):
                devices[f"{prefix}{index}"] = targets
        actions = tuple(
            ControlAction(identifier, target)
            for identifier in sorted(devices, key=str.casefold)
            for target in sorted(set(devices[identifier]))
        )
        if not actions:
            raise ValueError("Control policy contains no authorized actions")
        if len(actions) != len(set(actions)):
            raise ValueError("Control policy contains duplicate actions")
        return cls(version, actions, source)

    @classmethod
    def load(cls, path: Path | str) -> "ControlPolicy":
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Control policy does not exist: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), str(path))

    @classmethod
    def allow_all(cls, device_ids: Iterable[str]) -> "ControlPolicy":
        return cls(
            1,
            tuple(
                ControlAction(identifier, target)
                for identifier in sorted(set(device_ids), key=str.casefold)
                for target in (0, 1)
            ),
            "generated:allow-all",
        )

    def validate(self, discovered_device_ids: Iterable[str]) -> None:
        discovered = set(discovered_device_ids)
        unknown = sorted(
            {action.device_id for action in self.actions} - discovered,
            key=str.casefold,
        )
        if unknown:
            preview = ", ".join(unknown[:5])
            raise ValueError(f"Control policy references unknown shunt devices: {preview}")

    @property
    def digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "actions": [
                [action.device_id, action.target_status] for action in self.actions
            ],
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

