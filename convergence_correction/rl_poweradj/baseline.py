from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    converged: bool
    actions: tuple[int, ...]
    device_actions: tuple[tuple[str, int], ...]
    final_error: float | None
    evaluated_sequences: int
    environment_steps: int


@dataclass(frozen=True)
class _Node:
    actions: tuple[int, ...]
    converged: bool
    error: float | None
    action_mask: np.ndarray
    device_actions: tuple[tuple[str, int], ...]


def _run_sequence(
    env_factory: Callable[[], Any],
    case_index: int,
    actions: tuple[int, ...],
) -> _Node:
    env = env_factory()
    device_actions: list[tuple[str, int]] = []
    try:
        _state, info = env.reset(case_index=case_index)
        converged = bool(info["converged"])
        for action in actions:
            if converged or not bool(info["action_mask"][action]):
                return _Node(actions, False, None, info["action_mask"], tuple(device_actions))
            _state, _reward, terminated, truncated, info = env.step(action)
            device_actions.append((str(info["device_id"]), int(info["target_status"])))
            converged = bool(terminated)
            if terminated or truncated:
                break
        raw_error = info.get("error")
        return _Node(
            actions,
            converged,
            None if raw_error is None else float(raw_error),
            np.asarray(info["action_mask"], dtype=bool),
            tuple(device_actions),
        )
    finally:
        env.close()


def beam_search(
    env_factory: Callable[[], Any],
    case_index: int = 0,
    max_steps: int = 10,
    beam_width: int = 4,
) -> SearchResult:
    """Deterministic residual-ranked search using the production environment."""
    if max_steps <= 0 or beam_width <= 0:
        raise ValueError("max_steps and beam_width must be positive")
    root = _run_sequence(env_factory, case_index, ())
    if root.converged:
        return SearchResult(True, (), (), root.error, 1, 0)
    frontier = [root]
    evaluated = 1
    environment_steps = 0
    for _depth in range(max_steps):
        candidates: list[_Node] = []
        for node in frontier:
            for action in np.flatnonzero(node.action_mask):
                sequence = node.actions + (int(action),)
                child = _run_sequence(env_factory, case_index, sequence)
                evaluated += 1
                environment_steps += len(sequence)
                if child.converged:
                    return SearchResult(
                        True, child.actions, child.device_actions, child.error,
                        evaluated, environment_steps,
                    )
                if child.error is not None:
                    candidates.append(child)
        if not candidates:
            break
        candidates.sort(key=lambda item: (abs(float(item.error)), item.actions))
        frontier = candidates[:beam_width]
    best = min(
        frontier,
        key=lambda item: (
            float("inf") if item.error is None else abs(float(item.error)), item.actions
        ),
    )
    return SearchResult(
        False, best.actions, best.device_actions, best.error,
        evaluated, environment_steps,
    )


def greedy_search(
    env_factory: Callable[[], Any],
    case_index: int = 0,
    max_steps: int = 10,
) -> SearchResult:
    env = env_factory()
    evaluated = 1
    actions: list[int] = []
    device_actions: list[tuple[str, int]] = []
    try:
        _state, info = env.reset(case_index=case_index)
        if info["converged"]:
            return SearchResult(True, (), (), float(info["error"]), evaluated, 0)
        for _ in range(max_steps):
            valid = np.flatnonzero(info["action_mask"])
            if not len(valid):
                break
            if not hasattr(env, "preview_action"):
                return beam_search(env_factory, case_index, max_steps, beam_width=1)
            candidates = []
            for action in valid:
                result = env.preview_action(int(action))
                evaluated += 1
                candidates.append((not result.converged, abs(float(result.error)), int(action)))
            _failed, _error, selected = min(candidates)
            actions.append(selected)
            _state, _reward, terminated, truncated, info = env.step(selected)
            identifier, target = env.registry.action_targets[selected]
            device_actions.append((identifier, target))
            if terminated or truncated:
                break
        raw_error = info.get("error")
        return SearchResult(
            bool(info["converged"]), tuple(actions), tuple(device_actions),
            None if raw_error is None else float(raw_error), evaluated, len(actions),
        )
    finally:
        env.close()
