"""Family-balanced, resource-bounded replay sampling for variable-size grids."""
from __future__ import annotations

import numpy as np

from .graph_dqn import GraphReplay
from .multifamily_sampler import TRAIN_FAMILIES


class MultiFamilyReplay:
    """Reserve equal transition capacity for each training family."""

    def __init__(self, capacity: int):
        if capacity < len(TRAIN_FAMILIES):
            raise ValueError("Replay capacity must cover all training families")
        base, extra = divmod(capacity, len(TRAIN_FAMILIES))
        self.capacity = capacity
        self.parts = {family: GraphReplay(base + (index < extra))
                      for index, family in enumerate(TRAIN_FAMILIES)}

    @property
    def items(self):
        return [transition for family in TRAIN_FAMILIES for transition in self.parts[family].items]

    def add(self, transition):
        if transition.family not in self.parts:
            raise ValueError(f"Rejected family in replay: {transition.family}")
        self.parts[transition.family].add(transition)


def sample_balanced(replay, rng, count: int, *, node_budget: int, edge_budget: int,
                    action_budget: int):
    if count < 1 or min(node_budget, edge_budget, action_budget) < 1:
        raise ValueError("Positive replay count and budgets required")
    pools = {family: [] for family in TRAIN_FAMILIES}
    limits = np.array([node_budget, edge_budget, action_budget], dtype=np.int64)
    for transition in replay.items:
        if transition.family not in pools:
            raise ValueError(f"Rejected family in replay: {transition.family}")
        cost = np.array([
            len(transition.state.nodes) + len(transition.next_state.nodes),
            len(transition.state.edges) + len(transition.next_state.edges),
            len(transition.state.targets) + len(transition.next_state.targets),
        ], dtype=np.int64)
        if (cost > limits).any():
            raise MemoryError("Replay transition exceeds declared resource budget")
        pools[transition.family].append((transition, cost))
    for family, items in pools.items():
        pools[family] = [items[int(i)] for i in rng.permutation(len(items))]
    selected = []
    used = np.zeros(3, dtype=np.int64)
    while len(selected) < count and any(pools.values()):
        progress = False
        for index in rng.permutation(len(TRAIN_FAMILIES)):
            family = TRAIN_FAMILIES[int(index)]
            pool = pools[family]
            while pool:
                transition, cost = pool.pop()
                if not (used + cost > limits).any():
                    break
            else:
                continue
            selected.append(transition)
            used += cost
            progress = True
            if len(selected) == count:
                break
        if not progress:
            break
    return selected
