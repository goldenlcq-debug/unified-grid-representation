"""Reproducible family/parent/child sampling for joint-grid training."""
from __future__ import annotations

import hashlib
import json
import numpy as np

TRAIN_FAMILIES = ("grid_2000", "grid_3120", "grid_300", "grid_118")


class MultiFamilySampler:
    def __init__(self, records, seed: int = 0):
        self.groups: dict[str, dict[str, list[int]]] = {family: {} for family in TRAIN_FAMILIES}
        for index, record in enumerate(records):
            if record["split"] != "train":
                continue
            family = record["grid_family_id"]
            if family not in self.groups:
                raise ValueError(f"Unapproved training family: {family}")
            self.groups[family].setdefault(record["parent_case_id"], []).append(index)
        if any(not parents for parents in self.groups.values()):
            raise ValueError("Every training family must contain parents")
        for family in self.groups:
            self.groups[family] = {parent: sorted(indices, key=lambda i: records[i]["case_id"])
                                   for parent, indices in sorted(self.groups[family].items())}
        identity = [(family, parent, [records[i]["case_id"] for i in indices])
                    for family, parents in self.groups.items() for parent, indices in parents.items()]
        self.identity = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
        self.rng = np.random.default_rng(seed)
        self.counts = {family: 0 for family in TRAIN_FAMILIES}

    def next(self) -> int:
        family = TRAIN_FAMILIES[int(self.rng.integers(len(TRAIN_FAMILIES)))]
        parents = self.groups[family]
        parent = tuple(parents)[int(self.rng.integers(len(parents)))]
        children = parents[parent]
        index = children[int(self.rng.integers(len(children)))]
        self.counts[family] += 1
        return index

    def state_dict(self) -> dict:
        return {"identity": self.identity, "rng": self.rng.bit_generator.state,
                "counts": dict(self.counts)}

    def load_state_dict(self, state: dict) -> None:
        if state.get("identity") != self.identity or set(state.get("counts", {})) != set(TRAIN_FAMILIES):
            raise ValueError("Sampler dataset identity changed")
        self.rng.bit_generator.state = state["rng"]
        self.counts = dict(state["counts"])
