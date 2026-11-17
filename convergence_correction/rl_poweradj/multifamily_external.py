"""Identity and input gates for a sealed model's unseen-grid evaluation."""
from __future__ import annotations

import json
from pathlib import Path

from .case_adapter import CaseAdapter
from .data_preflight import input_card_hashes
from .grid_types import digest


def require_unseen_grid(family: str, bus_count: int, *, training_families, training_bus_counts) -> None:
    """Accept any compatible new family and size outside the training set."""
    if not family or family in set(training_families):
        raise ValueError(f"External grid family was used in training: {family}")
    if type(bus_count) is not int or bus_count < 1 or bus_count in set(training_bus_counts):
        raise ValueError(f"External grid node count is not unseen: {bus_count}")


def audit_external_manifest(path, *, training_families, training_bus_counts) -> dict:
    """Verify an external dataset without imposing a whitelist of target grids."""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema") != 2 or data.get("split_protocol") != "unseen_grid_external_test_v1"
            or data.get("phase") != "external_test"):
        raise ValueError("Wrong unseen-grid test protocol")
    records = []
    seen_ids = set()
    for item in data.get("cases", []):
        case_id = item["case_id"]
        if case_id in seen_ids or item.get("split") != "external_test":
            raise ValueError("Duplicate or non-external case")
        seen_ids.add(case_id)
        case = (path.parent / item["source"]).resolve()
        if input_card_hashes(case) != item["input_cards"]:
            raise ValueError(f"Changed external input: {case_id}")
        adapter = CaseAdapter(case)
        require_unseen_grid(item["grid_family_id"], adapter.n,
                            training_families=training_families,
                            training_bus_counts=training_bus_counts)
        if item.get("bus_count") != adapter.n:
            raise ValueError(f"Wrong declared node count: {case_id}")
        adapter.authorize(item["policy"])
        records.append({**item, "source": str(case)})
    if not records:
        raise ValueError("Empty unseen-grid external test")
    return {"cases": records, "digest": digest(data),
            "families": sorted({row["grid_family_id"] for row in records}),
            "bus_counts": sorted({row["bus_count"] for row in records})}
