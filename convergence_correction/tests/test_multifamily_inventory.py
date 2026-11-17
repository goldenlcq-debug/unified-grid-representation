import pytest

from tools.build_multifamily_inventory import difference, group_splits


def test_mark_difference_rejects_non_mark_and_non_shunt_edits():
    base = b"  1,  5,  5, 5000, 0.0,\r\n  1,  5,  6, 5001, 0.0,\r\n"
    assert difference(base, b"  0,  5,  5, 5000, 0.0,\r\n  1,  5,  6, 5001, 0.0,\r\n") == [
        {"line": 1, "before": 1, "after": 0}
    ]
    with pytest.raises(ValueError, match="outside Mark"):
        difference(base, b"  0,  5,  5, 5000, 0.1,\r\n  1,  5,  6, 5001, 0.0,\r\n")
    with pytest.raises(ValueError, match="not a same-bus"):
        difference(base, b"  1,  5,  5, 5000, 0.0,\r\n  0,  5,  6, 5001, 0.0,\r\n")


def test_parent_splits_are_deterministic_and_disjoint():
    groups = [{"group_id": f"g{i:03d}", "baseline_hash": f"baseline-{i}"} for i in range(1, 11)]
    first = group_splits("grid_3120", groups, (6, 2, 2))
    assert first == group_splits("grid_3120", list(reversed(groups)), (6, 2, 2))
    assert [sum(split == target for split in first.values()) for target in ("train", "validation", "test")] == [6, 2, 2]
    with pytest.raises(ValueError, match="expected 10 parent groups"):
        group_splits("grid_3120", groups[:-1], (6, 2, 2))
    forced = {"g006", "g007", "g008", "g009", "g010"}
    pinned = group_splits("grid_3120", groups, (6, 2, 2), frozenset(forced))
    assert all(pinned[group] == "train" for group in forced)
    assert [sum(split == target for split in pinned.values()) for target in ("train", "validation", "test")] == [6, 2, 2]
