import pytest

from rl_poweradj.multifamily_sampler import MultiFamilySampler, TRAIN_FAMILIES


def records():
    return [
        {"case_id": f"{family}-{parent}-{child}", "grid_family_id": family,
         "parent_case_id": f"{family}-{parent}", "split": "train"}
        for family in TRAIN_FAMILIES for parent in range(2) for child in range(3)
    ] + [{"case_id": "validation-only", "grid_family_id": "grid_118",
          "parent_case_id": "heldout", "split": "validation"}]


def test_sampler_covers_families_and_restores_exact_sequence():
    rows = records()
    sampler = MultiFamilySampler(rows, seed=3)
    first = [sampler.next() for _ in range(100)]
    assert set(rows[i]["grid_family_id"] for i in first) == set(TRAIN_FAMILIES)
    assert all(rows[i]["split"] == "train" for i in first)
    state = sampler.state_dict()
    continued = [sampler.next() for _ in range(50)]
    restored = MultiFamilySampler(rows, seed=999)
    restored.load_state_dict(state)
    assert [restored.next() for _ in range(50)] == continued
    with pytest.raises(ValueError, match="identity changed"):
        MultiFamilySampler(rows[:-2], seed=3).load_state_dict(state)


def test_sampler_rejects_rejected_family():
    rows = records()
    rows[0]["grid_family_id"] = "grid_2869"
    with pytest.raises(ValueError, match="Unapproved training family"):
        MultiFamilySampler(rows)
