"""Predeclared family-balanced checkpoint ranking for joint-grid training."""
from __future__ import annotations

from statistics import mean

from .multifamily_sampler import TRAIN_FAMILIES


def rank_validation(result: dict, *, horizon: int, episode: int) -> tuple[tuple[float, ...], dict]:
    if horizon < 1 or episode < 0:
        raise ValueError("Invalid validation context")
    family_rows = {family: [] for family in TRAIN_FAMILIES}
    for row in result["cases"]:
        if row["family"] not in family_rows:
            raise ValueError(f"Unexpected validation family: {row['family']}")
        family_rows[row["family"]].append(row)
    family_results = {}
    for family, rows in family_rows.items():
        required = [row for row in rows if not row["initial_converged"]]
        if not required:
            raise ValueError(f"{family}: no action-required validation cases")
        successes = [row for row in required if row["converged"]]
        family_results[family] = {
            "action_required": len(required),
            "successes": len(successes),
            "success_rate": len(successes) / len(required),
            "mean_success_steps": mean(row["steps"] for row in successes) if successes else None,
        }
    rates = [value["success_rate"] for value in family_results.values()]
    # No successes receives an explicit horizon+1 step penalty. Earlier
    # checkpoints win exact ties, avoiding a post-hoc choice.
    steps = [value["mean_success_steps"] if value["mean_success_steps"] is not None else horizon + 1
             for value in family_results.values()]
    rank = (mean(rates), min(rates), -mean(steps), -episode)
    return rank, {"families": family_results, "macro_success_rate": mean(rates),
                  "worst_family_success_rate": min(rates), "macro_success_steps": mean(steps)}
