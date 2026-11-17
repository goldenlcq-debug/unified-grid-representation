"""Action statistics for completed graph evaluations (not solver-call counts)."""
from collections import Counter, defaultdict
from numbers import Integral
from statistics import mean, median


STEP_METRIC_DEFINITIONS = {
    "unit": "shunt actions, not PSASP calls or solver iterations",
    "success_population": "completed, initially nonconverged, finally converged cases",
    "all_required_population": "completed, initially nonconverged cases, including policy failures",
    "excluded": "initially converged, skipped and explicitly interrupted/error cases",
    "empty_population": "mean/median/min/max are null; counts and totals are zero",
    "mean_required_steps": "legacy alias of mean_all_required_steps, NOT mean_success_steps",
}


def summarize_action_steps(rows):
    """Summarize recorded completed cases; never impute steps for missing cases.

    Current evaluators do not append a case on solver exceptions. Offline callers
    may also supply explicitly marked skipped/interrupted/error records, which
    are excluded rather than turned into policy failures. A failed result file
    can still contain a valid prefix of completed cases.
    """
    required = []
    skipped = interrupted = initial = 0
    for row in rows:
        disposition = row.get("disposition")
        reason = row.get("reason")
        if disposition == "skipped" or reason == "skipped":
            skipped += 1
            continue
        if (disposition in {"interrupted", "error", "exception"}
                or reason in {"interrupted", "solver_error", "license_error", "input_error"}):
            interrupted += 1
            continue
        if type(row.get("initial_converged")) is not bool or type(row.get("converged")) is not bool:
            raise ValueError("Step statistics require completed case convergence flags")
        steps = row.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 0:
            raise ValueError("Case steps must be a nonnegative integer action count")
        if reason == "running":
            raise ValueError("Cannot summarize an unfinished case as a completed evaluation")
        if row["initial_converged"]:
            initial += 1
        else:
            required.append(row)
    successes = [r["steps"] for r in required if r["converged"]]
    failures = [r for r in required if not r["converged"]]
    failure_groups = defaultdict(list)
    for row in failures:
        failure_groups[row.get("reason", "unspecified")].append(row["steps"])
    return {
        "action_required": len(required),
        "successes": len(successes),
        "failures": len(failures),
        "initial_converged_cases": initial,
        "skipped_cases": skipped,
        "interrupted_cases": interrupted,
        "mean_success_steps": float(mean(successes)) if successes else None,
        "median_success_steps": float(median(successes)) if successes else None,
        "min_success_steps": int(min(successes)) if successes else None,
        "max_success_steps": int(max(successes)) if successes else None,
        "mean_all_required_steps": float(mean(r["steps"] for r in required)) if required else None,
        "total_success_steps": int(sum(successes)),
        "total_all_required_steps": int(sum(r["steps"] for r in required)),
        "mean_failed_steps": float(mean(r["steps"] for r in failures)) if failures else None,
        "total_failed_steps": int(sum(r["steps"] for r in failures)),
        "failure_reasons": dict(Counter(r.get("reason", "unspecified") for r in failures)),
        "failure_steps_by_reason": {
            reason: {"cases": len(values), "total_steps": int(sum(values)),
                     "mean_steps": float(mean(values))}
            for reason, values in sorted(failure_groups.items())
        },
    }
