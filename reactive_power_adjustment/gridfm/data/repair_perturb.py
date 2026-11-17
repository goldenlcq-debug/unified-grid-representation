"""A4 受控扰动生成与评测。

扰动只修改独立副本中的 LF.L5.PG、LF.L6.PL/QL；拓扑、型号和原始 donor 永不原位修改。
所有选择由固定 seed 决定，并把每个 before/after 动作写入 provenance。
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
from typing import Any, Dict, Iterable, Optional

import numpy as np

from ..fsutil import robust_copytree
from ..psasp_io.cards import GenView, LoadView
from ..psasp_io.case import PsaspCase
from ..psasp_io.lowlevel import Card
from .repair import DEFAULT_REPAIR_CONFIG, RepairTrace
from .repair_battery import BatteryCase, RepairBatterySpec
from .repair_eval import hash_case_tree
from .synth import psasp_referee, repair_case_convergent


PERTURB_SCHEMA_VERSION = "a_controlled_perturbation_v1"
CONTROLLED_RUN_SCHEMA_VERSION = "a_controlled_repair_run_v1"
PF_OUTPUT_FILES = (
    "LF.CAL", "LF.LP1", "LF.LP2", "LF.LP3", "LF.LP5", "LF.LP6", "LF.P1D")


def _active_loads(card: Card):
    return [
        LoadView(record, row)
        for row, record in enumerate(card.records)
        if record.get_int(0) == 1
    ]


def _active_gens(card: Card):
    return [
        GenView(record, row)
        for row, record in enumerate(card.records)
        if record.get_int(0) == 1
    ]


def _adjacency(case: PsaspCase) -> Dict[int, set[int]]:
    graph = {row: set() for row in range(1, case.n_buses + 1)}
    for branch in [*case.aclines(), *case.xfmrs()]:
        if branch.mark != 1:
            continue
        graph.setdefault(branch.i_no, set()).add(branch.j_no)
        graph.setdefault(branch.j_no, set()).add(branch.i_no)
    return graph


def _distances(graph: Dict[int, set[int]], anchor: int) -> Dict[int, int]:
    distance = {anchor: 0}
    queue = deque([anchor])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(graph.get(current, ())):
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    return distance


def _local_load_buses(
    case: PsaspCase, loads, rng: np.random.Generator, fraction: float = 0.25
) -> tuple[int, list[int]]:
    buses = sorted({load.bs_no for load in loads})
    if not buses:
        raise ValueError("方式没有可扰动的在运负荷")
    anchor = buses[int(rng.integers(0, len(buses)))]
    distance = _distances(_adjacency(case), anchor)
    count = max(1, math.ceil(len(buses) * fraction))
    selected = sorted(
        buses, key=lambda bus: (distance.get(bus, case.n_buses + 1), bus)
    )[:count]
    return anchor, sorted(selected)


def _action(
    action_type: str,
    object_kind: str,
    object_id: int,
    field: str,
    before: float,
    after: float,
    **metadata,
) -> Dict[str, Any]:
    return {
        "action_type": action_type,
        "object_kind": object_kind,
        "object_id": object_id,
        "field": field,
        "before": before,
        "after": after,
        "delta": after - before,
        "metadata": metadata,
    }


def _totals(loads, gens) -> Dict[str, float]:
    return {
        "pl": float(sum(load.pl for load in loads)),
        "ql": float(sum(load.ql for load in loads)),
        "pg_non_slack": float(sum(
            gen.pg for gen in gens if gen.type_code != 0)),
    }


def apply_controlled_perturbation(
    case_dir: Path,
    parameters: Dict[str, Any],
    seed: int,
    *,
    always_protected=(),
) -> Dict[str, Any]:
    """在独立方式副本上确定性施加一个冻结扰动并返回完整动作账。"""
    case_dir = Path(case_dir)
    case = PsaspCase(case_dir)
    l5 = case.cards.get("LF.L5")
    l6 = case.cards.get("LF.L6")
    if l5 is None or l6 is None:
        raise FileNotFoundError("受控扰动要求 LF.L5 与 LF.L6")
    loads = _active_loads(l6)
    gens = _active_gens(l5)
    if not loads:
        raise ValueError("方式没有在运负荷")
    rng = np.random.default_rng(seed)
    kind = parameters.get("kind")
    scope = parameters.get("scope", "global")
    actions = []
    selected_buses: list[int] = []
    anchor = None
    protected = set(int(bus) for bus in always_protected)
    totals_before = _totals(loads, gens)

    def select_load_buses() -> set[int]:
        nonlocal anchor, selected_buses
        if scope == "global":
            selected_buses = sorted({load.bs_no for load in loads})
        elif scope == "local":
            anchor, selected_buses = _local_load_buses(case, loads, rng)
        else:
            raise ValueError(f"未知负荷扰动 scope:{scope}")
        return set(selected_buses)

    def scale_load_fields(pl_factor: float, ql_factor: float) -> None:
        if pl_factor <= 0 or ql_factor <= 0:
            raise ValueError("负荷扰动倍率必须为正")
        selected = select_load_buses()
        for load in loads:
            if load.bs_no not in selected:
                continue
            if pl_factor != 1.0:
                before = load.pl
                load.pl = before * pl_factor
                actions.append(_action(
                    "perturb_load", "load_bus", load.bs_no, "LF.L6.PL",
                    before, load.pl, factor=pl_factor, scope=scope))
            if ql_factor != 1.0:
                before = load.ql
                load.ql = before * ql_factor
                actions.append(_action(
                    "perturb_q", "load_bus", load.bs_no, "LF.L6.QL",
                    before, load.ql, factor=ql_factor, scope=scope))

    if kind == "scale_load":
        factor = float(parameters["factor"])
        scale_load_fields(factor, factor)
    elif kind == "scale_q":
        scale_load_fields(1.0, float(parameters["factor"]))
    elif kind == "redispatch_p":
        candidates = [
            gen for gen in gens if gen.type_code != 0
            and gen.pg > gen.pmin and gen.pg < gen.pmax
        ]
        pairs = []
        for source in candidates:
            for sink in candidates:
                if source.row == sink.row:
                    continue
                transferable = min(
                    max(source.pg - source.pmin, 0.0),
                    max(sink.pmax - sink.pg, 0.0))
                if transferable > 1e-9:
                    pairs.append((source, sink, transferable))
        if not pairs:
            raise ValueError("没有满足上下限的有功再调度机组对")
        source, sink, transferable = pairs[int(rng.integers(0, len(pairs)))]
        amount = float(parameters["fraction"]) * transferable
        if not 0 < float(parameters["fraction"]) <= 1:
            raise ValueError("有功扰动 fraction 必须位于 (0,1]")
        source_before, sink_before = source.pg, sink.pg
        source.pg = source_before - amount
        sink.pg = sink_before + amount
        selected_buses = sorted({source.bs_no, sink.bs_no})
        actions.extend([
            _action(
                "perturb_p", "generator_bus", source.bs_no, "LF.L5.PG",
                source_before, source.pg, role="source"),
            _action(
                "perturb_p", "generator_bus", sink.bs_no, "LF.L5.PG",
                sink_before, sink.pg, role="sink"),
        ])
    elif kind == "mixed":
        scope = parameters.get("scope", "global")
        load_factor = float(parameters.get("load_factor", 1.15))
        q_factor = float(parameters.get("q_factor", 1.25))
        scale_load_fields(load_factor, q_factor)
        if parameters.get("control") == "limited":
            pool = sorted(
                {load.bs_no for load in loads}
                | {gen.bs_no for gen in gens if gen.type_code != 0})
            fraction = float(parameters.get("protected_fraction", 0.75))
            if not 0 <= fraction < 1:
                raise ValueError("protected_fraction 必须位于 [0,1)")
            count = min(max(len(pool) - 1, 0), math.ceil(len(pool) * fraction))
            if count:
                protected.update(
                    int(bus) for bus in rng.choice(
                        np.array(pool), size=count, replace=False))
    else:
        raise ValueError(f"未知受控扰动 kind:{kind}")

    l5.save(case._paths["LF.L5"])
    l6.save(case._paths["LF.L6"])
    totals_after = _totals(_active_loads(l6), _active_gens(l5))
    return {
        "schema_version": PERTURB_SCHEMA_VERSION,
        "kind": kind,
        "scope": scope,
        "seed": seed,
        "anchor_bus": anchor,
        "selected_buses": selected_buses,
        "protected_buses": sorted(protected),
        "totals_before": totals_before,
        "totals_after": totals_after,
        "actions": actions,
        "touched_files": sorted({
            action["field"].split(".", 2)[0] + "." +
            action["field"].split(".", 2)[1] for action in actions}),
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    temp.replace(path)


def _clear_stale_pf_outputs(case_dir: Path) -> None:
    """初判前移除 donor 自带结果卡，防止引擎静默失败时误读旧解。"""
    for name in PF_OUTPUT_FILES:
        (Path(case_dir) / name).unlink(missing_ok=True)


def controlled_code_hash(repo_root: Path) -> str:
    names = (
        "gridfm/data/repair.py",
        "gridfm/data/synth.py",
        "gridfm/data/repair_battery.py",
        "gridfm/data/repair_perturb.py",
        "gridfm/data/repair_eval.py",
        "scripts/run_pf_repair_controlled.py",
    )
    digest = hashlib.sha256()
    for name in names:
        path = Path(repo_root) / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _run_key(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _controlled_cases(spec: RepairBatterySpec) -> list[BatteryCase]:
    return [case for case in spec.cases if case.layer == "controlled"]


def _archive_stale(artifact: Path, out_root: Path, old_key: str) -> None:
    root = out_root / "_stale"
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{artifact.name}-{(old_key or 'unknown')[:12]}"
    target = root / stem
    suffix = 1
    while target.exists():
        target = root / f"{stem}-{suffix}"
        suffix += 1
    shutil.move(str(artifact), str(target))


def run_controlled_battery(
    spec: RepairBatterySpec,
    out_root: Path,
    repo_root: Path,
    solver: Any,
    *,
    resume: bool = True,
    limit: Optional[int] = None,
    case_ids: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """顺序运行受控电池；每例落盘后立即重建汇总。"""
    out_root, repo_root = Path(out_root), Path(repo_root)
    cases = _controlled_cases(spec)
    if case_ids is not None:
        known = {case.case_id for case in cases}
        unknown = case_ids - known
        if unknown:
            raise ValueError(f"未知受控 case_id:{sorted(unknown)}")
        cases = [case for case in cases if case.case_id in case_ids]
    if limit is not None:
        cases = cases[:limit]
    code_hash = controlled_code_hash(repo_root)
    current = _read_current(out_root, spec.content_hash(), code_hash)

    for battery_case in cases:
        short_id = (battery_case.case_id[len("controlled::"):]
                    if battery_case.case_id.startswith("controlled::")
                    else battery_case.case_id)
        short_id = short_id.replace("::", "__")
        artifact = out_root / "artifacts" / short_id
        result_path = artifact / "result.json"
        source = repo_root / battery_case.source
        source_hash = hash_case_tree(source)
        run_key = _run_key({
            "battery_hash": spec.content_hash(),
            "code_hash": code_hash,
            "source_hash": source_hash["tree_sha256"],
            "case": battery_case.to_dict(),
            "config_hash": DEFAULT_REPAIR_CONFIG.content_hash(),
        })
        if result_path.is_file():
            old = json.loads(result_path.read_text(encoding="utf-8"))
            if resume and old.get("run_key") == run_key:
                current[battery_case.case_id] = old
                write_controlled_summary(out_root, current.values())
                continue
            _archive_stale(artifact, out_root, old.get("run_key", ""))

        final_cards = artifact / "final_cards"
        robust_copytree(source, final_cards)
        trace = RepairTrace.for_config(DEFAULT_REPAIR_CONFIG)
        started = time.perf_counter()
        error = None
        initial = {}
        repair: Dict[str, Any] = {}
        perturbation: Dict[str, Any] = {}
        status = "engine_error"
        donor = battery_case.case_id.split("::")[1]
        always_protected = {119} if donor == "ieee118" else set()
        try:
            perturbation = apply_controlled_perturbation(
                final_cards, battery_case.parameters,
                int(battery_case.parameters["seed"]),
                always_protected=always_protected)
            _clear_stale_pf_outputs(final_cards)
            verdict = psasp_referee(final_cards, solver)
            initial = {
                key: verdict.get(key)
                for key in ("converged", "vmin", "vmax")
            }
            repair = repair_case_convergent(
                final_cards, solver,
                protected_buses=perturbation["protected_buses"],
                config=DEFAULT_REPAIR_CONFIG, trace=trace)
            status = "success" if repair.get("nr_ok") else "algorithm_failure"
        except (FileNotFoundError, ValueError) as exc:
            status = "bad_input_data"
            error = {"type": type(exc).__name__, "message": str(exc)}
        except Exception as exc:
            status = "engine_error"
            error = {"type": type(exc).__name__, "message": str(exc)}
            artifact.mkdir(parents=True, exist_ok=True)
            (artifact / "exception.log").write_text(
                traceback.format_exc(), encoding="utf-8")

        donor_after = hash_case_tree(source)
        donor_unchanged = donor_after == source_hash
        if not donor_unchanged:
            status = "engine_error"
            error = {
                "type": "DonorMutationError",
                "message": "受控扰动期间 donor 发生变化",
            }
        result = {
            "schema_version": CONTROLLED_RUN_SCHEMA_VERSION,
            "case_id": battery_case.case_id,
            "donor": donor,
            "profile": battery_case.case_id.split("::")[-1],
            "status": status,
            "run_key": run_key,
            "battery_hash": spec.content_hash(),
            "code_hash": code_hash,
            "config_hash": DEFAULT_REPAIR_CONFIG.content_hash(),
            "source_hash": source_hash["tree_sha256"],
            "output_hash": hash_case_tree(final_cards)["tree_sha256"],
            "donor_unchanged": donor_unchanged,
            "elapsed_seconds": time.perf_counter() - started,
            "initial": initial,
            "error": error,
            "perturbation": perturbation,
            "repair": {key: value for key, value in repair.items() if key != "trace"},
            "trace": repair.get("trace", trace.to_dict()),
        }
        _atomic_json(artifact / "perturbation.json", perturbation)
        _atomic_json(result_path, result)
        current[battery_case.case_id] = result
        write_controlled_summary(out_root, current.values())
    return write_controlled_summary(out_root, current.values())


def _read_current(
    out_root: Path, battery_hash: str, code_hash: str
) -> Dict[str, Dict[str, Any]]:
    results = {}
    for path in sorted((Path(out_root) / "artifacts").glob("*/result.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if (item.get("battery_hash") == battery_hash
                and item.get("code_hash") == code_hash):
            results[item["case_id"]] = item
    return results


def write_controlled_summary(
    out_root: Path, results: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    items = sorted(results, key=lambda item: item["case_id"])
    statuses = ("success", "algorithm_failure", "bad_input_data", "engine_error")
    summary = {
        "schema_version": CONTROLLED_RUN_SCHEMA_VERSION,
        "total": len(items),
        "counts": {
            status: sum(item["status"] == status for item in items)
            for status in statuses
        },
        "initial_converged": sum(
            item.get("initial", {}).get("converged") is True for item in items),
        "initial_diverged": sum(
            item.get("initial", {}).get("converged") is False for item in items),
        "donors": {
            donor: {
                "total": sum(item["donor"] == donor for item in items),
                **{
                    status: sum(
                        item["donor"] == donor and item["status"] == status
                        for item in items)
                    for status in statuses
                },
            }
            for donor in sorted({item["donor"] for item in items})
        },
    }
    _atomic_json(Path(out_root) / "summary.json", summary)
    return summary
