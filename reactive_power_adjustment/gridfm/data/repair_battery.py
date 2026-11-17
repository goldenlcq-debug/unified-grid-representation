"""A4 四层规则潮流修复电池的冻结定义与资产审计。

本模块只定义/审计测试集合，不运行 PSASP、不生成扰动，也不调整规则参数。任何实际运行均须
引用本定义的内容哈希；外部 manifest/provenance 还须通过预期 SHA-256，避免集合静默漂移。
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


BATTERY_SCHEMA_VERSION = "a_rule_repair_battery_v1"

SYNTH_REGRESSION = tuple(
    f"synth_n{size}_seed{seed}"
    for size in (150, 500, 1000)
    for seed in (0, 1, 2)
)

LIFT118_REGRESSION = tuple("""
ieee118_00000 ieee118_00001 ieee118_00002 ieee118_00003 ieee118_00004
ieee118_00006 ieee118_00007 ieee118_00008 ieee118_00010 ieee118_00011
ieee118_00012 ieee118_00013 ieee118_00014 ieee118_00015 ieee118_00017
ieee118_00018 ieee118_00019 ieee118_00022 ieee118_00023 ieee118_00024
ieee118_00025 ieee118_00026 ieee118_00027 ieee118_00028 ieee118_00030
ieee118_00031 ieee118_00032 ieee118_00033 ieee118_00034 ieee118_00036
ieee118_00037 ieee118_00038 ieee118_00040 ieee118_00041 ieee118_00042
ieee118_00043 ieee118_00045 ieee118_00047 ieee118_00048 ieee118_00049
ieee118_00050 ieee118_00051 ieee118_00052 ieee118_00053 ieee118_00055
ieee118_00056 ieee118_00057 ieee118_00058 ieee118_00059 ieee118_00060
ieee118_00062 ieee118_00064 ieee118_00065 ieee118_00066 ieee118_00067
ieee118_00068 ieee118_00069 ieee118_00071 ieee118_00072 ieee118_00073
ieee118_00074 ieee118_00075 ieee118_00076 ieee118_00077 ieee118_00078
ieee118_00080 ieee118_00081 ieee118_00082 ieee118_00083 ieee118_00084
ieee118_00085 ieee118_00086 ieee118_00087 ieee118_00088 ieee118_00090
ieee118_00091 ieee118_00092 ieee118_00093 ieee118_00094 ieee118_00095
ieee118_00096 ieee118_00098 ieee118_00099 ieee118_00101 ieee118_00102
ieee118_00103 ieee118_00104 ieee118_00105 ieee118_00106 ieee118_00110
ieee118_00111 ieee118_00112 ieee118_00116 ieee118_00117 ieee118_00118
ieee118_00119 ieee118_00120 ieee118_00121 ieee118_00122 ieee118_00125
ieee118_00126 ieee118_00127 ieee118_00128 ieee118_00129 ieee118_00130
ieee118_00131 ieee118_00132 ieee118_00133 ieee118_00134 ieee118_00135
ieee118_00136 ieee118_00137 ieee118_00138 ieee118_00139 ieee118_00140
ieee118_00141 ieee118_00142 ieee118_00143 ieee118_00144 ieee118_00145
ieee118_00149
""".split())

LIFT118_PLATEAU = tuple("""
ieee118_00019 ieee118_00027 ieee118_00040 ieee118_00059 ieee118_00062
ieee118_00069 ieee118_00074 ieee118_00078 ieee118_00094 ieee118_00098
ieee118_00102 ieee118_00103 ieee118_00121 ieee118_00144 ieee118_00149
""".split())

UNIT_RULE_CASES = (
    "q_under_voltage_direction",
    "q_over_voltage_direction",
    "q_step_clipping",
    "voltage_deadband",
    "voltage_island_filter",
    "protected_bus",
    "generator_bounds",
    "p_two_hop_search",
    "no_controllable_device",
    "determinism",
    "transaction_rollback",
    "repair_budget_boundary",
    "final_psasp_only",
)

EXPECTED_ASSET_HASHES = {
    "lib_cepri36_manifest": "4b5d84aaf635e26160f7d27ab56935def19b7f8b5642205247b3aaef95a3692c",
    "lib118_manifest": "f24c81f8b0c82f63e18beF68c6147a61e76f8962fa408e481e1e6e8f5f0a665b".lower(),
    "lib118_lift_provenance": "1cc9dbfb8be09e5c670c6c984cd865339747d0c3484def943d5a5d222e2e2b52",
}


@dataclass(frozen=True)
class BatteryCase:
    case_id: str
    layer: str
    source: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepairBatterySpec:
    version: str
    cases: Tuple[BatteryCase, ...]
    external_assets: Dict[str, Dict[str, Any]]

    def __post_init__(self) -> None:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("电池 case_id 不得重复")
        allowed = {"unit", "historical", "controlled", "real_library"}
        actual = {case.layer for case in self.cases}
        if actual != allowed:
            raise ValueError(f"电池层级必须完整为 {sorted(allowed)}，实际 {sorted(actual)}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": BATTERY_SCHEMA_VERSION,
            "version": self.version,
            "cases": [case.to_dict() for case in self.cases],
            "external_assets": self.external_assets,
        }

    def content_hash(self) -> str:
        raw = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def default_battery_spec() -> RepairBatterySpec:
    cases: List[BatteryCase] = []
    cases.extend(BatteryCase(
        f"unit::{name}", "unit", "pytest",
        expected={"test_id": name}) for name in UNIT_RULE_CASES)
    for size in (150, 500, 1000):
        for seed in (0, 1, 2):
            cases.append(BatteryCase(
                f"historical::synth_n{size}_seed{seed}", "historical",
                "tests/fixtures/ieee118_golden",
                parameters={
                    "n_buses": size, "seed": seed, "load_level": 0.55,
                    "gen_frac": 0.30, "q_comp": 0.25,
                    "experimental_allow_new_comp_rows": True,
                    "dc_backend": "dense_legacy",
                },
                expected={
                    "psasp_converged": True,
                    "not_for_transient_pipeline": True,
                }))
    for sample_id in LIFT118_REGRESSION:
        cases.append(BatteryCase(
            f"historical::lift::{sample_id}", "historical",
            f"lib118_pre_lift/cases/{sample_id}",
            parameters={"max_rounds": 12, "target": 0.94, "protected_buses": [119]},
            expected={
                "accepted_vmin": 0.93,
                "target_vmin": 0.94,
                "plateau": sample_id in LIFT118_PLATEAU,
            }))

    # A4.3 只在此冻结“要生成什么”；生成器在下一提交实现。每个条目均需跨三种 donor/seed。
    controlled_profiles = (
        ("load_global_mild", {"kind": "scale_load", "scope": "global", "factor": 1.05}),
        ("load_global_medium", {"kind": "scale_load", "scope": "global", "factor": 1.15}),
        ("load_global_severe", {"kind": "scale_load", "scope": "global", "factor": 1.30}),
        ("load_local_medium", {"kind": "scale_load", "scope": "local", "factor": 1.15}),
        ("q_global_medium", {"kind": "scale_q", "scope": "global", "factor": 1.25}),
        ("q_local_medium", {"kind": "scale_q", "scope": "local", "factor": 1.25}),
        ("p_imbalance", {"kind": "redispatch_p", "scope": "zonal", "fraction": 0.20}),
        ("mixed", {"kind": "mixed", "load_factor": 1.15, "q_factor": 1.25}),
        ("limited_control", {
            "kind": "mixed", "scope": "global",
            "load_factor": 1.15, "q_factor": 1.25,
            "control": "limited", "protected_fraction": 0.75,
        }),
    )
    for donor in ("cepri36", "ieee118", "ieee39"):
        for seed in (0, 1, 2):
            for name, params in controlled_profiles:
                cases.append(BatteryCase(
                    f"controlled::{donor}::s{seed}::{name}", "controlled",
                    f"tests/fixtures/{donor}_golden",
                    parameters={"seed": seed, **params},
                    expected={"retain_failures": True}))

    for library in ("lib_cepri36", "lib118"):
        cases.append(BatteryCase(
            f"real::{library}::solvable", "real_library",
            f"{library}/manifest.csv",
            parameters={"selector": "solvable=1"},
            expected={"report_family_separately": True}))
        cases.append(BatteryCase(
            f"real::{library}::native_failure", "real_library",
            f"{library}/manifest.csv",
            parameters={"selector": "solvable=0"},
            expected={"retain_failures": True, "report_family_separately": True}))
    cases.append(BatteryCase(
        "real::lowv_quarantine", "real_library",
        "lib118_quarantine_lowv/manifest.csv",
        parameters={"selector": "all"},
        expected={"retain_failures": True, "report_family_separately": True}))

    assets = {
        "lib_cepri36_manifest": {
            "path": "lib_cepri36/manifest.csv",
            "sha256": EXPECTED_ASSET_HASHES["lib_cepri36_manifest"],
        },
        "lib118_manifest": {
            "path": "lib118/manifest.csv",
            "sha256": EXPECTED_ASSET_HASHES["lib118_manifest"],
        },
        "lib118_lift_provenance": {
            "path": "lib118/lift_provenance.json",
            "sha256": EXPECTED_ASSET_HASHES["lib118_lift_provenance"],
        },
    }
    return RepairBatterySpec("a_rule_repair_battery_v1", tuple(cases), assets)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_counts(path: Path) -> Dict[str, int]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "total": len(rows),
        "solvable": sum(row.get("solvable", "1").strip() != "0" for row in rows),
        "native_failure": sum(row.get("solvable", "1").strip() == "0" for row in rows),
    }


def audit_battery_assets(spec: RepairBatterySpec, data_root: Path) -> Dict[str, Any]:
    """核验外部资产并明确指出历史 pre-lift 输入是否在位。"""
    data_root = Path(data_root)
    assets = {}
    for name, expected in spec.external_assets.items():
        path = data_root / expected["path"]
        actual = _sha256(path) if path.is_file() else None
        item = {
            "path": str(path),
            "exists": path.is_file(),
            "expected_sha256": expected["sha256"],
            "actual_sha256": actual,
            "hash_match": actual == expected["sha256"],
        }
        if path.is_file() and path.name == "manifest.csv":
            item["counts"] = _manifest_counts(path)
        assets[name] = item

    provenance_path = data_root / "lib118" / "lift_provenance.json"
    provenance_ids: List[str] = []
    plateau_ids: List[str] = []
    if assets["lib118_lift_provenance"]["hash_match"]:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance_ids = [item["sample_id"] for item in provenance.get("modes", [])]
        plateau_ids = [
            item["sample_id"] for item in provenance.get("modes", [])
            if item.get("vmin_after") is not None and item["vmin_after"] < 0.94
        ]
    pre_lift_root = data_root / "lib118_pre_lift" / "cases"
    pre_lift_available = all(
        (pre_lift_root / sample_id).is_dir() for sample_id in LIFT118_REGRESSION)
    return {
        "schema_version": BATTERY_SCHEMA_VERSION,
        "battery_hash": spec.content_hash(),
        "assets": assets,
        "lift118": {
            "expected_count": len(LIFT118_REGRESSION),
            "provenance_count": len(provenance_ids),
            "ids_match": tuple(provenance_ids) == LIFT118_REGRESSION,
            "plateau_ids_match": tuple(plateau_ids) == LIFT118_PLATEAU,
            "pre_lift_inputs_available": pre_lift_available,
            "pre_lift_root": str(pre_lift_root),
            "reproduction_blocked": not pre_lift_available,
        },
        "lowv_quarantine_available": (
            data_root / "lib118_quarantine_lowv" / "manifest.csv").is_file(),
        "ready_for_historical_reproduction": (
            all(item["hash_match"] for item in assets.values())
            and tuple(provenance_ids) == LIFT118_REGRESSION
            and tuple(plateau_ids) == LIFT118_PLATEAU
            and pre_lift_available
        ),
    }
