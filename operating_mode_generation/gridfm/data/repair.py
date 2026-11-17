"""A 线规则潮流修复的统一配置与可序列化轨迹协议。

本模块只定义协议，不调用 PSASP、不读写卡片。所有规则入口必须引用同一
RuleRepairConfig，禁止在脚本中复制阈值。RepairTrace 是 A2 起逐轮观察、动作和终止原因的
统一载体，也是后续监督学习与强化学习的数据接口。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TRACE_SCHEMA_VERSION = "a_repair_trace_v1"


def _jsonable(value: Any) -> Any:
    """把 numpy 标量/数组、Path 等递归转换为 JSON 安全对象。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


@dataclass(frozen=True)
class RuleRepairConfig:
    """rule_repair_v1 的唯一规则参数源。"""

    version: str = "rule_repair_v1"
    action_order: Tuple[str, ...] = (
        "adjust_q", "redispatch_p", "adjust_q_recheck", "scale_load", "final_psasp")

    voltage_target: float = 0.94
    voltage_low_trigger: float = 0.95
    voltage_high_trigger: float = 1.05
    voltage_max_soft: float = 1.06
    voltage_island: float = 0.20
    q_gain: float = 2.0
    q_step_cap: float = 0.30
    q_max_rounds: int = 8
    q_recheck_rounds: int = 4

    p_residual_threshold: float = 0.30
    p_adjust_fraction: float = 0.70
    p_search_hops: int = 2
    p_max_rounds: int = 4

    load_scale: float = 0.85
    max_load_repairs: int = 3
    lift_max_rounds: int = 10
    protected_buses: Tuple[int, ...] = ()
    allow_new_comp_rows: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.voltage_target <= self.voltage_low_trigger:
            raise ValueError("voltage_target 必须位于 (0, voltage_low_trigger]")
        if not self.voltage_low_trigger < self.voltage_high_trigger:
            raise ValueError("电压低触发阈值必须小于高触发阈值")
        if self.voltage_max_soft < self.voltage_high_trigger:
            raise ValueError("voltage_max_soft 不得低于高压触发阈值")
        if self.q_gain <= 0 or self.q_step_cap <= 0:
            raise ValueError("无功比例增益和单轮上限必须为正")
        if self.p_residual_threshold <= 0 or not 0 < self.p_adjust_fraction <= 1:
            raise ValueError("有功残差阈值须为正，调整比例须位于 (0,1]")
        if self.p_search_hops < 1:
            raise ValueError("有功邻域搜索至少为 1 跳")
        if not 0 < self.load_scale < 1:
            raise ValueError("降载倍率必须位于 (0,1)")
        for name in ("q_max_rounds", "q_recheck_rounds", "p_max_rounds",
                     "max_load_repairs", "lift_max_rounds"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不得为负")
        if len(set(self.protected_buses)) != len(self.protected_buses):
            raise ValueError("protected_buses 不得重复")
        if not isinstance(self.allow_new_comp_rows, bool):
            raise ValueError("allow_new_comp_rows 必须为布尔值")

    def with_protected(self, buses) -> "RuleRepairConfig":
        merged = tuple(sorted(set(self.protected_buses) | {int(b) for b in buses}))
        return replace(self, protected_buses=merged)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    def content_hash(self) -> str:
        raw = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


DEFAULT_REPAIR_CONFIG = RuleRepairConfig()


@dataclass
class RepairObservation:
    stage: str
    round_index: int
    source: str
    converged: bool
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    voltage: Optional[Any] = None
    diagnosis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class RepairAction:
    action_type: str
    stage: str
    round_index: int
    object_kind: str
    object_id: int
    field: str
    before: Optional[float]
    after: Optional[float]
    delta: Optional[float]
    trigger: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    status: str = "applied"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass
class RepairTrace:
    config_hash: str
    schema_version: str = TRACE_SCHEMA_VERSION
    observations: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    termination_reason: str = ""
    final_converged: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_config(cls, config: RuleRepairConfig) -> "RepairTrace":
        return cls(
            config_hash=config.content_hash(),
            metadata={"config_version": config.version,
                      "action_order": list(config.action_order)})

    def add_observation(self, observation: RepairObservation) -> None:
        self.observations.append(observation.to_dict())

    def add_action(self, action: RepairAction) -> None:
        self.actions.append(action.to_dict())

    def finish(self, reason: str, converged: Optional[bool]) -> None:
        self.termination_reason = reason
        self.final_converged = converged

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent)

    def replay_touched_values(self, atol: float = 1e-9) -> Dict[str, float]:
        """按动作顺序重放所有已应用的数值字段，并校验 before→after 链连续。

        返回的键为 ``object_kind:object_id:field``。跳过/回滚动作不参与重放；若同一字段后一
        动作的 before 与前一动作 after 不一致，说明轨迹不完整或顺序损坏，立即报错。
        """
        state: Dict[str, float] = {}
        for action in self.actions:
            if action.get("status") != "applied":
                continue
            before = action.get("before")
            after = action.get("after")
            if before is None or after is None:
                raise ValueError("applied 动作必须包含 before/after")
            key = (
                f"{action['object_kind']}:{action['object_id']}:"
                f"{action['field']}")
            if key in state and not math.isclose(
                    state[key], float(before), rel_tol=0.0, abs_tol=atol):
                raise ValueError(f"动作链不连续:{key}")
            state[key] = float(after)
        return state
