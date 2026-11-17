"""Mandatory unified frozen-encoder integration for RL_POWERADJ."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from .environment import PowerAdjustmentEnv
from .model import DETERMINISTIC_OBSERVATION_METADATA, encode_observation


ENCODER_BLAKE2B16 = "ccf4260f2a8f5db5b6650a33b17d7a0f"
NORMALIZER_BLAKE2B16 = "12281dd483206efe09fdf19475472ebc"
FOUNDATION_FEATURE_SIZE = 16 * 192
FOUNDATION_OBSERVATION_SCHEMA = 3
APPROVED_GRIDFM_SHA256 = "fe305799ba1ca1e8697da4c6750b059f597c8324577405ea8483ed622036e3ac"


def _blake16(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted((root / "gridfm").rglob("*.py"))
    if not files:
        raise FileNotFoundError(f"No gridfm Python sources found under {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def default_foundation_paths(project_root: Path | str) -> tuple[Path, Path]:
    project_root = Path(project_root).resolve()
    assets = Path(os.getenv(
        "RL_POWERADJ_FOUNDATION_ASSETS",
        project_root / "foundation_assets" / "encoder_v1",
    )).resolve()
    source = Path(os.getenv(
        "RL_POWERADJ_GRIDFM_ROOT",
        project_root / "vendor" / "grid-foundation",
    )).resolve()
    return assets, source


def verify_foundation_assets(directory: Path | str) -> dict[str, str]:
    directory = Path(directory).resolve()
    encoder = directory / "v1.pt"
    normalizer = directory / "normalizer.npz"
    if not encoder.is_file() or not normalizer.is_file():
        raise FileNotFoundError(f"Unified frozen-encoder assets are incomplete: {directory}")
    actual_encoder = _blake16(encoder)
    actual_normalizer = _blake16(normalizer)
    if actual_encoder != ENCODER_BLAKE2B16 or actual_normalizer != NORMALIZER_BLAKE2B16:
        raise ValueError(
            "Unified encoder/normalizer fingerprint mismatch: "
            f"encoder={actual_encoder}, normalizer={actual_normalizer}"
        )
    return {
        "encoder_blake2b16": actual_encoder,
        "normalizer_blake2b16": actual_normalizer,
    }


class UnifiedFrozenEncoder:
    """Load the canonical GridEncoder and immutable normalizer with fail-fast identity checks."""

    def __init__(
        self,
        directory: Path | str,
        gridfm_root: Path | str,
        device: str = "cpu",
    ) -> None:
        self.directory = Path(directory).resolve()
        self.gridfm_root = Path(gridfm_root).resolve()
        assets = verify_foundation_assets(self.directory)
        interface = self.gridfm_root / "gridfm" / "reason" / "b_line" / "encoder_io.py"
        if not interface.is_file():
            raise FileNotFoundError(f"Frozen encoder interface is missing: {interface}")
        source_hash = _source_digest(self.gridfm_root)
        if source_hash != APPROVED_GRIDFM_SHA256:
            raise ValueError('Unapproved gridfm source fingerprint')
        loaded = sys.modules.get("gridfm")
        expected_package = self.gridfm_root / "gridfm"
        if loaded is not None and Path(loaded.__file__).resolve().parent != expected_package:
            raise RuntimeError("A different gridfm package is already loaded")
        if str(self.gridfm_root) not in sys.path:
            sys.path.insert(0, str(self.gridfm_root))
        from gridfm.reason.b_line.encoder_io import FrozenBEncoder, assemble_b_sample

        if str(device).startswith('cuda'):
            import torch
            torch.use_deterministic_algorithms(True)
        self.bundle = FrozenBEncoder(self.directory, device=device)
        self.assemble = assemble_b_sample
        self.size = self.bundle.config.n_global * self.bundle.config.d
        if (self.bundle.config.n_global, self.bundle.config.d) != (16, 192):
            raise ValueError(
                f"Unified encoder returned contract size {self.size}, expected {FOUNDATION_FEATURE_SIZE}"
            )
        self.contract = {
            "schema_version": FOUNDATION_OBSERVATION_SCHEMA,
            "kind": "deterministic_observation_plus_unified_frozen_global_tokens",
            "trainable": False,
            "base_observation": dict(DETERMINISTIC_OBSERVATION_METADATA),
            "view": "node_and_branch_STATE_missing_after_normalization",
            "pooling": "global_tokens_row_major",
            "global_tokens": 16,
            "token_width": 192,
            "foundation_size": self.size,
            "gridfm_source_sha256": _source_digest(self.gridfm_root),
            **assets,
        }
        self._normalizer_hash = self.normalizer_hash()
        self.verify_frozen()

    def normalizer_hash(self):
        digest = hashlib.sha256()
        for name in ('node_mean', 'node_std', 'branch_mean', 'branch_std'):
            array = np.asarray(getattr(self.bundle.normalizer, name))
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite frozen normalizer statistics')
            digest.update(name.encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    @classmethod
    def from_project(cls, project_root: Path | str, device: str = "cpu") -> "UnifiedFrozenEncoder":
        assets, source = default_foundation_paths(project_root)
        return cls(assets, source, device=device)

    def encode(self, case_dir: Path | str) -> np.ndarray:
        self.assert_runtime_frozen()
        raw = self.assemble(
            case_dir,
            sample_id="rl-poweradj",
            family_id="rl-poweradj",
        )
        _node_tokens, global_tokens, _edge_tokens = self.bundle.encode(raw)
        result = global_tokens.numpy().reshape(-1).astype(np.float32)
        if result.shape != (self.size,) or not np.isfinite(result).all():
            raise ValueError("Unified frozen encoder returned invalid global tokens")
        return result

    def assert_runtime_frozen(self):
        if self.bundle.encoder.training:
            raise RuntimeError('Unified encoder must remain in eval mode')
        if any(p.requires_grad or p.grad is not None for p in self.bundle.encoder.parameters()):
            raise RuntimeError('Unified encoder has trainable parameters or gradients')

    @property
    def graph_contract(self):
        return {**{k:v for k,v in self.contract.items() if k!='base_observation'},
                'kind':'unified_frozen_node_edge_global_tokens',
                'task_input':'variable_graph_no_flat_observation',
                'node_mapping':'bus_number_minus_one',
                'edge_mapping':'L2_then_L3_forward_block_then_reverse_block_mean',
                'random_identity':'fixed_rl_poweradj_sample_and_family',
                'graph_interface_schema':1}

    def encode_graph(self, case_dir):
        """Expose frozen local tokens without changing upstream weights or view."""
        from .grid_types import FrozenGraph
        self.assert_runtime_frozen()
        raw = self.assemble(case_dir, sample_id='rl-poweradj', family_id='rl-poweradj')
        from collections import OrderedDict
        cache = getattr(self, '_graph_cache', None)
        if cache is None:
            cache = self._graph_cache = OrderedDict()
        content = hashlib.sha256(repr(sorted(self.graph_contract.items())).encode())
        for value in (raw.node_feat, raw.branch_feat, raw.edge_index,
                      raw.node_missing, raw.branch_missing, raw.node_masked, raw.branch_masked,
                      raw.random_features(self.bundle.config.random_dim),
                      raw.branch_random_features(self.bundle.config.branch_random_dim)):
            value = np.ascontiguousarray(value)
            content.update(str((value.shape,str(value.dtype))).encode()); content.update(value.tobytes())
        cache_key = content.hexdigest()
        if cache_key in cache:
            cache.move_to_end(cache_key)
            return cache[cache_key]
        h, g, e = self.bundle.encode(raw)
        m = raw.edge_index.shape[1]
        if tuple(e.shape) != (2*m, 192) or tuple(g.shape) != (1, 16, 192):
            raise ValueError('Unexpected upstream directed-edge/global layout')
        nodes = h.numpy().astype(np.float32)
        edges = ((e[:m]+e[m:])/2).numpy().astype(np.float32)
        glob = g[0].numpy().astype(np.float32)
        index = raw.edge_index.astype(np.int64)
        digest = hashlib.sha256(repr(sorted(self.graph_contract.items())).encode())
        # Hash actual representation AND mapping; no directory/name cache keys.
        for value in (nodes, edges, glob, index):
            digest.update(str(value.shape).encode()); digest.update(value.tobytes())
        result = FrozenGraph(digest.hexdigest(), nodes, edges, glob, index)
        result.validate()
        for value in (nodes, edges, glob, index):
            value.flags.writeable = False
        cache[cache_key] = result
        while len(cache) > 8:
            cache.popitem(last=False)
        return result

    def verify_frozen(self) -> None:
        self.assert_runtime_frozen()
        if self.normalizer_hash() != self._normalizer_hash:
            raise RuntimeError('Frozen normalizer statistics changed')
        verify_foundation_assets(self.directory)
        self.bundle.assert_still_frozen()
        if any(parameter.requires_grad for parameter in self.bundle.encoder.parameters()):
            raise RuntimeError("Unified encoder contains trainable parameters")


class FoundationPowerAdjustmentEnv(PowerAdjustmentEnv):
    """Production environment with mandatory unified frozen global tokens."""

    def __init__(self, *args, foundation_encoder: UnifiedFrozenEncoder, **kwargs):
        self.foundation_encoder = foundation_encoder
        super().__init__(*args, **kwargs)

    @property
    def encoder_contract(self) -> dict[str, object]:
        return self.foundation_encoder.contract

    @property
    def observation_size(self) -> int:
        return self.registry.observation_size + self.foundation_encoder.size

    def _observation(self) -> np.ndarray:
        base = encode_observation(
            self.registry,
            self.l5_rows,
            self.l2_rows,
            self.previous_error,
            self.l6_rows,
            self.l3_rows,
            self.dc_plans,
            previous_error=self.prior_error,
            step_fraction=self.steps / max(self.max_steps, 1),
        )
        foundation = self.foundation_encoder.encode(self.case_dir)
        result = np.concatenate((base, foundation)).astype(np.float32, copy=False)
        if result.shape != (self.observation_size,):
            raise RuntimeError("Unified observation violated the frozen encoder contract")
        return result
