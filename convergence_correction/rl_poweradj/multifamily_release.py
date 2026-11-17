"""Content-bound release gate for the new four-family training protocol."""
from __future__ import annotations

import json
from pathlib import Path

from .foundation import (APPROVED_GRIDFM_SHA256, _source_digest,
                         default_foundation_paths, verify_foundation_assets)
from .grid_types import GRAPH_SCHEMA
from .multifamily_sampler import TRAIN_FAMILIES
from .readiness import code_manifest, runtime_identity
from .run_store import file_hash

EVIDENCE_KINDS = {"tests", "data_audit", "lifecycle", "parity", "prerun", "resources"}


def release_identity(root, manifest, solver, recipe) -> dict:
    root = Path(root).resolve()
    assets, source = default_foundation_paths(root)
    if _source_digest(source) != APPROVED_GRIDFM_SHA256:
        raise ValueError("Unapproved GridEncoder source")
    return {
        "protocol": "multi_family_joint_training_v1",
        "training_family_ids": list(TRAIN_FAMILIES),
        "code": code_manifest(root),
        "runtime": runtime_identity(),
        "graph_schema": GRAPH_SCHEMA,
        "manifest_sha256": file_hash(manifest),
        "solver_sha256": file_hash(solver),
        "recipe": recipe.to_dict(),
        "assets": verify_foundation_assets(assets),
        "gridfm_source_sha256": APPROVED_GRIDFM_SHA256,
    }


def verify_release(root, path, manifest, solver, recipe) -> dict:
    if path is None:
        raise ValueError("Formal multi-family training requires a new readiness release")
    root = Path(root).resolve()
    path = Path(path).resolve()
    release = json.loads(path.read_text(encoding="utf-8"))
    if release.get("status") != "ready_for_training" or release.get("kind") != "multifamily_joint_release_v1":
        raise ValueError("Wrong multi-family readiness release")
    if release.get("identity") != release_identity(root, manifest, solver, recipe):
        raise ValueError("Stale multi-family readiness identity")
    evidence = release.get("evidence", {})
    if set(evidence) != EVIDENCE_KINDS:
        raise ValueError("Incomplete multi-family readiness evidence")
    for item in evidence.values():
        file = (root / item["path"]).resolve()
        if not file.is_relative_to(root) or file_hash(file) != item["sha256"]:
            raise ValueError("Changed multi-family readiness evidence")
    return {"path": str(path), "sha256": file_hash(path)}
