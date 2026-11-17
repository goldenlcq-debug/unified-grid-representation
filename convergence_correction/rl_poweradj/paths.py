from __future__ import annotations

import os
import json
from hashlib import sha256
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Sequence


CASE_MARKERS = ("LF.L0", "LF.L1", "LF.L2", "LF.L5", "LF.L6")


def discover_cases(root: Path | str, required: Sequence[str] = CASE_MARKERS) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    cases = {
        marker.parent
        for marker in root.rglob(required[0])
        if all((marker.parent / name).is_file() for name in required)
    }
    result = sorted(cases, key=lambda path: str(path).casefold())
    if not result:
        raise FileNotFoundError(f"No valid PSASP cases found under {root}")
    return result


def read_case_catalog(
    catalog: Path | str,
    dataset_root: Path | str,
    required: Sequence[str] = CASE_MARKERS,
) -> list[Path]:
    """Read and validate a one-case-per-line dataset catalog.

    Catalog entries may be absolute or relative to ``dataset_root``, but every
    resolved case must be an immediate child of that dataset. This prevents a
    stale catalog from silently labelling a different dataset with matching
    numeric directory names.
    """
    catalog = Path(catalog).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not catalog.is_file():
        raise FileNotFoundError(f"Case catalog does not exist: {catalog}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    result: list[Path] = []
    seen: set[Path] = set()
    for line_number, raw in enumerate(catalog.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = raw.strip()
        if not text:
            continue
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = dataset_root / candidate
        candidate = candidate.resolve()
        if candidate.parent != dataset_root:
            raise ValueError(
                f"Catalog entry is outside dataset_root: {catalog}:{line_number}: {candidate}"
            )
        if candidate in seen:
            raise ValueError(f"Duplicate case in catalog: {catalog}:{line_number}: {candidate}")
        if not candidate.is_dir() or not all((candidate / name).is_file() for name in required):
            raise FileNotFoundError(f"Invalid PSASP case in catalog: {candidate}")
        seen.add(candidate)
        result.append(candidate)
    if not result:
        raise ValueError(f"Case catalog is empty: {catalog}")
    return result


def case_input_fingerprint(case: Path | str) -> bytes:
    """Hash electrical inputs so duplicate operating points stay in one split."""
    case = Path(case)
    digest = sha256()
    for name in ("LF.L0", "LF.L1", "LF.L2", "LF.L3", "LF.L5", "LF.L6", "LF.NL4"):
        path = case / name
        if path.is_file():
            digest.update(name.encode("ascii"))
            digest.update(path.read_bytes())
    return digest.digest()


def deterministic_partition(
    cases: Sequence[Path], fractions: Sequence[float] = (0.7, 0.15, 0.15)
) -> tuple[list[Path], ...]:
    """Create stable grouped partitions without duplicate-input leakage."""
    if len(fractions) < 2 or any(value <= 0.0 for value in fractions):
        raise ValueError("At least two positive partition fractions are required")
    if not isclose(sum(fractions), 1.0):
        raise ValueError("Partition fractions must sum to one")
    if len(cases) < len(fractions):
        raise ValueError("There are fewer cases than requested partitions")
    groups: dict[bytes, list[Path]] = {}
    for case in cases:
        groups.setdefault(case_input_fingerprint(case), []).append(case)
    targets = [int(round(len(cases) * value)) for value in fractions[:-1]]
    targets.append(len(cases) - sum(targets))
    partitions: list[list[Path]] = [[] for _ in fractions]
    for fingerprint in sorted(groups):
        group = sorted(groups[fingerprint], key=lambda path: path.name.casefold())
        destination = next(
            (
                index for index, target in enumerate(targets)
                if len(partitions[index]) + len(group) <= target
            ),
            len(partitions) - 1,
        )
        partitions[destination].extend(group)
    if any(not partition for partition in partitions):
        raise ValueError("Grouped split produced an empty partition")
    return tuple(partitions)


def deterministic_split(cases: Sequence[Path], train_fraction: float = 0.8) -> tuple[list[Path], list[Path]]:
    """Backward-compatible two-way grouped split."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    first, second = deterministic_partition(cases, (train_fraction, 1.0 - train_fraction))
    return first, second


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    train_root: Path
    test_root: Path
    validation_root: Path | None
    solver: Path
    model_dir: Path
    result_dir: Path
    dataset_root: Path | None
    converged_catalog: Path | None
    nonconverged_catalog: Path | None
    catalog_split: bool
    control_policy: Path
    flow_manifest: Path
    split_catalogs: dict[str, Path] | None = None
    parent_catalogs: dict[str, Path] | None = None
    converged_root: Path | None = None

    @classmethod
    def from_root(cls, root: Path | str | None = None) -> "ProjectPaths":
        project_root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        dataset_root = project_root / "600nodesample" / "dataset2"
        converged_catalog = project_root / "600nodesample" / "sampleset-600-ok.txt"
        nonconverged_catalog = project_root / "600nodesample" / "sampleset-600-bad.txt"
        train_override = os.getenv("RL_POWERADJ_TRAIN_DATA")
        test_override = os.getenv("RL_POWERADJ_TEST_DATA")
        validation_override = os.getenv("RL_POWERADJ_VALIDATION_DATA")
        lfsample_root = project_root / "lfsamples"
        lfsample_bad = lfsample_root / "lfsample-bad"
        lfsample_ok = lfsample_root / "lfsample-ok"
        lfsample_split_catalogs = {
            split: lfsample_root / f"{split}-cases.txt"
            for split in ("train", "validation", "test")
        }
        lfsample_parent_catalogs = {
            split: lfsample_root / f"{split}-parents.txt"
            for split in ("train", "validation", "test")
        }
        use_lfsamples = (
            train_override is None
            and test_override is None
            and validation_override is None
            and lfsample_bad.is_dir()
            and lfsample_ok.is_dir()
            and all(path.is_file() for path in lfsample_split_catalogs.values())
            and all(path.is_file() for path in lfsample_parent_catalogs.values())
            and (lfsample_root / "converged-cases.txt").is_file()
            and (lfsample_root / "formal-flow-manifest.json").is_file()
        )
        if use_lfsamples:
            dataset_root = lfsample_bad
            converged_catalog = lfsample_root / "converged-cases.txt"
            nonconverged_catalog = None
        catalog_split = use_lfsamples or (
            train_override is None
            and test_override is None
            and validation_override is None
            and dataset_root.is_dir()
            and converged_catalog.is_file()
            and nonconverged_catalog is not None
            and nonconverged_catalog.is_file()
        )
        train_root = Path(train_override or (dataset_root if catalog_split else project_root / "db_200")).resolve()
        test_root = Path(test_override or (dataset_root if catalog_split else project_root / "db_200ceshi")).resolve()
        validation_root = (
            dataset_root if use_lfsamples
            else None if validation_override is None
            else Path(validation_override).resolve()
        )
        solver = Path(os.getenv(
            "RL_POWERADJ_SOLVER",
            project_root / "PSASP" / "PSASP" / "WMLFRTMsg.exe",
        )).resolve()
        return cls(
            root=project_root,
            train_root=train_root,
            test_root=test_root,
            validation_root=validation_root,
            solver=solver,
            model_dir=project_root / "artifacts" / "models",
            result_dir=project_root / "artifacts" / "results",
            dataset_root=dataset_root if dataset_root.is_dir() else None,
            converged_catalog=converged_catalog if converged_catalog.is_file() else None,
            nonconverged_catalog=(
                nonconverged_catalog
                if nonconverged_catalog is not None and nonconverged_catalog.is_file()
                else None
            ),
            catalog_split=catalog_split,
            control_policy=Path(os.getenv(
                "RL_POWERADJ_CONTROL_POLICY",
                project_root / "config" / (
                    "reactive-control-policy-lfsamples.json"
                    if use_lfsamples else "reactive-control-policy.json"
                ),
            )).resolve(),
            flow_manifest=Path(os.getenv(
                "RL_POWERADJ_FLOW_MANIFEST",
                (
                    lfsample_root / "formal-flow-manifest.json"
                    if use_lfsamples
                    else project_root / "600nodesample" / "formal-flow-manifest.json"
                ),
            )).resolve(),
            split_catalogs=lfsample_split_catalogs if use_lfsamples else None,
            parent_catalogs=lfsample_parent_catalogs if use_lfsamples else None,
            converged_root=lfsample_ok if use_lfsamples else None,
        )

    def cases(self, split: str) -> list[Path]:
        if split == "all":
            if self.dataset_root is None:
                raise FileNotFoundError("The dataset is unavailable")
            if self.split_catalogs is not None:
                bad = [case for name in ("train", "validation", "test") for case in self.cases(name)]
                return bad + self.cases("converged")
            return discover_cases(self.dataset_root)
        if split == "converged":
            if self.dataset_root is None or self.converged_catalog is None:
                raise FileNotFoundError("The converged-case catalog is unavailable")
            return read_case_catalog(
                self.converged_catalog, self.converged_root or self.dataset_root
            )
        if self.split_catalogs is not None and split in self.split_catalogs:
            if self.dataset_root is None:
                raise FileNotFoundError("The bad-case dataset is unavailable")
            return read_case_catalog(self.split_catalogs[split], self.dataset_root)
        if self.catalog_split and split in ("train", "validation", "test"):
            if self.dataset_root is None or self.nonconverged_catalog is None:
                raise FileNotFoundError("The non-converged-case catalog is unavailable")
            train, validation, test = deterministic_partition(
                read_case_catalog(self.nonconverged_catalog, self.dataset_root)
            )
            return {"train": train, "validation": validation, "test": test}[split]
        if split == "train":
            return discover_cases(self.train_root)
        if split == "test":
            return discover_cases(self.test_root)
        if split == "validation":
            if self.validation_root is None:
                raise FileNotFoundError(
                    "A validation dataset is required; set RL_POWERADJ_VALIDATION_DATA"
                )
            return discover_cases(self.validation_root)
        raise ValueError("split must be 'train', 'validation', 'test', 'converged', or 'all'")

    def registry_cases(self, split: str = "train") -> list[Path]:
        if self.parent_catalogs is not None and self.converged_root is not None:
            # build_lfsamples_dataset.py verifies all 188 parents have the
            # same generator/load/branch/transformer identity contract.
            return read_case_catalog(self.parent_catalogs[split], self.converged_root)[:1]
        return self.cases(split)

    def ensure_outputs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def reactive_cases(self, split: str) -> list[Path]:
        """Return active-valid split members whose unchanged first AC solve fails."""
        if not self.flow_manifest.is_file():
            raise FileNotFoundError(
                f"Formal-flow manifest is required for DQN training: {self.flow_manifest}"
            )
        payload = json.loads(self.flow_manifest.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 2:
            raise ValueError(
                "The flow manifest predates the RL-only input contract; rerun `manifest`"
            )
        records = {
            str(item["case"]): item for item in payload.get("records", ())
        }
        split_cases = self.cases(split)
        missing = [case.name for case in split_cases if case.name not in records]
        invalid = [
            case.name for case in split_cases
            if records.get(case.name, {}).get("active_power_valid") is not True
        ]
        if missing:
            raise ValueError(f"Flow manifest is missing split cases: {', '.join(missing[:10])}")
        if invalid:
            raise ValueError(
                "RL-only datasets must already satisfy the active-power contract; "
                f"invalid cases: {', '.join(invalid[:10])}"
            )
        allowed = {
            str(item["case"])
            for item in records.values()
            if item.get("category") == "requires_reactive_recovery"
        }
        cases = [case for case in split_cases if case.name in allowed]
        if not cases:
            raise ValueError(f"No reactive-recovery cases remain in split {split}")
        return cases
