"""数据集加载(方案 7.5)。从方式库目录(.npz + manifest.csv)加载 GridSample,拟合/施加归一化。"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

from ..schema.sample import GridSample
from .normalize import Normalizer


def library_files(lib_dir: Path) -> List[Path]:
    """库目录内**将被加载**的 .npz 路径(manifest 驱动选择,与 load_library 严格同源)。

    冻结评估集哈希必须用本函数(8.9①):对目录裸 glob 会把 manifest 之外的文件算进哈希,
    哈希证的就不再是被评估的那批字节(2026-07-09 对抗审查发现)。
    """
    lib_dir = Path(lib_dir)
    manifest = lib_dir / "manifest.csv"
    if manifest.exists():
        rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
        ids = [r["sample_id"] for r in rows]
    else:
        ids = [p.stem for p in sorted(lib_dir.glob("*.npz"))]
    return [lib_dir / f"{sid}.npz" for sid in ids if (lib_dir / f"{sid}.npz").exists()]


def load_library(lib_dir: Path, split: Optional[str] = None) -> List[GridSample]:
    """加载库目录下所有 .npz(manifest 驱动)。split 预留。"""
    return [GridSample.load_npz(p) for p in library_files(lib_dir)]


def family_files(root: Path, split: Optional[str] = None) -> List[Path]:
    """多家族树内将被加载的 .npz 路径(与 load_families 严格同源,含 families/ 与扁平两种布局)。"""
    root = Path(root)
    fam_root = root / "families" if (root / "families").exists() else root
    splits = [split] if split else sorted(d.name for d in fam_root.iterdir() if d.is_dir())
    out: List[Path] = []
    for sp in splits:
        sp_dir = fam_root / sp
        if not sp_dir.exists():
            continue
        for fam_dir in sorted(d for d in sp_dir.iterdir() if d.is_dir()):
            out.extend(library_files(fam_dir))
    return out


def load_families(root: Path, split: Optional[str] = None) -> List[GridSample]:
    """加载 gen_families.py 产出的多家族树:root/families/{split}/{family_id}/*.npz。

    split=None → 全部;否则只加载该 split(train/val/transfer)。每个家族目录含自己的 manifest.csv。
    """
    return [GridSample.load_npz(p) for p in family_files(root, split)]


def fit_and_transform(samples: List[GridSample]) -> Tuple[List[GridSample], Normalizer]:
    """在样本集上拟合归一化并原地施加(训练前调用)。"""
    norm = Normalizer().fit(samples)
    for s in samples:
        norm.transform(s)
    return samples, norm
