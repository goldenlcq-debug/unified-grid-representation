"""pytest 夹具定位(方案 tests/)。优先用仓库内 118 golden fixture(离线、自足)。"""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ieee118_golden"


@pytest.fixture(scope="session")
def case_dir() -> Path:
    assert FIXTURE_DIR.exists(), f"缺少 golden fixture:{FIXTURE_DIR}"
    return FIXTURE_DIR
