from __future__ import annotations

import pathlib
from typing import List

import pytest
from _pytest.config import Config
from _pytest.python import Function

from kelp import consts

MARKERS = ["unit", "integration", "e2e"]


def pytest_collection_modifyitems(config: Config, items: List[Function]) -> None:
    rootdir = pathlib.Path(consts.directories.ROOT_DIR)
    for item in items:
        rel_path = pathlib.Path(item.fspath).relative_to(rootdir)
        mark_name = rel_path.as_posix().split("/")[1]
        if mark_name in MARKERS:
            mark = getattr(pytest.mark, mark_name)
            item.add_marker(mark)
