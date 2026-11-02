"""`@pytest.mark.dataset`: run where the data is on disk, skip where it is not.

The marker used to be a bare label that only CI's `-m "not dataset"` acted on,
so a plain local run tried to download the data and failed. Now each package
conftest names the files its dataset tests read, and calls
`skip_unless_present` from `pytest_collection_modifyitems`. If any file is
missing, the marked tests are skipped and the skip reason lists the missing
paths.

Set BENCHER_REQUIRE_DATASETS=1 to turn that skip into a collection error. The
container legs set it: the image is supposed to have the data baked in, so a
skip there would hide a broken bake rather than report it.

Loaded by package conftests via sys.path, like grpc_harness.py. Runs under
every package interpreter, 3.8 included.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pytest

REQUIRE_ENV_VAR = "BENCHER_REQUIRE_DATASETS"


def absent(patterns: Iterable[Path]) -> list[str]:
    """The patterns with no match on disk. A pattern may glob its last part."""
    return [str(p) for p in patterns if not any(p.parent.glob(p.name))]


def skip_unless_present(items: list[pytest.Item], missing: list[str]) -> None:
    """Skips every `dataset`-marked item when `missing` is non-empty."""
    marked = [item for item in items if item.get_closest_marker("dataset")]
    if not marked or not missing:
        return
    reason = "dataset not on disk: " + ", ".join(missing)
    if os.environ.get(REQUIRE_ENV_VAR) == "1":
        raise pytest.UsageError(f"{REQUIRE_ENV_VAR}=1 but {reason}")
    for item in marked:
        item.add_marker(pytest.mark.skip(reason=reason))
