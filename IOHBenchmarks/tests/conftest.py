"""Fixtures for IOHBenchmarks' tests, including the goldens plumbing."""
import json
import pathlib
import sys

import numpy as np
import pytest

from iohbenchmarks.main import IOHServiceServicer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
import goldens as goldens_store  # noqa: E402

GOLDENS_PATH = pathlib.Path(__file__).resolve().parent / "goldens.json"
# These benchmarks are arithmetic, not chaotic: one recording serves every
# platform. See tests/goldens.py.
GOLDENS_PLATFORM_SPECIFIC = False
REGISTRY_PATH = (pathlib.Path(__file__).resolve().parents[2]
                 / "BencherServer" / "benchmark-registry.json")

REGISTRY = json.loads(REGISTRY_PATH.read_text())
# bbob/pbo are dimension-agnostic (registry records null); these match the
# dimensions tests/benchmarks.py drives them at, so Tier 1 and e2e agree.
DEFAULT_DIMENSIONS = {"bbob-": 10, "pbo-": 16}
IOH_NAMES = sorted(n for n, p in REGISTRY.items() if p["port"] == 50059)


def point_for(name):
    entry = REGISTRY[name]
    dimensions = entry["dimensions"]
    if dimensions is None:
        dimensions = next(d for prefix, d in DEFAULT_DIMENSIONS.items()
                          if name.startswith(prefix))
    fill = 0.5 if entry["type"] == "purely_continuous" else 1
    return np.full(dimensions, fill)


def pytest_addoption(parser):
    parser.addoption("--update-goldens", action="store_true",
                     help="rewrite goldens.json from this run")


@pytest.fixture(scope="module")
def servicer():
    return IOHServiceServicer()


@pytest.fixture(scope="session")
def goldens():
    return goldens_store.load(GOLDENS_PATH)


@pytest.fixture(scope="session")
def golden_for(goldens):
    """Looks a golden up for this platform; None when none is recorded."""
    def _lookup(name):
        return goldens_store.lookup(goldens, name, GOLDENS_PLATFORM_SPECIFIC)
    return _lookup


@pytest.fixture(scope="session")
def recorder(request):
    """Collects values during --update-goldens and writes them out once.

    Writing from a session finalizer rather than per test keeps a partial run
    from losing values, and the store merges rather than overwrites.
    """
    updating = request.config.getoption("--update-goldens")
    recorded = {}
    yield (recorded if updating else None)
    if updating and recorded:
        section = goldens_store.record(GOLDENS_PATH, recorded, GOLDENS_PLATFORM_SPECIFIC)
        print(f"\nwrote {len(recorded)} goldens to {GOLDENS_PATH} [{section}]")


