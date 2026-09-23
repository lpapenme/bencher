"""Fixtures for IOHBenchmarks' tests, including the goldens plumbing."""
import json
import pathlib

import numpy as np
import pytest

from iohbenchmarks.main import IOHServiceServicer

GOLDENS_PATH = pathlib.Path(__file__).resolve().parent / "goldens.json"
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
    return json.loads(GOLDENS_PATH.read_text()) if GOLDENS_PATH.is_file() else {}


@pytest.fixture(scope="session")
def recorder(request):
    updating = request.config.getoption("--update-goldens")
    recorded = {}
    yield (recorded if updating else None)
    if updating and recorded:
        GOLDENS_PATH.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {len(recorded)} goldens to {GOLDENS_PATH}")


