"""Fixtures for MaxSATBenchmarks' tests, including the goldens plumbing."""
import json
import pathlib

import numpy as np
import pytest

from maxsatbenchmarks.main import BENCHMARKS, MaxSATServiceServicer

GOLDENS_PATH = pathlib.Path(__file__).resolve().parent / "goldens.json"
NAMES = sorted(BENCHMARKS)


def point_for(name):
    """All-ones: a valid binary assignment of the declared dimensionality."""
    return np.ones(BENCHMARKS[name]['dimensions'])


def pytest_addoption(parser):
    parser.addoption("--update-goldens", action="store_true",
                     help="rewrite goldens.json from this run")


@pytest.fixture(scope="module")
def servicer():
    return MaxSATServiceServicer()


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
