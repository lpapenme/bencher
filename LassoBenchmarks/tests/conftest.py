"""Fixtures for LassoBenchmarks' tests."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from lassobenchmarks.main import BENCHMARKS, LassoServiceServicer

import goldens as goldens_store  # noqa: E402

GOLDENS_PATH = pathlib.Path(__file__).resolve().parent / "goldens.json"
# These benchmarks are arithmetic, not chaotic: one recording serves every
# platform. See tests/goldens.py.
GOLDENS_PLATFORM_SPECIFIC = False

# Synthetic problems build in-process; the "real" ones fetch a libsvm dataset on
# first use. The big three are excluded entirely -- lasso-rcv1 is 47236
# dimensions and downloads hundreds of MB.
SYNTHETIC = ["lasso-simple", "lasso-medium", "lasso-high", "lasso-hard"]
SMALL_REAL = ["lasso-diabetes", "lasso-breastcancer"]
LARGE_REAL = ["lasso-dna", "lasso-leukemia", "lasso-rcv1"]


def point_for(name):
    """Mid-domain: 0.5 in [0, 1] maps to 0.0 once rescaled to [-1, 1]."""
    return np.full(BENCHMARKS[name]['dimensions'], 0.5)


def pytest_addoption(parser):
    parser.addoption("--update-goldens", action="store_true",
                     help="rewrite goldens.json from this run")


@pytest.fixture(scope="module")
def servicer():
    return LassoServiceServicer()


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
