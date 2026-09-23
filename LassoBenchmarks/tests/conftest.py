"""Fixtures for LassoBenchmarks' tests."""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from lassobenchmarks.main import BENCHMARKS, LassoServiceServicer

GOLDENS_PATH = pathlib.Path(__file__).resolve().parent / "goldens.json"

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
    return json.loads(GOLDENS_PATH.read_text()) if GOLDENS_PATH.is_file() else {}


@pytest.fixture(scope="session")
def recorder(request):
    updating = request.config.getoption("--update-goldens")
    recorded = {}
    yield (recorded if updating else None)
    if updating and recorded:
        merged = {}
        if GOLDENS_PATH.is_file():
            merged.update(json.loads(GOLDENS_PATH.read_text()))
        merged.update(recorded)
        GOLDENS_PATH.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {len(recorded)} goldens to {GOLDENS_PATH}")
