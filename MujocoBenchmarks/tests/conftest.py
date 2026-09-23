"""Fixtures for MujocoBenchmarks' tests.

Every benchmark here is a stochastic rollout, so goldens are meaningful only at a
fixed seed -- and only on the platform they were recorded on. See
test_rollouts.py for the measured drift, and tests/goldens.py for the storage.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
import goldens as goldens_store  # noqa: E402

from mujocobenchmarks.main import BENCHMARKS, MujocoServiceServicer  # noqa: E402

GOLDENS_PATH = pathlib.Path(__file__).resolve().parent / "goldens.json"
# Unlike every other package: a physics rollout amplifies floating-point
# differences, so a value recorded on one architecture does not reproduce on
# another. Goldens are stored per platform and simply absent elsewhere.
GOLDENS_PLATFORM_SPECIFIC = True

NAMES = sorted(BENCHMARKS)

# Fixed so a recorded value is reproducible by inspection.
GOLDEN_SEED = 7

# Rollouts scale with policy size; the largest are slow enough that pinning them
# is not worth the wall clock on every PR.
SLOW = {"mujoco-humanoid", "mujoco-ant"}
FAST_NAMES = [n for n in NAMES if n not in SLOW]


def point_for(name):
    """Mid-domain point: 0.5 in [0, 1] maps to the centre of the native bounds."""
    return np.full(BENCHMARKS[name]['dimensions'], 0.5)


def pytest_addoption(parser):
    parser.addoption("--include-slow", action="store_true",
                     help="also drive mujoco-ant and mujoco-humanoid")
    parser.addoption("--update-goldens", action="store_true",
                     help="rewrite this platform's section of goldens.json")


@pytest.fixture(scope="module")
def servicer():
    return MujocoServiceServicer()


@pytest.fixture(scope="session")
def goldens():
    return goldens_store.load(GOLDENS_PATH)


@pytest.fixture(scope="session")
def golden_for(goldens):
    """This platform's golden for `name`, or None if none was recorded here."""
    def _lookup(name):
        return goldens_store.lookup(goldens, name, GOLDENS_PLATFORM_SPECIFIC)
    return _lookup


@pytest.fixture(scope="session")
def recorder(request):
    updating = request.config.getoption("--update-goldens")
    recorded = {}
    yield (recorded if updating else None)
    if updating and recorded:
        section = goldens_store.record(GOLDENS_PATH, recorded, GOLDENS_PLATFORM_SPECIFIC)
        print(f"\nwrote {len(recorded)} goldens to {GOLDENS_PATH} [{section}]")




def pytest_generate_tests(metafunc):
    if "name" not in metafunc.fixturenames:
        return
    names = NAMES if metafunc.config.getoption("--include-slow") else FAST_NAMES
    metafunc.parametrize("name", names)
