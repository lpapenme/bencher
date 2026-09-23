"""Fixtures for MujocoBenchmarks' tests.

Every benchmark here is a stochastic rollout. See test_rollouts.py for why this
is the one package with no value goldens.
"""
import pathlib

import numpy as np
import pytest

from mujocobenchmarks.main import BENCHMARKS, MujocoServiceServicer

NAMES = sorted(BENCHMARKS)

# Fixed across every golden so a recorded value is reproducible by inspection.
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


@pytest.fixture(scope="module")
def servicer():
    return MujocoServiceServicer()




def pytest_generate_tests(metafunc):
    if "name" not in metafunc.fixturenames:
        return
    names = NAMES if metafunc.config.getoption("--include-slow") else FAST_NAMES
    metafunc.parametrize("name", names)
