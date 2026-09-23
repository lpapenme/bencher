"""Fixtures for MujocoBenchmarks' tests, including the goldens plumbing.

Every benchmark here is a stochastic rollout, so goldens are only meaningful
when a seed is supplied -- which is what `random_seed` on BenchmarkRequest is
for. The seed is therefore part of the recorded key, not an afterthought.
"""
import json
import pathlib

import numpy as np
import pytest

from mujocobenchmarks.main import BENCHMARKS, MujocoServiceServicer

GOLDENS_PATH = pathlib.Path(__file__).resolve().parent / "goldens.json"
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
    parser.addoption("--update-goldens", action="store_true",
                     help="rewrite goldens.json from this run")
    parser.addoption("--include-slow", action="store_true",
                     help="also drive mujoco-ant and mujoco-humanoid")


@pytest.fixture(scope="module")
def servicer():
    return MujocoServiceServicer()


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


def pytest_generate_tests(metafunc):
    if "name" not in metafunc.fixturenames:
        return
    names = NAMES if metafunc.config.getoption("--include-slow") else FAST_NAMES
    metafunc.parametrize("name", names)
