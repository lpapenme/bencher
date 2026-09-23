"""Tier 1 for EboBenchmarks: dispatch and dimension validation.

Both benchmarks are stochastic simulators, so no value can be pinned until
random_seed is threaded through. What can be tested cheaply is everything before
the rollout -- and it is testable at all only because the simulators are now
built lazily: `__init__` used to construct PushReward and a 60-D rover domain
eagerly, so the servicer could not be created without the full `ebo` stack
working.
"""
import numpy as np
import pytest

from ebobenchmarks.main import BENCHMARKS, EboServiceServicer


@pytest.fixture
def servicer():
    return EboServiceServicer()


def test_servicer_constructs_without_building_a_simulator(servicer):
    """The lazy-property refactor: construction must stay cheap."""
    assert servicer._push_reward is None
    assert servicer._rover_domain is None


def test_registry_shape_is_declared():
    assert BENCHMARKS == {
        "robotpushing": {"dimensions": 14, "type": "purely_continuous"},
        "rover": {"dimensions": 60, "type": "purely_continuous"},
    }


def test_unknown_benchmark_is_rejected(servicer):
    with pytest.raises(ValueError, match="Invalid benchmark name"):
        servicer.evaluate("not-a-benchmark", np.full(14, 0.5))


@pytest.mark.parametrize("name,dimensions", [("robotpushing", 14), ("rover", 60)])
def test_wrong_dimensionality_is_rejected(servicer, name, dimensions):
    """Caught before the simulator runs, so this needs no rollout."""
    with pytest.raises(AssertionError, match="dimensions"):
        servicer.evaluate(name, np.full(dimensions + 1, 0.5))


@pytest.mark.dataset
@pytest.mark.parametrize("name,dimensions", [("robotpushing", 14), ("rover", 60)])
def test_benchmark_evaluates(servicer, name, dimensions):
    """Builds the simulator, so it only runs where `ebo` is fully working."""
    assert np.isfinite(servicer.evaluate(name, np.full(dimensions, 0.5)))
