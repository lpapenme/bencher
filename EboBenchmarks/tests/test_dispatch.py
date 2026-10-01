"""Tier 1 for EboBenchmarks: dispatch, dimension validation, and seeding."""

import numpy as np
import pytest
from ebobenchmarks.main import BENCHMARKS, EboServiceServicer

SEED_POINTS = {
    "robotpushing": np.random.default_rng(0).uniform(size=14),
    "rover": np.random.default_rng(0).uniform(size=60),
}
SEED = 7


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


@pytest.mark.parametrize("name,dimensions", [("robotpushing", 14), ("rover", 60)])
def test_benchmark_evaluates(servicer, name, dimensions):
    """Builds the simulator, so it only runs where `ebo` is fully working."""
    assert np.isfinite(servicer.evaluate(name, np.full(dimensions, 0.5)))


@pytest.mark.parametrize("name", sorted(SEED_POINTS))
def test_a_seed_makes_an_evaluation_reproducible(servicer, name):
    x = SEED_POINTS[name]
    assert servicer.evaluate(name, x, seed=SEED) == servicer.evaluate(
        name, x, seed=SEED
    )


@pytest.mark.parametrize("name", sorted(SEED_POINTS))
def test_different_seeds_give_different_evaluations(servicer, name):
    x = SEED_POINTS[name]
    assert servicer.evaluate(name, x, seed=SEED) != servicer.evaluate(
        name, x, seed=SEED + 1
    )


@pytest.mark.parametrize("name", sorted(SEED_POINTS))
def test_without_a_seed_the_evaluation_still_varies(servicer, name):
    x = SEED_POINTS[name]
    assert servicer.evaluate(name, x) != servicer.evaluate(name, x)
