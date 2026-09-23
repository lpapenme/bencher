"""Tier 1 for LassoBenchmarks: dispatch, rescaling and goldens.

Lasso rescales a [0, 1] point onto [-1, 1] (`x = 2x - 1`), which is the kind of
transform that breaks silently, so it is pinned here rather than only end to end.

Regenerate with:  uv run --group dev pytest tests --update-goldens
"""
import numpy as np
import pytest
from conftest import SMALL_REAL, SYNTHETIC, point_for

from lassobenchmarks.main import BENCHMARKS

CHEAP = SYNTHETIC + SMALL_REAL


def test_registry_shape_is_declared():
    assert len(BENCHMARKS) == 9
    assert BENCHMARKS["lasso-diabetes"]["dimensions"] == 8


def test_unknown_benchmark_is_rejected(servicer):
    with pytest.raises(ValueError, match="Invalid benchmark name"):
        servicer.evaluate("lasso-nope", np.zeros(8))


@pytest.mark.parametrize("name", CHEAP)
def test_value_matches_its_golden(servicer, golden_for, recorder, name):
    value = servicer.evaluate(name, point_for(name))
    if recorder is not None:
        recorder[name] = value
        return
    expected = golden_for(name)
    assert expected is not None, (
        f"no golden recorded for {name} on this platform; "
        f"run with --update-goldens")
    assert value == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("name", CHEAP)
def test_is_deterministic(servicer, name):
    assert servicer.evaluate(name, point_for(name)) == \
           servicer.evaluate(name, point_for(name))


@pytest.mark.dataset
@pytest.mark.parametrize("name", ["lasso-dna"])
def test_large_real_benchmarks_evaluate(servicer, name):
    """Needs a libsvm dataset, so it only runs where one is baked in."""
    assert np.isfinite(servicer.evaluate(name, point_for(name)))
