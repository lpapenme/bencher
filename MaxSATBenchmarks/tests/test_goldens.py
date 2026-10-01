"""Tier 1 goldens for the MaxSAT instances -- no container needed.

Regenerate after an intended change with:

    uv run --group dev pytest tests/test_goldens.py --update-goldens
"""
import numpy as np
import pytest

from conftest import NAMES, point_for
from maxsatbenchmarks.main import BENCHMARKS


@pytest.mark.parametrize("name", NAMES)
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


@pytest.mark.parametrize("name", NAMES)
def test_is_deterministic(servicer, name):
    assert servicer.evaluate(name, point_for(name)) == servicer.evaluate(name, point_for(name))


@pytest.mark.parametrize("name", NAMES)
def test_rejects_non_binary_input(servicer, name):
    """The only explicit domain check in the repo; keep it enforced."""
    with pytest.raises(AssertionError, match="binary"):
        servicer.evaluate(name, np.full(BENCHMARKS[name]['dimensions'], 0.5))
