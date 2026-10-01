"""Tier 1 goldens: every IOH benchmark, without a container.

These call the servicer's evaluate() directly, so they run in the package's own
virtualenv in seconds and gate every PR -- unlike tests/e2e, which needs a built
image. Regenerate after an intended change with:

    uv run --group dev pytest tests/test_goldens.py --update-goldens
"""
import pytest

from conftest import IOH_NAMES, point_for


@pytest.mark.parametrize("name", IOH_NAMES)
def test_value_matches_its_golden(servicer, golden_for, recorder, name):
    value = servicer.evaluate(name, point_for(name))
    if recorder is not None:
        recorder[name] = value
        return
    expected = golden_for(name)
    assert expected is not None, (
        f"no golden recorded for {name} on this platform; "
        f"run with --update-goldens")
    assert value == pytest.approx(expected, rel=1e-9), (
        f"{name} returned {value!r}, golden is {goldens[name]!r}")


@pytest.mark.parametrize("name", IOH_NAMES)
def test_is_deterministic(servicer, name):
    """IOH problems are deterministic; a golden would be meaningless otherwise."""
    assert servicer.evaluate(name, point_for(name)) == servicer.evaluate(name, point_for(name))
