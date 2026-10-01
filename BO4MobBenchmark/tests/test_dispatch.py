"""Tier 1 for BO4MobBenchmark: name matching and OD-template dimensions.

Names are templated -- 420 of them over 5 networks x 14 dates x 3 hours x 2
metrics -- so this service matches by regex rather than enumerating. Both the
regex and the per-network dimensionality are checkable without running a SUMO
simulation, which is what everything past that point needs.
"""
import numpy as np
import pytest

from bo4mobbenchmark.main import BO4MOBServiceServicer, NETWORKS


@pytest.fixture
def servicer():
    return BO4MOBServiceServicer()


def test_networks_are_declared():
    assert set(NETWORKS) == {"1ramp", "2corridor", "3junction", "4smallRegion", "5fullRegion"}
    assert NETWORKS["1ramp"]["dimensions"] == 3


@pytest.mark.parametrize("name", [
    "1ramp_221008_06-07_count",
    "2corridor_221021_17-18_speed",
    "5fullRegion_221014_08-09_count",
])
def test_valid_names_are_accepted_past_validation(servicer, name):
    """A well-formed name must fail on dimensionality, not on the name."""
    with pytest.raises(AssertionError, match="does not match"):
        servicer.evaluate(name, [1.0])          # deliberately wrong length


@pytest.mark.parametrize("name", [
    "1ramp_221008_06-07_volume",     # unknown metric
    "6nowhere_221008_06-07_count",   # unknown network
    "1ramp_2210_06-07_count",        # malformed date
    "1ramp_221008_05-06_count",      # hour outside the three windows
    "1ramp",                         # bare network name
])
def test_malformed_names_are_rejected(servicer, name):
    with pytest.raises(ValueError, match="Invalid benchmark name"):
        servicer.evaluate(name, [1.0, 2.0, 3.0])


def test_dimensionality_must_match_the_od_template(servicer):
    """The OD template's row count is the benchmark's dimensionality."""
    with pytest.raises(AssertionError, match="does not match"):
        servicer.evaluate("1ramp_221008_06-07_count", np.ones(4))
