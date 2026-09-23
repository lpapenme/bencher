"""Tier 1 for SVMBenchmarks: the coordinate -> hyperparameter mapping.

The last three coordinates are decoded non-linearly into C, gamma and epsilon.
It is easy to transpose or misplace one, and the effect is a silently different
model rather than an error, so the mapping is pinned here. None of this needs
the dataset.
"""
import numpy as np
import pytest

from svmbenchmarks.main import BENCHMARKS, SvmServiceServicer


def test_registry_shape_is_declared():
    assert BENCHMARKS["svm"]["dimensions"] == 388
    assert BENCHMARKS["svmmixed"]["dimensions"] == 53
    assert BENCHMARKS["svmmixed"]["type"] == "mixed"


def test_unknown_benchmark_is_rejected(servicer):
    with pytest.raises(ValueError, match="Invalid benchmark name"):
        servicer.evaluate("svm-nope", np.zeros(10))


@pytest.mark.parametrize("fill,expected", [
    (0.0, (0.01, 0.1, 0.01)),                  # bottom of each range
    (1.0, (5.0, 3.0, 1.0)),                    # top of each range
    (0.5, (0.2236067977499790, 0.5477225575051661, 0.1)),
])
def test_hyperparameters_at_the_domain_edges(fill, expected):
    """C = 0.01*500**x[-1], gamma = 0.1*30**x[-2], epsilon = 0.01*100**x[-3]."""
    got = SvmServiceServicer.hyperparameters(np.full(388, fill))
    assert got == pytest.approx(expected, rel=1e-12)


def test_each_hyperparameter_reads_its_own_coordinate():
    """Guards against the three trailing coordinates being transposed."""
    x = np.full(388, 0.0)
    x[-1] = 1.0                                 # C only
    C, gamma, epsilon = SvmServiceServicer.hyperparameters(x)
    assert (C, gamma, epsilon) == pytest.approx((5.0, 0.1, 0.01), rel=1e-12)


def test_hyperparameters_are_monotonic_in_their_coordinate():
    low = SvmServiceServicer.hyperparameters(np.full(388, 0.25))
    high = SvmServiceServicer.hyperparameters(np.full(388, 0.75))
    assert all(h > l for h, l in zip(high, low))


@pytest.mark.dataset
def test_svm_evaluates_against_the_real_dataset(servicer):
    value = servicer.evaluate("svm", np.full(388, 0.5))
    assert np.isfinite(value) and value > 0


@pytest.mark.dataset
def test_svmmixed_with_no_features_selected_short_circuits(servicer):
    """An all-zero mask selects nothing; the service returns a constant 1.0."""
    x = np.zeros(53)
    x[-3:] = 0.5
    assert servicer.evaluate("svmmixed", x) == 1.0
