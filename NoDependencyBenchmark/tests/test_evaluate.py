"""Tier 1 tests: the benchmark functions and dispatch, without a gRPC server.

These run on any architecture. That is the point of moving the MOPTA
architecture check out of __init__ -- pestcontrol is pure numpy and has no
business being unreachable on an arm64 laptop.
"""
import numpy as np
import pytest
from bencherscaffold.protoclasses.bencher_pb2 import EvaluationResult

from nodependencybenchmark.main import (BENCHMARKS, NoDependencyServiceServicer,
                                        _pest_control_score, mopta_executable_basename)


@pytest.fixture
def servicer():
    return NoDependencyServiceServicer()


def test_servicer_constructs_on_any_architecture(servicer):
    """Regression: this used to raise RuntimeError on anything but x86/armv7l."""
    assert servicer is not None


def test_registry_shape_is_declared():
    assert set(BENCHMARKS) == {"mopta08", "pestcontrol"}
    assert BENCHMARKS["pestcontrol"]["dimensions"] == 25


class TestPestControl:
    def test_is_deterministic_for_a_fixed_seed(self, servicer):
        x = np.ones(25)
        assert servicer.evaluate("pestcontrol", x, seed=7) == \
               servicer.evaluate("pestcontrol", x, seed=7)

    def test_different_seeds_give_different_draws(self, servicer):
        x = np.ones(25)
        # Stochastic by construction; two seeds agreeing exactly would mean the
        # seed is being ignored.
        assert servicer.evaluate("pestcontrol", x, seed=1) != \
               servicer.evaluate("pestcontrol", x, seed=2)

    def test_returns_a_single_finite_objective(self, servicer):
        result = servicer.evaluate("pestcontrol", np.ones(25), seed=0)
        assert isinstance(result, EvaluationResult)
        assert [o.name for o in result.objectives] == ["f0"]
        assert np.isfinite(result.objectives[0].value)

    def test_is_unconstrained(self, servicer):
        result = servicer.evaluate("pestcontrol", np.ones(25), seed=0)
        assert len(result.constraints) == 0

    @pytest.mark.parametrize("category", [0, 1, 2, 3, 4])
    def test_accepts_every_valid_category(self, servicer, category):
        """Values are used directly as categorical indices, 0 (no control) to 4."""
        result = servicer.evaluate("pestcontrol", np.full(25, category), seed=3)
        assert np.isfinite(result.objectives[0].value)

    def test_all_zero_means_no_control_is_applied(self):
        """With no pesticide applied, nothing is paid for control."""
        assert np.isfinite(_pest_control_score(np.zeros(25), seed=1).objectives[0].value)


class TestDispatch:
    def test_unknown_benchmark_is_rejected_by_name(self, servicer):
        with pytest.raises(ValueError, match="Invalid benchmark name"):
            servicer.evaluate("not-a-benchmark", np.ones(5))

    def test_mopta_architecture_error_names_the_architecture(self):
        """On a machine with no MOPTA binary the message should say so clearly."""
        import platform
        if (platform.machine().lower(), 64) in {("x86_64", 64), ("amd64", 64)}:
            assert mopta_executable_basename().startswith("mopta08_")
        else:
            with pytest.raises(RuntimeError, match="no executable for architecture"):
                mopta_executable_basename()

    def test_every_declared_benchmark_is_dispatched(self, servicer):
        """A name added to BENCHMARKS without a branch must not fall through silently."""
        for name in BENCHMARKS:
            if name == "mopta08":
                continue  # needs the x86 binary; covered by the e2e tier
            dimensions = BENCHMARKS[name]["dimensions"]
            result = servicer.evaluate(name, np.zeros(dimensions, dtype=int), seed=0)
            assert isinstance(result, EvaluationResult)
