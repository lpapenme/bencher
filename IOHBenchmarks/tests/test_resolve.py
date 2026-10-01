"""Tier 1 tests for IOH problem resolution.

IOH is the one family whose benchmark names cannot be enumerated statically --
they are matched by prefix against the installed `ioh` package. The repo-wide
contract test can therefore only check prefixes; the exact registry-to-problem
check lives here, where `ioh` is installed.
"""
import json
import pathlib

import numpy as np
import pytest

from iohbenchmarks.main import IOHServiceServicer, known_names, resolve_problem

REGISTRY = json.loads(
    (pathlib.Path(__file__).resolve().parents[2] / "BencherServer" /
     "benchmark-registry.json").read_text())
IOH_NAMES = sorted(n for n, props in REGISTRY.items() if props["port"] == 50059)


@pytest.fixture(scope="module")
def servicer():
    return IOHServiceServicer()


def test_registry_lists_some_ioh_benchmarks():
    assert IOH_NAMES, "no benchmarks registered on the IOH port"


@pytest.mark.parametrize("name", IOH_NAMES)
def test_every_registered_name_resolves_to_a_problem(name):
    """The check the static contract test cannot do without `ioh` installed."""
    problem_name, problem_id, problem_class, point_type, rescale = resolve_problem(name)
    assert problem_name and isinstance(problem_id, int)


def test_registered_names_are_a_subset_of_what_the_service_can_serve():
    unknown = sorted(set(IOH_NAMES) - known_names())
    assert not unknown, f"registry advertises IOH problems that do not exist: {unknown}"


class TestNameResolution:
    def test_exact_match_wins_over_a_longer_prefix_match(self):
        """Regression: prefix-only matching made this order-dependent.

        `rosenbrock` is a prefix of both Rosenbrock and RosenbrockRotated, so
        `bbob-rosenbrock` could resolve to either depending on dict ordering.
        """
        assert resolve_problem("bbob-rosenbrock")[0] == "Rosenbrock"
        assert resolve_problem("bbob-rosenbrockrotated")[0] == "RosenbrockRotated"

    def test_unknown_prefix_is_rejected(self):
        with pytest.raises(ValueError, match="expected one of the prefixes"):
            resolve_problem("nosuchfamily-sphere")

    def test_unknown_problem_within_a_known_family_is_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            resolve_problem("bbob-definitelynotaproblem")


class TestEvaluation:
    def test_sphere_is_minimised_towards_the_centre(self, servicer):
        centre = servicer.evaluate("bbob-sphere", np.full(10, 0.5))
        corner = servicer.evaluate("bbob-sphere", np.zeros(10))
        assert centre < corner

    def test_dimension_is_taken_from_the_point(self, servicer):
        """bbob/pbo are registered with dimensions: null for exactly this reason."""
        for dim in (2, 5, 20):
            assert np.isfinite(servicer.evaluate("bbob-sphere", np.full(dim, 0.5)))

    def test_onemax_counts_the_ones(self, servicer):
        assert servicer.evaluate("pbo-onemax", np.ones(16)) == 16.0

    def test_graph_problems_are_not_rescaled(self, servicer):
        """Regression: every graph-* benchmark used to raise TypeError.

        Graph problems report int32 sentinels instead of real bounds, so
        rescaling a [0,1] point against them produced values around -2**31 that
        the C++ binding rejected. They take 0/1 directly, like pbo.
        """
        assert resolve_problem("graph-maxcut2000")[4] is False
        assert np.isfinite(servicer.evaluate("graph-maxcut2000", np.ones(800)))

    def test_every_registered_graph_benchmark_evaluates(self, servicer):
        """All seven were broken; check the whole set, not just one."""
        graph = [n for n in IOH_NAMES if n.startswith("graph-")]
        assert graph
        for name in graph:
            dims = REGISTRY[name]["dimensions"]
            assert np.isfinite(servicer.evaluate(name, np.ones(dims))), name

    def test_bbob_is_still_rescaled_onto_its_own_bounds(self):
        """The continuous family must keep its rescaling."""
        assert resolve_problem("bbob-sphere")[4] is True
