"""Tier 1 tests: dispatch, rescaling and constraint conversion, without a gRPC server.

botorch reports constraint *slack* (feasible iff >= 0); bencher reports
constraint *values* (feasible iff <= 0). The sign flip between the two breaks
silently, so it is pinned here against hand-computed values rather than only
against botorch itself.
"""

import math

import numpy as np
import pytest
import torch
from bencherscaffold.protoclasses.bencher_pb2 import ConstraintType, EvaluationResult
from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.test_functions.synthetic import PressureVessel
from conftest import NAMES, point_for

from botorchbenchmarks.main import BENCHMARKS


class TestDispatch:
    """Generic over BENCHMARKS, so a new botorch problem is covered by adding it there."""

    def test_unknown_benchmark_is_rejected_by_name(self, servicer):
        with pytest.raises(ValueError, match="Invalid benchmark name"):
            servicer.evaluate("not-a-benchmark", np.ones(4))

    @pytest.mark.parametrize("name", NAMES)
    def test_wrong_dimension_is_rejected(self, servicer, name):
        with pytest.raises(ValueError, match="does not match expected shape"):
            servicer.evaluate(name, np.full(BENCHMARKS[name]["dimensions"] + 1, 0.5))

    @pytest.mark.parametrize("name", NAMES)
    def test_declared_dimensions_match_the_botorch_problem(self, name):
        """BENCHMARKS is what the registry is checked against; it must agree
        with what the problem actually takes."""
        assert BENCHMARKS[name]["factory"]().dim == BENCHMARKS[name]["dimensions"]

    @pytest.mark.parametrize("name", NAMES)
    def test_factory_does_not_negate(self, name):
        """evaluate() calls evaluate_true, which skips `negate` -- botorch applies
        it only in forward(). A factory passing negate=True would have the flag
        silently dropped, so it is rejected here instead."""
        assert not BENCHMARKS[name]["factory"]().negate

    @pytest.mark.parametrize("name", NAMES)
    def test_returns_a_single_finite_objective(self, servicer, name):
        result = servicer.evaluate(name, point_for(name))
        assert isinstance(result, EvaluationResult)
        assert [o.name for o in result.objectives] == ["f0"]
        assert np.isfinite(result.objectives[0].value)

    @pytest.mark.parametrize("name", NAMES)
    def test_constraints_are_reported_iff_the_problem_has_them(self, servicer, name):
        problem = BENCHMARKS[name]["factory"]()
        expected = (
            problem.num_constraints
            if isinstance(problem, ConstrainedBaseTestProblem)
            else 0
        )
        result = servicer.evaluate(name, point_for(name))
        assert [c.name for c in result.constraints] == [
            f"g{i}" for i in range(expected)
        ]
        assert all(c.type == ConstraintType.INEQUALITY for c in result.constraints)

    @pytest.mark.parametrize("name", NAMES)
    def test_is_deterministic_and_ignores_the_seed(self, servicer, name):
        x = point_for(name)
        assert servicer.evaluate(name, x) == servicer.evaluate(name, x, seed=7)


_PV_LB = np.array([0.0, 0.0, 10.0, 150.0])
_PV_UB = np.array([10.0, 10.0, 50.0, 200.0])


def _pv_unit(x_native):
    """A point in PressureVessel's native bounds, as the [0, 1]^4 a client sends."""
    return (np.asarray(x_native, dtype=float) - _PV_LB) / (_PV_UB - _PV_LB)


def _botorch_native(x_native):
    return torch.as_tensor([x_native], dtype=torch.float64)


class TestPressureVessel:
    """Coello Coello (2002): min cost over (x1, x2, x3, x4) subject to

    g0 = -x1 + 0.0193 x3                              <= 0
    g1 = -x2 + 0.00954 x3                             <= 0
    g2 = -pi x3^2 x4 - 4/3 pi x3^3 + 1296000          <= 0
    g3 = x4 - 240                                     <= 0
    """

    NAME = "pressure-vessel"

    @pytest.mark.parametrize(
        "u, f, g",
        [
            # x = (0, 0, 10, 150): no material, so no cost -- and the thickness
            # and volume constraints are all violated.
            (
                0.0,
                0.0,
                [
                    0.193,
                    0.0954,
                    1296000 - math.pi * 100 * 150 - 4 / 3 * math.pi * 1000,
                    -90.0,
                ],
            ),
            # x = (10, 10, 50, 200): everything feasible, at maximum cost.
            (
                1.0,
                0.6224 * 10 * 50 * 200
                + 1.7781 * 10 * 50**2
                + 3.1661 * 10**2 * 200
                + 19.84 * 10**2 * 50,
                [
                    -10 + 0.0193 * 50,
                    -10 + 0.00954 * 50,
                    1296000 - math.pi * 50**2 * 200 - 4 / 3 * math.pi * 50**3,
                    -40.0,
                ],
            ),
        ],
    )
    def test_known_values_at_the_corners(self, servicer, u, f, g):
        result = servicer.evaluate(self.NAME, np.full(4, u))
        assert result.objectives[0].value == pytest.approx(f)
        assert [c.value for c in result.constraints] == pytest.approx(g)

    def test_published_optimum(self, servicer):
        """At x* = (0.8125, 0.4375, 42.098446, 176.636596), g0 and g2 are
        active and g1, g3 strictly satisfied."""
        result = servicer.evaluate(
            self.NAME, _pv_unit([0.8125, 0.4375, 42.098446, 176.636596])
        )
        assert result.objectives[0].value == pytest.approx(6059.7143, abs=1e-3)
        g0, g1, g2, g3 = (c.value for c in result.constraints)
        # The published digits put x* on g0 and within ~0.03 of g2 (whose
        # scale is ~1e6), so "active" is up to that rounding.
        assert abs(g0) < 1e-6
        assert abs(g2) < 0.1
        assert g1 < 0 and g3 < 0

    def test_constraints_see_the_rounded_thicknesses(self, servicer):
        """Regression: botorch rounds x1, x2 to multiples of 0.0625 in the
        objective only. With unrounded constraints x1 = 0.8437 counted as 0.8437
        for feasibility while the objective priced 0.8125, which admitted
        'feasible' designs cheaper than the published optimum (f ~ 5872)."""
        x3 = 43.7151
        result = servicer.evaluate(self.NAME, _pv_unit([0.8437, 0.4680, x3, 157.6]))
        g0, g1 = result.constraints[0].value, result.constraints[1].value
        assert g0 == pytest.approx(-0.8125 + 0.0193 * x3)
        assert g1 == pytest.approx(-0.4375 + 0.00954 * x3)
        assert g0 > 0, "the design actually priced violates g0"

    def test_objective_is_botorchs(self, servicer):
        """The subclass changes only the constraints."""
        x = [3.3, 6.7, 18.0, 195.0]
        expected = PressureVessel().evaluate_true(_botorch_native(x)).item()
        result = servicer.evaluate(self.NAME, _pv_unit(x))
        assert result.objectives[0].value == pytest.approx(expected, rel=1e-12)
