"""Tier 1 tests: the benchmark functions and dispatch, without a gRPC server.

These run on any architecture. That is the point of moving the MOPTA
architecture check out of __init__ -- pestcontrol is pure numpy and has no
business being unreachable on an arm64 laptop.
"""

import numpy as np
import pytest
from bencherscaffold.protoclasses.bencher_pb2 import ConstraintType, EvaluationResult

from nodependencybenchmark import lennard_jones, lennard_jones_sr, windfarm
from nodependencybenchmark.main import (
    BENCHMARKS,
    NoDependencyServiceServicer,
    _pest_control_score,
    mopta_executable_basename,
)


@pytest.fixture
def servicer():
    return NoDependencyServiceServicer()


def test_servicer_constructs_on_any_architecture(servicer):
    """Regression: this used to raise RuntimeError on anything but x86/armv7l."""
    assert servicer is not None


def test_registry_shape_is_declared():
    assert set(BENCHMARKS) == {
        "mopta08", "pestcontrol", "gramacy", "lj-7", "lj-13", "lj-38", "lj-55",
        "lj-13-sr", "lj-38-sr", "windfarm-10", "windfarm-30", "windfarm-50",
    }
    assert BENCHMARKS["pestcontrol"]["dimensions"] == 25


class TestPestControl:
    def test_is_deterministic_for_a_fixed_seed(self, servicer):
        x = np.ones(25)
        assert servicer.evaluate("pestcontrol", x, seed=7) == servicer.evaluate(
            "pestcontrol", x, seed=7
        )

    def test_different_seeds_give_different_draws(self, servicer):
        x = np.ones(25)
        # Stochastic by construction; two seeds agreeing exactly would mean the
        # seed is being ignored.
        assert servicer.evaluate("pestcontrol", x, seed=1) != servicer.evaluate(
            "pestcontrol", x, seed=2
        )

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
        assert np.isfinite(
            _pest_control_score(np.zeros(25), seed=1).objectives[0].value
        )


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


class TestGramacy:
    @pytest.mark.parametrize(
        "x, f, g0, g1",
        [
            ((0.5, 0.5), 1.0, -0.5, -1.0),  # feasible
            ((0.0, 0.0), 0.0, 1.5, -1.5),  # violates g0 only
            ((1.0, 1.0), 2.0, -1.5, 0.5),  # violates g1 only
        ],
    )
    def test_known_values(self, servicer, x, f, g0, g1):
        result = servicer.evaluate("gramacy", np.array(x))
        assert result.objectives[0].value == pytest.approx(f)
        assert [c.value for c in result.constraints] == pytest.approx([g0, g1])

    @pytest.mark.parametrize(
        "x, f",
        [
            ((0.1954, 0.4044), 0.5998),  # x^A, the global minimizer
            ((0.7197, 0.1411), 0.8609),  # x^B
            ((0.0, 0.75), 0.75),  # x^C, also bound by x1 >= 0
        ],
    )
    def test_published_local_optima_lie_on_g0(self, servicer, x, f):
        """Gramacy et al. (2016), Fig. 1: g0 is active and g1 strictly satisfied
        at every local minimizer."""
        result = servicer.evaluate("gramacy", np.array(x))
        assert result.objectives[0].value == pytest.approx(f, abs=1e-4)
        g0, g1 = (c.value for c in result.constraints)
        assert abs(g0) < 1e-3 and g1 < 0

    def test_constraints_are_named_inequalities(self, servicer):
        result = servicer.evaluate("gramacy", np.array([0.5, 0.5]))
        assert [c.name for c in result.constraints] == ["g0", "g1"]
        assert all(c.type == ConstraintType.INEQUALITY for c in result.constraints)

    def test_wrong_dimension_is_rejected(self, servicer):
        with pytest.raises(ValueError, match="does not match expected shape"):
            servicer.evaluate("gramacy", np.ones(3))


def _icosahedron(radius: float) -> np.ndarray:
    """Centre atom plus the 12 vertices of an icosahedron of the given radius."""
    phi = (1 + 5**0.5) / 2
    vertices = np.array(
        [
            p
            for a in (1, -1)
            for b in (1, -1)
            for p in ((0, a, b * phi), (a, b * phi, 0), (b * phi, 0, a))
        ],
        dtype=float,
    )
    vertices *= radius / np.linalg.norm(vertices[0])
    return np.vstack([np.zeros(3), vertices])


def _to_unit_cube(positions: np.ndarray) -> np.ndarray:
    """Centre a cluster in its benchmark's box and flatten it to [0, 1]^(3N)."""
    side = lennard_jones.box_side(len(positions))
    return ((positions - positions.mean(0)) / side + 0.5).ravel()


class TestLennardJones:
    def test_declared_dimensions_are_three_per_atom(self):
        for name in lennard_jones.GLOBAL_MINIMA:
            assert BENCHMARKS[name]["dimensions"] == 3 * int(name.split("-")[1])

    def test_dimer_minimum(self):
        """Two atoms at the pair-potential minimum r = 2^(1/6) have energy -1."""
        assert lennard_jones.energy(
            np.array([[0.0, 0.0, 0.0], [2 ** (1 / 6), 0.0, 0.0]])
        ) == pytest.approx(-1.0)

    def test_lj13_global_minimum_is_the_icosahedron(self, servicer):
        """The LJ-13 global minimum is a perfect icosahedron, so a 1-d scan over
        its radius must bottom out at the tabulated energy -- and not below it."""
        radii = np.linspace(1.0, 1.2, 2001)
        energies = [
            servicer.evaluate("lj-13", _to_unit_cube(_icosahedron(r)))
            .objectives[0]
            .value
            for r in radii
        ]
        assert min(energies) == pytest.approx(
            lennard_jones.GLOBAL_MINIMA["lj-13"], abs=1e-5
        )

    def test_is_invariant_to_translation_and_relabelling(self, servicer):
        x = _to_unit_cube(_icosahedron(1.08)).reshape(13, 3)
        reference = servicer.evaluate("lj-13", x.ravel()).objectives[0].value
        shifted = servicer.evaluate("lj-13", (x + 0.1).ravel()).objectives[0].value
        relabelled = servicer.evaluate("lj-13", x[::-1].ravel()).objectives[0].value
        assert shifted == pytest.approx(reference)
        assert relabelled == pytest.approx(reference)

    def test_soft_core_is_finite_and_continuous(self):
        """Coincident atoms must not overflow, and the linear continuation must
        meet the true potential at R_CORE in value and slope."""
        pair = lambda r: lennard_jones.energy(np.array([[0.0, 0, 0], [r, 0, 0]]))
        assert np.isfinite(pair(0.0)) and pair(0.0) < 1e3
        rc, h = lennard_jones.R_CORE, 1e-6
        assert pair(rc - h) == pytest.approx(pair(rc + h), rel=1e-4)
        inner = (pair(rc - h) - pair(rc - 2 * h)) / h
        outer = (pair(rc + 2 * h) - pair(rc + h)) / h
        assert inner == pytest.approx(outer, rel=1e-3)

    def test_soft_core_leaves_the_minimum_untouched(self):
        """No pair in the icosahedron is inside the core at its optimal radius."""
        positions = _icosahedron(1.08)
        distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
        assert distances[np.triu_indices(13, k=1)].min() > lennard_jones.R_CORE

    def test_every_size_is_finite_at_the_centre_and_unconstrained(self, servicer):
        for name in lennard_jones.GLOBAL_MINIMA:
            result = servicer.evaluate(
                name, np.full(BENCHMARKS[name]["dimensions"], 0.5)
            )
            assert np.isfinite(result.objectives[0].value)
            assert len(result.constraints) == 0

    def test_out_of_bounds_and_wrong_shape_are_rejected(self, servicer):
        with pytest.raises(ValueError, match="in \\[0, 1\\]"):
            servicer.evaluate("lj-7", np.full(21, 1.5))
        with pytest.raises(ValueError, match="does not match expected shape"):
            servicer.evaluate("lj-7", np.ones(20))


class TestLennardJonesSymmetryReduced:
    def test_declared_dimensions_drop_six_degrees_of_freedom(self):
        for name in lennard_jones_sr.GLOBAL_MINIMA:
            n = lennard_jones_sr.n_atoms(name)
            assert BENCHMARKS[name]["dimensions"] == 3 * n - 6

    def test_sorting_transform_is_a_bijection_onto_sorted_points(self):
        u = np.random.RandomState(0).rand(40)
        t = lennard_jones_sr.sorted_from_unit(u)
        assert np.all(np.diff(t) >= 0) and 0 <= t[0] and t[-1] <= 1
        assert lennard_jones_sr.unit_from_sorted(t) == pytest.approx(u)

    def test_icosahedron_is_representable_and_has_the_tabulated_energy(self, servicer):
        """Any labelling of the LJ-13 minimum that fits the reduced box must
        reproduce the tabulated energy exactly."""
        radius = 1.0823  # close to the optimal icosahedron radius
        scan = np.linspace(1.0, 1.2, 401)
        energies = [lennard_jones.energy(_icosahedron(r)) for r in scan]
        best = _icosahedron(scan[int(np.argmin(energies))])
        x = lennard_jones_sr.encode("lj-13-sr", best)
        value = servicer.evaluate("lj-13-sr", x).objectives[0].value
        assert value == pytest.approx(min(energies))
        assert value == pytest.approx(lennard_jones.GLOBAL_MINIMA["lj-13"], abs=1e-3)
        assert radius == pytest.approx(scan[int(np.argmin(energies))], abs=5e-3)

    def test_energy_matches_the_unreduced_benchmark(self):
        """The reduced parametrisation only chooses positions; the energy of the
        positions it produces is the plain Lennard-Jones energy."""
        x = np.random.RandomState(1).rand(lennard_jones_sr.dimensions("lj-38-sr"))
        pos = lennard_jones_sr.positions("lj-38-sr", x)
        assert pos[0] == pytest.approx(np.zeros(3))
        assert pos[1, 1:] == pytest.approx(np.zeros(2)) and pos[2, 2] == 0.0
        assert np.all(np.diff((pos[3:] @ lennard_jones_sr.FREE_FRAME.T)[:, 0]) >= 0)
        assert lennard_jones_sr.evaluate("lj-38-sr", x) == pytest.approx(
            lennard_jones.energy(pos)
        )

    def test_wrong_dimension_is_rejected(self, servicer):
        with pytest.raises(ValueError, match="does not match expected shape"):
            servicer.evaluate("lj-13-sr", np.full(39, 0.5))


class TestWindFarm:
    def test_declared_dimensions_are_two_per_turbine(self):
        for name in windfarm.NAMES:
            assert BENCHMARKS[name]["dimensions"] == 2 * int(name.split("-")[1])

    def test_widely_separated_turbines_have_negligible_wake_loss(self):
        """Wakes recover with distance: 200 rotor diameters apart, the loss over
        the whole wind rose is far below anything a layout could gain."""
        pos = np.array([[0.0, 0.0], [200.0, 0.0], [400.0, 0.0]])
        assert 0.0 <= windfarm.wake_loss(pos) < 1e-3
        assert windfarm.wake_loss(np.array([[0.0, 0.0]])) == 0.0

    def test_turbine_directly_downstream_loses_power(self):
        apart = windfarm.wake_loss(np.array([[0.0, 0.0], [0.0, 5.0]]))
        farther = windfarm.wake_loss(np.array([[0.0, 0.0], [0.0, 10.0]]))
        assert 0.0 < farther < apart < 1.0

    def test_objective_is_invariant_to_relabelling_turbines(self, servicer):
        x = np.random.RandomState(0).rand(30, 2)
        a = servicer.evaluate("windfarm-30", x.ravel()).objectives[0].value
        b = servicer.evaluate("windfarm-30", x[::-1].ravel()).objectives[0].value
        assert a == pytest.approx(b)

    def test_spacing_penalty_only_applies_below_two_diameters(self):
        assert windfarm.spacing_penalty(np.array([[0.0, 0.0], [2.0, 0.0]])) == 0.0
        assert windfarm.spacing_penalty(np.array([[0.0, 0.0], [1.0, 0.0]])) == pytest.approx(0.5)

    def test_regular_grid_beats_stacking_all_turbines(self, servicer):
        n = 10
        g = int(np.ceil(np.sqrt(n)))
        grid = np.array([[(i + 0.5) / g, (j + 0.5) / g] for i in range(g) for j in range(g)])[:n]
        good = servicer.evaluate("windfarm-10", grid.ravel()).objectives[0].value
        bad = servicer.evaluate("windfarm-10", np.full(2 * n, 0.5)).objectives[0].value
        assert 0.0 < good < 1.0 < bad

    def test_wrong_dimension_is_rejected(self, servicer):
        with pytest.raises(ValueError, match="does not match expected shape"):
            servicer.evaluate("windfarm-10", np.full(21, 0.5))


# ---------------------------------------------------------------------------
# Reference-value tests for the new benchmarks (Lennard-Jones, symmetry-reduced
# Lennard-Jones, wind farm). Fixtures in tests/data name their sources.
# ---------------------------------------------------------------------------

import json as _json
from pathlib import Path as _Path

_DATA = _Path(__file__).parent / "data"


def _ccd():
    return _json.loads((_DATA / "lj_ccd_minima.json").read_text())


def _rotation(seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.RandomState(seed).randn(3, 3))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


class TestLennardJonesPublishedMinima:
    @pytest.mark.parametrize("name", ["lj-7", "lj-13", "lj-38", "lj-55"])
    def test_ccd_minimum_has_the_published_energy(self, servicer, name):
        """Cambridge Cluster Database coordinates, centred in the benchmark's box,
        evaluate to the published energy (relative tolerance 1e-6)."""
        data = _ccd()
        x = _to_unit_cube(np.array(data["coordinates"][name]))
        assert np.all((x > 0.0) & (x < 1.0)), "published minimum does not fit the box"
        value = servicer.evaluate(name, x).objectives[0].value
        assert value == pytest.approx(data["energies"][name], rel=1e-6)
        assert lennard_jones.GLOBAL_MINIMA[name] == pytest.approx(data["energies"][name], rel=1e-9)


def _sr_roles(cluster: np.ndarray) -> np.ndarray:
    """Relabel a cluster so that atom 1 is the one closest to the centroid, atom 2
    its nearest neighbour and atom 3 the next-nearest neighbour that is not
    collinear with the first two."""
    c = cluster - cluster.mean(0)
    i1 = int(np.argmin(np.linalg.norm(c, axis=1)))
    d = np.linalg.norm(c - c[i1], axis=1)
    order = [int(k) for k in np.argsort(d) if k != i1]
    i2 = order[0]
    e = (c[i2] - c[i1]) / np.linalg.norm(c[i2] - c[i1])
    for i3 in order[1:]:
        v = c[i3] - c[i1]
        if np.linalg.norm(v - (v @ e) * e) > 1e-3:
            break
    rest = [k for k in range(len(c)) if k not in (i1, i2, i3)]
    return c[[i1, i2, i3] + rest]


class TestLennardJonesSymmetryReducedMap:
    @pytest.mark.parametrize("name", ["lj-13-sr", "lj-38-sr"])
    def test_decoded_points_are_valid_configurations(self, name):
        """10^4 random points of the cube decode to configurations with atom 1 at
        the origin, atom 2 on the positive x-axis, atom 3 in the half-plane
        z = 0, y >= 0, atoms 4..N sorted along the first axis of the free frame
        and inside the box in that frame."""
        n = lennard_jones_sr.n_atoms(name)
        half = lennard_jones.box_side(n) / 2
        rng = np.random.RandomState(0)
        for _ in range(10_000):
            p = lennard_jones_sr.positions(name, rng.rand(lennard_jones_sr.dimensions(name)))
            assert np.all(p[0] == 0.0)
            assert p[1, 0] >= 0.0 and p[1, 1] == 0.0 and p[1, 2] == 0.0
            assert p[2, 1] >= 0.0 and p[2, 2] == 0.0
            free = p[3:] @ lennard_jones_sr.FREE_FRAME.T
            assert np.all(np.diff(free[:, 0]) >= 0.0)
            assert np.all(np.abs(free) <= half + 1e-12)
            assert np.all(np.abs(p[:3]) <= half + 1e-12)

    @pytest.mark.parametrize("name", ["lj-13-sr", "lj-38-sr"])
    def test_encode_inverts_decode(self, name):
        rng = np.random.RandomState(1)
        worst = 0.0
        for _ in range(10_000):
            u = rng.rand(lennard_jones_sr.dimensions(name))
            back = lennard_jones_sr.encode(name, lennard_jones_sr.positions(name, u))
            worst = max(worst, float(np.abs(back - u).max()))
        assert worst < 1e-9

    @pytest.mark.parametrize("name,seed", [("lj-13-sr", 7), ("lj-38-sr", 7)])
    def test_published_minimum_is_reachable_and_not_on_a_bound(self, servicer, name, seed):
        """The CCD minimum encodes to a point strictly inside the cube and evaluates
        to the published energy. (The rotation is irrelevant to the encoding, which
        is rotation invariant; it only checks that invariance.)"""
        data = _ccd()
        base = name[: -len("-sr")]
        cluster = _sr_roles(np.array(data["coordinates"][base]) @ _rotation(seed).T)
        u = lennard_jones_sr.encode(name, cluster)
        assert np.all((u > 1e-6) & (u < 1.0 - 1e-6)), f"min distance to a bound {min(u.min(), 1 - u.max())}"
        value = servicer.evaluate(name, u).objectives[0].value
        assert value == pytest.approx(data["energies"][base], rel=1e-6)


def _pywake_reference():
    return _json.loads((_DATA / "windfarm_pywake_reference.json").read_text())


class TestWindFarmAgainstPyWake:
    def test_single_wake_deficit_matches_pywake(self, monkeypatch):
        """Deficit behind one turbine at the rotor centre of a second, against
        PyWake's BastankhahGaussianDeficit with the same C_T, k* and epsilon.
        The wind rose is narrowed to a single westerly sector, so the second
        turbine's power fraction is (1 - deficit)^3 and the farm loss is
        (1 - (1 - deficit)^3) / 2."""
        monkeypatch.setattr(windfarm, "WIND_DIRECTIONS", np.array([270.0]))
        monkeypatch.setattr(windfarm, "WIND_ROSE", np.array([1.0]))
        for case in _pywake_reference()["single_wake"]:
            pos = np.array([[0.0, 0.0], [case["dx"], case["dy"]]])
            loss = windfarm.wake_loss(pos)
            deficit = 1.0 - (1.0 - 2.0 * loss) ** (1.0 / 3.0)
            assert deficit == pytest.approx(case["pywake"], rel=1e-9)

    def test_farm_wake_loss_matches_pywake(self):
        """Whole-farm wake loss over the 12-sector wind rose, root-sum-square
        superposition, for six random feasible layouts of 10 and 30 turbines."""
        for case in _pywake_reference()["farm"]:
            pos = np.array(case["positions"])
            assert windfarm.spacing_penalty(pos) == 0.0
            assert windfarm.wake_loss(pos) == pytest.approx(case["wake_loss"], rel=1e-9)
