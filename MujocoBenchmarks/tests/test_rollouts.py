"""Tier 1 for the MuJoCo benchmarks: seeding, and the rescaling around it.

There are deliberately **no value goldens here**, unlike every other package.
A physics rollout amplifies floating-point differences exponentially, so a value
recorded on one machine does not reproduce on another. Measured across
macOS/arm64 and the linux/amd64 container at the same seed:

    lunarlander          87.79251775658189  ->  87.79232914609621   (2e-6)
    mujoco-halfcheetah   -0.1824251507142239 -> -0.18298116106765652 (3e-3)
    mujoco-hopper        -145.43122334242307 -> -146.28735420150608  (6e-3)
    mujoco-swimmer       -5.9325575192963615 -> -5.805574154153944   (2e-2)
    mujoco-walker         4.309704043416025  -> -1.7118133581203339  (sign flip)

No tolerance is both meaningful and passing across that, so goldens here are
stored *per platform* (see tests/goldens.py) rather than shared. The other
packages' goldens do travel -- IOH's 56, Lasso's 6 and MaxSAT's 2 pass on both --
because they are arithmetic, not chaotic dynamics.

On a platform with no recorded section the golden test skips and the relative
assertions below carry the load: a seed makes a rollout reproducible *on one
machine*, and the [0,1] -> native-bounds rescaling is exact. Both hold
everywhere, because neither compares against a stored constant. Between them they
catch what a golden would: the seed failing to reach the rollout, and the domain
mapping being wrong.
"""
import numpy as np
import pytest

from conftest import GOLDEN_SEED, goldens_store, point_for
from mujocobenchmarks.main import BENCHMARKS


def test_a_seed_makes_a_rollout_reproducible(servicer, name):
    """The point of the seeding support: same seed, same value.

    This is the check that catches the seed failing to reach the rollout. It is
    how the merge of the seeding PR with the dispatch refactor was caught: both
    applied cleanly, but `func_factory(x, seed=seed)` had become
    `spec['factory']()(x)`, so every MuJoCo benchmark was silently back to being
    non-reproducible while lunarlander still looked fine.
    """
    x = point_for(name)
    assert servicer.evaluate(name, x, seed=GOLDEN_SEED) == \
           servicer.evaluate(name, x, seed=GOLDEN_SEED)


def test_different_seeds_give_different_rollouts(servicer, name):
    """Guards against a seed that is honoured but collapses to one value.

    Note this does NOT catch a dropped seed -- unseeded rollouts differ anyway,
    so it passes for the wrong reason. That case is caught by the
    reproducibility test above.
    """
    x = point_for(name)
    assert servicer.evaluate(name, x, seed=GOLDEN_SEED) != \
           servicer.evaluate(name, x, seed=GOLDEN_SEED + 1)


def test_without_a_seed_the_rollout_still_varies(servicer, name):
    """Seeding is optional; omitting it must not silently fix the seed."""
    x = point_for(name)
    assert servicer.evaluate(name, x) != servicer.evaluate(name, x)


def test_seeded_value_matches_this_platforms_golden(servicer, golden_for, recorder, name):
    """Pins the actual trajectory -- the one thing the relative tests cannot.

    Skips where no golden has been recorded for this platform, which is the
    designed fallback rather than a gap: the assertions above still run, and
    recording a value from the wrong architecture would be worse than none.
    """
    value = servicer.evaluate(name, point_for(name), seed=GOLDEN_SEED)
    if recorder is not None:
        recorder[name] = value
        return

    expected = golden_for(name)
    if expected is None:
        pytest.skip(
            f"no golden for {name} on {goldens_store.PLATFORM_KEY}; "
            f"record one with --update-goldens, or rely on the relative "
            f"determinism tests")
    assert value == pytest.approx(expected, rel=1e-9), (
        f"{name} at seed {GOLDEN_SEED} returned {value!r}, "
        f"golden for {goldens_store.PLATFORM_KEY} is {expected!r}")


def test_the_value_is_finite(servicer, name):
    value = servicer.evaluate(name, point_for(name), seed=GOLDEN_SEED)
    assert isinstance(value, float) and np.isfinite(value)


# Pinned independently of BENCHMARKS, so that changing a bound in the source
# fails here instead of silently moving the expectation with it. These are the
# native domains each policy is optimised over; getting one wrong changes every
# value the benchmark returns, which is exactly what a value golden would have
# caught if rollouts were reproducible across machines.
EXPECTED_BOUNDS = {
    "mujoco-ant": (-1, 1),
    "mujoco-hopper": (-1.4, 1.4),
    "mujoco-walker": (-1.8, 0.9),
    "mujoco-halfcheetah": (-1, 1),
    "mujoco-swimmer": (-1, 1),
    "mujoco-humanoid": (-1, 1),
    "lunarlander": None,          # a gym environment, no rescaling
}


def test_declared_bounds_match_the_pinned_table(name):
    """A bound may only change deliberately, with this table updated too."""
    assert name in EXPECTED_BOUNDS, f"{name} has no pinned bounds; add it"
    assert BENCHMARKS[name]["bounds"] == EXPECTED_BOUNDS[name]


class _RecordingRollout:
    """Stands in for a MuJoCo rollout and records the point it was handed."""

    def __init__(self):
        self.x = None
        self.seed = None

    def __call__(self, x, seed=None):
        self.x, self.seed = np.array(x, copy=True), seed
        return np.array([[0.0]])


@pytest.mark.parametrize("fill,position", [(0.0, "lower"), (1.0, "upper"), (0.5, "centre")])
def test_the_point_is_rescaled_onto_the_benchmarks_bounds(monkeypatch, servicer, name,
                                                          fill, position):
    """Clients send [0, 1]; each benchmark has its own native bounds.

    This is the deterministic half of what a golden would have protected -- a
    wrong or transposed bound changes every value the benchmark ever returns --
    and unlike a golden it is identical on every architecture.
    """
    if BENCHMARKS[name]["factory"] is None:
        pytest.skip(f"{name} is a gym environment, not a bounded rollout")

    recorder = _RecordingRollout()
    monkeypatch.setitem(BENCHMARKS[name], "factory", lambda: recorder)

    servicer.evaluate(name, np.full(BENCHMARKS[name]["dimensions"], fill), seed=GOLDEN_SEED)

    # From the pinned table, not from BENCHMARKS, so this cannot move with the
    # source it is checking.
    lower, upper = EXPECTED_BOUNDS[name]
    expected = {"lower": lower, "upper": upper, "centre": (lower + upper) / 2}[position]
    assert recorder.x == pytest.approx(np.full_like(recorder.x, expected))
    assert recorder.seed == GOLDEN_SEED, "the seed must reach the rollout"
