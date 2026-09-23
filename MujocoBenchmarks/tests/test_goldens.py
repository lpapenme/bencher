"""Tier 1 goldens for the MuJoCo benchmarks, using the seeding support.

These rollouts are stochastic, so before seeding existed no value could be
pinned. With a seed they are reproducible, which is what makes these goldens --
and the determinism checks below -- possible.

Regenerate after an intended change with:

    uv run --group dev pytest tests/test_goldens.py --update-goldens
"""
import pytest

from conftest import GOLDEN_SEED, point_for


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


def test_seeded_value_matches_its_golden(servicer, goldens, recorder, name):
    value = servicer.evaluate(name, point_for(name), seed=GOLDEN_SEED)
    if recorder is not None:
        recorder[name] = value
        return
    assert name in goldens, f"no golden for {name}; run with --update-goldens"
    assert value == pytest.approx(goldens[name], rel=1e-9), (
        f"{name} at seed {GOLDEN_SEED} returned {value!r}, golden is {goldens[name]!r}")
