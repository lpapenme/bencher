"""Tier 1 goldens for BoTorchBenchmarks: every declared benchmark at mid-domain.

Parametrized over BENCHMARKS, so a problem added there fails here until a value
is recorded for it -- and a botorch upgrade that changes a test function fails
here too. Constraint values are pinned alongside the objective, as `<name>/g<i>`.

Regenerate with:  uv run --group dev pytest tests --update-goldens
"""

import pytest
from conftest import NAMES, point_for


def _outputs(name, result):
    values = {name: result.objectives[0].value}
    values.update({f"{name}/{c.name}": c.value for c in result.constraints})
    return values


@pytest.mark.parametrize("name", NAMES)
def test_values_match_their_goldens(servicer, golden_for, recorder, name):
    values = _outputs(name, servicer.evaluate(name, point_for(name)))
    if recorder is not None:
        recorder.update(values)
        return
    for key, value in values.items():
        expected = golden_for(key)
        assert expected is not None, (
            f"no golden recorded for {key} on this platform; run with --update-goldens"
        )
        assert value == pytest.approx(expected, rel=1e-9), key
