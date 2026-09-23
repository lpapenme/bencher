"""Tier 2: drive MujocoBenchmarks over a real gRPC channel.

Tier 1 calls evaluate() directly, so it never sees request parsing or the
Value -> evaluate() hop. This service registers as a SecondLevelBencherServicer
(not Bencher), so it is driven with a SecondLevelBencherStub rather than
BencherClient, which only speaks to the front door.
"""
import sys
from pathlib import Path

import grpc
import pytest
from bencherscaffold.protoclasses.bencher_pb2 import EvaluationResult, ValueType

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from grpc_harness import request_for, second_level_service  # noqa: E402

from mujocobenchmarks.main import MujocoServiceServicer  # noqa: E402

# Two code paths live behind this service and they seed differently:
# lunarlander runs a gym episode, everything else goes through a MuJoCo rollout
# factory. Testing only one hides a regression in the other -- the seed-drop bug
# was on the factory path, and a lunarlander-only test passed straight through
# it. `mujoco-swimmer` is the cheapest factory-path benchmark.
BENCHMARK = "mujoco-swimmer"
DIMENSIONS = 16
VALUE_TYPE = ValueType.CONTINUOUS
FILL = 0.5

SEEDED_BENCHMARKS = [
    ("mujoco-swimmer", 16),   # factory path
    ("lunarlander", 12),      # gym path
]


@pytest.fixture(scope="module")
def stub():
    with second_level_service(MujocoServiceServicer()) as (_port, stub):
        yield stub


def test_a_real_request_returns_a_single_objective(stub):
    result = stub.evaluate_point(
        request_for(BENCHMARK, [FILL] * DIMENSIONS, VALUE_TYPE))
    assert isinstance(result, EvaluationResult)
    assert len(result.objectives) == 1
    assert result.objectives[0].name == "f0"


@pytest.mark.parametrize("benchmark,dimensions", SEEDED_BENCHMARKS)
def test_random_seed_makes_the_rollout_reproducible_over_grpc(stub, benchmark, dimensions):
    """The full path: random_seed on the wire -> HasField -> evaluate(seed=...).

    Tier 1 covers evaluate(seed=...) directly, so it cannot catch the request
    field going unread. Reading it also requires scaffold >= 0.6.4, which every
    lockfile pinned below until recently -- that would have raised ValueError
    here on every call.
    """
    def value(seed):
        request = request_for(benchmark, [FILL] * dimensions, VALUE_TYPE, seed=seed)
        return stub.evaluate_point(request).objectives[0].value

    assert value(7) == value(7)
    assert value(7) != value(8)


@pytest.mark.parametrize("benchmark,dimensions", SEEDED_BENCHMARKS)
def test_without_a_seed_the_rollout_still_varies(stub, benchmark, dimensions):
    """Seeding stays optional: an unseeded request must not be pinned."""
    request = request_for(benchmark, [FILL] * dimensions, VALUE_TYPE)
    assert stub.evaluate_point(request).objectives[0].value != \
           stub.evaluate_point(request).objectives[0].value


def test_an_unknown_benchmark_fails_rather_than_hangs(stub):
    with pytest.raises(grpc.RpcError):
        stub.evaluate_point(request_for("definitely-not-a-benchmark", [FILL]))
