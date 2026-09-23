"""Tier 2: drive IOHBenchmarks over a real gRPC channel.

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

from iohbenchmarks.main import IOHServiceServicer  # noqa: E402

BENCHMARK = "bbob-sphere"
DIMENSIONS = 10
VALUE_TYPE = ValueType.CONTINUOUS
FILL = 0.5


@pytest.fixture(scope="module")
def stub():
    with second_level_service(IOHServiceServicer()) as (_port, stub):
        yield stub


def test_a_real_request_returns_a_single_objective(stub):
    result = stub.evaluate_point(
        request_for(BENCHMARK, [FILL] * DIMENSIONS, VALUE_TYPE))
    assert isinstance(result, EvaluationResult)
    assert len(result.objectives) == 1
    assert result.objectives[0].name == "f0"


def test_a_request_carrying_random_seed_is_accepted(stub):
    """This service does not use the seed, but must not reject a request that
    carries one -- and reading it at all requires scaffold >= 0.6.4, which every
    lockfile pinned below until recently. Tier 1 never sees this, because it
    passes seed= directly rather than through a BenchmarkRequest."""
    result = stub.evaluate_point(
        request_for(BENCHMARK, [FILL] * DIMENSIONS, VALUE_TYPE, seed=7))
    assert len(result.objectives) == 1


def test_an_unknown_benchmark_fails_rather_than_hangs(stub):
    with pytest.raises(grpc.RpcError):
        stub.evaluate_point(request_for("definitely-not-a-benchmark", [FILL]))
