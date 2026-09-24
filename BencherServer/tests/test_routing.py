"""Tier 2: drive the front door over real gRPC.

Nothing else covers this path. Tier 1 calls each service's evaluate() directly
and Tier 3 needs a built image, so the routing that turns a client request into
a call on the right family service has never been tested.

Fake family services stand in for the real ones, so these run anywhere in
seconds -- no benchmarks, no datasets.
"""
import grpc
import pytest
from bencherscaffold.client import BencherClient
from bencherscaffold.protoclasses.bencher_pb2 import EvaluationResult, Value, ValueType
from conftest import EchoSecondLevel, front_door, second_level_service

from bencherserver.server import BencherServer


def _point(n, fill=0.5):
    return [Value(type=ValueType.CONTINUOUS, value=fill) for _ in range(n)]


def test_a_request_reaches_the_service_that_serves_it(echo_service):
    with echo_service(value=10.0) as (servicer, host, port):
        server = BencherServer()
        server.register_stub(["lasso-dna"], host, port)
        with front_door(server) as door:
            client = BencherClient(address="127.0.0.1", port=door, max_retries=1)
            result = client.evaluate_point("lasso-dna", _point(3))

    assert isinstance(result, EvaluationResult)
    assert result.objectives[0].name == "f0"
    assert result.objectives[0].value == 13.0          # 10.0 + 3 values
    assert len(servicer.requests) == 1
    assert servicer.requests[0].benchmark.name == "lasso-dna"


def test_each_benchmark_routes_to_its_own_service(echo_service):
    """The whole point of the registry: different names, different backends."""
    with echo_service(value=100.0) as (lasso, lasso_host, lasso_port), \
         echo_service(value=200.0) as (svm, svm_host, svm_port):
        server = BencherServer()
        server.register_stub(["lasso-dna"], lasso_host, lasso_port)
        server.register_stub(["svm"], svm_host, svm_port)
        with front_door(server) as door:
            client = BencherClient(address="127.0.0.1", port=door, max_retries=1)
            lasso_value = client.evaluate_point("lasso-dna", _point(1)).objectives[0].value
            svm_value = client.evaluate_point("svm", _point(1)).objectives[0].value

    assert (lasso_value, svm_value) == (101.0, 201.0)
    assert len(lasso.requests) == 1 and len(svm.requests) == 1


def test_the_point_survives_the_hop_unmodified(echo_service):
    """The front door forwards; it must not rescale or reinterpret values."""
    sent = [0.0, 0.25, 1.0]
    with echo_service() as (servicer, host, port):
        server = BencherServer()
        server.register_stub(["b"], host, port)
        with front_door(server) as door:
            BencherClient(address="127.0.0.1", port=door, max_retries=1) \
                .evaluate_point("b", [Value(type=ValueType.CONTINUOUS, value=v) for v in sent])

    assert [v.value for v in servicer.requests[0].point.values] == sent


def test_random_seed_is_forwarded_to_the_service(echo_service):
    """Added in scaffold 0.6.4; the front door must pass it through untouched."""
    with echo_service() as (servicer, host, port):
        server = BencherServer()
        server.register_stub(["b"], host, port)
        with front_door(server) as door:
            BencherClient(address="127.0.0.1", port=door, max_retries=1) \
                .evaluate_point("b", _point(2), random_seed=4242)

    request = servicer.requests[0]
    assert request.HasField("random_seed")
    assert request.random_seed == 4242


def test_an_unknown_benchmark_is_rejected(echo_service):
    with echo_service() as (_servicer, host, port):
        server = BencherServer()
        server.register_stub(["known"], host, port)
        with front_door(server) as door:
            client = BencherClient(address="127.0.0.1", port=door, max_retries=1)
            with pytest.raises(grpc.RpcError):
                client.evaluate_point("not-registered", _point(1))


def test_a_failing_service_surfaces_as_an_error_not_a_hang(echo_service):
    """A backend that aborts must propagate, not stall the caller."""
    with echo_service(fail_with=grpc.StatusCode.UNAVAILABLE) as (_s, host, port):
        server = BencherServer()
        server.register_stub(["b"], host, port)
        with front_door(server) as door:
            client = BencherClient(address="127.0.0.1", port=door,
                                   max_retries=1, wait_time=0)
            with pytest.raises(grpc.RpcError):
                client.evaluate_point("b", _point(1))


def test_registering_a_name_twice_is_refused():
    """Two services claiming one benchmark would route non-deterministically."""
    server = BencherServer()
    server.register_stub(["dup"], "127.0.0.1", 1)
    with pytest.raises(AssertionError, match="already registered"):
        server.register_stub(["dup"], "127.0.0.1", 2)


def test_an_unreachable_backend_does_not_mask_the_error(echo_service):
    """Regression: evaluate_point dereferenced a None context in its except
    block, replacing the real gRPC failure with an AttributeError."""
    server = BencherServer()
    server.register_stub(["b"], "127.0.0.1", 1)      # nothing listening
    from bencherscaffold.protoclasses.bencher_pb2 import Benchmark, BenchmarkRequest, Point
    request = BenchmarkRequest(benchmark=Benchmark(name="b"), point=Point(values=_point(1)))
    with pytest.raises(grpc.RpcError):
        server.evaluate_point(request, context=None)
