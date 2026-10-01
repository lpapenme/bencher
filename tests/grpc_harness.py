"""Shared harness for Tier 2: drive a real servicer over a real gRPC channel.

Tier 1 calls `evaluate()` directly and Tier 3 needs a built image; this covers
what sits between them -- request parsing, the Value -> evaluate() path, and
`HasField('random_seed')` actually being read off a real BenchmarkRequest.

Two things to know about the services here:

* Family services register as `SecondLevelBencherServicer` (via GRCPService),
  *not* `Bencher`. Only the front door speaks `Bencher`, which is why
  BencherClient cannot be pointed at a family service directly.
* `DualStackGRCPService.serve()` binds its configured port and then blocks on
  `wait_for_termination()`, so tests build their own server rather than call it.

Servers bind 127.0.0.1:0 so the OS assigns a free port. That keeps tests off the
hardcoded 50053-50060 range, which collides with whatever else is running -- on
one machine here 50059 is held by a VPN helper.

Loaded by package conftests via sys.path, the same way tests/e2e/conftest.py
loads benchmarks.py. It needs only grpc and bencherscaffold, both present in
every package virtualenv.
"""
import contextlib
from concurrent.futures import ThreadPoolExecutor

import grpc
from bencherscaffold.protoclasses import bencher_pb2_grpc, second_level_services_pb2_grpc
from bencherscaffold.protoclasses.bencher_pb2 import (Benchmark, BenchmarkRequest,
                                                      EvaluationResult, ObjectiveValue,
                                                      Point, Value, ValueType)


@contextlib.contextmanager
def _serving(servicer, register, max_workers=4):
    """Serve `servicer` on an OS-assigned port; yield that port."""
    server = grpc.server(ThreadPoolExecutor(max_workers=max_workers))
    register(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    if port == 0:
        server.stop(None).wait(timeout=5)
        raise AssertionError("could not bind an ephemeral port on 127.0.0.1")
    server.start()
    try:
        yield port
    finally:
        # stop(None) cancels in-flight RPCs immediately; a grace period here is
        # the usual cause of slow or hanging teardown.
        server.stop(None).wait(timeout=10)


@contextlib.contextmanager
def second_level_service(servicer):
    """Serve a family service; yield (port, SecondLevelBencherStub)."""
    register = second_level_services_pb2_grpc.add_SecondLevelBencherServicer_to_server
    with _serving(servicer, register) as port:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            yield port, second_level_services_pb2_grpc.SecondLevelBencherStub(channel)


@contextlib.contextmanager
def front_door(servicer):
    """Serve a BencherServer; yield the port a BencherClient can connect to."""
    with _serving(servicer, bencher_pb2_grpc.add_BencherServicer_to_server) as port:
        yield port


def request_for(name, values, value_type=ValueType.CONTINUOUS, seed=None):
    """A BenchmarkRequest as a client would send it: values in [0, 1]."""
    request = BenchmarkRequest(
        benchmark=Benchmark(name=name),
        point=Point(values=[Value(type=value_type, value=v) for v in values]),
    )
    if seed is not None:
        request.random_seed = seed
    return request


def single_objective(value, name="f0"):
    """The shape every service is expected to answer with."""
    return EvaluationResult(objectives=[ObjectiveValue(name=name, value=value)])


class EchoSecondLevel(second_level_services_pb2_grpc.SecondLevelBencherServicer):
    """A stand-in family service: records requests, answers predictably.

    Used by the front-door tests so routing can be checked without standing up a
    real benchmark.
    """

    def __init__(self, value=42.0, fail_with=None):
        self.requests = []
        self.value = value
        self.fail_with = fail_with

    def evaluate_point(self, request, context):
        self.requests.append(request)
        if self.fail_with is not None:
            context.abort(self.fail_with, "configured to fail")
        return single_objective(self.value + len(request.point.values))
