import traceback

import grpc
import os

from bencherscaffold.protoclasses import second_level_services_pb2_grpc
from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult
from bencherscaffold.protoclasses.bencher_pb2_grpc import BencherServicer


class BencherServer(BencherServicer):

    def __init__(
            self,
            port: int = 50051,
            n_cores: int | None = None,
            stubs: dict[str, second_level_services_pb2_grpc.SecondLevelBencherStub] | None = None
    ):
        """
        Args:
            port (int): The port number to start the server on. Default is 50051.
            n_cores (int | None): The number of CPU cores to use. If None, it will use the maximum number of CPU cores available on the system. Default is None.
            stubs (dict[str, second_level_services_pb2_grpc.SecondLevelBencherStub] | None): A dictionary containing the stubs for second level services. Each key is a string representing the
        * name of the service, and each value is the corresponding stub object. If None, an empty dictionary will be created. Default is None.
        """
        self.stubs = stubs or {}
        self.port = port
        self.n_cores = n_cores or os.cpu_count()
        self.server = None

    def register_stub(
            self,
            names: list[str],
            host: str,
            port: int
    ):
        """
        Registers a stub for a given list of names and port.

        Args:
            names (list[str]): A list of names to register the stub.
            host (str): The host on which the stub is running.
            port (int): The port on which the stub is running.

        Returns:
            None
        """
        target = f"{host}:{port}"
        stub = second_level_services_pb2_grpc.SecondLevelBencherStub(
            grpc.insecure_channel(target)
        )
        for name in names:
            assert name not in self.stubs, f"Name {name} already registered"
            self.stubs[name] = stub

    def evaluate_point(
            self,
            request: BenchmarkRequest,
            context: grpc.ServicerContext | None = None
    ) -> EvaluationResult:
        """
        Args:
            request: The BenchmarkRequest object containing the details of the benchmark evaluation request.
            context: The grpc.ServicerContext object representing the context of the evaluation request.

        Returns:
            An EvaluationResult object representing the result of the evaluation.

        Raises:
            AssertionError: If the specified benchmark name is not valid.

        """
        benchmark_name = request.benchmark.name

        assert benchmark_name in self.stubs, f"Invalid benchmark name {benchmark_name}, available: {list(self.stubs.keys())}"
        stub = self.stubs[benchmark_name]
        try:
            response = stub.evaluate_point(request)
        except grpc.RpcError as e:
            stack_trace = traceback.format_exc()
            context.set_details(stack_trace)
            context.set_code(grpc.StatusCode.INTERNAL)
            raise e
        return response
