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
            stubs: dict[str, second_level_services_pb2_grpc.SecondLevelBencherStub] | None = None,
            prefix_stubs: list[tuple[str, second_level_services_pb2_grpc.SecondLevelBencherStub]] | None = None,
    ):
        """
        Args:
            port (int): The port number to start the server on. Default is 50051.
            n_cores (int | None): The number of CPU cores to use. If None, it will use the maximum number of CPU cores available on the system. Default is None.
            stubs (dict[str, second_level_services_pb2_grpc.SecondLevelBencherStub] | None): A dictionary containing the stubs for second level services. Each key is a string representing the
        * name of the service, and each value is the corresponding stub object. If None, an empty dictionary will be created. Default is None.
            prefix_stubs: Optional list of (prefix, stub) pairs used as a fallback for benchmark names not found in `stubs`. The longest matching prefix wins. Usually populated via `register_prefix_stub`.
        """
        self.stubs = stubs or {}
        # Fallback routes: list of (prefix, stub), kept sorted by descending prefix length
        # so the longest matching prefix is found first on lookup.
        self.prefix_stubs: list[tuple[str, second_level_services_pb2_grpc.SecondLevelBencherStub]] = list(prefix_stubs or [])
        self._sort_prefix_stubs()
        self.port = port
        self.n_cores = n_cores or os.cpu_count()
        self.server = None

    def _sort_prefix_stubs(self) -> None:
        self.prefix_stubs.sort(key=lambda p: len(p[0]), reverse=True)

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

    def register_prefix_stub(
            self,
            prefix: str,
            host: str,
            port: int,
    ) -> None:
        """
        Registers a fallback stub for any benchmark name beginning with `prefix`.

        Exact-name stubs (registered via `register_stub`) always take precedence over prefix
        stubs; the prefix table is only consulted when no exact match exists. Among multiple
        matching prefixes the longest one wins (so more specific prefixes can override broader
        ones).
        """
        target = f"{host}:{port}"
        stub = second_level_services_pb2_grpc.SecondLevelBencherStub(
            grpc.insecure_channel(target)
        )
        self.prefix_stubs.append((prefix, stub))
        self._sort_prefix_stubs()

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

        stub = self.stubs.get(benchmark_name)
        if stub is None:
            for prefix, s in self.prefix_stubs:
                if benchmark_name.startswith(prefix):
                    stub = s
                    break
        assert stub is not None, (
            f"Invalid benchmark name {benchmark_name}. "
            f"Exact stubs: {len(self.stubs)} registered. "
            f"Prefix stubs: {[p for p, _ in self.prefix_stubs]}."
        )
        try:
            response = stub.evaluate_point(request)
        except grpc.RpcError as e:
            stack_trace = traceback.format_exc()
            context.set_details(stack_trace)
            context.set_code(grpc.StatusCode.INTERNAL)
            raise e
        return response
