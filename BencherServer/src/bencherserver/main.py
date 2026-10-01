import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import grpc
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Mapping

from bencherscaffold.dual_stack_service import (_normalize_hosts, add_listen_argument,
                                                grpc_target, resolve_listen_entries)
from bencherscaffold.protoclasses import bencher_pb2_grpc

from bencherserver.server import BencherServer

# Mapping of default benchmark service ports to their environment variable overrides.
# When a benchmark in the registry uses one of these default ports, the corresponding
# env var (if set) will override it so the BencherServer connects to the right place.
_BENCHMARK_PORT_ENV_VARS: dict[int, str] = {
    50053: 'BENCHER_LASSO_PORT',
    50054: 'BENCHER_NODEP_PORT',
    50055: 'BENCHER_MAXSAT_PORT',
    50056: 'BENCHER_EBO_PORT',
    50057: 'BENCHER_MUJOCO_PORT',
    50058: 'BENCHER_SVM_PORT',
    50059: 'BENCHER_IOH_PORT',
    50060: 'BENCHER_BO4MOB_PORT',
}

_BENCHMARK_HOST_ENV_VARS: dict[int, str] = {
    50053: 'BENCHER_LASSO_HOST',
    50054: 'BENCHER_NODEP_HOST',
    50055: 'BENCHER_MAXSAT_HOST',
    50056: 'BENCHER_EBO_HOST',
    50057: 'BENCHER_MUJOCO_HOST',
    50058: 'BENCHER_SVM_HOST',
    50059: 'BENCHER_IOH_HOST',
    50060: 'BENCHER_BO4MOB_HOST',
}

DEFAULT_REGISTRY_PATH = Path(__file__).parent.parent.parent / 'benchmark-registry.json'


def resolve_targets(
        benchmark_names_to_properties: dict,
        env: Mapping[str, str] | None = None,
) -> dict[tuple[str, int], list[str]]:
    """Group registry entries by the (host, port) that serves them.

    Extracted from serve() so the env-var override logic is testable: it used to
    sit inline above a blocking server start, which meant nothing could exercise
    it. Two details worth knowing --

    * the host env var is looked up by the entry's *original* port, because
      `port` may already have been overridden above;
    * `resolve_listen_entries(...)` may return several hosts but only the first
      is used here, so a comma-separated BENCHER_*_HOST collapses to one target.
    """
    env = os.environ if env is None else env
    targets_to_benchmarks: dict[tuple[str, int], list[str]] = defaultdict(list)

    for benchmark_name, properties in benchmark_names_to_properties.items():
        port = properties['port']
        env_var = _BENCHMARK_PORT_ENV_VARS.get(port)
        if env_var:
            port = int(env.get(env_var, port))
        host = properties.get('host', 'localhost')
        host_env_var = _BENCHMARK_HOST_ENV_VARS.get(properties['port'])
        if host_env_var and host_env_var in env:
            host = _normalize_hosts(env[host_env_var].split(","))[0]
        targets_to_benchmarks[(host, port)].append(benchmark_name)

    return dict(targets_to_benchmarks)


def load_registry(registry_path: Path | None = None) -> dict:
    with open(registry_path or DEFAULT_REGISTRY_PATH, 'r') as f:
        return json.load(f)


def build_server(
        registry: dict | None = None,
        env: Mapping[str, str] | None = None,
        announce: bool = True,
) -> BencherServer:
    """A BencherServer with one stub registered per (host, port) target."""
    server = BencherServer()
    for (host, port), benchmarks in resolve_targets(
            registry if registry is not None else load_registry(), env).items():
        if announce:
            print(f"registering {benchmarks} on {grpc_target(host, port)}")
        server.register_stub(benchmarks, host, port)
    return server


def serve():
    argparse = ArgumentParser()
    argparse.add_argument(
        '-p',
        '--port',
        type=int,
        required=False,
        help='The port number to start the server on. Default is 50051. '
             'Can also be set via the BENCHER_SERVER_PORT environment variable.',
        default=int(os.environ.get('BENCHER_SERVER_PORT', 50051))
    )
    argparse.add_argument(
        '-c',
        '--cores',
        type=int,
        required=False,
        help='The number of CPU cores to use. If None, it will use the maximum number of CPU cores available on the system. Default is cpu_count()',
        default=os.cpu_count()
    )
    add_listen_argument(
        argparse,
        env_var='BENCHER_SERVER_HOST',
        option='--listen-address',
        dest='listen_addresses',
        value_name='Address',
    )
    args = argparse.parse_args()

    bencher_server = build_server()

    port = str(args.port)
    listen_addresses = resolve_listen_entries(args.listen_addresses, env_var='BENCHER_SERVER_HOST')
    n_cores = args.cores
    server = grpc.server(ThreadPoolExecutor(max_workers=n_cores))
    bencher_pb2_grpc.add_BencherServicer_to_server(bencher_server, server)
    bound = 0
    for address in listen_addresses:
        bound += server.add_insecure_port(grpc_target(address, args.port))
    if bound == 0:
        raise RuntimeError(f"Could not bind BencherServer on port {port} for addresses {listen_addresses}")
    server.start()
    addresses_str = ", ".join(listen_addresses)
    print(f"Server started, listening on {port} via {addresses_str}")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
