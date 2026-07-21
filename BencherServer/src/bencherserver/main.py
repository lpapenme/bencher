import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import grpc
import os
from argparse import ArgumentParser
from pathlib import Path

from bencherscaffold.dual_stack_service import add_listen_argument, grpc_target, resolve_listen_entries
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

    bencher_server = BencherServer()

    file_path = Path(__file__).parent.parent.parent / 'benchmark-registry.json'
    with open(file_path, 'r') as f:
        benchmark_names_to_properties = json.load(f)

    # structure: {benchmark_name: {port: int, dimensions: int}}
    targets_to_benchmarks: dict[tuple[str, int], list[str]] = defaultdict(list)

    for benchmark_name, properties in benchmark_names_to_properties.items():
        port = properties['port']
        env_var = _BENCHMARK_PORT_ENV_VARS.get(port)
        if env_var:
            port = int(os.environ.get(env_var, port))
        host = properties.get('host', 'localhost')
        host_env_var = _BENCHMARK_HOST_ENV_VARS.get(properties['port'])
        if host_env_var:
            host = resolve_listen_entries(None, env_var=host_env_var, default=(host,))[0]
        targets_to_benchmarks[(host, port)].append(benchmark_name)

    for (host, port), benchmarks in targets_to_benchmarks.items():
        print(f"registering {benchmarks} on {grpc_target(host, port)}")
        bencher_server.register_stub(benchmarks, host, port)

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
