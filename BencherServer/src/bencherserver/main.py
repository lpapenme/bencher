import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import grpc
import os
from argparse import ArgumentParser
from pathlib import Path

from bencherscaffold.protoclasses import bencher_pb2_grpc

from bencherserver.server import BencherServer


def serve():
    argparse = ArgumentParser()
    argparse.add_argument(
        '-p',
        '--port',
        type=int,
        required=False,
        help='The port number to start the server on. Default is 50051.',
        default=50051
    )
    argparse.add_argument(
        '-c',
        '--cores',
        type=int,
        required=False,
        help='The number of CPU cores to use. If None, it will use the maximum number of CPU cores available on the system. Default is cpu_count()',
        default=os.cpu_count()
    )
    argparse.add_argument(
        '--listen-address',
        dest='listen_addresses',
        action='append',
        required=False,
        help='Explicit address to bind on (e.g., 127.0.0.1, [::]). Defaults to binding both IPv4 and IPv6 loopback.',
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
        host = properties.get('host', 'localhost')
        targets_to_benchmarks[(host, port)].append(benchmark_name)

    for (host, port), benchmarks in targets_to_benchmarks.items():
        print(f"registering {benchmarks} on {host}:{port}")
        bencher_server.register_stub(benchmarks, host, port)

    port = str(args.port)
    listen_addresses = args.listen_addresses or ['0.0.0.0', '[::]']
    listen_addresses = [address.strip() for address in listen_addresses if address and address.strip()]
    n_cores = args.cores
    server = grpc.server(ThreadPoolExecutor(max_workers=n_cores))
    bencher_pb2_grpc.add_BencherServicer_to_server(bencher_server, server)
    bound = 0
    for address in listen_addresses:
        bound += server.add_insecure_port(f"{address}:{port}")
    if bound == 0:
        raise RuntimeError(f"Could not bind BencherServer on port {port} for addresses {listen_addresses}")
    server.start()
    addresses_str = ", ".join(listen_addresses)
    print(f"Server started, listening on {port} via {addresses_str}")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
