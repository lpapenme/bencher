import json
from concurrent.futures import ThreadPoolExecutor

import grpc
import os
from argparse import ArgumentParser

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
    args = argparse.parse_args()

    bencher_server = BencherServer()

    # load relative to this file
    benchmark_names_to_properties = json.load(
        open(os.path.join(os.path.dirname(__file__).parent, 'benchmark-registry.json'), 'r'),
    )

    # structure: {benchmark_name: {port: int, dimensions: int}}
    ports_to_benchmarks = dict()

    for benchmark_name, properties in benchmark_names_to_properties.items():
        port = properties['port']
        if port not in ports_to_benchmarks:
            ports_to_benchmarks[port] = []
        ports_to_benchmarks[port].append(benchmark_name)

    for port, benchmarks in ports_to_benchmarks.items():
        print(f"registering {benchmarks} on port {port}")
        bencher_server.register_stub(benchmarks, port)

    port = str(args.port)
    n_cores = args.cores
    server = grpc.server(ThreadPoolExecutor(max_workers=n_cores))
    bencher_pb2_grpc.add_BencherServicer_to_server(bencher_server, server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
