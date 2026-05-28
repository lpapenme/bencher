import logging
import os
import re
from argparse import ArgumentParser

import ioh.iohcpp
import numpy as np
from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult
from bencherscaffold.dual_stack_service import DualStackGRCPService
from ioh import get_problem, ProblemClass
from ioh.iohcpp.problem import MaxCoverage

# Matches `bbob-<fn>` (instance defaults to 1) and `bbob-<fn>-i<N>` (instance N).
# The function name token is [a-z0-9]+ so it does not consume the optional `-i<digits>`
# suffix. Examples: bbob-sphere, bbob-sphere-i7, bbob-schaffers10, bbob-schaffers10-i3.
BBOB_NAME_RE = re.compile(r'^bbob-([a-z0-9]+)(?:-i(\d+))?$')


class IOHServiceServicer(DualStackGRCPService):

    def __init__(
            self,
            port: int = 50059
    ):
        super().__init__(port=port, n_cores=1)

    def evaluate_point(
            self,
            request: BenchmarkRequest,
            context
    ) -> EvaluationResult:
        x = [v.value for v in request.point.values]
        x = np.array(x)
        dimension = x.shape[0]
        name = request.benchmark.name.strip()

        if name.startswith('bbob-'):
            m = BBOB_NAME_RE.match(name)
            if m is None:
                raise ValueError(
                    f"Invalid BBOB benchmark name {name!r}. Expected 'bbob-<fn>' or 'bbob-<fn>-i<N>'."
                )
            bname_trunc = m.group(1)
            instance = int(m.group(2)) if m.group(2) else 1
            print(f"Evaluating {name} (fn={bname_trunc}, instance={instance}, dim={dimension})")
            benchmark_candidate = ioh.iohcpp.problem.BBOB.problems
            pname_pid = [
                (n, pid) for pid, n in benchmark_candidate.items() if n.lower().startswith(bname_trunc)
            ]
            if len(pname_pid) == 0:
                raise ValueError(
                    f"Unknown BBOB function in {name!r}. Supported: {list(benchmark_candidate.values())}"
                )
            pname, _pid = pname_pid[0]
            problemclass = ProblemClass.BBOB
            point_type = np.float64
        elif name.startswith('pbo-'):
            print(f"Evaluating {name} with dimension {dimension}")
            bname_trunc = name.split('-')[1]
            benchmark_candidate = ioh.iohcpp.problem.PBO.problems
            pname_pid = [
                (n, pid) for pid, n in benchmark_candidate.items() if n.lower().startswith(bname_trunc)
            ]
            if len(pname_pid) == 0:
                raise ValueError(
                    f"Benchmark {name} not supported. Supported benchmarks are: {list(benchmark_candidate.values())}"
                )
            pname, pid = pname_pid[0]
            # PBO: preserve pre-existing behavior (instance == function id). Not extended
            # to instance-suffix naming in this change; that's a separate ask.
            instance = pid
            problemclass = ProblemClass.PBO
            point_type = np.int64
        elif name.startswith('graph-'):
            print(f"Evaluating {name} with dimension {dimension}")
            bname_trunc = name.split('-')[1]
            benchmark_candidate = ioh.iohcpp.problem.GraphProblem.problems
            pname_pid = [
                (n, pid) for pid, n in benchmark_candidate.items() if n.lower().startswith(bname_trunc)
            ]
            if len(pname_pid) == 0:
                raise ValueError(
                    f"Benchmark {name} not supported. Supported benchmarks are: {list(benchmark_candidate.values())}"
                )
            pname, pid = pname_pid[0]
            # GRAPH: same as PBO — preserve pre-existing behavior.
            instance = pid
            problemclass = ProblemClass.GRAPH
            point_type = np.int64
        else:
            raise ValueError(
                f"Benchmark {name} not supported. Supported benchmarks are: {list(ioh.iohcpp.problem.BBOB.problems.values()) + list(ioh.iohcpp.problem.PBO.problems.values())}"
            )

        benchmark = get_problem(pname, instance, dimension, problemclass)
        bounds = benchmark.bounds
        if bounds is not None:
            x = bounds.lb + x * (bounds.ub - bounds.lb)
            y = benchmark(x.astype(point_type))
        result = EvaluationResult(
            value=y,
        )
        return result


def serve():
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=int(os.environ.get('BENCHER_IOH_PORT', 50059)),
        help='The port number to start the server on. Default is 50059. '
             'Can also be set via the BENCHER_IOH_PORT environment variable.',
    )
    args = parser.parse_args()

    logging.basicConfig()
    ioh_service = IOHServiceServicer(port=args.port)
    ioh_service.serve()


if __name__ == '__main__':
    serve()
