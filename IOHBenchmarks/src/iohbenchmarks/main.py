import logging

import ioh.iohcpp
import numpy as np
from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult
from bencherscaffold.dual_stack_service import DualStackGRCPService
from ioh import get_problem, ProblemClass
from ioh.iohcpp.problem import MaxCoverage


class IOHServiceServicer(DualStackGRCPService):

    def __init__(
            self
    ):
        super().__init__(port=50059, n_cores=1)

    def evaluate_point(
            self,
            request: BenchmarkRequest,
            context
    ) -> EvaluationResult:
        x = [v.value for v in request.point.values]
        x = np.array(x)
        dimension = x.shape[0]
        if request.benchmark.name.strip().startswith('bbob-'):
            print(f"Evaluating {request.benchmark.name} with dimension {dimension}")
            bname_trunc = request.benchmark.name.split('-')[1]
            benchmark_candidate = ioh.iohcpp.problem.BBOB.problems

            pname_pid = [
                (name, pid) for pid, name in benchmark_candidate.items() if name.lower().startswith(bname_trunc)
            ]
            if len(pname_pid) == 0:
                raise ValueError(
                    f"Benchmark {request.benchmark.name} not supported. Supported benchmarks are: {list(benchmark_candidate.values())}"
                )
            pname, pid = pname_pid[0]
            problemclass = ProblemClass.BBOB
            point_type = np.float64
        elif request.benchmark.name.strip().startswith('pbo-'):
            print(f"Evaluating {request.benchmark.name} with dimension {dimension}")
            bname_trunc = request.benchmark.name.split('-')[1]
            benchmark_candidate = ioh.iohcpp.problem.PBO.problems

            pname_pid = [
                (name, pid) for pid, name in benchmark_candidate.items() if name.lower().startswith(bname_trunc)
            ]
            if len(pname_pid) == 0:
                raise ValueError(
                    f"Benchmark {request.benchmark.name} not supported. Supported benchmarks are: {list(benchmark_candidate.values())}"
                )
            pname, pid = pname_pid[0]
            problemclass = ProblemClass.PBO
            point_type = np.int64
        elif request.benchmark.name.strip().startswith('graph-'):
            print(f"Evaluating {request.benchmark.name} with dimension {dimension}")
            bname_trunc = request.benchmark.name.split('-')[1]
            benchmark_candidate = ioh.iohcpp.problem.GraphProblem.problems

            pname_pid = [
                (name, pid) for pid, name in benchmark_candidate.items() if name.lower().startswith(bname_trunc)
            ]
            if len(pname_pid) == 0:
                raise ValueError(
                    f"Benchmark {request.benchmark.name} not supported. Supported benchmarks are: {list(benchmark_candidate.values())}"
                )
            pname, pid = pname_pid[0]
            problemclass = ProblemClass.GRAPH
            point_type = np.int64

        else:
            raise ValueError(
                f"Benchmark {request.benchmark.name} not supported. Supported benchmarks are: {list(ioh.iohcpp.problem.BBOB.problems.values()) + list(ioh.iohcpp.problem.PBO.problems.values())}"
            )

        benchmark = get_problem(pname, pid, dimension, problemclass)
        bounds = benchmark.bounds
        MaxCoverage
        if bounds is not None:
            x = (x - bounds.lb) / (bounds.ub - bounds.lb)
            y = benchmark(x.astype(point_type))
        result = EvaluationResult(
            value=y,
        )
        return result


def serve():
    logging.basicConfig()
    ioh = IOHServiceServicer()
    ioh.serve()


if __name__ == '__main__':
    serve()
