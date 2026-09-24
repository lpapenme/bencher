import logging
import os
from argparse import ArgumentParser

import ioh.iohcpp
import numpy as np
from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult, ObjectiveValue
from bencherscaffold.dual_stack_service import DualStackGRCPService, add_listen_argument, resolve_listen_entries
from ioh import get_problem, ProblemClass
from ioh.iohcpp.problem import MaxCoverage

LISTEN_HOST_ENV_VAR = 'BENCHER_IOH_HOST'

# The three IOH families this service exposes. Names are matched by prefix
# because IOH problems are parameterised, so unlike the other services there is
# no finite BENCHMARKS mapping; known_names() enumerates them for the tests.
# The last field says whether a point in [0, 1]^d must be rescaled onto the
# problem's own bounds. Only the continuous family needs it: the discrete
# families take 0/1 directly from the client. Graph problems in particular
# report int32 sentinels (-2147483648 .. 2147483647) rather than real bounds, so
# rescaling against them produced values the C++ binding rejected outright --
# every graph-* benchmark raised TypeError before this flag existed.
PROBLEM_FAMILIES = {
    'bbob-': (lambda: ioh.iohcpp.problem.BBOB.problems, ProblemClass.BBOB, np.float64, True),
    'pbo-': (lambda: ioh.iohcpp.problem.PBO.problems, ProblemClass.PBO, np.int64, False),
    'graph-': (lambda: ioh.iohcpp.problem.GraphProblem.problems, ProblemClass.GRAPH, np.int64, False),
}


def resolve_problem(name: str):
    """Map a bencher benchmark name onto an IOH (name, id, class, dtype).

    Matching prefers an exact case-insensitive match and only then falls back to
    a unique prefix. Prefix-only matching was ambiguous: `bbob-rosenbrock` could
    resolve to either Rosenbrock or RosenbrockRotated depending on dict order.
    """
    name = name.strip()
    for prefix, (problems, problem_class, point_type, rescale) in PROBLEM_FAMILIES.items():
        if not name.startswith(prefix):
            continue
        # split once: the suffix itself may contain hyphens.
        suffix = name.split('-', 1)[1].lower()
        candidates = problems()

        exact = [(n, pid) for pid, n in candidates.items() if n.lower() == suffix]
        if exact:
            problem_name, problem_id = exact[0]
            return problem_name, problem_id, problem_class, point_type, rescale

        partial = sorted((n, pid) for pid, n in candidates.items()
                         if n.lower().startswith(suffix))
        if len(partial) == 1:
            problem_name, problem_id = partial[0]
            return problem_name, problem_id, problem_class, point_type, rescale
        if len(partial) > 1:
            raise ValueError(
                f"Benchmark {name} is ambiguous; it matches {[n for n, _ in partial]}. "
                f"Use the full problem name.")
        raise ValueError(
            f"Benchmark {name} not supported. Supported benchmarks are: "
            f"{sorted(candidates.values())}")

    raise ValueError(
        f"Benchmark {name} not supported: expected one of the prefixes "
        f"{sorted(PROBLEM_FAMILIES)}")


def known_names() -> set:
    """Every benchmark name this service can serve, in bencher's naming."""
    return {
        f"{prefix}{problem_name.lower()}"
        for prefix, (problems, _, _, _) in PROBLEM_FAMILIES.items()
        for problem_name in problems().values()
    }


class IOHServiceServicer(DualStackGRCPService):

    def __init__(
            self,
            port: int = 50059,
            listen_hosts=None
    ):
        super().__init__(port=port, n_cores=1, listen_hosts=listen_hosts)

    def evaluate_point(
            self,
            request: BenchmarkRequest,
            context
    ) -> EvaluationResult:
        x = np.array([v.value for v in request.point.values])
        value = self.evaluate(request.benchmark.name, x)
        return EvaluationResult(
            objectives=[ObjectiveValue(name="f0", value=value)],
        )

    def evaluate(
            self,
            name: str,
            x: np.ndarray,
            seed: int | None = None
    ) -> float:
        """Evaluate a point given in [0, 1]^d against an IOH problem.

        IOH problems are dimension-agnostic: the dimension is taken from the
        point, which is why the registry records `dimensions: null` for bbob and
        pbo. Split out of evaluate_point so it is testable without a server.
        """
        dimension = x.shape[0]
        problem_name, problem_id, problem_class, point_type, rescale = resolve_problem(name)
        print(f"Evaluating {name} with dimension {dimension}")

        benchmark = get_problem(problem_name, problem_id, dimension, problem_class)
        bounds = benchmark.bounds
        if rescale and bounds is not None:
            x = bounds.lb + x * (bounds.ub - bounds.lb)
        # `y` used to be assigned only inside the branch above, so a problem
        # reporting no bounds raised UnboundLocalError instead of evaluating.
        return benchmark(x.astype(point_type))


def serve():
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=int(os.environ.get('BENCHER_IOH_PORT', 50059)),
        help='The port number to start the server on. Default is 50059. '
             'Can also be set via the BENCHER_IOH_PORT environment variable.',
    )
    add_listen_argument(parser, env_var=LISTEN_HOST_ENV_VAR)
    args = parser.parse_args()

    logging.basicConfig()
    ioh_service = IOHServiceServicer(port=args.port, listen_hosts=resolve_listen_entries(args.listen_hosts, env_var=LISTEN_HOST_ENV_VAR))
    ioh_service.serve()


if __name__ == '__main__':
    serve()
