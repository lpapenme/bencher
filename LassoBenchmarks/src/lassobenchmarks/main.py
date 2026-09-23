import logging
import os
from argparse import ArgumentParser

import LassoBench
import numpy as np
from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult, ObjectiveValue
from bencherscaffold.dual_stack_service import DualStackGRCPService, add_listen_argument, resolve_listen_entries

LISTEN_HOST_ENV_VAR = 'BENCHER_LASSO_HOST'

def eval_lasso(
        x: np.ndarray,
        benchmark
):
    return benchmark.evaluate(x)


# The benchmarks this service serves, as data: dimensionality and type mirror
# benchmark-registry.json (tests/test_registry.py checks they agree), and
# `factory` builds the underlying LassoBench problem on demand.
BENCHMARKS = {
    'lasso-dna': {
        'dimensions': 180, 'type': 'purely_continuous',
        'factory': lambda: LassoBench.RealBenchmark(pick_data='dna', mf_opt='discrete_fidelity')},
    'lasso-simple': {
        'dimensions': 60, 'type': 'purely_continuous',
        'factory': lambda: LassoBench.SyntheticBenchmark(pick_bench='synt_simple')},
    'lasso-medium': {
        'dimensions': 100, 'type': 'purely_continuous',
        'factory': lambda: LassoBench.SyntheticBenchmark(pick_bench='synt_medium')},
    'lasso-high': {
        'dimensions': 300, 'type': 'purely_continuous',
        'factory': lambda: LassoBench.SyntheticBenchmark(pick_bench='synt_high')},
    'lasso-hard': {
        'dimensions': 1000, 'type': 'purely_continuous',
        'factory': lambda: LassoBench.SyntheticBenchmark(pick_bench='synt_hard')},
    'lasso-leukemia': {
        'dimensions': 7129, 'type': 'purely_continuous',
        'factory': lambda: LassoBench.RealBenchmark(pick_data='leukemia', mf_opt='discrete_fidelity')},
    'lasso-rcv1': {
        'dimensions': 47236, 'type': 'purely_continuous',
        'factory': lambda: LassoBench.RealBenchmark(pick_data='rcv1', mf_opt='discrete_fidelity')},
    'lasso-diabetes': {
        'dimensions': 8, 'type': 'purely_continuous',
        'factory': lambda: LassoBench.RealBenchmark(pick_data='diabetes', mf_opt='discrete_fidelity')},
    'lasso-breastcancer': {
        'dimensions': 10, 'type': 'purely_continuous',
        'factory': lambda: LassoBench.RealBenchmark(pick_data='breast_cancer', mf_opt='discrete_fidelity')},
}


class LassoServiceServicer(DualStackGRCPService):

    def __init__(
            self,
            port: int = 50053,
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
        """Evaluate a point given in [0, 1]^d.

        Split out of evaluate_point so the rescaling can be tested without a
        gRPC server. LassoBench problems are deterministic, so `seed` is unused.
        """
        if name not in BENCHMARKS:
            raise ValueError(
                f"Invalid benchmark name {name!r}; this service serves {sorted(BENCHMARKS)}")
        # lasso benchmarks are in [-1, 1] while x is in [0, 1], so we need to scale it
        x = 2 * x - 1
        return eval_lasso(x, BENCHMARKS[name]['factory']())


def serve():
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=int(os.environ.get('BENCHER_LASSO_PORT', 50053)),
        help='The port number to start the server on. Default is 50053. '
             'Can also be set via the BENCHER_LASSO_PORT environment variable.',
    )
    add_listen_argument(parser, env_var=LISTEN_HOST_ENV_VAR)
    args = parser.parse_args()

    logging.basicConfig()
    lasso = LassoServiceServicer(port=args.port, listen_hosts=resolve_listen_entries(args.listen_hosts, env_var=LISTEN_HOST_ENV_VAR))
    lasso.serve()


if __name__ == '__main__':
    serve()
