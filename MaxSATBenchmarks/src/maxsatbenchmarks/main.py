import threading

import logging
import numpy as np
import os
import pathlib
from argparse import ArgumentParser
from functools import lru_cache

from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult, ObjectiveValue
from bencherscaffold.dual_stack_service import DualStackGRCPService, add_listen_argument, resolve_listen_entries

from maxsatbenchmarks.data_loading import download_maxsat60_data, download_maxsat125_data
from maxsatbenchmarks.wcnf import WCNF

LISTEN_HOST_ENV_VAR = 'BENCHER_MAXSAT_HOST'
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "maxsat"
DATA_DIR.mkdir(parents=True, exist_ok=True)
directory_name = str(DATA_DIR)

# The benchmarks this service serves, as data. One entry per benchmark rather
# than four dicts keyed the same way, so adding a corpus cannot half-land and
# benchmark-registry.json stays checkable without importing anything
# (see tests/test_registry.py).
BENCHMARKS = {
    'maxsat60': {
        'dimensions': 60,
        'type': 'purely_binary',
        'filename': 'frb10-6-4.wcnf',
        'normalize_weights': True,
        'negative_weights': False,
        'data_loader': download_maxsat60_data,
    },
    'maxsat125': {
        'dimensions': 125,
        'type': 'purely_binary',
        'filename': 'cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf',
        'normalize_weights': False,
        'negative_weights': True,
        'data_loader': download_maxsat125_data,
    },
}

lock = threading.Lock()


def eval(
        x: np.ndarray,
        weights: np.ndarray,
        total_weight: float,
        clauseidxs: np.ndarray,
        clauses: np.ndarray,
        negative_weights: bool

) -> float:
    """
    Evaluate the function with the given input.

    :param x: Input array.
    :type x: np.ndarray
    :return: The evaluated result.
    :rtype: float
    """
    x = x.squeeze()
    assert x.ndim == 1
    weights_sum = np.sum(
        weights
        * [
            np.any(np.equal(x[ci], clauses[i, ci]))
            for i, ci in enumerate(clauseidxs)
        ]
    )
    if negative_weights:
        # weights of unsatisfied clauses
        weight_diff = total_weight - weights_sum
        fx = weight_diff
    else:
        fx = -weights_sum
    return fx


class MaxSATServiceServicer(DualStackGRCPService):
    """
    MaxSATServiceServicer class for maximum satisfiability problem service.

    This class provides methods for evaluating and solving maximum satisfiability problems.

    """

    def __init__(
            self,
            port: int = 50055,
            listen_hosts=None
    ):
        super().__init__(port=port, listen_hosts=listen_hosts)

    @lru_cache(maxsize=2)
    def get_wcnf_weights_totalweight_clauseidxs_clauses(
            self,
            benchmark: str
    ) -> (np.ndarray, float, np.ndarray, np.ndarray):
        """
        :param benchmark: The name of the benchmark to retrieve the data for.
        :return: A tuple containing four objects:
            - weights: An array of weights for each variable in the benchmark.
            - total_weight: The sum of all the weights.
            - clause_idxs: An array of indices indicating which variables are present in each clause.
            - clauses: A matrix representing the clauses where each row corresponds to a clause and each column corresponds to a variable.

        """
        assert benchmark in BENCHMARKS, "Invalid benchmark name"
        spec = BENCHMARKS[benchmark]
        fname = spec['filename']
        dataloader = spec['data_loader']
        # download data if not present
        with lock:
            dataloader(directory_name)

        wcnf = WCNF(
            os.path.join(
                directory_name, fname
            )
        )
        dim = wcnf.nv

        normalize_weights = spec['normalize_weights']

        weights = np.array(wcnf.weights, dtype=np.float64)
        total_weight = weights.sum()

        if normalize_weights:
            weights = (weights - weights.mean()) / weights.std()

        clauses = np.zeros((len(wcnf.clauses), dim), dtype=np.bool_)

        clause_idxs = []

        for i, clause in enumerate(wcnf.clauses):
            _clause_idxs = np.abs(np.array(clause)) - 1
            clauses[i, _clause_idxs] = np.array(clause) > 0
            clause_idxs.append(_clause_idxs)

        return weights, total_weight, clause_idxs, clauses

    def evaluate_point(
            self,
            request: BenchmarkRequest,
            context
    ) -> EvaluationResult:
        """
        :param request: Instance of the BenchmarkRequest class, containing the benchmark name and point values.
        :param context: The context in which the evaluation is being performed.
        :return: Instance of the EvaluationResult class, containing the evaluated value.
        """
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
        """Evaluate a binary assignment against a MaxSAT instance.

        Split out of evaluate_point so the binary check and weighting are
        testable without a gRPC server. MaxSAT is deterministic, so `seed` is
        unused.
        """
        if name not in BENCHMARKS:
            raise ValueError(
                f"Invalid benchmark name {name!r}; this service serves {sorted(BENCHMARKS)}")
        # check that x is binary
        assert np.all(np.logical_or(x == 0, x == 1)), "Input must be binary"

        weights, total_weight, clauseidxs, clauses = \
            self.get_wcnf_weights_totalweight_clauseidxs_clauses(name)
        return eval(x, weights, total_weight, clauseidxs, clauses,
                    BENCHMARKS[name]['negative_weights'])


def serve():
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=int(os.environ.get('BENCHER_MAXSAT_PORT', 50055)),
        help='The port number to start the server on. Default is 50055. '
             'Can also be set via the BENCHER_MAXSAT_PORT environment variable.',
    )
    add_listen_argument(parser, env_var=LISTEN_HOST_ENV_VAR)
    args = parser.parse_args()

    logging.basicConfig()
    maxsat = MaxSATServiceServicer(port=args.port, listen_hosts=resolve_listen_entries(args.listen_hosts, env_var=LISTEN_HOST_ENV_VAR))
    maxsat.serve()


if __name__ == '__main__':
    serve()
