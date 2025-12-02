import threading

import logging
import numpy as np
import os
import tempfile
from functools import lru_cache

from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult
from bencherscaffold.dual_stack_service import DualStackGRCPService

from maxsatbenchmarks.data_loading import download_maxsat60_data, download_maxsat125_data
from maxsatbenchmarks.wcnf import WCNF

directory_file_descriptor = tempfile.TemporaryDirectory()
directory_name = directory_file_descriptor.name

filename_map = {
    'maxsat60' : 'frb10-6-4.wcnf',
    'maxsat125': 'cluster-expansion-IS1_5.0.5.0.0.5_softer_periodic.wcnf'
}

normalize_weights_map = {
    'maxsat60' : True,
    'maxsat125': False
}

negative_weights_map = {
    'maxsat60' : False,
    'maxsat125': True
}

data_loader_map = {
    'maxsat60' : download_maxsat60_data,
    'maxsat125': download_maxsat125_data
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
            self
    ):
        super().__init__(port=50055)

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
        assert benchmark in filename_map.keys(), "Invalid benchmark name"
        fname = filename_map[benchmark]
        dataloader = data_loader_map[benchmark]
        # download data if not present
        with lock:
            dataloader(directory_name)

        wcnf = WCNF(
            os.path.join(
                directory_name, fname
            )
        )
        dim = wcnf.nv

        normalize_weights = normalize_weights_map[benchmark]

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
        assert request.benchmark.name in filename_map.keys(), "Invalid benchmark name"
        x = [v.value for v in request.point.values]
        x = np.array(x)
        # check that x is binary
        assert np.all(np.logical_or(x == 0, x == 1)), "Input must be binary"

        weights, total_weight, clauseidxs, clauses = self.get_wcnf_weights_totalweight_clauseidxs_clauses(
            request.benchmark.name
        )

        negative_weights = negative_weights_map[request.benchmark.name]

        result = EvaluationResult(
            value=eval(x, weights, total_weight, clauseidxs, clauses, negative_weights)
        )
        return result


def serve():
    logging.basicConfig()
    maxsat = MaxSATServiceServicer()
    maxsat.serve()


if __name__ == '__main__':
    serve()
