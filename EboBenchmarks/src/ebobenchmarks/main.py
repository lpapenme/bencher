# This package pins Python 3.8, where PEP 604 annotations (`int | None`)
# raise TypeError at definition time. Deferring annotation evaluation keeps
# the modern syntax working here.
from __future__ import annotations

import logging
import os
from argparse import ArgumentParser

import numpy as np
from bencherscaffold.protoclasses.bencher_pb2 import EvaluationResult, ObjectiveValue, BenchmarkRequest
from bencherscaffold.dual_stack_service import DualStackGRCPService, add_listen_argument, resolve_listen_entries
from ebo.test_functions.push_function import PushReward
from ebo.test_functions.rover_function import create_large_domain
from ebo.test_functions.rover_utils import RoverDomain

LISTEN_HOST_ENV_VAR = 'BENCHER_EBO_HOST'

def eval_lasso(
        x: np.ndarray,
        benchmark
):
    return benchmark.evaluate(x)


# The benchmarks this service serves, as data. Dimensions and type mirror
# benchmark-registry.json, which tests/test_registry.py cross-checks.
BENCHMARKS = {
    'robotpushing': {'dimensions': 14, 'type': 'purely_continuous'},
    'rover': {'dimensions': 60, 'type': 'purely_continuous'},
}


def _l2cost(x, point):
    return 10 * np.linalg.norm(x - point, 1)


class EboServiceServicer(DualStackGRCPService):

    def __init__(
            self,
            port: int = 50056,
            listen_hosts=None
    ):
        super().__init__(port=port, n_cores=1, listen_hosts=listen_hosts)
        # Both simulators used to be constructed here, which made the servicer
        # impossible to build -- and therefore to test -- without the full `ebo`
        # package importable and working. They are built on first use instead.
        self._push_reward = None
        self._rover_domain = None

    @property
    def push_reward(self) -> PushReward:
        if self._push_reward is None:
            self._push_reward = PushReward()
        return self._push_reward

    @property
    def rover_domain(self) -> RoverDomain:
        if self._rover_domain is None:
            self._rover_domain = create_large_domain(
                force_start=False,
                force_goal=False,
                start_miss_cost=_l2cost,
                goal_miss_cost=_l2cost,
            )
        return self._rover_domain

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
        """Evaluate a point given in [0, 1]^d; returns a cost (negated reward).

        Split out of evaluate_point so the rescaling and shape checks are
        testable without a gRPC server. Both simulators are stochastic; `seed`
        is threaded through for reproducibility once the services accept
        BenchmarkRequest.random_seed.
        """
        if name not in BENCHMARKS:
            raise ValueError(
                f"Invalid benchmark name {name!r}; this service serves {sorted(BENCHMARKS)}")

        x = np.array(x).squeeze()
        expected = BENCHMARKS[name]['dimensions']
        assert x.shape[0] == expected, (
            f"{name} expects {expected} dimensions, got {x.shape[0]}")

        if name == 'robotpushing':
            lb = np.array(self.push_reward.xmin)
            ub = np.array(self.push_reward.xmax)
            # x is in [0, 1] space, so we need to scale it to the domain
            x = lb + (ub - lb) * x
            return -self.push_reward(x)

        # bounds are [0, 1] for the rover, so we don't need to scale
        return -self.rover_domain(x)


def serve():
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=int(os.environ.get('BENCHER_EBO_PORT', 50056)),
        help='The port number to start the server on. Default is 50056. '
             'Can also be set via the BENCHER_EBO_PORT environment variable.',
    )
    add_listen_argument(parser, env_var=LISTEN_HOST_ENV_VAR)
    args = parser.parse_args()

    logging.basicConfig()
    ebo = EboServiceServicer(port=args.port, listen_hosts=resolve_listen_entries(args.listen_hosts, env_var=LISTEN_HOST_ENV_VAR))
    ebo.serve()


if __name__ == '__main__':
    serve()
