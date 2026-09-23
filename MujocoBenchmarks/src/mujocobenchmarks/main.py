# This package pins Python 3.8, where PEP 604 annotations (`int | None`)
# raise TypeError at definition time. Deferring annotation evaluation keeps
# the modern syntax working here.
from __future__ import annotations

import logging
import os
from argparse import ArgumentParser
from typing import Optional

import gym
import numpy as np
from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult, ObjectiveValue
from bencherscaffold.dual_stack_service import DualStackGRCPService, add_listen_argument, resolve_listen_entries

from mujocobenchmarks.functions import func_factories

LISTEN_HOST_ENV_VAR = 'BENCHER_MUJOCO_HOST'
# The benchmarks this service serves, as data. `bounds` is the native domain a
# point in [0, 1]^d is mapped onto; `factory` builds the MuJoCo rollout. The
# lunarlander entry has no factory -- it is a gym environment, handled below --
# but it belongs in the same mapping so the registry stays checkable.
# Dimensions and type mirror benchmark-registry.json.
BENCHMARKS = {
    'mujoco-ant': {
        'dimensions': 888, 'type': 'purely_continuous', 'bounds': (-1, 1),
        'factory': lambda: func_factories["ant"].make_object()},
    'mujoco-hopper': {
        'dimensions': 33, 'type': 'purely_continuous', 'bounds': (-1.4, 1.4),
        'factory': lambda: func_factories["hopper"].make_object()},
    'mujoco-walker': {
        'dimensions': 102, 'type': 'purely_continuous', 'bounds': (-1.8, 0.9),
        'factory': lambda: func_factories["walker_2d"].make_object()},
    'mujoco-halfcheetah': {
        'dimensions': 102, 'type': 'purely_continuous', 'bounds': (-1, 1),
        'factory': lambda: func_factories["half_cheetah"].make_object()},
    'mujoco-swimmer': {
        'dimensions': 16, 'type': 'purely_continuous', 'bounds': (-1, 1),
        'factory': lambda: func_factories["swimmer"].make_object()},
    'mujoco-humanoid': {
        'dimensions': 6392, 'type': 'purely_continuous', 'bounds': (-1, 1),
        'factory': lambda: func_factories["humanoid"].make_object()},
    'lunarlander': {
        'dimensions': 12, 'type': 'purely_continuous', 'bounds': None,
        'factory': None},
}


def heuristic_controller(
        state: np.ndarray,
        x: np.ndarray
) -> int:
    angle_targ = state[0] * x[0] + state[2] * x[1]
    if angle_targ > x[2]:
        angle_targ = x[2]
    if angle_targ < -x[2]:
        angle_targ = -x[2]
    hover_targ = x[3] * np.abs(state[0])

    angle_todo = (angle_targ - state[4]) * x[4] - (state[5]) * x[5]
    hover_todo = (hover_targ - state[1]) * x[6] - (state[3]) * x[7]

    if state[6] or state[7]:
        angle_todo = x[8]
        hover_todo = -(state[3]) * x[9]

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > x[10]:
        a = 2
    elif angle_todo < -x[11]:
        a = 3
    elif angle_todo > +x[11]:
        a = 1
    return a


class MujocoServiceServicer(DualStackGRCPService):

    def __init__(
            self,
            port: int = 50057,
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
            seed: Optional[int] = None
    ) -> float:
        """Evaluate a point given in [0, 1]^d; returns a cost (negated reward).

        Split out of evaluate_point so the per-benchmark rescaling is testable
        without standing up a gRPC server. Both the MuJoCo rollouts and the gym
        LunarLander episode are stochastic; `seed` is threaded through for
        reproducibility once the services accept BenchmarkRequest.random_seed.
        """
        if name not in BENCHMARKS:
            raise ValueError("Invalid benchmark name")
        spec = BENCHMARKS[name]
        x = np.array(x).reshape(1, -1)

        if spec['factory'] is not None:
            # x is in [0, 1] space, we need to map it to the benchmark space
            lb, ub = spec['bounds']
            x = lb + (ub - lb) * x
            return -float(spec['factory']()(x)[0].squeeze())

        return -self._lunarlander_reward(x.squeeze(), seed=seed)

    @staticmethod
    def _lunarlander_reward(weights: np.ndarray, seed: Optional[int] = None) -> float:
        """Total reward of one LunarLander episode under a heuristic controller."""
        env = gym.make("LunarLander-v2")
        try:
            total_reward = 0
            s = env.reset() if seed is None else env.reset(seed=seed)
            while True:
                a = heuristic_controller(s, weights)
                s, r, terminated, _ = env.step(a)
                total_reward += r
                if terminated:
                    break
        finally:
            env.close()
        return total_reward


def serve():
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=int(os.environ.get('BENCHER_MUJOCO_PORT', 50057)),
        help='The port number to start the server on. Default is 50057. '
             'Can also be set via the BENCHER_MUJOCO_PORT environment variable.',
    )
    add_listen_argument(parser, env_var=LISTEN_HOST_ENV_VAR)
    args = parser.parse_args()

    logging.basicConfig()
    mujoco = MujocoServiceServicer(port=args.port, listen_hosts=resolve_listen_entries(args.listen_hosts, env_var=LISTEN_HOST_ENV_VAR))
    mujoco.serve()


if __name__ == '__main__':
    serve()
