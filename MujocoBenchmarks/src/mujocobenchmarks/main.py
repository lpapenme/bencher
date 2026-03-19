import logging
import os
from argparse import ArgumentParser

import numpy as np
from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult
from bencherscaffold.dual_stack_service import DualStackGRCPService, add_listen_argument, resolve_listen_entries
from gym.envs.box2d import LunarLander

from mujocobenchmarks.functions import func_factories

LISTEN_HOST_ENV_VAR = 'BENCHER_MUJOCO_HOST'
func_factory_map = {
    'mujoco-ant': lambda
        _: func_factories["ant"].make_object(),
    'mujoco-hopper': lambda
        _: func_factories["hopper"].make_object(),
    'mujoco-walker': lambda
        _: func_factories["walker_2d"].make_object(),
    'mujoco-halfcheetah': lambda
        _: func_factories["half_cheetah"].make_object(),
    'mujoco-swimmer': lambda
        _: func_factories["swimmer"].make_object(),
    'mujoco-humanoid': lambda
        _: func_factories["humanoid"].make_object(),
}

benchmark_bounds = {
    'mujoco-ant': (-1, 1),
    'mujoco-hopper': (-1.4, 1.4),
    'mujoco-walker': (-1.8, 0.9),
    'mujoco-halfcheetah': (-1, 1),
    'mujoco-swimmer': (-1, 1),
    'mujoco-humanoid': (-1, 1),
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
        x = [v.value for v in request.point.values]
        x = np.array(x).reshape(1, -1)
        if request.benchmark.name in func_factory_map.keys():
            # x is in [0, 1] space, we need to map it to the benchmark space
            lb, ub = benchmark_bounds[request.benchmark.name]
            x = lb + (ub - lb) * x
            func_factory = func_factory_map[request.benchmark.name](None)
            result = EvaluationResult(
                value=-float(func_factory(x)[0].squeeze()),
            )
        elif request.benchmark.name == 'lunarlander':
            env = LunarLander()
            total_reward = 0
            steps = 0
            s = env.reset()
            while True:
                a = heuristic_controller(s, x.squeeze())
                s, r, terminated, _ = env.step(a)
                total_reward += r

                steps += 1
                if terminated:
                    break
            result = EvaluationResult(
                value=total_reward
            )
        else:
            raise ValueError("Invalid benchmark name")

        return result


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
