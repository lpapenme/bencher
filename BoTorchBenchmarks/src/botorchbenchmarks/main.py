import logging
import os
from argparse import ArgumentParser

import numpy as np
import torch
from bencherscaffold.dual_stack_service import (
    DualStackGRCPService,
    add_listen_argument,
    resolve_listen_entries,
)
from bencherscaffold.protoclasses.bencher_pb2 import (
    BenchmarkRequest,
    Constraint,
    ConstraintType,
    EvaluationResult,
    ObjectiveValue,
)
from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.test_functions.synthetic import PressureVessel
from botorch.test_functions.utils import round_nearest

LISTEN_HOST_ENV_VAR = "BENCHER_BOTORCH_HOST"


class _PressureVessel(PressureVessel):
    """botorch rounds x1, x2 to multiples of 0.0625 in the objective only.

    Without rounding them in the constraints too, a point can look feasible
    while the design the objective prices is not (f ~ 5872 < optimum ~ 6060).
    """

    def _evaluate_slack_true(self, X: torch.Tensor) -> torch.Tensor:
        X = X.clone()
        for i in (0, 1):
            X[..., i] = round_nearest(
                X[..., i], increment=0.0625, bounds=self._bounds[i]
            )
        return super()._evaluate_slack_true(X)


# The benchmarks this service serves, as data. `dimensions` and `type` mirror
# benchmark-registry.json (tests/test_registry.py checks they agree), and
# `factory` builds the underlying botorch test problem on demand.
BENCHMARKS = {
    "pressure-vessel": {
        "dimensions": 4,
        "type": "purely_continuous",
        "factory": lambda: _PressureVessel(),
    },
}

SUPPORTED_BENCHMARKS = list(BENCHMARKS)


class BoTorchServiceServicer(DualStackGRCPService):
    def __init__(self, port: int = 50061, listen_hosts=None):
        super().__init__(port=port, n_cores=1, listen_hosts=listen_hosts)

    def evaluate_point(self, request: BenchmarkRequest, context) -> EvaluationResult:
        x = np.array([v.value for v in request.point.values])
        seed = request.random_seed if request.HasField("random_seed") else None
        return self.evaluate(request.benchmark.name, x, seed=seed)

    def evaluate(
        self, name: str, x: np.ndarray, seed: int | None = None
    ) -> EvaluationResult:
        """Evaluate a point given in [0, 1]^d.

        Split out of evaluate_point so normalisation and dispatch can be tested
        without standing up a gRPC server. The botorch problems are evaluated
        noise-free, so `seed` is unused.

        Constrained problems fill `constraints`; inequality constraints are
        feasible iff value <= 0. Unconstrained problems leave the field empty.
        """
        if name not in BENCHMARKS:
            raise ValueError(
                f"Invalid benchmark name {name!r}; this service serves {SUPPORTED_BENCHMARKS}"
            )
        expected_shape = (BENCHMARKS[name]["dimensions"],)
        if x.shape != expected_shape:
            raise ValueError(
                f"Input shape {x.shape} does not match expected shape {expected_shape}"
            )

        problem = BENCHMARKS[name]["factory"]()
        # x is in [0, 1]^d, so we need to scale it to the problem's bounds
        lb, ub = problem.bounds
        X = (lb + torch.as_tensor(x, dtype=lb.dtype) * (ub - lb)).unsqueeze(0)

        constraints = []
        if isinstance(problem, ConstrainedBaseTestProblem):
            # botorch slack is feasible iff >= 0; bencher constraints are feasible iff <= 0.
            g = -problem.evaluate_slack_true(X)[0]
            constraints = [
                Constraint(name=f"g{i}", type=ConstraintType.INEQUALITY, value=float(v))
                for i, v in enumerate(g)
            ]

        return EvaluationResult(
            objectives=[
                ObjectiveValue(name="f0", value=float(problem.evaluate_true(X).item()))
            ],
            constraints=constraints,
        )


def serve():
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=int(os.environ.get("BENCHER_BOTORCH_PORT", 50061)),
        help="The port number to start the server on. Default is 50061. "
        "Can also be set via the BENCHER_BOTORCH_PORT environment variable.",
    )
    add_listen_argument(parser, env_var=LISTEN_HOST_ENV_VAR)
    args = parser.parse_args()

    logging.basicConfig()
    botorch_service = BoTorchServiceServicer(
        port=args.port,
        listen_hosts=resolve_listen_entries(
            args.listen_hosts, env_var=LISTEN_HOST_ENV_VAR
        ),
    )
    botorch_service.serve()


if __name__ == "__main__":
    serve()
