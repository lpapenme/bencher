import logging
import os
import shutil
import tempfile
import threading
from argparse import ArgumentParser

import numpy as np
from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult
from bencherscaffold.dual_stack_service import DualStackGRCPService
from sob import get_problem

# benchmark name -> (MECHBench model_type, objective metric).
# Each objective is the problem's canonical (paper-default) target: scalar,
# FEM-based, minimization, and permitted for that problem (the metric is
# validated against the model's forbidden_output_data upstream).
#   1 = Star Box, 2 = Three-Point Bending, 3 = Crash Tube
benchmark_map = {
    'mechbench-starbox': (1, 'penalized_sea'),
    'mechbench-bending': (2, 'penalized_mass'),
    'mechbench-crashtube': (3, 'load_uniformity'),
}

# Path to a local OpenRadioss install (set in the container via the Dockerfile).
# If unset, MECHBench tries to download the solver into the cwd on first use,
# which only works on Linux/Windows -- so real evaluations are container-only.
OPENRADIOSS_PATH = os.environ.get('BENCHER_MECHBENCH_OPENRADIOSS_PATH')


class MECHServiceServicer(DualStackGRCPService):

    def __init__(
            self,
            port: int = 50061
    ):
        super().__init__(port=port, n_cores=1)
        # A MECHBench evaluation chdir()s into a per-deck working directory and
        # writes simulation files there. os.chdir is process-global, so we
        # serialise calls with a lock, give each its own temp root_folder, and
        # restore the cwd afterwards.
        self._lock = threading.Lock()
        self._deck_counter = 0
        self._home = os.getcwd()

    def evaluate_point(
            self,
            request: BenchmarkRequest,
            context
    ) -> EvaluationResult:
        assert request.benchmark.name in benchmark_map.keys(), "Invalid benchmark name"
        model_type, metric = benchmark_map[request.benchmark.name]

        x = np.array([v.value for v in request.point.values])
        # MECHBench's search space is [-5, 5] for every design variable, while
        # the client sends x in [0, 1]; rescale. The dimension is taken from the
        # length of the point (StarBox 1-34, Bending 1-40, CrashTube ~2-30).
        x_native = (10.0 * x - 5.0).tolist()

        with self._lock:
            self._deck_counter += 1
            run_dir = tempfile.mkdtemp(prefix='mechbench_')
            try:
                problem = get_problem(
                    model_type,
                    len(x_native),
                    output_data=metric,
                    runner_options={
                        'open_radioss_main_path': OPENRADIOSS_PATH,
                        'np': 1,
                        'nt': 1,
                        'h_level': 1,
                        'gmsh_verbosity': 0,
                        'write_vtk': 0,
                    },
                    root_folder=run_dir,
                )
                value = float(problem(x_native, deck_id=self._deck_counter))
            finally:
                # MECHBench leaves the cwd inside run_dir; restore it before
                # removing the directory so the next call starts somewhere valid.
                os.chdir(self._home)
                shutil.rmtree(run_dir, ignore_errors=True)

        return EvaluationResult(value=value)


def serve():
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=int(os.environ.get('BENCHER_MECHBENCH_PORT', 50061)),
        help='The port number to start the server on. Default is 50061. '
             'Can also be set via the BENCHER_MECHBENCH_PORT environment variable.',
    )
    args = parser.parse_args()

    logging.basicConfig()
    mech = MECHServiceServicer(port=args.port)
    mech.serve()


if __name__ == '__main__':
    serve()
