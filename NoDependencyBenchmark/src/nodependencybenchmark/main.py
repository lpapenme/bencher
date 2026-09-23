import logging
import numpy as np
import os
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from platform import machine

from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult, ObjectiveValue
from bencherscaffold.dual_stack_service import DualStackGRCPService, add_listen_argument, resolve_listen_entries

LISTEN_HOST_ENV_VAR = 'BENCHER_NODEP_HOST'
_env_directory = os.environ.get("MOPTA_DATA_DIR")
if _env_directory:
    os.makedirs(_env_directory, exist_ok=True)
    directory_file_descriptor = None
    directory_name = _env_directory
else:
    directory_file_descriptor = tempfile.TemporaryDirectory()
    directory_name = directory_file_descriptor.name

# The benchmarks this service serves, as data. One declarative mapping per
# service keeps benchmark-registry.json checkable without importing anything
# (see tests/test_registry.py).
BENCHMARKS = {
    'mopta08': {'dimensions': 124, 'type': 'purely_continuous'},
    'pestcontrol': {'dimensions': 25, 'type': 'purely_categorical'},
}

SUPPORTED_BENCHMARKS = list(BENCHMARKS)


# Source: https://github.com/aryandeshwal/BODi/blob/main/bodi/pestcontrol.py

def _pest_spread(
        curr_pest_frac,
        spread_rate,
        control_rate,
        apply_control
):
    if apply_control:
        next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
        next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac


def _pest_control_score(
        x: np.ndarray,
        seed=None
):
    U = 0.1
    n_stages = x.size
    n_simulations = 100

    init_pest_frac_alpha = 1.0
    init_pest_frac_beta = 30.0
    spread_alpha = 1.0
    spread_beta = 17.0 / 3.0

    control_alpha = 1.0
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    # below two changes over stages according to x
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0
    above_threshold = 0

    if seed is not None:
        init_pest_frac = np.random.RandomState(seed).beta(
            init_pest_frac_alpha,
            init_pest_frac_beta,
            size=(n_simulations,)
        )
    else:
        init_pest_frac = np.random.beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
        if seed is not None:
            spread_rate = np.random.RandomState(seed).beta(spread_alpha, spread_beta, size=(n_simulations,))
        else:
            spread_rate = np.random.beta(spread_alpha, spread_beta, size=(n_simulations,))
        do_control = x[i] > 0
        if do_control:
            if seed is not None:
                control_rate = np.random.RandomState(seed).beta(
                    control_alpha,
                    control_beta[x[i]],
                    size=(n_simulations,)
                )
            else:
                control_rate = np.random.beta(control_alpha, control_beta[x[i]], size=(n_simulations,))
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, control_rate, True)
            # torelance has been developed for pesticide type 1
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            # you will get discount
            payed_price = control_price[x[i]] * (
                    1.0 - control_price_max_discount[x[i]] / float(n_stages) * float(np.sum(x == x[i])))
        else:
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, 0, False)
            payed_price = 0
        payed_price_sum += payed_price
        above_threshold += np.mean(curr_pest_frac > U)
        curr_pest_frac = next_pest_frac

    return payed_price_sum + above_threshold


def download_mopta_executable(
        executable_name: str,
):
    """
    Download MOPTA Executable

    :param executable_name: The name of the executable file to be downloaded.
    :return: None

    This method downloads the specified MOPTA executable file from a remote server. If the executable file does not exist in the specified directory, it will be downloaded and saved there
    *. The file will be downloaded using the provided `executable_name` and stored in the `directory_name` directory.

    Example usage:
        download_mopta_executable("mopta.exe")

    This will download the executable file "mopta.exe" and save it in the current working directory.
    """

    if not os.path.exists(os.path.join(directory_name, executable_name)):
        print(f"{executable_name} not found. Downloading...")
        url = f"http://mopta-executables.s3-website.eu-north-1.amazonaws.com/{executable_name}"
        print(f"Downloading {url}")

        import requests
        response = requests.get(url, verify=False)

        with open(os.path.join(directory_name, executable_name), "wb") as file:
            file.write(response.content)
        # make executable
        os.chmod(os.path.join(directory_name, executable_name), 0o755)
        print(f"Downloaded {executable_name}")


# Which MOPTA binary a given architecture needs. Kept as data so the lookup can
# be done without constructing a servicer (the Docker prefetch does exactly that).
_MOPTA_EXECUTABLES = {
    ("armv7l", 32): "mopta08_armhf.bin",
    ("x86_64", 64): "mopta08_elf64.bin",
    ("i386", 32): "mopta08_elf32.bin",
    ("amd64", 64): "mopta08_amd64.exe",
}


def mopta_executable_basename() -> str:
    """Name of the MOPTA binary for the running architecture.

    Raises only when MOPTA is actually needed, so architectures without a binary
    can still serve the benchmarks that do not use one.
    """
    sysarch = 64 if sys.maxsize > 2 ** 32 else 32
    key = (machine().lower(), sysarch)
    if key not in _MOPTA_EXECUTABLES:
        raise RuntimeError(
            f"mopta08 has no executable for architecture {key[0]!r} ({key[1]}-bit); "
            f"supported: {sorted({m for m, _ in _MOPTA_EXECUTABLES})}"
        )
    return _MOPTA_EXECUTABLES[key]


class NoDependencyServiceServicer(DualStackGRCPService):

    def __init__(
            self,
            port: int = 50054,
            listen_hosts=None
    ):
        super().__init__(port=port, n_cores=1, listen_hosts=listen_hosts)

        self.sysarch = 64 if sys.maxsize > 2 ** 32 else 32
        self.machine = machine().lower()

        # The MOPTA binary is x86-only, but pestcontrol is pure numpy. Resolving
        # the architecture here used to raise on anything else -- including Apple
        # Silicon -- so the servicer could not even be constructed for local
        # testing. It is resolved lazily now, when mopta08 is actually requested.
        self.directory_file_descriptor = tempfile.TemporaryDirectory()
        self.directory_name = self.directory_file_descriptor.name

    @property
    def _mopta_exectutable_basename(self) -> str:
        return mopta_executable_basename()

    @property
    def _mopta_exectutable(self) -> str:
        return os.path.join(directory_name, self._mopta_exectutable_basename)

    def evaluate_point(
            self,
            request: BenchmarkRequest,
            context
    ) -> EvaluationResult:
        """

        .. function:: evaluate_point(self, request: BenchmarkRequest, context) -> EvaluationResult

            Evaluate the given point against the specified benchmark.

            :param request: The benchmark request object containing the point to evaluate.
            :type request: BenchmarkRequest
            :param context: The evaluation context.
            :type context: Any
            :return: The evaluation result.
            :rtype: EvaluationResult

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
        """Evaluate a point in the benchmark's native domain.

        Split out of evaluate_point so normalisation and dispatch can be tested
        without standing up a gRPC server. `seed` is accepted for the benchmarks
        that are stochastic; it is threaded from BenchmarkRequest.random_seed.
        """
        if name not in BENCHMARKS:
            raise ValueError(
                f"Invalid benchmark name {name!r}; this service serves {SUPPORTED_BENCHMARKS}")

        if name == "mopta08":
            # mopta is in [0, 1]^n so we don't need to scale
            download_mopta_executable(self._mopta_exectutable_basename)
            return self.eval_mopta08(x)
        return _pest_control_score(x, seed=seed)

    def eval_mopta08(
            self,
            x: np.ndarray
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
        # write input to file in dir
        with open(os.path.join(self.directory_name, "input.txt"), "w+") as tmp_file:
            for _x in x:
                tmp_file.write(f"{_x}\n")
        # pass directory as working directory to process
        popen = subprocess.Popen(
            self._mopta_exectutable,
            stdout=subprocess.PIPE,
            cwd=self.directory_name,
        )
        popen.wait()
        # read and parse output file
        output = (
            open(os.path.join(self.directory_name, "output.txt"), "r")
            .read()
            .split("\n")
        )
        output = [x.strip() for x in output]
        output = np.array([float(x) for x in output if len(x) > 0])
        value = output[0]
        constraints = output[1:]
        # see https://arxiv.org/pdf/2103.00349.pdf E.7
        return float(value + 10 * np.sum(np.clip(constraints, a_min=0, a_max=None)))


def serve():
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=int(os.environ.get('BENCHER_NODEP_PORT', 50054)),
        help='The port number to start the server on. Default is 50054. '
             'Can also be set via the BENCHER_NODEP_PORT environment variable.',
    )
    add_listen_argument(parser, env_var=LISTEN_HOST_ENV_VAR)
    args = parser.parse_args()

    logging.basicConfig()
    nodep = NoDependencyServiceServicer(port=args.port, listen_hosts=resolve_listen_entries(args.listen_hosts, env_var=LISTEN_HOST_ENV_VAR))
    nodep.serve()


if __name__ == '__main__':
    serve()
