import logging
import numpy as np
import os
import subprocess
import sys
import tempfile
from platform import machine

from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult
from bencherscaffold.protoclasses.grcp_service import GRCPService

directory_file_descriptor = tempfile.TemporaryDirectory()
directory_name = directory_file_descriptor.name

SUPPORTED_BENCHMARKS = [
    'mopta08',
    'pestcontrol'
]


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


class NoDependencyServiceServicer(GRCPService):

    def __init__(
            self
    ):
        super().__init__(port=50054, n_cores=1)

        self.sysarch = 64 if sys.maxsize > 2 ** 32 else 32
        self.machine = machine().lower()

        if self.machine == "armv7l":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable_basename = "mopta08_armhf.bin"
        elif self.machine == "x86_64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable_basename = "mopta08_elf64.bin"
        elif self.machine == "i386":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable_basename = "mopta08_elf32.bin"
        elif self.machine == "amd64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable_basename = "mopta08_amd64.exe"
        else:
            raise RuntimeError("Machine with this architecture is not supported")

        self._mopta_exectutable = os.path.join(
            directory_name, self._mopta_exectutable_basename
        )

        self.directory_file_descriptor = tempfile.TemporaryDirectory()
        self.directory_name = self.directory_file_descriptor.name

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
        assert request.benchmark.name in SUPPORTED_BENCHMARKS, "Invalid benchmark name"

        x = [v.value for v in request.point.values]
        x = np.array(x)

        match request.benchmark.name:
            case "mopta08":
                download_mopta_executable(self._mopta_exectutable_basename)
                # mopta is in [0, 1]^n so we don't need to scale
                fun = self.eval_mopta08
            case "pestcontrol":
                fun = _pest_control_score
            case _:
                raise ValueError("Invalid benchmark name")

        result = EvaluationResult(
            value=fun(x)
        )
        return result

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
    logging.basicConfig()
    nodep = NoDependencyServiceServicer()
    nodep.serve()


if __name__ == '__main__':
    serve()
