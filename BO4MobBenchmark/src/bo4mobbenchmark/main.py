import glob
import logging
import os
import re
import shutil
import tempfile
from argparse import ArgumentParser
from importlib import resources

import pandas as pd
from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult, ObjectiveValue
from bencherscaffold.dual_stack_service import DualStackGRCPService, add_listen_argument, resolve_listen_entries
from bo4mob import single_od_run

LISTEN_HOST_ENV_VAR = 'BENCHER_BO4MOB_HOST'
# Benchmark names are templated: BASE-NAME_DATE_HOUR_EVAL-TYPE, e.g.
# 1ramp_221008_08-09_count. There are 420 of them (5 networks x 14 dates x 3
# hours x 2 metrics), so unlike the other services this one matches by regex
# rather than enumerating. The networks are the declarative part: `dimensions`
# is the number of OD pairs in that network's CSV template, and mirrors
# benchmark-registry.json.
NETWORKS = {
    '1ramp': {'dimensions': 3, 'type': 'purely_integer'},
    '2corridor': {'dimensions': 21, 'type': 'purely_integer'},
    '3junction': {'dimensions': 44, 'type': 'purely_integer'},
    '4smallRegion': {'dimensions': 151, 'type': 'purely_integer'},
    '5fullRegion': {'dimensions': 10100, 'type': 'purely_integer'},
}

# Everything after the network name: a yymmdd date, one of three hour windows,
# and the evaluation metric.
NAME_SUFFIX_PATTERN = r"_\d{6}_(06-07|08-09|17-18)_(count|speed)"

valid_benchmark_expressions = [network + NAME_SUFFIX_PATTERN for network in NETWORKS]


class BO4MOBServiceServicer(DualStackGRCPService):

    def __init__(
            self,
            port: int = 50060,
            listen_hosts=None
    ):
        super().__init__(port=port, n_cores=1, listen_hosts=listen_hosts)

    def evaluate_point(
            self,
            request: BenchmarkRequest,
            context
    ) -> EvaluationResult:
        x = [v.value for v in request.point.values]
        value = self.evaluate(request.benchmark.name, x)
        return EvaluationResult(
            objectives=[ObjectiveValue(name="f0", value=value)],
        )

    def evaluate(
            self,
            name: str,
            x,
            seed: int | None = None
    ) -> float:
        """Run one SUMO simulation and return its NRMSE.

        Split out of evaluate_point so name validation and the OD-template
        wiring are testable without a gRPC server.
        """
        if not any(re.fullmatch(expr, name) for expr in valid_benchmark_expressions):
            raise ValueError(f"Invalid benchmark name: {name}")
        request_name = name
        print(f"Received point with {len(x)} values for benchmark {name} and contents: {x}")
        # we have "template" csv files od_1ramp.csv, od_2corridor.csv, ...in csv_templates folder
        csv_filename = f"od_{request_name.split('_')[0]}.csv"
        package_root = resources.files("bo4mobbenchmark")
        template_csv_path = package_root / "csv_templates" / csv_filename
        # replace the values in the "flow" column of the template csv with the values from x, use pandas
        df = pd.read_csv(template_csv_path)
        assert len(x) == len(df), f"Length of x ({len(x)}) does not match number of OD pairs ({len(df)})"
        df["flow"] = x

        benchmark_date = request_name.split("_")[1]
        benchmark_hour = request_name.split("_")[2]
        benchmark_eval_type = request_name.split("_")[3]

        # save to TemporaryFile
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_csv_path = f"{tmpdir}/od_input.csv"
            df.to_csv(temp_csv_path, index=False)
            single_od_run.run_single_simulation(
                network_name=request_name.split("_")[0],
                date=benchmark_date,
                hour=benchmark_hour,
                eval_measure=benchmark_eval_type,
                routes_per_od="single",
                od_csv=temp_csv_path,
            )
            # now there's a NMRSE_{nrmse_val}.txt file in output/single_od_run/network_1ramp_221014_08-09_count_multiple_od_1ramp_values/result
            # just get the file via wildcards and read the value
            try:
                nrmse_dir = glob.glob("output/*/*/result")
                nrmse_file = glob.glob(f"{nrmse_dir[0]}/NRMSE_*.txt")[0]
                # just get nrmse value via the filename
                nrmse_value = float(re.search(r"NRMSE_(\d+\.\d+).txt", nrmse_file).group(1))
            finally:
                # remove everything within the output folder but not the output folder itself
                output_dir = "output"
                for item in os.listdir(output_dir):
                    item_path = os.path.join(output_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
        return nrmse_value


def serve():
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=int(os.environ.get('BENCHER_BO4MOB_PORT', 50060)),
        help='The port number to start the server on. Default is 50060. '
             'Can also be set via the BENCHER_BO4MOB_PORT environment variable.',
    )
    add_listen_argument(parser, env_var=LISTEN_HOST_ENV_VAR)
    args = parser.parse_args()

    logging.basicConfig()
    bo4mob = BO4MOBServiceServicer(port=args.port, listen_hosts=resolve_listen_entries(args.listen_hosts, env_var=LISTEN_HOST_ENV_VAR))
    bo4mob.serve()


if __name__ == '__main__':
    serve()
