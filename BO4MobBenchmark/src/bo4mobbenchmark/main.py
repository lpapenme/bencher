import glob
import logging
import os
import re
import shutil
import tempfile
from importlib import resources

import pandas as pd
from bencherscaffold.protoclasses.bencher_pb2 import BenchmarkRequest, EvaluationResult
from bencherscaffold.protoclasses.grcp_service import GRCPService
from bo4mob import single_od_run

# our benchmarks will have names like 1ramp_221008_08-09_count where everything after 1ramp can vary
# well match the names via regex
valid_benchmark_expressions = [
    r"1ramp_\d{6}_(06-07|08-09|17-18)_(count|speed)",
    r"2corridor_\d{6}_(06-07|08-09|17-18)_(count|speed)",
    r"3junction_\d{6}_(06-07|08-09|17-18)_(count|speed)",
    r"4smallRegion_\d{6}_(06-07|08-09|17-18)_(count|speed)",
    r"5fullRegion_\d{6}_(06-07|08-09|17-18)_(count|speed)",
]


class BO4MOBServiceServicer(GRCPService):

    def __init__(
            self
    ):
        super().__init__(port=50060, n_cores=1)

    def evaluate_point(
            self,
            request: BenchmarkRequest,
            context
    ) -> EvaluationResult:
        assert any(re.fullmatch(expr, request.benchmark.name) for expr in valid_benchmark_expressions), \
            f"Invalid benchmark name: {request.benchmark.name}"
        x = [v.value for v in request.point.values]
        print(f"Received point with {len(x)} values for benchmark {request.benchmark.name} and contents: {x}")
        # we have "template" csv files od_1ramp.csv, od_2corridor.csv, ...in csv_templates folder
        csv_filename = f"od_{request.benchmark.name.split('_')[0]}.csv"
        package_root = resources.files("bo4mobbenchmark")
        template_csv_path = package_root / "csv_templates" / csv_filename
        # replace the values in the "flow" column of the template csv with the values from x, use pandas
        df = pd.read_csv(template_csv_path)
        assert len(x) == len(df), f"Length of x ({len(x)}) does not match number of OD pairs ({len(df)})"
        df["flow"] = x

        benchmark_date = request.benchmark.name.split("_")[1]
        benchmark_hour = request.benchmark.name.split("_")[2]
        benchmark_eval_type = request.benchmark.name.split("_")[3]

        # save to TemporaryFile
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_csv_path = f"{tmpdir}/od_input.csv"
            df.to_csv(temp_csv_path, index=False)
            single_od_run.run_single_simulation(
                network_name=request.benchmark.name.split("_")[0],
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
        result = EvaluationResult(
            value=nrmse_value
        )
        return result


def serve():
    logging.basicConfig()
    bo4mob = BO4MOBServiceServicer()
    bo4mob.serve()


if __name__ == '__main__':
    serve()
