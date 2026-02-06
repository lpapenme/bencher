<!-- Builds currently failing due to the container size exceeding the size of gh runners -->
<!-- # [![Docker Build](https://github.com/LeoIV/bencher/actions/workflows/docker_build.yml/badge.svg)](https://github.com/LeoIV/bencher/actions/workflows/docker_build.yml) -->

Bencher is a lightweight benchmarking framework for black-box optimization designed to make *benchmark execution* simple and reproducible, without forcing benchmark dependencies into your optimizer’s environment. It follows a client–server architecture: benchmarks run in an isolated, containerized server, while optimizers communicate with the server through a stable gRPC interface via a small Python client (`bencherscaffold`).

Each benchmark (or compatible benchmark group) is executed in its own dedicated Python environment, which avoids dependency conflicts and makes it easy to mix benchmarks with different (and sometimes outdated) requirements. The Bencher server can be deployed locally via Docker or on HPC systems via Singularity/Apptainer, enabling the same benchmark setup across machines and runs. This repository contains the Bencher server and the benchmark implementations

* See the paper for details: https://arxiv.org/abs/2505.21321
* See this blogpost for an example of how you can integrate `bencher` in your workflow: https://leonard.papenmeier.io/2026/02/05/adding-bencher-to-existing-code.html

# Docker Container

The Docker container can be pulled from the [Docker Hub](https://hub.docker.com/r/gaunab/bencher) or built locally.
It contains all benchmarks and dependencies and exposes the benchmark server via port 50051.

We give an exemplary usage of the Docker container in the [bencherclient](https://github.com/LeoIV/bencherclient)
repository.

```shell
pwd # /path/to/bencher
docker build -t bencher .
# always keep the container running, can be stopped with docker stop <container-id>
docker run -p 50051:50051 --restart always -d bencher:latest
```

**or**

```shell
docker pull gaunab/bencher:latest
# always keep the container running, can be stopped with docker stop <container-id>
docker run -p 50051:50051 --restart always -d gaunab/bencher:latest
```

# Apptainer / Singularity Container

You can build an Apptainer container from the Docker image:

```shell
Bootstrap: docker
From: gaunab/bencher:latest
Namespace:
Stage: build

%environment
    export LANG=C.UTF-8
    export PATH="/root/.local/bin:$PATH"

%post
    cd /opt
    git clone your-repo
    cd your-repo
    pip install bencherscaffold # you'll need bencherscaffold to call bencher
    pip install your-dependencies

%startscript
    bash -c "/docker-entrypoint.sh"

%runscript
    bash -c "your-command-to-run-your-app"
```

This will create an Apptainer container with the Docker image `gaunab/bencher:latest` and the repository `your-repo`
with the dependencies `your-dependencies` installed.

## Usage

### Starting the instance

```shell
apptainer build container.sif your-apptainer-file
```

### Start the Apptainer instance

This starts all the benchmarks in the container (as defined in the `startscript` of the Apptainer file).

```shell
apptainer instance start container.sif your-instance-name
```

### Run your command that depends on the benchmarks

This runs your command in the instance `your-instance-name` as defined in the `runscript` of the Apptainer file.

```shell
apptainer run instance://your-instance-name
```

### Evaluating a benchmark

We show how to run all benchmarks in the [`bencherclient`](https://github.com/LeoIV/bencherclient) repository.
You don't need to use this repository, it is mainly used to test the benchmarks.
The general setup to evaluate a benchmark is as follows.
First, install the [`bencherscaffold`](https://github.com/LeoIV/BencherScaffold) package:

```shell
pip install git+https://github.com/LeoIV/BencherScaffold
```

Then, you can use the following code to evaluate a benchmark:

```python
from bencherscaffold.client import BencherClient
from bencherscaffold.protoclasses.bencher_pb2 import Value, ValueType

# Create a client to communicate with the Bencher server
# By default, it connects to 127.0.0.1:50051
client = BencherClient()

# Create a list of values to evaluate
values = [Value(type=ValueType.CONTINUOUS, value=0.5) for _ in range(180)]
# The benchmark name is the name of the benchmark you want to evaluate
benchmark_name = 'lasso-dna'

# Evaluate the benchmark with the given values
# This will send the values to the server and return the result
# If the server is not running, it will raise an error
result = client.evaluate_point(
    benchmark_name=benchmark_name,
    point=values,
)
print(f"Result: {result}")
```

### Available Benchmarks

The urban mobility benchmarks (`1ramp_*`, `2corridor_*`, etc.) follow a templated naming convention: `BASE-NAME_DATE_HOUR_EVAL-TYPE` [^14].

-   **`BASE-NAME`**: Defines the traffic scenario (`1ramp`, `2corridor`, `3junction`, `4smallRegion`, `5fullRegion`).
-   **`DATE`**: A date in `yymmdd` format, from `221008` to `221021`.
-   **`HOUR`**: The time of day (`06-07`, `08-09`, or `17-18`).
-   **`EVAL-TYPE`**: The evaluation metric (`count` or `speed`).

For example, a valid benchmark name is `1ramp_221008_08-09_count`.

The following benchmarks are available:

| Benchmark Name             | # Dimensions | Type        | Source(s)      | Noisy    |
|----------------------------|--------------|-------------|----------------|----------|
| lasso-dna                  | 180          | continuous  | [^1],[^5]      | &#x2612; |
| lasso-simple               | 60           | continuous  | [^1]           | &#x2612; |
| lasso-medium               | 100          | continuous  | [^1]           | &#x2612; |
| lasso-high                 | 300          | continuous  | [^1],[^5]      | &#x2612; |
| lasso-hard                 | 1000         | continuous  | [^1],[^5]      | &#x2612; |
| lasso-leukemia             | 7129         | continuous  | [^1]           | &#x2612; |
| lasso-rcv1                 | 47236        | continuous  | [^1],[^2]      | &#x2612; |
| lasso-diabetes             | 8            | continuous  | [^1]           | &#x2612; |
| lasso-breastcancer         | 10           | continuous  | [^1]           | &#x2612; |
| mopta08                    | 124          | continuous  | [^4],[^5]      | &#x2612; |
| maxsat60                   | 60           | binary      | [^6],[^7]      | &#x2612; |
| maxsat125                  | 125          | binary      | [^7]           | &#x2612; |
| robotpushing               | 14           | continuous  | [^3]           | &#x2611; |
| lunarlander                | 12           | continuous  | [^3]           | &#x2611; |
| rover                      | 60           | continuous  | [^3]           | &#x2612; |
| mujoco-ant                 | 888          | continuous  | [^9],[^5]      | &#x2611; |
| mujoco-hopper              | 33           | continuous  | [^9],[^5]      | &#x2611; |
| mujoco-walker              | 102          | continuous  | [^9],[^5]      | &#x2611; |
| mujoco-halfcheetah         | 102          | continuous  | [^9],[^5]      | &#x2611; |
| mujoco-swimmer             | 16           | continuous  | [^9],[^5]      | &#x2611; |
| mujoco-humanoid            | 6392         | continuous  | [^9],[^5]      | &#x2611; |
| svm                        | 388          | continuous  | [^4],[^5],[^8] | &#x2612; |
| svmmixed                   | 53           | mixed       | [^6],[^7]      | &#x2612; |
| 1ramp_*                    | 3            | integer     | [^14]          | &#x2612; |
| 2corridor_*                | 21           | integer     | [^14]          | &#x2612; |
| 3junction_*                | 44           | integer     | [^14]          | &#x2612; |
| 4smallRegion_*             | 151          | integer     | [^14]          | &#x2612; |
| 5fullRegion_*              | 10100        | integer     | [^14]          | &#x2612; |
| pestcontrol                | 25           | categorical | [^10],[^13]    | &#x2612; |
| bbob-sphere                | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-ellipsoid             | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-rastrigin             | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-buecherastrigin       | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-linearslope           | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-attractivesector      | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-stepellipsoid         | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-rosenbrock            | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-rosenbrockrotated     | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-ellipsoidrotated      | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-discus                | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-bentcigar             | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-sharpridge            | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-differentpowers       | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-rastriginrotated      | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-weierstrass           | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-schaffers10           | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-schaffers1000         | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-griewankrosenbrock    | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-schwefel              | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-gallagher101          | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-gallagher21           | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-katsuura              | any          | continuous  | [^11],[^12]    | &#x2612; |
| bbob-lunacekbirastrigin    | any          | continuous  | [^11],[^12]    | &#x2612; |
| pbo-onemax                 | any          | binary      | [^11]          | &#x2612; |
| pbo-leadingones            | any          | binary      | [^11]          | &#x2612; |
| pbo-linear                 | any          | binary      | [^11]          | &#x2612; |
| pbo-onemaxdummy1           | any          | binary      | [^11]          | &#x2612; |
| pbo-onemaxdummy2           | any          | binary      | [^11]          | &#x2612; |
| pbo-onemaxneutrality       | any          | binary      | [^11]          | &#x2612; |
| pbo-onemaxepistasis        | any          | binary      | [^11]          | &#x2612; |
| pbo-onemaxruggedness1      | any          | binary      | [^11]          | &#x2612; |
| pbo-onemaxruggedness2      | any          | binary      | [^11]          | &#x2612; |
| pbo-onemaxruggedness3      | any          | binary      | [^11]          | &#x2612; |
| pbo-leadingonesdummy1      | any          | binary      | [^11]          | &#x2612; |
| pbo-leadingonesdummy2      | any          | binary      | [^11]          | &#x2612; |
| pbo-leadingonesneutrality  | any          | binary      | [^11]          | &#x2612; |
| pbo-leadingonesepistasis   | any          | binary      | [^11]          | &#x2612; |
| pbo-leadingonesruggedness1 | any          | binary      | [^11]          | &#x2612; |
| pbo-leadingonesruggedness2 | any          | binary      | [^11]          | &#x2612; |
| pbo-leadingonesruggedness3 | any          | binary      | [^11]          | &#x2612; |
| pbo-labs                   | any          | binary      | [^11]          | &#x2612; |
| pbo-isingring              | any          | binary      | [^11]          | &#x2612; |
| pbo-isingtorus             | any          | binary      | [^11]          | &#x2612; |
| pbo-isingtriangular        | any          | binary      | [^11]          | &#x2612; |
| pbo-mis                    | any          | binary      | [^11]          | &#x2612; |
| pbo-nqueens                | any          | binary      | [^11]          | &#x2612; |
| pbo-concatenatedtrap       | any          | binary      | [^11]          | &#x2612; |
| pbo-nklandscapes           | any          | binary      | [^11]          | &#x2612; |
| graph-maxcut2000           | 800          | binary      | [^11]          | &#x2612; |
| graph-maxcut2001           | 800          | binary      | [^11]          | &#x2612; |
| graph-maxcut2002           | 800          | binary      | [^11]          | &#x2612; |
| graph-maxcut2003           | 800          | binary      | [^11]          | &#x2612; |
| graph-maxcut2004           | 800          | binary      | [^11]          | &#x2612; |
| graph-maxcoverage2100      | 800          | binary      | [^11]          | &#x2612; |
| graph-maxcoverage2101      | 800          | binary      | [^11]          | &#x2612; |

# Citation
If you use this repository or the benchmarks in your research, please cite the following [paper](https://arxiv.org/abs/2505.21321):

```bibtex
@misc{papenmeier2025bencher,
      title={Bencher: Simple and Reproducible Benchmarking for Black-Box Optimization}, 
      author={Leonard Papenmeier and Luigi Nardi},
      year={2025},
      eprint={2505.21321},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.21321}, 
}
```

# Building on MacOS (under development)

`brew install swig gfortran openblas pkg-config glfw libomp`

To allow the build tools to find OpenBLAS, you must run:
`brew info openblas | grep PKG_CONFIG_PATH`

and set the `PKG_CONFIG_PATH` environment variable accordingly, e.g.:
`export PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig"`

## Mujoco

Download `https://github.com/google-deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-macos-universal2.dmg` and mount it.

Then, copy the dynamic library and headers to `~/.mujoco/mujoco210/`:

```
mkdir -p ~/.mujoco/mujoco210/bin
cp /Volumes/MuJoCo/MuJoCo.framework/Versions/Current/libmujoco.2.1.1.dylib ~/.mujoco/mujoco210/bin/
ln -sf ~/.mujoco/mujoco210/bin/libmujoco.2.1.1.dylib ~/.mujoco/mujoco210/bin/libmujoco.dylib
mkdir -p ~/.mujoco/mujoco210/bin/MuJoCo.framework/Versions/A/
ln -s ~/.mujoco/mujoco210/bin/libmujoco.2.1.1.dylib ~/.mujoco/mujoco210/bin/MuJoCo.framework/Versions/A/libmujoco.2.1.1.dylib
cp -r /Volumes/MuJoCo/MuJoCo.framework/Versions/Current/Headers ~/.mujoco/mujoco210/include
```

You probably have to allow access to the library in the Security & Privacy settings.


```
export CC=/opt/homebrew/opt/llvm/bin/clang 
```

## Toubleshooting

One main problem during the compilation occurs if you use a x86_64 Python on an ARM Mac.

[^1]: [`LassoBench`](https://github.com/ksehic/LassoBench) (`
Šehić Kenan, Gramfort Alexandre, Salmon Joseph and Nardi Luigi, "LassoBench: A High-Dimensional Hyperparameter Optimization Benchmark Suite for Lasso", AutoML conference, 2022.`)
[^2]: The LassoBench paper states 19,959 features, but the number of features in the RCV1 dataset is 47,236.
[^3]: [`TurBO`](https://github.com/uber-research/TuRBO) (
`David Eriksson, Michael Pearce, Jacob Gardner, Ryan D Turner and Matthias Poloczek, "Scalable Global Optimization via Local Bayesian Optimization." NeurIPS 2019`)
[^4]: [`SAASBO`](https://github.com/martinjankowiak/saasbo)
`David Eriksson and Martin Jankowiak, "High-dimensional Bayesian optimization with sparse axis-aligned subspaces", UAI 2021`
[^5]: [`BAxUS`](https://github.com/LeoIV/BAxUS)
`Leonard Papenmeier, Luigi Nardi, and Matthias Poloczek, "Increasing the Scope as You Learn: Adaptive Bayesian Optimization in Nested Subspaces", NeurIPS 2022`
[^6]: [`BODi`](https://github.com/aryandeshwal/BODi)
`Aryan Deshwal, Sebastian Ament, Maximilian Balandat, Eytan Bakshy, Janardhan Rao Doppa, and David Eriksson, "Bayesian Optimization over High-Dimensional Combinatorial Spaces via Dictionary-based Embeddings", AISTATS 2023`
[^7]: [`Bounce`](https://github.com/LeoIV/bounce)
`Leonard Papenmeier, Luigi Nardi and Matthias Poloczek, "Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and Mixed Spaces", NeurIPS 2023`
[^8]: The SVM benchmark is not included in the repository and was obtained by corresponding with the authors of the
paper.
[^9]: [`LA-MCTS`](https://github.com/facebookresearch/LA-MCTS)
`Linnan Wang, Rodrigo Fonseca, and Yuandong Tian, "Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search", NeurIPS 2020`
[^10]: Oh, Changyong, et al. "Combinatorial bayesian optimization using the graph cartesian product." Advances in Neural
Information Processing Systems 32 (2019).
[^11]: de Nobel, Jacob, et al. "Iohexperimenter: Benchmarking platform for iterative optimization heuristics."
Evolutionary Computation 32.3 (2024): 205-210.
[^12]: Hansen, Nikolaus, et al. "COCO: A platform for comparing continuous optimizers in a black-box setting."
Optimization Methods and Software 36.1 (2021): 114-144.
[^13]: Each category has 5 possible values. The benchmark expects an integer between 0 and 4 for each category.
[^14]: Ryu, Seunghee, et al. "BO4Mob: Bayesian Optimization Benchmarks for High-Dimensional Urban Mobility Problem." *arXiv preprint arXiv:2510.18824* (2025). For 1ramp, values should be integers between 1 and 2500. For the other scenarios, values should be integers between 1 and 2000.
