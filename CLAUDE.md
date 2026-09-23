# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Bencher is a collection of black-box optimization benchmarks packaged behind a single gRPC façade. Clients (typically using `bencherscaffold`) talk to one front-door server on `:50051`; that server fans out to per-family benchmark services on internal ports. The whole stack is normally consumed as a Docker/Apptainer image (`gaunab/bencher` on Docker Hub).

## Architecture

The repo is a multi-package monorepo. Each top-level directory whose name ends in `Benchmark(s)` is an independent Python project with its own `pyproject.toml`, `uv.lock`, `.python-version`, and `.venv`. They do **not** share a virtualenv — each family pins its own Python (3.8, 3.10, or 3.11) because the upstream benchmark libraries are mutually incompatible.

Request flow:

1. Client calls `BencherServer` (`BencherServer/src/bencherserver/server.py`) on port 50051.
2. `BencherServer.serve()` reads `BencherServer/benchmark-registry.json` and registers a gRPC stub for every `{benchmark_name → (host, port)}` entry.
3. Each per-family service is a `DualStackGRCPService` (from `bencherscaffold`) that owns one internal port and dispatches by `request.benchmark.name` via a local `benchmark_map`.

Current port allocation (from `benchmark-registry.json`):

| Port  | Family / package                  |
|-------|-----------------------------------|
| 50051 | `BencherServer` (front door)      |
| 50053 | `LassoBenchmarks`                 |
| 50054 | `NoDependencyBenchmark` (mopta08, pestcontrol) |
| 50055 | `MaxSATBenchmarks`                |
| 50056 | `EboBenchmarks` (robotpushing, rover) |
| 50057 | `MujocoBenchmarks` (+ lunarlander) |
| 50058 | `SVMBenchmarks`                   |
| 50059 | `IOHBenchmarks` (bbob-*, pbo-*, graph-*) |
| 50060 | `BO4MobBenchmark` (urban mobility)|

**When adding or renaming a benchmark, you must update `BencherServer/benchmark-registry.json` — the front door only advertises what's listed there.** `dimensions: any` benchmarks (bbob, pbo) are represented per-instance in the registry by the IOH service itself.

Input convention: clients send `Value`s in `[0, 1]` (or as ints for binary/integer/categorical). Each service rescales to its native domain in its `evaluate()` method (e.g. `LassoServiceServicer.evaluate` does `x = 2*x - 1` for `[-1, 1]`). When adding a benchmark, replicate this normalization contract.

## Running the stack

`entrypoint.py` is what runs inside the container. It scans `/opt/bencher/*/pyproject.toml`, reads each `[project.scripts].start-benchmark-service` to find the module to run, and launches each one as a subprocess using that package's own `.venv/bin/python -m <module>`. If any thread dies, the entrypoint exits with status 1 (Docker restart policy is expected to bring it back). `start_all.sh` is an older zsh variant that does the same thing by invoking the venv binary directly — prefer `entrypoint.py`.

## Common commands

Build & run the full container (needed only for the end-to-end tier; the contract and per-package tiers run without it):

```shell
docker build -t bencher .
docker run -p 50051:50051 --restart always -d bencher:latest
```

Work on a single benchmark family locally (no Docker):

```shell
cd LassoBenchmarks       # or MujocoBenchmarks, IOHBenchmarks, etc.
uv sync                  # recreates .venv from uv.lock against the pinned .python-version
uv run start-benchmark-service   # starts that family's service on its own internal port
```

To run the front door against locally-running family services, also `cd BencherServer && uv sync && uv run start-benchmark-service` — but note that the registry hardcodes `host: localhost` (default) and the family ports above must be free.

Apptainer / Singularity is documented in `README.md`; the same image is the base.

## Cross-cutting things to know

- **The Dockerfile's layer order is load-bearing.** Stages run system → interpreters →
  dependencies → datasets → final, so an ordinary source change reuses 31 of 42 layers
  and rebuilds in seconds rather than recompiling three CPythons and re-downloading
  datasets. Keep `COPY . /opt/bencher` below the dependency sync, keep `.dockerignore`
  excluding `.venv` and `tests/`, and chmod each tree in the stage that creates it —
  a single recursive chmod at the end cost 462s of every rebuild. `tests/test_dockerfile.py`
  guards these.

- **Python versions differ per package.** `EboBenchmarks` and `MujocoBenchmarks` pin 3.8.20, `LassoBenchmarks` pins 3.10.19, the rest 3.11.13. Syntax must match: `X | Y` annotations need 3.10, so the 3.8 packages carry `from __future__ import annotations`, and `tests/test_python_compatibility.py` enforces it. The Dockerfile uses uv-managed Python installations for each package's `.python-version`. Don't bump a `.python-version` without verifying the upstream library still works.
- **Heavy datasets are baked into the image at build time** to avoid runtime downloads: MaxSAT corpora, libsvm datasets (via `libsvmdata`), the SVM slice-localization dataset, and the MOPTA executable. The relevant env vars (`LIBSVMDATA_HOME`, `SVM_DATA_DIR`, `MOPTA_DATA_DIR`) are set in the Dockerfile and must match what the benchmark code reads at runtime. Editing those paths means editing both.
- **MuJoCo** lives at `/opt/mujoco210` inside the container; locally on macOS see the `~/.mujoco/mujoco210` setup in `README.md`. `mujoco-py` is finicky about `LD_LIBRARY_PATH` and `MUJOCO_PY_MUJOCO_PATH` — both are set in the Dockerfile.
- **Native build deps.** macOS local builds need Homebrew `swig`, `gfortran`, `openblas`, `pkg-config`, `glfw`, `libomp`, and `PKG_CONFIG_PATH` exported to point at the OpenBLAS pkgconfig directory. A `x86_64` Python on Apple Silicon will fail builds — use an arm64 Python.
- **Tests are tiered; run Tier 0 first.** `uv run pytest tests -q` at the repo root takes ~0.3s and needs no benchmark dependencies — it parses source and config against the installed `bencherscaffold`. It catches scaffold API drift, lock/pin divergence, registry drift, and syntax newer than a package's pinned interpreter. Per-package suites are `cd <Package> && uv run --locked --group dev pytest tests -q`; `tests/e2e/` needs a running container. See `AGENTS.md` for the full tiering.
- **Services expose a pure `evaluate(name, x, seed=None)`** alongside `evaluate_point`, plus a declarative `BENCHMARKS` mapping (IOH uses `PROBLEM_FAMILIES`, BO4Mob `NETWORKS`, since their names are generated). Dimensions and types declared there are cross-checked against `benchmark-registry.json` by Tier 0 — keep the two in step.
- **Goldens are sectioned by platform.** Per-package `tests/goldens.json`, shape
  `{"any": {...}, "linux-x86_64": {...}}`, read through `tests/goldens.py`.
  Arithmetic benchmarks record under `any` -- one value serves every machine.
  MuJoCo sets `GOLDENS_PLATFORM_SPECIFIC = True` because a physics rollout
  amplifies floating-point differences: the same seed gives `+4.31` on
  macOS/arm64 and `-1.71` on linux/amd64 for `mujoco-walker`. Where the running
  platform has no section the golden test **skips** and the relative assertions
  (same seed twice is equal, different seeds differ) carry the load.
  Regenerate with `pytest tests --update-goldens`; it merges, never truncates.
  Record for another architecture with the `record-goldens` workflow_dispatch
  input, which captures on a real runner and uploads an artifact -- never under
  emulation, where results may differ from real hardware.
- **CI runs three workflows.** `pr-checks.yml` on every PR: Tier 0, plus a per-package matrix derived from each package's `[tool.bencher.ci] tier`, plus an informational container leg. `docker_build.yml` builds the image and runs `tests/e2e` nightly, on `main`, on tags, and on demand. `update_scaffold_version.yml` bumps `bencherscaffold` across all ten projects, gates on Tier 0, and opens a PR — it no longer pushes to `main`.
- **gRPC types come from `bencherscaffold`**, not this repo. Don't try to regenerate protobufs here. Family services implement `SecondLevelBencher`; only `BencherServer` implements `Bencher`, which is why `BencherClient` can only talk to the front door.

## Conventions (from AGENTS.md)

- Two-space indentation in shell scripts; type-annotated Python.
- Benchmark identifiers are kebab-case (`lasso-dna`, `mujoco-ant`).
- Commits are short and imperative (e.g. `fix container setup`, `upgrade bo4mob`); call out Docker/registry changes in the body.
