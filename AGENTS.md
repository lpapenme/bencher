# Repository Guidelines

## Project Structure & Module Organization
Bencher groups benchmark families in peer directories such as `BO4MobBenchmark`, `EboBenchmarks`, `MaxSATBenchmarks`, and `MujocoBenchmarks`. Each package is an isolated project with its own `pyproject.toml`, `src/` tree, `uv.lock` and `.venv`. They do not share an interpreter: `EboBenchmarks` and `MujocoBenchmarks` pin Python 3.8, `LassoBenchmarks` 3.10, the rest 3.11, because the upstream libraries are mutually incompatible. `BencherServer/src/bencherserver` exposes the gRPC interface defined in `benchmark-registry.json`. Container collateral (`Dockerfile`, `container.sdef`, `entrypoint.py`, `start_all.sh`) lives at the repo root; keep new assets there so the orchestration scripts can locate them.

## Build, Test, and Development Commands
`uv sync` run inside any benchmark directory recreates the local env from `uv.lock`. `uv run pytest tests -q` at the repo root runs the contract suite in well under a second and is the first thing to run after any change. `uv run start-benchmark-service` bootstraps a single benchmark service for local debugging. `docker build -t bencher .` builds the full stack container, and `docker run -p 50051:50051 --restart always -d bencher:latest` keeps it reachable by clients. Use `docker pull gaunab/bencher:latest` when you only need the published image. `apptainer build container.sif container.sdef` plus `apptainer instance start container.sif bencher` reproduces the Singularity workflow described in `README.md`.

## Coding Style & Naming Conventions
Favor type annotations, and keep modules under `src/<package>/`. Note the interpreter a package pins before using newer syntax — `X | Y` annotations need 3.10, so the 3.8 packages carry `from __future__ import annotations`. Use `snake_case` for files and functions, `PascalCase` for classes, and descriptive benchmark identifiers such as `lasso-dna`. Container scripts are POSIX shell (the provided `start_all.sh` is zsh-compatible); stick to the existing two-space indentation. Update `benchmark-registry.json` whenever you add or rename a service entry so the server can advertise it.

## Testing Guidelines
Tests are tiered by cost. Run the cheap ones constantly; the expensive one runs in CI.

**Tier 0 — contract (`tests/`, repo root).** `uv run pytest tests -q`. ~0.3s, and deliberately needs no benchmark dependencies: it parses source and config against the installed `bencherscaffold`. Catches scaffold API drift, `uv.lock` disagreeing with `pyproject.toml`, `benchmark-registry.json` drifting from what services serve, and source using syntax newer than the interpreter a package pins.

**Tier 1/2 — per package (`<Package>/tests/`).** `cd <Package> && uv run --locked --group dev pytest tests -q`. Tier 1 calls the service's `evaluate()` directly; Tier 2 starts the real servicer on an ephemeral port and drives it over gRPC via `tests/grpc_harness.py`. Every package has a suite, and all nine run on every PR.

**Tier 3 — end to end (`tests/e2e/`).** Needs a running container: `uv run pytest tests/e2e -q --target localhost:50051`. Drives every benchmark the registry advertises, so a new registry entry is covered automatically.

Adding a benchmark: register it in `benchmark-registry.json`, declare it in the service's `BENCHMARKS` mapping (dimensions and type are cross-checked against the registry), and record a golden if it is deterministic — `pytest tests --update-goldens` regenerates them. Stochastic benchmarks take a `seed`; pin their goldens at a fixed seed rather than leaving them unchecked. Mark anything needing a baked-in dataset `@pytest.mark.dataset` so it only runs where the data exists.

Adding a package: give it `tests/`, a `dev` dependency group with pytest, `[tool.pytest.ini_options]`, and `[tool.bencher.ci] tier = "runner"` (or `"container"` if it needs MuJoCo binaries or SUMO). CI derives its matrix from these, and Tier 0 fails if any are missing.

## Commit & Pull Request Guidelines
Follow the existing short, imperative style (`fix container setup`, `upgrade bo4mob`). Keep one change per commit; mention Docker or registry updates in the message body whenever deployment behavior changes. PRs should summarize the benchmark(s) touched, list the commands you ran (build/test), and link the relevant issue. Attach logs or screenshots for new client-facing behavior and call out any follow-up tasks.

## Security & Configuration Tips
Never hard-code credentials or proprietary datasets. Mac builds rely on Homebrew OpenBLAS—export `PKG_CONFIG_PATH="$(brew --prefix openblas)/lib/pkgconfig"` before compiling native extras. Mujoco assets belong under `~/.mujoco/mujoco210`; do not commit them. Keep gRPC endpoints on `localhost:50051` while developing and avoid exposing containers without authentication.
