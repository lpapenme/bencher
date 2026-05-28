# MECHBenchmarks

bencher service wrapping [MECHBench](https://github.com/BayesOptApp/MECHBench) — structural
mechanics optimization problems solved with the OpenRadioss explicit-dynamics FEM solver.

Runs on port `50061` (override with `BENCHER_MECHBENCH_PORT`).

## Benchmarks

| name | MECHBench problem | objective (minimize) | dims |
| --- | --- | --- | --- |
| `mechbench-starbox`   | Star Box (1)            | `penalized_sea`    | 1–34 |
| `mechbench-bending`   | Three-Point Bending (2) | `penalized_mass`   | 1–40 |
| `mechbench-crashtube` | Crash Tube (3)          | `load_uniformity`  | ~2–30 |

Dimensionality is variable: the MECHBench `dimension` is taken from the length of the query
point, so each benchmark is registered with `dimensions: null`. Clients send points in
`[0, 1]`; the service rescales to MECHBench's `[-5, 5]` search space (`x_native = 10·x − 5`).

## OpenRadioss

Every objective runs at least the OpenRadioss starter, and OpenRadioss ships only Linux/Windows
binaries — so evaluations work in the bencher **Linux container**, not on macOS. The container
bundles OpenRadioss and points the service at it via `BENCHER_MECHBENCH_OPENRADIOSS_PATH`. If
that variable is unset, MECHBench attempts to download the solver into the working directory on
first use (Linux/Windows only).

## Dependency

`mechbench` is pulled from a packaging-fixed fork
(<https://github.com/lpapenme/MECHBench>) — upstream installs as `sob` but imports itself as
`src.sob`, so it is not importable once installed; the fork fixes the import paths and ships the
`.rad` input decks as package data.
