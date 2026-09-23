"""One source of truth for which benchmarks the tests drive, and with what point.

Derived from BencherServer/benchmark-registry.json rather than duplicating it:
the registry already carries every benchmark's name, dimensionality and type.
This module adds only what testing needs on top -- the fill value to send, which
benchmarks are stochastic, and which are too expensive to run routinely.

Both the e2e suite and capture_goldens.py read this, so the point a golden was
captured at cannot drift from the point the test replays.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "BencherServer" / "benchmark-registry.json"

# Registry `type` -> the ValueType name a client should send. Resolved to the
# real enum by the caller, so this module stays importable without grpc.
FILL_BY_TYPE = {
    "purely_continuous": ("CONTINUOUS", 0.5),
    "purely_binary": ("BINARY", 1),
    "purely_integer": ("INTEGER", 1),
    "purely_categorical": ("CATEGORICAL", 1),
    "purely_ordinal_real": ("CONTINUOUS", 0.5),
    "purely_ordinal_int": ("INTEGER", 1),
    # svmmixed: leading coordinates are a binary feature mask, trailing three are
    # continuous hyperparameters. A constant 1 fill selects every feature, which
    # is a valid point and avoids the empty-selection short circuit.
    "mixed": ("CONTINUOUS", 1),
}

# Benchmarks whose value legitimately changes between identical calls, so no
# golden can be pinned until BenchmarkRequest.random_seed is threaded through.
# Everything else is expected to be reproducible.
STOCHASTIC = {
    "pestcontrol",
    "rover",
    "robotpushing",
    "lunarlander",
    "mujoco-ant", "mujoco-hopper", "mujoco-walker",
    "mujoco-halfcheetah", "mujoco-swimmer", "mujoco-humanoid",
}

# Too slow or too large to run on every sweep. Dimensionality is a decent proxy
# (a fresh LassoBench problem is built per evaluation, and a MuJoCo rollout
# scales with the policy size), plus a few known-slow small ones.
EXPENSIVE_ABOVE_DIMENSIONS = 1000
EXPENSIVE = {
    "svmmixed",          # fits an XGBoost model on 10k rows before the SVR
    "mujoco-humanoid",   # 6392-d policy rollout
}

# BO4Mob templates 420 names over 5 networks x 14 dates x 3 hours x 2 metrics.
# They share one code path, so one representative per network is enough; the
# rest add SUMO simulation time without adding coverage.
BO4MOB_PORT = 50060


def _bo4mob_representatives(registry: dict) -> set:
    """One benchmark per BO4Mob network -- the alphabetically first."""
    by_network: dict[str, str] = {}
    for name, props in registry.items():
        if props["port"] != BO4MOB_PORT:
            continue
        network = name.split("_")[0]
        if network not in by_network or name < by_network[network]:
            by_network[network] = name
    return set(by_network.values())


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def all_benchmarks(registry: dict | None = None) -> list[dict]:
    """Every benchmark worth driving, with the point to drive it with.

    Each entry: name, dimensions, value_type (a ValueType *name*), fill,
    deterministic, expensive.
    """
    registry = registry if registry is not None else load_registry()
    representatives = _bo4mob_representatives(registry)

    entries = []
    for name, props in sorted(registry.items()):
        if props["port"] == BO4MOB_PORT and name not in representatives:
            continue
        value_type, fill = FILL_BY_TYPE[props["type"]]
        # bbob/pbo are dimension-agnostic (registry records null); 10 and 16 are
        # small enough to be fast and large enough to be meaningful.
        dimensions = props["dimensions"]
        if dimensions is None:
            dimensions = 16 if value_type in ("BINARY", "INTEGER") else 10
        entries.append({
            "name": name,
            "dimensions": dimensions,
            "value_type": value_type,
            "fill": fill,
            "deterministic": name not in STOCHASTIC,
            "expensive": name in EXPENSIVE or dimensions > EXPENSIVE_ABOVE_DIMENSIONS,
        })
    return entries


def selected(include_expensive: bool = False, deterministic_only: bool = False) -> list[dict]:
    return [
        e for e in all_benchmarks()
        if (include_expensive or not e["expensive"])
        and (not deterministic_only or e["deterministic"])
    ]
