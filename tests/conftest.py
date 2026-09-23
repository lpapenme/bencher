"""Shared helpers for the Tier 0 contract tests.

These tests are deliberately *static*: they parse source and config rather than
importing benchmark modules, so the whole suite runs against a venv containing
only pytest and bencherscaffold. That is what lets them gate every PR in seconds
instead of waiting ~40 minutes for a container build.
"""
import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "BencherServer" / "benchmark-registry.json"


def package_dirs() -> list[Path]:
    """Every benchmark/server package: a top-level dir with a pyproject.toml.

    Excludes the repo root itself (which holds these contract tests) and dirs
    like MECHBenchmarks that have no pyproject and are skipped by entrypoint.py.
    """
    return sorted(
        p.parent
        for p in REPO_ROOT.glob("*/pyproject.toml")
        if p.parent.name != "tests"
    )


def service_main_files() -> list[Path]:
    """The `evaluate_point` implementations: */src/*/main.py."""
    return sorted(REPO_ROOT.glob("*/src/*/main.py"))


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def load_registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def registry_by_port(registry: dict | None = None) -> dict[int, set[str]]:
    registry = registry if registry is not None else load_registry()
    out: dict[int, set[str]] = {}
    for name, props in registry.items():
        out.setdefault(props["port"], set()).add(name)
    return out


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _is_benchmark_name_expr(node: ast.expr) -> bool:
    """True for `request.benchmark.name` (the value every service dispatches on)."""
    return isinstance(node, ast.Attribute) and node.attr == "name"


def dispatch_names(path: Path, variables: set[str]) -> set[str]:
    """Benchmark names a service dispatches on, extracted without importing it.

    Covers the idioms actually present in this repo:
      1. a named dict or list literal (`benchmark_map`, `SUPPORTED_BENCHMARKS`,
         `filename_map`, `func_factory_map`, `valid_benchmark_names`) -- including
         ones assigned inside a function body, as SVMBenchmarks does;
      2. `... .name in ['a', 'b']`, as EboBenchmarks asserts inline;
      3. `... .name == 'a'`, which is how MujocoBenchmarks serves `lunarlander`
         and SVMBenchmarks branches between svm/svmmixed.

    Once dispatch is normalised to a single declarative BENCHMARKS mapping per
    service (see the refactor step), cases 2 and 3 become dead code here.
    """
    names: set[str] = set()
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in variables:
                    names |= _constants_in(node.value)
        elif isinstance(node, ast.Compare) and _is_benchmark_name_expr(node.left):
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.Eq)):
                    names |= _constants_in(comparator)
    return names


def _constants_in(node: ast.expr) -> set[str]:
    """String constants in a literal: a bare str, a list/tuple, or dict keys."""
    if isinstance(node, ast.Constant):
        return {node.value} if isinstance(node.value, str) else set()
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return {e.value for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    if isinstance(node, ast.Dict):
        return {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return set()


def declared_specs(path: Path, variable: str = "BENCHMARKS") -> dict[str, dict]:
    """The constant fields of a service's declarative BENCHMARKS mapping.

    Returns {benchmark_name: {field: value}} for literal values only, so callers
    can cross-check `dimensions` and `type` against the registry. Non-literal
    entries (a `factory` lambda, say) are simply omitted.
    """
    specs: dict[str, dict] = {}
    for node in ast.walk(parse(path)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key, value in zip(node.value.keys, node.value.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            fields = {}
            if isinstance(value, ast.Dict):
                for field, field_value in zip(value.keys, value.values):
                    if (isinstance(field, ast.Constant)
                            and isinstance(field_value, ast.Constant)):
                        fields[field.value] = field_value.value
            specs[key.value] = fields
    return specs


def string_list_literal(path: Path, variable: str) -> list[str]:
    """The string elements of a named list literal, in source order."""
    for node in ast.walk(parse(path)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable:
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return [e.value for e in node.value.elts
                                if isinstance(e, ast.Constant)]
    raise AssertionError(f"no list literal named {variable!r} in {rel(path)}")


@pytest.fixture(scope="session")
def registry() -> dict:
    return load_registry()
