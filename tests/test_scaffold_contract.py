"""Guards against bencherscaffold API drift.

bencherscaffold is an external dependency whose releases are auto-merged here by
.github/workflows/update_scaffold_version.yml. This test suite ensures that the services are compatible with the installed version of the scaffold.
"""
import ast

import pytest
from bencherscaffold.protoclasses import bencher_pb2

from conftest import REPO_ROOT, package_dirs, parse, rel, service_main_files

# Scaffold message types the services construct directly.
CONSTRUCTED_MESSAGES = ("EvaluationResult", "ObjectiveValue", "Constraint")

# Every service reports a single objective under this name. Keeping it uniform is
# what lets a client do `result.objectives[0]` without knowing the family.
SINGLE_OBJECTIVE_NAME = "f0"


def _fields(message_name: str) -> set[str]:
    return {f.name for f in getattr(bencher_pb2, message_name).DESCRIPTOR.fields}


def _construction_calls(path):
    """Every `EvaluationResult(...)` / `ObjectiveValue(...)` / `Constraint(...)` call."""
    for node in ast.walk(parse(path)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in CONSTRUCTED_MESSAGES):
            yield node


@pytest.mark.parametrize("path", service_main_files(), ids=rel)
def test_constructor_keywords_exist_on_installed_scaffold(path):
    """No service may pass a keyword the installed scaffold does not define.

    This is the check that fails loudly on a 0.6.2-style or 0.6.3-style change,
    naming every offending line, before anything is built or pushed.
    """
    violations = []
    for call in _construction_calls(path):
        allowed = _fields(call.func.id)
        used = {kw.arg for kw in call.keywords if kw.arg is not None}
        unknown = used - allowed
        if unknown:
            violations.append(
                f"{rel(path)}:{call.lineno}: {call.func.id}(...) passes "
                f"{sorted(unknown)}, but the installed scaffold defines {sorted(allowed)}"
            )
    assert not violations, "scaffold API drift:\n" + "\n".join(violations)


@pytest.mark.parametrize("path", service_main_files(), ids=rel)
def test_constructed_messages_are_imported(path):
    """A service that constructs a scaffold message must import it by name.

    Catches the half-migrated state where a call site is updated but the import
    line is not -- a NameError that would otherwise only appear at RPC time.
    """
    imported = {
        alias.asname or alias.name
        for node in ast.walk(parse(path))
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("bencherscaffold")
        for alias in node.names
    }
    constructed = {call.func.id for call in _construction_calls(path)}
    missing = constructed - imported
    assert not missing, f"{rel(path)} constructs {sorted(missing)} without importing it"


def test_every_service_constructs_an_evaluation_result():
    """Each family service must actually build a result -- no silently dead service."""
    missing = [
        rel(p) for p in service_main_files()
        if p.parent.parent.parent.name != "BencherServer"
        and not any(c.func.id == "EvaluationResult" for c in _construction_calls(p))
    ]
    assert not missing, f"services that never construct an EvaluationResult: {missing}"


@pytest.mark.parametrize("path", service_main_files(), ids=rel)
def test_single_objective_naming_is_uniform(path):
    """Every ObjectiveValue is named "f0", matching the scaffold's own convention.

    bencherscaffold's test helper is `single_objective(value, name="f0")`; clients
    rely on objectives[0] being the objective regardless of family.
    """
    wrong = []
    for call in _construction_calls(path):
        if call.func.id != "ObjectiveValue":
            continue
        for kw in call.keywords:
            if kw.arg == "name":
                if not (isinstance(kw.value, ast.Constant)
                        and kw.value.value == SINGLE_OBJECTIVE_NAME):
                    got = getattr(kw.value, "value", "<non-literal>")
                    wrong.append(f"{rel(path)}:{call.lineno}: name={got!r}")
    assert not wrong, (
        f"ObjectiveValue must be named {SINGLE_OBJECTIVE_NAME!r}:\n" + "\n".join(wrong))


def test_evaluation_result_shape_is_what_services_assume():
    """Pin the scaffold shape the services are written against.

    If a future release reshapes EvaluationResult again, this fails with a direct
    statement of what changed rather than a confusing downstream error.
    """
    assert _fields("EvaluationResult") == {"objectives", "constraints"}
    assert _fields("ObjectiveValue") == {"name", "value"}


def _locked_scaffold_version(pyproject_dir) -> str | None:
    import tomllib
    lock = pyproject_dir / "uv.lock"
    if not lock.is_file():
        return None
    data = tomllib.loads(lock.read_text())
    for package in data.get("package", []):
        if package.get("name") == "bencherscaffold":
            return package.get("version")
    return None


def test_tier0_runs_the_same_scaffold_the_services_will():
    """These tests are only meaningful if they check the version that ships.

    The root test environment and every package must lock the *same*
    bencherscaffold. When they diverge, Tier 0 validates one API while the
    container runs another -- which is exactly how a `HasField('random_seed')`
    call passed locally against 0.6.4 while every service locked 0.6.3 and would
    have raised ValueError on every evaluation.
    """
    from importlib.metadata import version as installed_version

    root = REPO_ROOT if (REPO_ROOT / "uv.lock").is_file() else None
    versions = {}
    if root is not None:
        versions["<root test env>"] = _locked_scaffold_version(root)
    for pkg in package_dirs():
        versions[pkg.name] = _locked_scaffold_version(pkg)

    locked = {v for v in versions.values() if v is not None}
    assert len(locked) == 1, (
        "packages lock different bencherscaffold versions, so Tier 0 does not "
        f"test what the container runs: {versions}")

    running = installed_version("bencherscaffold")
    assert running == locked.pop(), (
        f"these tests are running against bencherscaffold {running}, but the "
        f"packages lock {versions}. Re-sync the root environment.")


@pytest.mark.parametrize("path", service_main_files(), ids=rel)
def test_hasfield_names_exist_on_the_installed_scaffold(path):
    """`HasField("x")` raises ValueError if the field does not exist.

    Unlike a missing attribute this is not caught by any import check -- it
    fails at RPC time, once per request.
    """
    known = set()
    for message in ("BenchmarkRequest", "EvaluationResult", "ObjectiveValue",
                    "Constraint", "Benchmark", "Point", "Value"):
        descriptor = getattr(bencher_pb2, message, None)
        if descriptor is not None:
            known |= {f.name for f in descriptor.DESCRIPTOR.fields}

    bad = []
    for node in ast.walk(parse(path)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "HasField"
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            field = node.args[0].value
            if field not in known:
                bad.append(f"{rel(path)}:{node.lineno}: HasField({field!r})")
    assert not bad, (
        "HasField on a field the installed scaffold does not define:\n"
        + "\n".join(bad))
