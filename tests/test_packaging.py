"""Guards the nine packages' metadata and lockfiles.

Today the only thing that validates lock/pyproject agreement is
`uv sync --frozen` inside the Docker build, ~40 minutes in. These checks take
milliseconds. They matter because update_scaffold_version.yml moves `uv.lock`
*without* touching the `pyproject.toml` pin, so the two can silently diverge.
"""
import shutil
import subprocess
import tomllib

import pytest

from conftest import REPO_ROOT, package_dirs, rel

SCAFFOLD = "bencherscaffold"
SERVICE_SCRIPT = "start-benchmark-service"

PACKAGES = package_dirs()
PACKAGE_IDS = [p.name for p in PACKAGES]


def _pyproject(pkg):
    return tomllib.loads((pkg / "pyproject.toml").read_text())


def _scaffold_specifier(pkg) -> str | None:
    for dep in _pyproject(pkg).get("project", {}).get("dependencies", []):
        if dep.replace("-", "_").lower().startswith(SCAFFOLD):
            return dep
    return None


def _locked_scaffold_version(pkg) -> str | None:
    """The bencherscaffold version pinned in uv.lock, read without running uv."""
    lock = tomllib.loads((pkg / "uv.lock").read_text())
    for package in lock.get("package", []):
        if package.get("name") == SCAFFOLD:
            return package.get("version")
    return None


@pytest.mark.parametrize("pkg", PACKAGES, ids=PACKAGE_IDS)
def test_package_declares_a_service_entry_point(pkg):
    """entrypoint.py launches each service via [project.scripts]; it must exist."""
    scripts = _pyproject(pkg).get("project", {}).get("scripts", {})
    assert SERVICE_SCRIPT in scripts, (
        f"{rel(pkg)}/pyproject.toml has no [project.scripts] {SERVICE_SCRIPT}; "
        f"entrypoint.py would skip it silently")
    assert ":" in scripts[SERVICE_SCRIPT], (
        f"{rel(pkg)}: {SERVICE_SCRIPT} must be 'module:function', "
        f"got {scripts[SERVICE_SCRIPT]!r}")


@pytest.mark.parametrize("pkg", PACKAGES, ids=PACKAGE_IDS)
def test_package_has_a_lockfile(pkg):
    assert (pkg / "uv.lock").is_file(), f"{rel(pkg)} has no uv.lock"


@pytest.mark.parametrize("pkg", PACKAGES, ids=PACKAGE_IDS)
def test_scaffold_pin_present_and_satisfied_by_the_lock(pkg):
    """The pyproject pin and the locked version must agree.

    update_scaffold_version.yml only runs `uv lock --upgrade-package`, so the lock
    can advance past the pin. `uv sync --frozen` does not catch that -- this does.
    """
    spec = _scaffold_specifier(pkg)
    assert spec, f"{rel(pkg)} does not depend on {SCAFFOLD}"

    locked = _locked_scaffold_version(pkg)
    assert locked, f"{rel(pkg)}/uv.lock does not contain {SCAFFOLD}"

    # Specifiers here are all of the form `bencherscaffold~=X.Y.Z`.
    assert "~=" in spec, f"{rel(pkg)}: expected a compatible-release pin, got {spec!r}"
    pinned = spec.split("~=", 1)[1].strip().strip('"').strip("'")
    pin_parts = [int(x) for x in pinned.split(".")]
    lock_parts = [int(x) for x in locked.split(".")]
    # `~=X.Y.Z` means >=X.Y.Z, ==X.Y.*
    assert lock_parts[:2] == pin_parts[:2] and lock_parts >= pin_parts, (
        f"{rel(pkg)}: pyproject pins {SCAFFOLD}{spec.split(SCAFFOLD)[1]} but "
        f"uv.lock has {locked}")


def test_all_packages_pin_the_same_scaffold_version():
    """A split-brain scaffold version across services is a wire-compat hazard."""
    pins = {p.name: _scaffold_specifier(p) for p in PACKAGES}
    distinct = set(pins.values())
    assert len(distinct) == 1, f"packages disagree on the {SCAFFOLD} pin: {pins}"


@pytest.mark.parametrize("pkg", PACKAGES, ids=PACKAGE_IDS)
def test_python_version_is_compatible_with_requires_python(pkg):
    """The pinned interpreter must satisfy requires-python.

    The Dockerfile installs whatever `.python-version` says and then runs
    `uv sync --frozen` against it; a mismatch fails the build deep in the run.
    """
    version_file = pkg / ".python-version"
    if not version_file.is_file():
        pytest.skip(f"{rel(pkg)} has no .python-version (uses the default interpreter)")
    pinned = version_file.read_text().strip()
    requires = _pyproject(pkg).get("project", {}).get("requires-python")
    assert requires, f"{rel(pkg)} has .python-version but no requires-python"

    from packaging.specifiers import SpecifierSet  # noqa: PLC0415
    from packaging.version import Version  # noqa: PLC0415
    assert Version(pinned) in SpecifierSet(requires), (
        f"{rel(pkg)}: .python-version {pinned} does not satisfy "
        f"requires-python {requires!r}")


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
@pytest.mark.parametrize("pkg", PACKAGES, ids=PACKAGE_IDS)
def test_lockfile_is_up_to_date_with_pyproject(pkg):
    """`uv lock --check` -- the authoritative lock/pyproject agreement check."""
    # No --offline: on a cold CI cache uv needs the index to verify the
    # resolution. Runs in milliseconds once the cache is warm.
    result = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=pkg, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"{rel(pkg)}: uv.lock is out of date with pyproject.toml.\n"
        f"Run `cd {rel(pkg)} && uv lock`.\n{result.stderr.strip()}")
