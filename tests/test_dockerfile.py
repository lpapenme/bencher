"""Guards the Dockerfile's per-package COPY lists and the build context.

The Dockerfile copies each package's pyproject.toml, uv.lock and .python-version
explicitly rather than by glob, because a glob would flatten nine identically
named files into one. Explicit lists drift when a package is added, so these
tests keep them honest -- a new benchmark that is not wired into the Dockerfile
would otherwise build an image silently missing that service.
"""
import re

import pytest

from conftest import REPO_ROOT, package_dirs

DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text()
DOCKERIGNORE = (REPO_ROOT / ".dockerignore").read_text()

PACKAGES = package_dirs()
PACKAGE_IDS = [p.name for p in PACKAGES]


@pytest.mark.parametrize("pkg", PACKAGES, ids=PACKAGE_IDS)
def test_python_version_is_copied_for_every_package(pkg):
    """Without this the interpreter layer would miss a package's Python."""
    assert re.search(rf"^COPY\s+{re.escape(pkg.name)}/\.python-version\s", DOCKERFILE, re.M), (
        f"Dockerfile never copies {pkg.name}/.python-version; add it to the "
        f"interpreters stage")


@pytest.mark.parametrize("pkg", PACKAGES, ids=PACKAGE_IDS)
def test_dependency_files_are_copied_for_every_package(pkg):
    """The dependency layer must see this package's pyproject.toml and uv.lock."""
    line = re.search(rf"^COPY\s+{re.escape(pkg.name)}/pyproject\.toml\s+.*$",
                     DOCKERFILE, re.M)
    assert line, f"Dockerfile never copies {pkg.name}/pyproject.toml"
    assert f"{pkg.name}/uv.lock" in line.group(0), (
        f"Dockerfile copies {pkg.name}/pyproject.toml without its uv.lock; "
        f"`uv sync --frozen` would fail")


def test_source_is_copied_after_dependencies_are_installed():
    """The whole point of the layout: deps must install before source lands.

    If `COPY . /opt/bencher` moves back above the dependency sync, every source
    change rebuilds every virtualenv again.
    """
    # Match instructions, not the prose in this file's header comment.
    lines = DOCKERFILE.splitlines()
    deps_sync = next(i for i, l in enumerate(lines)
                     if l.strip().startswith(("RUN", "COPY", "uv", "PYENV_VERSION"))
                     and "--no-install-project" in l)
    full_copy = next(i for i, l in enumerate(lines)
                     if re.match(r"^COPY\s+\.\s+/opt/bencher\s*$", l))
    assert deps_sync < full_copy, (
        "`COPY . /opt/bencher` must come after the dependency-only sync, "
        "otherwise a source edit invalidates the whole dependency build")


def test_dockerignore_excludes_local_virtualenvs():
    """Excluding .venv keeps the context small AND stops the source copy from
    clobbering the virtualenvs built in the dependency layer."""
    assert re.search(r"^\*/\.venv/?$", DOCKERIGNORE, re.M), (
        ".dockerignore must exclude */.venv -- otherwise `COPY . /opt/bencher` "
        "overwrites the virtualenvs built earlier in the build")


@pytest.mark.parametrize("script", [
    "prefetch_libsvm.py", "prefetch_maxsat.py", "prefetch_svm.py", "prefetch_mopta.py",
])
def test_referenced_prefetch_scripts_exist(script):
    assert f"/opt/bencher-build/{script}" in DOCKERFILE, (
        f"docker/{script} exists but the Dockerfile does not run it")
    assert (REPO_ROOT / "docker" / script).is_file(), (
        f"Dockerfile runs {script} but docker/{script} does not exist")
