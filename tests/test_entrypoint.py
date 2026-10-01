"""Tests entrypoint.py's service discovery.

entrypoint.py is what actually runs inside the container: it scans /opt/bencher
for packages and launches each one's service from that package's own .venv. Two
of its failure paths print and return silently, which the watchdog then reports
as a crashed thread -- worth pinning so the behaviour is deliberate.
"""
import importlib.util
import sys

import pytest

from conftest import REPO_ROOT, package_dirs


def _load_entrypoint():
    """Import entrypoint.py by path; it is a script at the repo root, not a package."""
    spec = importlib.util.spec_from_file_location("bencher_entrypoint",
                                                  REPO_ROOT / "entrypoint.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def entrypoint():
    return _load_entrypoint()


def test_entrypoint_imports_cleanly(entrypoint):
    """Guards against a syntax/stdlib error that would break every service at once."""
    assert hasattr(entrypoint, "ServiceThread")


def test_every_package_exposes_the_module_entrypoint_resolves(entrypoint):
    """entrypoint.py derives `python -m <module>` from [project.scripts]; that
    module must exist on disk, or the service dies at container start."""
    missing = []
    for pkg in package_dirs():
        import tomllib
        data = tomllib.loads((pkg / "pyproject.toml").read_text())
        script = data["project"]["scripts"]["start-benchmark-service"]
        module = script.split(":")[0]
        module_path = pkg / "src" / module.replace(".", "/")
        if not module_path.with_suffix(".py").is_file():
            missing.append(f"{pkg.name}: {script} -> {module_path}.py does not exist")
    assert not missing, "entrypoint would fail to launch:\n" + "\n".join(missing)


def test_service_thread_skips_a_directory_without_pyproject(entrypoint, tmp_path, capsys):
    """A dir with no pyproject.toml is skipped, not crashed on.

    MECHBenchmarks is exactly this case today.
    """
    thread = entrypoint.ServiceThread(str(tmp_path))
    thread.run()
    assert "pyproject.toml not found" in capsys.readouterr().out


def test_service_thread_skips_a_package_without_the_script_key(entrypoint, tmp_path, capsys):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    thread = entrypoint.ServiceThread(str(tmp_path))
    thread.run()
    assert "Missing key" in capsys.readouterr().out
