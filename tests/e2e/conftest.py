"""Fixtures for the end-to-end tier.

These talk to a *running* bencher container over gRPC, so they are excluded from
the default pytest run (see addopts in the root pyproject.toml). Run them with:

    pytest tests/e2e --target localhost:50051

and regenerate the recorded values with:

    pytest tests/e2e --target localhost:50051 --update-goldens

They live here rather than in the bencherclient repo so that a change to a
service and the test that covers it land in the same commit.
"""
import json
import sys
from pathlib import Path

import pytest
from bencherscaffold.client import BencherClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks import selected  # noqa: E402  (needs the path above)

GOLDENS_PATH = Path(__file__).resolve().parent / "goldens.json"


def pytest_addoption(parser):
    parser.addoption(
        "--target", default="localhost:50051",
        help="host:port of the bencher front door (default: localhost:50051)")
    parser.addoption(
        "--include-expensive", action="store_true",
        help="also drive the benchmarks excluded for cost (see tests/benchmarks.py)")
    parser.addoption(
        "--update-goldens", action="store_true",
        help="rewrite tests/e2e/goldens.json from this run instead of asserting")


@pytest.fixture(scope="session")
def target(request) -> tuple[str, int]:
    host, _, port = request.config.getoption("--target").rpartition(":")
    return host or "localhost", int(port)


@pytest.fixture(scope="session")
def client(target) -> BencherClient:
    host, port = target
    # A few retries: the container starts every service concurrently, so the
    # first call to a slow family can land before it is listening.
    return BencherClient(address=host, port=port, max_retries=3, wait_time=5)


@pytest.fixture(scope="session")
def goldens() -> dict:
    if GOLDENS_PATH.is_file():
        return json.loads(GOLDENS_PATH.read_text())
    return {}


@pytest.fixture(scope="session")
def recorder(request):
    """Collects values during an --update-goldens run and writes them out once.

    Writing from a session finalizer rather than per test keeps a partial run
    from truncating the file: goldens.json is only replaced if the sweep got
    far enough to produce at least one value.
    """
    updating = request.config.getoption("--update-goldens")
    recorded: dict[str, float] = {}

    yield (recorded if updating else None)

    if updating and recorded:
        merged = {}
        if GOLDENS_PATH.is_file():
            merged.update(json.loads(GOLDENS_PATH.read_text()))
        merged.update(recorded)
        GOLDENS_PATH.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {len(recorded)} goldens to {GOLDENS_PATH}")


def pytest_generate_tests(metafunc):
    """Parametrize over the manifest, honouring --include-expensive."""
    if "benchmark" not in metafunc.fixturenames:
        return
    include_expensive = metafunc.config.getoption("--include-expensive")
    entries = selected(include_expensive=include_expensive)
    metafunc.parametrize("benchmark", entries, ids=[e["name"] for e in entries])
