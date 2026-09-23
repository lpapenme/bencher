"""Fixtures for the end-to-end tier.

These talk to a *running* bencher container over gRPC, so they are excluded from
the default pytest run (see addopts in the root pyproject.toml). Run them with:

    pytest tests/e2e --target localhost:50051

and regenerate the recorded values with:

    pytest tests/e2e --target localhost:50051 --update-goldens

They live here rather than in the bencherclient repo so that a change to a
service and the test that covers it land in the same commit.
"""
import sys
from pathlib import Path

import pytest
from bencherscaffold.client import BencherClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import goldens as goldens_store  # noqa: E402
from benchmarks import selected  # noqa: E402  (needs the path above)

GOLDENS_PATH = Path(__file__).resolve().parent / "goldens.json"
# The e2e sweep drives the deterministic benchmarks only, and those travel,
# so one shared section serves every platform. See tests/goldens.py.
GOLDENS_PLATFORM_SPECIFIC = False


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
    return goldens_store.load(GOLDENS_PATH)


@pytest.fixture(scope="session")
def golden_for(goldens):
    """The recorded value for a benchmark, or None if none exists."""
    def _lookup(name):
        return goldens_store.lookup(goldens, name, GOLDENS_PLATFORM_SPECIFIC)
    return _lookup


@pytest.fixture(scope="session")
def recorder(request):
    """Collects values during an --update-goldens run and writes them out once.

    Writing from a session finalizer rather than per test keeps a partial run
    from losing values, and the store merges rather than overwrites.
    """
    updating = request.config.getoption("--update-goldens")
    recorded: dict[str, float] = {}

    yield (recorded if updating else None)

    if updating and recorded:
        section = goldens_store.record(GOLDENS_PATH, recorded, GOLDENS_PLATFORM_SPECIFIC)
        print(f"\nwrote {len(recorded)} goldens to {GOLDENS_PATH} [{section}]")


def pytest_generate_tests(metafunc):
    """Parametrize over the manifest, honouring --include-expensive."""
    if "benchmark" not in metafunc.fixturenames:
        return
    include_expensive = metafunc.config.getoption("--include-expensive")
    entries = selected(include_expensive=include_expensive)
    metafunc.parametrize("benchmark", entries, ids=[e["name"] for e in entries])
