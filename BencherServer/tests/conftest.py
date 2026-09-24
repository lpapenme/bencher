"""Fixtures for BencherServer's tests."""
import sys
from pathlib import Path

import pytest

# The shared Tier 2 harness lives at the repo root, the same way
# tests/e2e/conftest.py reaches benchmarks.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from grpc_harness import (EchoSecondLevel, front_door,  # noqa: E402
                          request_for, second_level_service)


@pytest.fixture
def echo_service():
    """Starts a stand-in family service; yields (servicer, host, port)."""
    import contextlib

    @contextlib.contextmanager
    def _start(**kwargs):
        servicer = EchoSecondLevel(**kwargs)
        with second_level_service(servicer) as (port, _stub):
            yield servicer, "127.0.0.1", port

    return _start
