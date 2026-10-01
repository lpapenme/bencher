"""Fixtures for SVMBenchmarks' tests.

Anything that calls evaluate() needs the CT-slice dataset (~200 MB), so those
cases carry @pytest.mark.dataset and run only where it is baked in -- the
container tier. The hyperparameter mapping and dispatch need no data and run
everywhere.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from svmbenchmarks.main import SvmServiceServicer


@pytest.fixture(scope="module")
def servicer():
    return SvmServiceServicer()
