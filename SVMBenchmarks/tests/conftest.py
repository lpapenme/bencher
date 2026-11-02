"""Fixtures for SVMBenchmarks' tests.

Anything that calls evaluate() needs the CT-slice dataset (~200 MB), so those
cases carry @pytest.mark.dataset and are skipped unless $SVM_DATA_DIR holds it,
as it does in the image. Without that variable the service downloads into a
fresh temp directory, so there is nothing to find. The hyperparameter mapping and dispatch need no data and run
everywhere.
"""
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from svmbenchmarks.main import SvmServiceServicer

import datasets  # noqa: E402


def pytest_collection_modifyitems(config, items):
    data_dir = os.environ.get("SVM_DATA_DIR")
    if not data_dir:
        missing = ["$SVM_DATA_DIR is unset"]
    else:
        missing = datasets.absent(pathlib.Path(data_dir) / name
                                  for name in ("CT_slice_X.npy", "CT_slice_y.npy"))
    datasets.skip_unless_present(items, missing)


@pytest.fixture(scope="module")
def servicer():
    return SvmServiceServicer()
