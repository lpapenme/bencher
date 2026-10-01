"""Bake the SVM slice-localization dataset into the image (SVMBenchmarks venv).

Invoked with SVM_DATA_DIR pointed at a cache-mounted directory; argv[1] is the
real destination baked into the image.
"""
import shutil
import sys
from pathlib import Path

from svmbenchmarks.main import directory_name, download_slice_localization_data

download_slice_localization_data()

destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
shutil.copytree(directory_name, destination, dirs_exist_ok=True)
print(f"slice localization data -> {destination}")
