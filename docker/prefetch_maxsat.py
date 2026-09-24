"""Bake the MaxSAT corpora into the image (run with the MaxSATBenchmarks venv).

argv[1] is a download cache directory, which the Dockerfile backs with a cache
mount so the corpora are not re-downloaded when this layer is invalidated. The
final destination is whatever the package itself says it reads from.
"""
import shutil
import sys
from pathlib import Path

from maxsatbenchmarks.data_loading import download_maxsat125_data, download_maxsat60_data
from maxsatbenchmarks.main import directory_name

cache = Path(sys.argv[1])
cache.mkdir(parents=True, exist_ok=True)
download_maxsat60_data(str(cache))
download_maxsat125_data(str(cache))

destination = Path(directory_name)
destination.mkdir(parents=True, exist_ok=True)
shutil.copytree(cache, destination, dirs_exist_ok=True)
print(f"maxsat corpora -> {destination}")
