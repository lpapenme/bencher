"""Bake the MOPTA executable into the image (NoDependencyBenchmark venv).

Invoked with MOPTA_DATA_DIR pointed at a cache-mounted directory; argv[1] is the
real destination. Must run on the image's target architecture, since which
binary is fetched depends on it.
"""
import shutil
import sys
from pathlib import Path

from nodependencybenchmark.main import (directory_name, download_mopta_executable,
                                        mopta_executable_basename)

download_mopta_executable(mopta_executable_basename())

destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
shutil.copytree(directory_name, destination, dirs_exist_ok=True)
print(f"mopta executable -> {destination}")
