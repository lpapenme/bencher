"""Bake the libsvm datasets into the image (run with the LassoBenchmarks venv).

Needs only that package's dependencies, not bencher source, so it can run above
the source copy in the Dockerfile. Destination comes from LIBSVMDATA_HOME.
"""
from libsvmdata import fetch_libsvm

DATASETS = ["diabetes_scale", "breast-cancer_scale", "leukemia_test", "rcv1.binary", "dna"]

for name in DATASETS:
    print(f"fetching {name}...", flush=True)
    fetch_libsvm(name)
