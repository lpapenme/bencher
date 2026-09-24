# Bencher container.

############################  system  ############################
# OS packages, MuJoCo, uv. Changes only when this file does.
# This used to be duplicated verbatim in a second stage, so the SUMO PPA and the
# full apt install ran twice per build.
FROM debian:bookworm-slim AS system

ENV LANG=C.UTF-8 \
    MUJOCO_PY_MUJOCO_PATH=/opt/mujoco210 \
    LD_LIBRARY_PATH="/opt/mujoco210/bin:/bin/usr/local/nvidia/lib64:/usr/lib/nvidia:${LD_LIBRARY_PATH-}" \
    LIBSVMDATA_HOME=/opt/bencher-cache/libsvm \
    MOPTA_DATA_DIR=/opt/bencher/NoDependencyBenchmark/data \
    SVM_DATA_DIR=/opt/bencher-cache/svm \
    SUMO_HOME=/usr/share/sumo

ARG UV_VERSION="0.9.2"
ARG PPA_DEPENDENCIES="software-properties-common python3-launchpadlib gnupg"
ARG RUNTIME_DEPENDENCIES="git curl g++ build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl \
    libglew-dev patchelf python3-dev libglfw3 gcc libosmesa6-dev libgl1-mesa-glx sumo sumo-tools swig"

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends $PPA_DEPENDENCIES && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends $RUNTIME_DEPENDENCIES && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN curl -LO https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
    tar -xf mujoco210-linux-x86_64.tar.gz && \
    rm mujoco210-linux-x86_64.tar.gz

# uv and its managed interpreters are globally readable for Docker and Apptainer.
ENV UV_INSTALL_DIR="/usr/local/bin"
ENV UV_PYTHON_INSTALL_DIR="/opt/uv-python"
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh

######################  interpreters  ############################
# This layer depends ONLY on the .python-version files, so it survives every
# dependency and source change. Packages are listed explicitly because a glob
# COPY would flatten nine identically-named files into one;
# tests/test_dockerfile.py asserts this list stays in step with the packages
# that actually exist.
FROM system AS interpreters

COPY BO4MobBenchmark/.python-version        /opt/bencher/BO4MobBenchmark/.python-version
COPY BencherServer/.python-version          /opt/bencher/BencherServer/.python-version
COPY EboBenchmarks/.python-version          /opt/bencher/EboBenchmarks/.python-version
COPY IOHBenchmarks/.python-version          /opt/bencher/IOHBenchmarks/.python-version
COPY LassoBenchmarks/.python-version        /opt/bencher/LassoBenchmarks/.python-version
COPY MaxSATBenchmarks/.python-version       /opt/bencher/MaxSATBenchmarks/.python-version
COPY MujocoBenchmarks/.python-version       /opt/bencher/MujocoBenchmarks/.python-version
COPY NoDependencyBenchmark/.python-version  /opt/bencher/NoDependencyBenchmark/.python-version
COPY SVMBenchmarks/.python-version          /opt/bencher/SVMBenchmarks/.python-version

RUN --mount=type=cache,target=/root/.cache \
    for version in $(sort -u /opt/bencher/*/.python-version); do \
        uv python install "$version"; \
    done && \
    chmod -R a+rX "$UV_PYTHON_INSTALL_DIR"

#########################  dependencies  #########################
# Third-party dependencies only -- no project source, so editing a service does
# not rebuild torch, mujoco-py, box2d, celer or GPy. Invalidated only by a
# pyproject.toml or uv.lock change.
FROM interpreters AS dependencies

COPY BO4MobBenchmark/pyproject.toml BO4MobBenchmark/uv.lock              /opt/bencher/BO4MobBenchmark/
COPY BencherServer/pyproject.toml BencherServer/uv.lock                  /opt/bencher/BencherServer/
COPY EboBenchmarks/pyproject.toml EboBenchmarks/uv.lock                  /opt/bencher/EboBenchmarks/
COPY IOHBenchmarks/pyproject.toml IOHBenchmarks/uv.lock                  /opt/bencher/IOHBenchmarks/
COPY LassoBenchmarks/pyproject.toml LassoBenchmarks/uv.lock              /opt/bencher/LassoBenchmarks/
COPY MaxSATBenchmarks/pyproject.toml MaxSATBenchmarks/uv.lock            /opt/bencher/MaxSATBenchmarks/
COPY MujocoBenchmarks/pyproject.toml MujocoBenchmarks/uv.lock            /opt/bencher/MujocoBenchmarks/
COPY NoDependencyBenchmark/pyproject.toml NoDependencyBenchmark/uv.lock  /opt/bencher/NoDependencyBenchmark/
COPY SVMBenchmarks/pyproject.toml SVMBenchmarks/uv.lock                  /opt/bencher/SVMBenchmarks/

RUN --mount=type=cache,target=/root/.cache \
    for dir in /opt/bencher/*; do \
        [ -f "$dir/pyproject.toml" ] || continue; \
        cd "$dir"; \
        version=$(tr -d '[:space:]' < .python-version); \
        echo "Installing dependencies for $(basename $dir) with Python ${version}..."; \
        UV_MANAGED_PYTHON=1 UV_PYTHON_DOWNLOADS=never \
          uv sync --python "$version" --frozen --compile-bytecode --no-dev --no-install-project; \
    done && \
    # Force mujoco-py to compile its cymj extension now. It builds the .so into
    # its own site-packages on first import, which succeeds in Docker's writable
    # layer but fails with EROFS on a read-only Apptainer .sif -- so without this
    # every MuJoCo benchmark works under Docker and dies under Apptainer. Doing
    # it here also keeps the cost off the first request.
    /opt/bencher/MujocoBenchmarks/.venv/bin/python -c "import mujoco_py" && \
    chmod -R a+rX /opt/bencher

##########################  datasets  ############################
# Baked in so nothing is downloaded at runtime. The fetch logic lives in
# docker/*.py rather than inline heredocs: a Dockerfile RUN supports only one
# heredoc, and real files are far easier to read and change.
FROM dependencies AS datasets

COPY docker /opt/bencher-build

# libsvm needs only LassoBenchmarks' dependencies, so it sits above any source.
RUN --mount=type=cache,target=/root/.cache \
    mkdir -p "$LIBSVMDATA_HOME" && \
    /opt/bencher/LassoBenchmarks/.venv/bin/python /opt/bencher-build/prefetch_libsvm.py

# The remaining three read URLs and paths from package code, so they need that
# package's source -- but only these three trees, not the whole repo. Each writes
# into a cache mount first and is then copied into the image, so even when this
# layer is invalidated the bytes (several hundred MB) are not re-downloaded.
COPY MaxSATBenchmarks/src       /opt/bencher/MaxSATBenchmarks/src
COPY SVMBenchmarks/src          /opt/bencher/SVMBenchmarks/src
COPY NoDependencyBenchmark/src  /opt/bencher/NoDependencyBenchmark/src

RUN --mount=type=cache,target=/opt/dl-cache \
    set -eu && \
    PYTHONPATH=/opt/bencher/MaxSATBenchmarks/src \
      /opt/bencher/MaxSATBenchmarks/.venv/bin/python \
      /opt/bencher-build/prefetch_maxsat.py /opt/dl-cache/maxsat && \
    SVM_DATA_DIR=/opt/dl-cache/svm PYTHONPATH=/opt/bencher/SVMBenchmarks/src \
      /opt/bencher/SVMBenchmarks/.venv/bin/python \
      /opt/bencher-build/prefetch_svm.py "/opt/bencher-cache/svm" && \
    MOPTA_DATA_DIR=/opt/dl-cache/mopta PYTHONPATH=/opt/bencher/NoDependencyBenchmark/src \
      /opt/bencher/NoDependencyBenchmark/.venv/bin/python \
      /opt/bencher-build/prefetch_mopta.py "/opt/bencher/NoDependencyBenchmark/data" && \
    chmod -R a+rX /opt/bencher-cache /opt/bencher/MaxSATBenchmarks /opt/bencher/NoDependencyBenchmark

###########################  final  ##############################
# Project source last: a change here re-runs only the project install, which is
# a handful of seconds, plus the permission fixups.
FROM datasets AS final

COPY . /opt/bencher
COPY entrypoint.py /entrypoint.py

# Installs just the projects themselves into the virtualenvs built above.
# umask 022 so the handful of files this writes are world-readable without
# needing another recursive chmod.
RUN --mount=type=cache,target=/root/.cache \
    umask 022 && \
    for dir in /opt/bencher/*; do \
        [ -f "$dir/pyproject.toml" ] || continue; \
        cd "$dir"; \
        version=$(tr -d '[:space:]' < .python-version); \
        UV_MANAGED_PYTHON=1 UV_PYTHON_DOWNLOADS=never \
          uv sync --python "$version" --frozen --compile-bytecode --no-dev; \
    done

# Apptainer runs as a non-root user, so everything must be world-readable. The
# expensive interpreter, virtualenv, and dataset trees are already chmod'ed in
# the stages that create them, and those layers are cached; only the source
# copied just above still needs it. Doing all of it here instead cost 462s on
# every source change -- 96% of the rebuild.
RUN chmod +x /entrypoint.py && \
    find /opt/bencher -name .venv -prune -o -print0 | xargs -0 chmod a+rX

# Runtime-only settings. Deliberately after every build step: setting
# UV_CACHE_DIR earlier would point uv away from the cache mounts above.
ENV UV_CACHE_DIR=/tmp/.uv-cache \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/bencher
EXPOSE 50051
ENTRYPOINT ["/opt/bencher/BencherServer/.venv/bin/python", "/entrypoint.py"]
