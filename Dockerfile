# Stage 1: The Builder
FROM debian:bookworm-slim AS builder

# --- CHANGE 1: Move pyenv out of /root ---
# We use /opt/pyenv so it is globally readable by any user (Docker or Apptainer)
ENV PYENV_ROOT="/opt/pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:/root/.local/bin:$PATH"
ENV LANG=C.UTF-8 \
    MUJOCO_PY_MUJOCO_PATH=/opt/mujoco210 \
    LD_LIBRARY_PATH="/opt/mujoco210/bin:/bin/usr/local/nvidia/lib64:/usr/lib/nvidia:${LD_LIBRARY_PATH-}" \
    LIBSVMDATA_HOME=/tmp \
    SUMO_HOME=/usr/share/sumo

# ... (Arg definitions and apt-get installs remain the same) ...
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

# Install pyenv
RUN curl https://pyenv.run | bash

WORKDIR /opt

# ... (Mujoco install remains the same) ...
RUN curl -LO https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
    tar -xf mujoco210-linux-x86_64.tar.gz && \
    rm mujoco210-linux-x86_64.tar.gz

# --- CHANGE 2: Install uv to a global location ---
# By default uv installs to ~/.local/bin. We force it to /usr/local/bin
ENV UV_INSTALL_DIR="/usr/local/bin"
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY . /opt/bencher
WORKDIR /opt/bencher

# --- CHANGE 3: Update Cache Mounts ---
# Update cache mounts to point to the new location or keep them in root (caches are fine in root if only used during build)
# Note: We must ensure the permissions of /opt/pyenv allow reading by others
RUN --mount=type=cache,target=/root/.pyenv/cache \
    --mount=type=cache,target=/root/.cache \
    for dir in /opt/bencher/*; do \
        if [ -d "$dir" ] && [ -f "$dir/pyproject.toml" ]; then \
            cd "$dir"; \
            if [ -f ".python-version" ]; then \
                PYTHON_VERSION=$(cat .python-version | tr -d '[:space:]'); \
                echo "Found .python-version in $(basename $dir), requires Python $PYTHON_VERSION"; \
                if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then \
                    echo "Installing Python $PYTHON_VERSION..."; \
                    pyenv install $PYTHON_VERSION; \
                fi; \
                echo "Installing dependencies for $(basename $dir) with Python $PYTHON_VERSION..."; \
                PYENV_VERSION=$PYTHON_VERSION uv sync --frozen --compile-bytecode --no-dev; \
            else \
                echo "No .python-version found in $(basename $dir), using default system python (3.11)..."; \
                uv sync --frozen --compile-bytecode --no-dev; \
            fi; \
            cd ..; \
        fi; \
    done

# Pre-fetch MaxSAT datasets into the package data directory
RUN cd /opt/bencher/MaxSATBenchmarks && \
    uv run python - <<'PY'
from maxsatbenchmarks.main import directory_name
from maxsatbenchmarks.data_loading import download_maxsat60_data, download_maxsat125_data

download_maxsat60_data(directory_name)
download_maxsat125_data(directory_name)
PY

# --- CHANGE 4: Permission Fix ---
# Crucial for Apptainer: Ensure /opt/pyenv is readable by non-root users
RUN chmod -R a+rX /opt/pyenv

COPY entrypoint.py /entrypoint.py
RUN chmod +x /entrypoint.py

# ---
# Stage 2: The Final Image
FROM debian:bookworm-slim AS final

# --- CHANGE 5: Environment variables in Final Stage ---
ENV PYENV_ROOT="/opt/pyenv"
# Removed /root/.local/bin from PATH as we moved uv to /usr/local/bin
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
ENV LANG=C.UTF-8 \
    MUJOCO_PY_MUJOCO_PATH=/opt/mujoco210 \
    LD_LIBRARY_PATH="/opt/mujoco210/bin:/bin/usr/local/nvidia/lib64:/usr/lib/nvidia:${LD_LIBRARY_PATH-}" \
    LIBSVMDATA_HOME=/tmp \
    SUMO_HOME=/usr/share/sumo
ENV UV_CACHE_DIR=/tmp/.uv-cache \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1

# ... (Runtime dependency install remains the same) ...
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

COPY --from=builder /opt/mujoco210 /opt/mujoco210

# --- CHANGE 6: Copy uv from /usr/local/bin ---
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

# --- CHANGE 7: Copy pyenv to /opt/pyenv ---
COPY --from=builder /opt/pyenv /opt/pyenv

COPY --from=builder /opt/bencher /opt/bencher
COPY --from=builder /entrypoint.py /entrypoint.py

# Pre-fetch libsvm datasets into /tmp to avoid download at runtime using LassoBenchmarks env
RUN cd /opt/bencher/LassoBenchmarks && \
    LIBSVMDATA_HOME=/tmp uv run python - <<'PY'
from libsvmdata import fetch_libsvm
for name in ["diabetes_scale", "breast-cancer_scale", "leukemia_test", "rcv1.binary", "dna"]:
    fetch_libsvm(name)
PY

# --- CHANGE 8: Final Permission sanity check ---
# Just to be absolutely sure permissions didn't get messed up during COPY
RUN chmod -R a+rX /opt/pyenv /opt/bencher

WORKDIR /opt/bencher
EXPOSE 50051
ENTRYPOINT ["python3.11", "/entrypoint.py"]
