# Stage 1: The Builder
# This stage installs all dependencies and prepares the application.
FROM python:3.11-slim-bookworm AS builder

# Set environment variables consistently across stages
# Using ${VAR:-} syntax to avoid warnings if VAR is unset.
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:/root/.local/bin:$PATH"
ENV LANG=C.UTF-8 \
    MUJOCO_PY_MUJOCO_PATH=/opt/mujoco210 \
    LD_LIBRARY_PATH="/opt/mujoco210/bin:/bin/usr/local/nvidia/lib64:/usr/lib/nvidia:${LD_LIBRARY_PATH-}" \
    LIBSVMDATA_HOME=/tmp \
    SUMO_HOME=/usr/share/sumo

# Group all dependency ARGs at the top for clarity
ARG PPA_DEPENDENCIES="software-properties-common python3-launchpadlib gnupg"
# Added git for pyenv installation
ARG RUNTIME_DEPENDENCIES="git curl g++ build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl \
    libglew-dev patchelf python3-dev libglfw3 gcc libosmesa6-dev libgl1-mesa-glx sumo sumo-tools swig"

# This entire layer for system dependencies will be cached after the first successful run.
# Note: Removed undefined $BUILD_DEPENDENCIES variable from the original file
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends $PPA_DEPENDENCIES && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends $RUNTIME_DEPENDENCIES && \
    rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash

WORKDIR /opt

# These layers for external tools are also highly cacheable.
RUN curl -LO https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
    tar -xf mujoco210-linux-x86_64.tar.gz && \
    rm mujoco210-linux-x86_64.tar.gz

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy all your code at once. It's less granular for caching but guaranteed to work.
COPY . /opt/bencher
WORKDIR /opt/bencher

# Install Python dependencies using pyenv and uv.
# This loop checks each subdirectory for a .python-version file.
# If found, it installs that Python version with pyenv and uses it to install dependencies with uv.
# Mount pyenv's cache directory and uv's cache directory for faster subsequent builds.
RUN --mount=type=cache,target=/root/.pyenv/cache \
    --mount=type=cache,target=/root/.cache \
    for dir in /opt/bencher/*; do \
        if [ -d "$dir" ] && [ -f "$dir/pyproject.toml" ]; then \
            cd "$dir"; \
            if [ -f ".python-version" ]; then \
                PYTHON_VERSION=$(cat .python-version | tr -d '[:space:]'); \
                echo "Found .python-version in $(basename $dir), requires Python $PYTHON_VERSION"; \
                # Check if the required Python version is already installed before trying to install
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


# Make the entrypoint executable
COPY entrypoint.py /entrypoint.py
RUN chmod +x /entrypoint.py

# ---
# Stage 2: The Final Image
# This stage creates the final, small runtime image.
FROM python:3.11-slim-bookworm AS final

# Re-establish environment variables and set up the pyenv environment
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:/root/.local/bin:$PATH"
ENV LANG=C.UTF-8 \
    MUJOCO_PY_MUJOCO_PATH=/opt/mujoco210 \
    LD_LIBRARY_PATH="/opt/mujoco210/bin:/bin/usr/local/nvidia/lib64:/usr/lib/nvidia:${LD_LIBRARY_PATH-}" \
    LIBSVMDATA_HOME=/tmp \
    SUMO_HOME=/usr/share/sumo
ENV UV_CACHE_DIR=/tmp/.uv-cache \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1

ARG PPA_DEPENDENCIES="software-properties-common python3-launchpadlib gnupg"
# Added git as a pyenv runtime dependency
ARG RUNTIME_DEPENDENCIES="git curl g++ build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl \
    libglew-dev patchelf python3-dev libglfw3 gcc libosmesa6-dev libgl1-mesa-glx sumo sumo-tools swig"


# Install only the necessary RUNTIME and PPA dependencies.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends $PPA_DEPENDENCIES && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends $RUNTIME_DEPENDENCIES && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-built artifacts from the 'builder' stage
COPY --from=builder /opt/mujoco210 /opt/mujoco210
COPY --from=builder /root/.local/bin/uv /root/.local/bin/uv
# Copy the entire pyenv installation, including all installed Python versions
COPY --from=builder /root/.pyenv /root/.pyenv
COPY --from=builder /opt/bencher /opt/bencher
COPY --from=builder /entrypoint.py /entrypoint.py

WORKDIR /opt/bencher
EXPOSE 50051
ENTRYPOINT ["python3.11", "/entrypoint.py"]