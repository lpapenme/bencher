# Stage 1: The Builder
# This stage installs all dependencies and prepares the application.
FROM python:3.11-slim-bookworm AS builder

# Set environment variables consistently across stages
# Using ${VAR:-} syntax to avoid warnings if VAR is unset.
ENV LANG=C.UTF-8 \
    PATH="/root/.local/bin:$PATH" \
    MUJOCO_PY_MUJOCO_PATH=/opt/mujoco210 \
    LD_LIBRARY_PATH="/opt/mujoco210/bin:/bin/usr/local/nvidia/lib64:/usr/lib/nvidia:${LD_LIBRARY_PATH-}" \
    LIBSVMDATA_HOME=/tmp \
    SUMO_HOME=/usr/share/sumo

# Group all dependency ARGs at the top for clarity
ARG PPA_DEPENDENCIES="software-properties-common python3-launchpadlib gnupg"
ARG BUILD_DEPENDENCIES="git curl g++ build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl swig \
    libglew-dev patchelf python3-dev"
ARG RUNTIME_DEPENDENCIES="libglfw3 gcc libosmesa6-dev libgl1-mesa-glx sumo sumo-tools"

# This entire layer for system dependencies will be cached after the first successful run.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends $PPA_DEPENDENCIES && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends $BUILD_DEPENDENCIES $RUNTIME_DEPENDENCIES && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# These layers for external tools are also highly cacheable.
RUN curl -LO https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz && \
    tar -xf mujoco210-linux-x86_64.tar.gz && \
    rm mujoco210-linux-x86_64.tar.gz

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy all your code at once. It's less granular for caching but guaranteed to work.
COPY . /opt/bencher
WORKDIR /opt/bencher

# Install Python dependencies.
RUN --mount=type=cache,target=/root/.cache \
    for dir in /opt/bencher/*; do \
        if [ -d "$dir" ] && [ -f "$dir/pyproject.toml" ]; then \
            cd "$dir" && uv sync && cd ..; \
        fi; \
    done

# Make the entrypoint executable
COPY entrypoint.py /entrypoint.py
RUN chmod +x /entrypoint.py

# ---
# Stage 2: The Final Image
# This stage creates the final, small runtime image.
FROM python:3.11-slim-bookworm AS final

# Re-establish environment variables
ENV LANG=C.UTF-8 \
    PATH="/root/.local/bin:$PATH" \
    MUJOCO_PY_MUJOCO_PATH=/opt/mujoco210 \
    LD_LIBRARY_PATH="/opt/mujoco210/bin:/bin/usr/local/nvidia/lib64:/usr/lib/nvidia:${LD_LIBRARY_PATH-}" \
    LIBSVMDATA_HOME=/tmp \
    SUMO_HOME=/usr/share/sumo

ARG PPA_DEPENDENCIES="software-properties-common python3-launchpadlib gnupg"
ARG RUNTIME_DEPENDENCIES="libglfw3 gcc libosmesa6-dev libgl1-mesa-glx sumo sumo-tools"

# Install only the necessary RUNTIME and PPA dependencies.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends $PPA_DEPENDENCIES && \
    add-apt-repository -y ppa:sumo/stable && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends $RUNTIME_DEPENDENCIES && \
    rm -rf /var/lib/apt/lists/*

# Copy pre-built artifacts from the 'builder' stage
COPY --from=builder /opt/mujoco210 /opt/mujoco210
# NEW: Copy the uv binary to the final image
COPY --from=builder /root/.local/bin/uv /root/.local/bin/uv
COPY --from=builder /opt/bencher /opt/bencher
COPY --from=builder /entrypoint.py /entrypoint.py

WORKDIR /opt/bencher
EXPOSE 50051
ENTRYPOINT ["python3.11", "/entrypoint.py"]