FROM ghcr.io/astral-sh/uv:latest AS uv

FROM mcr.microsoft.com/devcontainers/base:ubuntu26.04

# install needed packages
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install \
    clang \
    clang-format \
    clang-tidy \
    cmake \
    doxygen \
    g++ \
    gcc \
    gdb \
    just \
    lcov \
    libboost-dev \
    libboost-test-dev \
    libhwloc-dev \
    libmsgpack-cxx-dev \
    libopenmpi-dev \
    ninja-build \
    openmpi-bin \
    pkg-config \
    python3 \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv --chown=vscode: /uv /uvx /bin/
