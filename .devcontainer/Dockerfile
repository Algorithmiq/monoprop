FROM ghcr.io/astral-sh/uv:latest AS uv

FROM mcr.microsoft.com/devcontainers/base:ubuntu26.04

# install needed packages
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install \
    clang \
    clang-tidy \
    clang-format \
    cmake \
    doxygen \
    gcc \
    g++ \
    gdb \
    just \
    lcov \
    ninja-build \
    python3 \
    openmpi-bin \
    libboost-dev \
    libboost-test-dev \
    libmsgpack-cxx-dev \
    libopenmpi-dev \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv --chown=vscode: /uv /uvx /bin/
