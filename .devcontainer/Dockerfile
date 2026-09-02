FROM ghcr.io/astral-sh/uv:latest AS uv

FROM mcr.microsoft.com/devcontainers/base:ubuntu26.04

COPY tools/packages/apt.txt tools/packages/apt-mpi.txt /tmp/packages/

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && sed -e 's/#.*//' -e 's/[[:space:]]*$//' /tmp/packages/*.txt \
    | grep -v '^$' \
    | xargs apt-get -y install \
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
    && rm -rf /tmp/packages \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv --chown=vscode: /uv /uvx /bin/
