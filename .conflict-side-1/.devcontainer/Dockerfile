FROM ghcr.io/astral-sh/uv:latest AS uv

FROM mcr.microsoft.com/devcontainers/base:ubuntu26.04

COPY tools/install-deps.sh /usr/local/bin/install-deps.sh

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
    ninja-build \
    python3 \
    openmpi-bin \
    libboost-dev \
    libboost-test-dev \
    libmsgpack-cxx-dev \
    libopenmpi-dev \
    && /usr/local/bin/install-deps.sh --skip-boost-unordered --skip-boost-test --skip-msgpack \
    && if [ "$(uname -m)" = "x86_64" ] && [ -f /opt/intel/oneapi/setvars.sh ]; then \
        echo '. /opt/intel/oneapi/setvars.sh > /dev/null 2>&1' >> /etc/zsh/zshrc; \
        echo '. /opt/intel/oneapi/setvars.sh > /dev/null 2>&1' >> /etc/bash.bashrc; \
        fi \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv --chown=vscode: /uv /uvx /bin/
