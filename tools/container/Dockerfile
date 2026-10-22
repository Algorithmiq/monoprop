# syntax=docker/dockerfile:1

# monoprop on top of the MPI base image rendered from tools/container/base.py. The build
# context holds only the sdist; container.yml passes BASE_IMAGE (content-addressed) and SDIST.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SDIST
# MareNostrum5's GPP and ACC partitions are both Intel Sapphire Rapids.
ARG MARCH=sapphirerapids

ENV VIRTUAL_ENV=/opt/monoprop
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

# monoprop_ENABLE_MPI must come from the environment: it is what activates the
# scikit-build override that adds mpi4py to the build requirements.
# PIP_ONLY_BINARY makes pip, including its isolated build environment, fail instead
# of compiling mpi4py when its PyPI wheel is missing.
RUN --mount=type=bind,source=${SDIST},target=/tmp/${SDIST} \
    python3 -m venv "${VIRTUAL_ENV}" && \
    PIP_ONLY_BINARY=mpi4py monoprop_ENABLE_MPI=ON pip install --no-cache-dir --verbose \
        --config-settings=cmake.build-type=Release \
        --config-settings=cmake.define.monoprop_ENABLE_ARCH_FLAGS=OFF \
        --config-settings=cmake.define.monoprop_ENABLE_CXX_UNIT_TESTS=OFF \
        "--config-settings=cmake.define.EXTRA_CXXFLAGS=-march=${MARCH}" \
        "--config-settings=build.tool-args=-j$(nproc)" \
        "monoprop[mpi,qiskit] @ file:///tmp/${SDIST}"
