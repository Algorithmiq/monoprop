# syntax=docker/dockerfile:1

# monoprop on top of the MPI base image rendered from tools/container/base.py. The build
# context holds only the sdist and constraints.txt (exported from uv.lock); container.yml
# passes BASE_IMAGE (content-addressed) and SDIST.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SDIST
# MareNostrum5's GPP and ACC partitions are both Intel Sapphire Rapids.
ARG MARCH=sapphirerapids

ENV VIRTUAL_ENV=/opt/monoprop
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

# monoprop.mpi also adds mpi4py to the build requirements.
# constraints.txt is exported from uv.lock, so the runtime dependencies are the
# versions CI tests against. Both PIP_* variables also reach pip's isolated build
# environment: mpi4py is built against the locked version, and pip fails instead of
# compiling it when its PyPI wheel is missing.
RUN --mount=type=bind,source=${SDIST},target=/tmp/${SDIST} \
    --mount=type=bind,source=constraints.txt,target=/tmp/constraints.txt \
    python3 -m venv "${VIRTUAL_ENV}" && \
    PIP_CONSTRAINT=/tmp/constraints.txt PIP_ONLY_BINARY=mpi4py \
    pip install --no-cache-dir --verbose \
        --config-settings=cmake.build-type=Release \
        --config-settings=monoprop.mpi=ON \
        --config-settings=monoprop.arch-flags=OFF \
        --config-settings=monoprop.cxx-unit-tests=OFF \
        "--config-settings=cmake.define.EXTRA_CXXFLAGS=-march=${MARCH}" \
        "--config-settings=build.tool-args=-j$(nproc)" \
        "monoprop[mpi,qiskit] @ file:///tmp/${SDIST}"
