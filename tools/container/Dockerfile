# syntax=docker/dockerfile:1

# monoprop on top of the MPI base image rendered from tools/container/base.py. The build
# context holds only the sdist and requirements.txt and constraints.txt, both exported
# from uv.lock; container.yml passes BASE_IMAGE (content-addressed) and SDIST.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SDIST
# MareNostrum5's GPP and ACC partitions are both Intel Sapphire Rapids.
ARG MARCH=sapphirerapids

ENV VIRTUAL_ENV=/opt/monoprop
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

# The runtime dependencies are exactly the locked set, the versions CI tests against,
# installed from hash-checked files; monoprop itself then installs with --no-deps, so a
# dependency missing from the lock fails pip check instead of floating to the newest
# release. monoprop.mpi adds mpi4py to the build requirements. PIP_CONSTRAINT is an
# environment variable because pip forwards no constraint option to its isolated build
# environment, where it pins mpi4py too. pip does forward --only-binary there, so a
# missing mpi4py wheel fails the build instead of compiling mpi4py.
RUN --mount=type=bind,source=${SDIST},target=/tmp/${SDIST} \
    --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    --mount=type=bind,source=constraints.txt,target=/tmp/constraints.txt \
    python3 -m venv "${VIRTUAL_ENV}" && \
    pip install --no-cache-dir --only-binary=mpi4py --require-hashes \
        --requirement /tmp/requirements.txt && \
    PIP_CONSTRAINT=/tmp/constraints.txt \
    pip install --no-cache-dir --verbose --no-deps \
        --only-binary=mpi4py \
        --config-settings=cmake.build-type=Release \
        --config-settings=monoprop.mpi=ON \
        --config-settings=monoprop.arch-flags=OFF \
        --config-settings=monoprop.cxx-unit-tests=OFF \
        "--config-settings=cmake.define.EXTRA_CXXFLAGS=-march=${MARCH}" \
        "--config-settings=build.tool-args=-j$(nproc)" \
        "monoprop[mpi,qiskit] @ file:///tmp/${SDIST}"
