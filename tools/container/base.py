# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HPCCM recipe for monoprop's MPI base image.

Render it with ``uvx hpccm==26.5.0 --recipe tools/container/base.py --format docker``.
container.yml tags the image with a hash of that output, so changing anything here (or
the hpccm version) produces a new base image.

The compilers, Python, Boost and hwloc come from Ubuntu. The communication path is
pinned: DOCA-OFED's InfiniBand user space, and UCX, PMIx and Open MPI built from source.
Ranks are launched by Slurm with ``srun --mpi=pmix``, so no PMI2 library is built.
"""

# hpccm runs this file with ``Stage0`` predefined.
# ruff: noqa: INP001, F821

from __future__ import annotations

from hpccm.building_blocks import gnu, openmpi, packages, pmix, python, ucx
from hpccm.primitives import baseimage, comment, environment, label, shell

UBUNTU_IMAGE = "ubuntu:26.04"
DOCA_OFED_VERSION = "3.5.0"
UCX_VERSION = "1.22.0"
PMIX_VERSION = "5.0.11"
OPENMPI_VERSION = "5.0.11"

UCX_PREFIX = "/usr/local/ucx"
PMIX_PREFIX = "/usr/local/pmix"

Stage0 += baseimage(image=UBUNTU_IMAGE, _distro="ubuntu26")
Stage0 += label(
    metadata={
        "org.opencontainers.image.source": "https://github.com/Algorithmiq/monoprop",
        "org.opencontainers.image.description": '"MPI base image (UCX, PMIx, Open MPI) for monoprop"',
    }
)

compiler = gnu(fortran=False)
Stage0 += compiler
Stage0 += python(python2=False, devel=True)
Stage0 += packages(
    apt=[
        "libboost-dev",
        "libevent-dev",
        "libhwloc-dev",
        "libnuma-dev",
        "pkg-config",
        "python3-venv",
    ]
)

# DOCA-OFED's InfiniBand user space. hpccm's doca_ofed block cannot be used on Ubuntu 26.04:
# it trusts the legacy Mellanox key, which does not sign the 3.5.0 repository (the
# repository ships its own doca_keyring.gpg), and apt 3 on 26.04 aborts the TLS download
# of the repository because linux.mellanox.com closes connections without a close_notify.
# The repository stays signed, so apt fetches it over plain HTTP.
DOCA_REPOSITORY = (
    f"linux.mellanox.com/public/repo/doca/{DOCA_OFED_VERSION}/ubuntu26.04/x86_64"
)
DOCA_KEYRING = "/usr/share/keyrings/doca_keyring.gpg"
DOCA_PACKAGES = (
    "ibverbs-providers ibverbs-utils libibmad-dev libibmad5 libibumad-dev libibumad3"
    " libibverbs-dev libibverbs1 librdmacm-dev librdmacm1"
)
Stage0 += comment(f"DOCA OFED version {DOCA_OFED_VERSION}")
Stage0 += packages(ospackages=["ca-certificates", "wget"])
Stage0 += shell(
    commands=[
        f"wget -qO {DOCA_KEYRING} https://{DOCA_REPOSITORY}/doca_keyring.gpg",
        f'echo "deb [signed-by={DOCA_KEYRING}] http://{DOCA_REPOSITORY}/ ./" > /etc/apt/sources.list.d/doca.list',
        "apt-get update -y",
        f"DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends {DOCA_PACKAGES}",
        "rm -rf /var/lib/apt/lists/*",
    ]
)

Stage0 += ucx(
    version=UCX_VERSION,
    prefix=UCX_PREFIX,
    ofed=True,
    enable_mt=True,
    cuda=False,
    gdrcopy=False,
    knem=False,
    xpmem=False,
    toolchain=compiler.toolchain,
)
Stage0 += pmix(version=PMIX_VERSION, prefix=PMIX_PREFIX, toolchain=compiler.toolchain)
# Slurm may offer munge authentication to the PMIx client, which is built without it; the
# client then warns before falling back to native. Ask for native directly.
Stage0 += environment(variables={"PMIX_MCA_psec": "native"})
Stage0 += openmpi(
    version=OPENMPI_VERSION,
    pmix=PMIX_PREFIX,
    ucx=UCX_PREFIX,
    cuda=False,
    with_slurm=True,
    toolchain=compiler.toolchain,
)
