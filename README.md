# monoprop

[![Documentation](https://github.com/Algorithmiq/monoprop/actions/workflows/docpages.yml/badge.svg)](https://docs.algorithmiq.fi/monoprop)
[![Test monoprop](https://github.com/Algorithmiq/monoprop/actions/workflows/test.yml/badge.svg)](https://github.com/Algorithmiq/monoprop/actions/workflows/test.yml)

[![Quality gate](https://sonarcloud.io/api/project_badges/quality_gate?project=Algorithmiq_monoprop&token=4639fc1ce1c87d33bf3da90eaa516f2ee77dabb0)](https://sonarcloud.io/summary/new_code?id=Algorithmiq_monoprop)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Algorithmiq_monoprop&metric=coverage&token=4639fc1ce1c87d33bf3da90eaa516f2ee77dabb0)](https://sonarcloud.io/summary/new_code?id=Algorithmiq_monoprop)


## Setting up the development environment

The code has a [DevContainer] configuration that will get you up and running
with all dependencies installed and configured, including sane defaults for the
editor.

You will need:

1. A working [Docker] installation:
   - For macOS and Windows, install [Docker Desktop](https://docs.docker.com/get-docker/)
   - For Linux, install [Docker Engine](https://docs.docker.com/engine/install/#server) following the instructions for your specific distro.
2. The [Visual Studio Code] editor. A recent version is recommended, _e.g._ >=1.78
3. The VSCode [DevContainers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
4. (Optional, but **highly recommended**) The [GitHub CLI] tool.

You can clone the repository with:

```
git clone https://github.com/Algorithmiq/monoprop.git
```

We recommend using a Git credential manager, such as [GitHub CLI], configured to
use HTTPS as protocol for Git operations.

Once the code is locally available, you can open its containing folder in
[Visual Studio Code]. The editor will then set up the [DevContainer] for you.
The first time you open the folder the startup will take a few minutes. Once the
process is done, you will have _all_ project dependencies installed, including
`prek` hooks.
[Visual Studio Code] will be already configured with all the extensions helpful for Python development.

**Note** that the order in which Visual Studio Code loads the extensions in the
DevContainer is non-deterministic.  You might have to execute the *Reload
Window* command to get everything to work as expected after a fresh build of the
container.

## Documentation

We use [Sphinx] to generate our documentation pages. You can find the latest version [at this link].

The documentation is now organized by reader task:

- start with [docs/getting-started.rst](docs/getting-started.rst) to choose the right path
- use [docs/quickstart.rst](docs/quickstart.rst) for the fastest install-to-first-run path
- use [docs/user-guide.rst](docs/user-guide.rst) for public workflows
- use [docs/concepts.rst](docs/concepts.rst) and [docs/internals.rst](docs/internals.rst) for runtime design and implementation details

We encourage you to build the documentation locally, so you can check that newer
documentation you might have added looks as it should.

To do so, open a terminal in [Visual Studio Code] and run:

```
just build-docs
```

You can then see the generated website by opening the file `build/docs/html/index.html` with your browser.


[DevContainer]: https://containers.dev/
[Docker]: https://docs.docker.com/get-docker/
[Visual Studio Code]: https://code.visualstudio.com/
[GitHub CLI]: https://cli.github.com/
[Sphinx]: https://www.sphinx-doc.org/en/master/index.html
[at this link]: https://docs.algorithmiq.fi/monoprop

## Quickstart tutorial: use, test, run

The DevContainer ships with everything pre-installed, so you only need a few commands to get productive.

### 1. Install & build the project

```bash
uv sync --all-groups --all-extras -v
```

`uv` creates the virtual environment, installs Python dependencies (including `mpi4py`), and configures scikit-build so the nanobind extension is compiled with MPI support in editable mode. Re-run this command whenever the dependency graph or C++ bindings change. To opt out of MPI (for example on platforms without an MPI toolchain), append `--config-settings-package="monoprop:cmake.define.monoprop_ENABLE_MPI=OFF"`.

### 2. Use the Python API

```python
#TODO: code example
```

### 3. Test the codebase

- **Python tests** (default serial run):

   ```bash
   uv run pytest -v
   ```

- **Python tests under MPI** (full suite, pytest-mpi enabled):

   ```bash
   mpiexec --allow-run-as-root -n 2 /home/vscode/.venv/bin/python -m pytest tests --with-mpi -v
   ```

- **Python MPI marker rank matrix** (pytest-mpi marker coverage):

   ```bash
   for r in 1 2 4; do
       mpiexec --allow-run-as-root -n "$r" /home/vscode/.venv/bin/python -m pytest tests --with-mpi -m mpi -v
   done
   ```

- **C++ unit tests**:

   ```bash
   cmake --preset release-gcc
   cmake --build --preset release-gcc --target unit_tests.x
   ctest --preset release-gcc -R unit_tests
   ```

   In VS Code you can switch the CMake preset between `release-gcc` (MPI disabled) and `release-gcc-mpi` (same flags with `monoprop_ENABLE_MPI=ON`) to make the appropriate test set appear in the CTest panel.

- **Both MPI configurations via CTest** (Release builds for `monoprop_ENABLE_MPI=OFF` and `ON`):

   ```bash
   ctest -S tools/ctest-mpi-matrix.cmake -VV
   ```

   The script configures out-of-source builds under `build/nompi` and `build/mpi`, builds `unit_tests.x` in each, and executes the corresponding `ctest` suites (including `unit_tests_mpi_pare` when MPI is enabled).

### 4. Run the distributed example

```bash
cmake --preset release-gcc-mpi
cmake --build --preset release-gcc-mpi --target example.x
mpiexec -n 4 ./build/release-gcc-mpi/bin/example.x lih_fermionic_spin_exact tests/data
```

This mirrors the Python workflow: the executable loads MsgPack reference data, evolves the distributed operator, and prints energies/gradients per rank. Adjust `-n` to scale across more ranks and export `monoprop_REF_DATA_PATH` if you want to avoid passing the data directory.

For runtime experiments, the example also accepts `--pare-threshold VALUE` to enable masked exact paring during functional setup and `--track-memory` to print current and peak RSS across ranks. The same options are available through `PARE_THRESHOLD` and `TRACK_MEMORY`.

## MPI tutorial (C++ & Python)

The repository ships with a complete MPI toolchain in the devcontainer. If you are building outside the container, make sure OpenMPI (or an MPI implementation compatible with `mpiexec`) is available together with a C++23 compiler and Python 3.11+.

### C++ workflow

1. Configure and build the MPI preset:

   ```bash
   cmake --preset release-gcc-mpi
   cmake --build --preset release-gcc-mpi --target example.x
   ```

2. Run the example across `N` ranks. The first argument is the molecule tag, the second is the folder containing the MsgPack reference data. If you omit the path, the binary looks for `monoprop_REF_DATA_PATH`.

   ```bash
   mpiexec -n 4 ./build/release-gcc-mpi/bin/example.x S0_14e14o_majoranic_c6 tests/data
   ```

   The program prints the parallel runtime configuration, evolution wall times, energies, gradients, and the distributed operator size so you can confirm the run is balanced. Use the regular CMake presets (for example `release-gcc`) when you do not need MPI.

### Python workflow

1. Install the Python package (and the compiled extension) with [uv](https://github.com/astral-sh/uv):

   ```bash
   uv sync --all-groups --all-extras -v
   ```

   The default build enables MPI so the Python and C++ layers share the same communicator handling. Pass `--config-settings=cmake.define.ENABLE_MPI=OFF` only if you explicitly need a non-MPI build.

2. Create a small driver script (for example `scripts/run_mpi_sim.py`) that loads MsgPack data, instantiates the simulator, and evaluates an observable:

   ```python
   #TODO: code example
   ```

3. Launch the script with MPI. `uv run` ensures the virtual environment is active on every rank:

   ```bash
   mpiexec -n 4 uv run python scripts/run_mpi_sim.py
   ```

   The Python API shares the same C++ backend, so the MPI behaviour mirrors the C++ example. You can adjust cutoffs, tolerances, or basis changes in pure Python before handing work to the nanobind bindings.

### Tips

- Set `NUM_THREADS` explicitly when benchmarking so each MPI rank uses a predictable number of worker threads.
- Large datasets live under `tests/data/`; point the executables or scripts at that directory or export `monoprop_REF_DATA_PATH`.
- When profiling MPI runs, prefer Release builds (`--preset release-gcc-mpi`) to match production flags.


## CI and Testing Requirements

The GitHub Actions CI automatically tests the codebase with MPI enabled. If you're running tests locally or contributing to CI configuration, ensure the following dependencies are installed:

### Ubuntu/Debian Systems

```bash
sudo apt-get update
sudo apt-get install -y libopenmpi-dev openmpi-bin pkg-config
```

These packages provide:
- `libopenmpi-dev`: MPI C++ development headers and libraries
- `openmpi-bin`: OpenMPI runtime binaries (including `mpiexec`)
- `pkg-config`: Required for CMake to locate MPI packages

After installing MPI, you can build and test with:

```bash
# Build with MPI enabled (default)
uv sync --all-groups --all-extras -v

# Run tests
uv run pytest -v

# Run full suite with MPI
mpiexec --allow-run-as-root -n 2 /home/vscode/.venv/bin/python -m pytest tests --with-mpi -v

# Run MPI-marker tests across rank matrix
for r in 1 2 4; do
   mpiexec --allow-run-as-root -n "$r" /home/vscode/.venv/bin/python -m pytest tests --with-mpi -m mpi -v
done
```

To build **without** MPI (e.g., on systems where MPI is unavailable):

```bash
uv sync --all-groups --all-extras -v --config-settings=cmake.define.ENABLE_MPI=OFF
```


## Using the Linaro-MAP profiler
- Log into polaris using (if you want to use the GUI log in outside)
```
ssh -X -Y polaris.algorithmiq.fi
```
- Run this (potentially there might be a new version so be careful)
```
. /opt/intel/oneapi/2025.3/oneapi-vars.sh
```
- You may have to install a version of cmake
```
~/cmake-4.0.2-linux-x86_64/bin/cmake -S. -Bbuild -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=icpx
```
- There are a variety of ways to use the profiler:
```
# MAP Profiling Commands
# Profile with command-line MAP (no GUI needed)
/opt/linaro/forge/24.1.2/bin/map --profile --output=monoprop_profile.map ./build/bin/example.x

# Generate HTML report from profile
/opt/linaro/forge/24.1.2/bin/map --export=monoprop_profile.html monoprop_profile.map

# Try to launch MAP GUI
/opt/linaro/forge/24.1.2/bin/map &
```
