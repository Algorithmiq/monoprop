#!/usr/bin/env bash
# Checks a built monoprop image without executing monoprop code: the image is compiled
# for -march=$MARCH, which the machine running the check (a CI runner) may not implement.
#
# Usage, inside the image: smoke-test.sh <expected monoprop version>

set -euo pipefail

if (( $# != 1 )); then
    echo "usage: $0 <expected monoprop version>" >&2
    exit 2
fi
expected_version="$1"

# The checks run as root in a throwaway container.
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

fail() {
    echo "smoke test failed: $*" >&2
    exit 1
}

echo "== monoprop distribution"
installed_version="$(python3 -c 'import importlib.metadata as m; print(m.version("monoprop"))')"
[[ "$installed_version" == "$expected_version" ]] \
    || fail "expected monoprop $expected_version, found $installed_version"
python3 -m pip check

echo "== shared libraries"
# find_spec locates the package without importing it.
package_dir="$(python3 -c 'import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec("monoprop").origin).parent)')"
mapfile -t libraries < <(find "$package_dir" -type f -name '*.so*')
(( ${#libraries[@]} > 0 )) || fail "no shared libraries under $package_dir"
all_dependencies=""
for library in "${libraries[@]}"; do
    dependencies="$(ldd "$library")"
    if grep -q 'not found' <<< "$dependencies"; then
        echo "$dependencies" >&2
        fail "unresolved dependency in $library"
    fi
    # Collected in a variable: grep -q on a pipe makes ldd die of SIGPIPE, failing pipefail.
    all_dependencies+="$dependencies"$'\n'
done
grep -q 'libmpi\.so' <<< "$all_dependencies" || fail "monoprop is not linked against libmpi"

echo "== MPI library loaded by mpi4py"
ompi_details="$(ompi_info)"
ompi_version="$(awk '/^ *Open MPI:/ {print $3; exit}' <<< "$ompi_details")"
[[ -n "$ompi_version" ]] || fail "ompi_info reported no version"
mpi4py_library="$(python3 -c 'import mpi4py; mpi4py.rc.initialize = False; from mpi4py import MPI; print(MPI.Get_library_version().replace(chr(0), ""))')"
echo "$mpi4py_library"
[[ "$mpi4py_library" == *"Open MPI v${ompi_version}"* ]] \
    || fail "mpi4py did not load the image's Open MPI v$ompi_version"

echo "== Open MPI build"
grep -q 'MCA pml: ucx' <<< "$ompi_details" || fail "Open MPI was built without the UCX PML"
# The base image builds PMIx into /usr/local/pmix; srun --mpi=pmix talks to it.
mpi_dependencies="$(ldd "$(dirname "$(command -v ompi_info)")/../lib/libmpi.so")"
grep -q '/usr/local/pmix/' <<< "$mpi_dependencies" \
    || fail "Open MPI is not linked against the base image's PMIx"

echo "== two-rank MPI job"
mpiexec -n 2 --map-by :OVERSUBSCRIBE python3 -c '
from mpi4py import MPI
comm = MPI.COMM_WORLD
assert comm.Get_size() == 2, comm.Get_size()
assert comm.allreduce(1) == 2
'

echo "smoke test passed"
