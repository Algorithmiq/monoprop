#!/bin/bash
# Fan the Fig. 6 octave sweep out across a Slurm partition, one exclusive node per work item.
#
#     scripts/slurm/submit_octave.sh /path/to/site.env [layers=7]
#
# ppvm N=1024 goes first: at ~47 min it is the long pole and sets the wall clock. N <= 128 is
# grouped per engine (seconds of work, not worth a node boot each). PauliPropagation.jl has no
# N=1024 item: at seven layers that single point extrapolates to days. When every job is
# done, merge the per-job files into data/ -- see the README.
set -euo pipefail
SITE="$(realpath "${1:?usage: submit_octave.sh <site.env> [layers]}")"
LAYERS="${2:-7}"
# shellcheck source=/dev/null
source "$SITE"
: "${PARTITION:?} ${RUN_DIR:?}"
PLOTS="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$RUN_DIR/logs"

sub() {
    local name="$1"
    shift
    printf '%-12s ' "$name"
    sbatch --partition="$PARTITION" --job-name="$name" --chdir="$PLOTS" \
        --output="$RUN_DIR/logs/%x-%j.out" --error="$RUN_DIR/logs/%x-%j.err" \
        "$PLOTS/scripts/slurm/octave_point.sbatch" "$SITE" "$LAYERS" "$@"
}

sub ppvm-1024 ppvm 1024
sub ppvm-512 ppvm 512
sub ppvm-256 ppvm 256
sub ppvm-lo ppvm 32 64 128
sub julia-512 julia 512
sub julia-256 julia 256
sub julia-lo julia 32 64 128
sub mono-1024 monoprop 1024
sub mono-512 monoprop 512
sub mono-256 monoprop 256
sub mono-lo monoprop 32 64 128
