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

# Shared across the third-party Julia benchmark scripts in this project (pauli_prop,
# majorana_prop). Included via `include(joinpath(@__DIR__, "..", "bench_common.jl"))`.

if Threads.nthreads() == 1
    @warn "Julia is running with only 1 thread: the peak-memory sampler runs as a background " *
          "task, but a single-threaded Julia can only switch to it between steps, not while the " *
          "propagation call itself is running, so the memory figures will undercount any " *
          "transient spike freed before that call returns. Re-run with `julia --threads=auto` " *
          "(or set JULIA_NUM_THREADS) for accurate peak-memory measurements."
end

# Linux-only: reads the kernel's live RSS for this process directly, in kB.
function current_rss_bytes()::Int
    for line in eachline("/proc/self/status")
        if startswith(line, "VmRSS:")
            return parse(Int, split(line)[2]) * 1024
        end
    end
    return 0
end

# Tracks this process's peak RSS within a resettable window: the kernel's own high-water mark
# (Sys.maxrss()) is monotonic for the whole process and can't be reset per step, so this instead
# polls the *current* RSS from a background task and keeps the max seen since the last reset!.
mutable struct RssPeakSampler
    peak_bytes::Threads.Atomic{Int}
    running::Threads.Atomic{Bool}
end

function start_sampler(interval_s::Float64=1e-3)
    sampler = RssPeakSampler(Threads.Atomic{Int}(current_rss_bytes()), Threads.Atomic{Bool}(true))
    Threads.@spawn begin
        while sampler.running[]
            rss = current_rss_bytes()
            if rss > sampler.peak_bytes[]
                sampler.peak_bytes[] = rss
            end
            sleep(interval_s)
        end
    end
    return sampler
end

reset!(sampler::RssPeakSampler) = (sampler.peak_bytes[] = current_rss_bytes())
peak_mb(sampler::RssPeakSampler) = sampler.peak_bytes[] / 1024^2
stop!(sampler::RssPeakSampler) = (sampler.running[] = false)