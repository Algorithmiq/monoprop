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
#
# Line-for-line mirror of `monoprop_bench_tools.memory.cpu`. The two must stay in step: a cross-language
# memory comparison is only meaningful if both sides measure the same quantity the same way,
# down to how the process is settled before the window opens.

# Reset VmHWM to the current RSS, starting a new measurement window
const _CLEAR_REFS_MM_HIWATER_RSS = "5\n"

"""Return a `/proc/self` size field (kB -> bytes); 0 if unavailable."""
function proc_field(path::AbstractString, key::AbstractString)::Int
    try
        for line in eachline(path)
            if startswith(line, key)
                return parse(Int, split(line)[2]) * 1024  # values are in kB
            end
        end
    catch
        return 0  # non-Linux or restricted /proc
    end
    return 0
end

"""Return this process's current resident set size (RSS) in bytes."""
rss_bytes()::Int = proc_field("/proc/self/status", "VmRSS:")

"""Return the kernel's high-water mark of this process's RSS, in bytes.

VmHWM is maintained by the kernel on every RSS increase, so it is exact: unlike a polling
sampler it cannot miss a transient that is allocated and freed between two observations.
"""
peak_rss_bytes()::Int = proc_field("/proc/self/status", "VmHWM:")

"""Reset VmHWM to the current RSS, starting a new measurement window.

Returns `false` where `/proc/self/clear_refs` is unavailable (non-Linux, kernel < 4.0, or a
restricted sandbox), in which case VmHWM keeps counting from process start and callers must
fall back.
"""
function reset_peak_rss()::Bool
    try
        write("/proc/self/clear_refs", _CLEAR_REFS_MM_HIWATER_RSS)
    catch
        return false
    end
    return true
end

"""Ask the C allocator to return unused heap pages to the OS."""
function heap_trim()
    try
        ccall((:malloc_trim, "libc"), Cint, (Csize_t,), 0)
    catch
        nothing  # unsupported platform / allocator
    end
    return nothing
end

"""Return current RSS after collecting garbage and trimming the C heap."""
function resting_rss_bytes()::Int
    GC.gc()
    heap_trim()
    return rss_bytes()
end

"""Exact peak RSS over a measurement window, straight from the kernel.

`start!` settles the process (`GC.gc()` + `malloc_trim`) and resets VmHWM to that floor, so
the peak reported is this window's own and not an earlier window's retained garbage. The
settling is what makes the number comparable against the Python side: without it the figure
tracks the GC's willingness to return pages more than what the code needed.

`exact` is `false` when the kernel would not reset the window (see `reset_peak_rss`); the
peak then degrades to the RSS observed at `stop!`, which is a lower bound.
"""
mutable struct HighWaterMark
    baseline_bytes::Int
    peak_bytes::Int
    exact::Bool
end

HighWaterMark() = HighWaterMark(0, 0, false)

"""Open the window: settle, reset VmHWM, and record the floor."""
function start!(hwm::HighWaterMark)
    hwm.baseline_bytes = resting_rss_bytes()
    hwm.exact = reset_peak_rss()
    hwm.peak_bytes = hwm.baseline_bytes
    return hwm
end

"""Close the window and latch the peak."""
function stop!(hwm::HighWaterMark)
    observed = hwm.exact ? peak_rss_bytes() : rss_bytes()
    hwm.peak_bytes = max(hwm.baseline_bytes, observed)
    return hwm
end

peak_mb(hwm::HighWaterMark) = hwm.peak_bytes / 1024^2
baseline_mb(hwm::HighWaterMark) = hwm.baseline_bytes / 1024^2
delta_mb(hwm::HighWaterMark) = (hwm.peak_bytes - hwm.baseline_bytes) / 1024^2
