// Copyright 2026 Algorithmiq
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "monoprop/detail/ProcessMemory.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace monoprop::detail {
namespace {

auto slurp(const char *path) -> std::string {
    std::ifstream in(path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

auto digits_at(std::string_view text, size_t at) -> size_t {
    size_t value = 0;
    for (; at < text.size() && text[at] >= '0' && text[at] <= '9'; ++at) {
        value = (value * 10) + static_cast<size_t>(text[at] - '0');
    }
    return value;
}

// /proc reports these in kB.
auto status_field(std::string_view text, std::string_view key) -> size_t {
    const auto at = text.find(key);
    if (at == std::string_view::npos) {
        return 0uz;
    }
    return digits_at(text, text.find_first_of("0123456789", at + key.size())) * 1024;
}

constexpr std::string_view kSizeAttr{R"(size=")"};
constexpr std::string_view kHeapTag{R"(<heap nr=)"};

// The per-arena element names repeat once more in the process-wide roll-up, which is LAST.
auto last_size_attr(std::string_view xml, std::string_view tag) -> size_t {
    const auto at = xml.rfind(tag);
    if (at == std::string_view::npos) {
        return 0uz;
    }
    const auto size_at = xml.find(kSizeAttr, at);
    return size_at == std::string_view::npos ? 0uz : digits_at(xml, size_at + kSizeAttr.size());
}

auto malloc_info_xml() -> std::string {
#if defined(__GLIBC__)
    // open_memstream itself allocates, so this reads a few KiB above the state it describes.
    char *buf = nullptr;
    size_t len = 0;
    FILE *stream = ::open_memstream(&buf, &len);
    if (stream == nullptr) {
        return {};
    }
    const int rc = ::malloc_info(0, stream);
    (void)std::fclose(stream);
    // Owns the buffer before the copy below: constructing the string can throw, and free() must still run.
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc) -- open_memstream's buffer is malloc'd
    const std::unique_ptr<char, decltype(&std::free)> owned(buf, &std::free);
    return (rc == 0 && buf != nullptr) ? std::string(buf, len) : std::string{};
#else
    return {};
#endif
}

auto read_process_memory() -> ProcessMemory {
    ProcessMemory out;
#if defined(__linux__)
    const std::string status = slurp("/proc/self/status");
    out.rss_bytes = status_field(status, "VmRSS:");
    out.peak_rss_bytes = status_field(status, "VmHWM:");
#endif
    const std::string xml = malloc_info_xml();
    if (xml.empty()) {
        return out;
    }
    // `system current` is arenas only; mmap'd chunks are separate and never free (free() unmaps them).
    const size_t mmapped = last_size_attr(xml, R"(<total type="mmap")");
    out.alloc_system_bytes = last_size_attr(xml, R"(<system type="current")") + mmapped;
    out.alloc_retained_bytes =
        last_size_attr(xml, R"(<total type="rest")") + last_size_attr(xml, R"(<total type="fast")");
    out.alloc_in_use_bytes = out.alloc_system_bytes - std::min(out.alloc_system_bytes, out.alloc_retained_bytes);
    for (auto at = xml.find(kHeapTag); at != std::string::npos; at = xml.find(kHeapTag, at + kHeapTag.size())) {
        ++out.alloc_arenas;
    }
    return out;
}

} // namespace

// Allocates, so the header's "never throws" is held here rather than asserted: all-or-nothing, since a
// partially-filled result would break the identity its consumers check.
auto process_memory() noexcept -> ProcessMemory {
    try {
        return read_process_memory();
    }
    catch (...) {
        return {};
    }
}

} // namespace monoprop::detail
