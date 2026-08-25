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
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace monoprop::detail {
namespace {

auto slurp(const char *path) -> std::string {
    std::string out;
    FILE *file = std::fopen(path, "re");
    if (file == nullptr) {
        return out;
    }
    std::array<char, 4096> buf{};
    for (size_t n = 0; (n = std::fread(buf.data(), 1, buf.size(), file)) > 0;) {
        out.append(buf.data(), n);
    }
    (void)std::fclose(file);
    return out;
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

// The per-arena element names repeat once more in the process-wide roll-up, which is LAST.
auto last_size_attr(std::string_view xml, std::string_view tag) -> size_t {
    const auto at = xml.rfind(tag);
    if (at == std::string_view::npos) {
        return 0uz;
    }
    const auto size_at = xml.find("size=\"", at);
    return size_at == std::string_view::npos ? 0uz : digits_at(xml, size_at + 6);
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
    std::string out = (rc == 0 && buf != nullptr) ? std::string(buf, len) : std::string{};
    std::free(buf); // NOLINT(cppcoreguidelines-no-malloc) -- open_memstream's buffer is malloc'd
    return out;
#else
    return {};
#endif
}

} // namespace

auto process_memory() -> ProcessMemory {
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
    const size_t mmapped = last_size_attr(xml, "<total type=\"mmap\"");
    out.alloc_system_bytes = last_size_attr(xml, "<system type=\"current\"") + mmapped;
    out.alloc_retained_bytes =
        last_size_attr(xml, "<total type=\"rest\"") + last_size_attr(xml, "<total type=\"fast\"");
    out.alloc_in_use_bytes = out.alloc_system_bytes - std::min(out.alloc_system_bytes, out.alloc_retained_bytes);
    for (auto at = xml.find("<heap nr="); at != std::string::npos; at = xml.find("<heap nr=", at + 1)) {
        ++out.alloc_arenas;
    }
    return out;
}

} // namespace monoprop::detail
