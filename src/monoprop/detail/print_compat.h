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

#pragma once
// std::print polyfill for compilers that lack <print> (GCC < 14).
#if __has_include(<print>)
#include <print>
#else
#include <cstdio>
#include <format>
namespace std { // NOLINT(cert-dcl58-cpp)
template <class... Args>
void print(FILE* f, format_string<Args...> fmt, Args&&... args) {
    auto s = std::vformat(fmt.get(), std::make_format_args(args...));
    std::fwrite(s.data(), 1, s.size(), f);
}
template <class... Args>
void print(format_string<Args...> fmt, Args&&... args) {
    ::std::print(stdout, fmt, std::forward<Args>(args)...);
}
} // namespace std
#endif
