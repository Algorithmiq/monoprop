#pragma once
// std::print polyfill for compilers that lack <print> (GCC < 14).
// When <print> is available natively, we just include it.
#if __has_include(<print>)
#    include <print>
#else
#    include <cstdio>
#    include <format>
namespace std {  // NOLINT(cert-dcl58-cpp)
template <class... Args>
void print(FILE* f, format_string<Args...> fmt, Args&&... args) {
    auto s = std::vformat(fmt.get(), std::make_format_args(args...));
    std::fwrite(s.data(), 1, s.size(), f);
}
template <class... Args>
void print(format_string<Args...> fmt, Args&&... args) {
    ::std::print(stdout, fmt, std::forward<Args>(args)...);
}
}  // namespace std
#endif
