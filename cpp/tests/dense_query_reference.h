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

// THE DENSE QUERY RECORD, AS A TEST-ONLY REFERENCE.
//
// This was the production wire format until the compact record replaced it outright: W monomial words
// plus one phase word, at a fixed stride, so the q'th query starts at q * kQueryWords. It lived in
// detail/evolution/layer_build/Common.h and was already marked there as "A DENSE-ONLY ORACLE, NOT
// PRODUCTION CODE" -- kept solely so the codec could be pinned against an independently written second
// implementation. Deleting the format from production did not delete that value, so the block moved
// here instead of going away.
//
// WHY IT IS WORTH KEEPING. CompactQuery is now the only format, which means a round-trip test on it
// can only ever ask whether it agrees with ITSELF: encode, decode, compare. That catches a decoder
// that mirrors an encoder's mistake not at all. This reference shares no code with it -- different
// layout, different width, different offset arithmetic -- so `compact_query_agrees_with_the_dense_codec`
// is a genuine second opinion about what a monomial and a phase mean on the wire, and it is the only
// remaining test that can catch the two of them being wrong together.
//
// It is deliberately in `monoprop::test_ref` and deliberately under cpp/tests/. Nothing in the shipped
// headers may include it. A constant-stride reader sitting in a production header is exactly how this
// work produced two knob-blind oracles: both read a
// layout-selected buffer at a hardcoded stride and passed silently once the layout changed under them.
// The namespace and the directory are the guardrail against a third.
//
// This is frozen. It is not a second implementation to keep in sync with anything -- if the compact
// format changes, this file does NOT follow it, and the differential test is expected to be rewritten
// or retired rather than patched to agree.

#pragma once

#include <cstddef>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/core/Monomial.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/mpi/MPIUtils.h"

namespace monoprop::test_ref {

using monoprop::VecZ;

// W monomial words + one ±1 phase word.
template <size_t NumModes>
inline constexpr size_t kQueryWords = mpi_detail::kWords<NumModes> + 1;

// The plain record plus one trailing word holding the source's pre-cos coeff, bit-cast from double, so
// query and value rode a single alltoallv instead of two.
template <size_t NumModes>
inline constexpr size_t kQueryWordsFused = kQueryWords<NumModes> + 1;

template <size_t NumModes>
inline auto query_push(VecZ &buf, const Monomial<NumModes> &mono, int phase) -> void {
    mpi_detail::append_monomial_words<NumModes>(mono, buf);
    buf.push_back(detail::encode_phase(phase));
}

// The mono + phase words occupy the same leading offsets in the plain and fused record, so readers
// differ only in the per-record stride QW (defaulted to the plain width).
template <size_t NumModes, size_t QW = kQueryWords<NumModes>>
inline auto query_read(const VecZ &buf, size_t q, Monomial<NumModes> &mono_out, int &phase_out) -> void {
    const size_t base = q * QW;
    mono_out = mpi_detail::read_monomial_from_words<NumModes>(buf, base);
    phase_out = detail::decode_phase(buf[base + mpi_detail::kWords<NumModes>]);
}

template <size_t NumModes>
inline auto query_value(const VecZ &buf, size_t q) -> double {
    return detail::decode_value(buf[q * kQueryWordsFused<NumModes> + mpi_detail::kWords<NumModes> + 1]);
}

// Requires v.size() == q.size()/kQueryWords: exactly one value per query record.
template <size_t NumModes>
inline auto build_fused_query_value(const VecZ &q, const std::vector<double> &v, VecZ &out) -> void {
    constexpr size_t W = kQueryWords<NumModes>;
    const size_t nq = q.empty() ? 0 : q.size() / W;
    out.clear();
    out.reserve(nq * kQueryWordsFused<NumModes>);
    for (size_t i = 0; i < nq; ++i) {
        out.insert(out.end(),
                   q.begin() + static_cast<std::ptrdiff_t>(i * W),
                   q.begin() + static_cast<std::ptrdiff_t>((i + 1) * W));
        out.push_back(detail::encode_value(v[i]));
    }
}

} // namespace monoprop::test_ref
