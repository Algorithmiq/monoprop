#include <boost/test/unit_test.hpp>

#include <cmath>
#include <cstdint>
#include <numbers>
#include <span>
#include <vector>

#include "monoprop/MajoranaAlgebra.h" // MajoranaSet
#include "monoprop/TypeAliases.h"
#include "monoprop/detail/evolution/CoeffFrame.h"

// Unit tests for the lazy-cosine CoeffFrame: the on-demand coefficient reconstruction
//   true_value(i) = stored × ∏ cos(2θ_f)  over firings f ∈ (stamp, now]  with  a(M,G_f)=1
// checked against a straight brute-force product; the barrier materialize_all sweep; and the
// magnitude-byte upper-bound helpers. The reconstruction is a pure function of (stored, stamp,
// positions, log), so these tests exercise it directly with no operator store.

using namespace monoprop;            // NOLINT
using namespace monoprop::detail;    // NOLINT

namespace {
constexpr size_t N = 32; // 2N = 64 majorana columns ⇒ generators/terms use raw bit positions [0,64)

// A firing described independently of the frame, for the brute-force reference.
struct Firing {
    std::vector<size_t> gen; // generator support (raw bit positions)
    double two_theta;
};

MajoranaSet<N> gen_set(const std::vector<size_t> &positions) {
    MajoranaSet<N> g{};
    for (const size_t p : positions) {
        g.set(p);
    }
    return g;
}

// a(M, G) = parity(|M ∩ G|) ⊕ (|G| odd ? parity(|M|) : 0).
bool anti(const std::vector<uint32_t> &term, bool m_odd, const std::vector<size_t> &gen) {
    size_t overlap = 0;
    for (const uint32_t p : term) {
        for (const size_t q : gen) {
            overlap += (static_cast<size_t>(p) == q) ? 1 : 0;
        }
    }
    bool a = (overlap & 1U) != 0;
    if (m_odd && (gen.size() & 1U) != 0) {
        a = !a;
    }
    return a;
}

// stored × ∏ cos(2θ_f) over firings f ∈ [stamp, nfirings) with a(term, G_f)=1.
double brute_force(double stored,
                   uint32_t stamp,
                   const std::vector<uint32_t> &term,
                   const std::vector<Firing> &firings) {
    const bool m_odd = (term.size() & 1U) != 0;
    double v = stored;
    for (size_t f = stamp; f < firings.size(); ++f) {
        if (anti(term, m_odd, firings[f].gen)) {
            v *= std::cos(firings[f].two_theta);
        }
    }
    return v;
}

CoeffFrame<N> build_frame(const std::vector<Firing> &firings) {
    CoeffFrame<N> frame;
    for (const auto &f : firings) {
        frame.append_firing(f.two_theta, gen_set(f.gen));
    }
    return frame;
}

double reconstruct(const CoeffFrame<N> &frame, double stored, uint32_t stamp, const std::vector<uint32_t> &term) {
    const bool m_odd = (term.size() & 1U) != 0;
    return frame.reconstruct(stored, stamp, std::span<const uint32_t>(term.data(), term.size()), m_odd);
}

// A minimal duck-typed store for true_value / materialize_all (needs only for_each_position).
struct MockStore {
    std::vector<std::vector<uint32_t>> rows;
    template <typename Fn>
    void for_each_position(size_t i, Fn fn) const {
        for (const uint32_t p : rows[i]) {
            fn(static_cast<size_t>(p));
        }
    }
};
} // namespace

// Reconstruction over a mix of even and odd generators must equal the brute-force cosine product,
// for terms that anticommute with none / some / all firings.
BOOST_AUTO_TEST_CASE(coeff_frame_reconstruct_matches_brute_force) {
    const std::vector<Firing> firings{
        {{0, 1}, 0.31},          // even |G|
        {{2, 3, 4}, -0.72},      // odd  |G|
        {{0, 5}, 1.10},          // even
        {{1, 2, 6, 7}, 0.05},    // even
        {{3}, 0.93},             // odd
        {{0, 1, 2, 3, 4}, -1.4}, // odd
    };
    const auto frame = build_frame(firings);

    const std::vector<std::vector<uint32_t>> terms{
        {},              // empty support (m_odd = false; anti only via odd-|G| correction when m_odd... none)
        {8, 9},          // touches no generator column
        {0},             // odd, overlaps several
        {0, 1, 2, 3},    // even
        {2, 3, 4, 5, 6}, // odd
    };
    for (const auto &term : terms) {
        for (uint32_t stamp = 0; stamp <= firings.size(); ++stamp) {
            const double got = reconstruct(frame, /*stored=*/0.5, stamp, term);
            const double want = brute_force(0.5, stamp, term, firings);
            BOOST_TEST(got == want, boost::test_tools::tolerance(1e-15));
        }
    }
}

// The θ = π/4 kicked-Ising extreme: cos(2θ) = cos(π/2) ≈ 6e-17. A term that anticommutes with such a
// firing is driven to (near) zero — reconstruction must reproduce the tiny product with no clamp/NaN.
BOOST_AUTO_TEST_CASE(coeff_frame_reconstruct_extreme_cosine) {
    const double quarter_pi = std::numbers::pi / 4.0;
    const std::vector<Firing> firings{
        {{0, 1}, 2.0 * quarter_pi}, // cos(π/2) ≈ 6.1e-17
        {{0, 2}, 0.5},
    };
    const auto frame = build_frame(firings);
    const std::vector<uint32_t> term{0}; // anticommutes with firing 0 (overlap 1) and firing 1
    const double got = reconstruct(frame, 1.0, 0, term);
    const double want = brute_force(1.0, 0, term, firings);
    BOOST_TEST(got == want);        // exact same product order ⇒ bit-identical
    BOOST_TEST(std::abs(got) < 1e-15); // annihilated, finite (no clamp, no NaN)
    BOOST_TEST(std::isfinite(got));
}

// Window edges: a log that crosses the 64-firing word boundary must apply exactly the firings in
// [stamp, nfirings) at stamps 0, 63, 64, 65 and the last index.
BOOST_AUTO_TEST_CASE(coeff_frame_reconstruct_window_edges) {
    std::vector<Firing> firings;
    firings.reserve(70);
    for (size_t f = 0; f < 70; ++f) {
        // Alternate even/odd generators; vary the angle so the products are non-degenerate.
        std::vector<size_t> gen = (f % 3 == 0) ? std::vector<size_t>{0, 1, 2} : std::vector<size_t>{0, 3};
        firings.push_back({gen, 0.1 + 0.01 * static_cast<double>(f)});
    }
    const auto frame = build_frame(firings);
    const std::vector<uint32_t> term{0, 4}; // anticommutes with the {0,..} generators
    for (const uint32_t stamp : {0U, 1U, 62U, 63U, 64U, 65U, 69U, 70U}) {
        const double got = reconstruct(frame, -0.75, stamp, term);
        const double want = brute_force(-0.75, stamp, term, firings);
        BOOST_TEST(got == want, boost::test_tools::tolerance(1e-14));
    }
}

// stamp >= nfirings (a term born after the last logged firing) returns stored unchanged; stored==0 too.
BOOST_AUTO_TEST_CASE(coeff_frame_reconstruct_fresh_and_zero) {
    const std::vector<Firing> firings{{{0, 1}, 0.4}, {{1, 2}, 0.8}};
    const auto frame = build_frame(firings);
    const std::vector<uint32_t> term{1};
    BOOST_TEST(reconstruct(frame, 3.14, /*stamp=*/2, term) == 3.14); // stamp == nfirings
    BOOST_TEST(reconstruct(frame, 3.14, /*stamp=*/5, term) == 3.14); // stamp > nfirings
    BOOST_TEST(reconstruct(frame, 0.0, /*stamp=*/0, term) == 0.0);   // stored 0 stays 0
}

// materialize_all writes each term's true value into coeffs, empties the log, resets stamps, and is
// idempotent (a second sweep changes nothing).
BOOST_AUTO_TEST_CASE(coeff_frame_materialize_all_and_idempotent) {
    const std::vector<Firing> firings{
        {{0, 1}, 0.30},
        {{1, 2}, -0.60},
        {{0, 2, 3}, 0.90}, // odd
    };
    MockStore store{{{0, 1}, {1}, {2, 3}, {8}}}; // 4 terms; term 3 touches no generator column
    VecD coeffs{0.5, -0.25, 0.75, 1.0};
    const VecD stored = coeffs;

    auto frame = build_frame(firings);
    frame.stamp.assign(coeffs.size(), 0);

    // Reference true values before the sweep.
    std::vector<double> want(coeffs.size());
    for (size_t i = 0; i < coeffs.size(); ++i) {
        const bool m_odd = (store.rows[i].size() & 1U) != 0;
        want[i] = brute_force(stored[i], 0, store.rows[i], firings);
        (void)m_odd;
    }

    frame.materialize_all(store, coeffs, /*use_mag=*/true);

    BOOST_TEST(!frame.active());          // log emptied
    BOOST_TEST(frame.nfirings == 0U);
    BOOST_TEST(frame.mag.size() == coeffs.size()); // byte rebuilt from the materialized coeffs
    for (size_t i = 0; i < coeffs.size(); ++i) {
        BOOST_TEST(coeffs[i] == want[i], boost::test_tools::tolerance(1e-15));
        BOOST_TEST(frame.stamp[i] == 0U);
    }

    // Idempotent: with an empty log the second sweep leaves coeffs untouched.
    const VecD after = coeffs;
    frame.materialize_all(store, coeffs, /*use_mag=*/true);
    for (size_t i = 0; i < coeffs.size(); ++i) {
        BOOST_TEST(coeffs[i] == after[i]);
    }
}

// mag_byte is a valid upper bound: 2^(byte - kMagBias) >= |c|, with 0/denormal in the zero bucket
// and large values clamped into range.
BOOST_AUTO_TEST_CASE(coeff_frame_mag_byte_upper_bound) {
    BOOST_TEST(mag_byte(0.0) == 0);
    BOOST_TEST(mag_byte(5e-324) == 0); // smallest denormal folds into the deep-tail bucket

    for (const double c : {1.0, -1.0, 0.5, 1e-5, -3.2e-8, 1234.5, 1e15, -1e-30, std::ldexp(1.0, 55)}) {
        const uint8_t b = mag_byte(c);
        BOOST_TEST(b > 0);
        const double bound = std::ldexp(1.0, static_cast<int>(b) - kMagBias);
        BOOST_TEST(bound >= std::abs(c)); // decoded bound is a true upper bound
    }

    // The reject threshold rejects iff (double)byte <= rt: a term below |sin|·bound ≤ atol.
    const double rt = mag_reject_threshold(/*check_atol=*/true, /*abs_sin=*/0.5, /*atol=*/1e-6);
    BOOST_TEST(static_cast<double>(mag_byte(1e-9)) <= rt);  // tiny ⇒ rejected
    BOOST_TEST(static_cast<double>(mag_byte(1.0)) > rt);    // O(1) ⇒ kept
    // Inactive gate never rejects.
    BOOST_TEST(mag_reject_threshold(false, 0.5, 1e-6) == -std::numeric_limits<double>::infinity());
}

// The engage gate switches eager→lazy exactly when the operator's row+coeff working set outgrows the
// detected last-level cache. Expressed relative to the detected budget so it holds on any machine.
BOOST_AUTO_TEST_CASE(coeff_frame_engage_cache_gate) {
    const size_t budget = last_level_cache_bytes();
    BOOST_TEST(budget > 0U); // a cache size was detected (or the conservative fallback)

    constexpr size_t row_bytes = ((2 * N + 63) / 64) * 8;
    constexpr size_t bytes_per_term = row_bytes + sizeof(double);
    const size_t knee = budget / bytes_per_term; // largest term count that still fits in cache

    BOOST_TEST(!engage_lazy_frame<N>(0));       // empty operator ⇒ eager
    BOOST_TEST(!engage_lazy_frame<N>(knee));    // fits in cache ⇒ eager
    BOOST_TEST(engage_lazy_frame<N>(knee + 1)); // just over cache ⇒ lazy
    BOOST_TEST(engage_lazy_frame<N>(knee * 4)); // well over ⇒ lazy

    // Wider majorana rows (more modes) reach the knee at FEWER terms, so an operator that just crossed
    // for 2N=64 columns is comfortably over for 2N=256.
    BOOST_TEST(engage_lazy_frame<128>(knee + 1));
}
