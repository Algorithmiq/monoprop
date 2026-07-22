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

#include <cstddef>
#include <limits>
#include <vector>

#include "monoprop/TypeAliases.h"
#include "monoprop/algebra/Algebra.h" // is_paired / get_hf_mask (common) + algebra_hf_phase (fresh Schrödinger miss coeff)
#include "monoprop/detail/evolution/EvolutionHelpers.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/operator/MPOperator.h"

namespace monoprop::detail {

// ─── Shared incoming-record probe (Phases 1-2 + insert) ───────────────────────
// resolve_incoming_queries and its fused twin share the picture-independent
// probe/insert machinery: deserialize every incoming record, batch-find it in the local operator, and
// assign each miss the next index base+j in a serial (sender,record)-order prefix. None of this does
// floating-point math, so all resolvers reuse it verbatim — the deterministic base+j assignment (and
// thus multi-rank bit-exactness) then CANNOT drift between them. Each resolver supplies its own Phase-3
// scatter (the only part that differs: graph acc entries + TermIndex responses vs. half-rotation records
// + value responses) BETWEEN the probe and insert_incoming_misses.
//
// PARALLELISM (load-bearing): all queries in one pass are source⊕G for globally-distinct sources (each
// term owned by one rank) and ⊕G is injective ⇒ queries pairwise distinct ⇒ misses distinct and absent.
// So miss j (in fixed (s,q) order) gets index base+j, byte-identical to a serial current_size++ loop.
template <size_t NumModes>
struct IncomingProbe {
    std::vector<size_t> goff;                  // rank_count+1 flat offsets: g = goff[s] + q
    DefaultInitVector<uint32_t> sender_of;     // g → sender rank
    DefaultInitVector<Monomial<NumModes>> maj; // g → deserialized query monomial
    DefaultInitVector<int> phase_of;           // g → query phase
    DefaultInitVector<size_t> idx_of;          // g → resolved index (HIT: < base; MISS: base+j)
    std::vector<TermIndex> miss_g;             // j → the g that became miss j (Phase 4 reads maj[miss_g[j]])
    size_t base = 0;                           // op size before the miss inserts (the miss-index base)
    size_t nq_total = 0;
};

// Phases 1-2 for QUERY records: deserialize + batch-find every incoming record and assign miss indices.
// Read-only w.r.t. the operator's contents (probes only); the caller runs its Phase-3 scatter, then
// insert_incoming_misses. QW = per-record stride: the plain query width, or kQueryWordsFused for the fused
// resolver (trailing v_src word). Keeping this single copy keeps the deterministic serial (s,q) miss-prefix
// — and thus multi-rank bit-exactness — consistent across resolvers. The value word (if any) is read by
// the caller (see resolve_incoming_queries_fused), never here.
template <size_t NumModes, size_t QW = kQueryWords<NumModes>>
auto probe_incoming_queries(const std::vector<VecZ> &incoming, // serialized, one VecZ per sender
                            MPOperator<NumModes> &op,
                            size_t rank_count) -> IncomingProbe<NumModes> {
    constexpr size_t W = QW;
    IncomingProbe<NumModes> pr;

    // Per-sender query counts and flat (sender-major, query-minor) offsets: g = goff[s] + q.
    pr.goff.assign(rank_count + 1, 0);
    for (size_t s = 0; s < rank_count; ++s) {
        const size_t nq = incoming[s].empty() ? 0 : incoming[s].size() / W;
        pr.goff[s + 1] = pr.goff[s] + nq;
    }
    pr.nq_total = pr.goff[rank_count];
    if (pr.nq_total == 0) {
        return pr;
    }

    // ── Deterministic PARALLEL resolve (see PARALLELISM above) ── probes run lock-free; the table is
    // not mutated during this phase. Only the miss-rank prefix (Phase 2) is serial.
    pr.sender_of.resize(pr.nq_total);
    for (size_t s = 0; s < rank_count; ++s) {
        std::fill(pr.sender_of.begin() + static_cast<std::ptrdiff_t>(pr.goff[s]),
                  pr.sender_of.begin() + static_cast<std::ptrdiff_t>(pr.goff[s + 1]),
                  static_cast<uint32_t>(s));
    }

    // Phase 1 (parallel, read-only): deserialize, then probe with the group-prefetch batch find
    // (chunked so each task pipelines its own probes; the table is not mutated during this phase).
    pr.maj.resize(pr.nq_total);
    pr.phase_of.resize(pr.nq_total);
    pr.idx_of.resize(pr.nq_total);
    for (size_t g = 0; g < pr.nq_total; ++g) {
        const size_t s = pr.sender_of[g];
        const size_t q = g - pr.goff[s];
        Monomial<NumModes> m;
        int ph = 0;
        query_read<NumModes, QW>(incoming[s], q, m, ph);
        pr.maj[g] = m;
        pr.phase_of[g] = ph;
    }
    {
        const size_t op_size = op.store->size();
        op.store->find_batch(pr.maj.data(), pr.nq_total, pr.idx_of.data());
        for (size_t g = 0; g < pr.nq_total; ++g) {
            if (pr.idx_of[g] >= op_size) { // kNotFound is size_t max → also lands here
                pr.idx_of[g] = kMissingIndex;
            }
        }
    }

    // Phase 2 (serial prefix, (sender,query) order): each miss takes the next index base+j. miss_g[j]
    // records which query g became miss j, so Phase 4 reads the deserialized maj[miss_g[j]] directly.
    pr.base = op.store->size(); // LOCAL insert base into the op being mutated
    for (size_t g = 0; g < pr.nq_total; ++g) {
        if (pr.idx_of[g] == kMissingIndex) {
            pr.idx_of[g] = pr.base + pr.miss_g.size();
            pr.miss_g.push_back(static_cast<TermIndex>(g));
        }
    }
    return pr;
}

// Phase 4 (parallel bulk insert of the distinct absent terms): scatter majs into the disjoint op slots
// [base, base+n_miss), insert keys into disjoint map shards, resync the inverted index. Atomics-free (disjoint
// op slots / map shards / inverted index words, as insert_deferred_self_misses). Call AFTER the caller's
// Phase-3 scatter: that scatter reads pre-insert op_coeffs for hits and needs base still == op.size().
template <size_t NumModes>
auto insert_incoming_misses(MPOperator<NumModes> &op, const IncomingProbe<NumModes> &pr) -> void {
    const size_t n_miss = pr.miss_g.size();
    if (n_miss == 0) {
        return;
    }
    // Grow → scatter → index → resync (see insert_absent_terms). pr.base was captured at Phase 2 for the
    // miss-index assignment and no insert has run since, so op.size() still equals pr.base == the insert
    // base here. One writer per miss slot base+j; the staged dense majorana is read straight out of the
    // deserialization buffer via miss_g — the packed row is written once, never re-read here.
    insert_absent_terms<NumModes>(
        op,
        n_miss,
        [&](size_t j) -> const Monomial<NumModes> & { return pr.maj[pr.miss_g[j]]; },
        [&](size_t j, size_t base) { assign_row<NumModes>(*op.store, base + j, pr.maj[pr.miss_g[j]]); });
}

// ─── resolve_incoming_queries ─────────────────────────────────────────────────
// Resolver rank: for each query from sender s, look up M' locally; found → return its index, absent →
// INSERT it (new index i') and return i' in the SAME response round (the resolver is the sole inserter
// of cross-rank absent terms). It also records its inbound entry acc[s].in_entries in query order, so
// build_layer can assemble CrossRankPartnerData without a separate cycle-exchange round.
//
// ORDERING CONTRACT (load-bearing): the B/D exchange is positional — querier A's out_indices[k] must
// pair with resolver B's in_indices[k]. alltoallv preserves per-source order, so responses[s][q]
// answers incoming[s][q]. Every query yields exactly one resolution — DO NOT skip, reorder, or
// partition found vs. absent, or the pairing breaks and multi-rank energy diverges.
//
// Returns per-sender response buffers — one TermIndex per query, each a REAL local index after the
// insert-on-miss (check_index_fits keeps it below the TermIndex ceiling; the element widens under
// monoprop_WIDE_TERM_INDEX). Symmetric-pair dedup is structural.
template <size_t NumModes>
auto resolve_incoming_queries(const std::vector<VecZ> &incoming, // serialized, one VecZ per sender
                              MPOperator<NumModes> &op,
                              size_t rank_count,
                              bool is_leader_pass,
                              MatchedEpochSet &matched,
                              size_t combined_size, // pre-layer op size: bounds the matched set
                              std::vector<PartnerAcc> &acc) -> std::vector<std::vector<TermIndex>> {
    const IncomingProbe<NumModes> pr = probe_incoming_queries<NumModes>(incoming, op, rank_count);
    std::vector<std::vector<TermIndex>> responses(rank_count);
    for (size_t s = 0; s < rank_count; ++s) {
        responses[s].assign(pr.goff[s + 1] - pr.goff[s], std::numeric_limits<TermIndex>::max());
    }
    if (pr.nq_total == 0) {
        return responses;
    }

    // Phase 3 (parallel scatter): responses, resolver IN entries (q order), and matched-follower marks.
    // Found indices are distinct so the matched set has ≤1 writer per slot; freshly inserted partners
    // (ip ≥ base ≥ combined_size) are skipped by the bound check.
    std::vector<size_t> in_base(rank_count);
    for (size_t s = 0; s < rank_count; ++s) {
        in_base[s] = acc[s].in_entries.size();
        acc[s].in_entries.resize(in_base[s] + responses[s].size());
    }
    for (size_t g = 0; g < pr.nq_total; ++g) {
        const size_t s = pr.sender_of[g];
        const size_t q = g - pr.goff[s];
        const size_t ip = pr.idx_of[g];
        responses[s][q] = static_cast<TermIndex>(ip); // real index (post phase-2), fits by check_index_fits
        acc[s].in_entries[in_base[s] + q] = {ip, pr.phase_of[g]};
        if (is_leader_pass && ip < combined_size) {
            matched.mark(ip);
        }
    }

    insert_incoming_misses<NumModes>(op, pr);
    return responses;
}

// ─── resolve_incoming_queries_fused (R>1 ContractImmediately) ─────────────────
// Fused twin of resolve_incoming_queries: shares the exact probe/insert machinery (probe_incoming_queries
// + insert_incoming_misses, so the deterministic base+j miss-index assignment and the inverted index resync are
// literally the same code), but instead of the acc in/out entries + a TermIndex response it emits one
// half-rotation per query into `fc.cross_half` (the resolver's +φ half on the target it owns, v_src known
// from the query) and returns a per-query VALUE stream = the querier half's v_partner (the target coeff):
//   HIT  (target already local, ip < base): resp_val[s][q] = the target's PRE-cos coeff — op_coeffs[ip]
//        directly, or op_coeffs[ip]·inv_cos when the fused cos sweep already scaled it (fused_scale).
//   MISS (target freshly inserted, ip = base+j): the fresh term's picture coeff, which is a pure function
//        of the majorana — 0 in Heisenberg; is_paired ? hf_phase : 0 in Schrödinger (∈ {−1,0,+1}, exactly
//        get_state's scoring). Computed here from the query's own majorana, so it flows back in THIS round
//        with no post-extension second exchange. (base == pre-insert op size, so ip < base ⇔ HIT; distinct
//        sources ⟹ distinct targets ⟹ no query hits a same-layer insert, so a HIT's op_coeffs[ip] is in
//        bounds.) matched.mark is kept for the leader pass (byte-identical to the non-fused resolver).
template <size_t NumModes>
auto resolve_incoming_queries_fused(const std::vector<VecZ> &incoming,
                                    MPOperator<NumModes> &op,
                                    size_t rank_count,
                                    bool is_leader_pass,
                                    MatchedEpochSet &matched,
                                    size_t combined_size,
                                    const VecD &op_coeffs,
                                    bool schrodinger,
                                    FusedContract &fc,
                                    bool fused_scale = false,
                                    double inv_cos = 1.0,
                                    Basis basis = Basis::Majorana) -> std::vector<std::vector<double>> {
    // Incoming records are fused (maj, phase, v_src): probe at the fused stride, read v_src per record.
    const IncomingProbe<NumModes> pr =
        probe_incoming_queries<NumModes, kQueryWordsFused<NumModes>>(incoming, op, rank_count);
    std::vector<std::vector<double>> resp_val(rank_count);
    for (size_t s = 0; s < rank_count; ++s) {
        resp_val[s].resize(pr.goff[s + 1] - pr.goff[s]);
    }
    if (pr.nq_total == 0) {
        return resp_val;
    }

    // Schrödinger fresh-insert coeff = is_paired ? hf_phase : 0, a pure ±1/0 function of the majorana
    // (get_state's scoring). Precompute the HF mask once; unused (empty) in the Heisenberg picture.
    const auto hf_mask = schrodinger ? get_hf_mask<NumModes>(op.slater_determinant) : Monomial<NumModes>{};

    // Phase 3 (parallel scatter): resp_val + one resolver +φ half per query + matched marks. Deterministic
    // resize+indexed-scatter keyed by the flat g (append base = current cross_half size); never a shared
    // push_back.
    const size_t cross_base = fc.cross_half.size();
    fc.cross_half.resize(cross_base + pr.nq_total);
    for (size_t g = 0; g < pr.nq_total; ++g) {
        const size_t s = pr.sender_of[g];
        const size_t q = g - pr.goff[s];
        const size_t ip = pr.idx_of[g];
        double v_tgt;
        if (ip < pr.base) {
            // HIT: the target's PRE-cos coeff. Under the fused cos sweep the resolver's own scan already
            // scaled this slot (an existing anticommuting term), so recover the pre-cos value with the
            // inverse factor — the wire ships pre-cos values exactly as the two-pass path did. MISS values
            // below are computed fresh (never swept) and must NOT be un-scaled.
            v_tgt = fused_scale ? op_coeffs[ip] * inv_cos : op_coeffs[ip];
        }
        else if (schrodinger) {
            // Fresh Schrödinger insert coeff = ⟨b|P|b⟩ scoring, ±1/0. For a Z-only (is_paired) term the
            // Pauli phase omits the Majorana pairing sign (see pauli_hf_phase); off-diagonal terms score 0.
            v_tgt = is_paired<NumModes>(pr.maj[g]) ? algebra_hf_phase<NumModes>(basis, pr.maj[g], hf_mask) : 0.0;
        }
        else {
            v_tgt = 0.0; // Heisenberg fresh insert
        }
        resp_val[s][q] = v_tgt;
        // A MISS half's local slot is a fresh insert (born after the sweep) — flag it so the apply folds
        // the gate's cos into the slot itself; hit halves' slots were swept and take the plain add.
        fc.cross_half[cross_base + g] = HalfRotationRec{ip,
                                                        query_value<NumModes>(incoming[s], q),
                                                        static_cast<int32_t>(pr.phase_of[g]),
                                                        /*is_insert=*/ip >= pr.base};
        if (is_leader_pass && ip < combined_size) {
            matched.mark(ip);
        }
    }

    insert_incoming_misses<NumModes>(op, pr);
    return resp_val;
}

// ─── process_query_responses ──────────────────────────────────────────────────
// Querier rank: turn each resolver response (always a real partner index, since the resolver inserts
// on miss) into a querier-side OUT entry (source idx + query phase) in the shared per-rank PartnerAcc.
// The self/local rank was already resolved inline, so it is skipped here.
template <size_t NumModes>
auto process_query_responses(const std::vector<std::vector<TermIndex>> &responses,
                             const std::vector<std::vector<size_t>> &src_idx,
                             const std::vector<VecZ> &queries, // serialized query buffers (for phase recovery)
                             size_t rank_count,
                             size_t my_rank,
                             std::vector<PartnerAcc> &acc) -> void {
    for (size_t r = 0; r < rank_count; ++r) {
        if (r == my_rank) {
            continue;
        } // local already handled inline
        const auto &resp = responses[r];
        const auto &srcs = src_idx[r];
        const auto &qbuf = queries[r];
        const size_t nq = resp.size();
        if (nq == 0) {
            continue;
        }
        // OUT block (querier side), in response (== q) order, appended after any earlier pass's
        // entries. Resize once + indexed scatter (mirrors resolve_incoming_queries' in_entries fill):
        // every query yields exactly one OUT entry, so the slot for q is base+q — parallelizable with
        // no ordering hazard. Only source_idx + the trailing phase word feed it; the resolver inserts
        // on miss so found_idx is always a real index and is not needed downstream (see the assert,
        // which reconstructs nothing).
        auto &out = acc[r].out_entries;
        const size_t base = out.size();
        out.resize(base + nq);
        for (size_t q = 0; q < nq; ++q) {
            assert(resp[q] != std::numeric_limits<TermIndex>::max() && "resolver must insert absent cross-rank terms");
            out[base + q] = {srcs[q], query_phase<NumModes>(qbuf, q)};
        }
    }
}

// ─── process_query_responses_fused (R>1 ContractImmediately) ──────────────────
// Fused twin of process_query_responses: turns each resolver value response into a querier half-rotation
// on the source slot THIS rank owns. inc_rval[r] is parallel to this rank's queries to r (resp_val came
// back in query order, so inc_rval[r][q] answers query q), and every value is the target coeff v_tgt — the
// resolver computed the fresh-insert coeff on the spot, so there is no NaN sentinel and no second round.
// For each r != my_rank, q ascending, append a querier half {S=src_idx[r][q], v_tgt, −φ}: c[S] +=
// sin·(−φ)·v_tgt is applied later with the resolver halves. (A Heisenberg fresh-insert v_tgt is 0 ⟹ a
// harmless no-op add.) Serial per r in q-order (no parallel push_back into the shared cross_half).
template <size_t NumModes>
auto process_query_responses_fused(const std::vector<std::vector<double>> &inc_rval,
                                   const std::vector<std::vector<size_t>> &src_idx,
                                   const std::vector<VecZ> &queries,
                                   size_t rank_count,
                                   size_t my_rank,
                                   FusedContract &fc) -> void {
    // The resolver already resize()d cross_half to exact size; reserve the querier-half count (one per
    // incoming response) up front so these push_backs don't reallocate that block per rank.
    size_t incoming = 0;
    for (size_t r = 0; r < rank_count; ++r) {
        if (r != my_rank) {
            incoming += inc_rval[r].size();
        }
    }
    fc.cross_half.reserve(fc.cross_half.size() + incoming);
    for (size_t r = 0; r < rank_count; ++r) {
        if (r == my_rank) {
            continue;
        }
        const auto &rval = inc_rval[r];
        const auto &srcs = src_idx[r];
        const auto &qbuf = queries[r];
        const size_t nq = rval.size();
        for (size_t q = 0; q < nq; ++q) {
            const auto nphase = static_cast<int32_t>(-query_phase<NumModes>(qbuf, q));
            // The local slot this half writes is the querier's SOURCE — an existing pre-gate term
            // (< combined_size) the cos sweep covered, so it always takes the plain add (is_insert=false).
            fc.cross_half.push_back(HalfRotationRec{srcs[q], rval[q], nphase, /*is_insert=*/false});
        }
    }
}

} // namespace monoprop::detail
