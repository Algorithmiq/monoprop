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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "monoprop/Validation.h"
#include "monoprop/algebra/Algebra.h"
#include "monoprop/detail/evolution/CutoffContext.h"
#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/GateScratch.h"
#include "monoprop/detail/evolution/layer_build/GateSinks.h"
#include "monoprop/detail/evolution/layer_build/PartnerMerge.h"
#include "monoprop/detail/evolution/layer_build/QueryWire.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/evolution/layer_build/TableJoin.h"
#include "monoprop/detail/mpi/MPIUtils.h"
#include "monoprop/detail/mpi/Routing.h"
#include "monoprop/detail/operator/MPOperator.h"

namespace monoprop::detail {

/*! @brief The gate exchange, over a compile-time Sink policy (GateSinks.h).
 *
 *  For a gate G, a tracked term v anticommuting with G has partner u = v ^ G owned by one flat slot, and
 *  the pair rotates iff E(v) or E(u), both adds using the PRE-cos values and phi_v = -phi_u.
 *
 *  One and a half rounds. Only an EMITTING v sends a record (key u, phi_v, rot = E(v), [val_v]) to
 *  owner(u), so round 1 is O(|emitted|), not O(|Anti(G)|). Each slot probes what it received against its
 *  persistent term table once per record: a hit applies +phi_rec*val_rec onto u iff rot_rec or E(u), and
 *  if u is SILENT (it sent nothing) stages a response carrying u's pre-gate coefficient; a miss mints u.
 *  Round 2 delivers the responses, and the absence pass over the records v sent finds the partners nobody
 *  could answer for. Per pair that is the same two adds with the same values whichever branch supplies
 *  them, so the three outcomes are exact, not an approximation of the symmetric protocol. The narrative
 *  argument is in docs/content/docs/features/parallelism.mdx.
 *
 *  Graph mode has no coefficients, so no term is silent and no response can be needed: it keeps the
 *  symmetric one-round predicate its positional replay is proved on. `Sink::wants_responses` switches.
 */
template <size_t NumModes, typename Sink>
struct LayerBuildEngine {
    using RowPosT = typename OperatorIndex<NumModes>::PosT;

    MPOperator<NumModes> &local_op; // scanned, looked up, and grown by the inserts
    mpi::Comm comm;
    size_t R;
    size_t my_rank;
    // This gate's join and the other per-gate scratch, propagator-owned so capacity survives the gate;
    // the protocol's per-row state is in scratch.marks (GateScratch.h).
    GateScratch<NumModes> &scratch;
    size_t combined_size; // the pre-layer operator size
    // Which destination ranks this gate's records can reach. Dense unless the router is GF(2)-linear;
    // see mpi::PeerPlan. Derived once per layer in build_layer, never per record.
    mpi::PeerPlan plan;
    Sink sink;

    LayerBuildEngine(MPOperator<NumModes> &local_op_,
                     mpi::Comm comm_,
                     size_t R_,
                     size_t my_rank_,
                     GateScratch<NumModes> &scratch_,
                     size_t combined_size_,
                     Sink &&sink_,
                     mpi::PeerPlan plan_ = {}) // dense by default: the tests build the engine directly
        : local_op(local_op_),
          comm(comm_),
          R(R_),
          my_rank(my_rank_),
          scratch(scratch_),
          combined_size(combined_size_),
          plan(plan_),
          sink(std::move(sink_)) {}

    /*! @brief The round and a half.
     *
     *  Exchange the scan's records, probe every record against this slot's term table in ONE batched
     *  pass, apply the receiver rule to the self stage and then the incoming records in that fixed order,
     *  exchange the responses the silent hits staged, insert the misses while that is in flight, apply
     *  the responses, walk the sent records. Taking the scan result by value is what makes the sequence
     *  unmissable -- nothing else can read the records once they are here.
     */
    auto exchange_and_join(FusedScanResult<NumModes> &&scan) -> void {
        RowMarks &marks = scratch.marks;
        TableJoin<NumModes> &join = scratch.join;
        MissStage<NumModes> &misses = scratch.misses;
        IncomingRecords<NumModes> &pr = scratch.incoming_records;
        const size_t base = local_op.store->size();
        misses.clear();
        pr.nq_total = 0;
        pr.goff.assign(R + 1, 0);
        assert(scan.queries[my_rank].empty() && "self-owned records are staged, never encoded");

        std::optional<mpi::PendingAlltoallv<size_t>> pending;
        // Round 1 opens here and closes past the decode: the post, the wait and the decode are one stage
        // because no gate can overlap them with work of its own -- the join's row side needs every record.
        if (R > 1) {
            pending.emplace(
                mpi::begin_alltoallv(scan.queries, comm, /*skip_self=*/false, /*known_recv_counts=*/nullptr, plan));
        }
        std::vector<std::span<const size_t>> views;
        if (pending.has_value()) {
            scratch.reuse_incoming_wire(R);
            pending->wait_into(scratch.incoming_wire);
            decode_incoming_records<NumModes>(slot_streams(scratch.incoming_wire, views), sink.incoming_form(), pr);
        }

        // Query order IS mint order: the self stage first, then the incoming sources in ascending slot
        // order, each in its sender's stream order.
        const size_t n_self = scan.self.size();
        const std::span<const SentRecord> sent_self(scan.sent[my_rank]);
        assert(sent_self.size() == n_self && "one sent record per self query");
        join.begin_queries(n_self + pr.nq_total);
        // One batched probe of the term table over the whole query space, in resolve order. The table
        // covers every row of the store (the previous gate's mints were appended by reindex_after_growth),
        // and a record's partner, if tracked anywhere on this slot, is one of them.
        join.run(
            local_op.term_table(),
            *local_op.store,
            [&](size_t q) -> uint32_t { return (q < n_self) ? scan.self.tag_of[q] : pr.tag_of[q - n_self]; },
            [&](size_t q) -> std::span<const RowPosT> {
                return (q < n_self) ? scan.self.positions_at(q) : pr.positions_at(q - n_self);
            },
            [&](size_t /*q*/, size_t row) { marks.set_matched(row); });

        // Both output buffers of the resolve phase are sized here, before the first push, because the
        // join has just settled the two counts they depend on. Every sink call is keyed on either a
        // distinct query -- a hit pushes at most one half, a miss mints at most one -- or a distinct
        // record this slot sent, since a record is answered (round 2, or a self silent hit) or absent but
        // never both, and a row sends exactly one record because it has exactly one partner. So
        // |Q| + |sent| bounds the gate's halves, and |Q| - |hits| bounds its mints exactly.
        size_t n_sent = 0;
        for (const auto &records : scan.sent) {
            n_sent += records.size();
        }
        sink.reserve_halves(join.queries() + n_sent);
        misses.reserve(join.queries() - join.hits());

        // Round 2's send buffer. A sink that answers nothing keeps an empty local rather than naming the
        // shared one, so nothing charges it what an earlier fused call left there.
        std::vector<VecZ> responses;
        if constexpr (Sink::wants_responses) {
            responses.assign(R, VecZ{});
        }
        size_t answered = 0;
        if (n_self != 0) {
            answered =
                join_self<NumModes>(scan.self, join, /*q_base=*/0, marks, my_rank, base, misses, sink, sent_self);
        }
        answered += join_incoming<NumModes>(pr, join, /*q_base=*/n_self, marks, base, misses, sink, responses);

        // Round 1 at its widest: the scan's arrays, the delivered records and the staged responses.
        stamp_gate_buffers_(scan, scratch.incoming_wire, responses, /*extra=*/0);
        run_round_two_(scan, responses, views, base);
        absence_pass<NumModes>(marks, scan.sent, scan.sent_c0, sink);

        scratch.counters.gates += 1;
        scratch.counters.records += n_sent;
        if constexpr (Sink::wants_responses) {
            scratch.counters.responses += answered;
        }
    }

    auto finish(CosMask &&cos_all, CosMask *out_cos = nullptr) -> std::shared_ptr<LayerCore> {
        return sink.finalize(std::move(cos_all), out_cos, combined_size, local_op);
    }

private:
    /*! @brief Round 2: the responses out, the misses inserted under them, the answers applied.
     *
     *  The same verb with the response counts, posted before the inserts so the store's growth overlaps
     *  the peers' answers. Its slot set is round 1's: the rank shift is an involution, so a slot a record
     *  came from is a slot this one can send to. The inserts' order against apply_responses is what the
     *  result depends on, and that is what fixes their place here (mint indices were assigned against
     *  `base` before this call).
     */
    auto run_round_two_(const FusedScanResult<NumModes> &scan,
                        std::vector<VecZ> &responses,
                        std::vector<std::span<const size_t>> &views,
                        size_t base) -> void {
        std::optional<mpi::PendingAlltoallv<size_t>> answering;
        if constexpr (Sink::wants_responses) {
            if (R > 1) {
                answering.emplace(
                    mpi::begin_alltoallv(responses, comm, /*skip_self=*/false, /*known_recv_counts=*/nullptr, plan));
            }
        }
        insert_misses<NumModes>(local_op, scratch.misses, base);
        if constexpr (Sink::wants_responses) {
            if (answering.has_value()) {
                std::vector<VecZ> answers;
                answering->wait_into(answers);
                // Round 2 at its widest: round 1's buffers are all still in scope under the answers.
                stamp_gate_buffers_(scan, scratch.incoming_wire, responses, wire_bytes_(answers));
                apply_responses<NumModes>(scratch.marks, scan.sent, slot_streams(answers, views), sink);
            }
        }
    }

    static auto wire_bytes_(const std::vector<VecZ> &wire) -> size_t {
        size_t bytes = wire.capacity() * sizeof(VecZ);
        for (const VecZ &slot : wire) {
            bytes += slot.capacity() * sizeof(size_t);
        }
        return bytes;
    }

    /*! @brief Stamps the gate's per-gate buffers at one instant into the call's high-water mark.
     *
     *  @param extra Round 2's answers, which are live only at the second sample.
     *
     *  None of what it counts can be read back once the gate is over, which is the whole reason the
     *  field exists; it overlaps the buffers the scratch owns for their capacity, so it is a `d_`
     *  diagnostic beside the ledger and never a term in it.
     */
    auto stamp_gate_buffers_(const FusedScanResult<NumModes> &scan,
                             const std::vector<VecZ> &incoming,
                             const std::vector<VecZ> &responses,
                             size_t extra) -> void {
        size_t bytes = extra + wire_bytes_(scan.queries) + wire_bytes_(incoming) + wire_bytes_(responses)
                       + scratch.misses.memory_bytes() + scratch.incoming_records.memory_bytes()
                       + (scan.self.pos_flat.capacity() * sizeof(RowPosT))
                       + (scan.self.pos_off.capacity() * sizeof(size_t)) + (scan.self.k_of.capacity() * 2)
                       + (scan.self.phase_of.capacity()) + (scan.self.rot_of.capacity())
                       + (scan.self.val_of.capacity() * sizeof(double)) + (scan.self.tag_of.capacity() * 4);
        for (const auto &records : scan.sent) {
            bytes += records.capacity() * sizeof(SentRecord);
        }
        for (const auto &c0 : scan.sent_c0) {
            bytes += c0.capacity() * sizeof(double);
        }
        scratch.buffers_hwm_bytes = std::max(scratch.buffers_hwm_bytes, bytes);
    }
};

static inline auto empty_coeffs() -> const VecD & {
    static const VecD coeffs;
    return coeffs;
}

/*! @brief Primary-path layer builder: one fused scan, one exchange, one join into the chosen sink.
 *
 *  See LayerBuilder.h. `over_cutoff_possible` is the caller's per-call flag
 *  (MonomialPropagator::run_gate_loop_): true when a tracked term may fail the structural cutoff, which
 *  widens the scan's send predicate to every anticommuting term. It must be agreed across the comm, or
 *  one side drops records the other expects.
 */
template <size_t NumModes>
auto build_layer(MPOperator<NumModes> &local_op,
                 const Monomial<NumModes> &gen,
                 const CutoffFn<NumModes> &cutoff_fn,
                 const std::optional<double> &atol,
                 std::optional<std::reference_wrapper<const VecD>> local_coeffs,
                 const std::optional<double> &upper_atol,
                 const std::optional<double> &param,
                 std::optional<size_t> only_rotate_len_k,
                 bool over_cutoff_possible,
                 GateScratch<NumModes> &scratch,
                 mpi::Comm comm,
                 CosMask *out_cos = nullptr,
                 FusedContract *fused_contract = nullptr,
                 bool schrodinger = false,
                 VecD *fused_scale_coeffs = nullptr,
                 bool *fused_scale_out = nullptr,
                 Basis basis = Basis::Majorana) -> std::shared_ptr<LayerCore> {
    validate_only_rotate_len_k_(only_rotate_len_k, 2 * NumModes);
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm));
    const size_t R = static_cast<size_t>(mpi::size(comm));
    // R is the FLAT world (ranks x partitions); the router is what splits it back into the two levels.
    // Hoisted here because mpi::geometry can reach the communicator, so it must never run per term.
    const routing::Router router = router_for<NumModes>(comm);
    assert(router.flat_world() == R);
    // Under linear routing every record for THIS generator lands on the rank this rank's own index XOR
    // rank_shift(gen), so the exchange knows its peer before it starts. Dense otherwise.
    const auto plan =
        mpi::PeerPlan{.sparse = router.is_linear(), .shift = static_cast<int>(router.rank_shift<NumModes>(gen))};
    // Fused contraction runs at all rank counts (R>1 via the cross-rank half-rotation exchange).
    const bool use_fused = (fused_contract != nullptr);
    const auto cut_st = build_majorana_evolution_cutoff_state(atol, local_coeffs, upper_atol, param);
    const auto &coeffs = local_coeffs.value_or(empty_coeffs()).get();
    const CutoffEvaluator<NumModes> cut_eval{cutoff_fn};

    // Fused cos sweep: fold the per-gate cosine into the scan's own coefficient pass. No length cap only
    // (a popcount>k term is outside the per-index cos set) and cos!=0. cos is even, so the sweep's
    // cos(2*build_angle) matches the apply's cos(2*apply_angle) bit-for-bit.
    const double cos_build = (use_fused && param.has_value()) ? std::cos(2.0 * param.value()) : 1.0;
    const bool fused_scale = use_fused && !only_rotate_len_k.has_value() && fused_scale_coeffs != nullptr
                             && param.has_value() && cos_build != 0.0;
    // build_layer is the single authority for this decision; the fused caller must drive its apply from it.
    if (fused_scale_out != nullptr) {
        *fused_scale_out = fused_scale;
    }
    assert(fused_scale_coeffs == nullptr || (local_coeffs && &local_coeffs->get() == fused_scale_coeffs));

    // An identity generator anticommutes with nothing, so there is nothing to scan or exchange. The
    // generator list is replicated, so skipping it needs no agreement.
    const bool identity_gen = !gen.any();

    FusedScanResult<NumModes> fused;
    CosMask cos_all;
    // An identity generator skips the scan, so the fold's words it would have produced are emptied here.
    scratch.nz.clear();
    if (!identity_gen) {
        double *const sweep_ptr = fused_scale ? fused_scale_coeffs->data() : nullptr;
        // The fresh partner's pre-gate coefficient is state-scored only in the Schrödinger picture, and
        // only the fused path needs it on the sender side (graph replay reads it off the extended vector).
        std::optional<Monomial<NumModes>> state_mask;
        if (use_fused && schrodinger) {
            state_mask = initial_state_mask<NumModes>(local_op.initial_state);
        }
        fused = with_algebra<NumModes>(basis, [&]<typename A>() {
            // capture_values selects the fused sink downstream, so it is a compile-time property of the
            // whole scan rather than a flag each record re-tests.
            auto run_scan = [&]<bool CaptureValues>() {
                return fused_find_and_collect<NumModes, A, CaptureValues>(local_op,
                                                                          gen,
                                                                          cut_eval,
                                                                          cut_st,
                                                                          coeffs,
                                                                          only_rotate_len_k,
                                                                          over_cutoff_possible,
                                                                          my_rank,
                                                                          router,
                                                                          scratch,
                                                                          sweep_ptr,
                                                                          cos_build,
                                                                          state_mask ? &*state_mask : nullptr);
            };
            return use_fused ? run_scan.template operator()<true>() : run_scan.template operator()<false>();
        });
        if (fused.cos_blocks.size() == 1) {
            cos_all = std::move(fused.cos_blocks[0]);
        }
        else {
            // Cosine block sets are disjoint and ascending; concatenate in order.
            for (const auto &block : fused.cos_blocks) {
                cos_all.total_count += block.total_count;
                cos_all.blocks.insert(cos_all.blocks.end(), block.blocks.begin(), block.blocks.end());
            }
        }
        fused.cos_blocks = std::vector<CosMask>{};
    }

    auto run = [&]<typename Sink>(Sink sink) -> std::shared_ptr<LayerCore> {
        LayerBuildEngine<NumModes, Sink> eng(local_op,
                                             comm,
                                             R,
                                             my_rank,
                                             scratch,
                                             /*combined_size=*/local_op.store->size(),
                                             std::move(sink),
                                             plan);
        if (!identity_gen) {
            eng.exchange_and_join(std::move(fused));
        }
        return eng.finish(std::move(cos_all), out_cos);
    };

    std::shared_ptr<LayerCore> storage;
    if (use_fused) {
        // fl(1/cos_build), once per gate, which is exactly where the two-pass protocol computed it.
        const double inv_cos = fused_scale ? 1.0 / cos_build : 1.0;
        storage = run(ContractSink<NumModes>{.fc = *fused_contract,
                                             .fused_scale = fused_scale,
                                             .op_coeffs = coeffs,
                                             .cos_build = cos_build,
                                             .inv_cos = inv_cos,
                                             .absences_carry_c0 = schrodinger});
    }
    else {
        storage = run(GraphSink<NumModes>{R, my_rank});
    }
    // Recompute metadata rides with the layer so it survives every graph transform. scaled_count is the
    // post-insert operator size: the fold truncated to it reproduces the "all anticommuting" cos
    // bit-for-bit with no stored bitmap.
    if (storage != nullptr) {
        storage->generator_words.assign(gen.data(), gen.data() + mpi_detail::kWords<NumModes>);
        storage->scaled_count = static_cast<uint64_t>(local_op.store->size());
    }

    return storage;
}

} // namespace monoprop::detail
