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
#include <bit>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <utility>

#include "monoprop/MonomialPropagator.h"
#include "monoprop/detail/EnvConfig.h"
#include "monoprop/detail/evolution/CoeffFrame.h"
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/evolution/LayerBuilder.h"
#include "monoprop/detail/evolution/layer_build/FusedApply.h"

namespace monoprop {

template <size_t NumModes>
MonomialPropagator<NumModes>::MonomialPropagator(const FermiOperatorMap &initial_operator,
                                                 unsigned int cutoff,
                                                 const VecZ &slater_determinant,
                                                 std::optional<unsigned int> schrodinger_cutoff,
                                                 MPI_Comm comm,
                                                 std::optional<double> lower_atol,
                                                 std::optional<double> upper_atol,
                                                 CutoffType cutoff_type,
                                                 std::optional<std::vector<VecZ>> basis_change,
                                                 size_t logical_num_modes)
    : schrodinger_{schrodinger_cutoff.has_value()},
      comm_{comm},
      mp_op_{},
      graph_(schrodinger_cutoff.has_value()),
      cutoff_{cutoff},
      lower_atol_{lower_atol},
      upper_atol_{upper_atol},
      logical_num_modes_{logical_num_modes},
      cutoff_type_{cutoff_type},
      basis_change_{basis_change} {
    if (logical_num_modes_ == 0 || logical_num_modes_ > NumModes) {
        throw std::runtime_error(
            std::format("logical_num_modes ({}) must be in the range [1, {}].", logical_num_modes_, NumModes));
    }

    // Validate atol parameters
    if (upper_atol.has_value() && lower_atol.has_value() && (upper_atol.value() < lower_atol.value())) {
        throw std::runtime_error(std::format("upper_atol ({}) must be greater than or equal to lower_atol ({}).",
                                             upper_atol.value(),
                                             lower_atol.value()));
    }

    const size_t num_ranks = static_cast<size_t>(mpi::size(comm_));
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm_));
    MajoranaVector<NumModes> local_heisenberg_terms;

    // convert the operator to the internal format
    double core_term = 0.0;
    for (const auto &[indices, coefficient] : initial_operator) {
        for (const auto &index : indices) {
            if (index >= 2 * logical_num_modes_) {
                throw std::runtime_error(
                    std::format("Operator term contains an index greater than {}", 2 * logical_num_modes_));
            }
        }
        const auto majorana_bitset = indices_to_bitset<NumModes>(indices);
        const auto encoded_coeff = encode_coeff<NumModes>(coefficient, majorana_bitset);

        // Store the core term separately as it is orders of magnitude larger than the other terms
        if (indices.empty()) {
            core_term = encoded_coeff;
            continue;
        }
        if (my_rank == find_rank<NumModes>(majorana_bitset, num_ranks)) {
            mp_op_.init_op_map[majorana_bitset] = encoded_coeff;
            local_heisenberg_terms.push_back(majorana_bitset);
        }
    }

    auto sc = schrodinger_cutoff.value_or(cutoff + 2);
    sc = std::min(sc, static_cast<unsigned int>(2 * logical_num_modes_));
    auto op = schrodinger_ ? generate_paired_op<NumModes>(sc / 2 + sc % 2, logical_num_modes_) : local_heisenberg_terms;

    const size_t expected_local_terms = std::max<size_t>(1, op.size() / std::max<size_t>(1, num_ranks));
    // Build the cutoff function BEFORE the store: packed_inline_width_() derives the packed-row width
    // from cutoff_fn_, so it must be populated first — otherwise the width silently falls back to the
    // loose kMaxInlinePositions and the cutoff-adaptive row narrowing is dead. Inputs (cutoff_type_,
    // cutoff_, basis_change_, logical_num_modes_) are all set by now; nothing below re-touches them.
    regenerate_cutoff_fn_();
    // Re-init the store with this run's cutoff-adaptive packed-row width (terms with popcount <=
    // cutoff are the common case for length/mode cutoffs; longer fully-paired terms spill to
    // overflow losslessly, so a tight width only ever helps). Width is a construction invariant,
    // so a fresh store sets it; then size rows + index once to the expected per-rank count.
    mp_op_.store = std::make_unique<detail::OperatorIndex<NumModes>>(packed_inline_width_());
    mp_op_.store->reserve(expected_local_terms);

    size_t i = 0;
    // The initial operator is a sum over DISTINCT Majorana monomials (Hamiltonian / paired-ham
    // basis), so each maj is unique on this rank and emplace (insert-if-absent) is equivalent to
    // an assigning insert — the duplicate-key distinction does not arise.
    for (size_t r = 0; r < op.size(); ++r) {
        const auto &maj = materialize_row<NumModes>(op, r);
        if (my_rank == find_rank<NumModes>(maj, num_ranks)) {
            mp_op_.append_term(maj);
            mp_op_.store->emplace(maj, i++);
        }
    }

    // Initialize this rank's MPOperator
    mp_op_.slater_determinant = slater_determinant;
    core_term_ = core_term;

    // (cutoff function was built above, before the store, so packed_inline_width_ could use it)
    initialize_operator_caches_();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::packed_inline_width_() const -> size_t {
    constexpr size_t kMax = detail::OperatorIndex<NumModes>::kMaxInlinePositions;
    if (schrodinger_) {
        return kMax;
    }
    const auto bound = detail::CutoffEvaluator<NumModes>(cutoff_fn_).max_positions_bound();
    return bound ? std::min<size_t>(*bound, kMax) : kMax;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::apply_initial_operator_(const FermiOperatorMap &op_dict)
    -> std::pair<MajoranaVector<NumModes>, VecD> {
    const size_t num_ranks = static_cast<size_t>(mpi::size(comm_));
    const size_t my_rank = static_cast<size_t>(mpi::rank(comm_));

    // Convert the input operator to the internal format and distribute terms to ranks
    FermiOperatorMap new_op;
    for (const auto &[ind, coeff] : op_dict) {
        const auto maj = indices_to_bitset<NumModes>(ind);
        if (ind.empty()) { // Core term, store in all
            core_term_ = encode_coeff<NumModes>(coeff, maj);
            continue;
        }
        if (my_rank == find_rank<NumModes>(maj, num_ranks)) {
            const auto maj_indices = bitset_to_indices<NumModes>(maj);
            new_op[maj_indices] = coeff;
        }
    }

    // Update this rank's operator
    auto res = mp_op_.update_initial_operator(new_op, schrodinger_);
    return std::move(std::get<2>(res));
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::graph_data() const -> std::vector<LayerData> {
    std::vector<LayerData> layers;
    const auto num_layers = graph_.layers();
    layers.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        const auto traversal = graph_.get_layer_traversal(i);
        const size_t rank_count = traversal.cross_rank_rank_count();

        // Local cycles are folded into cross_rank[my_rank] — no separate local_cycles slot.
        std::vector<LocalCycleData> local_cyc_data;

        std::vector<CrossRankData> b_data, d_data;
        b_data.reserve(rank_count);
        d_data.reserve(rank_count);
        for (size_t rank = 0; rank < rank_count; ++rank) {
            VecZ sin_send_indices(traversal.cross_rank_sin_send_size(rank));
            VecI b_phases(traversal.cross_rank_sin_send_size(rank), 0);
            VecZ d_indices(traversal.cross_rank_sin_recv_size(rank));
            VecI sin_recv_phases(traversal.cross_rank_sin_recv_size(rank));

            traversal.for_each_cross_rank_sin_send_range(
                rank,
                0,
                traversal.cross_rank_sin_send_size(rank),
                [&](size_t logical_idx, size_t value_idx) { sin_send_indices[logical_idx] = value_idx; });
            traversal.for_each_cross_rank_sin_recv_range(rank,
                                                  0,
                                                  traversal.cross_rank_sin_recv_size(rank),
                                                  [&](size_t logical_idx, size_t value_idx, int phase) {
                                                      d_indices[logical_idx] = value_idx;
                                                      sin_recv_phases[logical_idx] = phase;
                                                  });

            b_data.emplace_back(std::move(sin_send_indices), std::move(b_phases));
            d_data.emplace_back(std::move(d_indices), std::move(sin_recv_phases));
        }
        // cos is no longer stored per-layer (the main path moves it out transiently and the
        // replay paths recompute from inverted indexes). Recompute the full cosine index set here from
        // the persistent even-parity inverted index fold using this layer's recompute metadata
        // (generator_words + scaled_count). graph_ is the main graph (single-inverted index fold).
        VecZ cos_inds;
        const auto &gw = traversal.generator_words();
        if (!gw.empty()) {
            profiling::ScopedRegion prof_cos(profiling::Region::CosRecompute);
            MajoranaSet<NumModes> gen{};
            std::memcpy(gen.data(), gw.data(), gw.size() * sizeof(uint64_t));
            auto p = detail::make_fold_cache<NumModes>(mp_op_.inverted_index(), gen, traversal.scaled_count());
            cos_inds = detail::fold_to_indices<NumModes>(p);
        }
        layers.emplace_back(std::move(cos_inds), std::move(local_cyc_data), std::move(b_data), std::move(d_data));
    }
    return layers;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::regenerate_cutoff_fn_() -> void {
    if (basis_change_.has_value()) {
        MajoranaVector<NumModes> basis;
        basis.reserve(2 * logical_num_modes_);
        for (size_t i = 0; i < 2 * logical_num_modes_; ++i) {
            basis.push_back(indices_to_bitset<NumModes>(basis_change_.value()[i]));
        }
        cutoff_fn_ = detail::cutoff_function_basis_change<NumModes>(cutoff_type_, cutoff_, basis, logical_num_modes_);
    }
    else {
        cutoff_fn_ = detail::cutoff_function<NumModes>(cutoff_type_, cutoff_, logical_num_modes_);
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::initialize_operator_caches_() -> void {
    // Lazy-cosine barrier: materialize the active picture's coefficients (replay the firing log, write
    // back each term's true value, reset stamps), then empty the log so every external reader —
    // expectation, get_operator, deepcopy — sees a plain eager coeff vector, and the next step's
    // reconstruction windows restart at one Trotter step. This is the invariant that keeps the whole
    // public API oblivious to the frame: frames are empty at every boundary. Runs BEFORE get_operator().
    // materialize_all also rebuilds the magnitude byte from the now-exact coeffs (restores selectivity
    // the shrinking cosines left loose). The coeff vector already has size mp_op_.size() here (grown by
    // extend during the build); resize is a defensive no-op.
    {
        detail::CoeffFrame<NumModes> &frame = schrodinger_ ? mp_op_.state_frame : mp_op_.op_frame;
        VecD &lazy = schrodinger_ ? mp_op_.state_coeffs : mp_op_.op_coeffs;
        if (frame.active() || !frame.mag.empty()) {
            lazy.resize(mp_op_.size(), 0.0);
            frame.materialize_all(*mp_op_.store, lazy, detail::kFrameUseMagByte);
        }
    }

    // Pre-warm the lazy operator/state/inverted index caches (results discarded) so later eval-time
    // recompute hits them already built, then trim the now-stable coeff vectors' slack.
    (void)mp_op_.get_operator();
    (void)mp_op_.get_state();
    (void)mp_op_.inverted_index();
    mp_op_.op_coeffs.shrink_to_fit();
    mp_op_.state_coeffs.shrink_to_fit();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::current_picture_coeffs_() -> const VecD & {
    return schrodinger_ ? mp_op_.get_state() : mp_op_.get_operator();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::extend_coeffs_from_current_picture_if_needed_(VecD &coeffs) -> void {
    if (coeffs.size() >= mp_op_.size()) {
        return;
    }

    const auto &current = current_picture_coeffs_();
    if (&coeffs == &current) {
        return;
    }

    if (coeffs.size() < current.size()) {
        coeffs.insert(coeffs.end(), current.begin() + static_cast<std::ptrdiff_t>(coeffs.size()), current.end());
    }
    coeffs.resize(mp_op_.size(), 0.0);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_mode_build_graph_(const std::vector<VecZ> &majoranas,
                                                            const VecZ &parameter_mapping,
                                                            const VecD &gen_coeffs,
                                                            const VecZ &gate_indices,
                                                            int only_rotate_len_k) -> void {
    const auto majoranas_size = majoranas.size();
    propagate_with_timing_(
        majoranas,
        only_rotate_len_k,
        [this, &parameter_mapping, &gen_coeffs, &gate_indices, majoranas_size](const VecZ &maj, int rot_len, size_t i) {
            const auto idx = !schrodinger_ ? majoranas_size - 1 - i : i;
            propagate_one_(maj,
                           rot_len,
                           std::nullopt,
                           std::nullopt,
                           parameter_mapping[idx],
                           gen_coeffs[idx],
                           gate_indices[idx]);
        });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_mode_graph_with_coeffs_(const std::vector<VecZ> &majoranas,
                                                                  const VecZ &parameter_mapping,
                                                                  const VecD &gen_coeffs,
                                                                  const VecZ &gate_indices,
                                                                  const VecD &parameters,
                                                                  const VecD &operator_coeffs,
                                                                  int only_rotate_len_k) -> void {
    auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, 1.0);
    auto coeffs = operator_coeffs;
    const auto majoranas_size = majoranas.size();

    propagate_with_timing_(
        majoranas,
        only_rotate_len_k,
        [this, &parameter_mapping, &gen_coeffs, &gate_indices, &mapped_params, &coeffs, majoranas_size](
            const VecZ &maj, int rot_len, size_t i) {
            const auto idx = !schrodinger_ ? majoranas_size - 1 - i : i;
            const auto [build_angle, apply_angle] = gate_angle_(mapped_params, i, majoranas_size);
            // The cos word list is no longer persisted on the layer; the builder MOVES it out transiently
            // (the per-layer recompute metadata still rides on the storage and is appended to the graph).
            // The immediate evolve_step scales that transient word list in parallel via scale_cos_mask,
            // rather than reading the layer's (now empty) stored cos_data. Gate info (param index, gen
            // coeff, gate index) is recorded on the layer here so the graph owns it and evaluation needs
            // only the variational parameters.
            auto cos = std::make_shared<CosMask>();
            auto storage = build_evolve_result_(maj, rot_len, std::cref(coeffs), build_angle, cos.get());
            graph_.append(storage, parameter_mapping[idx], gen_coeffs[idx], gate_indices[idx]);

            extend_coeffs_from_current_picture_if_needed_(coeffs);

            Layer layer(std::move(storage));
            detail::LayerCosScale cos_scale = [cos](size_t, double *c, double v) {
                detail::scale_cos_mask(c, *cos, v); // parallel; build-produced list is 64-aligned & disjoint
            };
            evolve_step(coeffs, layer, apply_angle, cos_scale, comm_);
        });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_mode_contract_immediately_(const std::vector<VecZ> &majoranas,
                                                                     const VecZ &parameter_mapping,
                                                                     const VecD &gen_coeffs,
                                                                     const VecD &parameters,
                                                                     int only_rotate_len_k) -> void {
    auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, 1.0);
    VecD *op_coeffs = schrodinger_ ? &mp_op_.state_coeffs : &mp_op_.op_coeffs;
    *op_coeffs = current_picture_coeffs_();
    // Lazy cosine IS this mode's cos implementation — the frame is the active picture's, no env gate,
    // and it runs at every rank count (build_layer + the fused resolver reconstruct the cross-rank
    // partner/local values before the log append; the wire still carries true pre-cos values as before).
    detail::CoeffFrame<NumModes> *framep = schrodinger_ ? &mp_op_.state_frame : &mp_op_.op_frame;
    const auto majoranas_size = majoranas.size();
    // ContractImmediately is a SINGLE fused contraction path at all rank counts. build_evolve_result_
    // emits rotation records (self-rank full rotations + R>1 cross-rank half rotations, the latter
    // carrying partner values via the build-time exchange) plus the inserted-endpoint cos directly — no
    // transient LayerCore (storage is nullptr) — and apply_fused_contract applies them in place,
    // replacing the old build_layer + evolve_step. cos is consumed synchronously, so a plain local suffices.
    propagate_with_timing_(
        majoranas,
        only_rotate_len_k,
        [this, &mapped_params, op_coeffs, framep, majoranas_size](const VecZ &maj, int rot_len, size_t i) {
            const auto [build_angle, apply_angle] = gate_angle_(mapped_params, i, majoranas_size);
            // build_evolve_result_ performs the self-rank operator inserts that grow the operator;
            // extend_coeffs must run AFTER that grow and BEFORE the apply.
            CosMask cos;
            detail::FusedContract fc;
            // Lazy cosine engages only at k==0 once the operator outgrows the last-level cache. build_layer
            // owns that decision (and the frame setup) and reports it back through `implicit`, so the apply
            // below drives frozen-vs-eager coeffs from the SAME decision the build used — they cannot disagree.
            bool implicit = false;
            build_evolve_result_(maj, rot_len, std::cref(*op_coeffs), build_angle, &cos, &fc, &implicit);
            // In the lazy frame build_layer already appended this firing to the log; nfirings is its
            // post-append size (the epoch freshly-touched/inserted terms stamp at).
            const uint32_t nf = implicit ? static_cast<uint32_t>(framep->nfirings) : 0;
            {
                profiling::ScopedRegion prof_ext(profiling::Region::Extend);
                extend_coeffs_from_current_picture_if_needed_(*op_coeffs);
                // Grow the frame's per-term arrays over the freshly-inserted terms: they are exact NOW
                // (born this firing), so stamp them at nf and take their byte from the fresh coeff.
                if (implicit) {
                    framep->extend_new_terms(*op_coeffs, nf);
                }
            }
            detail::FrameRefs refs;
            if (implicit) {
                refs.stamp = framep->stamp.data();
                refs.mag = framep->mag.empty() ? nullptr : framep->mag.data();
                refs.nfirings = nf;
            }
            detail::apply_fused_contract(fc, *op_coeffs, cos, apply_angle, schrodinger_, refs);
            // Firing cap: bound the reconstruction window. A single propagate() with thousands of gates
            // (e.g. Pauli/kicked-Ising, all layers in one call) would otherwise grow the log unboundedly,
            // making per-gate reconstruction cost quadratic in the gate count. Flushing at the cap keeps
            // windows short. Workloads that call propagate() per Trotter step (Hubbard, ~476 firings/call)
            // never reach the cap before their per-step barrier, so their measured win is unaffected.
            if (implicit && framep->nfirings >= detail::kFrameFlushFirings) {
                framep->materialize_all(*mp_op_.store, *op_coeffs, detail::kFrameUseMagByte);
            }
        });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::build_graph(const std::vector<VecZ> &majoranas,
                                               const VecZ &parameter_mapping,
                                               const VecD &gen_coeffs,
                                               std::optional<VecZ> gate_indices,
                                               std::optional<VecD> parameters,
                                               int only_rotate_len_k) -> void {
    if (majoranas.empty()) {
        return;
    }
    validate_coefficient_lengths(parameter_mapping, gen_coeffs);

    // Resolve gate indices: default to one gate per generator (iota). These are local and
    // 0-based per call; offset by the gate count already in the graph so they are absolute.
    VecZ local_gates;
    if (gate_indices.has_value()) {
        local_gates = std::move(*gate_indices);
    }
    else {
        local_gates.resize(majoranas.size());
        std::iota(local_gates.begin(), local_gates.end(), size_t{0});
    }
    validate_gate_indices(local_gates, majoranas.size());
    const size_t gate_offset = n_gates();
    for (auto &g : local_gates) {
        g += gate_offset;
    }

    if (!parameters.has_value()) {
        // Pure structural build: append layers recording their gate information. Run inside a
        // persistent, loop-scoped task arena so TBB workers stay attached across the (often many)
        // small per-gate parallel regions instead of parking/waking between them.
        tbb::task_arena arena(static_cast<int>(threading::effective_parallelism()));
        arena.execute(
            [&] { evolve_mode_build_graph_(majoranas, parameter_mapping, gen_coeffs, local_gates, only_rotate_len_k); });
    }
    else {
        // Guard the coefficient-informed path the same way propagate() guards its parameters:
        // evolve_mode_graph_with_coeffs_ -> map_params()/fill_mapped_params() index
        // `parameters` by parameter_mapping, so a too-short vector reads out of bounds.
        validate_parameters_length(*parameters, parameter_mapping);
        // Coefficient-informed build: regenerate the seed by contracting the existing graph
        // (replacing the former operator_coeffs input), then build+contract the new layers into it
        // so atol truncation sees realistic coefficients. The existing graph references the
        // parameter prefix [0, m); slice `parameters` to that so the exact-length check passes.
        VecD seed;
        if (graph_layers() > 0) {
            const auto existing = graph_gate_arrays_();
            const size_t m = expected_num_params(existing.first);
            const VecD existing_params(
                parameters->begin(),
                parameters->begin() + static_cast<std::ptrdiff_t>(std::min(m, parameters->size())));
            seed = contract_partially(existing_params, false);
        }
        else {
            seed = current_picture_coeffs_();
        }
        // Build inside the same loop-scoped task arena as the structural path; the MPI-driven seed
        // regeneration above already ran outside it.
        tbb::task_arena arena(static_cast<int>(threading::effective_parallelism()));
        arena.execute([&] {
            evolve_mode_graph_with_coeffs_(majoranas,
                                           parameter_mapping,
                                           gen_coeffs,
                                           local_gates,
                                           *parameters,
                                           seed,
                                           only_rotate_len_k);
        });
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::propagate(const std::vector<VecZ> &majoranas,
                                             const VecZ &parameter_mapping,
                                             const VecD &gen_coeffs,
                                             const VecD &parameters,
                                             int only_rotate_len_k) -> void {
    if (majoranas.empty()) {
        return;
    }
    validate_coefficient_lengths(parameter_mapping, gen_coeffs);
    validate_parameters_length(parameters, parameter_mapping);
    // propagate() evolves and contracts in place assuming no stored graph: its contract loop
    // slices and consumes graph layers from the front (see evolve_mode_contract_immediately_),
    // so running it on a graph built by build_graph() would consume those pre-existing layers
    // and silently corrupt the result. Reject it -- build_graph() is the extend path, and
    // contract_partially() folds an existing graph.
    if (graph_layers() > 0) {
        throw std::runtime_error(std::format("Cannot propagate() on top of a non-empty graph of {} layer(s): "
                                             "propagate() evolves and contracts in place and assumes no stored graph. "
                                             "Call contract_partially() to fold the existing graph first, or use "
                                             "build_graph() to extend it.",
                                             graph_layers()));
    }
    // Contract inside a persistent, loop-scoped task arena so TBB workers stay attached across the
    // (often many) small per-gate parallel regions rather than parking/waking between them.
    tbb::task_arena arena(static_cast<int>(threading::effective_parallelism()));
    arena.execute(
        [&] { evolve_mode_contract_immediately_(majoranas, parameter_mapping, gen_coeffs, parameters, only_rotate_len_k); });
}

template <size_t NumModes>
template <typename EvolutionFunc>
auto MonomialPropagator<NumModes>::propagate_with_timing_(const std::vector<VecZ> &majoranas,
                                                          int only_rotate_len_k,
                                                          EvolutionFunc evolution_func) -> void {
    for (size_t i = 0; i < majoranas.size(); ++i) {
        const auto idx = !schrodinger_ ? majoranas.size() - 1 - i : i;
        const auto &maj = majoranas[idx];

        evolution_func(maj, only_rotate_len_k, i);
    }

    initialize_operator_caches_();
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::build_evolve_result_(const VecZ &gen_vec,
                                                        int only_rotate_len_k,
                                                        std::optional<std::reference_wrapper<const VecD>> coeffs,
                                                        std::optional<double> param,
                                                        CosMask *out_cos,
                                                        detail::FusedContract *fused_contract,
                                                        bool *engaged_frame)
    -> std::shared_ptr<LayerCore> {
    const auto gen_maj = indices_to_bitset<NumModes>(gen_vec);

    // Unified build pass (paper Algorithm 2). Both parities go through the
    // parity-corrected fastpath inverted index scan + pivot-bit leader/follower split
    // (odd generators apply the g_odd parity(|M|) correction in the fold). The per-layer recompute
    // metadata (generator words + scaled_count) is written onto the returned LayerCore by the
    // builder, so it travels with the layer through every graph transform.
    return detail::build_layer<NumModes>(mp_op_,
                                                     gen_maj,
                                                     cutoff_fn_,
                                                     lower_atol_,
                                                     coeffs,
                                                     upper_atol_,
                                                     param,
                                                     only_rotate_len_k,
                                                     matched_scratch_,
                                                     comm_,
                                                     out_cos,
                                                     fused_contract,
                                                     schrodinger_,
                                                     engaged_frame);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::propagate_one_(const VecZ &gen_vec,
                                                  int only_rotate_len_k,
                                                  std::optional<std::reference_wrapper<const VecD>> coeffs,
                                                  std::optional<double> param,
                                                  size_t param_index,
                                                  double gen_coeff,
                                                  size_t gate_index) -> void {
    // The per-layer recompute metadata (generator words + scaled_count) is stored ON the layer's
    // LayerCore by the builder, so it travels with the layer through every graph transform — no
    // separate lockstep append is needed here. Gate info (param index, gen coeff, gate index) is
    // recorded on the layer so the graph owns it and evaluation needs only the variational parameters.
    graph_.append(build_evolve_result_(gen_vec, only_rotate_len_k, coeffs, param), param_index, gen_coeff, gate_index);
}

// Defined below; forward-declared so the operator-evolution replay can build its scale callback
// through the same budget-honoring path as the energy/gradient functionals.
template <size_t NumModes>
auto build_cos_callbacks(const detail::InvertedIndex<NumModes> &inverted_index, const MPGraphView &graph)
    -> std::pair<detail::LayerCosScale, detail::LayerCosAccumulate>;

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_operator_with_recompute_(VecD &&coeffs,
                                                                   const MPGraphView &graph,
                                                                   const VecD &params) -> VecD {
    const auto &inverted_index = mp_op_.inverted_index();
    // Only the scale side is consumed (replay applies rotations to coeffs); build the pair through
    // the shared builder so the fold-cache budget gate is honored here too.
    auto cos_scale = build_cos_callbacks<NumModes>(inverted_index, graph).first;
    return evolve_operator(std::move(coeffs), graph, params, cos_scale, comm_);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::n_gates() const -> size_t {
    const size_t count = graph_.layers();
    size_t max_gate = 0;
    bool any = false;
    for (size_t layer = 0; layer < count; ++layer) {
        const size_t g = graph_.get_layer_traversal(layer).gate_index();
        max_gate = any ? std::max(max_gate, g) : g;
        any = true;
    }
    return any ? max_gate + 1 : 0;
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::set_parameter_mapping(const VecZ &parameter_mapping) -> void {
    const size_t count = graph_.layers();
    const size_t gates = n_gates();

    // Relabel one layer's parameter index. The LayerCore is a shared immutable core (sliced/unioned
    // graphs may share it), so relabeling copies the core, sets the new parameter index, and replaces
    // the layer's core in place — preserving generator coeff, gate index, and any stored pruned cos.
    auto relabel = [this](size_t layer, size_t new_param_index) {
        auto &target = graph_.get_layer(layer);
        auto new_core = std::make_shared<LayerCore>(target.core());
        new_core->param_index = new_param_index;
        if (const CosMask *pruned = target.pruned_cos()) {
            target = Layer(std::move(new_core), *pruned);
        }
        else {
            target = Layer(std::move(new_core));
        }
    };

    if (parameter_mapping.size() == count) {
        // Per-layer mapping in optimizer order; layer `layer` holds optimizer index
        // count-1-layer (see graph_gate_arrays_).
        for (size_t layer = 0; layer < count; ++layer) {
            relabel(layer, parameter_mapping[count - 1 - layer]);
        }
    }
    else if (parameter_mapping.size() == gates) {
        // Per-gate mapping indexed by absolute gate index: relabel each layer via its own
        // stored gate index (order-agnostic, correct in both pictures and across builds).
        for (size_t layer = 0; layer < count; ++layer) {
            relabel(layer, parameter_mapping[graph_.get_layer(layer).gate_index()]);
        }
    }
    else {
        throw std::runtime_error(std::format("parameter_mapping has {} entries; expected {} (per graph "
                                             "layer) or {} (per gate).",
                                             parameter_mapping.size(),
                                             count,
                                             gates));
    }
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::graph_gate_arrays_() const -> std::pair<VecZ, VecD> {
    const size_t count = graph_.layers();
    VecZ parameter_mapping(count);
    VecD gen_coeffs(count);
    // Layers store gate info in simulation order; the evaluation machinery expects it in
    // optimizer order (the reverse), so place layer `layer` at optimizer index count-1-layer.
    for (size_t layer = 0; layer < count; ++layer) {
        const auto traversal = graph_.get_layer_traversal(layer);
        const size_t optimizer_index = count - 1 - layer;
        parameter_mapping[optimizer_index] = traversal.param_index();
        gen_coeffs[optimizer_index] = traversal.gen_coeff();
    }
    return {std::move(parameter_mapping), std::move(gen_coeffs)};
}

// Build the per-layer (scale, accumulate) cos callbacks for a replayed graph. Handles all three
// layer kinds uniformly, so the pare and non-pare functional paths share ONE implementation:
//   - PRUNED  (pruned_cos() != nullptr): cos is a stored filtered CosMask, scaled in parallel;
//   - FOLD, cached (below the memory budget): a FoldCache buffer per layer;
//   - FOLD, recompute (above the budget): a LazyFold (no buffer; fused/blocked recompute).
// The non-pare graph simply has no pruned layers. The returned closures capture only the cache +
// inverted index pointer + recompute flag; the GRAPH's lifetime is owned by the caller's functional capture.
template <size_t NumModes>
auto build_cos_callbacks(const detail::InvertedIndex<NumModes> &inverted_index, const MPGraphView &graph)
    -> std::pair<detail::LayerCosScale, detail::LayerCosAccumulate> {
    // Memory-budget gate: Σ mask_words · 8 B for the fold layers. Below it caching is cheapest; above
    // it the cache is a multi-GB cold burden, so recompute each fold on the fly (bit-identical).
    size_t recompute_cache_words = 0;
    for (size_t i = 0; i < graph.layers(); ++i) {
        const auto &l = graph.get_layer(i);
        if (l.pruned_cos() == nullptr) {
            recompute_cache_words += std::min(inverted_index.words(), static_cast<size_t>((l.scaled_count() + 63) / 64));
        }
    }
    const bool recompute = recompute_cache_words * sizeof(uint64_t) > detail::recompute_cache_budget_bytes();
    if (recompute) {
        inverted_index.ensure_sorted_columns();
    }

    struct LayerCos {
        bool recomputes_cos = false;
        detail::FoldCache<NumModes> combined{}; // used iff recomputes_cos && !recompute
        detail::LazyFold<NumModes> recipe{};   // used iff recomputes_cos && recompute
        const CosMask *filtered = nullptr; // points into a pruned layer's stored cos
    };
    auto cache = std::make_shared<std::vector<LayerCos>>();
    cache->reserve(graph.layers());
    for (size_t i = 0; i < graph.layers(); ++i) {
        const auto &layer = graph.get_layer(i);
        LayerCos entry;
        if (const CosMask *pruned = layer.pruned_cos(); pruned != nullptr) {
            entry.recomputes_cos = false;
            entry.filtered = pruned;
        }
        else {
            entry.recomputes_cos = true;
            const auto gen = detail::generator_from_words<NumModes>(layer.generator_words());
            if (recompute) {
                entry.recipe = detail::make_lazy_fold<NumModes>(inverted_index, gen, layer.scaled_count());
            }
            else {
                entry.combined = detail::make_fold_cache<NumModes>(inverted_index, gen, layer.scaled_count());
            }
        }
        cache->push_back(std::move(entry));
    }

    const auto *sc = &inverted_index;
    detail::LayerCosScale cos_scale = [cache, sc, recompute](size_t i, double *c, double v) {
        const auto &e = (*cache)[i];
        if (!e.recomputes_cos) {
            detail::scale_cos_mask(c, *e.filtered, v);
        }
        else if (recompute) {
            detail::scale_cos_lazy<NumModes>(*sc, e.recipe, c, v);
        }
        else {
            detail::scale_cos_cached<NumModes>(e.combined, c, v);
        }
    };
    detail::LayerCosAccumulate cos_acc = [cache, sc, recompute](size_t i, double *s, double *h, double v, double sec) {
        const auto &e = (*cache)[i];
        if (!e.recomputes_cos) {
            return detail::accumulate_cos_mask(s, h, *e.filtered, v, sec);
        }
        if (recompute) {
            return detail::accumulate_cos_lazy<NumModes>(*sc, e.recipe, s, h, v, sec);
        }
        return detail::accumulate_cos_cached<NumModes>(e.combined, s, h, v, sec);
    };
    return {std::move(cos_scale), std::move(cos_acc)};
}

template <size_t NumModes>
template <typename Fn, typename R>
auto MonomialPropagator<NumModes>::make_functional_(Fn &&func, std::optional<double> pare_threshold)
    -> std::function<R(const VecD &)> {
    auto gate_arrays = graph_gate_arrays_();
    auto parameter_mapping = std::move(gate_arrays.first);
    auto gen_coeffs = std::move(gate_arrays.second);
    const auto num_params = expected_num_params(parameter_mapping);

    VecD state = mp_op_.get_state();
    VecD op = mp_op_.get_operator();
    const auto core_term = this->core_term();
    const auto comm = comm_;

    const auto expected_layers = graph_layers();
    const auto &inverted_index = mp_op_.inverted_index();

    // Resolve the graph the functional replays, as ONE owning handle so the pare and non-pare paths
    // share a single tail (build cos callbacks → capture into the functional):
    //   • pare: a streaming backward keep-set sweep (get_pared_graph) produces a typed-layer MPGraph —
    //     each layer a FoldLayer (cos recomputed from the fold) or a PrunedLayer (cos trimmed, stored).
    //     It is heap-owned via shared_ptr so build_cos_callbacks can hold raw pointers into each
    //     PrunedLayer's stored cos with no copy and no dangling: the functional captures this same
    //     shared_ptr, so every copy of the std::function keeps the graph (and those cos pointers) alive.
    //   • non-pare: an ALIASING (non-owning) shared_ptr to graph_ — identical lifetime to capturing
    //     &graph_ by reference (graph_ must outlive the functional either way), but it lets the tail
    //     capture one `graph` uniformly. graph_ has no pruned layers, so every layer takes the fold path.
    // The pared graph keeps the original layer count, so validate_expected_graph_layers is identical.
    std::shared_ptr<const MPGraph> graph;
    if (pare_threshold.has_value()) {
        auto full_cos_of_layer = [this, &inverted_index](size_t i) -> CosMask {
            const auto &layer = graph_.get_layer(i);
            const auto gen = detail::generator_from_words<NumModes>(layer.generator_words());
            const auto combined = detail::make_fold_cache<NumModes>(inverted_index, gen, layer.scaled_count());
            return detail::fold_to_cos_mask<NumModes>(combined);
        };
        graph = std::make_shared<const MPGraph>(
            get_pared_graph(state, op, *pare_threshold, graph_, schrodinger_, comm_, full_cos_of_layer));
    }
    else {
        graph = std::shared_ptr<const MPGraph>(std::shared_ptr<const void>{}, &graph_);
    }

    // Per-layer cos callbacks (fold-cached / fold-recompute / stored filtered cos — see
    // build_cos_callbacks). The inverted index is persistent (pre-warmed in initialize_operator_caches_)
    // and outlives this simulator; the folds reference its dense columns by pointer and own their sparse
    // columns, so they stay valid for the captured closure.
    auto callbacks = build_cos_callbacks<NumModes>(inverted_index, graph->replay_view());
    detail::LayerCosScale cos_scale = std::move(callbacks.first);
    detail::LayerCosAccumulate cos_acc = std::move(callbacks.second);

    return make_parameter_validated_functional(
        num_params,
        [func = std::move(func),
         core_term,
         state = std::move(state),
         op = std::move(op),
         graph = std::move(graph),
         parameter_mapping,
         gen_coeffs,
         expected_layers,
         cos_scale = std::move(cos_scale),
         cos_acc = std::move(cos_acc),
         comm](const VecD &params) -> R {
            validate_expected_graph_layers(graph->layers(), expected_layers);
            return func(core_term, state, op, parameter_mapping, gen_coeffs, *graph, params, comm, cos_scale, cos_acc);
        });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_functional(std::optional<double> pare_threshold)
    -> std::function<double(const VecD &)> {
    return make_functional_(ev_fn, pare_threshold);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_and_gradient_functional(std::optional<double> pare_threshold)
    -> std::function<std::pair<double, VecD>(const VecD &)> {
    return make_functional_(ev_and_grad_fn, pare_threshold);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value(const VecD &parameters) -> double {
    return expectation_value_functional(std::nullopt)(parameters);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::expectation_value_and_gradient(const VecD &parameters) -> std::pair<double, VecD> {
    return expectation_value_and_gradient_functional(std::nullopt)(parameters);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::contract_partially(const VecD &parameters, bool inplace) -> VecD {
    const auto gate_arrays = graph_gate_arrays_();
    const auto &parameter_mapping = gate_arrays.first;
    const auto &gen_coeffs = gate_arrays.second;
    validate_parameters_length(parameters, parameter_mapping);

    if (parameters.empty()) {
        return current_picture_coeffs_();
    }

    const size_t num_majoranas = parameter_mapping.size();
    // Main-built layers no longer store the cosine bitmap, so the replay recomputes each layer's cosine
    // set from the persistent inverted-index fold (evolve_operator_with_recompute_). Both replay paths
    // take an MPGraphView. Inplace contraction slices into an owned MPGraph, so it must be bound to a
    // named local before viewing (never view a temporary); the non-inplace path's slice_view() already
    // returns a view over this graph's still-live layers.
    if (schrodinger_) {
        const auto &state = mp_op_.get_state();
        const auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, -1.0);
        VecD evolved_state;
        if (inplace) {
            const MPGraph sliced = graph_.slice_graph(num_majoranas, true);
            evolved_state = evolve_operator_with_recompute_(VecD(state), sliced.replay_view(), mapped_params);
            mp_op_.state_coeffs = evolved_state;
        }
        else {
            evolved_state =
                evolve_operator_with_recompute_(VecD(state), graph_.slice_view(num_majoranas), mapped_params);
        }
        return evolved_state;
    }

    const auto &op = mp_op_.get_operator();
    const auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, 1.0, true);
    VecD evolved_op;
    if (inplace) {
        const MPGraph sliced = graph_.slice_graph(num_majoranas, true);
        evolved_op = evolve_operator_with_recompute_(VecD(op), sliced.replay_view(), mapped_params);
        mp_op_.op_coeffs = evolved_op;
    }
    else {
        evolved_op = evolve_operator_with_recompute_(VecD(op), graph_.slice_view(num_majoranas), mapped_params);
    }
    return evolved_op;
}

} // namespace monoprop
