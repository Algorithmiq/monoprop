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
#include "monoprop/detail/evolution/CosineRecompute.h"
#include "monoprop/detail/evolution/DistributedLayerBuilder.h"

namespace monoprop {

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
    // Pre-warm the lazy operator/state/sidecar caches (results discarded) so later eval-time recompute
    // hits them already built, then trim the now-stable coeff vectors' slack.
    (void)mp_op_.get_operator();
    (void)mp_op_.get_state();
    (void)mp_op_.even_parity_scan_sidecar();
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
auto MonomialPropagator<NumModes>::evolve_mode_graph_only_(const std::vector<VecZ> &majoranas, int only_rotate_len_k)
    -> void {
    propagate_with_timing_(majoranas, only_rotate_len_k, [this](const VecZ &maj, int rot_len, size_t) {
        propagate_one_(maj, rot_len);
    });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::evolve_mode_graph_with_coeffs_(const std::vector<VecZ> &majoranas,
                                                                  const VecZ &parameter_mapping,
                                                                  const VecD &gen_coeffs,
                                                                  const VecD &parameters,
                                                                  const VecD &operator_coeffs,
                                                                  int only_rotate_len_k) -> void {
    auto mapped_params = map_params(parameters, parameter_mapping, gen_coeffs, 1.0);
    auto coeffs = operator_coeffs;
    const auto majoranas_size = majoranas.size();

    propagate_with_timing_(
        majoranas,
        only_rotate_len_k,
        [this, &mapped_params, &coeffs, majoranas_size](const VecZ &maj, int rot_len, size_t i) {
            const auto idx = !schrodinger_ ? majoranas_size - 1 - i : i;
            // The cos word list is no longer persisted on the layer; the builder MOVES it out transiently
            // (the per-layer recompute metadata still rides on the storage and is appended to the graph).
            // The immediate evolve_step scales that transient word list in parallel via scale_cos_words,
            // rather than reading the layer's (now empty) stored cos_data.
            auto cos = std::make_shared<CosineWordList>();
            auto storage = build_evolve_result_(maj, rot_len, std::cref(coeffs), mapped_params[idx], cos.get());
            graph_.append(storage);

            const auto param = schrodinger_ ? -mapped_params[idx] : mapped_params[idx];
            extend_coeffs_from_current_picture_if_needed_(coeffs);

            Layer layer(std::move(storage));
            detail::LayerCosScale cos_scale = [cos](size_t, double *c, double v) {
                detail::scale_cos_words(c, *cos, v); // parallel; build-produced list is 64-aligned & disjoint
            };
            evolve_step(coeffs, layer, param, cos_scale, comm_);
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
    const auto majoranas_size = majoranas.size();

    propagate_with_timing_(
        majoranas,
        only_rotate_len_k,
        [this, &mapped_params, op_coeffs, majoranas_size](const VecZ &maj, int rot_len, size_t i) {
            const auto idx = !schrodinger_ ? majoranas_size - 1 - i : i;
            // build_evolve_result_ performs the self-rank operator inserts that grow the operator;
            // extend_coeffs must run AFTER that grow and BEFORE evolve_step.
            // The cos word list is no longer persisted; the builder moves it out transiently and the
            // immediate evolve_step scales it in parallel via scale_cos_words.
            auto cos = std::make_shared<CosineWordList>();
            auto storage = build_evolve_result_(maj, rot_len, std::cref(*op_coeffs), mapped_params[idx], cos.get());

            extend_coeffs_from_current_picture_if_needed_(*op_coeffs);

            const auto param = schrodinger_ ? -mapped_params[idx] : mapped_params[idx];
            Layer layer(std::move(storage));
            detail::LayerCosScale cos_scale = [cos](size_t, double *c, double v) {
                detail::scale_cos_words(c, *cos, v); // parallel; build-produced list is 64-aligned & disjoint
            };
            evolve_step(*op_coeffs, layer, param, cos_scale, comm_);
        });
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::execute_evolution_mode_(EvolutionMode mode,
                                                           const std::vector<VecZ> &majoranas,
                                                           const std::optional<VecZ> &parameter_mapping,
                                                           const std::optional<VecD> &gen_coeffs,
                                                           const std::optional<VecD> &parameters,
                                                           const std::optional<VecD> &operator_coeffs,
                                                           int only_rotate_len_k) -> void {
    switch (mode) {
        case EvolutionMode::GraphOnly:
            evolve_mode_graph_only_(majoranas, only_rotate_len_k);
            break;
        case EvolutionMode::GraphWithCoeffs:
            evolve_mode_graph_with_coeffs_(majoranas,
                                           parameter_mapping.value(),
                                           gen_coeffs.value(),
                                           parameters.value(),
                                           operator_coeffs.value(),
                                           only_rotate_len_k);
            break;
        case EvolutionMode::ContractImmediately:
            evolve_mode_contract_immediately_(majoranas,
                                              parameter_mapping.value(),
                                              gen_coeffs.value(),
                                              parameters.value(),
                                              only_rotate_len_k);
            break;
        default:
            throw std::runtime_error("Unknown evolution mode.");
    }
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
                                                        CosineWordList *out_cos) -> std::shared_ptr<LayerCore> {
    const auto gen_maj = indices_to_bitset<NumModes>(gen_vec);

    // Unified build pass (paper Algorithm 2). Both parities go through the
    // parity-corrected fastpath sidecar scan + pivot-bit leader/follower split
    // (odd generators apply the g_odd parity(|M|) correction in the fold). The per-layer recompute
    // metadata (generator words + cos_count) is written onto the returned LayerCore by the
    // builder, so it travels with the layer through every graph transform.
    return detail::build_distributed_layer<NumModes>(mp_op_,
                                                     gen_maj,
                                                     cutoff_fn_,
                                                     lower_atol_,
                                                     coeffs,
                                                     upper_atol_,
                                                     param,
                                                     only_rotate_len_k,
                                                     comm_,
                                                     out_cos);
}

template <size_t NumModes>
auto MonomialPropagator<NumModes>::propagate_one_(const VecZ &gen_vec,
                                                  int only_rotate_len_k,
                                                  std::optional<std::reference_wrapper<const VecD>> coeffs,
                                                  std::optional<double> param) -> void {
    // The per-layer recompute metadata (generator words + cos_count) is stored ON the layer's
    // LayerCore by the builder, so it travels with the layer through every graph transform — no
    // separate lockstep append is needed here.
    graph_.append(build_evolve_result_(gen_vec, only_rotate_len_k, coeffs, param));
}

template <size_t NumModes>
template <typename GraphLike>
auto MonomialPropagator<NumModes>::evolve_operator_with_recompute_(VecD &&coeffs,
                                                                   const GraphLike &graph,
                                                                   const VecD &params) -> VecD {
    const auto &sidecar = mp_op_.even_parity_scan_sidecar();
    auto cache = detail::build_layer_cos_cache<NumModes>(sidecar, graph);
    detail::LayerCosScale cos_scale = detail::make_cos_scale_from_cache(cache);
    return evolve_operator(std::move(coeffs), graph, params, cos_scale, comm_);
}

template <size_t NumModes>
template <typename Fn, typename R>
auto MonomialPropagator<NumModes>::make_functional_(const VecZ &parameter_mapping,
                                                    const VecD &gen_coeffs,
                                                    Fn &&func,
                                                    std::optional<double> pare_threshold)
    -> std::function<R(const VecD &)> {
    validate_coefficient_lengths(parameter_mapping, gen_coeffs);
    validate_propagation_params(parameter_mapping.size(), graph_layers());

    const auto num_params = expected_num_params(parameter_mapping);

    VecD state = mp_op_.get_state();
    VecD op = mp_op_.get_operator();
    const auto core_term = this->core_term();
    const auto comm = comm_;

    if (pare_threshold.has_value()) {
        // ── Streaming pare: build a typed-layer MPGraph (FoldLayer / PrunedLayer) ──
        // pare_graph runs the backward keep-set sweep and materializes each ORIGINAL graph layer's
        // full cosine set LAZILY (one layer at a time) via the provider below — no up-front
        // materialization of all N layers (the old memory spike). Each emitted layer is either a
        // FoldLayer (cos unchanged → recomputed from the fold at replay, exactly like the non-pare
        // path) or a PrunedLayer (cos trimmed → stored explicitly). The pared graph reuses the
        // source layer cores (shared_ptr), so capturing it by value is cheap.
        const auto &sidecar = mp_op_.even_parity_scan_sidecar();
        auto full_cos_of_layer = [this, &sidecar](size_t i) -> CosineWordList {
            const auto &layer = graph_.get_layer(i);
            MajoranaSet<NumModes> gen{};
            const auto &gw = layer.generator_words();
            std::memcpy(gen.data(), gw.data(), gw.size() * sizeof(uint64_t));
            const auto fold = detail::make_prepared_fold<NumModes>(sidecar, gen, layer.cos_count());
            return detail::fold_to_cos_words<NumModes>(fold);
        };

        // Own the pared graph through a shared_ptr so the cos cache below can hold raw pointers into
        // each PrunedLayer's stored cos with NO copy and NO dangling: the functional captures this
        // same shared_ptr, so every copy of the (copyable) std::function keeps the one heap-resident
        // MPGraph — and therefore those cos pointers — alive. The graph object never moves.
        auto pared = std::make_shared<const MPGraph>(
            get_pared_graph(state, op, *pare_threshold, graph_, schrodinger_, comm_, full_cos_of_layer));
        const auto expected_layers = graph_layers();

        // ── Per-functional cos cache over the PARED graph ──
        //   - FoldLayer (pruned_cos() == nullptr): cos == the full anticommuting set → recompute from
        //     the sidecar fold (PreparedFold), exactly like the non-pare path.
        //   - PrunedLayer (pruned_cos() != nullptr): cos == a FILTERED SUBSET stored on the layer as a
        //     CosineWordList; scaled in parallel.
        // The sidecar is persistent for the simulator's lifetime; the folds reference its dense
        // columns by pointer and own their materialized sparse columns, so they outlive the closure.
        struct ParedLayerCos {
            bool is_fold = false;
            detail::PreparedFold<NumModes> fold{};
            const CosineWordList *filtered = nullptr; // points into the pared layer's stored cos
        };
        auto cache = std::make_shared<std::vector<ParedLayerCos>>();
        cache->reserve(pared->layers());
        for (size_t i = 0; i < pared->layers(); ++i) {
            const auto &pared_layer = pared->get_layer(i);
            ParedLayerCos entry;
            if (const CosineWordList *pruned = pared_layer.pruned_cos(); pruned != nullptr) {
                // Pruned layer: scale the stored filtered cos in PARALLEL (the pruned set is often
                // still large at modest pare thresholds; a serial per-bit walk regresses the pared
                // gradient ~3x). `pared` (a shared_ptr) is captured by the functional below, so this
                // pointer into the heap-resident graph stays valid for every copy of the closure.
                entry.is_fold = false;
                entry.filtered = pruned;
            }
            else {
                // Fold layer: fold the operator's sidecar truncated to the layer's POST-insert size,
                // reconstructing the generator MajoranaSet from the underlying LayerCore words.
                MajoranaSet<NumModes> gen{};
                const auto &gw = pared_layer.generator_words();
                std::memcpy(gen.data(), gw.data(), gw.size() * sizeof(uint64_t));
                entry.is_fold = true;
                entry.fold = detail::make_prepared_fold<NumModes>(sidecar, gen, pared_layer.cos_count());
            }
            cache->push_back(std::move(entry));
        }

        detail::LayerCosScale cos_scale = [cache](size_t i, double *c, double v) {
            if ((*cache)[i].is_fold) {
                detail::scale_cos_fold<NumModes>((*cache)[i].fold, c, v);
            }
            else {
                detail::scale_cos_words(c, *(*cache)[i].filtered, v);
            }
        };
        detail::LayerCosAccumulate cos_acc = [cache](size_t i, double *s, double *h, double v, double sec) {
            return (*cache)[i].is_fold ? detail::accumulate_cos_fold<NumModes>((*cache)[i].fold, s, h, v, sec)
                                       : detail::accumulate_cos_words(s, h, *(*cache)[i].filtered, v, sec);
        };

        // Route the pared graph through the SAME MPGraph replay path the non-pare path uses (func is
        // the generic ev / ev_and_grad lambda — it dispatches to the const MPGraph& overload).
        return make_parameter_validated_functional(num_params,
                                                    [func = std::move(func),
                                                     core_term,
                                                     state = std::move(state),
                                                     op = std::move(op),
                                                     graph = std::move(pared),
                                                     parameter_mapping,
                                                     gen_coeffs,
                                                     expected_layers,
                                                     cos_scale = std::move(cos_scale),
                                                     cos_acc = std::move(cos_acc),
                                                     comm](const VecD &params) -> R {
                                                        validate_expected_graph_layers(graph->layers(), expected_layers);
                                                        return func(core_term,
                                                                    state,
                                                                    op,
                                                                    parameter_mapping,
                                                                    gen_coeffs,
                                                                    *graph,
                                                                    params,
                                                                    comm,
                                                                    cos_scale,
                                                                    cos_acc);
                                                    });
    }

    const auto expected_layers = graph_layers();

    // ── Build the per-layer prepared-fold cache + recompute callbacks (non-pare path) ──
    // The sidecar is persistent (pre-warmed in initialize_operator_caches_) and lives as long as this
    // simulator; the prepared folds reference its dense columns by pointer and own their materialized
    // sparse columns, so they stay valid for the captured closure's lifetime. cos is no longer stored —
    // the callbacks recompute the full cosine set from the sidecar fold at eval time.
    const auto &sidecar = mp_op_.even_parity_scan_sidecar();
    auto prepared = std::make_shared<std::vector<detail::PreparedFold<NumModes>>>();
    prepared->reserve(graph_.layers());
    for (size_t i = 0; i < graph_.layers(); ++i) {
        const auto &layer = graph_.get_layer(i);
        MajoranaSet<NumModes> gen{};
        const auto &gw = layer.generator_words();
        std::memcpy(gen.data(), gw.data(), gw.size() * sizeof(uint64_t));
        // cos_count is the layer's POST-insert operator size (build_distributed_layer): the cosine set
        // is "all anticommuting" including this layer's inserted rotation-target endpoints, so folding
        // the full operator's sidecar truncated to cos_count reproduces the full cosine set exactly
        // (both pictures). cos is not stored — this fold recomputes it at eval time.
        prepared->push_back(detail::make_prepared_fold<NumModes>(sidecar, gen, layer.cos_count()));
    }

    detail::LayerCosScale cos_scale = [prepared](size_t i, double *c, double v) {
        detail::scale_cos_fold<NumModes>((*prepared)[i], c, v);
    };
    detail::LayerCosAccumulate cos_acc = [prepared](size_t i, double *s, double *h, double v, double sec) {
        return detail::accumulate_cos_fold<NumModes>((*prepared)[i], s, h, v, sec);
    };

    return make_parameter_validated_functional(
        num_params,
        [func = std::move(func),
         core_term,
         state,
         op,
         &graph = graph_,
         parameter_mapping,
         gen_coeffs,
         expected_layers,
         cos_scale = std::move(cos_scale),
         cos_acc = std::move(cos_acc),
         comm](const VecD &params) -> R {
            validate_expected_graph_layers(graph.layers(), expected_layers);
            return func(core_term, state, op, parameter_mapping, gen_coeffs, graph, params, comm, cos_scale, cos_acc);
        });
}

} // namespace monoprop
