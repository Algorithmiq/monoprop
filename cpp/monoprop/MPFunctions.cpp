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

#include "monoprop/MPFunctions.h"

#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "monoprop/Evolution.h"
#include "monoprop/detail/evolution/CosineRecomputeCallbacks.h"

namespace monoprop {

// A per-layer cosine callback this evaluation path needs was left empty.
class MissingLayerCallback : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

// A state's rows and values disagree in length, a sparse row names a slot outside the state, or the
// operator being contracted is shorter than the state. One type because every case is a caller-supplied
// length or index that does not fit the rest of the call.
class EvalStateArgumentError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

namespace {

// What a layer's forward pass keeps for the reverse pass, as flags on cos_wanted.
constexpr uint8_t kRecordRotationsBelow = 1; ///< the coefficients the layer below rotates
constexpr uint8_t kRecordCosineSet = 2;      ///< its own cosine set, which a vanishing cosine destroys

// Per-thread scratch so repeated ev/grad calls stay allocation-light.
struct EvalScratch {
    VecD state;
    VecD op;
    VecD mapped_params;
    VecD gradient;
    std::vector<TermIndex> cos_indices; ///< recorded coefficient indices, layer slices back to back
    VecD cos_values;                    ///< the pre-layer coefficient at each of them
    VecZ cos_offset;                    ///< per layer: start of its slice, plus a tail entry
    std::vector<uint8_t> cos_wanted;    ///< per layer: which records the parameters earn it
};

auto eval_scratch() -> EvalScratch & {
    static thread_local EvalScratch scratch;
    return scratch;
}

// Flag the layers whose record can change an answer, and return whether any needs its cosine set.
// Layer j records what layer j-1 rotates, so it earns its keep only once the layers already reversed can
// amplify an error there by a full significand -- a bound read off the parameters, before any coefficient
// is touched.
auto layout_cos_records(size_t layers, EvalScratch &scratch) -> bool {
    scratch.cos_wanted.assign(layers, 0);
    scratch.cos_offset.assign(layers + 1, 0);
    scratch.cos_indices.clear();
    scratch.cos_values.clear();
    bool any_cosine_set = false;
    double spread = 0.0;
    for (size_t j = layers; j-- > 0;) {
        const double cos_val = std::cos(2 * scratch.mapped_params[j]);
        const double own = -std::log2(std::abs(cos_val));
        if (j > 0 && spread + own >= detail::kRecordSpreadBits) {
            scratch.cos_wanted[j] |= kRecordRotationsBelow;
        }
        if (std::abs(cos_val) < detail::kVanishingCos) {
            scratch.cos_wanted[j] |= kRecordCosineSet;
            any_cosine_set = true;
        }
        spread += own;
    }
    return any_cosine_set;
}

// Open `layer_idx`'s slice and fill it from the not-yet-stepped `op`. Called in layer order, so each
// slice starts where the previous one ended.
auto record_pre_layer(EvalScratch &scratch,
                      const MPGraphView &graph,
                      const detail::LayerCosIndices &cos_inds,
                      size_t layer_idx) -> void {
    scratch.cos_offset[layer_idx] = scratch.cos_values.size();
    const uint8_t wanted = scratch.cos_wanted[layer_idx];
    if (wanted == 0) {
        return;
    }
    const size_t begin = scratch.cos_indices.size();
    if ((wanted & kRecordRotationsBelow) != 0U) {
        const auto below = graph.get_layer_traversal(layer_idx - 1);
        const auto mark = [&scratch](size_t, size_t i, auto) {
            scratch.cos_indices.push_back(static_cast<TermIndex>(i));
        };
        for (size_t r = 0; r < below.cross_rank_rank_count(); ++r) {
            below.for_each_cross_rank_sin_recv_range(r, 0, below.cross_rank_sin_recv_size(r), mark);
        }
    }
    if ((wanted & kRecordCosineSet) != 0U) {
        cos_inds(layer_idx, scratch.cos_indices);
    }
    const size_t end = scratch.cos_indices.size();
    scratch.cos_values.resize(end);
    for (size_t k = begin; k < end; ++k) {
        scratch.cos_values[k] = scratch.op[scratch.cos_indices[k]];
    }
}

// What the reverse pass reads back for `layer_idx`.
auto layer_record_in(const EvalScratch &scratch, size_t layer_idx) -> detail::CosRecordView {
    const size_t begin = scratch.cos_offset[layer_idx];
    const size_t count = scratch.cos_offset[layer_idx + 1] - begin;
    if (count == 0) {
        return {};
    }
    return {.indices = scratch.cos_indices.data() + begin, .values = scratch.cos_values.data() + begin, .count = count};
}

// Graph is traversed in simulation order but parameter_mapping is stored in optimizer order; write the
// mapped coefficients forward or reversed accordingly.
auto fill_mapped_params(VecD &result,
                        const VecD &parameters,
                        const VecZ &parameter_mapping,
                        const VecD &gen_coeffs,
                        double phase,
                        bool reverse) -> void {
    const size_t count = parameter_mapping.size();
    result.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t dst = reverse ? (count - 1 - i) : i;
        result[dst] = phase * parameters[parameter_mapping[i]] * gen_coeffs[i];
    }
}

auto prepare_evolved_operator(const EvalRequest &request, mpi::Comm comm, const detail::CosCallbacks &cos, bool record)
    -> void {
    // Checked once here rather than at each caller: evolve_operator invokes cos_scale per layer, so an
    // empty callback would otherwise surface as a std::bad_function_call naming nothing.
    if (!cos.scale) {
        throw MissingLayerCallback("Evaluating at non-empty parameters requires a cos_scale (forward) callback.");
    }
    auto &scratch = eval_scratch();
    fill_mapped_params(scratch.mapped_params, request.params, request.parameter_mapping, request.gen_coeffs, 1.0, true);
    scratch.op = request.op;
    if (!record) {
        scratch.op = evolve_operator(std::move(scratch.op), request.graph, scratch.mapped_params, comm, cos.scale);
        return;
    }

    // Step layer by layer so each record is taken while op still holds that layer's pre-layer values.
    const size_t layers = request.graph.layers();
    if (layout_cos_records(layers, scratch) && !cos.indices) {
        throw MissingLayerCallback("A layer whose cosine vanishes requires a cos_indices (reverse) callback.");
    }
    for (size_t i = 0; i < layers; ++i) {
        record_pre_layer(scratch, request.graph, cos.indices, i);
        evolve_step(scratch.op, request.graph, scratch.mapped_params[i], i, comm, cos.scale);
    }
    scratch.cos_offset[layers] = scratch.cos_values.size();
}

} // namespace

auto inner_product(const VecD &v, const VecD &w) -> double {
    const auto *v_data = v.data();
    const auto *w_data = w.data();
    double result = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        result += v_data[i] * w_data[i];
    }
    return result;
}

auto indices_above(const VecD &v, double threshold) -> VecZ {
    VecZ inds;
    inds.reserve(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        if (std::abs(v[i]) > threshold) {
            inds.push_back(i);
        }
    }
    return inds;
}

auto EvalState::sparse(size_t length, std::span<const TermIndex> rows, std::span<const double> values) -> EvalState {
    if (rows.size() != values.size()) {
        throw EvalStateArgumentError("EvalState::sparse: rows and values must have the same length.");
    }

    EvalState state;
    state.length_ = length;
    state.is_dense_ = false;
    state.rows_.reserve(rows.size());
    for (const auto row : rows) {
        const auto widened = static_cast<size_t>(row);
        if (widened >= length) {
            throw EvalStateArgumentError("EvalState::sparse: row index is out of range for the given length.");
        }
        state.rows_.push_back(widened);
    }
    state.values_.assign(values.begin(), values.end());
    return state;
}

auto EvalState::dense(VecD values) -> EvalState {
    EvalState state;
    state.length_ = values.size();
    state.is_dense_ = true;
    state.values_ = std::move(values);
    return state;
}

auto EvalState::dot(const VecD &op) const -> double {
    if (op.size() < length_) {
        throw EvalStateArgumentError("EvalState::dot: the operator is shorter than the state.");
    }
    if (is_dense_) {
        return inner_product(values_, op);
    }

    double result = 0.0;
    for (size_t k = 0; k < rows_.size(); ++k) {
        result += values_[k] * op[rows_[k]];
    }
    return result;
}

auto EvalState::scatter_into(VecD &out) const -> void {
    if (is_dense_) {
        out.assign(values_.begin(), values_.end());
        return;
    }

    // assign, not resize: `out` is scratch that arrives holding a previous, possibly longer state, and
    // every entry this state does not name must read as an exact zero.
    out.assign(length_, 0.0);
    for (size_t k = 0; k < rows_.size(); ++k) {
        out[rows_[k]] = values_[k];
    }
}

auto EvalState::indices_above(double threshold) const -> VecZ {
    if (is_dense_) {
        return monoprop::indices_above(values_, threshold);
    }
    // |0.0| exceeds a negative threshold too, so the dense scan would keep every index; match it rather
    // than silently paring against a different keep-set.
    if (threshold < 0.0) {
        VecZ all(length_);
        std::iota(all.begin(), all.end(), 0uz);
        return all;
    }

    VecZ inds;
    inds.reserve(rows_.size());
    for (size_t k = 0; k < rows_.size(); ++k) {
        if (std::abs(values_[k]) > threshold) {
            inds.push_back(rows_[k]);
        }
    }
    return inds;
}

auto map_params(const VecD &parameters,
                const VecZ &parameter_mapping,
                const VecD &gen_coeffs,
                double phase,
                bool reverse) -> VecD {
    VecD result;
    fill_mapped_params(result, parameters, parameter_mapping, gen_coeffs, phase, reverse);
    return result;
}

auto ev(const EvalRequest &request, mpi::Comm comm, const detail::CosCallbacks &cos) -> double {
    // The allreduce is unconditional -- ShmComm's is barrier-synced, so short-circuiting an empty local
    // sum past it would deadlock every peer.
    if (request.params.empty()) {
        return request.e_core + mpi::allreduce_sum(request.state.dot(request.op), comm);
    }

    prepare_evolved_operator(request, comm, cos, /*record=*/false);
    return request.e_core + mpi::allreduce_sum(request.state.dot(eval_scratch().op), comm);
}

auto ev_and_grad(const EvalRequest &request, mpi::Comm comm, const detail::CosCallbacks &cos)
    -> std::pair<double, VecD> {
    if (request.params.empty()) {
        return {request.e_core + mpi::allreduce_sum(request.state.dot(request.op), comm), VecD(0)};
    }

    // cos.scale is checked in prepare_evolved_operator, shared by both paths; cos.accumulate is this path's own.
    if (!cos.accumulate) {
        throw MissingLayerCallback("ev_and_grad requires a cos_acc (reverse) callback.");
    }

    // The reverse pass back-evolves the state in place, which destroys sparsity at the first layer, so
    // this path needs it dense. Densifying into the shared scratch keeps energy-only runs from ever
    // building one.
    auto &scratch = eval_scratch();
    request.state.scatter_into(scratch.state);
    prepare_evolved_operator(request, comm, cos, /*record=*/true);

    auto &state_ = scratch.state;
    auto &op_ = scratch.op;
    const auto expectation_value = mpi::allreduce_sum(inner_product(state_, op_), comm);

    const auto &parameter_mapping = request.parameter_mapping;
    scratch.gradient.assign(request.params.size(), 0.0);
    for (size_t i = 0; i < parameter_mapping.size(); ++i) {
        const auto idx = parameter_mapping.size() - 1 - i;
        const auto param_ind = parameter_mapping[i];
        scratch.gradient[param_ind] +=
            state_operator_derivative_local(state_,
                                            op_,
                                            request.graph,
                                            idx,
                                            {.gen_coeff = request.gen_coeffs[i], .param = request.params[param_ind]},
                                            comm,
                                            cos.accumulate,
                                            layer_record_in(scratch, idx));
    }

    mpi::allreduce_sum_inplace(scratch.gradient, comm);
    return {request.e_core + expectation_value, scratch.gradient};
}

} // namespace monoprop
