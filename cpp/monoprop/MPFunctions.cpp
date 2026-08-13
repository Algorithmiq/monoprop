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
#include <numeric>
#include <stdexcept>

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

// Per-thread scratch so repeated ev/grad calls stay allocation-light.
struct EvalScratch {
    VecD state;
    VecD op;
    VecD mapped_params;
    VecD gradient;
};

auto eval_scratch() -> EvalScratch & {
    static thread_local EvalScratch scratch;
    return scratch;
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

auto prepare_evolved_operator(const EvalRequest &request, mpi::Comm comm, const detail::LayerCosScale &cos_scale)
    -> void {
    // Checked once here rather than at each caller: evolve_operator invokes cos_scale per layer, so an
    // empty callback would otherwise surface as a std::bad_function_call naming nothing.
    if (!cos_scale) {
        throw MissingLayerCallback("Evaluating at non-empty parameters requires a cos_scale (forward) callback.");
    }
    auto &scratch = eval_scratch();
    fill_mapped_params(scratch.mapped_params, request.params, request.parameter_mapping, request.gen_coeffs, 1.0, true);
    scratch.op = request.op;
    scratch.op = evolve_operator(std::move(scratch.op), request.graph, scratch.mapped_params, comm, cos_scale);
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
        std::iota(all.begin(), all.end(), size_t{0});
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

    prepare_evolved_operator(request, comm, cos.scale);
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
    prepare_evolved_operator(request, comm, cos.scale);

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
                                            cos.accumulate);
    }

    mpi::allreduce_sum_inplace(scratch.gradient, comm);
    return {request.e_core + expectation_value, scratch.gradient};
}

} // namespace monoprop
