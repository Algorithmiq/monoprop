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

#include "monoprop/detail/sharded/Evaluation.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <format>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "monoprop/detail/evolution/LayerReplay.h"
#include "monoprop/detail/mpi/OperationFailure.h"

namespace monoprop::detail::sharded {

namespace {

// Within-shard kernels run serially: the team's parallelism is across owners.
constexpr parallel::Options kSerial{};

auto notify(const EvaluationObserver *observer, EvaluationWork work, size_t step, size_t shard) -> void {
    if (observer != nullptr) {
        observer->visit(work, step, shard);
    }
}

// The block partners read from a board: `scale` values per endpoint of this owner's slot for them.
template <size_t Scale>
auto board_reader(const EndpointBoard &board, const LayerTraversal &layer, size_t flat_owner) {
    return [&board, &layer, flat_owner](size_t rank) -> const double * {
        return board.block(rank, flat_owner, Scale * layer.cross_rank_sin_send_size(rank));
    };
}

// One owner's working frame. Thread-local to the executing worker thread and retained across calls, as the legacy
// per-thread eval scratch is; every call rebinds it in the owner's first phase and resets what it reads.
struct OwnerFrame {
    VecD op;                                // the working operator, evolved in place
    VecD state;                             // gradient only: the dense working state
    ForwardScratch forward;                 // forward self sources
    replay::DerivativeSnapshotScratch snap; // reverse pre-cosine snapshots
    replay::CosRecords records;             // gradient only: selective pre-layer records
    PublishedPair published;                // read by partners
    double accumulated = 0.0;               // reverse: the raw accumulation (before ×sec) between begin and finish
    bool self_pre = false;                  // reverse: whether the self pass reads the snapshotted entry op
};

auto owner_frame() -> OwnerFrame & {
    static thread_local OwnerFrame frame;
    return frame;
}

// Pure argument checks, before the team.
auto same_bits(double a, double b) -> bool {
    return std::bit_cast<uint64_t>(a) == std::bit_cast<uint64_t>(b);
}

auto check_team(const char *what, parallel::Options options, size_t entries, size_t callbacks) -> void {
    if (options.threads < 1 || entries != static_cast<size_t>(options.threads)
        || callbacks != static_cast<size_t>(options.threads)) {
        throw std::invalid_argument(std::format("sharded::{}: expected {} requests and callback sets, got {} and {}",
                                                what,
                                                options.threads,
                                                entries,
                                                callbacks));
    }
}

// What every shard of one evaluation shares, decided on the caller from the replicated inputs.
struct EvaluationPlan {
    bool empty = false;          // no parameters: contract the un-evolved operator, no replay
    size_t layers = 0;           // forward steps L
    size_t steps = 0;            // L, or 2L with the reverse pass
    bool opaque = false;         // some callback set is opaque: callback work runs on the primary
    VecD mapped;                 // replay angle per forward step (layer order)
    std::vector<uint8_t> wanted; // gradient: per layer record flags
};

auto plan_evaluation(std::span<const EvalRequest> requests,
                     std::span<const CosCallbacks> callbacks,
                     parallel::Options options,
                     bool gradient) -> EvaluationPlan {
    check_team("evaluate", options, requests.size(), callbacks.size());
    const EvalRequest &first = requests.front();
    EvaluationPlan plan;
    plan.empty = first.params.empty();
    for (const EvalRequest &request : requests) {
        // The team runs one phase sequence, so everything that shapes it must agree on every shard.
        if (!same_bits(request.e_core, first.e_core) || request.params.size() != first.params.size()
            || !std::ranges::equal(request.params, first.params, same_bits)
            || request.parameter_mapping != first.parameter_mapping
            || !std::ranges::equal(request.gen_coeffs, first.gen_coeffs, same_bits)) {
            throw std::invalid_argument(
                "sharded::evaluate: the replicated identity, parameters, mapping and generator coefficients must "
                "agree on every shard");
        }
        if (request.state.length() > request.op.size()) {
            throw EvalStateArgumentError("EvalState::dot: the operator is shorter than the state.");
        }
        if (!plan.empty && request.graph.layers() != first.parameter_mapping.size()) {
            throw std::invalid_argument(std::format("sharded::evaluate: a graph has {} layers for {} mapped parameters",
                                                    request.graph.layers(),
                                                    first.parameter_mapping.size()));
        }
    }
    if (plan.empty) {
        return plan;
    }
    if (first.gen_coeffs.size() != first.parameter_mapping.size()) {
        throw std::invalid_argument("sharded::evaluate: parameter_mapping and gen_coeffs differ in length");
    }
    for (const size_t index : first.parameter_mapping) {
        if (index >= first.params.size()) {
            throw std::invalid_argument(
                std::format("sharded::evaluate: parameter index {} is out of range for {} parameters",
                            index,
                            first.params.size()));
        }
    }
    // The legacy evaluator's checks, in its order: accumulate (gradient), then scale, then indices.
    for (const CosCallbacks &cos : callbacks) {
        if (gradient && !cos.accumulate) {
            throw MissingLayerCallback("ev_and_grad requires a cos_acc (reverse) callback.");
        }
    }
    for (const CosCallbacks &cos : callbacks) {
        if (!cos.scale) {
            throw MissingLayerCallback("Evaluating at non-empty parameters requires a cos_scale (forward) callback.");
        }
    }
    plan.layers = first.parameter_mapping.size();
    plan.steps = gradient ? 2 * plan.layers : plan.layers;
    plan.mapped = map_params(first.params, first.parameter_mapping, first.gen_coeffs, 1.0, true);
    if (gradient && replay::plan_cos_records(plan.mapped, plan.wanted)) {
        for (const CosCallbacks &cos : callbacks) {
            if (!cos.indices) {
                throw MissingLayerCallback("A layer whose cosine vanishes requires a cos_indices (reverse) callback.");
            }
        }
    }
    plan.opaque = std::ranges::any_of(callbacks, [](const CosCallbacks &cos) { return !cos.owner_parallel; });
    return plan;
}

// The evaluation's shared, caller-owned state for one team.
struct EvaluationRun {
    std::span<const EvalRequest> requests;
    std::span<const CosCallbacks> callbacks;
    const EvaluationPlan &plan;
    bool gradient;
    const EvaluationObserver *observer;
    size_t first_local = 0;
    std::vector<OwnerFrame *> frames;         // entry t written by owner t in its frame phase
    std::vector<const PublishedPair *> pairs; // entry t written by owner t in its frame phase
    std::vector<double> *contributions;       // entry t written by owner t
    std::vector<VecD> *gradients;             // entry t written by owner t

    [[nodiscard]] auto board(size_t step) const -> EndpointBoard {
        return {.owners = pairs, .buffer = step % 2, .first_local = first_local};
    }

    // Reverse step j = step - L replays layer L - 1 - j at parameter mapping[j].
    [[nodiscard]] auto reverse_layer(size_t step) const -> size_t { return (2 * plan.layers) - 1 - step; }
    [[nodiscard]] auto reverse_trig(size_t step) const -> replay::TrigValues {
        const EvalRequest &request = requests.front();
        const size_t j = step - plan.layers;
        const size_t param_ind = request.parameter_mapping[j];
        return replay::TrigValues(request.params[param_ind], request.gen_coeffs[j]);
    }

    auto frame(size_t t) -> void {
        notify(observer, EvaluationWork::frame, kNoStep, t);
        OwnerFrame &f = owner_frame();
        const EvalRequest &request = requests[t];
        if (!plan.empty) {
            f.op = request.op;
            if (gradient) {
                request.state.scatter_into(f.state);
                replay::reset_records(f.records, plan.layers);
                (*gradients)[t].assign(request.params.size(), 0.0);
            }
        }
        f.accumulated = 0.0;
        f.self_pre = false;
        frames[t] = &f;
        pairs[t] = &f.published;
    }

    // The callback part of a begin: on the owner, or on the primary for an opaque set.
    auto callback_part(size_t s, size_t step) -> void {
        OwnerFrame &f = *frames[s];
        const EvalRequest &request = requests[s];
        const CosCallbacks &cos = callbacks[s];
        if (step < plan.layers) {
            if (gradient) {
                notify(observer, EvaluationWork::record, step, s);
                replay::record_pre_layer(f.records, plan.wanted, request.graph, cos.indices, step, f.op);
                if (step + 1 == plan.layers) {
                    replay::close_records(f.records, plan.layers);
                }
            }
            notify(observer, EvaluationWork::cosine, step, s);
            forward_scale(f.op, step, plan.mapped[step], cos.scale);
            return;
        }
        notify(observer, EvaluationWork::reverse_accumulate, step, s);
        const size_t idx = reverse_layer(step);
        f.accumulated = replay::accumulate_layer_raw(f.state,
                                                     f.op,
                                                     idx,
                                                     reverse_trig(step),
                                                     replay::layer_record_in(f.records, idx),
                                                     cos.accumulate);
    }

    auto begin(size_t t, size_t step) -> void {
        OwnerFrame &f = *frames[t];
        const EvalRequest &request = requests[t];
        const size_t flat = first_local + t;
        PublishedEndpoints &out = f.published[step % 2];
        if (step < plan.layers) {
            notify(observer, EvaluationWork::publish, step, t);
            forward_publish(f.op, request.graph.get_layer_traversal(step), flat, f.forward, out);
        }
        else {
            notify(observer, EvaluationWork::reverse_publish, step, t);
            const size_t idx = reverse_layer(step);
            const auto layer = request.graph.get_layer_traversal(idx);
            const auto record = replay::layer_record_in(f.records, idx);
            replay::snapshot_remote_endpoints(f.state, f.op, layer, flat, f.snap);
            // The self slot reads its entry op off the post-cos slots, which a record overwrites first.
            f.self_pre = record.count > 0 && flat < layer.cross_rank_rank_count();
            if (f.self_pre) {
                replay::snapshot_self_recv_op(f.op, layer, f.snap);
            }
            // Scale 2: each rotation endpoint carries both the state and the op value.
            derive_exchange_layout(layer.cross_rank(), flat, 2, out.layout, "Layer derivative exchange");
            out.values.resize(out.layout.total_count);
            replay::pack_derivative_payload(f.snap, layer, flat, [&out](size_t rank) {
                return out.values.data() + out.layout.displs[rank];
            });
        }
        if (!plan.opaque) {
            callback_part(t, step);
        }
    }

    auto finish(size_t t, size_t step) -> void {
        OwnerFrame &f = *frames[t];
        const EvalRequest &request = requests[t];
        const size_t flat = first_local + t;
        const EndpointBoard published = board(step);
        if (step < plan.layers) {
            notify(observer, EvaluationWork::finish, step, t);
            forward_finish(f.op,
                           request.graph.get_layer_traversal(step),
                           plan.mapped[step],
                           flat,
                           f.forward,
                           published);
            return;
        }
        notify(observer, EvaluationWork::reverse_finish, step, t);
        const size_t idx = reverse_layer(step);
        const auto layer = request.graph.get_layer_traversal(idx);
        const auto trig = reverse_trig(step);
        replay::EndpointContrib ep;
        if (flat < layer.cross_rank_rank_count()) {
            ep = replay::apply_self_slot_derivative_paired(f.state,
                                                           f.op,
                                                           layer,
                                                           trig,
                                                           f.self_pre ? f.snap.self_recv_op.data() : nullptr);
        }
        const auto remote = replay::apply_derivative_payload(f.state,
                                                             f.op,
                                                             layer,
                                                             f.snap,
                                                             trig,
                                                             flat,
                                                             board_reader<2>(published, layer, flat));
        ep = replay::combine_endpoint_contrib(ep, remote);
        const size_t param_ind = request.parameter_mapping[step - plan.layers];
        (*gradients)[t][param_ind] += replay::layer_derivative_from_raw(trig, f.accumulated, ep);
    }

    auto contribution(size_t t) -> void {
        notify(observer, EvaluationWork::contribution, kNoStep, t);
        const EvalRequest &request = requests[t];
        if (plan.empty) {
            (*contributions)[t] = request.state.dot(request.op);
            return;
        }
        const OwnerFrame &f = *frames[t];
        (*contributions)[t] = gradient ? inner_product(f.state, f.op) : request.state.dot(f.op);
    }

    // The whole sequence of one worker: frame, then phase p = finish(p - 1), contribution at p = L, begin(p).
    auto run(size_t t, TeamFailure &failure) noexcept -> void {
        if (!phase(failure, t, [&] { frame(t); })) {
            return;
        }
        for (size_t p = 0; p <= plan.steps; ++p) {
            const bool ok = phase(failure, t, [&] {
                if (p > 0) {
                    finish(t, p - 1);
                }
                if (p == plan.layers) {
                    contribution(t);
                }
                if (p < plan.steps) {
                    begin(t, p);
                }
            });
            if (!ok) {
                return;
            }
            // Opaque callbacks: the primary runs every shard's callback part while the other owners wait.
            if (plan.opaque && p < plan.steps
                && !phase(
                    failure,
                    t,
                    [&] {
                        if (t == 0) {
                            for (size_t s = 0; s < frames.size(); ++s) {
                                callback_part(s, p);
                            }
                        }
                    })) {
                return;
            }
        }
    }
};

// The forward replay of replay_forward_in_team() and replay_shards(): phase p = finish(p - 1), begin(p).
auto forward_steps(TeamFailure &failure,
                   size_t t,
                   ForwardReplayJob &job,
                   std::span<const PublishedPair *const> pairs,
                   const EvaluationObserver *observer) noexcept -> bool {
    const size_t steps = job.params.size();
    const size_t flat = job.first_local + t;
    const auto cosine = [&](size_t s, size_t step) {
        ForwardReplayOwner &owner = job.owners[s];
        notify(observer, EvaluationWork::cosine, step, s);
        forward_scale(*owner.coeffs, step, job.params[step], owner.callbacks->scale);
    };
    for (size_t p = 0; p <= steps; ++p) {
        const bool ok = phase(failure, t, [&] {
            ForwardReplayOwner &own = job.owners[t];
            if (p > 0) {
                notify(observer, EvaluationWork::finish, p - 1, t);
                const EndpointBoard board{.owners = pairs, .buffer = (p - 1) % 2, .first_local = job.first_local};
                forward_finish(*own.coeffs,
                               own.graph->get_layer_traversal(p - 1),
                               job.params[p - 1],
                               flat,
                               own.scratch,
                               board);
            }
            if (p < steps) {
                notify(observer, EvaluationWork::publish, p, t);
                forward_publish(*own.coeffs,
                                own.graph->get_layer_traversal(p),
                                flat,
                                own.scratch,
                                own.published[p % 2]);
                if (job.owner_parallel) {
                    cosine(t, p);
                }
            }
        });
        if (!ok) {
            return false;
        }
        if (!job.owner_parallel && p < steps
            && !phase(
                failure,
                t,
                [&] {
                    if (t == 0) {
                        for (size_t s = 0; s < job.owners.size(); ++s) {
                            cosine(s, p);
                        }
                    }
                })) {
            return false;
        }
    }
    return true;
}

auto check_replay_job(const ForwardReplayJob &job) -> void {
    for (const ForwardReplayOwner &owner : job.owners) {
        if (owner.coeffs == nullptr || owner.graph == nullptr || owner.callbacks == nullptr) {
            throw std::logic_error("sharded replay: an owner entry was not filled");
        }
        if (owner.graph->layers() != job.params.size()) {
            throw std::invalid_argument(std::format("sharded replay: a window has {} layers for {} angles",
                                                    owner.graph->layers(),
                                                    job.params.size()));
        }
        if (!job.params.empty() && !owner.callbacks->scale) {
            throw MissingLayerCallback("Evaluating at non-empty parameters requires a cos_scale (forward) callback.");
        }
    }
}

} // namespace

auto forward_publish(const VecD &coeffs,
                     const LayerTraversal &layer,
                     size_t flat_owner,
                     ForwardScratch &scratch,
                     PublishedEndpoints &out) -> void {
    replay::snapshot_self_sources(coeffs, layer, scratch.self_sources);
    derive_exchange_layout(layer.cross_rank(), flat_owner, 1, out.layout, "Layer exchange");
    out.values.resize(out.layout.total_count);
    replay::pack_evolution_payload(coeffs, layer, flat_owner, [&out](size_t rank) {
        return out.values.data() + out.layout.displs[rank];
    });
}

auto forward_scale(VecD &coeffs, size_t layer_idx, double param, const LayerCosScale &scale) -> void {
    const double cos_val = std::cos(2 * param);
    scale(layer_idx, coeffs.data(), cos_val);
}

auto forward_scale_mask(VecD &coeffs, const CosMask &cos, double param) -> void {
    const double cos_val = std::cos(2 * param);
    scale_cos_mask(coeffs.data(), cos, cos_val, kSerial);
}

auto forward_finish(VecD &coeffs,
                    const LayerTraversal &layer,
                    double param,
                    size_t flat_owner,
                    const ForwardScratch &scratch,
                    const EndpointBoard &board) -> void {
    const double sin_val = std::sin(2 * param);
    replay::apply_evolution_payload(coeffs, layer, sin_val, flat_owner, board_reader<1>(board, layer, flat_owner));
    replay::apply_self_sources(coeffs, layer, sin_val, scratch.self_sources);
}

auto replay_forward_in_team(TeamFailure &failure,
                            size_t shard,
                            ForwardReplayJob &job,
                            const EvaluationObserver *observer) noexcept -> bool {
    // The pair pointers are derived, not stored: every worker builds the same span over the caller's entries.
    std::vector<const PublishedPair *> pairs;
    bool ok = phase(failure, shard, [&] {
        if (shard == 0) {
            check_replay_job(job);
        }
        pairs.reserve(job.owners.size());
        for (const ForwardReplayOwner &owner : job.owners) {
            pairs.push_back(&owner.published);
        }
    });
    return ok && forward_steps(failure, shard, job, pairs, observer);
}

auto evaluate_shards(std::span<const EvalRequest> requests,
                     std::span<const CosCallbacks> callbacks,
                     parallel::Options options,
                     bool gradient,
                     const EvaluationObserver *observer) -> EvaluationOutcome {
    const EvaluationPlan plan = plan_evaluation(requests, callbacks, options, gradient);
    const size_t threads = requests.size();
    EvaluationOutcome outcome{.error = {}, .contributions = std::vector<double>(threads, 0.0), .gradients = {}};
    if (gradient) {
        outcome.gradients.resize(threads);
    }
    EvaluationRun run{.requests = requests,
                      .callbacks = callbacks,
                      .plan = plan,
                      .gradient = gradient,
                      .observer = observer,
                      .first_local = 0,
                      .frames = std::vector<OwnerFrame *>(threads, nullptr),
                      .pairs = std::vector<const PublishedPair *>(threads, nullptr),
                      .contributions = &outcome.contributions,
                      .gradients = &outcome.gradients};
    // Empty parameters leave every gradient empty, as the legacy evaluator returns.
    outcome.error = run_team(options, [&run](size_t t, TeamFailure &failure) noexcept { run.run(t, failure); });
    return outcome;
}

auto combine_contributions(std::span<const double> contributions) -> double {
    auto total = 0.0;
    for (const double contribution : contributions) {
        total += contribution;
    }
    return total;
}

auto combine_gradients(std::span<const VecD> gradients) -> VecD {
    if (gradients.empty()) {
        return {};
    }
    const size_t length = gradients.front().size();
    for (const VecD &g : gradients) {
        if (g.size() != length) {
            throw std::invalid_argument("sharded::combine_gradients: the shards' gradients differ in length");
        }
    }
    VecD total(length, 0.0);
    for (size_t k = 0; k < length; ++k) {
        auto acc = 0.0;
        for (const VecD &g : gradients) {
            acc += g[k];
        }
        total[k] = acc;
    }
    return total;
}

namespace {

auto check_single_rank(const char *what, const mpi::Comm &comm) -> void {
    if (comm.kind != mpi::Comm::Kind::Mpi || mpi::size(comm) != 1) {
        throw std::invalid_argument(std::format("sharded::{}: only a one-rank ordinary communicator is supported "
                                                "until the physical exchange exists",
                                                what));
    }
}

} // namespace

auto ev_sharded(std::span<const EvalRequest> requests,
                std::span<const CosCallbacks> callbacks,
                parallel::Options options,
                mpi::Comm comm,
                const EvaluationObserver *observer) -> double {
    check_single_rank("ev_sharded", comm);
    auto outcome = evaluate_shards(requests, callbacks, options, /*gradient=*/false, observer);
    if (outcome.error) {
        mpi::operation_failed(comm, outcome.error);
    }
    const double local = combine_contributions(outcome.contributions);
    return requests.front().e_core + mpi::allreduce_sum(local, comm);
}

auto ev_and_grad_sharded(std::span<const EvalRequest> requests,
                         std::span<const CosCallbacks> callbacks,
                         parallel::Options options,
                         mpi::Comm comm,
                         const EvaluationObserver *observer) -> std::pair<double, VecD> {
    check_single_rank("ev_and_grad_sharded", comm);
    auto outcome = evaluate_shards(requests, callbacks, options, /*gradient=*/true, observer);
    if (outcome.error) {
        mpi::operation_failed(comm, outcome.error);
    }
    const double local = combine_contributions(outcome.contributions);
    const double expectation_value = mpi::allreduce_sum(local, comm);
    VecD gradient = combine_gradients(outcome.gradients);
    if (!requests.front().params.empty()) {
        mpi::allreduce_sum_inplace(gradient, comm);
    }
    return {requests.front().e_core + expectation_value, std::move(gradient)};
}

auto replay_shards(std::span<const ReplayRequest> requests,
                   std::span<const double> mapped_params,
                   std::span<const CosCallbacks> callbacks,
                   parallel::Options options,
                   const EvaluationObserver *observer) -> ReplayOutcome {
    check_team("replay_shards", options, requests.size(), callbacks.size());
    const size_t threads = requests.size();
    ForwardReplayJob job{
        .params = mapped_params,
        .first_local = 0,
        .owner_parallel = std::ranges::all_of(callbacks, [](const CosCallbacks &cos) { return cos.owner_parallel; }),
        .owners = std::vector<ForwardReplayOwner>(threads)};
    for (size_t t = 0; t < threads; ++t) {
        job.owners[t].graph = &requests[t].graph;
        job.owners[t].callbacks = &callbacks[t];
    }
    // Slot t is filled by owner t in its frame phase, so its storage is first touched there.
    ReplayOutcome outcome{.error = {}, .coeffs = std::vector<VecD>(threads)};
    for (size_t t = 0; t < threads; ++t) {
        job.owners[t].coeffs = &outcome.coeffs[t];
    }
    check_replay_job(job);
    outcome.error = run_team(options, [&](size_t t, TeamFailure &failure) noexcept {
        if (!phase(failure, t, [&] {
                notify(observer, EvaluationWork::frame, kNoStep, t);
                outcome.coeffs[t] = requests[t].coeffs; // allocated on the owner
            })) {
            return;
        }
        static_cast<void>(replay_forward_in_team(failure, t, job, observer));
    });
    return outcome;
}

} // namespace monoprop::detail::sharded
