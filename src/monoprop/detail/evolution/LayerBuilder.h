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

// LayerBuilder.h — the paper's BuildDistributedLayer algorithm (arXiv:2503.18939, Algorithm 2)
//
// build_layer<NumModes>() implements Algorithm 2 in a single pass: one fused
// FindAnticommuting+cutoff scan feeds two MPI exchange passes and emits a graph layer directly.
//
// The layer's cosine set records ALL locally-anticommuting indices (endpoints included, not just the
// cutoff survivors). Contraction reads it as a PRE-rotation snapshot of the operator, so replay is
// independent of iteration order and rank count.
//
// FindAnticommuting is a single parity-agnostic pass: the inverted index XOR-column fold + pivot-bit
// leader/follower split. WHY the pivot bit is the whole trick: a term M and its rotation partner
// M⊕G differ in every column of G, including the pivot (G's lowest set column), so exactly one of
// the pair carries the pivot bit. The bit therefore assigns the two partners opposite roles —
// leader (pivot clear) vs follower (pivot set) — and visiting leaders then the still-unmatched
// followers touches each anticommuting pair exactly once, with no comparison sort and no dedup.
// Even generators use the plain fold; odd generators add the per-row parity(|M|) correction (g_odd)
// so the same kernel is correct for both parities.
//
// Emits a graph layer directly: returns std::shared_ptr<LayerCore> assembled in-pass from a
// uniform per-rank PartnerAcc, ready to append to the surrogate graph.
//
// Overview (paper Algorithm 2):
// 1) FindAnticommuting → partition the anticommuting terms into leaders L_r and followers F_r.
// 2) Leader pass: apply cutoffs → route surviving queries to owner of M' = M_i⊕G.
//    * Local leaders (owner == my_rank): resolve inline.
//    * Remote leaders: exchange queries via MPI (round 1a), receive responses (round 1b).
// 3) Follower pass: iterate F_r \ matched → same exchange (rounds 2a, 2b).
// 4) Insert-on-miss in the resolver: absent partners targeting REMOTE ranks are inserted by the
//    owner during resolve_incoming_queries and their real index is returned in the same response
//    round. Absent partners targeting THIS rank (self-rank queries, resolved inline) are deferred
//    and inserted after both passes, inside build_layer itself — never over the wire.

// Outcomes: cutoff applied → cosine only; otherwise → sine between ranks. A matched follower
// (found by its leader) is skipped — the leader already accounts for it.
//
// Umbrella header: the implementation lives in the sibling layer_build/ headers, included below in
// dependency order (Parallel → Common → Scan → Resolve → Engine). Include this for the full surface.

#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/evolution/layer_build/Engine.h"
