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
// build_layer<NumModes>() implements Algorithm 2 in a single pass: one fused FindAnticommuting+cutoff
// scan feeds two MPI exchange passes and emits a graph layer directly (a shared_ptr<LayerCore> assembled
// from a uniform per-rank PartnerAcc). The layer's cosine set records ALL locally-anticommuting indices
// (endpoints included), read at replay as a PRE-rotation snapshot, so replay is order- and rank-independent.
//
// WHY the pivot bit is the whole trick: M and its partner M⊕G differ in every column of G including the
// pivot (G's lowest set column), so exactly one of the pair carries it — leader (pivot clear) vs follower
// (pivot set). Visiting leaders then the still-unmatched followers touches each anticommuting pair once,
// no sort and no dedup. Even generators use the plain fold; odd add the per-row parity(|M|) correction (g_odd).
//
// Passes: (1) leader pass applies cutoffs and routes surviving queries to the owner of M'=M⊕G (local
// inline, remote via MPI); (2) follower pass repeats over F_r \ matched. Insert-on-miss: remote absent
// partners are inserted by the resolver in the same response round; self-rank absent partners are deferred
// and inserted after both passes inside build_layer — never over the wire. Cutoff applied → cosine only;
// otherwise → sine.
//
// Umbrella header: the implementation lives in the sibling layer_build/ headers, included below in
// dependency order (Common → Scan → Resolve → Engine). Include this for the full surface.

#include "monoprop/detail/evolution/layer_build/Common.h"
#include "monoprop/detail/evolution/layer_build/Engine.h"
#include "monoprop/detail/evolution/layer_build/Resolve.h"
#include "monoprop/detail/evolution/layer_build/Scan.h"
