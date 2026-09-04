# Engine history and operational notes (condensed from the coordinator's session memory)

## How the design got here (2026-09-01 → 09-04)

1. Phase 1 `1e7a7e98` hash-free kernel: resolve partners inside Anti(G) instead of a hash table.
   Bit-identical; on its own 9–11× slower for `propagate` with `lower_atol` (an AntiTable over all of Anti(G)).
2. Phase 2a `524ffc7b` exact one-round symmetric exchange + `db3f2798`/`4da2f2de` bucketed radix join with
   4-byte keys. Exact one-round sends ≈ 0.85·|Anti| records in the lower_atol regime: 20× slower at L1 and
   *more* peak RSS. A `monoprop_DROP_SILENT_RECORDS` knob was 1.9–5.9× slower and changed term counts
   (−5 % to +32 %) → disqualified and removed. Lesson: any protocol that sends records for silent rows loses.
3. `60254b20` the **1½-round exact protocol** (records only for emitting rows, tiny responses for hits on
   silent rows). Protocol cost gone; residual 2.1× at P=1 was the kernel (cache-bound, grows with terms per
   partition). Callgrind attribution drove everything after.
4. `bff293d6` join query-tag filter + inlined visit + fused cos sweep (`pre_cos`): 1.97× → 1.50–1.63×.
5. `ce500790` A5 reserves from the join's tally, 16-byte half-rotation records: −4.5 %.
6. `fd795841` S3: 32 filter bits per query, no bucket split when the smaller side fits 32 Ki slots:
   L 0.942 / S 0.915 — the join was 12 % of instructions, pass B 8 %, the record pipeline the rest.
7. Storage levers (`perf/storage-levers`, 11 commits): pooled chunked stores, size-adaptive chunks, ledger
   completeness, gate-buffer HWM, restride rows to the observed width. Ledger 0.67× base. Time-neutral.
8. W (`perf/wire-zero-copy`): the peak−ledger gap at S>1 (16–19 B/term) is copies of records in flight;
   `mpi::pair_exchange` (from `perf/pair-exchange` cfd50db7) removes the staging chain. Integration first
   *raised* the 8×16 peak (1.092) through double-buffered pools; fixed down to 0.983, last fix (91806a63)
   expected to clear the 0.95 gate but unmeasured.

Rejected with measurements (do not re-propose without a new mechanism): co-located {coeff,key} record
(A3, slower), software prefetch of the sparse streams (A4, slower), loop-invariant hoists (1 %), AVX2
blocks for pass B (≤ 2.3 % ceiling), PGO (≤ 1.8 %, regression on base), 64-bit query filter (tie),
inverted-index posting codecs (Elias-Fano, run coding, ANS — measured on earlier branches), the 5-byte
hash slot of `perf/operator-memory-nr2`.

## Older branches worth porting rather than re-deriving

- `perf/majorana-sign-from-positions` `dc85a86f`: `rotation_sign_positions` and a merge returning
  (k, overlap, d). A lambda inside the merge cost 275 M instructions — keep per-position work out of the
  merge. Input to round 3's sign-from-fold idea.
- `perf/operator-memory` `0b0557c1` (+ `c63a6b03`): chunked row store, kChunkRows = 4096 (superseded by
  the storage branch's ChunkedArray, kept for its measurements).
- `fix/schrodinger-paired-basis` `97db1e2b`: enumerate the Schrödinger paired basis instead of listing it
  per world slot (unmerged fix for an S× transient).

## Fast C++ iteration loop (Deucalion; the shape applies anywhere scikit-build-core is used)

- `uv sync` builds the whole tree and leaves a CMake cache pointing at scikit-build's deleted temporary
  Python, so a later plain `ninja` cannot reconfigure. Fix once per reinstall:
  `uv pip install --python $VENV/bin/python nanobind "mpi4py>=4.1.0"`, then
  `cmake -S . -B build/editable/Release-<tag> -DPython_EXECUTABLE=$VENV/bin/python
  -Dnanobind_DIR=$($VENV/bin/python -c 'import nanobind;print(nanobind.cmake_dir())')
  -DCMAKE_PREFIX_PATH=$VENV/lib/python3.11/site-packages`; after that
  `ninja -C build/editable/Release-<tag> libmonoprop.so monoprop_unit_tests.x _core.abi3.so` is incremental.
- The venv's `site-packages/monoprop/_core.abi3.so` and `lib64/libmonoprop.so` are copies; ninja does not
  update them. Reinstall or copy both before running Python tests; identify arms by md5 of those files.
- Never edit a tree while it builds; keep an untouched detached worktree per reference arm.
- Profiling: RelWithDebInfo is the Release flag set plus `-g3` (codegen unchanged); copy only
  `libmonoprop.so` and `_core.abi3.so` into a `.venv-dbg`. Callgrind: `--LL=16MiB` (one zen2 CCX),
  `--compress-strings=no --compress-pos=no`. Asserts: separate build dir with `-DEXTRA_CXXFLAGS:STRING=-UNDEBUG`.
  `EXTRA_CXXFLAGS` reaches only the compile line; link-time flags need `CMAKE_SHARED_LINKER_FLAGS` and
  `CMAKE_MODULE_LINKER_FLAGS`.

## Measurement facts that were re-derived wrongly at least once

- |Q| (queries per gate) is 15–17 % of |Anti| at the S/L cells, not 1.7 %.
- Row stride is already cutoff-driven (`packed_inline_width_` → `max_slot_bound`); what was loose was
  the bound (observed widths P99 ≈ 0.8 of bound for Pauli, even parity for Hubbard).
- `FusedContract::halves` is reserved from the slot's own tally; the flat world total is the invariant.
- A/B ratios in every table are arm/base with the arm named first: **> 1 in the time column means slower.**
