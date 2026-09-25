# index-miniapp (Task 0)

A standalone MPI-off experiment for the rank-local OpenMP plan
(`docs/superpowers/plans/2026-09-18-rank-local-openmp.md`, Task 0). One process with one OpenMP team of T workers
compares three ownership/index designs over monoprop's current packed rows. The program reuses the header-only
`OperatorIndex` and `Monomial` code without changing it. It does not build, link or import monoprop.

The app checks exact keys and IDs, which establishes **index-result agreement only**. That is not retained-operator
equivalence, which also requires coefficients. It is not a propagation, replay, gradient, MPI or full-library benchmark,
and it does not measure the current library.

## Variants

| `--variant` | Stores | Batch pipeline |
| --- | --- | --- |
| `per-thread` | T disjoint `OperatorIndex` stores; owner = `monomial_hash % T` (R=1 routing) | route/count → ordered prefixes → stable scatter to owner buckets → per owner: `find_batch_positions`, missing IDs in bucket order, `grow_rows_geometric`/`set_positions`, `bulk_insert_hashed` into bucket slots → each source chunk pulls its own results |
| `shared-serial` | one `OperatorIndex` | parallel `find_batch_positions` over balanced contiguous spans → caller assigns missing IDs in query order, grows/fills rows, `bulk_insert_hashed` |
| `shared-concurrent` | one packed row backing (`OperatorIndex` rows only) + `boost::concurrent_flat_map` | parallel hash + per-query `cvisit` → caller assigns IDs and fills rows → join → parallel `emplace` of preassigned IDs |

`--lookup bulk` (only with `shared-concurrent`) probes through Boost's bulk `cvisit` in 16-query chunks. The visitor
receives only the matched element, never the query, and Boost documents no callback order. So the visitor records
`(hash, row)` pairs, and the results are matched back to queries after each call, outside Boost's locks. A query that is
alone with that hash in its chunk is the match; several such queries (duplicates or a 32-bit collision) are resolved by
exact comparison. Misses produce no callback and stay absent. The default is `--lookup per-query`.

The concurrent index stores `{row ID, cached 32-bit fold hash}` handles, never monomials or query views. Transparent
hash and equality compare query positions against the packed row exactly, using a dense comparison for spilled rows. The
backing `OperatorIndex` keeps its constant empty 16-slot table (`harness_bytes_est`); the app never populates or
reserves it. Boost changes table layout and probing as well as synchronization, so its T=1 result is a
different-container result, not a pure locking cost.

Shared IDs are canonical: misses get IDs in query order. Per-thread IDs are owner-local and follow per-owner query
order, so the numeric IDs differ between shard counts while the decoded keys and (owner, local ID) order agree.
Nothing reserves capacity; every table grows natively.

## Build and self-test

```bash
cmake -S tools/index-miniapp -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --parallel
OMP_NUM_THREADS=1 ctest --test-dir "$BUILD" --output-on-failure --no-tests=error
for T in 1 2 3 4; do OMP_NUM_THREADS="$T" OMP_DYNAMIC=FALSE "$BUILD/index-miniapp" --self-test; done
```

The build needs GCC 14+ or Clang 18+, Boost 1.85+ and OpenMP. It refuses a `monoprop_ENABLE_MPI` definition. The
self-test runs every fixture through all three variants at requested teams 1–4, with 128 and 1024 modes (both
position widths). It also runs per-thread at a fixed three shards. A serial key oracle that exists only in the
self-test checks the results.

The fixtures cover the plan's A/B/C hit masks and IDs, repeated queries, duplicate-miss rejection, empty
inputs and stores, 15/16/17 and 63/64/65 query tails, and an equal-fold-hash collision pair before and after table
growth. They also cover spilled length-16 rows across backing reallocations, actual probe/publication worker
participation and joined worker exceptions. The checks use `require`, not `assert`, so they stay active in Release.
Under `OMP_THREAD_LIMIT` the fixtures still pass, and participation is reported as `SKIP`.

A diagnostic build with `-DCMAKE_CXX_FLAGS=-DBOOST_UNORDERED_ENABLE_STATS` (Boost 1.86 or newer) adds Boost's own container
statistics for `shared-concurrent`: `bs_*` CSV columns covering probe lengths and comparisons for hits, misses and
insertions. The statistics cover the measured batches. Boost also counts its own traffic: every `emplace` performs an
absence lookup, and rehash transfers count as insertions. Only the hit counts match adapter probes one-to-one. The
concurrent counters serialize on a lock, so never time a stats build.

Sanitizers use command-line flags, for example
`-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"`. Qualify a TSan/OpenMP-runtime pair with a
known-racy and a known-safe program first. GCC TSan with an uninstrumented libgomp reports false positives. Sanitizer
timings are not performance evidence.

## Measurement run

```bash
OMP_NUM_THREADS="$T" OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close \
  "$BUILD/index-miniapp" --variant shared-serial --mode grow --rows 65536 \
  --batch 4096 --batches 16 --miss-percent 10 --key-length 6 --modes 128 --seed 0 [--profile]
```

- The team is `omp_get_max_threads()`, read once at startup. Every region requests exactly that many workers. The app
  never calls `omp_set_*`, sets affinity or discovers topology. Verify placement externally, for example with
  `OMP_DISPLAY_AFFINITY` or `/proc/<pid>/task/*/status`.
- Keys are an injective synthetic encoding: `2*modes` positions split into `key-length` bands, with one ordinal digit
  per band after a seed-derived cyclic offset. The app streams the initial rows through the variant's own grow path.
  Each timed batch contains `batch - floor(batch*miss/100)` distinct existing ordinals and fresh ordinals for the
  misses. In `lookup` mode the misses stay absent. In `grow` mode they are published. This structured synthetic
  family makes no claim about real hit ratios or ownership skew.
- The app prepares each batch outside the timer. The timer covers the complete batch: hashing, ownership,
  bucketing/scatter, probing, ID assignment, row growth and fill, publication and the join. `--profile` adds phase
  timers. Profile a separate matching process: normal runs leave the phase fields empty, and so do phases that a
  variant or mode does not have. For per-thread, `route`, `prefix`, `scatter`, `owner_elapsed` and `gather` are
  elapsed and additive. Its `probe`…`publish` columns are summed across owners that run concurrently, so they are
  **nonadditive**. `owner_max` sums the slowest owner of each batch.
- Validation runs outside the timer and keeps no shadow map. Each batch checks every declared hit and miss, the exact
  key of every returned row, the owner and the per-owner publication order. At the end, the app checks the exact row
  count and then looks up every expected key again, in order. It prints one CSV row with `validation=ok` only on
  success. A failure prints `FAILED` and no CSV row.
- `row_bytes_est`, `index_bytes_est` and `harness_bytes_est` are estimates. The concurrent index figure is a formula
  over `bucket_count()`. Take peak memory from an external whole-process RSS measurement.
- Boost APIs used: the `concurrent_flat_map(n, hash, eq)` constructor, `emplace`, heterogeneous single and bulk
  `cvisit`, `size` and `bucket_count`. The code rejects Boost older than 1.85.
