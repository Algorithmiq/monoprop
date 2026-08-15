# The fixed-model campaign: Hubbard and kicked Ising, swept

2026-08-15, Deucalion x86. Harness `sbatch/models-ab.sh` driven by `sbatch/campaign.sh`,
collated by `tools/campaign_summary.py`. Arms are `origin/main` @ `6abd839` (0.8.1.dev37)
against `perf/invidx-memory`. Every cell: 6 interleaved reps, order flipped per (rep, cell),
`--bench-rounds=1`, `--cpu-bind=none`, 16 threads pinned per rank on both arms, term counts
identical across arms to the term.

One-factor-at-a-time around a per-model anchor rather than a cross product. Both anchors are
the measured ~100M rungs, and every sweep runs at the anchor's `lower_atol`, because at the
models' own default `1e-4` both cutoff and system size are saturated and would sweep a flat
line (`hubbard` c10 = c11 = 1,887,255; `num_sites` 60 = 90).

## What to read, and in what order

1. **`agree` before any ratio.** 6/6 is p=0.031 under a sign test and is the best six reps
   can do; 5/6 is not a result. The collator prints `(consistent)` for a unanimous difference
   outside +-1% precisely because `verdict`'s "flat" band would otherwise assert the absence
   of an effect the data resolved.
2. **RSS before B/term.** The ledger is capacity. It disagrees with the kernel in both
   directions and, on one cell here, in *sign*: pauli at 691k terms reads 0.93x on
   `total_bytes` (the branch larger) and 1.027x in RSS (the branch smaller), because the
   chunked row store's slack is never touched.
3. **Ratios cross layouts; absolute figures do not.** A 1x16 rung differs from an 8x16 one in
   cores, rank count and the flat world P at once, and `memhwm` is a sum over ranks. Compare
   within-cell main/branch ratios only.

## Results



## hubbard

Axis: cutoff, num_sites, observable_site

| cutoff | num_sites | observable_site | N | layout | terms | op | main ms | port ms | port/main | agree | main dmem | port dmem | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 20 | 15 | 1 | B_8x16 | 75,847,560 | propagate[hubbard] | 19287.0 | 20145.7 | 1.05x (flat) | 5/6 | 0.81 | 0.72 | 7.1 |
| 10 | 30 | 23 | 1 | B_8x16 | 90,154,750 | propagate[hubbard] | 22130.4 | 23045.3 | 1.04x (consistent) | 6/6 | 0.96 | 0.82 | 7.9 |
| 10 | 60 | 46 | 1 | B_8x16 | 96,981,051 | propagate[hubbard] | 28556.9 | 30961.8 | 1.09x (consistent) | 6/6 | 1.22 | 0.97 | 9.1 |
| 10 | 60 | 46 | 1 | X_1x16 | 96,981,051 | propagate[hubbard] | 39373.2 | 43411.5 | **1.10x slower** | 6/6 | 6.18 | 5.48 | 5.7 |
| 6 | 60 | 46 | 1 | B_8x16 | 23,933,586 | propagate[hubbard] | 15505.8 | 16425.5 | 1.05x (flat) | 5/6 | 0.28 | 0.23 | 3.3 |
| 8 | 60 | 46 | 1 | B_8x16 | 66,689,918 | propagate[hubbard] | 23357.6 | 24396.4 | 1.05x (consistent) | 6/6 | 0.73 | 0.59 | 6.2 |

### hubbard -- operator memory (B/term)

`main/port` above 1.00x means the port is smaller, for the ledger columns and for RSS alike. The ledger is CAPACITY and the RSS is the kernel's answer for the whole job: they answer different questions and are allowed to disagree. Where they do, the RSS ratio is the one a user feels.

| cutoff | num_sites | observable_site | N | layout | terms | field | main | port | main/port | main RSS | port RSS | RSS main/port |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 20 | 15 | 1 | B_8x16 | 75,847,560 | inverted_index_bytes | 13.30 | 6.17 | 2.15x | 7.9 | 7.1 | 1.111x |
|  |  |  |  |  |  | indexing_bytes | 14.16 | 8.85 | 1.60x |  |  |  |
|  |  |  |  |  |  | operator_terms_bytes | 13.18 | 11.03 | 1.20x |  |  |  |
|  |  |  |  |  |  | total_bytes | 48.65 | 36.06 | 1.35x |  |  |  |
| 10 | 30 | 23 | 1 | B_8x16 | 90,154,750 | inverted_index_bytes | 12.52 | 6.79 | 1.84x | 9.1 | 7.9 | 1.147x |
|  |  |  |  |  |  | indexing_bytes | 11.91 | 7.44 | 1.60x |  |  |  |
|  |  |  |  |  |  | operator_terms_bytes | 16.07 | 11.03 | 1.46x |  |  |  |
|  |  |  |  |  |  | total_bytes | 48.51 | 35.27 | 1.38x |  |  |  |
| 10 | 60 | 46 | 1 | B_8x16 | 96,981,051 | inverted_index_bytes | 13.94 | 7.01 | 1.99x | 11.1 | 9.1 | 1.210x |
|  |  |  |  |  |  | indexing_bytes | 22.14 | 13.84 | 1.60x |  |  |  |
|  |  |  |  |  |  | operator_terms_bytes | 14.94 | 11.03 | 1.35x |  |  |  |
|  |  |  |  |  |  | total_bytes | 59.03 | 41.88 | 1.41x |  |  |  |
| 10 | 60 | 46 | 1 | X_1x16 | 96,981,051 | inverted_index_bytes | 14.00 | 6.16 | 2.27x | 6.4 | 5.7 | 1.123x |
|  |  |  |  |  |  | indexing_bytes | 22.14 | 13.84 | 1.60x |  |  |  |
|  |  |  |  |  |  | operator_terms_bytes | 14.02 | 11.01 | 1.27x |  |  |  |
|  |  |  |  |  |  | total_bytes | 58.17 | 41.01 | 1.42x |  |  |  |
| 6 | 60 | 46 | 1 | B_8x16 | 23,933,586 | inverted_index_bytes | 15.72 | 6.61 | 2.38x | 3.8 | 3.3 | 1.133x |
|  |  |  |  |  |  | indexing_bytes | 22.43 | 14.02 | 1.60x |  |  |  |
|  |  |  |  |  |  | operator_terms_bytes | 7.87 | 7.06 | 1.12x |  |  |  |
|  |  |  |  |  |  | total_bytes | 54.02 | 37.69 | 1.43x |  |  |  |
| 8 | 60 | 46 | 1 | B_8x16 | 66,689,918 | inverted_index_bytes | 15.31 | 6.80 | 2.25x | 7.3 | 6.2 | 1.188x |
|  |  |  |  |  |  | indexing_bytes | 16.10 | 10.06 | 1.60x |  |  |  |
|  |  |  |  |  |  | operator_terms_bytes | 11.85 | 9.05 | 1.31x |  |  |  |
|  |  |  |  |  |  | total_bytes | 51.27 | 35.91 | 1.43x |  |  |  |

`total_bytes` understates the saving: the port sums `matched_scratch_bytes` (up to 2.00 B/term here) into its total and main counts it in no field at all. The `total_bytes` ratios above are a floor, not a best case.

## pauli

Axis: cutoff, num_layers

| cutoff | num_layers | N | layout | terms | op | main ms | port ms | port/main | agree | main dmem | port dmem | peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 20 | 1 | B_8x16 | 4,751,695 | build_graph[pauli] | 3849.1 | 3893.0 | 1.02x (flat) | 4/6 | 0.66 | 0.65 | 10.8 |
|  |  |  |  |  | propagate[pauli] | 3000.8 | 3120.3 | 1.04x (flat) | 3/6 | 0.06 | 0.04 |  |
|  |  |  |  |  | energy[pauli] | 307.9 | 304.0 | 0.98x (flat) | 5/6 | 0.01 | 0.01 |  |
|  |  |  |  |  | gradient[pauli] | 1033.2 | 1054.5 | 1.02x (flat) | 3/6 | 0.18 | 0.18 |  |
| 12 | 20 | 1 | B_8x16 | 29,385,590 | build_graph[pauli] | 7967.2 | 8030.5 | 1.03x (flat) | 4/6 | 1.08 | 1.09 | 17.4 |
|  |  |  |  |  | propagate[pauli] | 7536.7 | 7324.2 | 0.97x (flat) | 5/6 | 0.44 | 0.46 |  |
|  |  |  |  |  | energy[pauli] | 790.3 | 930.8 | **1.17x slower** | 6/6 | 0.03 | 0.03 |  |
|  |  |  |  |  | gradient[pauli] | 3055.7 | 3320.0 | 1.06x (consistent) | 6/6 | 0.22 | 0.22 |  |
| 12 | 20 | 1 | B_8x16 | 29,385,590 | build_graph[pauli] | 7968.8 | 8515.0 | 1.07x (flat) | 5/6 | 1.08 | 1.01 | 15.5 |
|  |  |  |  |  | propagate[pauli] | 7449.3 | 7819.2 | 1.05x (consistent) | 6/6 | 0.44 | 0.30 |  |
|  |  |  |  |  | energy[pauli] | 910.4 | 920.4 | 1.04x (flat) | 4/6 | 0.03 | 0.03 |  |
|  |  |  |  |  | gradient[pauli] | 3184.6 | 3227.9 | 1.01x (flat) | 4/6 | 0.22 | 0.22 |  |
| 12 | 20 | 1 | X_1x16 | 29,385,590 | build_graph[pauli] | 8398.4 | 9504.6 | **1.13x slower** | 6/6 | 4.01 | 3.52 | 6.7 |
|  |  |  |  |  | propagate[pauli] | 7707.3 | 8532.9 | **1.11x slower** | 5/6 | 2.54 | 2.40 |  |
|  |  |  |  |  | energy[pauli] | 1087.3 | 1244.1 | **1.15x slower** | 6/6 | 0.24 | 0.23 |  |
|  |  |  |  |  | gradient[pauli] | 5150.7 | 5390.6 | 1.05x (consistent) | 6/6 | 0.36 | 0.37 |  |
| 14 | 10 | 1 | B_8x16 | 70,924,929 | build_graph[pauli] | 6200.2 | 6349.7 | 1.03x (consistent) | 6/6 | 1.32 | 1.18 | 19.6 |
|  |  |  |  |  | propagate[pauli] | 5814.3 | 5942.7 | 1.02x (flat) | 5/6 | 0.98 | 0.73 |  |
|  |  |  |  |  | energy[pauli] | 877.9 | 918.2 | 1.04x (flat) | 4/6 | 0.08 | 0.08 |  |
|  |  |  |  |  | gradient[pauli] | 3216.0 | 3262.5 | 1.01x (flat) | 3/6 | 0.18 | 0.18 |  |
| 14 | 20 | 1 | B_8x16 | 91,273,861 | build_graph[pauli] | 18160.4 | 18682.4 | 1.03x (flat) | 5/6 | 1.99 | 1.83 | 27.2 |
|  |  |  |  |  | propagate[pauli] | 15570.6 | 15880.9 | 1.02x (flat) | 5/6 | 1.33 | 0.90 |  |
|  |  |  |  |  | energy[pauli] | 3748.4 | 3806.3 | 1.01x (flat) | 3/6 | 0.10 | 0.10 |  |
|  |  |  |  |  | gradient[pauli] | 13756.4 | 13967.2 | 1.00x (flat) | 3/6 | 0.29 | 0.29 |  |
| 14 | 20 | 2 | B_8x16 | 91,273,861 | build_graph[pauli] | 14725.4 | 15304.0 | 1.05x (flat) | 5/6 | 1.82 | 1.73 | 51.0 |
|  |  |  |  |  | propagate[pauli] | 12152.1 | 12273.3 | 1.01x (flat) | 5/6 | 0.66 | 0.43 |  |
|  |  |  |  |  | energy[pauli] | 1389.7 | 1456.4 | 1.05x (consistent) | 6/6 | 0.05 | 0.05 |  |
|  |  |  |  |  | gradient[pauli] | 6445.4 | 6756.9 | 1.05x (flat) | 5/6 | 0.47 | 0.47 |  |
| 14 | 20 | 4 | B_8x16 | 91,273,861 | build_graph[pauli] | 13912.1 | 14295.4 | 1.02x (consistent) | 6/6 | 2.60 | 2.56 | 134.2 |
|  |  |  |  |  | propagate[pauli] | 10394.8 | 10473.2 | 1.01x (flat) | 5/6 | 0.35 | 0.24 |  |
|  |  |  |  |  | energy[pauli] | 1621.0 | 1610.4 | 0.98x (flat) | 3/6 | 0.03 | 0.03 |  |
|  |  |  |  |  | gradient[pauli] | 4472.2 | 4490.0 | 1.01x (flat) | 5/6 | 0.87 | 0.87 |  |
| 14 | 20 | 1 | X_1x16 | 91,273,861 | build_graph[pauli] | 27265.4 | 29849.9 | 1.10x (consistent) | 6/6 | 11.06 | 9.82 | 17.7 |
|  |  |  |  |  | propagate[pauli] | 22517.9 | 24902.8 | **1.10x slower** | 6/6 | 7.11 | 6.37 |  |
|  |  |  |  |  | energy[pauli] | 8066.1 | 8399.0 | 1.04x (consistent) | 6/6 | 0.70 | 0.70 |  |
|  |  |  |  |  | gradient[pauli] | 25981.0 | 24549.9 | 0.94x (flat) | 4/6 | 0.98 | 0.99 |  |
| 14 | 5 | 1 | B_8x16 | 691,430 | build_graph[pauli] | 463.2 | 453.8 | 0.98x (flat) | 4/6 | 0.16 | 0.16 | 5.1 |
|  |  |  |  |  | propagate[pauli] | 284.1 | 277.1 | 0.97x (flat) | 5/6 | 0.02 | 0.01 |  |
|  |  |  |  |  | energy[pauli] | 57.1 | 57.6 | 1.02x (flat) | 3/6 | 0.00 | 0.00 |  |
|  |  |  |  |  | gradient[pauli] | 144.4 | 145.0 | 1.01x (flat) | 4/6 | 0.04 | 0.04 |  |

### pauli -- operator memory (B/term)

`main/port` above 1.00x means the port is smaller, for the ledger columns and for RSS alike. The ledger is CAPACITY and the RSS is the kernel's answer for the whole job: they answer different questions and are allowed to disagree. Where they do, the RSS ratio is the one a user feels.

| cutoff | num_layers | N | layout | terms | field | main | port | main/port | main RSS | port RSS | RSS main/port |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 20 | 1 | B_8x16 | 4,751,695 | inverted_index_bytes | 16.37 | 8.41 | 1.95x | 10.7 | 10.8 | 0.991x |
|  |  |  |  |  | indexing_bytes | 14.13 | 8.83 | 1.60x |  |  |  |
|  |  |  |  |  | operator_terms_bytes | 23.88 | 22.96 | 1.04x |  |  |  |
|  |  |  |  |  | total_bytes | 62.37 | 50.21 | 1.24x |  |  |  |
| 12 | 20 | 1 | B_8x16 | 29,385,590 | inverted_index_bytes | 15.14 | 15.14 | 1.00x | 17.3 | 17.4 | 0.993x |
|  |  |  |  |  | indexing_bytes | 18.27 | 18.27 | 1.00x |  |  |  |
|  |  |  |  |  | operator_terms_bytes | 32.64 | 32.64 | 1.00x |  |  |  |
|  |  |  |  |  | total_bytes | 74.05 | 74.05 | 1.00x |  |  |  |
| 12 | 20 | 1 | B_8x16 | 29,385,590 | inverted_index_bytes | 15.14 | 8.56 | 1.77x | 17.2 | 15.5 | 1.112x |
|  |  |  |  |  | indexing_bytes | 18.27 | 11.42 | 1.60x |  |  |  |
|  |  |  |  |  | operator_terms_bytes | 32.64 | 25.27 | 1.29x |  |  |  |
|  |  |  |  |  | total_bytes | 74.05 | 55.29 | 1.34x |  |  |  |
| 12 | 20 | 1 | X_1x16 | 29,385,590 | inverted_index_bytes | 15.25 | 8.54 | 1.78x | 7.6 | 6.7 | 1.131x |
|  |  |  |  |  | indexing_bytes | 18.27 | 11.42 | 1.60x |  |  |  |
|  |  |  |  |  | operator_terms_bytes | 33.11 | 25.03 | 1.32x |  |  |  |
|  |  |  |  |  | total_bytes | 74.62 | 55.00 | 1.36x |  |  |  |
| 14 | 10 | 1 | B_8x16 | 70,924,929 | inverted_index_bytes | 17.82 | 9.31 | 1.91x | 23.2 | 19.6 | 1.180x |
|  |  |  |  |  | indexing_bytes | 15.14 | 9.46 | 1.60x |  |  |  |
|  |  |  |  |  | operator_terms_bytes | 37.19 | 29.14 | 1.28x |  |  |  |
|  |  |  |  |  | total_bytes | 78.15 | 57.91 | 1.35x |  |  |  |
| 14 | 20 | 1 | B_8x16 | 91,273,861 | inverted_index_bytes | 16.66 | 10.08 | 1.65x | 31.8 | 27.2 | 1.168x |
|  |  |  |  |  | indexing_bytes | 11.76 | 7.35 | 1.60x |  |  |  |
|  |  |  |  |  | operator_terms_bytes | 39.98 | 29.09 | 1.37x |  |  |  |
|  |  |  |  |  | total_bytes | 76.40 | 56.57 | 1.35x |  |  |  |
| 14 | 20 | 2 | B_8x16 | 91,273,861 | inverted_index_bytes | 16.80 | 9.30 | 1.81x | 55.7 | 51.0 | 1.093x |
|  |  |  |  |  | indexing_bytes | 11.76 | 7.35 | 1.60x |  |  |  |
|  |  |  |  |  | operator_terms_bytes | 35.88 | 29.20 | 1.23x |  |  |  |
|  |  |  |  |  | total_bytes | 72.44 | 55.90 | 1.30x |  |  |  |
| 14 | 20 | 4 | B_8x16 | 91,273,861 | inverted_index_bytes | 16.38 | 9.66 | 1.70x | 138.7 | 134.2 | 1.034x |
|  |  |  |  |  | indexing_bytes | 11.76 | 7.35 | 1.60x |  |  |  |
|  |  |  |  |  | operator_terms_bytes | 35.28 | 29.32 | 1.20x |  |  |  |
|  |  |  |  |  | total_bytes | 71.42 | 56.37 | 1.27x |  |  |  |
| 14 | 20 | 1 | X_1x16 | 91,273,861 | inverted_index_bytes | 17.39 | 8.78 | 1.98x | 19.7 | 17.7 | 1.111x |
|  |  |  |  |  | indexing_bytes | 11.76 | 7.35 | 1.60x |  |  |  |
|  |  |  |  |  | operator_terms_bytes | 38.79 | 29.01 | 1.34x |  |  |  |
|  |  |  |  |  | total_bytes | 75.95 | 55.22 | 1.38x |  |  |  |
| 14 | 5 | 1 | B_8x16 | 691,430 | inverted_index_bytes | 6.70 | 5.25 | 1.28x | 5.3 | 5.1 | 1.027x |
|  |  |  |  |  | indexing_bytes | 12.16 | 7.61 | 1.60x |  |  |  |
|  |  |  |  |  | operator_terms_bytes | 35.24 | 43.98 | 0.80x |  |  |  |
|  |  |  |  |  | total_bytes | 62.11 | 66.85 | 0.93x |  |  |  |

`total_bytes` understates the saving: the port sums `matched_scratch_bytes` (up to 2.08 B/term here) into its total and main counts it in no field at all. The `total_bytes` ratios above are a floor, not a best case.

## Void

- `models-hubbard-hub-c10-anchor-N1`: no term counts recorded: cannot show the arms did the same work; placement was not recorded on at least one arm
- `models-hubbard-hub-c10-N2-N2`: no term counts recorded: cannot show the arms did the same work; placement was not recorded on at least one arm
- `models-hubbard-hub-c10-N4-N4`: no term counts recorded: cannot show the arms did the same work; placement was not recorded on at least one arm
- `models-hubbard-hub-c10-serial-N1`: no term counts recorded: cannot show the arms did the same work; placement was not recorded on at least one arm
- `models-hubbard-hub-c6-N1`: no term counts recorded: cannot show the arms did the same work; placement was not recorded on at least one arm
- `models-hubbard-hub-c6-serial-N1`: no term counts recorded: cannot show the arms did the same work; placement was not recorded on at least one arm
- `models-hubbard-hub-c8-N1`: no term counts recorded: cannot show the arms did the same work; placement was not recorded on at least one arm
- `models-hubbard-hub-s20-N1`: no term counts recorded: cannot show the arms did the same work; placement was not recorded on at least one arm
- `models-hubbard-hub-s30-N1`: no term counts recorded: cannot show the arms did the same work; placement was not recorded on at least one arm

These are NOT rows with no effect; they are cells whose arms cannot be compared. Their numbers are omitted above on purpose.
