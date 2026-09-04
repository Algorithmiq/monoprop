# A/B summary: ab-hashfree-1870817

```
key	value
arm_1_name	main
arm_1_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/base-296/.venv
arm_1_md5	f991ce3a65daa476f1229b26643f92ce
arm_1_rev	5ada3da3
arm_1_env	
arm_2_name	exact
arm_2_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/one-round/.venv
arm_2_md5	4dbb818c8ea90332442d0c023a1767f4
arm_2_rev	4da2f2de
arm_2_env	
arm_3_name	drop
arm_3_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/one-round/.venv
arm_3_md5	4dbb818c8ea90332442d0c023a1767f4
arm_3_rev	4da2f2de
arm_3_env	monoprop_DROP_SILENT_RECORDS=1
bench_tree	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/bench-317
bench_rev	e3588ecb
node	cnx001.deucalion.macc.fccn.pt
job	1870817
```

cells: 30 loaded, 0 skipped (failed cells, see stderr)

## A_1x1 hubbard fresh

### main vs exact  (3 paired reps)

terms: (9953109,)

| metric | main median | exact median | exact/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 25.950 s | 565.528 s | 21.825 | 3/3 | 0.250 |
| peak RSS (kernel) | 0.864 GiB | 1.047 GiB | 1.198 | 3/3 | 0.250 |
| operator ledger total | 0.469 GiB | 0.373 GiB | 0.796 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 125.5 MiB | 125.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 122.0 MiB | 122.0 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 75.9 MiB | 75.9 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 45.6 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 13.4 MiB | — | — | — |
| ledger matched_scratch_bytes | 29.2 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 29.2 MiB | 13.4 MiB | 0.458 | 3/3 | 0.250 |
| ledger bytes/term | 50.63 B/term | 40.29 B/term | 0.796 | 3/3 | 0.250 |
| kernel peak bytes/term | 93.24 B/term | 112.97 B/term | 1.198 | 3/3 | 0.250 |

### main vs drop  (3 paired reps)

!! rep 1: term counts differ main=(9953109,) drop=(13094510,)
!! rep 2: term counts differ main=(9953109,) drop=(13094510,)
!! rep 3: term counts differ main=(9953109,) drop=(13094510,)
terms: (13094510,)

| metric | main median | drop median | drop/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 25.950 s | 152.512 s | 5.888 | 3/3 | 0.250 |
| peak RSS (kernel) | 0.864 GiB | 0.929 GiB | 1.076 | 3/3 | 0.250 |
| operator ledger total | 0.469 GiB | 0.509 GiB | 1.084 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 125.5 MiB | 188.2 MiB | 1.500 | 3/3 | 0.250 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 122.0 MiB | 151.0 MiB | 1.238 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 75.9 MiB | 99.9 MiB | 1.316 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 68.4 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 13.4 MiB | — | — | — |
| ledger matched_scratch_bytes | 29.2 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 29.2 MiB | 13.4 MiB | 0.460 | 3/3 | 0.250 |
| ledger bytes/term | 38.49 B/term | 41.72 B/term | 1.084 | 3/3 | 0.250 |
| kernel peak bytes/term | 70.87 B/term | 76.21 B/term | 1.076 | 3/3 | 0.250 |

## A_1x1 pauli fresh

### main vs exact  (3 paired reps)

terms: (10069308,)

| metric | main median | exact median | exact/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[pauli] | 20.584 s | 416.671 s | 20.243 | 3/3 | 0.250 |
| peak RSS (kernel) | 1.016 GiB | 1.383 GiB | 1.353 | 3/3 | 0.250 |
| operator ledger total | 0.632 GiB | 0.539 GiB | 0.854 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 285.1 MiB | 285.1 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 132.4 MiB | 132.4 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 76.8 MiB | 76.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 45.6 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 12.2 MiB | — | — | — |
| ledger matched_scratch_bytes | 24.5 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 24.5 MiB | 12.2 MiB | 0.498 | 3/3 | 0.250 |
| ledger bytes/term | 67.36 B/term | 57.50 B/term | 0.854 | 3/3 | 0.250 |
| kernel peak bytes/term | 108.34 B/term | 147.43 B/term | 1.353 | 3/3 | 0.250 |

### main vs drop  (3 paired reps)

!! rep 1: term counts differ main=(10069308,) drop=(9537132,)
!! rep 2: term counts differ main=(10069308,) drop=(9537132,)
!! rep 3: term counts differ main=(10069308,) drop=(9537132,)
terms: (9537132,)

| metric | main median | drop median | drop/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[pauli] | 20.584 s | 69.356 s | 3.371 | 3/3 | 0.250 |
| peak RSS (kernel) | 1.016 GiB | 1.010 GiB | 0.992 | 2/3 | 1.000 |
| operator ledger total | 0.632 GiB | 0.534 GiB | 0.846 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 285.1 MiB | 285.1 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 132.4 MiB | 131.6 MiB | 0.994 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 76.8 MiB | 72.8 MiB | 0.947 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 45.6 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 12.2 MiB | — | — | — |
| ledger matched_scratch_bytes | 24.5 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 24.5 MiB | 12.2 MiB | 0.498 | 3/3 | 0.250 |
| ledger bytes/term | 71.12 B/term | 60.18 B/term | 0.846 | 3/3 | 0.250 |
| kernel peak bytes/term | 114.38 B/term | 113.75 B/term | 0.992 | 2/3 | 1.000 |

## A_1x1 randheis graph

### main vs exact  (3 paired reps)

terms: (19902244,)

| metric | main median | exact median | exact/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_random_gradient[heisenberg] | 11.139 s | 10.822 s | 0.972 | 3/3 | 0.250 |
| peak RSS (kernel) | 2.585 GiB | 2.392 GiB | 0.926 | 3/3 | 0.250 |
| operator ledger total | 1.760 GiB | 1.600 GiB | 0.909 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 340.5 MiB | 340.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 256.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 983.5 MiB | 983.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 151.8 MiB | 151.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 97.3 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 65.5 MiB | — | — | — |
| ledger matched_scratch_bytes | 70.8 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 70.8 MiB | 65.5 MiB | 0.926 | 3/3 | 0.250 |
| ledger bytes/term | 94.97 B/term | 86.33 B/term | 0.909 | 3/3 | 0.250 |
| kernel peak bytes/term | 139.44 B/term | 129.04 B/term | 0.926 | 3/3 | 0.250 |

### main vs drop  (3 paired reps)

terms: (19902244,)

| metric | main median | drop median | drop/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_random_gradient[heisenberg] | 11.139 s | 10.795 s | 0.970 | 3/3 | 0.250 |
| peak RSS (kernel) | 2.585 GiB | 2.375 GiB | 0.919 | 3/3 | 0.250 |
| operator ledger total | 1.760 GiB | 1.600 GiB | 0.909 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 340.5 MiB | 340.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 256.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 983.5 MiB | 983.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 151.8 MiB | 151.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 97.3 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 65.5 MiB | — | — | — |
| ledger matched_scratch_bytes | 70.8 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 70.8 MiB | 65.5 MiB | 0.926 | 3/3 | 0.250 |
| ledger bytes/term | 94.97 B/term | 86.33 B/term | 0.909 | 3/3 | 0.250 |
| kernel peak bytes/term | 139.44 B/term | 128.15 B/term | 0.919 | 3/3 | 0.250 |

## C_1x128 hubbard fresh

### main vs exact  (1 paired reps)

terms: (1001661534,)

| metric | main median | exact median | exact/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 148.291 s | 1473.701 s | 9.938 | 1/1 | 1.000 |
| peak RSS (kernel) | 56.231 GiB | 93.731 GiB | 1.667 | 1/1 | 1.000 |
| operator ledger total | 48.762 GiB | 34.857 GiB | 0.715 | 1/1 | 1.000 |
| ledger operator_terms_bytes | 10903.1 MiB | 10984.8 MiB | 1.007 | 1/1 | 1.000 |
| ledger indexing_bytes | 16384.0 MiB | 0.0 MiB | 0.000 | 1/1 | 1.000 |
| ledger inverted_index_bytes | 12201.2 MiB | 12177.6 MiB | 0.998 | 1/1 | 1.000 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 1/1 | 1.000 |
| ledger row_keys_bytes | — | 3994.5 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 893.7 MiB | — | — | — |
| ledger matched_scratch_bytes | 2801.9 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 2801.9 MiB | 893.7 MiB | 0.319 | 1/1 | 1.000 |
| ledger bytes/term | 52.27 B/term | 37.36 B/term | 0.715 | 1/1 | 1.000 |
| kernel peak bytes/term | 60.28 B/term | 100.48 B/term | 1.667 | 1/1 | 1.000 |

### main vs drop  (1 paired reps)

!! rep 1: term counts differ main=(1001661534,) drop=(1020270081,)
terms: (1020270081,)

| metric | main median | drop median | drop/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 148.291 s | 272.672 s | 1.839 | 1/1 | 1.000 |
| peak RSS (kernel) | 56.231 GiB | 53.415 GiB | 0.950 | 1/1 | 1.000 |
| operator ledger total | 48.762 GiB | 40.064 GiB | 0.822 | 1/1 | 1.000 |
| ledger operator_terms_bytes | 10903.1 MiB | 12323.0 MiB | 1.130 | 1/1 | 1.000 |
| ledger indexing_bytes | 16384.0 MiB | 0.0 MiB | 0.000 | 1/1 | 1.000 |
| ledger inverted_index_bytes | 12201.2 MiB | 15240.3 MiB | 1.249 | 1/1 | 1.000 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7784.0 MiB | 1.019 | 1/1 | 1.000 |
| ledger row_keys_bytes | — | 4481.1 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 1196.2 MiB | — | — | — |
| ledger matched_scratch_bytes | 2801.9 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 2801.9 MiB | 1196.2 MiB | 0.427 | 1/1 | 1.000 |
| ledger bytes/term | 51.32 B/term | 42.16 B/term | 0.822 | 1/1 | 1.000 |
| kernel peak bytes/term | 59.18 B/term | 56.21 B/term | 0.950 | 1/1 | 1.000 |

## Compact summary

| row | arm | time ratio | RSS ratio | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| A_1x1 hubbard fresh | exact | 21.825 | 1.198 | 0.796 |
| A_1x1 hubbard fresh | drop | 5.888 | 1.076 | 1.084 |
| A_1x1 pauli fresh | exact | 20.243 | 1.353 | 0.854 |
| A_1x1 pauli fresh | drop | 3.371 | 0.992 | 0.846 |
| A_1x1 randheis graph | exact | 0.972 | 0.926 | 0.909 |
| A_1x1 randheis graph | drop | 0.970 | 0.919 | 0.909 |
| C_1x128 hubbard fresh | exact | 9.938 | 1.667 | 0.715 |
| C_1x128 hubbard fresh | drop | 1.839 | 0.950 | 0.822 |

