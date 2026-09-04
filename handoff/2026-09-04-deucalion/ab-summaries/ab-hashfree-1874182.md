# A/B summary: ab-hashfree-1874182

```
key	value
arm_1_name	base
arm_1_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/base-296/.venv
arm_1_md5	f991ce3a65daa476f1229b26643f92ce
arm_1_rev	5ada3da3
arm_1_env	
arm_2_name	s3
arm_2_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/ref-fd795841/.venv
arm_2_md5	457429b288c9ae8616f01e297046fcd0
arm_2_rev	fd795841
arm_2_env	
bench_tree	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/bench-317
bench_rev	e3588ecb
node	cnx003.deucalion.macc.fccn.pt
job	1874182
```

cells: 22 loaded, 0 skipped (failed cells, see stderr)

## A_1x1 hubbard fresh  (3 paired reps)

terms: (9953109,)

| metric | base median | s3 median | s3/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 25.941 s | 41.581 s | 1.603 | 3/3 | 0.250 |
| peak RSS (kernel) | 0.861 GiB | 0.755 GiB | 0.875 | 3/3 | 0.250 |
| operator ledger total | 0.469 GiB | 0.378 GiB | 0.806 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 125.5 MiB | 125.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 122.0 MiB | 122.0 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 75.9 MiB | 75.9 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 45.6 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 18.1 MiB | — | — | — |
| ledger matched_scratch_bytes | 29.2 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 29.2 MiB | 18.1 MiB | 0.620 | 3/3 | 0.250 |
| ledger bytes/term | 50.63 B/term | 40.79 B/term | 0.806 | 3/3 | 0.250 |
| kernel peak bytes/term | 92.84 B/term | 81.42 B/term | 0.875 | 3/3 | 0.250 |

## A_1x1 pauli fresh  (3 paired reps)

terms: (10069308,)

| metric | base median | s3 median | s3/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[pauli] | 20.547 s | 29.882 s | 1.455 | 3/3 | 0.250 |
| peak RSS (kernel) | 1.008 GiB | 0.951 GiB | 0.944 | 3/3 | 0.250 |
| operator ledger total | 0.632 GiB | 0.543 GiB | 0.860 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 285.1 MiB | 285.1 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 132.4 MiB | 132.4 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 76.8 MiB | 76.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 45.6 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 16.2 MiB | — | — | — |
| ledger matched_scratch_bytes | 24.5 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 24.5 MiB | 16.2 MiB | 0.662 | 3/3 | 0.250 |
| ledger bytes/term | 67.36 B/term | 57.92 B/term | 0.860 | 3/3 | 0.250 |
| kernel peak bytes/term | 107.44 B/term | 101.41 B/term | 0.944 | 3/3 | 0.250 |

## A_1x1 randheis graph  (3 paired reps)

terms: (19902244,)

| metric | base median | s3 median | s3/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_random_gradient[heisenberg] | 11.137 s | 10.834 s | 0.972 | 3/3 | 0.250 |
| peak RSS (kernel) | 2.583 GiB | 2.333 GiB | 0.903 | 3/3 | 0.250 |
| operator ledger total | 1.760 GiB | 1.584 GiB | 0.900 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 340.5 MiB | 340.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 256.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 983.5 MiB | 983.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 151.8 MiB | 151.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 97.3 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 49.3 MiB | — | — | — |
| ledger matched_scratch_bytes | 70.8 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 70.8 MiB | 49.3 MiB | 0.696 | 3/3 | 0.250 |
| ledger bytes/term | 94.97 B/term | 85.48 B/term | 0.900 | 3/3 | 0.250 |
| kernel peak bytes/term | 139.38 B/term | 125.86 B/term | 0.903 | 3/3 | 0.250 |

## C_1x128 hubbard fresh  (2 paired reps)

terms: (1001661534,)

| metric | base median | s3 median | s3/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 144.064 s | 200.782 s | 1.394 | 2/2 | 0.500 |
| peak RSS (kernel) | 56.312 GiB | 50.780 GiB | 0.902 | 2/2 | 0.500 |
| operator ledger total | 48.762 GiB | 35.184 GiB | 0.722 | 2/2 | 0.500 |
| ledger operator_terms_bytes | 10903.1 MiB | 10984.8 MiB | 1.007 | 2/2 | 0.500 |
| ledger indexing_bytes | 16384.0 MiB | 0.0 MiB | 0.000 | 2/2 | 0.500 |
| ledger inverted_index_bytes | 12201.2 MiB | 12177.6 MiB | 0.998 | 2/2 | 0.500 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 2/2 | 0.500 |
| ledger row_keys_bytes | — | 3994.5 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 1228.6 MiB | — | — | — |
| ledger matched_scratch_bytes | 2801.9 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 2801.9 MiB | 1228.6 MiB | 0.438 | 2/2 | 0.500 |
| ledger bytes/term | 52.27 B/term | 37.72 B/term | 0.722 | 2/2 | 0.500 |
| kernel peak bytes/term | 60.36 B/term | 54.43 B/term | 0.902 | 2/2 | 0.500 |

## Compact summary

| row | arm | time ratio | RSS ratio | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| A_1x1 hubbard fresh | s3 | 1.603 | 0.875 | 0.806 |
| A_1x1 pauli fresh | s3 | 1.455 | 0.944 | 0.860 |
| A_1x1 randheis graph | s3 | 0.972 | 0.903 | 0.900 |
| C_1x128 hubbard fresh | s3 | 1.394 | 0.902 | 0.722 |

