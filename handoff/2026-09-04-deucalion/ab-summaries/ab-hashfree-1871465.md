# A/B summary: ab-hashfree-1871465

```
key	value
arm_1_name	base
arm_1_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/base-296/.venv
arm_1_md5	f991ce3a65daa476f1229b26643f92ce
arm_1_rev	5ada3da3
arm_1_env	
arm_2_name	k2
arm_2_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/ref-bff293d6/.venv
arm_2_md5	ca0d2b5286ef1e9a266bad039bfea466
arm_2_rev	bff293d6
arm_2_env	
bench_tree	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/bench-317
bench_rev	e3588ecb
node	cnx007.deucalion.macc.fccn.pt
job	1871465
```

cells: 22 loaded, 0 skipped (failed cells, see stderr)

## A_1x1 hubbard fresh  (3 paired reps)

terms: (9953109,)

| metric | base median | k2 median | k2/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 26.118 s | 51.330 s | 1.968 | 3/3 | 0.250 |
| peak RSS (kernel) | 0.871 GiB | 0.840 GiB | 0.976 | 2/3 | 1.000 |
| operator ledger total | 0.469 GiB | 0.378 GiB | 0.806 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 125.5 MiB | 125.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 122.0 MiB | 122.0 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 75.9 MiB | 75.9 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 45.6 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 18.2 MiB | — | — | — |
| ledger matched_scratch_bytes | 29.2 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 29.2 MiB | 18.2 MiB | 0.623 | 3/3 | 0.250 |
| ledger bytes/term | 50.63 B/term | 40.79 B/term | 0.806 | 3/3 | 0.250 |
| kernel peak bytes/term | 93.94 B/term | 90.57 B/term | 0.976 | 2/3 | 1.000 |

## A_1x1 pauli fresh  (3 paired reps)

terms: (10069308,)

| metric | base median | k2 median | k2/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[pauli] | 20.785 s | 33.500 s | 1.612 | 3/3 | 0.250 |
| peak RSS (kernel) | 1.013 GiB | 0.967 GiB | 0.946 | 3/3 | 0.250 |
| operator ledger total | 0.632 GiB | 0.543 GiB | 0.860 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 285.1 MiB | 285.1 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 132.4 MiB | 132.4 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 76.8 MiB | 76.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 45.6 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 16.3 MiB | — | — | — |
| ledger matched_scratch_bytes | 24.5 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 24.5 MiB | 16.3 MiB | 0.664 | 3/3 | 0.250 |
| ledger bytes/term | 67.36 B/term | 57.92 B/term | 0.860 | 3/3 | 0.250 |
| kernel peak bytes/term | 108.05 B/term | 103.13 B/term | 0.946 | 3/3 | 0.250 |

## A_1x1 randheis graph  (3 paired reps)

terms: (19902244,)

| metric | base median | k2 median | k2/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_random_gradient[heisenberg] | 11.161 s | 10.904 s | 0.975 | 3/3 | 0.250 |
| peak RSS (kernel) | 2.573 GiB | 2.373 GiB | 0.922 | 3/3 | 0.250 |
| operator ledger total | 1.760 GiB | 1.585 GiB | 0.900 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 340.5 MiB | 340.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 256.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 983.5 MiB | 983.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 151.8 MiB | 151.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 97.3 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 49.9 MiB | — | — | — |
| ledger matched_scratch_bytes | 70.8 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 70.8 MiB | 49.9 MiB | 0.706 | 3/3 | 0.250 |
| ledger bytes/term | 94.97 B/term | 85.51 B/term | 0.900 | 3/3 | 0.250 |
| kernel peak bytes/term | 138.84 B/term | 128.02 B/term | 0.922 | 3/3 | 0.250 |

## C_1x128 hubbard fresh  (2 paired reps)

terms: (1001661534,)

| metric | base median | k2 median | k2/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 146.932 s | 213.784 s | 1.456 | 2/2 | 0.500 |
| peak RSS (kernel) | 56.346 GiB | 51.059 GiB | 0.906 | 2/2 | 0.500 |
| operator ledger total | 48.762 GiB | 35.176 GiB | 0.721 | 2/2 | 0.500 |
| ledger operator_terms_bytes | 10903.1 MiB | 10984.8 MiB | 1.007 | 2/2 | 0.500 |
| ledger indexing_bytes | 16384.0 MiB | 0.0 MiB | 0.000 | 2/2 | 0.500 |
| ledger inverted_index_bytes | 12201.2 MiB | 12177.6 MiB | 0.998 | 2/2 | 0.500 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 2/2 | 0.500 |
| ledger row_keys_bytes | — | 3994.5 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 1220.7 MiB | — | — | — |
| ledger matched_scratch_bytes | 2801.9 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 2801.9 MiB | 1220.7 MiB | 0.436 | 2/2 | 0.500 |
| ledger bytes/term | 52.27 B/term | 37.71 B/term | 0.721 | 2/2 | 0.500 |
| kernel peak bytes/term | 60.40 B/term | 54.73 B/term | 0.906 | 2/2 | 0.500 |

## Compact summary

| row | arm | time ratio | RSS ratio | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| A_1x1 hubbard fresh | k2 | 1.968 | 0.976 | 0.806 |
| A_1x1 pauli fresh | k2 | 1.612 | 0.946 | 0.860 |
| A_1x1 randheis graph | k2 | 0.975 | 0.922 | 0.900 |
| C_1x128 hubbard fresh | k2 | 1.456 | 0.906 | 0.721 |

