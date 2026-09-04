# A/B summary: ab-hashfree-1871300

```
key	value
arm_1_name	base
arm_1_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/base-296/.venv
arm_1_md5	f991ce3a65daa476f1229b26643f92ce
arm_1_rev	5ada3da3
arm_1_env	
arm_2_name	r15
arm_2_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/one-round/.venv
arm_2_md5	f0508bd29598d83c7905a8a3e18fd974
arm_2_rev	60254b20
arm_2_env	
bench_tree	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/bench-317
bench_rev	e3588ecb
node	cnx001.deucalion.macc.fccn.pt
job	1871300
```

cells: 22 loaded, 0 skipped (failed cells, see stderr)

## A_1x1 hubbard fresh  (3 paired reps)

terms: (9953109,)

| metric | base median | r15 median | r15/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 25.928 s | 116.615 s | 4.486 | 3/3 | 0.250 |
| peak RSS (kernel) | 0.866 GiB | 0.824 GiB | 0.952 | 3/3 | 0.250 |
| operator ledger total | 0.469 GiB | 0.375 GiB | 0.799 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 125.5 MiB | 125.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 122.0 MiB | 122.0 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 75.9 MiB | 75.9 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 45.6 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 15.1 MiB | — | — | — |
| ledger matched_scratch_bytes | 29.2 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 29.2 MiB | 15.1 MiB | 0.519 | 3/3 | 0.250 |
| ledger bytes/term | 50.63 B/term | 40.47 B/term | 0.799 | 3/3 | 0.250 |
| kernel peak bytes/term | 93.43 B/term | 88.93 B/term | 0.952 | 3/3 | 0.250 |

## A_1x1 pauli fresh  (3 paired reps)

terms: (10069308,)

| metric | base median | r15 median | r15/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[pauli] | 20.608 s | 74.785 s | 3.629 | 3/3 | 0.250 |
| peak RSS (kernel) | 1.027 GiB | 0.995 GiB | 0.995 | 3/3 | 0.250 |
| operator ledger total | 0.632 GiB | 0.541 GiB | 0.856 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 285.1 MiB | 285.1 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 132.4 MiB | 132.4 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 76.8 MiB | 76.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 45.6 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 13.7 MiB | — | — | — |
| ledger matched_scratch_bytes | 24.5 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 24.5 MiB | 13.7 MiB | 0.561 | 3/3 | 0.250 |
| ledger bytes/term | 67.36 B/term | 57.66 B/term | 0.856 | 3/3 | 0.250 |
| kernel peak bytes/term | 109.56 B/term | 106.08 B/term | 0.995 | 3/3 | 0.250 |

## A_1x1 randheis graph  (3 paired reps)

terms: (19902244,)

| metric | base median | r15 median | r15/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_random_gradient[heisenberg] | 11.165 s | 10.859 s | 0.972 | 3/3 | 0.250 |
| peak RSS (kernel) | 2.579 GiB | 2.386 GiB | 0.924 | 3/3 | 0.250 |
| operator ledger total | 1.760 GiB | 1.605 GiB | 0.911 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 340.5 MiB | 340.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 256.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 983.5 MiB | 983.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 151.8 MiB | 151.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 97.3 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 69.9 MiB | — | — | — |
| ledger matched_scratch_bytes | 70.8 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 70.8 MiB | 69.9 MiB | 0.988 | 3/3 | 0.250 |
| ledger bytes/term | 94.97 B/term | 86.57 B/term | 0.911 | 3/3 | 0.250 |
| kernel peak bytes/term | 139.13 B/term | 128.75 B/term | 0.924 | 3/3 | 0.250 |

## C_1x128 hubbard fresh  (2 paired reps)

terms: (1001661534,)

| metric | base median | r15 median | r15/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 149.290 s | 322.406 s | 2.160 | 2/2 | 0.500 |
| peak RSS (kernel) | 56.415 GiB | 48.989 GiB | 0.868 | 2/2 | 0.500 |
| operator ledger total | 48.762 GiB | 34.973 GiB | 0.717 | 2/2 | 0.500 |
| ledger operator_terms_bytes | 10903.1 MiB | 10984.8 MiB | 1.007 | 2/2 | 0.500 |
| ledger indexing_bytes | 16384.0 MiB | 0.0 MiB | 0.000 | 2/2 | 0.500 |
| ledger inverted_index_bytes | 12201.2 MiB | 12177.6 MiB | 0.998 | 2/2 | 0.500 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 2/2 | 0.500 |
| ledger row_keys_bytes | — | 3994.5 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 1012.6 MiB | — | — | — |
| ledger matched_scratch_bytes | 2801.9 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 2801.9 MiB | 1012.6 MiB | 0.361 | 2/2 | 0.500 |
| ledger bytes/term | 52.27 B/term | 37.49 B/term | 0.717 | 2/2 | 0.500 |
| kernel peak bytes/term | 60.47 B/term | 52.51 B/term | 0.868 | 2/2 | 0.500 |

## Compact summary

| row | arm | time ratio | RSS ratio | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| A_1x1 hubbard fresh | r15 | 4.486 | 0.952 | 0.799 |
| A_1x1 pauli fresh | r15 | 3.629 | 0.995 | 0.856 |
| A_1x1 randheis graph | r15 | 0.972 | 0.924 | 0.911 |
| C_1x128 hubbard fresh | r15 | 2.160 | 0.868 | 0.717 |

