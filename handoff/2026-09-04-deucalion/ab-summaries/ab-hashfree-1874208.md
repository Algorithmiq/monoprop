# A/B summary: ab-hashfree-1874208

```
key	value
arm_1_name	base
arm_1_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/base-296/.venv
arm_1_md5	f991ce3a65daa476f1229b26643f92ce
arm_1_rev	5ada3da3
arm_1_env	
arm_2_name	stack
arm_2_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/ref-a1c122a9/.venv
arm_2_md5	eccf7a159a95c9cf14a8a7550d07d229
arm_2_rev	a1c122a9
arm_2_env	
bench_tree	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/bench-317
bench_rev	e3588ecb
node	cnx003.deucalion.macc.fccn.pt
job	1874208
```

cells: 22 loaded, 0 skipped (failed cells, see stderr)

## A_1x1 hubbard fresh  (3 paired reps)

terms: (9953109,)

| metric | base median | stack median | stack/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 25.909 s | 42.744 s | 1.649 | 3/3 | 0.250 |
| peak RSS (kernel) | 0.862 GiB | 0.681 GiB | 0.793 | 3/3 | 0.250 |
| operator ledger total | 0.469 GiB | 0.314 GiB | 0.668 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 125.5 MiB | 104.5 MiB | 0.833 | 3/3 | 0.250 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 122.0 MiB | 84.5 MiB | 0.693 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 75.9 MiB | 75.9 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 38.0 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 18.1 MiB | — | — | — |
| ledger matched_scratch_bytes | 29.2 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 29.2 MiB | 18.1 MiB | 0.620 | 3/3 | 0.250 |
| ledger bytes/term | 50.63 B/term | 33.83 B/term | 0.668 | 3/3 | 0.250 |
| kernel peak bytes/term | 92.95 B/term | 73.43 B/term | 0.793 | 3/3 | 0.250 |

## A_1x1 pauli fresh  (3 paired reps)

terms: (10069308,)

| metric | base median | stack median | stack/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[pauli] | 20.564 s | 29.987 s | 1.458 | 3/3 | 0.250 |
| peak RSS (kernel) | 1.010 GiB | 0.829 GiB | 0.821 | 3/3 | 0.250 |
| operator ledger total | 0.632 GiB | 0.427 GiB | 0.677 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 285.1 MiB | 200.2 MiB | 0.702 | 3/3 | 0.250 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 132.4 MiB | 105.5 MiB | 0.797 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 76.8 MiB | 76.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 39.0 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 16.2 MiB | — | — | — |
| ledger matched_scratch_bytes | 24.5 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 24.5 MiB | 16.2 MiB | 0.662 | 3/3 | 0.250 |
| ledger bytes/term | 67.36 B/term | 45.59 B/term | 0.677 | 3/3 | 0.250 |
| kernel peak bytes/term | 107.70 B/term | 88.40 B/term | 0.821 | 3/3 | 0.250 |

## A_1x1 randheis graph  (3 paired reps)

terms: (19902244,)

| metric | base median | stack median | stack/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_random_gradient[heisenberg] | 11.170 s | 10.658 s | 0.954 | 3/3 | 0.250 |
| peak RSS (kernel) | 2.580 GiB | 2.109 GiB | 0.820 | 3/3 | 0.250 |
| operator ledger total | 1.760 GiB | 1.190 GiB | 0.676 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 340.5 MiB | 266.0 MiB | 0.781 | 3/3 | 0.250 |
| ledger indexing_bytes | 256.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 983.5 MiB | 675.0 MiB | 0.686 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 151.8 MiB | 151.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 76.0 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 49.3 MiB | — | — | — |
| ledger matched_scratch_bytes | 70.8 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 70.8 MiB | 49.3 MiB | 0.696 | 3/3 | 0.250 |
| ledger bytes/term | 94.97 B/term | 64.18 B/term | 0.676 | 3/3 | 0.250 |
| kernel peak bytes/term | 139.21 B/term | 113.79 B/term | 0.820 | 3/3 | 0.250 |

## C_1x128 hubbard fresh  (2 paired reps)

terms: (1001661534,)

| metric | base median | stack median | stack/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 146.264 s | 192.238 s | 1.315 | 2/2 | 0.500 |
| peak RSS (kernel) | 56.125 GiB | 47.463 GiB | 0.846 | 2/2 | 0.500 |
| operator ledger total | 48.762 GiB | 32.574 GiB | 0.668 | 2/2 | 0.500 |
| ledger operator_terms_bytes | 10903.1 MiB | 10560.0 MiB | 0.969 | 2/2 | 0.500 |
| ledger indexing_bytes | 16384.0 MiB | 0.0 MiB | 0.000 | 2/2 | 0.500 |
| ledger inverted_index_bytes | 12201.2 MiB | 10084.9 MiB | 0.827 | 2/2 | 0.500 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 2/2 | 0.500 |
| ledger row_keys_bytes | — | 3840.0 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 1228.6 MiB | — | — | — |
| ledger matched_scratch_bytes | 2801.9 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 2801.9 MiB | 1228.6 MiB | 0.438 | 2/2 | 0.500 |
| ledger bytes/term | 52.27 B/term | 34.92 B/term | 0.668 | 2/2 | 0.500 |
| kernel peak bytes/term | 60.16 B/term | 50.88 B/term | 0.846 | 2/2 | 0.500 |

## Compact summary

| row | arm | time ratio | RSS ratio | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| A_1x1 hubbard fresh | stack | 1.649 | 0.793 | 0.668 |
| A_1x1 pauli fresh | stack | 1.458 | 0.821 | 0.677 |
| A_1x1 randheis graph | stack | 0.954 | 0.820 | 0.676 |
| C_1x128 hubbard fresh | stack | 1.315 | 0.846 | 0.668 |

