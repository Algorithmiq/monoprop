# A/B summary: ab-hashfree-1871456

```
key	value
arm_1_name	ref
arm_1_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/ref-60254b20/.venv
arm_1_md5	f0508bd29598d83c7905a8a3e18fd974
arm_1_rev	60254b20
arm_1_env	
arm_2_name	st2
arm_2_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/storage/.venv-2fd00060
arm_2_md5	219e4843f44b2a1cdf052e92d5f4b7c8
arm_2_rev	2fd00060
arm_2_env	
bench_tree	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/bench-317
bench_rev	e3588ecb
node	cnx001.deucalion.macc.fccn.pt
job	1871456
```

cells: 22 loaded, 0 skipped (failed cells, see stderr)

## A_1x1 hubbard fresh  (3 paired reps)

terms: (9953109,)

| metric | ref median | st2 median | st2/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 117.008 s | 117.222 s | 1.004 | 3/3 | 0.250 |
| peak RSS (kernel) | 0.829 GiB | 0.822 GiB | 0.994 | 2/3 | 1.000 |
| operator ledger total | 0.375 GiB | 0.375 GiB | 1.000 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 125.5 MiB | 125.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 122.0 MiB | 122.0 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 75.9 MiB | 75.9 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | 45.6 MiB | 45.6 MiB | 1.000 | 3/3 | 0.250 |
| ledger gate_scratch_bytes | 15.1 MiB | 15.1 MiB | 1.000 | 3/3 | 0.250 |
| ledger scratch (matched|gate) | 15.1 MiB | 15.1 MiB | 1.000 | 3/3 | 0.250 |
| ledger bytes/term | 40.47 B/term | 40.47 B/term | 1.000 | 3/3 | 0.250 |
| kernel peak bytes/term | 89.42 B/term | 88.68 B/term | 0.994 | 2/3 | 1.000 |

## A_1x1 pauli fresh  (3 paired reps)

terms: (10069308,)

| metric | ref median | st2 median | st2/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[pauli] | 74.600 s | 74.974 s | 1.006 | 2/3 | 1.000 |
| peak RSS (kernel) | 1.052 GiB | 1.018 GiB | 0.967 | 3/3 | 0.250 |
| operator ledger total | 0.541 GiB | 0.541 GiB | 1.000 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 285.1 MiB | 285.1 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 132.4 MiB | 132.4 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 76.8 MiB | 76.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | 45.6 MiB | 45.6 MiB | 1.000 | 3/3 | 0.250 |
| ledger gate_scratch_bytes | 13.7 MiB | 13.7 MiB | 1.000 | 3/3 | 0.250 |
| ledger scratch (matched|gate) | 13.7 MiB | 13.7 MiB | 1.000 | 3/3 | 0.250 |
| ledger bytes/term | 57.66 B/term | 57.66 B/term | 1.000 | 3/3 | 0.250 |
| kernel peak bytes/term | 112.19 B/term | 108.51 B/term | 0.967 | 3/3 | 0.250 |

## A_1x1 randheis graph  (3 paired reps)

terms: (19902244,)

| metric | ref median | st2 median | st2/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_random_gradient[heisenberg] | 10.819 s | 10.996 s | 1.018 | 3/3 | 0.250 |
| peak RSS (kernel) | 2.392 GiB | 2.387 GiB | 0.998 | 2/3 | 1.000 |
| operator ledger total | 1.605 GiB | 1.605 GiB | 1.000 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 340.5 MiB | 340.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 983.5 MiB | 983.5 MiB | 1.000 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 151.8 MiB | 151.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | 97.3 MiB | 97.3 MiB | 1.000 | 3/3 | 0.250 |
| ledger gate_scratch_bytes | 69.9 MiB | 69.9 MiB | 1.000 | 3/3 | 0.250 |
| ledger scratch (matched|gate) | 69.9 MiB | 69.9 MiB | 1.000 | 3/3 | 0.250 |
| ledger bytes/term | 86.57 B/term | 86.57 B/term | 1.000 | 3/3 | 0.250 |
| kernel peak bytes/term | 129.05 B/term | 128.80 B/term | 0.998 | 2/3 | 1.000 |

## C_1x128 hubbard fresh  (2 paired reps)

terms: (1001661534,)

| metric | ref median | st2 median | st2/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 319.288 s | 322.686 s | 1.011 | 2/2 | 0.500 |
| peak RSS (kernel) | 49.201 GiB | 49.573 GiB | 1.008 | 2/2 | 0.500 |
| operator ledger total | 34.973 GiB | 34.973 GiB | 1.000 | 2/2 | 0.500 |
| ledger operator_terms_bytes | 10984.8 MiB | 10984.8 MiB | 1.000 | 2/2 | 0.500 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger inverted_index_bytes | 12177.6 MiB | 12177.6 MiB | 1.000 | 2/2 | 0.500 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 2/2 | 0.500 |
| ledger row_keys_bytes | 3994.5 MiB | 3994.5 MiB | 1.000 | 2/2 | 0.500 |
| ledger gate_scratch_bytes | 1012.6 MiB | 1012.6 MiB | 1.000 | 2/2 | 0.500 |
| ledger scratch (matched|gate) | 1012.6 MiB | 1012.6 MiB | 1.000 | 2/2 | 0.500 |
| ledger bytes/term | 37.49 B/term | 37.49 B/term | 1.000 | 2/2 | 0.500 |
| kernel peak bytes/term | 52.74 B/term | 53.14 B/term | 1.008 | 2/2 | 0.500 |

## Compact summary

| row | arm | time ratio | RSS ratio | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| A_1x1 hubbard fresh | st2 | 1.004 | 0.994 | 1.000 |
| A_1x1 pauli fresh | st2 | 1.006 | 0.967 | 1.000 |
| A_1x1 randheis graph | st2 | 1.018 | 0.998 | 1.000 |
| C_1x128 hubbard fresh | st2 | 1.011 | 1.008 | 1.000 |

