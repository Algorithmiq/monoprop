# A/B summary: ab-hashfree-1873587

```
key	value
arm_1_name	ref
arm_1_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/ref-ce500790/.venv
arm_1_md5	d15ede5a9c6a1cd6546d0006e5d7ffca
arm_1_rev	ce500790
arm_1_env	
arm_2_name	st
arm_2_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/ref-ce16f0f6/.venv
arm_2_md5	97f6d9185da0796a0ec2e9b1e12d102e
arm_2_rev	ce16f0f6
arm_2_env	
bench_tree	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/bench-317
bench_rev	e3588ecb
node	cnx004.deucalion.macc.fccn.pt
job	1873587
```

cells: 22 loaded, 0 skipped (failed cells, see stderr)

## A_1x1 hubbard fresh  (3 paired reps)

terms: (9953109,)

| metric | ref median | st median | st/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 48.205 s | 48.674 s | 1.010 | 3/3 | 0.250 |
| peak RSS (kernel) | 0.767 GiB | 0.741 GiB | 0.966 | 3/3 | 0.250 |
| operator ledger total | 0.378 GiB | 0.314 GiB | 0.829 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 125.5 MiB | 104.5 MiB | 0.833 | 3/3 | 0.250 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.875 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 122.0 MiB | 84.5 MiB | 0.693 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 75.9 MiB | 75.9 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | 45.6 MiB | 38.0 MiB | 0.833 | 3/3 | 0.250 |
| ledger gate_scratch_bytes | 18.2 MiB | 18.2 MiB | 1.000 | 3/3 | 0.250 |
| ledger scratch (matched|gate) | 18.2 MiB | 18.2 MiB | 1.000 | 3/3 | 0.250 |
| ledger bytes/term | 40.79 B/term | 33.84 B/term | 0.829 | 3/3 | 0.250 |
| kernel peak bytes/term | 82.75 B/term | 79.91 B/term | 0.966 | 3/3 | 0.250 |

## A_1x1 pauli fresh  (3 paired reps)

terms: (10069308,)

| metric | ref median | st median | st/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[pauli] | 33.309 s | 33.499 s | 1.004 | 3/3 | 0.250 |
| peak RSS (kernel) | 0.919 GiB | 0.876 GiB | 0.953 | 3/3 | 0.250 |
| operator ledger total | 0.543 GiB | 0.470 GiB | 0.865 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 285.1 MiB | 243.8 MiB | 0.855 | 3/3 | 0.250 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.875 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 132.4 MiB | 105.5 MiB | 0.797 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 76.8 MiB | 76.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | 45.6 MiB | 39.0 MiB | 0.855 | 3/3 | 0.250 |
| ledger gate_scratch_bytes | 16.3 MiB | 16.3 MiB | 1.000 | 3/3 | 0.250 |
| ledger scratch (matched|gate) | 16.3 MiB | 16.3 MiB | 1.000 | 3/3 | 0.250 |
| ledger bytes/term | 57.92 B/term | 50.13 B/term | 0.865 | 3/3 | 0.250 |
| kernel peak bytes/term | 98.00 B/term | 93.42 B/term | 0.953 | 3/3 | 0.250 |

## A_1x1 randheis graph  (3 paired reps)

terms: (19902244,)

| metric | ref median | st median | st/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_random_gradient[heisenberg] | 10.877 s | 10.676 s | 0.983 | 3/3 | 0.250 |
| peak RSS (kernel) | 2.334 GiB | 2.112 GiB | 0.908 | 3/3 | 0.250 |
| operator ledger total | 1.585 GiB | 1.190 GiB | 0.751 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 340.5 MiB | 266.0 MiB | 0.781 | 3/3 | 0.250 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.875 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 983.5 MiB | 675.0 MiB | 0.686 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 151.8 MiB | 151.8 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | 97.3 MiB | 76.0 MiB | 0.781 | 3/3 | 0.250 |
| ledger gate_scratch_bytes | 49.9 MiB | 49.9 MiB | 1.000 | 3/3 | 0.250 |
| ledger scratch (matched|gate) | 49.9 MiB | 49.9 MiB | 1.000 | 3/3 | 0.250 |
| ledger bytes/term | 85.51 B/term | 64.21 B/term | 0.751 | 3/3 | 0.250 |
| kernel peak bytes/term | 125.90 B/term | 113.95 B/term | 0.908 | 3/3 | 0.250 |

## C_1x128 hubbard fresh  (2 paired reps)

terms: (1001661534,)

| metric | ref median | st median | st/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 205.950 s | 197.403 s | 0.959 | 2/2 | 0.500 |
| peak RSS (kernel) | 49.792 GiB | 47.953 GiB | 0.963 | 2/2 | 0.500 |
| operator ledger total | 35.176 GiB | 32.567 GiB | 0.926 | 2/2 | 0.500 |
| ledger operator_terms_bytes | 10984.8 MiB | 10560.0 MiB | 0.961 | 2/2 | 0.500 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.875 | 2/2 | 0.500 |
| ledger inverted_index_bytes | 12177.6 MiB | 10084.9 MiB | 0.828 | 2/2 | 0.500 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 2/2 | 0.500 |
| ledger row_keys_bytes | 3994.5 MiB | 3840.0 MiB | 0.961 | 2/2 | 0.500 |
| ledger gate_scratch_bytes | 1220.7 MiB | 1220.7 MiB | 1.000 | 2/2 | 0.500 |
| ledger scratch (matched|gate) | 1220.7 MiB | 1220.7 MiB | 1.000 | 2/2 | 0.500 |
| ledger bytes/term | 37.71 B/term | 34.91 B/term | 0.926 | 2/2 | 0.500 |
| kernel peak bytes/term | 53.37 B/term | 51.40 B/term | 0.963 | 2/2 | 0.500 |

## Compact summary

| row | arm | time ratio | RSS ratio | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| A_1x1 hubbard fresh | st | 1.010 | 0.966 | 0.829 |
| A_1x1 pauli fresh | st | 1.004 | 0.953 | 0.865 |
| A_1x1 randheis graph | st | 0.983 | 0.908 | 0.751 |
| C_1x128 hubbard fresh | st | 0.959 | 0.963 | 0.926 |

