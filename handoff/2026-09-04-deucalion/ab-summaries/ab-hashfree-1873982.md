# A/B summary: ab-hashfree-1873982

```
key	value
arm_1_name	ref
arm_1_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/ref-ce16f0f6/.venv
arm_1_md5	97f6d9185da0796a0ec2e9b1e12d102e
arm_1_rev	ce16f0f6
arm_1_env	
arm_2_name	wire
arm_2_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/wire/.venv
arm_2_md5	e41aa46a237d276bcb6dc880c25368c4
arm_2_rev	7de5a4e0
arm_2_env	
bench_tree	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/bench-317
bench_rev	e3588ecb
node	cnx003.deucalion.macc.fccn.pt
job	1873982
```

cells: 40 loaded, 0 skipped (failed cells, see stderr)

## A_1x1 hubbard fresh  (5 paired reps)

terms: (9953109,)

| metric | ref median | wire median | wire/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 48.570 s | 48.397 s | 0.996 | 4/5 | 0.375 |
| peak RSS (kernel) | 0.685 GiB | 0.689 GiB | 1.001 | 3/5 | 1.000 |
| operator ledger total | 0.314 GiB | 0.314 GiB | 1.000 | 5/5 | 0.062 |
| ledger operator_terms_bytes | 104.5 MiB | 104.5 MiB | 1.000 | 5/5 | 0.062 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger inverted_index_bytes | 84.5 MiB | 84.5 MiB | 1.000 | 5/5 | 0.062 |
| ledger op_coeffs_bytes | 75.9 MiB | 75.9 MiB | 1.000 | 5/5 | 0.062 |
| ledger row_keys_bytes | 38.0 MiB | 38.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger gate_scratch_bytes | 18.2 MiB | 18.2 MiB | 1.000 | 5/5 | 0.062 |
| ledger scratch (matched|gate) | 18.2 MiB | 18.2 MiB | 1.000 | 5/5 | 0.062 |
| ledger bytes/term | 33.84 B/term | 33.84 B/term | 1.000 | 5/5 | 0.062 |
| kernel peak bytes/term | 73.86 B/term | 74.35 B/term | 1.001 | 3/5 | 1.000 |

## A_1x1 pauli fresh  (5 paired reps)

terms: (10069308,)

| metric | ref median | wire median | wire/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[pauli] | 33.404 s | 33.461 s | 1.002 | 3/5 | 1.000 |
| peak RSS (kernel) | 0.871 GiB | 0.872 GiB | 1.000 | 3/5 | 1.000 |
| operator ledger total | 0.470 GiB | 0.470 GiB | 1.000 | 5/5 | 0.062 |
| ledger operator_terms_bytes | 243.8 MiB | 243.8 MiB | 1.000 | 5/5 | 0.062 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger inverted_index_bytes | 105.5 MiB | 105.5 MiB | 1.000 | 5/5 | 0.062 |
| ledger op_coeffs_bytes | 76.8 MiB | 76.8 MiB | 1.000 | 5/5 | 0.062 |
| ledger row_keys_bytes | 39.0 MiB | 39.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger gate_scratch_bytes | 16.3 MiB | 16.3 MiB | 1.000 | 5/5 | 0.062 |
| ledger scratch (matched|gate) | 16.3 MiB | 16.3 MiB | 1.000 | 5/5 | 0.062 |
| ledger bytes/term | 50.13 B/term | 50.13 B/term | 1.000 | 5/5 | 0.062 |
| kernel peak bytes/term | 92.93 B/term | 92.97 B/term | 1.000 | 3/5 | 1.000 |

## A_1x1 randheis graph  (5 paired reps)

terms: (19902244,)

| metric | ref median | wire median | wire/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_random_gradient[heisenberg] | 10.659 s | 10.620 s | 0.997 | 4/5 | 0.375 |
| peak RSS (kernel) | 2.114 GiB | 2.110 GiB | 0.998 | 4/5 | 0.375 |
| operator ledger total | 1.190 GiB | 1.190 GiB | 1.000 | 5/5 | 0.062 |
| ledger operator_terms_bytes | 266.0 MiB | 266.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger inverted_index_bytes | 675.0 MiB | 675.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger op_coeffs_bytes | 151.8 MiB | 151.8 MiB | 1.000 | 5/5 | 0.062 |
| ledger row_keys_bytes | 76.0 MiB | 76.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger gate_scratch_bytes | 49.9 MiB | 49.9 MiB | 1.000 | 5/5 | 0.062 |
| ledger scratch (matched|gate) | 49.9 MiB | 49.9 MiB | 1.000 | 5/5 | 0.062 |
| ledger bytes/term | 64.21 B/term | 64.21 B/term | 1.000 | 5/5 | 0.062 |
| kernel peak bytes/term | 114.04 B/term | 113.82 B/term | 0.998 | 4/5 | 0.375 |

## C_1x128 hubbard fresh  (5 paired reps)

terms: (1001661534,)

| metric | ref median | wire median | wire/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 199.604 s | 187.886 s | 0.944 | 5/5 | 0.062 |
| peak RSS (kernel) | 48.027 GiB | 47.897 GiB | 0.993 | 3/5 | 1.000 |
| operator ledger total | 32.567 GiB | 32.567 GiB | 1.000 | 5/5 | 0.062 |
| ledger operator_terms_bytes | 10560.0 MiB | 10560.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger inverted_index_bytes | 10084.9 MiB | 10084.9 MiB | 1.000 | 5/5 | 0.062 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 5/5 | 0.062 |
| ledger row_keys_bytes | 3840.0 MiB | 3840.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger gate_scratch_bytes | 1220.7 MiB | 1220.7 MiB | 1.000 | 5/5 | 0.062 |
| ledger scratch (matched|gate) | 1220.7 MiB | 1220.7 MiB | 1.000 | 5/5 | 0.062 |
| ledger bytes/term | 34.91 B/term | 34.91 B/term | 1.000 | 5/5 | 0.062 |
| kernel peak bytes/term | 51.48 B/term | 51.34 B/term | 0.993 | 3/5 | 1.000 |

## Compact summary

| row | arm | time ratio | RSS ratio | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| A_1x1 hubbard fresh | wire | 0.996 | 1.001 | 1.000 |
| A_1x1 pauli fresh | wire | 1.002 | 1.000 | 1.000 |
| A_1x1 randheis graph | wire | 0.997 | 0.998 | 1.000 |
| C_1x128 hubbard fresh | wire | 0.944 | 0.993 | 1.000 |

