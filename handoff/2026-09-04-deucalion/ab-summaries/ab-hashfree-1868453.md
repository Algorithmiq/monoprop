# A/B summary: ab-hashfree-1868453

```
key	value
main_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/base-296/.venv
port_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/hashfree-kernel/.venv
main_md5	f991ce3a65daa476f1229b26643f92ce
port_md5	20c87a4ae4ac32ea9012e3d70c25442d
bench_tree	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/bench-317
bench_rev	e3588ecb
main_rev	5ada3da3
port_rev	1e7a7e98
node	cnx001.deucalion.macc.fccn.pt
job	1868453
```

cells: 42 loaded, 1 skipped (failed cells, see stderr)

## A_1x1 hubbard fresh  (7 paired reps)

terms: (9953109,)

| metric | main median | port median | port/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 25.869 s | 276.973 s | 10.686 ** | 7/7 | 0.016 |
| peak RSS (kernel) | 0.862 GiB | 0.770 GiB | 0.892 ** | 7/7 | 0.016 |
| operator ledger total | 0.469 GiB | 0.316 GiB | 0.673 ** | 7/7 | 0.016 |
| ledger operator_terms_bytes | 125.5 MiB | 125.5 MiB | 1.000 | 7/7 | 0.016 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 ** | 7/7 | 0.016 |
| ledger inverted_index_bytes | 122.0 MiB | 122.0 MiB | 1.000 | 7/7 | 0.016 |
| ledger op_coeffs_bytes | 75.9 MiB | 75.9 MiB | 1.000 | 7/7 | 0.016 |
| ledger gate_scratch_bytes | — | 0.0 MiB | — | — | — |
| ledger matched_scratch_bytes | 29.2 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 29.2 MiB | 0.0 MiB | 0.000 ** | 7/7 | 0.016 |
| ledger bytes/term | 50.63 B/term | 34.07 B/term | 0.673 ** | 7/7 | 0.016 |
| kernel peak bytes/term | 92.99 B/term | 83.12 B/term | 0.892 ** | 7/7 | 0.016 |

## A_1x1 pauli fresh  (7 paired reps)

terms: (10069308,)

| metric | main median | port median | port/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[pauli] | 20.490 s | 180.507 s | 8.785 ** | 7/7 | 0.016 |
| peak RSS (kernel) | 1.023 GiB | 0.945 GiB | 0.921 ** | 7/7 | 0.016 |
| operator ledger total | 0.632 GiB | 0.483 GiB | 0.764 ** | 7/7 | 0.016 |
| ledger operator_terms_bytes | 285.1 MiB | 285.1 MiB | 1.000 | 7/7 | 0.016 |
| ledger indexing_bytes | 128.0 MiB | 0.0 MiB | 0.000 ** | 7/7 | 0.016 |
| ledger inverted_index_bytes | 132.4 MiB | 132.4 MiB | 1.000 | 7/7 | 0.016 |
| ledger op_coeffs_bytes | 76.8 MiB | 76.8 MiB | 1.000 | 7/7 | 0.016 |
| ledger gate_scratch_bytes | — | 0.0 MiB | — | — | — |
| ledger matched_scratch_bytes | 24.5 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 24.5 MiB | 0.0 MiB | 0.000 ** | 7/7 | 0.016 |
| ledger bytes/term | 67.36 B/term | 51.48 B/term | 0.764 ** | 7/7 | 0.016 |
| kernel peak bytes/term | 109.07 B/term | 100.78 B/term | 0.921 ** | 7/7 | 0.016 |

## A_1x1 randheis graph  (7 paired reps)

terms: (19902244,)

| metric | main median | port median | port/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_random_gradient[heisenberg] | 11.136 s | 11.127 s | 1.000 | 4/7 | 1.000 |
| peak RSS (kernel) | 2.578 GiB | 2.351 GiB | 0.911 ** | 7/7 | 0.016 |
| operator ledger total | 1.760 GiB | 1.496 GiB | 0.850 ** | 7/7 | 0.016 |
| ledger operator_terms_bytes | 340.5 MiB | 340.5 MiB | 1.000 | 7/7 | 0.016 |
| ledger indexing_bytes | 256.0 MiB | 0.0 MiB | 0.000 ** | 7/7 | 0.016 |
| ledger inverted_index_bytes | 983.5 MiB | 983.5 MiB | 1.000 | 7/7 | 0.016 |
| ledger op_coeffs_bytes | 151.8 MiB | 151.8 MiB | 1.000 | 7/7 | 0.016 |
| ledger gate_scratch_bytes | — | 56.2 MiB | — | — | — |
| ledger matched_scratch_bytes | 70.8 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 70.8 MiB | 56.2 MiB | 0.795 ** | 7/7 | 0.016 |
| ledger bytes/term | 94.97 B/term | 80.72 B/term | 0.850 ** | 7/7 | 0.016 |
| kernel peak bytes/term | 139.08 B/term | 126.83 B/term | 0.911 ** | 7/7 | 0.016 |

