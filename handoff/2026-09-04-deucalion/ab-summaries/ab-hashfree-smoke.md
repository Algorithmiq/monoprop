# A/B summary: ab-hashfree-smoke

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
node	ln04
job	smoke
```

cells: 8 loaded, 0 skipped (failed cells, see stderr)

## A_1x1 hubbard fresh  (2 paired reps)

terms: (229,)

| metric | main median | port median | port/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 0.004 s | 0.005 s | 1.361 | 2/2 | 0.500 |
| peak RSS (kernel) | 0.363 GiB | 0.366 GiB | 1.008 | 2/2 | 0.500 |
| operator ledger total | 0.000 GiB | 0.000 GiB | 0.613 | 2/2 | 0.500 |
| ledger operator_terms_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 0.025 | 2/2 | 0.500 |
| ledger inverted_index_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger op_coeffs_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger gate_scratch_bytes | — | 0.0 MiB | — | — | — |
| ledger matched_scratch_bytes | 0.0 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 0.0 MiB | 0.0 MiB | 1.693 | 2/2 | 0.500 |
| ledger bytes/term | 41.53 B/term | 25.44 B/term | 0.613 | 2/2 | 0.500 |
| kernel peak bytes/term | 1702406.71 B/term | 1716018.31 B/term | 1.008 | 2/2 | 0.500 |

## A_1x1 randheis graph  (2 paired reps)

terms: (2034,)

| metric | main median | port median | port/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_random_gradient[heisenberg] | 0.000 s | 0.000 s | 1.027 | 1/2 | 1.000 |
| peak RSS (kernel) | 0.368 GiB | 0.368 GiB | 1.000 | 1/2 | 1.000 |
| operator ledger total | 0.000 GiB | 0.000 GiB | 0.703 | 2/2 | 0.500 |
| ledger operator_terms_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 0.003 | 2/2 | 0.500 |
| ledger inverted_index_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger op_coeffs_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger gate_scratch_bytes | — | 0.0 MiB | — | — | — |
| ledger matched_scratch_bytes | 0.0 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 0.0 MiB | 0.0 MiB | 1.815 | 2/2 | 0.500 |
| ledger bytes/term | 43.56 B/term | 30.63 B/term | 0.703 | 2/2 | 0.500 |
| kernel peak bytes/term | 194036.42 B/term | 194030.38 B/term | 1.000 | 1/2 | 1.000 |

