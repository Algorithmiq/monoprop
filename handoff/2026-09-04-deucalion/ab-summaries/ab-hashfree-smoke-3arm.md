# A/B summary: ab-hashfree-smoke-3arm

```
key	value
arm_1_name	main
arm_1_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/base-296/.venv
arm_1_md5	f991ce3a65daa476f1229b26643f92ce
arm_1_rev	5ada3da3
arm_1_env	
arm_2_name	port
arm_2_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/hashfree-kernel/.venv
arm_2_md5	20c87a4ae4ac32ea9012e3d70c25442d
arm_2_rev	1e7a7e98
arm_2_env	
arm_3_name	alt
arm_3_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/hashfree-kernel/.venv
arm_3_md5	20c87a4ae4ac32ea9012e3d70c25442d
arm_3_rev	1e7a7e98
arm_3_env	monoprop_DUMMY=1
bench_tree	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/bench-317
bench_rev	e3588ecb
node	ln04
job	smoke-3arm
```

cells: 12 loaded, 0 skipped (failed cells, see stderr)

## A_1x1 hubbard fresh

### main vs port  (2 paired reps)

terms: (229,)

| metric | main median | port median | port/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 0.004 s | 0.005 s | 1.359 | 2/2 | 0.500 |
| peak RSS (kernel) | 0.363 GiB | 0.364 GiB | 1.003 | 2/2 | 0.500 |
| operator ledger total | 0.000 GiB | 0.000 GiB | 0.613 | 2/2 | 0.500 |
| ledger operator_terms_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 0.025 | 2/2 | 0.500 |
| ledger inverted_index_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger op_coeffs_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger gate_scratch_bytes | — | 0.0 MiB | — | — | — |
| ledger matched_scratch_bytes | 0.0 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 0.0 MiB | 0.0 MiB | 1.693 | 2/2 | 0.500 |
| ledger bytes/term | 41.53 B/term | 25.44 B/term | 0.613 | 2/2 | 0.500 |
| kernel peak bytes/term | 1703372.58 B/term | 1707674.27 B/term | 1.003 | 2/2 | 0.500 |

### main vs alt  (2 paired reps)

terms: (229,)

| metric | main median | alt median | alt/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 0.004 s | 0.005 s | 1.339 | 2/2 | 0.500 |
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
| kernel peak bytes/term | 1703372.58 B/term | 1716474.41 B/term | 1.008 | 2/2 | 0.500 |

## A_1x1 randheis graph

### main vs port  (2 paired reps)

terms: (2034,)

| metric | main median | port median | port/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_random_gradient[heisenberg] | 0.000 s | 0.000 s | 0.918 | 1/2 | 1.000 |
| peak RSS (kernel) | 0.367 GiB | 0.366 GiB | 0.998 | 1/2 | 1.000 |
| operator ledger total | 0.000 GiB | 0.000 GiB | 0.703 | 2/2 | 0.500 |
| ledger operator_terms_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 0.003 | 2/2 | 0.500 |
| ledger inverted_index_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger op_coeffs_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger gate_scratch_bytes | — | 0.0 MiB | — | — | — |
| ledger matched_scratch_bytes | 0.0 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 0.0 MiB | 0.0 MiB | 1.815 | 2/2 | 0.500 |
| ledger bytes/term | 43.56 B/term | 30.63 B/term | 0.703 | 2/2 | 0.500 |
| kernel peak bytes/term | 193686.03 B/term | 193261.12 B/term | 0.998 | 1/2 | 1.000 |

### main vs alt  (2 paired reps)

terms: (2034,)

| metric | main median | alt median | alt/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_random_gradient[heisenberg] | 0.000 s | 0.000 s | 0.958 | 2/2 | 0.500 |
| peak RSS (kernel) | 0.367 GiB | 0.365 GiB | 0.994 | 2/2 | 0.500 |
| operator ledger total | 0.000 GiB | 0.000 GiB | 0.703 | 2/2 | 0.500 |
| ledger operator_terms_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 0.003 | 2/2 | 0.500 |
| ledger inverted_index_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger op_coeffs_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 2/2 | 0.500 |
| ledger gate_scratch_bytes | — | 0.0 MiB | — | — | — |
| ledger matched_scratch_bytes | 0.0 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 0.0 MiB | 0.0 MiB | 1.815 | 2/2 | 0.500 |
| ledger bytes/term | 43.56 B/term | 30.63 B/term | 0.703 | 2/2 | 0.500 |
| kernel peak bytes/term | 193686.03 B/term | 192580.47 B/term | 0.994 | 2/2 | 0.500 |

## Compact summary

| row | arm | time ratio | RSS ratio | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| A_1x1 hubbard fresh | port | 1.359 | 1.003 | 0.613 |
| A_1x1 hubbard fresh | alt | 1.339 | 1.008 | 0.613 |
| A_1x1 randheis graph | port | 0.918 | 0.998 | 0.703 |
| A_1x1 randheis graph | alt | 0.958 | 0.994 | 0.703 |

