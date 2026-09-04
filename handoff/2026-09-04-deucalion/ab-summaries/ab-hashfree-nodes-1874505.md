# A/B summary: ab-hashfree-nodes-1874505

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
layout	B_8x16
rung	L2b
nodes	1
ranks	8
ranks_per_node	8
cores_per_rank	16
partitions_per_rank	16
reps	3
smoke	0
rss_source	bench-json memhwm (ranks' VmHWM summed)
partition	dev-x86
nodelist	cnx001
job	1874505
```

cells: 6 loaded, 0 skipped (failed cells, see stderr)

## B_8x16 hubbard fresh  (3 paired reps)

terms: (1001661534,)

| metric | base median | stack median | stack/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 130.299 s | 176.437 s | 1.354 | 3/3 | 0.250 |
| peak RSS (ranks summed) | 58.528 GiB | 48.161 GiB | 0.823 | 3/3 | 0.250 |
| operator ledger total | 48.764 GiB | 32.568 GiB | 0.668 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 10860.4 MiB | 10560.0 MiB | 0.972 | 3/3 | 0.250 |
| ledger indexing_bytes | 16384.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 12202.0 MiB | 10085.0 MiB | 0.827 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 3840.0 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 1221.7 MiB | — | — | — |
| ledger matched_scratch_bytes | 2845.6 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 2845.6 MiB | 1221.7 MiB | 0.429 | 3/3 | 0.250 |
| ledger bytes/term | 52.27 B/term | 34.91 B/term | 0.668 | 3/3 | 0.250 |
| peak bytes/term (ranks summed) | 62.74 B/term | 51.63 B/term | 0.823 | 3/3 | 0.250 |

## Compact summary

| row | arm | time ratio | RSS ratio (summed) | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| B_8x16 hubbard fresh | stack | 1.354 | 0.823 | 0.668 |

