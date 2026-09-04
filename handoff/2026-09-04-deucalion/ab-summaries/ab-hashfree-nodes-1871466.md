# A/B summary: ab-hashfree-nodes-1871466

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
nodelist	cnx008
job	1871466
```

cells: 6 loaded, 0 skipped (failed cells, see stderr)

## B_8x16 hubbard fresh  (3 paired reps)

terms: (1001661534,)

| metric | base median | k2 median | k2/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 131.044 s | 194.038 s | 1.480 | 3/3 | 0.250 |
| peak RSS (ranks summed) | 58.560 GiB | 54.242 GiB | 0.926 | 3/3 | 0.250 |
| operator ledger total | 48.764 GiB | 35.375 GiB | 0.725 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 10860.4 MiB | 11165.1 MiB | 1.028 | 3/3 | 0.250 |
| ledger indexing_bytes | 16384.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 12202.0 MiB | 12142.1 MiB | 0.995 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 4060.1 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 1213.7 MiB | — | — | — |
| ledger matched_scratch_bytes | 2845.6 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 2845.6 MiB | 1213.7 MiB | 0.427 | 3/3 | 0.250 |
| ledger bytes/term | 52.27 B/term | 37.92 B/term | 0.725 | 3/3 | 0.250 |
| peak bytes/term (ranks summed) | 62.77 B/term | 58.15 B/term | 0.926 | 3/3 | 0.250 |

## Compact summary

| row | arm | time ratio | RSS ratio (summed) | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| B_8x16 hubbard fresh | k2 | 1.480 | 0.926 | 0.725 |

