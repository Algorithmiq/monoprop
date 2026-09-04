# A/B summary: ab-hashfree-nodes-1874191

```
key	value
arm_1_name	base
arm_1_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/base-296/.venv
arm_1_md5	f991ce3a65daa476f1229b26643f92ce
arm_1_rev	5ada3da3
arm_1_env	
arm_2_name	s3
arm_2_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/ref-fd795841/.venv
arm_2_md5	457429b288c9ae8616f01e297046fcd0
arm_2_rev	fd795841
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
nodelist	cnx002
job	1874191
```

cells: 6 loaded, 0 skipped (failed cells, see stderr)

## B_8x16 hubbard fresh  (3 paired reps)

terms: (1001661534,)

| metric | base median | s3 median | s3/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 126.378 s | 180.681 s | 1.430 | 3/3 | 0.250 |
| peak RSS (ranks summed) | 58.494 GiB | 53.786 GiB | 0.920 | 3/3 | 0.250 |
| operator ledger total | 48.764 GiB | 35.382 GiB | 0.726 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 10860.4 MiB | 11165.1 MiB | 1.028 | 3/3 | 0.250 |
| ledger indexing_bytes | 16384.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 12202.0 MiB | 12142.1 MiB | 0.995 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 4060.1 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 1221.7 MiB | — | — | — |
| ledger matched_scratch_bytes | 2845.6 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 2845.6 MiB | 1221.7 MiB | 0.429 | 3/3 | 0.250 |
| ledger bytes/term | 52.27 B/term | 37.93 B/term | 0.726 | 3/3 | 0.250 |
| peak bytes/term (ranks summed) | 62.70 B/term | 57.66 B/term | 0.920 | 3/3 | 0.250 |

## Compact summary

| row | arm | time ratio | RSS ratio (summed) | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| B_8x16 hubbard fresh | s3 | 1.430 | 0.920 | 0.726 |

