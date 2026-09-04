# A/B summary: ab-hashfree-nodes-1870859

```
key	value
arm_1_name	main
arm_1_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/base-296/.venv
arm_1_md5	f991ce3a65daa476f1229b26643f92ce
arm_1_rev	5ada3da3
arm_1_env	
arm_2_name	drop
arm_2_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/one-round/.venv
arm_2_md5	4dbb818c8ea90332442d0c023a1767f4
arm_2_rev	4da2f2de
arm_2_env	monoprop_DROP_SILENT_RECORDS=1
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
nodelist	cnx003
job	1870859
```

cells: 6 loaded, 0 skipped (failed cells, see stderr)

## B_8x16 hubbard fresh  (3 paired reps)

!! rep 1: term counts differ main=(1001661534,) drop=(1020270081,)
!! rep 2: term counts differ main=(1001661534,) drop=(1020270081,)
!! rep 3: term counts differ main=(1001661534,) drop=(1020270081,)
terms: (1020270081,)

| metric | main median | drop median | drop/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 130.349 s | 245.239 s | 1.881 | 3/3 | 0.250 |
| peak RSS (ranks summed) | 58.501 GiB | 54.915 GiB | 0.940 | 3/3 | 0.250 |
| operator ledger total | 48.764 GiB | 39.918 GiB | 0.819 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 10860.4 MiB | 12168.8 MiB | 1.120 | 3/3 | 0.250 |
| ledger indexing_bytes | 16384.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 12202.0 MiB | 15294.5 MiB | 1.253 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7784.0 MiB | 1.019 | 3/3 | 0.250 |
| ledger gate_scratch_bytes | — | 1203.3 MiB | — | — | — |
| ledger matched_scratch_bytes | 2845.6 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 2845.6 MiB | 1203.3 MiB | 0.423 | 3/3 | 0.250 |
| ledger bytes/term | 51.32 B/term | 42.01 B/term | 0.819 | 3/3 | 0.250 |
| peak bytes/term (ranks summed) | 61.57 B/term | 57.79 B/term | 0.940 | 3/3 | 0.250 |

## Compact summary

| row | arm | time ratio | RSS ratio (summed) | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| B_8x16 hubbard fresh | drop | 1.881 | 0.940 | 0.819 |

