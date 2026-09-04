# A/B summary: ab-hashfree-nodes-1873589

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
nodelist	cnx005
job	1873589
```

cells: 6 loaded, 0 skipped (failed cells, see stderr)

## B_8x16 hubbard fresh  (3 paired reps)

terms: (1001661534,)

| metric | ref median | st median | st/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 182.737 s | 181.517 s | 0.993 | 2/3 | 1.000 |
| peak RSS (ranks summed) | 53.909 GiB | 48.253 GiB | 0.895 | 3/3 | 0.250 |
| operator ledger total | 35.375 GiB | 32.560 GiB | 0.920 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 11165.1 MiB | 10560.0 MiB | 0.946 | 3/3 | 0.250 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.875 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 12142.1 MiB | 10085.0 MiB | 0.831 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | 4060.1 MiB | 3840.0 MiB | 0.946 | 3/3 | 0.250 |
| ledger gate_scratch_bytes | 1213.7 MiB | 1213.7 MiB | 1.000 | 3/3 | 0.250 |
| ledger scratch (matched|gate) | 1213.7 MiB | 1213.7 MiB | 1.000 | 3/3 | 0.250 |
| ledger bytes/term | 37.92 B/term | 34.90 B/term | 0.920 | 3/3 | 0.250 |
| peak bytes/term (ranks summed) | 57.79 B/term | 51.73 B/term | 0.895 | 3/3 | 0.250 |

## Compact summary

| row | arm | time ratio | RSS ratio (summed) | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| B_8x16 hubbard fresh | st | 0.993 | 0.895 | 0.920 |

