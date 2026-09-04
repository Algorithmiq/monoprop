# A/B summary: ab-hashfree-nodes-1871480

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
layout	C_2n_8x16
rung	L3
nodes	2
ranks	16
ranks_per_node	8
cores_per_rank	16
partitions_per_rank	16
reps	3
smoke	0
rss_source	bench-json memhwm (ranks' VmHWM summed)
partition	dev-x86
nodelist	cnx[007-008]
job	1871480
```

cells: 6 loaded, 0 skipped (failed cells, see stderr)

## C_2n_8x16 hubbard fresh  (3 paired reps)

terms: (1001661534,)

| metric | base median | k2 median | k2/base (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 62.978 s | 95.695 s | 1.518 | 3/3 | 0.250 |
| peak RSS (ranks summed) | 64.661 GiB | 57.414 GiB | 0.887 | 3/3 | 0.250 |
| operator ledger total | 51.808 GiB | 38.734 GiB | 0.748 | 3/3 | 0.250 |
| ledger operator_terms_bytes | 13891.9 MiB | 13680.9 MiB | 0.985 | 3/3 | 0.250 |
| ledger indexing_bytes | 16384.0 MiB | 0.0 MiB | 0.000 | 3/3 | 0.250 |
| ledger inverted_index_bytes | 12270.1 MiB | 12150.4 MiB | 0.990 | 3/3 | 0.250 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 3/3 | 0.250 |
| ledger row_keys_bytes | — | 4974.9 MiB | — | — | — |
| ledger gate_scratch_bytes | — | 1215.3 MiB | — | — | — |
| ledger matched_scratch_bytes | 2863.2 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 2863.2 MiB | 1215.3 MiB | 0.424 | 3/3 | 0.250 |
| ledger bytes/term | 55.54 B/term | 41.52 B/term | 0.748 | 3/3 | 0.250 |
| peak bytes/term (ranks summed) | 69.31 B/term | 61.55 B/term | 0.887 | 3/3 | 0.250 |

## Compact summary

| row | arm | time ratio | RSS ratio (summed) | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| C_2n_8x16 hubbard fresh | k2 | 1.518 | 0.887 | 0.748 |

