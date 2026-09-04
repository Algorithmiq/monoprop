# A/B summary: ab-hashfree-nodes-smoke-1870857

```
key	value
arm_1_name	main
arm_1_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/base-296/.venv
arm_1_md5	f991ce3a65daa476f1229b26643f92ce
arm_1_rev	5ada3da3
arm_1_env	
arm_2_name	port
arm_2_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/one-round/.venv
arm_2_md5	4dbb818c8ea90332442d0c023a1767f4
arm_2_rev	4da2f2de
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
reps	1
smoke	1
rss_source	bench-json memhwm (ranks' VmHWM summed)
partition	dev-x86
nodelist	cnx003
job	1870857
```

cells: 4 loaded, 0 skipped (failed cells, see stderr)

## B_8x16 hubbard fresh  (1 paired reps)

terms: (19532,)

| metric | main median | port median | port/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 0.439 s | 0.157 s | 0.359 | 1/1 | 1.000 |
| peak RSS (ranks summed) | 1.664 GiB | 1.689 GiB | 1.015 | 1/1 | 1.000 |
| operator ledger total | 0.001 GiB | 0.001 GiB | 1.142 | 1/1 | 1.000 |
| ledger operator_terms_bytes | 0.2 MiB | 0.2 MiB | 1.001 | 1/1 | 1.000 |
| ledger indexing_bytes | 0.3 MiB | 0.0 MiB | 0.058 | 1/1 | 1.000 |
| ledger inverted_index_bytes | 0.2 MiB | 0.2 MiB | 1.004 | 1/1 | 1.000 |
| ledger op_coeffs_bytes | 0.1 MiB | 0.1 MiB | 1.000 | 1/1 | 1.000 |
| ledger gate_scratch_bytes | — | 0.3 MiB | — | — | — |
| ledger matched_scratch_bytes | 0.1 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 0.1 MiB | 0.3 MiB | 5.810 | 1/1 | 1.000 |
| ledger bytes/term | 47.81 B/term | 54.60 B/term | 1.142 | 1/1 | 1.000 |
| peak bytes/term (ranks summed) | 91493.34 B/term | 92873.63 B/term | 1.015 | 1/1 | 1.000 |

## B_8x16 pauli fresh  (1 paired reps)

terms: (588,)

| metric | main median | port median | port/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[pauli] | 0.195 s | 4.457 s | 22.891 | 1/1 | 1.000 |
| peak RSS (ranks summed) | 1.682 GiB | 1.697 GiB | 1.009 | 1/1 | 1.000 |
| operator ledger total | 0.000 GiB | 0.000 GiB | 2.185 | 1/1 | 1.000 |
| ledger operator_terms_bytes | 0.0 MiB | 0.0 MiB | 1.014 | 1/1 | 1.000 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 0.471 | 1/1 | 1.000 |
| ledger inverted_index_bytes | 0.0 MiB | 0.0 MiB | 1.074 | 1/1 | 1.000 |
| ledger op_coeffs_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 1/1 | 1.000 |
| ledger gate_scratch_bytes | — | 0.1 MiB | — | — | — |
| ledger matched_scratch_bytes | 0.0 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 0.0 MiB | 0.1 MiB | 66.085 | 1/1 | 1.000 |
| ledger bytes/term | 124.41 B/term | 271.87 B/term | 2.185 | 1/1 | 1.000 |
| peak bytes/term (ranks summed) | 3071853.71 B/term | 3098916.57 B/term | 1.009 | 1/1 | 1.000 |

## Compact summary

| row | arm | time ratio | RSS ratio (summed) | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| B_8x16 hubbard fresh | port | 0.359 | 1.015 | 1.142 |
| B_8x16 pauli fresh | port | 22.891 | 1.009 | 2.185 |

