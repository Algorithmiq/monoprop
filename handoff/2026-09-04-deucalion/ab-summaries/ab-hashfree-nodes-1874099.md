# A/B summary: ab-hashfree-nodes-1874099

```
key	value
arm_1_name	ref
arm_1_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/ref-ce16f0f6/.venv
arm_1_md5	97f6d9185da0796a0ec2e9b1e12d102e
arm_1_rev	ce16f0f6
arm_1_env	
arm_2_name	wire
arm_2_venv	/projects/EEHPC-DEV-2026D08-260/aaron/worktrees/wire/.venv
arm_2_md5	ef0334654d16f2ba7daf44a6bc74e59d
arm_2_rev	29d225ec
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
reps	5
smoke	0
rss_source	bench-json memhwm (ranks' VmHWM summed)
partition	dev-x86
nodelist	cnx002
job	1874099
```

cells: 20 loaded, 0 skipped (failed cells, see stderr)

## B_8x16 hubbard fresh  (5 paired reps)

terms: (1001661534,)

| metric | ref median | wire median | wire/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 186.153 s | 186.340 s | 1.001 | 3/5 | 1.000 |
| peak RSS (ranks summed) | 48.575 GiB | 49.187 GiB | 1.014 | 5/5 | 0.062 |
| operator ledger total | 32.560 GiB | 32.562 GiB | 1.000 | 5/5 | 0.062 |
| ledger operator_terms_bytes | 10560.0 MiB | 10560.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger inverted_index_bytes | 10085.0 MiB | 10085.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7642.1 MiB | 1.000 | 5/5 | 0.062 |
| ledger row_keys_bytes | 3840.0 MiB | 3840.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger gate_scratch_bytes | 1213.7 MiB | 1215.7 MiB | 1.002 | 5/5 | 0.062 |
| ledger scratch (matched|gate) | 1213.7 MiB | 1215.7 MiB | 1.002 | 5/5 | 0.062 |
| ledger bytes/term | 34.90 B/term | 34.90 B/term | 1.000 | 5/5 | 0.062 |
| peak bytes/term (ranks summed) | 52.07 B/term | 52.73 B/term | 1.014 | 5/5 | 0.062 |

## B_8x16 pauli fresh  (5 paired reps)

terms: (985970588,)

| metric | ref median | wire median | wire/ref (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[pauli] | 117.259 s | 117.350 s | 1.000 | 3/5 | 1.000 |
| peak RSS (ranks summed) | 73.656 GiB | 74.344 GiB | 1.009 | 5/5 | 0.062 |
| operator ledger total | 53.550 GiB | 53.551 GiB | 1.000 | 5/5 | 0.062 |
| ledger operator_terms_bytes | 27840.0 MiB | 27840.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger indexing_bytes | 0.0 MiB | 0.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger inverted_index_bytes | 13801.1 MiB | 13801.1 MiB | 1.000 | 5/5 | 0.062 |
| ledger op_coeffs_bytes | 7522.4 MiB | 7522.4 MiB | 1.000 | 5/5 | 0.062 |
| ledger row_keys_bytes | 3840.0 MiB | 3840.0 MiB | 1.000 | 5/5 | 0.062 |
| ledger gate_scratch_bytes | 1831.1 MiB | 1832.1 MiB | 1.001 | 5/5 | 0.062 |
| ledger scratch (matched|gate) | 1831.1 MiB | 1832.1 MiB | 1.001 | 5/5 | 0.062 |
| ledger bytes/term | 58.32 B/term | 58.32 B/term | 1.000 | 5/5 | 0.062 |
| peak bytes/term (ranks summed) | 80.21 B/term | 80.96 B/term | 1.009 | 5/5 | 0.062 |

## Compact summary

| row | arm | time ratio | RSS ratio (summed) | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| B_8x16 hubbard fresh | wire | 1.001 | 1.014 | 1.000 |
| B_8x16 pauli fresh | wire | 1.000 | 1.009 | 1.000 |

