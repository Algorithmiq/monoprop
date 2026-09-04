# A/B summary: ab-hashfree-1870846

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
node	cnx002.deucalion.macc.fccn.pt
job	1870846
```

cells: 4 loaded, 0 skipped (failed cells, see stderr)

## C_1x128 hubbard fresh  (2 paired reps)

!! rep 1: term counts differ main=(1001661534,) drop=(1020270081,)
!! rep 2: term counts differ main=(1001661534,) drop=(1020270081,)
terms: (1020270081,)

| metric | main median | drop median | drop/main (paired median) | agree | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| time test_model_propagate[hubbard] | 147.347 s | 273.743 s | 1.858 | 2/2 | 0.500 |
| peak RSS (kernel) | 56.497 GiB | 53.182 GiB | 0.941 | 2/2 | 0.500 |
| operator ledger total | 48.762 GiB | 40.064 GiB | 0.822 | 2/2 | 0.500 |
| ledger operator_terms_bytes | 10903.1 MiB | 12323.0 MiB | 1.130 | 2/2 | 0.500 |
| ledger indexing_bytes | 16384.0 MiB | 0.0 MiB | 0.000 | 2/2 | 0.500 |
| ledger inverted_index_bytes | 12201.2 MiB | 15240.3 MiB | 1.249 | 2/2 | 0.500 |
| ledger op_coeffs_bytes | 7642.1 MiB | 7784.0 MiB | 1.019 | 2/2 | 0.500 |
| ledger gate_scratch_bytes | — | 1196.2 MiB | — | — | — |
| ledger matched_scratch_bytes | 2801.9 MiB | — | — | — | — |
| ledger scratch (matched|gate) | 2801.9 MiB | 1196.2 MiB | 0.427 | 2/2 | 0.500 |
| ledger bytes/term | 51.32 B/term | 42.16 B/term | 0.822 | 2/2 | 0.500 |
| kernel peak bytes/term | 59.46 B/term | 55.97 B/term | 0.941 | 2/2 | 0.500 |

## Compact summary

| row | arm | time ratio | RSS ratio | bytes/term ratio |
| --- | --- | ---: | ---: | ---: |
| C_1x128 hubbard fresh | drop | 1.858 | 0.941 | 0.822 |

