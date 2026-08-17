# Measured code that exists in no commit

Both patches below were **built, gated and measured**, and both lived only as uncommitted
modifications in a `/projects` worktree until they were saved here. A `git clean` or `git checkout`
in that worktree would have destroyed them and, with them, the reproducibility of every number in
`RESULTS-invidx-arena-segments.md` §4-§6. They are kept here because this branch is the measurement
record; nothing here ships in a PR.

The branch names in this area actively mislead: `perf/invidx-arena-segments` contains **no** arena
segments (it is `24bef81`, the a1 index commit, exactly), and `perf/invidx-a1f-a2` contains **no**
a1f (it is `24bef81 + eff6898`). Read the patch, not the branch name.

## `invidx-arena-segments.patch`

Apply to `24bef81` (`perf/invidx-arena-segments` tip). Also applies, modulo hunk offsets, to
`eff6898` and to `a4fea7c` — the copies in `mp-a1f`, `mp-s2` and `mp-s2p` were the same change; the
latter two were byte-identical to each other.

    cpp/monoprop/detail/operator/InvertedIndex.h   +79/-56
    cpp/tests/inverted_index_tests.cpp

What it fixes: `open_segment_`'s whole-arena `reserve` never fired on the incremental `append_rows`
path, so a monolithic `std::vector` grew on a 1/8 floor and paid a ~920 MB copy at the top. It
inverts a1's peak-RSS regression on Pauli from **+7.0 to -4.5 B/term** at R=8 (jobs 1826752 /
1826756) and improves Hubbard from -5.13 to -6.60.

Gate that passed on the built arm (`_core.so` md5 `ea724dc4515f4138f363d8d0408729cc`, job
`mp-s2-check-1826785`): 212/212 `ctest -L unit`, 211/211 `ctest -L serial`.

## `invidx-bitmap-premium.patch`

Apply to `050eea9` (`perf/layer-profile-branch` tip). Adds `monoprop_INVIDX_BITMAP_PREMIUM`
(sixteenths, default 16) plus profiling hunks in `EnvConfig.h`, `LayerProfile.h`, `HybridComm.h`,
`InvertedIndex.h`.

**This is an instrument, not a candidate.** Its sweep has already closed it: free at R=1 (0.9881x,
4/4) and refuted at production R=8 (1.0047x, 3/4) — a threshold tuned at one `P` does not transfer,
and 16 is the pure-bytes argmin, i.e. no knob at all. The `p016 -> pinf` sweep (job 1826877) removed
99.6% of delta bytes to recover 1.8% of an 11.3% regression. It is kept only so the sweep can be
re-run; it must never ship.

Note it inherits `parse_positive_int`'s silent-default behaviour: values above 1e6 fall back to the
default, so an arm can run the shipped rule while reporting the experimental one. Prove a counter
moved before believing an arm.
