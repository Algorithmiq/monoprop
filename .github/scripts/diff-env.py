#!/usr/bin/env python3
"""Diff two environment files and append changed variables to GITHUB_ENV."""

from __future__ import annotations

import os
from pathlib import Path

before_path = Path(os.environ["BEFORE_ENV"])
after_path = Path(os.environ["AFTER_ENV"])
github_env = Path(os.environ["GITHUB_ENV"])


def parse_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k] = v
    return out


before = parse_env(before_path)
after = parse_env(after_path)

changed = {k: v for k, v in after.items() if before.get(k) != v}

with github_env.open("a", encoding="utf-8") as out:
    out.writelines(f"{k}<<__EOF__\n{v}\n__EOF__\n" for k, v in sorted(changed.items()))
