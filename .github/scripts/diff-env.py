#!/usr/bin/env python3
"""Diff two environment files and append changed variables to GITHUB_ENV."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("before_path", type=Path)
    parser.add_argument("after_path", type=Path)
    args = parser.parse_args()

    before = parse_env(args.before_path)
    after = parse_env(args.after_path)

    changed = {k: v for k, v in after.items() if before.get(k) != v}

    github_env = Path(os.environ["GITHUB_ENV"])

    with github_env.open("a", encoding="utf-8") as out:
        out.writelines(
            f"{k}<<__EOF__\n{v}\n__EOF__\n" for k, v in sorted(changed.items())
        )
