#!/usr/bin/env python3
# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diff two environment files and append changed variables to GITHUB_ENV."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_env(path: Path) -> dict[str, str]:
    """Parse an environment file into a mapping of variable names to values."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
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
