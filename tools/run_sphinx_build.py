#!/usr/bin/env python

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

"""Run sphinx-build and replay MyST-NB error logs to stderr on failure.

The purpose of this wrapper is to make it easier to debug MyST-NB notebook execution
failures during documentation builds. By default, MyST-NB writes execution error logs to
separate files in a reports directory. This wrapper captures the state of those logs
before and after running sphinx-build, and replays any new or updated logs to stderr.
This allows developers to see notebook execution errors directly in the build output,
without having to manually find and open the log files.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def find_error_logs(reports_dir: Path) -> list[Path]:
    """Return notebook execution error logs ordered from newest to oldest."""
    if not reports_dir.exists():
        return []
    return sorted(
        reports_dir.rglob("*.err.log"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )


def snapshot_logs(reports_dir: Path) -> dict[Path, int]:
    """Capture the current modification times for known error logs."""
    return {
        path.resolve(): path.stat().st_mtime_ns for path in find_error_logs(reports_dir)
    }


def extract_notebook_exception(log_content: str) -> str | None:
    """Return the notebook path and cell-level error from a sphinx exception log.

    Strips out the sphinx framework frames so only the failing cell code and
    the kernel traceback are shown. Returns None if the log does not contain a
    notebook execution error.
    """
    notebook_match = re.search(r"ExecutionError: (.+\.ipynb)", log_content)
    # Match from CellExecutionError up to (but not including) the outer sphinx
    # traceback that starts with "The above exception was the direct cause".
    cell_match = re.search(
        r"([ \t]*\S*CellExecutionError: An error occurred while executing the following cell:\n.*?)"
        r"(?=\n[ \t]*The above exception was the direct cause|\Z)",
        log_content,
        re.DOTALL,
    )
    if not notebook_match and not cell_match:
        return None

    parts: list[str] = []
    if notebook_match:
        parts.append(f"Notebook: {notebook_match.group(1).strip()}")
    if cell_match:
        cleaned = _ANSI_ESCAPE.sub("", cell_match.group(1))
        cleaned = textwrap.dedent(cleaned).strip()
        parts.append("")
        parts.append(cleaned)
    return "\n".join(parts)


def find_sphinx_exception_logs(created_after: float) -> list[Path]:
    """Return sphinx exception logs written to /tmp after the given timestamp."""
    tmp_dir = Path("/tmp")  # noqa: S108
    logs: list[tuple[float, Path]] = []
    for path in tmp_dir.glob("sphinx-err-*.log"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= created_after:
            logs.append((mtime, path))
    logs.sort(key=lambda item: item[0])
    return [path for _, path in logs]


def print_error_logs(error_logs: list[Path]) -> None:
    """Write notebook execution error logs to stderr."""
    if not error_logs:
        return

    sys.stderr.write("\nNotebook execution logs:\n")
    for error_log in error_logs:
        sys.stderr.write(f"\n=== {error_log} ===\n")
        try:
            contents = error_log.read_text()
        except OSError as exc:
            sys.stderr.write(f"Unable to read notebook error log: {exc}\n")
            continue

        sys.stderr.write(contents)
        if not contents.endswith("\n"):
            sys.stderr.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Execute the wrapped command and replay notebook failures on stderr."""
    parser = argparse.ArgumentParser(
        description="Run sphinx-build and replay MyST-NB error logs to stderr on failure."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="Directory containing MyST-NB execution reports.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to execute. Prefix with '--' to separate wrapper arguments.",
    )
    args = parser.parse_args(argv)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a sphinx-build command is required")

    start_time = time.time()
    before = snapshot_logs(args.reports_dir)
    completed = subprocess.run(command, check=False)  # noqa: S603
    if completed.returncode == 0:
        return 0

    after = find_error_logs(args.reports_dir)
    changed_logs = [
        path
        for path in after
        if path.resolve() not in before
        or path.stat().st_mtime_ns > before[path.resolve()]
    ]
    print_error_logs(changed_logs or after)

    sphinx_logs = find_sphinx_exception_logs(start_time)
    if sphinx_logs:
        sys.stderr.write("\nSphinx exception logs:\n")
        for log in sphinx_logs:
            sys.stderr.write(f"\n=== {log} ===\n")
            try:
                contents = log.read_text()
            except OSError as exc:
                sys.stderr.write(f"Unable to read sphinx exception log: {exc}\n")
                continue
            extracted = extract_notebook_exception(contents)
            output = extracted if extracted is not None else contents
            sys.stderr.write(output)
            if not output.endswith("\n"):
                sys.stderr.write("\n")

    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
