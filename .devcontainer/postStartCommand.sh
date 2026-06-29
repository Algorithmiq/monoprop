#!/usr/bin/env bash

set -euxo pipefail

echo "Set up development environment in $WORKSPACE"

# install project
# NOTE the docs dependency group is excluded by default (it requires Python 3.12
# and we develop against 3.11); it is built separately via `just build-docs`.
# dev/test/interactive are the default groups (see [tool.uv] default-groups).
uv sync --all-extras -v

# install prek hooks
uv run prek install --install-hooks
