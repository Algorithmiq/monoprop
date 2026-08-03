#!/usr/bin/env bash

set -euxo pipefail

echo "Set up development environment in $WORKSPACE"

# install project
uv sync --all-groups --all-extras -v

# install prek hooks
uv run prek install --install-hooks --overwrite
