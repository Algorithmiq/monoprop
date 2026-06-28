#!/usr/bin/env bash

set -euxo pipefail

echo "Set up development environment in $WORKSPACE"

# install project
# NOTE docs dependency group excluded, as it requires Python 3.12 and we develop against 3.11
uv sync --group dev --group test --group interactive --all-extras -v

# install prek hooks
uv run prek install --install-hooks
