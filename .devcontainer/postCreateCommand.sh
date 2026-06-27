#!/usr/bin/env bash

set -euxo pipefail

# install uv and uvx shell completions
echo 'eval "$(uv generate-shell-completion zsh)"' >> /home/vscode/.zshrc
echo 'eval "$(uvx --generate-shell-completion zsh)"' >> /home/vscode/.zshrc

# just completions
mkdir -p ~/.zsh/completions
just --completions zsh > ~/.zsh/completions/_just
cat <<EOT >> /home/vscode/.zshrc
fpath=(~/.zsh/completions \$fpath)
autoload -U compinit
compinit
EOT

# set up GDB to view contents of STL containers
cat <<EOT >> /home/vscode/.gdbinit
python
import sys
sys.path.insert(0, "/usr/share/gcc/python")
from libstdcxx.v6.printers import register_libstdcxx_printers
register_libstdcxx_printers (None)
end
EOT
