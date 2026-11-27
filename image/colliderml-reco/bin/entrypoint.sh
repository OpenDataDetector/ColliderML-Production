#!/bin/bash
# Container entrypoint: load the reconstruction environment, then run the given command.
# With no command it prints the colliderml-reco help.

# ROOT and Gaudi write small cache files to $HOME; an arbitrary --user has none.
if [[ ! -w "${HOME:-/nonexistent}" ]]; then
  export HOME=/tmp
fi

source /opt/colliderml-reco/env/setup.sh

if [[ $# -eq 0 ]]; then
  exec colliderml-reco --help
fi
exec "$@"
