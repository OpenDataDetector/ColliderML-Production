#!/bin/bash
# Record the base image's key4hep environment once, so that containers start without
# calling spack (several seconds per start, and spack writes to the caller's home directory).
source /opt/colliderml-reco/steps/common.sh

set +u
source /opt/setup_spack.sh
spack load --sh key4hep-stack > "$RECO_HOME/env/key4hep-stack.sh"
set -u

test -s "$RECO_HOME/env/key4hep-stack.sh"
load_key4hep
for tool in k4run ddsim cmake ninja; do
  command -v "$tool" >/dev/null
done
echo "key4hep environment recorded: $(wc -l < "$RECO_HOME/env/key4hep-stack.sh") lines"
