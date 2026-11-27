# Runtime environment for reconstruction: key4hep, the Pandora stack, ODD and k4ODD.
# Usage: source /opt/colliderml-reco/env/setup.sh

# The environment scripts of the base image read unset variables.
case $- in
  *u*) colliderml_reco_nounset=1; set +u ;;
  *) colliderml_reco_nounset=0 ;;
esac

source /opt/key4hep-shim.sh
source /opt/pandora-stack/install/setup_stack.sh
source /opt/odd-install/bin/this_odd.sh

export K4ODD_PATH=/opt/k4ODD
export ODD_INSTALL_DIR=/opt/odd-install
export LD_LIBRARY_PATH="/opt/k4ODD/install/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="/opt/k4ODD/install/python:/opt/reco-pylibs${PYTHONPATH:+:$PYTHONPATH}"
export PATH="/opt/colliderml-reco/bin:${PATH}"

if [[ "$colliderml_reco_nounset" == 1 ]]; then
  set -u
fi
unset colliderml_reco_nounset
