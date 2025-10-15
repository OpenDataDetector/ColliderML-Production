#!/bin/bash
# Example commands for running calorimeter performance plots
# for different particle types

# Electrons (PDG = 11)
echo "Generating electron plots..."
python plot_calo_performance.py calo_performance_config.yaml

# To run for photons, create a photon config or modify PDG code inline:
# Photons (PDG = 22)
# python plot_calo_performance.py calo_performance_config_photon.yaml

# To run for pions, create a pion config:
# Pions (PDG = 211)
# python plot_calo_performance.py calo_performance_config_pion.yaml

echo "Done!"

