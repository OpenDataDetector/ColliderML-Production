#!/bin/bash
set -x
conda run -n collider-env python plot_calo_performance.py calo_performance_config.yaml
