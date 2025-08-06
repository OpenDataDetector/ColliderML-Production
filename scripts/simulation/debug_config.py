#!/usr/bin/env python3

import sys
sys.path.append('../cli')
from utils.config import create_base_parser, load_config
import yaml

def main():
    # Load the config directly
    config_path = "/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software/colliderml_dev/configs_production/full_pileup_pilot/ttbar/madgraph_init_config.yaml"
    
    print("=== Loading config directly with yaml ===")
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    print("Card customizations:")
    print(raw_config.get('card_customizations', {}))
    print()
    print("Shower card section:")
    shower_card = raw_config.get('card_customizations', {}).get('shower_card', {})
    print(shower_card)
    print()
    print("time_shower_me_corrections value:")
    print(repr(shower_card.get('time_shower_me_corrections')))
    
    print("\n=== Loading with load_config function ===")
    # Simulate argparse
    class Args:
        config = config_path
        output = None
        log_level = None
    
    args = Args()
    config = load_config(args)
    
    print("Config object type:", type(config))
    print("Has card_customizations:", hasattr(config, 'card_customizations'))
    if hasattr(config, 'card_customizations'):
        print("card_customizations:", config.card_customizations)
        shower_config = config.card_customizations.get('shower_card', {})
        print("shower_card config:", shower_config)
        print("time_shower_me_corrections in shower_card:", 'time_shower_me_corrections' in shower_config)

if __name__ == "__main__":
    main()