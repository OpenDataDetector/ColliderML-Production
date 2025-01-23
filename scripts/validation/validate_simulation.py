import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_simulation(run_dir, node_idx, runs_per_node):
    """
    Validation steps for simulation stage:
    - Check that the expected run directories exist
    - Check that each run has required HepMC3 files
    - Check that the events are valid using HepMC3 (TODO)
    """
    start_run = node_idx * runs_per_node
    end_run = start_run + runs_per_node
    
    logger.info(f"Validating runs {start_run} through {end_run-1}")
    
    # Check each expected run directory
    for run_id in range(start_run, end_run):
        run_path = Path(run_dir) / f"{run_id}"
        
        if not run_path.exists():
            logger.error(f"Missing run directory: {run_path}")
            continue
            
        # Check for required files
        required_files = ["edm4hep.root"]
        for file in required_files:
            file_path = run_path / file
            if not file_path.exists():
                logger.error(f"Missing required file {file} in {run_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", help="Pipeline stage being validated")
    parser.add_argument("--run-dir", help="Directory containing run outputs")
    parser.add_argument("--node-idx", type=int, help="Node index being validated")
    parser.add_argument("--runs-per-node", type=int, help="Number of runs per node")
    
    args = parser.parse_args()
    
    logger.info(f"Validating simulation stage for node {args.node_idx}")
    logger.info(f"Run directory: {args.run_dir}")
    logger.info(f"Runs per node: {args.runs_per_node}")

    validate_simulation(args.run_dir, args.node_idx, args.runs_per_node)

if __name__ == "__main__":
    main() 