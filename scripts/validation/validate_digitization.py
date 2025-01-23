import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", help="Pipeline stage being validated")
    parser.add_argument("--stage-dir", help="Directory containing stage outputs")
    parser.add_argument("--job-id", help="Job ID being validated")
    args = parser.parse_args()
    
    logger.info(f"Hello from validation script!")
    logger.info(f"Validating stage: {args.stage}")
    logger.info(f"Stage directory: {args.stage_dir}")
    logger.info(f"Job ID: {args.job_id}")

if __name__ == "__main__":
    main() 