import argparse
import logging
from pathlib import Path
import sys
from tqdm import tqdm
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_hepmc3_file(hepmc_path: Path) -> list:
    """
    Validate a HepMC3 file by checking basic structure.
    
    Args:
        hepmc_path: Path to the HepMC3 file
        
    Returns:
        List of issues found (empty if valid)
    """
    issues = []
    
    # Check file is not empty
    if hepmc_path.stat().st_size == 0:
        issues.append(f"File is empty: {hepmc_path}")
        return issues
    
    try:
        with open(hepmc_path, 'r') as f:
            # Read first few lines to check HepMC3 header
            first_lines = []
            for i, line in enumerate(f):
                if i >= 10:  # Only check first 10 lines
                    break
                first_lines.append(line)
            
            # Check for HepMC3 format markers
            header_found = any("HepMC" in line for line in first_lines)
            if not header_found:
                issues.append(f"No HepMC header found in {hepmc_path}")
            
            # Check for event records (should have 'E' lines for events)
            has_events = any(line.startswith('E ') for line in first_lines)
            if not has_events:
                issues.append(f"No event records found in {hepmc_path}")
                
    except Exception as e:
        issues.append(f"Failed to read {hepmc_path}: {e}")
    
    return issues


def validate_run(run_dir: Path) -> tuple[list, int]:
    """
    Validate a single run directory from pythia_generation stage.
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Tuple of (list of issues found, file size in bytes)
    """
    issues = []
    file_size = 0
    
    # Check for merged_events.hepmc3 (primary output from pythia_generation)
    merged_file = run_dir / "merged_events.hepmc3"
    if not merged_file.exists():
        issues.append(f"Missing merged_events.hepmc3 in {run_dir}")
        return issues, file_size
    
    # Get file size
    file_size = merged_file.stat().st_size
    
    # Validate the HepMC3 file structure
    hepmc_issues = validate_hepmc3_file(merged_file)
    issues.extend(hepmc_issues)
    
    # Check for timing summary (optional but good practice)
    timing_file = run_dir / "timing_summary.txt"
    if timing_file.exists():
        logger.debug(f"Found timing_summary.txt in {run_dir}")
    
    return issues, file_size


def main():
    parser = argparse.ArgumentParser(description="Validate pythia_generation stage outputs")
    parser.add_argument("--stage", required=False, help="Pipeline stage being validated")
    parser.add_argument("--runs-dir", help="Directory containing run subdirectories")
    # Legacy args for backwards compatibility
    parser.add_argument("--run-dir", help="Legacy: Directory containing run outputs")
    parser.add_argument("--node-idx", type=int, help="Legacy: Node index being validated")
    parser.add_argument("--runs-per-node", type=int, help="Legacy: Number of runs per node")
    
    args = parser.parse_args()

    # Determine runs directory
    runs_dir = None
    if args.runs_dir:
        runs_dir = Path(args.runs_dir)
    elif args.run_dir:
        runs_dir = Path(args.run_dir)
    else:
        logger.error("Must provide --runs-dir (or legacy --run-dir)")
        sys.exit(2)

    if not runs_dir.is_dir():
        logger.error(f"Runs directory not found: {runs_dir}")
        sys.exit(2)

    logger.info(f"Validating pythia_generation; scanning runs in: {runs_dir}")

    # Find all numeric run directories
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    run_dirs = sorted(run_dirs, key=lambda p: int(p.name))
    
    if not run_dirs:
        logger.error(f"No run directories found in {runs_dir}")
        sys.exit(3)

    logger.info(f"Found {len(run_dirs)} run directories to validate")

    # Validate each run and collect file sizes
    issues_total = []
    runs_with_issues = []
    file_sizes = {}
    
    for run_dir in tqdm(run_dirs, desc="Validating runs", unit="run"):
        issues, file_size = validate_run(run_dir)
        file_sizes[run_dir.name] = file_size
        
        if issues:
            for issue in issues:
                logger.warning(issue)
            runs_with_issues.append(run_dir.name)
            issues_total.extend(issues)
        else:
            logger.debug(f"Run {run_dir.name}: OK")

    # Check file size consistency
    valid_sizes = [size for run_name, size in file_sizes.items() if size > 0 and run_name not in runs_with_issues]
    
    if len(valid_sizes) > 0:
        median_size = statistics.median(valid_sizes)
        threshold = 0.8 * median_size
        
        logger.info(f"File size statistics:")
        logger.info(f"  Median size: {median_size / (1024**2):.2f} MB")
        logger.info(f"  Threshold (80% of median): {threshold / (1024**2):.2f} MB")
        
        size_outliers = []
        for run_name, size in file_sizes.items():
            if size > 0 and size < threshold:
                size_outliers.append(run_name)
                issue = f"Run {run_name}: File size {size / (1024**2):.2f} MB is below 80% of median ({median_size / (1024**2):.2f} MB)"
                logger.warning(issue)
                if run_name not in runs_with_issues:
                    runs_with_issues.append(run_name)
                    issues_total.append(issue)
        
        if size_outliers:
            logger.warning(f"Found {len(size_outliers)} runs with undersized files")

    # Summary
    logger.info(f"Validated {len(run_dirs)} runs")
    logger.info(f"Successful: {len(run_dirs) - len(runs_with_issues)}")
    logger.info(f"Failed: {len(runs_with_issues)}")
    
    if issues_total:
        bad_sorted = sorted(runs_with_issues, key=int)
        logger.error(f"Runs with issues ({len(bad_sorted)}): {' '.join(bad_sorted)}")
        logger.error(f"Validation FAILED with {len(issues_total)} issue(s)")
        sys.exit(1)
    else:
        logger.info("Validation PASSED - all runs have valid outputs")
        sys.exit(0)


if __name__ == "__main__":
    main()

