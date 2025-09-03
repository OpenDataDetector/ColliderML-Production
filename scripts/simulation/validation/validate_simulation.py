import argparse
import logging
from pathlib import Path
import sys

import uproot
import awkward as ak
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_run(run_dir: Path) -> list:
    issues = []
    edm_file = run_dir / "edm4hep.root"
    if not edm_file.exists():
        issues.append(f"Missing edm4hep.root in {run_dir}")
        return issues

    try:
        with uproot.open(edm_file) as f:
            if "events" not in f:
                issues.append(f"'events' tree not found in {edm_file}")
                return issues
            events_tree = f["events"]
            n = events_tree.num_entries
            if n <= 0:
                issues.append(f"No entries in {edm_file}")
                return issues

            # Lightweight read: first and last event of any branch to ensure readability
            _ = events_tree.arrays(entry_start=0, entry_stop=1)
            _ = events_tree.arrays(entry_start=max(0, n-1), entry_stop=n)

            # Check for PixelBarrelReadout collection and compute total hits across all events
            det = "PixelBarrelReadout"
            if det not in events_tree:
                issues.append(f"Missing {det} collection in {edm_file}")
            else:
                try:
                    arr = events_tree[det].arrays([f"{det}.cellID"], entry_start=0, entry_stop=None, library="ak")
                    total_hits = ak.sum(ak.num(arr[f"{det}.cellID"]))
                    logger.info(f"{det}: total hits across file = {int(total_hits)}")
                except Exception as e:
                    issues.append(f"Failed reading {det} from {edm_file}: {e}")
    except Exception as e:
        issues.append(f"Failed to read edm4hep.root in {run_dir}: {e}")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=False, help="Pipeline stage being validated")
    # New unified arg
    parser.add_argument("--runs-dir", help="Directory containing run subdirectories")
    # Back-compat legacy args (optional)
    parser.add_argument("--run-dir", help="Legacy: Directory containing run outputs")
    parser.add_argument("--node-idx", type=int, help="Legacy: Node index being validated")
    parser.add_argument("--runs-per-node", type=int, help="Legacy: Number of runs per node")
    
    args = parser.parse_args()

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

    logger.info(f"Validating simulation; scanning runs in: {runs_dir}")

    # Only numeric-named subdirectories, sorted numerically
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    run_dirs = sorted(run_dirs, key=lambda p: int(p.name))
    if not run_dirs:
        logger.error(f"No run directories found in {runs_dir}")
        sys.exit(3)

    issues_total = []
    runs_with_issues = []
    for d in tqdm(run_dirs, desc="Validating runs", unit="run"):
        issues = validate_run(d)
        for issue in issues:
            logger.warning(issue)
        if issues:
            runs_with_issues.append(d.name)
            issues_total.extend(issues)

    if issues_total:
        bad_sorted = sorted(runs_with_issues, key=int)
        logger.error(f"Runs with issues ({len(bad_sorted)}): {', '.join(bad_sorted)}")
        logger.error(f"Validation FAILED with {len(issues_total)} issue(s)")
        sys.exit(1)
    else:
        logger.info("Validation PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main() 