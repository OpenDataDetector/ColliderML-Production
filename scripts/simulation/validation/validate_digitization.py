import argparse
import logging
from pathlib import Path
import sys

import uproot
from tqdm import tqdm

# Optional: import loading_utils if available for ROOT loading
try:
    from colliderml_dev.notebooks.loading_utils import load_root_file
except Exception:
    load_root_file = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_first_last_csv(run_dir: Path):
    csvs = sorted(run_dir.glob("*.csv"))
    if not csvs:
        return None, None
    return csvs[0], csvs[-1]


def check_tprofile(file_path: Path, tprofile_path: str):
    try:
        with uproot.open(file_path) as f:
            # uproot uses "/" separator for nested objects
            obj = f[tprofile_path]
            # Access bins to ensure it's readable
            _ = obj.values()
        return True, ""
    except Exception as e:
        return False, str(e)


def validate_run_dir(run_dir: Path) -> list:
    issues = []

    # 1) First and last CSV present
    first_csv, last_csv = find_first_last_csv(run_dir)
    if first_csv is None or last_csv is None:
        issues.append(f"No CSVs found in {run_dir}")
    else:
        if not first_csv.exists():
            issues.append(f"First CSV missing: {first_csv}")
        if not last_csv.exists():
            issues.append(f"Last CSV missing: {last_csv}")

    # 2) measurements.root via loading_utils.load_root_file if available
    meas_root = run_dir / "measurements.root"
    if load_root_file is not None:
        try:
            _ = load_root_file(meas_root)
        except Exception as e:
            issues.append(f"Failed to load measurements.root with loading_utils: {e}")
    else:
        if not meas_root.exists():
            issues.append(f"Missing measurements.root: {meas_root}")

    # 3) purity_vs_eta TProfile inside performance_finding_ckf.root
    perf_root = run_dir / "performance_finding_ckf.root"
    if perf_root.exists():
        ok, msg = check_tprofile(perf_root, "purity_vs_eta")
        if not ok:
            issues.append(f"purity_vs_eta check failed in {perf_root}: {msg}")
    else:
        issues.append(f"Missing performance_finding_ckf.root: {perf_root}")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, help="Pipeline stage being validated")
    parser.add_argument("--runs-dir", required=True, help="Directory containing run subdirectories")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_dir():
        logger.error(f"Runs directory not found: {runs_dir}")
        sys.exit(2)

    logger.info(f"Validating stage: {args.stage}")
    logger.info(f"Scanning runs in: {runs_dir}")

    issues_total = []
    # Only numeric directories and sort numerically
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    run_dirs = sorted(run_dirs, key=lambda p: int(p.name))
    if not run_dirs:
        logger.error(f"No run directories found in {runs_dir}")
        sys.exit(3)

    runs_with_issues = []

    # Check each run directory with progress bar
    for d in tqdm(run_dirs, desc="Validating runs", unit="run"):
        issues = validate_run_dir(d)
        for issue in issues:
            logger.warning(issue)
        if issues:
            runs_with_issues.append(d.name)
            issues_total.extend(issues)

    if issues_total:
        bad_sorted = sorted(runs_with_issues, key=int)
        logger.error(f"Runs with issues ({len(bad_sorted)}): {' '.join(bad_sorted)}")
        logger.error(f"Validation FAILED with {len(issues_total)} issue(s)")
        sys.exit(1)
    else:
        logger.info("Validation PASSED")
        sys.exit(0)

if __name__ == "__main__":
    main() 