"""One-off edit (2026-11-06): point the v22 Pandora / reco_tables / packaging configs at the fixed reco image
20261106 and the staging tree, set work_root and the 100k run count. Prints a diff summary per file."""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ST = "/global/cfs/cdirs/m4958/data/ColliderML/staging/simulation"
PUB = "/global/cfs/cdirs/m4958/data/ColliderML/simulation"
SCR = "/pscratch/sd/d/danieltm/colliderml_pandora_work"
NOTE = "   # staging rerun 2026-11-06 (reco 20261106); published copy swapped in after verification"

edits = {
    "full_pileup/ttbar/v22/pandora_reco_config.yaml": dict(out=True, n_runs=1563, work_root=True),
    "full_pileup/ttbar/v22/reco_tables_config.yaml": dict(out=True, n_runs=1563),
    "hard_scatter/ttbar/v22/pandora_reco_config.yaml": dict(out=True, work_root=True),
    "hard_scatter/ttbar/v22/reco_tables_config.yaml": dict(out=True),
}
for rel, e in edits.items():
    p = REPO / "configs_production" / rel
    s0 = s = p.read_text()
    s = s.replace("colliderml/reco:20261103b", "colliderml/reco:20261106")
    s = s.replace("colliderml_reco_20261103b.tar", "colliderml_reco_20261106.tar")
    if e.get("out"):
        s = s.replace(f'  output_base_dir: "{PUB}"', f'  output_base_dir: "{ST}"{NOTE}')
    if e.get("n_runs"):
        s = re.sub(r"^  n_runs: \d+", f"  n_runs: {e['n_runs']}", s, flags=re.M)
    if e.get("work_root") and "work_root:" not in s:
        s = s.rstrip("\n") + f"\nwork_root: {SCR}   # per-slice work dirs off CFS (inodes)\n"
    if s == s0:
        sys.exit(f"no change in {rel}")
    p.write_text(s)
    n = re.search(r"n_runs: (\d+)", s).group(1)
    print(f"{rel}: image 20261106, output_base_dir {'staging' if ST in s else 'UNCHANGED'}, "
          f"n_runs {n}, work_root {'yes' if 'work_root:' in s else 'no'}")
