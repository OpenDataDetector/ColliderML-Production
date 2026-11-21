#!/usr/bin/env python3
"""Write <version>/.git_commit_success once, after the whole packaging stage has succeeded.

For every release chunk 0..N-1 (N = ceil(release_events / chunk_size) of the packaging config) there
must be an accepted provenance record (status "complete", return_code 0) and a chunk manifest with
status "ok"; all accepted records must name the same git SHA. Only then is the marker created
(exclusive create, never overwritten) and made read-only (0444). Exit 1 with the list of problems
otherwise; nothing is written.

  python finalize_version.py <package_parquet_config.yaml> [--dry-run]
"""
import datetime
import glob
import json
import os
import sys
from pathlib import Path

import yaml


def main() -> int:
    cfg_path = Path(sys.argv[1])
    dry = "--dry-run" in sys.argv
    c = yaml.safe_load(cfg_path.read_text())
    base = (c.get("common") or {}).get("output_base_dir") or c["output_base_dir"]
    vdir = Path(base) / c["campaign"] / c["dataset"] / c["version"]
    n = (int(c["release_events"]) + int(c["chunk_size"]) - 1) // int(c["chunk_size"])
    pdir = vdir / "provenance" / "package_parquet"
    problems, shas = [], set()
    for i in range(n):
        recs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(pdir / f"chunk{i}_attempt*.json")))]
        ok = [r for r in recs if r.get("status") == "complete" and r.get("return_code") == 0]
        if not ok:
            problems.append(f"chunk {i}: no accepted record ({len(recs)} attempts)")
            continue
        shas.add((ok[-1].get("git") or {}).get("sha"))
        man = pdir / f"chunk{i}_inputs.json"
        if not man.exists() or json.loads(man.read_text()).get("status") != "ok":
            problems.append(f"chunk {i}: chunk manifest missing or not ok")
    if len(shas) > 1:
        problems.append(f"accepted records name {len(shas)} different git SHAs: {sorted(shas)}")
    marker = vdir / ".git_commit_success"
    if marker.exists():
        problems.append(f"{marker} already exists; it is never overwritten")
    if problems:
        print("\n".join(problems[:50]))
        print(f"{len(problems)} problem(s); marker not written")
        return 1
    sha = shas.pop()
    print(f"{n} chunks accepted, code {sha}")
    if dry:
        return 0
    with open(marker, "x") as f:
        f.write(f"Finalized at {datetime.datetime.now()}\nGit Hash: {sha}\nConfig: configs/{cfg_path.name}\n"
                f"Chunks: {n}\nStage: package_parquet (finalize_version.py)\n")
    os.chmod(marker, 0o444)
    print(f"wrote {marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
