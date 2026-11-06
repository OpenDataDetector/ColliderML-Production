"""Remove run dirs that hold nothing but an input symlink made by link_version_objects.py.
Refuses any dir with other content or a non-symlink. Usage: unlink_runs.py <version_dir> <first> <last> [--apply]"""
import os
import sys

V, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
apply = "--apply" in sys.argv
n = 0
for r in range(a, b + 1):
    d = f"{V}/runs/{r}"
    entries = os.listdir(d)
    if entries != ["edm4hep.root"] or not os.path.islink(f"{d}/edm4hep.root"):
        sys.exit(f"refusing {d}: contents {entries}")
    if apply:
        os.unlink(f"{d}/edm4hep.root")
        os.rmdir(d)
    n += 1
print(f"{'removed' if apply else 'would remove'} {n} link-only run dirs in {V}")
