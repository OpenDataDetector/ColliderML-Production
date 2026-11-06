"""Build staging run dirs for a Pandora rerun: <staging>/<campaign>/<dataset>/<version>/runs/<N>/ with
symlinks to the tracking-stage outputs of the published run dir, so pandora_reco, reco_tables and
package_parquet (output_base_dir / input_base_dir = staging) write only into staging while the
published run dirs and parquet stay untouched. Links may dangle for runs whose tracking is still
running (they resolve when it finishes). Never overwrites a path that is not already the same link.

Usage: stage_runs.py <published_version_dir> <staging_version_dir> <first> <last> [--apply]
"""
import os
import sys

LINKED = ["edm4hep.root", "sim_with_tracks.root", "particles", "particles.root", "measurements.root",
          "tracker_hits", "tracker_simhits", "tracks", "truth_tracks", "tracksummary_ambi.root",
          "tracksummary_truth.root", "performance_finding_ambi.root", "performance_fitting_ambi.root",
          "performance_measurements.root", "generator_particles"]

src_v, dst_v, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
apply = "--apply" in sys.argv
made = same = 0
for r in range(a, b + 1):
    d = f"{dst_v}/runs/{r}"
    if apply:
        os.makedirs(d, exist_ok=True)
    for name in LINKED:
        target, link = f"{src_v}/runs/{r}/{name}", f"{d}/{name}"
        if os.path.lexists(link):
            if os.path.islink(link) and os.readlink(link) == target:
                same += 1
                continue
            sys.exit(f"refusing to overwrite {link}")
        if apply:
            os.symlink(target, link)
        made += 1
print(f"{'made' if apply else 'would make'} {made} links ({same} already in place) for runs {a}-{b} in {dst_v}")
