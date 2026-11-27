#!/usr/bin/env python3
"""
Pandora particle-flow reconstruction using k4ODD.

Runs k4run ODDreconstruction.py (digitisation + DDPandoraPFANewAlgorithm) on a simulated
EDM4hep file, producing GaudiPandoraPFOs / GaudiPandoraClusters / GaudiPandoraStartVertices
plus the digitised calo cells and their digi<->sim truth links. Runs AFTER simulation (and
after ACTS tracking if charged PF is wanted: the input file must then carry an edm4hep
Track collection, e.g. ActsTracks from the EDM4hepTrackOutputConverter).

Environment (loaded by the pipeline via env_setup.yaml):
  - Key4hep stack
  - K4ODD_PATH        -> k4ODD checkout (options/ODDreconstruction.py); its install must be
                         on LD_LIBRARY_PATH/PYTHONPATH (k4ODDPlugins)
  - the pinned Pandora stack (source $PANDORA_STACK_PREFIX/setup_stack.sh after key4hep;
    built by k4ODD ci/build_pandora_stack.sh from ci/pandora_stack.env) so the optimized
    LCContent (byte-identical 5.66x at mu=200) and patched k4GaudiPandora are used
  - ODD geometry (this_odd.sh / OpenDataDetector env)

Calibration: ODDreconstruction.py defaults ARE the calibrated realistic-digi constants;
no env knobs needed. Config keys (yaml):
  events                  event cap (default: all)
  track_collection        "ActsTracks" (charged PF, the default and REQUIRED mode: the stage
                          errors out if the input file has no such collection) or, ONLY when
                          set explicitly, "EmptyTracks" (calo-only neutral PF)
  pandora_settings        settings XML (default: PandoraSettingsCLD.xml for ActsTracks,
                          ODDreconstruction.py default Minimal for EmptyTracks)
  max_track_sigma_pop     track-quality cut (default 999 for ActsTracks: masks the
                          inflated ACTS omega covariance, issue #25)
  input_name              input file name in the run dir (default: sim_with_tracks.root
                          if present, else edm4hep.root)
  processes               concurrent k4run processes for this run (default 1). k4run is
                          single-threaded, so a run is split into contiguous entry slices
                          (IOSvc.FirstEventEntry + --events), one process each, and the
                          parts are merged in entry order with podio-merge-files.
  merge_slices            default true. false: keep the slices as <run>/reco_parts/
                          reco_part_NNN.root (reco_tables reads them directly).
  work_root               where the per-slice work dirs go (default: <run>/pandora_work);
                          e.g. a $PSCRATCH path to keep their inodes off CFS.

Isolation: every k4run gets its own K4ODD_OUTPUT_DIR (<run>/pandora_work/slice_NNN) for
ddcalodigi_hist.root and its log. Sharing one histogram file killed 18 of 64 processes at
finalize in the 2026-09-22 pilot. The cwd stays the k4ODD root (relative resources).
"""

import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.app_logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config


def _count_entries(path):
    """Number of events in a podio ROOT file."""
    try:
        import uproot
        return int(uproot.open(path)["events"].num_entries)
    except ImportError:
        import ROOT
        f = ROOT.TFile.Open(str(path))
        n = int(f.Get("events").GetEntries())
        f.Close()
        return n


def _remove_stale_outputs(output_dir, keep, n_parts, logger):
    """reco_tables prefers reco_parts/ over reco_edm4hep.root, so after a successful run remove
    the other output form and any slice files beyond this run's slice count; otherwise an older
    run's output could be published (Codex audit 2026-10-01, finding 2). Only files this stage
    writes are touched."""
    parts_dir = Path(output_dir) / "reco_parts"
    stale = []
    if keep == "merged":
        stale += sorted(parts_dir.glob("reco_part_*.root"))
    else:
        merged = Path(output_dir) / "reco_edm4hep.root"
        if merged.exists():
            stale.append(merged)
        stale += [p for p in sorted(parts_dir.glob("reco_part_*.root")) if int(p.stem.split("_")[-1]) >= n_parts]
    for p in stale:
        logger.info(f"removing stale Pandora output {p}")
        p.unlink()


def run_pandora_reco(input_file, output_dir, config, logger):
    k4odd_base = os.environ.get("K4ODD_PATH")
    if not k4odd_base:
        raise ValueError("K4ODD_PATH not found in environment. This should be set by env_setup.yaml")

    k4odd_script = Path(k4odd_base) / "k4ODD/options/ODDreconstruction.py"
    if not k4odd_script.exists():
        raise FileNotFoundError(f"k4ODD script not found: {k4odd_script}")

    events = getattr(config, "events", -1)
    # Charged particle flow is the production mode. Calo-only is NEVER a fallback:
    # it must be requested explicitly (track_collection: EmptyTracks in the config).
    track_collection = getattr(config, "track_collection", None) or "ActsTracks"
    charged = track_collection != "EmptyTracks"

    # Always pass an ABSOLUTE settings path. ODDreconstruction.py's default resolves it
    # via os.path.abspath(__file__), but k4run exec()s the options file so __file__ points
    # at the k4FWCore package dir, not k4ODD/options -> the default path is wrong and
    # Pandora dies with "Invalid xml file". Setting K4ODD_PANDORA_SETTINGS (below) sidesteps
    # that. Charged -> CLD; calo-only (EmptyTracks) -> Minimal.
    pandora_settings = getattr(config, "pandora_settings", None)
    if pandora_settings is None:
        _settings_name = "PandoraSettingsCLD.xml" if charged else "PandoraSettingsMinimal.xml"
        pandora_settings = str(Path(k4odd_base) / "k4ODD/options" / _settings_name)

    input_file = input_file.resolve()
    output_file = (output_dir / "reco_edm4hep.root").resolve()

    if charged:
        # Hard guard: refuse to run charged PF on an input without the track collection.
        # (Silently degrading to calo-only would produce wrong-looking PFOs downstream.)
        dump = subprocess.run(["podio-dump", str(input_file)], capture_output=True, text=True)
        if track_collection not in dump.stdout:
            raise RuntimeError(
                f"Input {input_file} has no '{track_collection}' collection - run the ACTS "
                f"tracking step first. Calo-only reconstruction requires explicitly setting "
                f"track_collection: EmptyTracks in the stage config.")

    env = os.environ.copy()
    if charged:
        env["K4ODD_TRACK_CREATOR"] = "DDTrackCreatorCLIC"
        env["K4ODD_TRACK_COLLECTION"] = track_collection
        # mask the inflated ACTS->edm4hep omega covariance (#25); override via config
        env["K4ODD_MAX_TRACK_SIGMA_POVERP"] = str(getattr(config, "max_track_sigma_pop", 999))
    if pandora_settings:
        env["K4ODD_PANDORA_SETTINGS"] = str(pandora_settings)
    if getattr(config, "preload_jemalloc", False):
        # jemalloc in k4run only: ~14% faster at PU200 with bit-identical output (2026-11-03).
        jemalloc = Path("/opt/libjemalloc.so.2")
        if not jemalloc.exists():
            raise RuntimeError(f"preload_jemalloc set but {jemalloc} not found (image built before 2026-11-03?)")
        env["LD_PRELOAD"] = ":".join(p for p in (str(jemalloc), env.get("LD_PRELOAD", "")) if p)

    logger.info("Pandora particle-flow reconstruction (k4ODD)")
    logger.info(f"  Input:    {input_file}")
    logger.info(f"  Output:   {output_file}")
    logger.info(f"  Events:   {events if events and int(events) > 0 else 'all'}")
    logger.info(f"  Tracks:   {track_collection} ({'charged PF' if charged else 'calo-only'})")
    logger.info(f"  Settings: {pandora_settings or 'ODDreconstruction.py default'}")
    if not env.get("PANDORA_STACK_PREFIX"):
        logger.warning("PANDORA_STACK_PREFIX not set: running on the STOCK cvmfs Pandora "
                       "(correct physics, but ~5.7x slower at mu=200 than the pinned stack)")

    n_input = _count_entries(input_file)
    n_events = n_input if not events or int(events) < 0 else min(int(events), n_input)
    n_proc = max(1, min(int(getattr(config, "processes", 1) or 1), n_events))
    bounds = [round(i * n_events / n_proc) for i in range(n_proc + 1)]
    # work_root (config): put the per-slice work dirs (~3 files per slice) outside the run dir,
    # e.g. on $PSCRATCH, to save CFS inodes. Same layout below it as the run dir's absolute path.
    work_base = getattr(config, "work_root", None)
    work_root = (Path(work_base) / str(output_dir.resolve()).lstrip("/") / "pandora_work") if work_base \
        else output_dir / "pandora_work"
    logger.info(f"  Slices:   {n_proc} process(es) over {n_events} of {n_input} entries")
    logger.info(f"  Work dir: {work_root}")

    def launch(i):
        first, count = bounds[i], bounds[i + 1] - bounds[i]
        work = work_root / f"slice_{i:03d}"
        work.mkdir(parents=True, exist_ok=True)
        part = (work / "reco_part.root") if n_proc > 1 else output_file
        if part.exists():  # a part from an earlier attempt must not pass as this one's output
            part.unlink()
        cmd = ["k4run", str(k4odd_script),
               f"--inputFile={input_file}",
               f"--outputFile={part}",
               f"--events={count}",
               f"--IOSvc.FirstEventEntry={first}"]
        penv = dict(env, K4ODD_OUTPUT_DIR=str(work))
        log = open(work / "k4run.log", "w")
        if i == 0:
            logger.info(f"Running (slice 0 of {n_proc}): {' '.join(cmd)}")
        # ODDreconstruction.py and the Pandora settings XMLs resolve some resources
        # (photon-likelihood XML etc.) relative to the k4ODD repo root - run from there.
        return (i, first, count, part, log,
                subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=penv, cwd=k4odd_base))

    # Per-slice record (v23 provenance): which source entries each k4run process read, and its
    # return code, every attempt included. Written whether or not the slices succeed.
    slice_log = []

    def _digest(path):
        import hashlib
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        except OSError:
            return None

    # the settings file and the photon-likelihood file it names, as k4run sees them (relative
    # paths resolve against k4odd_base, the k4run working directory), hashed in this container
    import re as _re
    _lik = None
    try:
        _m = _re.search(r"<HistogramFile>\s*([^<\s]+)\s*</HistogramFile>", Path(pandora_settings).read_text())
        _lik = (Path(k4odd_base) / _m.group(1)) if _m else None
    except OSError:
        pass
    resources = {"pandora_settings": {"path": str(pandora_settings), "sha256": _digest(pandora_settings)},
                 "photon_likelihood": {"path": str(_lik) if _lik else None,
                                       "sha256": _digest(_lik) if _lik else None}}

    def write_slice_record(status):
        import json
        rec = {"input_file": str(input_file), "n_input_entries": n_input, "n_events": n_events,
               "n_slices": n_proc, "resources": resources, "status": status, "slices": slice_log}
        tmp = output_dir / ".pandora_slices.json.tmp"
        tmp.write_text(json.dumps(rec, indent=1) + "\n")
        os.replace(tmp, output_dir / "pandora_slices.json")

    def finish(slice_procs):
        failed = []
        for i, first, count, part, log, proc in slice_procs:
            rc = proc.wait()
            log.close()
            slice_log.append({"slice": i, "first_entry": first, "count": count, "rc": rc,
                              "output_exists": part.exists()})
            if rc != 0 or not part.exists():
                failed.append(i)
                tail = (work_root / f"slice_{i:03d}" / "k4run.log").read_text()[-2000:]
                logger.error(f"slice {i} (entries {first}..{first + count - 1}) failed rc={rc}:\n{tail}")
        return failed

    procs = [launch(i) for i in range(n_proc)]
    failed = finish(procs)
    if failed:
        # At PU200 a few of 64 concurrent slices can run out of node memory (fixed config peaks
        # ~11 GB per event): retry each failed slice once, alone, after the others have finished.
        logger.warning(f"retrying {len(failed)} failed slice(s) one at a time: {failed}")
        failed = [i for i in failed if finish([launch(i)])]
    if failed:
        write_slice_record("failed")
        raise RuntimeError(f"Pandora reconstruction failed in {len(failed)} of {n_proc} slices: {failed}")
    write_slice_record("slices_done")   # final status written after the output checks below

    merge = bool(getattr(config, "merge_slices", True))
    if n_proc > 1 and not merge:
        # Keep the slices as <run>/reco_parts/reco_part_NNN.root (entry order = slice order);
        # reco_tables reads them directly. At PU200 podio-merge-files (a python frame copy)
        # took longer than the reconstruction itself.
        parts_dir = output_dir / "reco_parts"
        parts_dir.mkdir(exist_ok=True)
        n_out = 0
        for i, _, count, part, _, _ in procs:
            dst = parts_dir / f"reco_part_{i:03d}.root"
            shutil.move(part, dst)                      # work_root may be another file system
            n = _count_entries(dst)
            if n != count:
                write_slice_record(f"failed: slice {i} wrote {n} events, expected {count}")
                raise RuntimeError(f"slice {i}: {n} events written, expected {count}")
            dump = subprocess.run(["podio-dump", str(dst)], capture_output=True, text=True, env=env)
            missing = [c for c in ("GaudiPandoraPFOs", "GaudiPandoraClusters") if c not in dump.stdout]
            if missing:
                write_slice_record(f"failed: slice {i} lacks {missing}")
                raise RuntimeError(f"slice {i}: output missing collections {missing}")
            n_out += n
        write_slice_record("ok")
        logger.info(f"✓ {n_proc} slice files ({n_out} events) in {parts_dir} (merge_slices: false)")
        _remove_stale_outputs(output_dir, keep="parts", n_parts=n_proc, logger=logger)
        return parts_dir
    if n_proc > 1:
        parts = [str(part) for _, _, _, part, _, _ in procs]
        merged = subprocess.run(["podio-merge-files", "--output-file", str(output_file)] + parts,
                                capture_output=True, text=True, env=env)
        if merged.returncode != 0:
            logger.error(f"podio-merge-files failed: {merged.stderr[-2000:]}")
            raise RuntimeError("merging the Pandora slices failed")
        n_out = _count_entries(output_file)
        if n_out != n_events:
            raise RuntimeError(f"merged file has {n_out} events, expected {n_events}")
        for p in parts:
            Path(p).unlink()
    if not output_file.exists():
        raise RuntimeError(f"Output file not created: {output_file}")

    # smoke-gate: PFO collections must be present (exit-0 silent failures have bitten us)
    check = subprocess.run(["podio-dump", str(output_file)], capture_output=True, text=True, env=env)
    for coll in ("GaudiPandoraPFOs", "GaudiPandoraClusters"):
        if coll not in check.stdout:
            raise RuntimeError(f"Output missing collection {coll} - reco silently failed")

    _remove_stale_outputs(output_dir, keep="merged", n_parts=0, logger=logger)
    write_slice_record("ok")
    out_mb = output_file.stat().st_size / 1024**2
    logger.info(f"✓ Pandora reco complete ({out_mb:.1f} MB)")
    return output_file


def main():
    logger = setup_logging()
    try:
        parser = create_base_parser("Pandora particle-flow reconstruction via k4ODD")
        parser.add_argument("--input-file", type=Path, default=None,
                            help="Input EDM4hep file (default: {output_dir}/sim_with_tracks.root "
                                 "if present, else {output_dir}/edm4hep.root)")
        parser.add_argument("--processes", type=int, default=None,
                            help="Concurrent k4run processes (overrides the config key)")
        args = parser.parse_args()
        config = load_config(args)

        output_dir = Path(args.output)
        if hasattr(args, "output_subdir") and args.output_subdir:
            output_dir = output_dir / args.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.input_file:
            input_file = Path(args.input_file)
        else:
            input_name = getattr(config, "input_name", None)
            if input_name:
                input_file = output_dir / input_name
            else:
                tracked = output_dir / "sim_with_tracks.root"
                input_file = tracked if tracked.exists() else output_dir / "edm4hep.root"
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        logger.info("=" * 80)
        logger.info("Starting Pandora particle-flow reconstruction")
        logger.info("=" * 80)

        timer = TimingRecorder(output_dir)
        with timer.record("Pandora Reconstruction"):
            output_file = run_pandora_reco(input_file, output_dir, config, logger)
        timer.write_report()

        logger.info("=" * 80)
        logger.info("✓ Pandora reconstruction completed successfully")
        logger.info(f"  Output: {output_file}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Fatal error in pandora reco: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
