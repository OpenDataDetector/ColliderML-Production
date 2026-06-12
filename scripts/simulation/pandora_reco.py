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
"""

import os
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.app_logging import setup_logging, TimingRecorder
from utils.config import create_base_parser, load_config


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

    pandora_settings = getattr(config, "pandora_settings", None)
    if pandora_settings is None and charged:
        pandora_settings = str(Path(k4odd_base) / "k4ODD/options/PandoraSettingsCLD.xml")

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

    logger.info("Pandora particle-flow reconstruction (k4ODD)")
    logger.info(f"  Input:    {input_file}")
    logger.info(f"  Output:   {output_file}")
    logger.info(f"  Events:   {events if events and int(events) > 0 else 'all'}")
    logger.info(f"  Tracks:   {track_collection} ({'charged PF' if charged else 'calo-only'})")
    logger.info(f"  Settings: {pandora_settings or 'ODDreconstruction.py default'}")
    if not env.get("PANDORA_STACK_PREFIX"):
        logger.warning("PANDORA_STACK_PREFIX not set: running on the STOCK cvmfs Pandora "
                       "(correct physics, but ~5.7x slower at mu=200 than the pinned stack)")

    cmd = ["k4run", str(k4odd_script),
           f"--inputFile={input_file}",
           f"--outputFile={output_file}",
           f"--events={events if events and int(events) > 0 else -1}"]
    logger.info(f"Running: {' '.join(cmd)}")

    # ODDreconstruction.py and the Pandora settings XMLs resolve some resources
    # (photon-likelihood XML etc.) relative to the k4ODD repo root - run from there.
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=k4odd_base)
    if result.returncode != 0:
        logger.error(f"k4run failed with return code {result.returncode}")
        logger.error(f"STDOUT (last 2000 chars): {result.stdout[-2000:]}")
        logger.error(f"STDERR (last 2000 chars): {result.stderr[-2000:]}")
        raise RuntimeError("Pandora reconstruction failed")
    if not output_file.exists():
        raise RuntimeError(f"Output file not created: {output_file}")

    # smoke-gate: PFO collections must be present (exit-0 silent failures have bitten us)
    check = subprocess.run(["podio-dump", str(output_file)], capture_output=True, text=True, env=env)
    for coll in ("GaudiPandoraPFOs", "GaudiPandoraClusters"):
        if coll not in check.stdout:
            raise RuntimeError(f"Output missing collection {coll} - reco silently failed")

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
