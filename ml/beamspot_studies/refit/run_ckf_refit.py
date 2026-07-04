#!/usr/bin/env python3
"""Beamspot-constrained CKF refit on ColliderML EDM4hep files (runs inside sw:pr-8).

Truth-seeded CKF -> Kalman refit with an optional beamspot constraint (the beamspot
enters as an extra 2D measurement on a perigee surface at the origin — ACTS PR #5342),
following D. Elitez's study harness (delitez/acts@ckfRefitClean), ported to the
ColliderML production container:
  - ODD v6 from /opt/odd (B = 3 T — verified at startup, hard gate)
  - single-file PodioReader (container ACTS) -> one job per edm4hep file, outputs
    are per-file; downstream analysis row-concatenates tracksummary TTrees (never
    merge histogram files — that corrupted widths in the original study)

Constraint configs (--constraint):
  none       unconstrained refit (reference)
  corrected  covariance diag(sigma_xy^2, sigma_z^2) = diag(0.0125^2, 55.5^2) mm^2
             — matches the full_pileup/ttbar/v1 generation beamspot exactly
  doga       diag(0.0125, 55.5) mm^2 — the original study's matrix, which passed the
             beamspot SIGMAS as a COVARIANCE (d0 ~9x looser, z0 ~7.4x tighter than
             truth). Kept for the cross-check that quantifies that bug.

Usage (inside the container, after sourcing setup_container_env.sh and stripping
/cache from LD_LIBRARY_PATH — see REPRODUCE.md):
  python3 run_ckf_refit.py --input .../runs/0/edm4hep.root --output out/0 \
      --constraint corrected [--events 10] [--odd-dir /opt/odd]
"""

import argparse
import sys
from pathlib import Path

import acts
import acts.examples
from acts.examples.edm4hep import EDM4hepSimInputConverter, PodioReader
from acts.examples.simulation import (
    addDigitization,
    addSimParticleSelection,
    addDigiParticleSelection,
    ParticleSelectorConfig,
)
from acts.examples.reconstruction import (
    addSeeding,
    addCKFTracks,
    TrackSelectorConfig,
    CkfConfig,
)
from acts.examples.root import (
    RootTrackSummaryWriter,
    RootTrackFitterPerformanceWriter,
)
from acts.examples.odd import getOpenDataDetector

u = acts.UnitConstants

# full_pileup/ttbar/v1 generation beamspot (pythia_config.yaml)
BEAMSPOT_SIGMA_XY_MM = 0.0125
BEAMSPOT_SIGMA_Z_MM = 55.5

CONSTRAINTS = {
    "none": None,
    "corrected": [BEAMSPOT_SIGMA_XY_MM**2, BEAMSPOT_SIGMA_Z_MM**2],
    "doga": [BEAMSPOT_SIGMA_XY_MM, BEAMSPOT_SIGMA_Z_MM],
}


def build_sequencer(args):
    odd_dir = Path(args.odd_dir)
    material_deco = acts.IMaterialDecorator.fromFile(
        odd_dir / "data/odd-material-maps.root"
    )
    detector = getOpenDataDetector(odd_dir=odd_dir, materialDecorator=material_deco)
    trackingGeometry = detector.trackingGeometry()
    field = detector.field

    # Hard gate: the original study lost weeks to a 2T/3T mismatch. Refuse to run
    # if the loaded geometry's field is not the 3T ColliderML configuration.
    ctx = acts.MagneticFieldContext()
    cache = field.makeCache(ctx)
    bz_T = field.getField(acts.Vector3(0, 0, 0), cache)[2] / u.T
    print(f"[field gate] B(0,0,0) = {bz_T:.4f} T")
    if abs(bz_T - 3.0) > 0.05:
        raise RuntimeError(f"B-field gate failed: Bz = {bz_T:.3f} T, expected 3 T")

    s = acts.examples.Sequencer(
        events=args.events,
        skip=args.skip,
        numThreads=1,  # PodioReader path; parallelism is across per-file jobs
        logLevel=acts.logging.INFO,
        trackFpes=False,
    )
    rnd = acts.examples.RandomNumbers(seed=args.seed)
    outputDir = Path(args.output)
    outputDir.mkdir(parents=True, exist_ok=True)

    s.addReader(
        PodioReader(
            level=acts.logging.INFO,
            inputPath=str(args.input),
            outputFrame="events",
            category="events",
        )
    )
    s.addAlgorithm(
        EDM4hepSimInputConverter(
            level=acts.logging.INFO,
            inputFrame="events",
            inputSimHits=[
                "PixelBarrelReadout",
                "PixelEndcapReadout",
                "ShortStripBarrelReadout",
                "ShortStripEndcapReadout",
                "LongStripBarrelReadout",
                "LongStripEndcapReadout",
            ],
            outputParticlesGenerator="particles_generated",
            outputParticlesSimulation="particles_simulated",
            outputSimHits="simhits",
            outputSimVertices="vertices_truth",
            dd4hepDetector=detector,
            trackingGeometry=trackingGeometry,
            sortSimHitsInTime=False,
        )
    )
    s.addWhiteboardAlias("particles", "particles_simulated")

    # Same particle selection as the original study
    addSimParticleSelection(
        s,
        ParticleSelectorConfig(
            rho=(0.0, 24 * u.mm),
            absZ=(0.0, 1.0 * u.m),
            eta=(-3.0, 3.0),
            removeNeutral=True,
        ),
    )

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=odd_dir / "config/odd-digi-smearing-config.json",
        rnd=rnd,
    )
    addDigiParticleSelection(
        s,
        ParticleSelectorConfig(
            pt=(0.9 * u.GeV, None),
            measurements=(7, None),
            removeNeutral=True,
            removeSecondaries=True,
        ),
    )

    addSeeding(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        inputParticles="particles_generated",
        particleHypothesis=acts.ParticleHypothesis.muon,
        initialSigmas=[
            1 * u.mm,
            1 * u.mm,
            1 * u.degree,
            1 * u.degree,
            0 / u.GeV,
            1 * u.ns,
        ],
        initialSigmaQoverPt=0.1 / u.GeV,
        initialSigmaPtRel=0.1,
        initialVarInflation=[1e0] * 6,
        geoSelectionConfigFile=odd_dir / "config/odd-seeding-config.json",
    )

    addCKFTracks(
        s,
        trackingGeometry,
        field,
        TrackSelectorConfig(
            pt=(1.0 * u.GeV, None),
            absEta=(None, 3.0),
            loc0=(-4.0 * u.mm, 4.0 * u.mm),
            nMeasurementsMin=7,
            maxHoles=2,
            maxOutliers=2,
        ),
        CkfConfig(
            chi2CutOffMeasurement=15.0,
            chi2CutOffOutlier=25.0,
            numMeasurementsCutOff=2,
            pixelVolumes=[16, 17, 18],
            stripVolumes=[23, 24, 25],
            maxPixelHoles=1,
            maxStripHoles=2,
            constrainToVolumes=[
                2,  # beam pipe
                32,
                4,  # beam pipe gap
                16, 17, 18,  # pixel
                20,  # PST
                23, 24, 25,  # short strip
                26, 8,  # long strip gap
                28, 29, 30,  # long strip
            ],
        ),
        outputDirRoot=None,  # explicit writers below
        writeTrackSummary=False,
        writeTrackStates=False,
        writePerformance=False,
        writeCovMat=False,
    )

    # Base (pre-refit) CKF reference output
    s.addWriter(
        RootTrackSummaryWriter(
            level=acts.logging.INFO,
            inputTracks="ckf_tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="track_particle_matching",
            filePath=str(outputDir / "tracksummary_ckf_base.root"),
        )
    )

    # Kalman refit of the CKF tracks, optionally beamspot-constrained.
    # `--constraint all` runs the three configs off the SAME CKF pass (the CKF+seeding
    # dominate the wall clock; each extra refit costs ~7%).
    kalmanOptions = {
        "multipleScattering": True,
        "energyLoss": True,
        "reverseFilteringMomThreshold": float("inf"),
        "reverseFilteringCovarianceScaling": 100.0,
        "freeToBoundCorrection": acts.examples.FreeToBoundCorrection(False),
        "level": acts.logging.INFO,
        "chi2Cut": float("inf"),
        "useJosephFormulation": False,
    }
    configs = list(CONSTRAINTS) if args.constraint == "all" else [args.constraint]
    for tag in configs:
        diag = CONSTRAINTS[tag]
        refit_kwargs = {}
        if diag is not None:
            refit_kwargs["beamSpotConstraint"] = acts.SquareMatrix2(
                [[diag[0], 0.0], [0.0, diag[1]]]
            )
        print(f"[constraint] {tag}: cov diag = {diag} mm^2")

        s.addAlgorithm(
            acts.examples.RefittingAlgorithm(
                level=acts.logging.INFO,
                inputTracks="ckf_tracks",
                outputTracks=f"ckf_refit_tracks_{tag}",
                initialVarInflation=6 * [100.0],
                fit=acts.examples.makeKalmanFitterFunction(
                    trackingGeometry, field, **kalmanOptions
                ),
                **refit_kwargs,
            )
        )
        s.addAlgorithm(
            acts.examples.TrackTruthMatcher(
                level=acts.logging.INFO,
                inputTracks=f"ckf_refit_tracks_{tag}",
                inputParticles="particles_selected",
                inputMeasurementParticlesMap="measurement_particles_map",
                outputTrackParticleMatching=f"refit_track_particle_matching_{tag}",
                outputParticleTrackMatching=f"refit_particle_track_matching_{tag}",
            )
        )
        s.addWriter(
            RootTrackSummaryWriter(
                level=acts.logging.INFO,
                inputTracks=f"ckf_refit_tracks_{tag}",
                inputParticles="particles_selected",
                inputTrackParticleMatching=f"refit_track_particle_matching_{tag}",
                filePath=str(outputDir / f"tracksummary_ckf_refit_{tag}.root"),
            )
        )
        s.addWriter(
            RootTrackFitterPerformanceWriter(
                level=acts.logging.INFO,
                inputTracks=f"ckf_refit_tracks_{tag}",
                inputParticles="particles_selected",
                inputTrackParticleMatching=f"refit_track_particle_matching_{tag}",
                filePath=str(outputDir / f"performance_ckf_refit_{tag}.root"),
            )
        )
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="one edm4hep.root file")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument(
        "--constraint",
        choices=sorted(CONSTRAINTS) + ["all"],
        default="all",
        help="beamspot-constraint config; 'all' runs the three off one CKF pass",
    )
    p.add_argument("--events", type=int, default=10)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--odd-dir", default="/opt/odd")
    args = p.parse_args()

    if not Path(args.input).is_file():
        sys.exit(f"input not found: {args.input}")
    build_sequencer(args).run()


if __name__ == "__main__":
    main()
