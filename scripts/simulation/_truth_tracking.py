"""Attach a truth-found track collection to a Sequencer.

Truth is used for *track finding only* — deciding which measurements belong to
which particle. It is deliberately kept out of the fit, including the fit's
starting parameters:

  * ``TruthSeedingAlgorithm`` groups measurements per particle via
    ``particle_measurements_map`` (this is the truth-informed step), orders them
    by sim-hit time, and picks three real space points off that group to form a
    seed.
  * ``TrackParamsEstimationAlgorithm`` turns that seed into starting parameters
    with a three-point helix estimate from tracking geometry and the B-field.
    No truth kinematics enter here.
  * ``TrackFittingAlgorithm`` fits the measurements from those starting parameters.

This is NOT the upstream ``SeedingAlgorithm.TruthSmeared`` path used by ACTS'
truth_tracking_kalman.py / truth_tracking_gx2f.py examples. That path runs
ParticleTrackParamExtractor + TrackParameterSmearing, i.e. it seeds the fit from
smeared *true* parameters. It is excluded on purpose.

One subtlety worth keeping: TruthSeedingAlgorithm falls back to the true PID for
the per-seed particle hypothesis (``m_cfg.particleHypothesis.value_or(
particle.hypothesis())``), which would be another truth leak into the fit. We
always pass an explicit hypothesis to override it.

This module is imported only when ``truth_tracking`` is set in the config.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import acts

logger = logging.getLogger(__name__)

# Whiteboard keys owned by this module. Deliberately distinct from the CKF
# path's keys ("estimatedparameters", "ckf_tracks", "tracks", ...) because the
# EventStore is write-once and both chains run in the same pass.
PROTO_TRACKS = "truth_proto_tracks"
SEEDS = "truth_seeds"
SEED_HYPOTHESES = "truth_seed_hypotheses"
SEEDED_PARTICLES = "truth_seeded_particles"
ESTIMATED_PARAMS = "truth_estimatedparameters"
ESTIMATED_PROTO_TRACKS = "truth_estimated_proto_tracks"

# Fitters we know how to build. Keys are the accepted config values.
_FITTERS = ("kf", "gx2f", "both")


def _make_fit_function(kind: str, tracking_geometry: Any, field: Any, log_level: Any):
    """Build an ACTS fitter function. Neither fitter sees truth."""
    if kind == "kf":
        return acts.examples.makeKalmanFitterFunction(
            tracking_geometry,
            field,
            multipleScattering=True,
            energyLoss=True,
            reverseFilteringMomThreshold=0 * acts.UnitConstants.GeV,
            reverseFilteringCovarianceScaling=100.0,
            freeToBoundCorrection=acts.examples.FreeToBoundCorrection(False),
            level=log_level,
            chi2Cut=float("inf"),
            useJosephFormulation=False,
        )
    if kind == "gx2f":
        # multipleScattering/energyLoss off here mirrors ACTS' own addGx2fTracks
        # defaults; the global fit is not set up to absorb them the way the KF is.
        return acts.examples.makeGlobalChiSquareFitterFunction(
            tracking_geometry,
            field,
            multipleScattering=False,
            energyLoss=False,
            freeToBoundCorrection=acts.examples.FreeToBoundCorrection(False),
            nUpdateMax=5,
            relChi2changeCutOff=1e-7,
            level=log_level,
        )
    raise ValueError(f"unknown fitter {kind!r}")


def _add_fit_and_match(
    s: Any,
    *,
    tracking_geometry: Any,
    field: Any,
    fitter: str,
    suffix: str,
    track_selector_config: Any,
    log_level: Any,
) -> tuple[str, str]:
    """Fit the truth proto tracks, select, and truth-match. Returns (tracks, matching)."""
    raw_tracks = f"truth_tracks_{suffix}_raw"
    selected_tracks = f"truth_tracks_{suffix}"
    matching = f"truth_track_particle_matching_{suffix}"

    s.addAlgorithm(
        acts.examples.TrackFittingAlgorithm(
            level=log_level,
            inputMeasurements="measurements",
            # Post-estimation proto tracks: index-aligned with the parameters
            # below, because TrackParamsEstimationAlgorithm drops any seed whose
            # estimate failed and republishes only the survivors. Feeding it the
            # pre-estimation proto tracks would silently misalign the two.
            inputProtoTracks=ESTIMATED_PROTO_TRACKS,
            inputInitialTrackParameters=ESTIMATED_PARAMS,
            inputClusters="",
            outputTracks=raw_tracks,
            pickTrack=-1,
            fit=_make_fit_function(fitter, tracking_geometry, field, log_level),
            calibrator=acts.examples.makePassThroughCalibrator(),
        )
    )

    # Match the CKF path's acceptance. The CKF applies these cuts during
    # finding (via trackSelectorCfg); here they are a post-fit filter. Same
    # surviving acceptance, which is what makes the two collections comparable.
    from acts.examples.reconstruction import addTrackSelection

    addTrackSelection(
        s,
        track_selector_config,
        inputTracks=raw_tracks,
        outputTracks=selected_tracks,
        logLevel=log_level,
    )

    s.addAlgorithm(
        acts.examples.TrackTruthMatcher(
            level=log_level,
            inputTracks=selected_tracks,
            inputParticles="particles",
            inputMeasurementParticlesMap="measurement_particles_map",
            outputTrackParticleMatching=matching,
            outputParticleTrackMatching=f"truth_particle_track_matching_{suffix}",
            doubleMatching=True,
        )
    )

    return selected_tracks, matching


def add_truth_tracking(
    s: Any,
    *,
    tracking_geometry: Any,
    field: Any,
    track_selector_config: Any,
    selected_particles: str = "particles_selected",
    space_points: str = "spacepoints",
    initial_sigmas: Optional[Sequence[float]] = None,
    initial_sigma_qoverpt: Optional[float] = None,
    initial_sigma_ptrel: Optional[float] = None,
    initial_var_inflation: Optional[Sequence[float]] = None,
    particle_hypothesis: Any = None,
    delta_r: tuple = (10.0, None),
    fitter: str = "kf",
    log_level: Any = None,
) -> dict[str, str]:
    """Wire truth finding -> geometric seed estimate -> fit -> truth match.

    Must be called AFTER addSeeding, which is what creates ``space_points``.

    Deliberately does not go through addSeeding / addKalmanTracks / addGx2fTracks:
    those write the whiteboard key ``estimatedparameters`` and register the alias
    ``tracks``, both already owned by the CKF path in the same sequence.

    Returns a mapping of {suffix: (tracks, matching)} flattened into a dict of
    ``{"tracks": ..., "matching": ...}`` per fitter, keyed by fitter name.
    """
    if fitter not in _FITTERS:
        raise ValueError(f"truth_tracking_fitter must be one of {_FITTERS}, got {fitter!r}")

    if log_level is None:
        log_level = acts.logging.INFO

    if particle_hypothesis is None:
        particle_hypothesis = acts.ParticleHypothesis.pion

    # --- Step 1: truth-informed hit grouping (the ONLY truth-informed step) ---
    truth_seeding_kwargs = dict(
        level=log_level,
        inputParticles=selected_particles,
        inputParticleMeasurementsMap="particle_measurements_map",
        inputSpacePoints=space_points,
        inputSimHits="simhits",
        inputMeasurementSimHitsMap="measurement_simhits_map",
        outputParticles=SEEDED_PARTICLES,
        outputProtoTracks=PROTO_TRACKS,
        outputSeeds=SEEDS,
        outputParticleHypotheses=SEED_HYPOTHESES,
        # Explicit override: without it the hypothesis falls back to the true
        # PID, which would put truth back into the fit.
        particleHypothesis=particle_hypothesis,
    )
    if delta_r is not None:
        if delta_r[0] is not None:
            truth_seeding_kwargs["deltaRMin"] = delta_r[0]
        if delta_r[1] is not None:
            truth_seeding_kwargs["deltaRMax"] = delta_r[1]
    s.addAlgorithm(acts.examples.TruthSeedingAlgorithm(**truth_seeding_kwargs))

    # --- Step 2: starting parameters from real space points, no truth ---
    par_kwargs = dict(
        level=log_level,
        inputSeeds=SEEDS,
        inputProtoTracks=PROTO_TRACKS,
        inputParticleHypotheses=SEED_HYPOTHESES,
        outputTrackParameters=ESTIMATED_PARAMS,
        outputProtoTracks=ESTIMATED_PROTO_TRACKS,
        trackingGeometry=tracking_geometry,
        magneticField=field,
        particleHypothesis=particle_hypothesis,
    )
    # Match the CKF path's initial uncertainties so the two chains start from
    # comparably-sized covariances.
    if initial_sigmas is not None:
        par_kwargs["initialSigmas"] = list(initial_sigmas)
    if initial_sigma_qoverpt is not None:
        par_kwargs["initialSigmaQoverPt"] = initial_sigma_qoverpt
    if initial_sigma_ptrel is not None:
        par_kwargs["initialSigmaPtRel"] = initial_sigma_ptrel
    if initial_var_inflation is not None:
        par_kwargs["initialVarInflation"] = list(initial_var_inflation)
    s.addAlgorithm(acts.examples.TrackParamsEstimationAlgorithm(**par_kwargs))

    # --- Step 3+4: fit, select, truth-match. No ambiguity resolution: truth
    # finding yields one track per particle, so there is nothing to resolve.
    outputs: dict[str, str] = {}
    kinds = ("kf", "gx2f") if fitter == "both" else (fitter,)
    for kind in kinds:
        tracks, matching = _add_fit_and_match(
            s,
            tracking_geometry=tracking_geometry,
            field=field,
            fitter=kind,
            suffix=kind,
            track_selector_config=track_selector_config,
            log_level=log_level,
        )
        outputs[f"{kind}_tracks"] = tracks
        outputs[f"{kind}_matching"] = matching

    logger.info(
        "Truth tracking attached (fitter=%s): truth finding only, fit seeded from "
        "a geometric three-point estimate; outputs %s",
        fitter,
        sorted(outputs.values()),
    )
    return outputs
