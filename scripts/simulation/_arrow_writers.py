"""Attach ACTS-native Arrow + Parquet writers to a Sequencer.

Wraps the ArrowParticleOutputConverter / ArrowSimHitOutputConverter /
ArrowTrackOutputConverter / ArrowCaloHitOutputConverter machinery from
acts-project/acts PR #5410 (+ #5441 for calo). All four converters park
arrow::Table objects on the EventStore; one ParquetWriter then drains
them to ``<output_dir>/<collection>/<event_id>.parquet`` shards using
the ACTS-canonical schemas.

This module is imported only when ``output_parquet_arrow`` is set in the
config, and ``acts.examples.arrow`` is therefore importable — which in
turn requires the ACTS build to be the Arrow-enabled image (see
``docker/acts-arrow/Dockerfile``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import acts
from acts import UnitConstants as u

logger = logging.getLogger(__name__)


# ODD volume_id → per-subsystem detector enum. Same numbering as
# colliderml-production's encode_tracker_detector(), so downstream
# parquet consumers don't have to remap.
_ODD_TRACKER_VOLUME_MAP: dict[int, int] = {
    7: 0,   # PixelEndcapNeg
    8: 1,   # PixelBarrel
    9: 2,   # PixelEndcapPos
    12: 3,  # ShortStripEndcapNeg
    13: 4,  # ShortStripBarrel
    14: 5,  # ShortStripEndcapPos
    16: 6,  # LongStripEndcapNeg
    17: 7,  # LongStripBarrel
    18: 8,  # LongStripEndcapPos
}


def add_arrow_writers(
    s: Any,
    *,
    output_dir: Path,
    field: Any,
    tracking_geometry: Any,
    has_reco: bool = True,
    log_level: Any = None,
) -> None:
    """Attach Arrow output converters + a ParquetWriter to ``s``.

    Pre-conditions on the EventStore (all produced by the existing
    digi_and_reco.py pipeline before this is called):
      - ``simhits``                — SimHitContainer
      - ``particles_simulated``    — SimParticleContainer
      - ``measurements``           — MeasurementContainer
      - ``simhit_measurements_map``— forward sim-hit → measurement map
      - ``measurement_simhits_map``— inverse, needed for tracks
      - ``tracks``                 — TrackContainer (only when has_reco)
      - ``track_particle_matching``— track ↔ truth-particle match
    """
    try:
        from acts.arrow import (
            particleSchema,
            simHitSchema,
            trackSchema,
        )
        from acts.examples.arrow import (
            ArrowParticleOutputConverter,
            ArrowSimHitOutputConverter,
            ArrowTrackOutputConverter,
            makeVolumeIdDetectorResolver,
            ParquetWriter,
        )
    except ImportError as e:
        raise RuntimeError(
            "Arrow output requested but acts.examples.arrow is not importable; "
            "this runner needs the Arrow-enabled ACTS image (see "
            "docker/acts-arrow/Dockerfile)."
        ) from e

    if log_level is None:
        log_level = acts.logging.INFO

    detector_resolver = makeVolumeIdDetectorResolver(
        _ODD_TRACKER_VOLUME_MAP, 255
    )

    arr_particles = ArrowParticleOutputConverter(
        level=log_level,
        inputParticles="particles_simulated",
        outputTable="particles_arrow",
        writeHelixParameters=True,
        # Match the existing RootParticleWriter knobs in digi_and_reco.py:
        # propagate truth helix → perigee surface at the origin.
        minHelixTransverseMomentum=500 * u.MeV,
        maxHelixEta=5.0,
        bField=field,
    )
    s.addAlgorithm(arr_particles)

    arr_simhits = ArrowSimHitOutputConverter(
        level=log_level,
        inputSimHits="simhits",
        inputParticles="particles_simulated",
        inputMeasurements="measurements",
        inputSimHitMeasurementsMap="simhit_measurements_map",
        outputTable="simhits_arrow",
        trackingGeometry=tracking_geometry,
        detectorResolver=detector_resolver,
    )
    s.addAlgorithm(arr_simhits)

    collections: dict[str, str] = {
        arr_particles.config.outputTable: "particles",
        arr_simhits.config.outputTable: "tracker_hits",
    }
    expected: dict[str, Any] = {
        arr_particles.config.outputTable: particleSchema(),
        arr_simhits.config.outputTable: simHitSchema(),
    }

    if has_reco:
        arr_tracks = ArrowTrackOutputConverter(
            level=log_level,
            inputTracks="tracks",
            inputTrackParticleMatching="track_particle_matching",
            inputParticles="particles_simulated",
            inputMeasurementSimHitsMap="measurement_simhits_map",
            outputTable="tracks_arrow",
        )
        s.addAlgorithm(arr_tracks)
        collections[arr_tracks.config.outputTable] = "tracks"
        expected[arr_tracks.config.outputTable] = trackSchema()

    # Calo writer needs EDM4hepCaloHitInputConverter parked upstream to
    # produce the "calo_hits" EventStore key. Not yet wired in our
    # digi_and_reco.py — left out until we add the input converter. The
    # tracker side (particles, simhits, tracks) is enough to validate the
    # ACTS-native parquet path against convert_all.py.

    s.addWriter(
        ParquetWriter(
            level=log_level,
            outputDir=str(output_dir),
            collections=collections,
            expectedSchemas=expected,
        )
    )
    logger.info(
        "Arrow writers attached: %s -> %s",
        list(collections.keys()),
        output_dir,
    )
