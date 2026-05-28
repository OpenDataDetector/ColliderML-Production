"""Shared fixtures for the ACTS-native ↔ v1 parquet regression suite."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

ENV_V1 = "COLLIDERML_V1_PARQUET_DIR"
ENV_ACTS = "COLLIDERML_ACTSNATIVE_PARQUET_DIR"


def _require_dir(env: str) -> Path:
    raw = os.environ.get(env, "").strip()
    if not raw:
        pytest.skip(f"{env} not set — see tests/regression/README.md")
    p = Path(raw)
    if not p.is_dir():
        pytest.skip(f"{env}={raw} is not a directory")
    return p


@pytest.fixture(scope="session")
def v1_parquet_root() -> Path:
    """v1 layout: ``<root>/<truth|reco>/<object>/...events*.parquet``."""
    return _require_dir(ENV_V1)


@pytest.fixture(scope="session")
def acts_parquet_root() -> Path:
    """ACTS-native layout: ``<root>/<object>/event*.parquet`` (one shard
    per event per object, written by acts.examples.arrow.ParquetWriter)."""
    return _require_dir(ENV_ACTS)


def _read_v1_object(root: Path, kind: str, subdir: str):
    """Read every v1 shard for an object into one polars DataFrame.

    ``kind`` is the directory name under truth/ or reco/ (e.g. ``particles``,
    ``tracker_hits``, ``calo_hits``, ``tracks``). ``subdir`` is ``truth`` or
    ``reco``.
    """
    import polars as pl

    shards = sorted((root / subdir / kind).glob("*.parquet"))
    if not shards:
        pytest.skip(f"v1 has no shards under {root}/{subdir}/{kind}")
    return pl.concat([pl.read_parquet(p) for p in shards]).sort("event_id")


def _read_acts_object(root: Path, kind: str):
    """Read every ACTS-native shard for an object into one polars DataFrame."""
    import polars as pl

    shards = sorted((root / kind).glob("*.parquet"))
    if not shards:
        pytest.skip(f"ACTS-native has no shards under {root}/{kind}")
    return pl.concat([pl.read_parquet(p) for p in shards]).sort("event_id")


@pytest.fixture(scope="session")
def v1_particles(v1_parquet_root):
    return _read_v1_object(v1_parquet_root, "particles", "truth")


@pytest.fixture(scope="session")
def v1_tracker_hits(v1_parquet_root):
    return _read_v1_object(v1_parquet_root, "tracker_hits", "reco")


@pytest.fixture(scope="session")
def v1_tracks(v1_parquet_root):
    return _read_v1_object(v1_parquet_root, "tracks", "reco")


@pytest.fixture(scope="session")
def v1_calo_hits(v1_parquet_root):
    return _read_v1_object(v1_parquet_root, "calo_hits", "reco")


@pytest.fixture(scope="session")
def acts_particles(acts_parquet_root):
    return _read_acts_object(acts_parquet_root, "particles")


@pytest.fixture(scope="session")
def acts_tracker_hits(acts_parquet_root):
    return _read_acts_object(acts_parquet_root, "tracker_hits")


@pytest.fixture(scope="session")
def acts_tracks(acts_parquet_root):
    return _read_acts_object(acts_parquet_root, "tracks")


@pytest.fixture(scope="session")
def acts_calo_hits(acts_parquet_root):
    return _read_acts_object(acts_parquet_root, "calo_hits")
