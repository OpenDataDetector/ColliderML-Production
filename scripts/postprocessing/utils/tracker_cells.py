"""Convert the ACTS cluster ROOT output to ColliderML tracker cell shards."""

from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .parquet_schemas import TRACKER_CELLS_PARQUET_TYPES
from .parquet_utils import write_parquet_table
from .track_utils import load_root_file

GEOMETRY_FIELDS = ("volume_id", "layer_id", "surface_id", "extra_id")
MEASUREMENT_COORDINATES = ("rec_gx", "rec_gy", "rec_gz")
CLUSTER_COORDINATES = ("clus_gx", "clus_gy", "clus_gz")
CELL_BRANCHES = ("channel_loc0", "channel_loc1", "channel_value")


def write_tracker_cells(
    cells: pd.DataFrame, path: Path, *, row_group_size: int | None
) -> None:
    write_parquet_table(
        cells,
        str(path),
        schema_overrides=TRACKER_CELLS_PARQUET_TYPES,
        row_group_size=row_group_size,
    )


def _build_cell_frame(
    clusters: ak.Array, local_events: tuple[int, int], event_offset: int
) -> pd.DataFrame:
    sizes = ak.to_numpy(clusters["clus_size"])
    if not all(
        np.array_equal(sizes, ak.to_numpy(ak.num(clusters[field], axis=1)))
        for field in CELL_BRANCHES
    ):
        raise ValueError("ACTS cluster size disagrees with its cell branch lengths")

    start, stop = local_events
    events = np.arange(start, stop, dtype=np.int64)
    # Locate each event's first cluster, then convert cluster offsets to cell offsets.
    cluster_offsets = np.searchsorted(
        ak.to_numpy(clusters["event_nr"]), np.arange(start, stop + 1)
    )
    cell_offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    event_cell_offsets = cell_offsets[cluster_offsets]
    values = {"measurement_id": np.repeat(ak.to_numpy(clusters["cluster_id"]), sizes)}
    for name, branch in zip(("channel0", "channel1", "value"), CELL_BRANCHES):
        values[name] = ak.to_numpy(ak.flatten(clusters[branch]))
    columns = {
        name: [
            data[lo:hi] for lo, hi in zip(event_cell_offsets, event_cell_offsets[1:])
        ]
        for name, data in values.items()
    }
    return pd.DataFrame({"event_id": events + event_offset, **columns})


def load_digitized_tracker(
    run_dir: Path,
    local_events: tuple[int, int],
    event_offset: int,
    *,
    measurement_columns: list[str] | None,
    include_cells: bool,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Read a run slice and join split ACTS outputs by local event and measurement ID."""
    measurements_path = run_dir / "measurements.root"
    clusters_path = run_dir / "clusters.root"
    start, stop = local_events
    if include_cells:
        with uproot.open(run_dir / "edm4hep.root") as source:
            if stop > source["events"].num_entries:
                raise ValueError(
                    f"{run_dir}: tracker_cells event range exceeds edm4hep.root"
                )
    cut = f"(event_nr >= {start}) & (event_nr < {stop})"
    requested = list(measurement_columns or ())
    with uproot.open(measurements_path) as source:
        tree = source["measurements"]
        use_cluster_coordinates = not set(MEASUREMENT_COORDINATES).issubset(tree.keys())
        if use_cluster_coordinates or include_cells:
            join_fields = ["event_nr", "measurement_id", *GEOMETRY_FIELDS]
            branches = list(
                dict.fromkeys(join_fields + [key for key in requested if key in tree])
            )
            measures = pd.DataFrame(tree.arrays(branches, cut=cut, library="np"))
            measures = measures.sort_values(
                ["event_nr", "measurement_id"], ignore_index=True
            )

    if not use_cluster_coordinates and not include_cells:
        return load_root_file(str(measurements_path), included_columns=requested), None

    cluster_fields = ["event_nr", "cluster_id", *GEOMETRY_FIELDS]
    if use_cluster_coordinates and measurement_columns is not None:
        cluster_fields.extend(CLUSTER_COORDINATES)
    if include_cells:
        cluster_fields.extend(("clus_size", *CELL_BRANCHES))
    with uproot.open(clusters_path) as source:
        clusters = source["clusters"].arrays(cluster_fields, cut=cut, library="ak")
    cluster_order = np.lexsort(
        (ak.to_numpy(clusters["cluster_id"]), ak.to_numpy(clusters["event_nr"]))
    )
    clusters = clusters[cluster_order]

    if not all(
        np.array_equal(
            measures[measure_key].to_numpy(), ak.to_numpy(clusters[cluster_key])
        )
        for measure_key, cluster_key in zip(join_fields, cluster_fields)
    ):
        raise ValueError(
            f"{run_dir}: ACTS cluster and measurement IDs or geometry disagree"
        )
    hit_positions = measures.groupby("event_nr", sort=False).cumcount()
    if not np.array_equal(measures["measurement_id"], hit_positions):
        raise ValueError(
            f"{measurements_path}: measurement IDs are not consecutive within events"
        )

    hits = None
    if measurement_columns is not None:
        if use_cluster_coordinates:
            for measurement_key, cluster_key in zip(
                MEASUREMENT_COORDINATES, CLUSTER_COORDINATES
            ):
                measures[measurement_key] = ak.to_numpy(clusters[cluster_key])
        columns = [key for key in requested if key in measures]
        hits = measures[columns].rename(columns={"event_nr": "event_id"})

    cells = (
        _build_cell_frame(clusters, local_events, event_offset)
        if include_cells
        else None
    )
    return hits, cells
