"""calo_hits validator: detector codes 9-14, equal contribution lengths, optional event count."""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "dataset"))

from validate_calo_hits import check_calo_hits_file, main  # noqa: E402


def write_calo(path, detector, ids, energies, times=None, event_ids=(0, 1)):
    """One row per event; detector/ids/energies hold the per-hit lists of each event."""
    cols = {
        "event_id": pa.array(list(event_ids), pa.uint32()),
        "detector": pa.array(detector, pa.list_(pa.uint8())),
        "contrib_particle_ids": pa.array(ids, pa.list_(pa.list_(pa.uint64()))),
        "contrib_energies": pa.array(energies, pa.list_(pa.list_(pa.float32()))),
    }
    if times is not None:
        cols["contrib_times"] = pa.array(times, pa.list_(pa.list_(pa.float32())))
    pq.write_table(pa.table(cols), path)
    return path


def good_file(path):
    return write_calo(path, [[9, 10], [14]], [[[1, 2], [3]], [[4]]], [[[0.1, 0.2], [0.3]], [[0.4]]],
                      times=[[[1.0, 2.0], [3.0]], [[4.0]]])


def test_clean_file_passes(tmp_path):
    ok, msg = check_calo_hits_file(good_file(tmp_path / "ok.parquet"), expected_events=2)
    assert ok, msg


def test_detector_255_fails(tmp_path):
    f = write_calo(tmp_path / "det.parquet", [[9, 255], [255]], [[[1], [2]], [[3]]], [[[0.1], [0.2]], [[0.3]]])
    ok, msg = check_calo_hits_file(f)
    assert not ok and "bad_detector_hits 2" in msg


def test_mismatched_contribution_lengths_fail(tmp_path):
    # Release 1 patterns: ids present but energies empty (1/0) and the reverse (0/1)
    f = write_calo(tmp_path / "len.parquet", [[9, 10], [11]], [[[1], []], [[3]]], [[[], [0.2]], [[0.3]]])
    ok, msg = check_calo_hits_file(f)
    assert not ok and "bad_contrib_hits 2" in msg


def test_times_length_checked_when_present(tmp_path):
    f = write_calo(tmp_path / "t.parquet", [[9], [10]], [[[1, 2]], [[3]]], [[[0.1, 0.2]], [[0.3]]],
                   times=[[[1.0]], [[2.0]]])
    ok, msg = check_calo_hits_file(f)
    assert not ok and "bad_contrib_hits 1" in msg


def test_short_file_fails_event_count(tmp_path):
    ok, msg = check_calo_hits_file(good_file(tmp_path / "ok.parquet"), expected_events=3)
    assert not ok and "events 2/3" in msg


def test_cli_exit_status(tmp_path):
    good = good_file(tmp_path / "ok.parquet")
    bad = write_calo(tmp_path / "bad.parquet", [[255], [9]], [[[1]], [[2]]], [[[0.1]], [[0.2]]])
    assert main([str(good)]) == 0
    assert main([str(good), str(bad)]) == 1
