"""Check of the ddsim output file: each requested event must be written exactly once.

The ddsim of the sw:pr-8 image writes each event twice when it runs with one thread
(observed 2026-11-27: 10 events requested, 20 entries, event numbers 0,0,1,1,...).
Its log shows Geant4 falling back to the serial run manager; we suspect the output
action is then registered twice, but have not looked into DD4hep to confirm it. With
two or more threads the file is as expected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EVENT_NUMBER_BRANCH = "EventHeader/EventHeader.eventNumber"


class SimOutputError(RuntimeError):
    """The simulation output is missing, unreadable, short, or repeats an event."""


def duplicated_event_numbers(event_numbers) -> list[int]:
    """Event numbers that occur more than once, ascending."""
    values, counts = np.unique(np.asarray(event_numbers, dtype=np.int64), return_counts=True)
    return values[counts > 1].tolist()


def read_event_numbers(sim_file: Path) -> np.ndarray:
    """Event number of every entry of a podio ROOT file (one EventHeader per entry)."""
    import uproot

    try:
        with uproot.open(sim_file) as f:
            per_entry = f["events"][EVENT_NUMBER_BRANCH].array(library="np")
    except Exception as exc:
        raise SimOutputError(f"{sim_file}: cannot read the event numbers ({type(exc).__name__}: {exc})") from exc
    if any(len(x) != 1 for x in per_entry):
        raise SimOutputError(f"{sim_file}: an entry does not have exactly one EventHeader")
    return np.array([int(x[0]) for x in per_entry], dtype=np.int64)


def check_sim_output(sim_file: Path, requested_events=None) -> int:
    """Raise SimOutputError unless ``sim_file`` holds each event once and, if
    ``requested_events`` is a positive number N, exactly the events 0 to N-1 (ddsim numbers
    the events of its input file from 0). Returns the event count."""
    numbers = read_event_numbers(sim_file)
    duplicated = duplicated_event_numbers(numbers)
    if duplicated:
        raise SimOutputError(
            f"{sim_file}: {len(numbers)} entries but {len(np.unique(numbers))} distinct event numbers; "
            f"written more than once: {duplicated[:10]}. With the sw:pr-8 image this happens when "
            "ddsim runs with one thread: set threads to 2 or more.")
    if len(numbers) == 0:
        raise SimOutputError(f"{sim_file}: no events")
    if requested_events and int(requested_events) > 0:
        n = int(requested_events)
        if len(numbers) != n:
            raise SimOutputError(f"{sim_file}: {len(numbers)} events, {n} requested")
        if set(numbers.tolist()) != set(range(n)):
            raise SimOutputError(f"{sim_file}: event numbers {sorted(numbers.tolist())[:10]}... are not 0 to {n - 1}")
    return len(numbers)
