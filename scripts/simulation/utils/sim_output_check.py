"""Check of the ddsim output file: every event must be written once.

The ddsim of the sw:pr-8 image writes each event twice when it runs with one thread
(observed 2026-11-27: 10 events requested, 20 entries, event numbers 0,0,1,1,...).
Geant4 falls back to its serial run manager there and the output action ends up
registered twice. With two or more threads the file is as expected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

EVENT_NUMBER_BRANCH = "EventHeader/EventHeader.eventNumber"


class DuplicateEventsError(RuntimeError):
    """The simulation output holds an event number more than once."""


def duplicated_event_numbers(event_numbers) -> list[int]:
    """Event numbers that occur more than once, ascending."""
    values, counts = np.unique(np.asarray(event_numbers, dtype=np.int64), return_counts=True)
    return values[counts > 1].tolist()


def check_unique_event_numbers(sim_file: Path) -> int:
    """Raise DuplicateEventsError if ``sim_file`` holds an event number twice.

    Returns the number of entries."""
    import uproot

    with uproot.open(sim_file) as f:
        numbers = np.concatenate(f["events"][EVENT_NUMBER_BRANCH].array(library="np")) \
            if f["events"].num_entries else np.zeros(0, np.int64)
    duplicated = duplicated_event_numbers(numbers)
    if duplicated:
        raise DuplicateEventsError(
            f"{sim_file}: {len(numbers)} entries but {len(np.unique(numbers))} distinct event numbers; "
            f"written more than once: {duplicated[:10]}. With the sw:pr-8 image this happens when "
            "ddsim runs with one thread: set threads to 2 or more.")
    return len(numbers)
