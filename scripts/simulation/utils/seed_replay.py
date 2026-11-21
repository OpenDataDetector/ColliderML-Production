"""Replay a run's historical digitisation seed string.

The run_stage / cli_utils seed string is ``{dataset}_{version}_run{N}``, so a new
version (v23) would draw new digitisation random numbers (Gaussian charge
smearing in the geometric digitisation) and change tracker_hits and truth_tracks
on every event. To make v23 differ from v22 only by the intended changes, a
config can name the seed string each run had in the version it replaces:

    digitization_seed_replay:
      - {runs: [0, 15],    seed_string: "ttbar_v20_run{run}"}
      - {runs: [16, 1562], seed_string: "ttbar_v22_run{run}"}

Ranges are inclusive and must not overlap. A run outside every range is an
error (never a silent fall-back to the version seed). Pure python, no ACTS, so
it is unit-testable outside the container.
"""
from __future__ import annotations


class SeedReplayError(ValueError):
    pass


def _ranges(replay):
    if not isinstance(replay, list) or not replay:
        raise SeedReplayError("digitization_seed_replay must be a non-empty list of {runs, seed_string}")
    out = []
    for i, entry in enumerate(replay):
        try:
            lo, hi = (int(x) for x in entry["runs"])
            template = str(entry["seed_string"])
        except (KeyError, TypeError, ValueError) as e:
            raise SeedReplayError(f"digitization_seed_replay[{i}] malformed: {entry!r}") from e
        if lo > hi:
            raise SeedReplayError(f"digitization_seed_replay[{i}]: runs {lo} > {hi}")
        if "{run}" not in template:
            raise SeedReplayError(f"digitization_seed_replay[{i}]: seed_string lacks '{{run}}'")
        out.append((lo, hi, template))
    out.sort()
    for (a_lo, a_hi, _), (b_lo, b_hi, _) in zip(out, out[1:]):
        if b_lo <= a_hi:
            raise SeedReplayError(f"digitization_seed_replay: ranges [{a_lo},{a_hi}] and [{b_lo},{b_hi}] overlap")
    return out


def replay_seed_string(replay, run) -> str:
    """Return the historical seed string for ``run`` (an int or a digit string)."""
    try:
        run_i = int(str(run))
    except ValueError as e:
        raise SeedReplayError(f"seed replay needs a numeric run (--output-subdir), got {run!r}") from e
    if str(run_i) != str(run).strip():
        raise SeedReplayError(f"seed replay needs a plain run number, got {run!r}")
    for lo, hi, template in _ranges(replay):
        if lo <= run_i <= hi:
            return template.format(run=run_i)
    raise SeedReplayError(f"run {run_i} is not covered by digitization_seed_replay")
