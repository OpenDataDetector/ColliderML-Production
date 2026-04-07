"""
Compute cost estimation and the canonical credit unit.

  1 credit = 1 node-hour on Perlmutter (cpu queue)
          ≈ 100 hard-scatter events @ pu=0
          ≈  20 hard-scatter events @ pu=200

These numbers are empirical estimates from the test runs in CLAUDE.md and
should be recalibrated as real user jobs run. Overestimates are conservative
(better to refund than charge more after the fact).
"""

from __future__ import annotations

# Seconds per event at pu=0 for each channel.
# Dominated by DDSim (Geant4) which is ~90% of total wall time.
_BASE_SECONDS_PER_EVENT = {
    "higgs_portal": 60.0,
    "ttbar": 90.0,
    "zmumu": 30.0,
    "zee": 30.0,
    "diphoton": 30.0,
    "jets": 45.0,
    "susy_gmsb": 60.0,
    "hidden_valley": 60.0,
    "zprime": 60.0,
    "single_muon": 5.0,
}

# MadGraph overhead per request (init + generation, independent of events).
# Only applies to channels that use MadGraph.
_MADGRAPH_OVERHEAD_SECONDS = {
    "ttbar": 300.0,
    "susy_gmsb": 300.0,
    "zprime": 300.0,
    "hidden_valley": 300.0,
    # Pythia-only channels have no overhead.
    "higgs_portal": 0.0,
    "zmumu": 0.0,
    "zee": 0.0,
    "diphoton": 0.0,
    "jets": 0.0,
    "single_muon": 0.0,
}


def estimate_node_hours(channel: str, events: int, pileup: int) -> float:
    """Return the estimated node-hours for a request.

    The pileup scaling (1 + pileup/50) comes from DDSim runtime with
    minbias overlay being roughly linear in the number of overlaid events.
    """
    base = _BASE_SECONDS_PER_EVENT.get(channel, 60.0)
    overhead = _MADGRAPH_OVERHEAD_SECONDS.get(channel, 0.0)
    pileup_factor = 1.0 + (pileup / 50.0)

    seconds = overhead + (base * events * pileup_factor)
    return round(seconds / 3600.0, 3)


def estimate_completion_seconds(channel: str, events: int, pileup: int) -> int:
    """Best-guess queue + run time in seconds, for user-facing ETAs.

    Assumes the debug queue (<30 min) for small jobs and regular otherwise.
    """
    node_hours = estimate_node_hours(channel, events, pileup)
    run_seconds = int(node_hours * 3600)
    # Add a generous queue buffer: 5 min for debug, 30 min for regular.
    queue_buffer = 5 * 60 if run_seconds < 25 * 60 else 30 * 60
    return run_seconds + queue_buffer
