#!/usr/bin/env python3
"""Port Daniel's measurementIDs patch onto Paul's arrow branch's
RootTrackSummaryWriter (github.com/murnanedaniel/acts@feat/root-writer-measurement-ids,
commit c12ad97).

A blind cherry-pick fails because the fork was cut from an older `main` whose
``majorityParticleId`` representation differs from Paul's branch (single branch
vs five split ``majorityParticleId_*`` branches). So we apply the substantive
change — a per-track ``measurementIDs`` branch holding each measurement state's
uncalibrated source-link index — at anchor lines that are identical in both
trees. Each anchor must match exactly once or we abort, so silent drift fails
the build instead of producing a half-patched writer.

This is what gives the v1 (convert_all-from-ROOT) track path its ``hit_ids`` so
it can be cross-checked against the native Arrow track writer.
"""

from __future__ import annotations

import sys

SRC = "/opt/acts-arrow-src"
HPP = f"{SRC}/Examples/Io/Root/include/ActsExamples/Io/Root/RootTrackSummaryWriter.hpp"
CPP = f"{SRC}/Examples/Io/Root/src/RootTrackSummaryWriter.cpp"


def patch(path: str, anchor: str, insert: str, *, after: bool = True) -> None:
    """Insert `insert` immediately before/after the unique line containing `anchor`."""
    with open(path) as f:
        text = f.read()
    n = text.count(anchor)
    if n != 1:
        sys.exit(f"PATCH ABORT: anchor {anchor!r} found {n}x (expected 1) in {path}")
    if insert.strip() in text:
        print(f"  already patched: {insert.strip()[:50]}…")
        return
    # operate line-wise so we preserve the anchor line's own indentation/newline
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if anchor in line:
            at = i + 1 if after else i
            lines.insert(at, insert if insert.endswith("\n") else insert + "\n")
            break
    with open(path, "w") as f:
        f.write("".join(lines))
    print(f"  patched {path.split('/')[-1]}: +{insert.strip()[:60]}")


# 1. header member
patch(
    HPP,
    "std::vector<std::vector<std::uint32_t>> m_outlierLayer;",
    "  /// The measurement IDs for each track\n"
    "  std::vector<std::vector<std::uint64_t>> m_measurementIDs;",
)

# 2. cpp include (matches the fork's placement before TruthMatching.hpp)
patch(
    CPP,
    '#include "ActsExamples/EventData/TruthMatching.hpp"',
    '#include "ActsExamples/EventData/IndexSourceLink.hpp"',
    after=False,
)

# 3. register the output branch
patch(
    CPP,
    'm_outputTree->Branch("outlierLayer", &m_outlierLayer);',
    '  m_outputTree->Branch("measurementIDs", &m_measurementIDs);',
)

# 4. declare the per-track accumulator inside the event loop
patch(
    CPP,
    "      std::vector<std::uint32_t> outlierLayer;",
    "      std::vector<std::uint64_t> measurementIDs;",
)

# 5. collect the source-link index for each measurement state
patch(
    CPP,
    "          measurementLayer.push_back(layer);",
    "          auto sl = state.getUncalibratedSourceLink()\n"
    "                        .template get<IndexSourceLink>();\n"
    "          measurementIDs.push_back(sl.index());",
)

# 6. move the accumulator into the member vector
patch(
    CPP,
    "      m_outlierLayer.push_back(std::move(outlierLayer));",
    "      m_measurementIDs.push_back(std::move(measurementIDs));",
)

# 7. clear between events
patch(
    CPP,
    "  m_outlierLayer.clear();",
    "  m_measurementIDs.clear();",
)

print("measurementIDs patch applied OK")
