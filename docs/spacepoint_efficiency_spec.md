# Spacepoint efficiency metric specification

This document defines what "good spacepoints" means for the ColliderML
strip detectors. It is the contract the digi/reco pipeline must satisfy
before downstream ML tracking work can use spacepoints. Locked 2026-04-27.

## Scope

- **Strip spacepoints only.** Pixel measurements in the ColliderML/ODD
  geometry are already 3D clusters and do not flow through
  `acts.examples.SpacePointMaker`. They are out of scope for this metric.
- **Per-run, per-event, microaveraged across events.**

## Primary metric: strip spacepoint efficiency

```
efficiency  =  N_correct_spacepoints  /  N_true_cluster_pairs
```

### Numerator — `N_correct_spacepoints`

A spacepoint counts as *correct* if the two source-link measurements
share at least one truth particle.

- Source: `spacepoints.root` (rows = strip pairs, with `measurement_id`,
  `measurement_id_2`).
- Truth: join measurements via `measurement_id` to `measurements.root`'s
  `particles` branch (`std::vector<std::vector<uint32_t>>` — one list per
  measurement of contributing truth-particle barcodes).
- A pair is *correct* iff `particles[m1] ∩ particles[m2] ≠ ∅`.
- We compute this in post-processing rather than relying on the writer's
  `fake` flag, because the flag depends on the writer being given a
  populated `MeasurementParticlesMap` — which on the v1 baseline appears
  to be missing/broken (100% `fake==true`).

### Denominator — `N_true_cluster_pairs`

For each `(event, truth_particle p, layer ℓ)`, count `1` if `p` hit *both*
faces of any stereo strip module on `ℓ`:

- Particle `p` has at least one measurement with `(vol_id=v, layer_id=ℓ, extra_id=1)`, *and*
- Particle `p` has at least one measurement with `(vol_id=v, layer_id=ℓ, extra_id=2)`,
- where `v` is the same in both cases (same volume).

Sum across all particles and stereo layers in the event.

- Source: `measurements.root` only (no need for simhits).
- For volumes where the geometry uses a different stereo encoding (e.g.
  vol 29 in v1 has `extra=0` for all surfaces), we treat them as having
  *no* stereo pairs — so they contribute `0` to the denominator and any
  built spacepoints in those volumes contribute purely to the fake rate.
  vol 29 should be removed from the spacepoint geometry-selection JSON.

## Secondary metric: fake rate

```
fake_rate  =  (N_total_spacepoints − N_correct_spacepoints)  /  N_total_spacepoints
```

A high efficiency with a high fake rate is unacceptable — both must be
satisfied to consider Phase A done.

## Particle-scope variants

Report **two** efficiencies and **two** fake rates, with identical
formulas but different particle filters on the denominator (and on the
"is correct" check):

1. **All-particles**: every truth particle counts. Used as the baseline
   reproducibility number.
2. **Primary + pT > 1 GeV**: only truth particles that are
   - "primary" in the ACTS barcode sense (`vertex_secondary == 0`,
     `generation == 0`, `subparticle == 0`), *and*
   - `sqrt(px² + py²) > 1.0 GeV/c`,
   contribute to the denominator. For the numerator, a paired spacepoint
   counts as correct only if at least one shared particle satisfies the
   filter. This is the "physics-relevant" efficiency.

Particle properties come from `particles.root` joined to measurements
via the truth-particle barcode.

## Reporting granularity

For each variant (all-particles and primary+pT>1GeV):

- **Headline**: efficiency and fake rate microaveraged over all events
  (sum numerators and denominators, then divide).
- **Per-(vol, layer)**: same two metrics broken out per layer in the
  spacepoint geometry-selection JSON. Diagnostic — bad layers stand out.
- **Per-event distribution**: efficiency histogram over events, to
  sanity-check uniformity.

Aggregation method: **microaverage** (sum-then-divide, not
per-event-then-average).

## Targets

| Metric                          | Target |
|---------------------------------|--------|
| Headline efficiency (both variants) | **≥ 99 %** |
| Headline fake rate (both variants)  | **≤ 1 %**  |
| Per-(vol, layer) efficiency         | **≥ 95 %** for every (vol, layer) in JSON |

If any per-layer efficiency is below 95 %, that's a layer-specific
debugging task before the headline is considered green.

## Out of scope (for now)

- **Pixel "spacepoints"** (already 3D — handled by tracker_hits parquet).
- **Differential efficiency vs η, pT, displaced vs prompt** — useful
  later, not needed to gate Phase A.
- **Timing performance** — separate concern.

## Implementation notes

- The analysis script reads three ROOT files per run: `spacepoints.root`,
  `measurements.root`, `particles.root`.
- All three must come from the **same digi run**. Joining across
  different runs is undefined behaviour.
- Particle barcode decoding follows ACTS's `Barcode.hpp` layout:
  vertex_primary (12) | vertex_secondary (12) | particle (16) |
  generation (8) | subparticle (16) — packed into 64 bits.
  In v1 measurements.root the `particles` field is `vector<vector<uint32_t>>`,
  which stores the lower-32-bit truncation of the barcode; the join key
  to `particles.root` is whatever that truncation collides with on the
  particles side. Verify against a single event before scaling up.
- Output: a single text/JSON report per run with headline + per-layer +
  per-event breakdown. No plots required for Phase A; CSV/JSON is enough.
