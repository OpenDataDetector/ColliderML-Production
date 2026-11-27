# Patches

Each patch applies to the commit pinned in `../sources.env` and carries the licence of the
project it changes. `SHA256SUMS` is checked before any patch is applied.

The Pandora libraries are built in one of two ways (`PANDORA_VARIANT` in `../Containerfile`):

- `optimised` (default): no patches. PandoraSDK, LCContent and k4GaudiPandora are fetched at
  the tag `colliderml-reco-20261127` of the OpenDataDetector forks (`*_OPT_COMMIT` in
  `../sources.env`). Each tag is a branch on top of the commit the reference variant uses, so
  `git log <reference commit>..colliderml-reco-20261127` in the fork shows every change.
- `reference`: the commits `*_COMMIT` with the two `*_reference` patches.

The optimised sources contain, in PandoraSDK: faster object management (`CaloHitList` as
`std::vector`, flat `OrderedCaloHitList`, pooled cluster-fit buffers, constant-time geometry
and cluster-list lookups). In LCContent: spatial indices (k-d trees, bounding-volume
hierarchies, grids) and AVX2 kernels for the clustering, topological-association,
fragment-removal and photon algorithms that dominate at 200 pileup, and bounded memory in
`SoftClusterMerging`. `PANDORA_STRICT_BYTE_IDENTICAL` selects the code paths written to keep
the floating-point operation order of the unpatched code. The LCContent tag also keeps two
things the output depends on: the 100 helix sampling points of the layer-crossing count, and,
in the algorithms where our validation found a difference, the moment at which a cluster is
first asked for its cached photon id and track comparison energy (these caches are not reset
when track associations change). In k4GaudiPandora: calorimeter hits keyed by object
(cluster-to-cell links pointed at wrong cells when collections had no ID yet), the MC-truth
energy map fix, `DDCaloDigi` and hit-creator speedups, and three switches, off by default:
`K4ODD_CALODIGI_HISTOGRAMS` (fill the DDCaloDigi histograms),
`K4ODD_RESET_RANDGAUSS_PER_EVENT` (reset the Gaussian generator state at each event),
`K4ODD_FORCE_MCPARTICLES` (pass MCParticles to Pandora even when the settings do not use them).
Its `fillECALGaps` shortcut is not reached in this image, because the ECAL gap correction is
switched off (`k4odd_ecal_gap_correction_off.patch`).

| Patch | Applies to | Licence | Content |
|---|---|---|---|
| `lccontent_reference.patch` | LCContent `3948e46` | GPL-3.0-or-later | Reference variant: bounded memory in `SoftClusterMerging` only. |
| `k4gaudipandora_reference.patch` | k4GaudiPandora `faf0d67` | Apache-2.0 | Reference variant: calorimeter hits keyed by object and the MC-truth energy map fix only. |
| `k4reco_gauditrkutils_only.patch` | k4Reco `4be5a1b` (v0.3.0) | Apache-2.0 | Build only the `GaudiTrkUtils` library and export its CMake config. |
| `k4odd_calo_last_layer.patch` | k4ODD `d92d205` | Apache-2.0 | `DDCaloDigi` layer edges 49 (ECAL) and 37 (HCAL), so the last layer of each calorimeter is calibrated. |
| `k4odd_ecal_gap_correction_off.patch` | k4ODD `d92d205`, after the patch above | Apache-2.0 | `ECALGapCorrection = 0` in the digitisation and reconstruction options. ODD has no dead gaps between ECAL cells, and the correction reads the hits through expired handles (key4hep/k4GaudiPandora#68). |
| `odd_calo_layerdata.patch` | OpenDataDetector `a1f2ca1` | MPL-2.0 | Per-layer radiation and interaction lengths in `LayeredCalorimeterData` (inner values were overwritten per slice). |
| `k4fwcore_gaudi40_eventloopmgr.patch` | k4FWCore 1.3 in the base image | Apache-2.0 | Tolerate the `EventLoopMgr.Warnings` property that Gaudi 40 removed. |

The Pandora sources as built stay in the image under `/opt/pandora-stack/src`, next to the
patches in `/opt/colliderml-reco/patches`.
