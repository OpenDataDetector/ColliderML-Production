# Sources and licences

The files of this recipe are under the MIT licence in `LICENSE`, except the patches. Each patch in `patches/` is a modification
of the project it applies to and carries that project's licence (table in
`patches/README.md`); the licence texts are in `LICENSES/`. The image contains the software
below, each under its own licence.

## Built from source in this image

| Component | Repository | Commit | Licence | Patch |
|---|---|---|---|---|
| PandoraSDK | https://github.com/OpenDataDetector/PandoraSDK (fork of PandoraPFA/PandoraSDK) | optimised: `0390e905e3c501a08ed1a65eada4beb9cfa18b6b` (tag `colliderml-reco-20261127`); reference: `d8c6466071ff3300c86c1049454d295e0288030a` | GPL-3.0-or-later | none |
| LCContent | https://github.com/OpenDataDetector/LCContent (fork of PandoraPFA/LCContent) | optimised: `dc6e6a304357a95fd97e298db72b1db8c480bfc8` (tag `colliderml-reco-20261127`); reference: `3948e46055bed9ab40df3a3d20b8d776d10053e3` | GPL-3.0-or-later | reference: `lccontent_reference.patch` |
| k4GaudiPandora | https://github.com/OpenDataDetector/k4GaudiPandora (fork of key4hep/k4GaudiPandora) | optimised: `1f04be1a4df4f11e01875893097bcedcfa626e67` (tag `colliderml-reco-20261127`); reference: `faf0d672bc64aa665dcfda51649e766c220664b9` | Apache-2.0 | reference: `k4gaudipandora_reference.patch` |
| k4DetectorPerformance | https://github.com/OpenDataDetector/k4DetectorPerformance (fork of key4hep/k4DetectorPerformance) | `d684e12e2562e1df0d5e49babed9fdab688f48c3` | Apache-2.0 | none |
| k4Reco (GaudiTrkUtils only) | https://github.com/key4hep/k4Reco | `4be5a1b02e2a6f63cfcfb4377be68c54f399cbed` (v0.3.0) | Apache-2.0 | `k4reco_gauditrkutils_only.patch` |
| k4ODD | https://github.com/OpenDataDetector/k4ODD | `d92d2056f14ebde9b9c3fc885d4363869fd08cf5` | Apache-2.0 | `k4odd_calo_last_layer.patch`, `k4odd_ecal_gap_correction_off.patch` |
| OpenDataDetector | https://gitlab.cern.ch/azaborow/OpenDataDetector (fork of acts/OpenDataDetector) | `a1f2ca18c17c09f805e6e6b12830eb6c9fff7cb7` | MPL-2.0 | `odd_calo_layerdata.patch` |

The licence text of each component is the `LICENSE` file in its source tree inside the image:
`/opt/pandora-stack/src/<name>`, `/opt/k4reco-src`, `/opt/k4ODD`, `/opt/odd-src`.

## Python packages added with pip

| Package | Version | Licence |
|---|---|---|
| pyarrow | 26.0.1 | Apache-2.0 |
| polars, polars-runtime-32 | 2.1.0 | MIT |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |

## Base image

`ghcr.io/key4hep/key4hep-sim-reco-ubuntu24@sha256:5b3a56c63a3337eca22bbfc14cfb3239e1049037a8734322e70ae8580ce96911`,
published by the Key4hep project: Ubuntu 24.04 with a spack installation of the Key4hep stack
(`key4hep-stack@2025-12-01`, several hundred packages) under `/opt/spack`. This image
redistributes those layers unchanged, apart from one file of k4FWCore
(`k4fwcore_gaudi40_eventloopmgr.patch`) and the journal mode of the Gaudi `*.confdb2` files.

`spack find` inside the image lists every package and version. The spack recipes, which
name the source location of each package, are in the image (`spack repo list`). Licences of
the packages this image calls directly, as stated by their projects: ROOT (LGPL-2.1), Geant4
(Geant4 Software License), Gaudi (Apache-2.0), DD4hep (LGPL-3.0), podio, EDM4hep and k4FWCore
(Apache-2.0), jemalloc (BSD-2-Clause).

The base image is referenced by digest in the `org.opencontainers.image.base.name` and
`org.opencontainers.image.base.digest` labels of this image. Its recipe and the package
recipes it is built from are public:

- https://github.com/key4hep/key4hep-images: the Containerfiles of the Key4hep images;
- https://github.com/key4hep/key4hep-spack: the Key4hep spack recipes and the
  `key4hep-stack` environment;
- https://github.com/spack/spack and https://github.com/spack/spack-packages: spack and its
  builtin package recipes, which name the upstream source of each package.

Source of the base-image binaries: we ship none. For a spack package, `spack info <name>`
in the image gives the source URL and `spack fetch <name>@<version>` downloads it; for an
Ubuntu package, `apt-get source <name>` after enabling the `deb-src` entries of Ubuntu 24.04.

Not verified by us: that these routes give the exact sources of every binary in the base
image, the licences of the base packages one by one, and the terms of each Geant4 data set.
The base also holds its own builds of PandoraSDK and LCContent (GPL-3.0-or-later), which this
reconstruction does not use, and the iLCSoft packages KalTest, DDKalTest and LCIO, which
GaudiTrkUtils links.

## GPL-3.0 and the reconstruction built here

PandoraSDK and LCContent are GPL-3.0-or-later. `libk4GaudiPandoraPlugins.so` links both, and the GNU
Scientific Library of the base image (GPL-3.0-or-later), so the reconstruction libraries
built by this recipe are a combined work distributed under GPL-3.0 terms. The image as a
whole is an aggregate of separately licensed works. Apache-2.0 code may be combined into a GPL-3.0 work. The ODD geometry plugin (MPL-2.0)
is a separate library. The corresponding source of the libraries built here is in the image:
the pinned commits with the patches applied under `/opt/pandora-stack/src`, the patches under
`/opt/colliderml-reco/patches`, the build scripts under `/opt/colliderml-reco/steps`, and the
licence texts under `/opt/colliderml-reco/LICENSES`. The same files are in this directory.
