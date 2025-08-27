#!/usr/bin/env python3

import pathlib

import acts
import acts.examples
from acts.examples.simulation import (
    ParticleSelectorConfig,
    addSimParticleSelection,
    addDigitization,
    addDigiParticleSelection,
)
from acts.examples.reconstruction import (
    SeedingAlgorithm,
    SeedFinderConfigArg,
    addSeeding,
    CkfConfig,
    addCKFTracks,
    TrackSelectorConfig,
    addAmbiguityResolution,
    AmbiguityResolutionConfig,
    addVertexFitting,
    VertexFinder,
)
from acts.examples.odd import getOpenDataDetector, getOpenDataDetectorDirectory
import acts.examples.edm4hep
from acts.examples.podio import PodioReader

u = acts.UnitConstants


inputFile = "/Users/andreas/Downloads/edm4hep_ttbar.root"
outputDir = pathlib.Path(__file__).parent / "colliderml_output"
geoDir = getOpenDataDetectorDirectory()
actsDir = pathlib.Path(__file__).parent

oddDigiConfig = actsDir / "Examples/Configs/odd-digi-smearing-config.json"
oddSeedingSel = actsDir / "Examples/Configs/odd-seeding-config.json"
oddMaterialMap = geoDir / "data/odd-material-maps.root"
oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

detector = getOpenDataDetector(odd_dir=geoDir, materialDecorator=oddMaterialDeco)
trackingGeometry = detector.trackingGeometry()
decorators = detector.contextDecorators()
field = detector.field
field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 3.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=42)

s = acts.examples.Sequencer(
    events=10,
    skip=0,
    numThreads=-1,
    outputDir=str(outputDir),
)

s.addReader(
    PodioReader(
        level=acts.logging.DEBUG,
        inputPath=str(inputFile),
        outputFrame="events",
        category="events",
    )
)

edm4hepReader = acts.examples.edm4hep.EDM4hepSimInputConverter(
    inputFrame="events",
    inputSimHits=[
        "PixelBarrelReadout",
        "PixelEndcapReadout",
        "ShortStripBarrelReadout",
        "ShortStripEndcapReadout",
        "LongStripBarrelReadout",
        "LongStripEndcapReadout",
    ],
    outputParticlesGenerator="particles_generated",
    outputParticlesSimulation="particles_simulated",
    outputSimHits="simhits",
    outputSimVertices="vertices_truth",
    dd4hepDetector=detector,
    trackingGeometry=trackingGeometry,
    sortSimHitsInTime=False,
    particleRMax=1080 * u.mm,
    particleZ=(-3030 * u.mm, 3030 * u.mm),
    particlePtMin=150 * u.MeV,
    level=acts.logging.DEBUG,
)
s.addAlgorithm(edm4hepReader)

s.addWhiteboardAlias("particles", edm4hepReader.config.outputParticlesSimulation)

addSimParticleSelection(
    s,
    ParticleSelectorConfig(),
)

addDigitization(
    s,
    trackingGeometry,
    field,
    digiConfigFile=oddDigiConfig,
    outputDirRoot=outputDir,
    rnd=rnd,
)

def make_geoid(vol=None, lay=None):
    geoid = acts.GeometryIdentifier()
    if vol is not None:
        geoid.volume = vol
    if lay is not None:
        geoid.layer = lay
    return geoid

measurementCounter = acts.examples.ParticleSelector.MeasurementCounter()
# At least 3 hits in the pixels
measurementCounter.addCounter(
    [
        make_geoid(16),
        make_geoid(17),
        make_geoid(18),
    ],
    3,
)

addDigiParticleSelection(
    s,
    ParticleSelectorConfig(
        # we are only interested in the hard scatter vertex
        #primaryVertexId=(1, 2),
        rho=(0.0, 24 * u.mm),
        absZ=(0.0, 1.0 * u.m),
        eta=(-3.0, 3.0),
        # using something close to 1 to include for sure
        pt=(0.999 * u.GeV, None),
        measurements=(6, None),
        removeNeutral=True,
        removeSecondaries=False,
        nMeasurementsGroupMin=measurementCounter,
    ),
)

addSeeding(
    s,
    trackingGeometry,
    field,
    seedingAlgorithm=SeedingAlgorithm.Default,
    particleHypothesis=acts.ParticleHypothesis.pion,
    seedFinderConfigArg=SeedFinderConfigArg(
        r=(33 * u.mm, 200 * u.mm),
        # kills efficiency at |eta|~2
        deltaR=(1 * u.mm, 300 * u.mm),
        collisionRegion=(-250 * u.mm, 250 * u.mm),
        z=(-2000 * u.mm, 2000 * u.mm),
        maxSeedsPerSpM=3,
        sigmaScattering=5,
        radLengthPerSeed=0.1,
        minPt=0.5 * u.GeV,
        impactMax=3 * u.mm,
    ),
    initialSigmas = [
        1 * u.mm,
        1 * u.mm,
        1 * u.degree,
        1 * u.degree,
        0.1 / u.GeV,
        1 * u.ns,
    ],
    initialSigmaQoverPt=0.1 * u.e / u.GeV,
    initialSigmaPtRel = 0.1,
    initialVarInflation = [1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
    geoSelectionConfigFile=oddSeedingSel,
    outputDirRoot=outputDir,
)

addCKFTracks(
    s,
    trackingGeometry,
    field,
    trackSelectorConfig=TrackSelectorConfig(
        pt=(0.7 * u.GeV, None),
        absEta=(None, 3.5),
        nMeasurementsMin=6,
        maxHolesAndOutliers=3,
    ),
    ckfConfig=CkfConfig(
        chi2CutOffMeasurement=15.0,
        chi2CutOffOutlier=25.0,
        numMeasurementsCutOff=1,
        seedDeduplication=True,
        stayOnSeed=True,
    ),
    twoWay=True,
    outputDirRoot=outputDir,
)

addAmbiguityResolution(
    s,
    config=AmbiguityResolutionConfig(
        maximumSharedHits=3,
        maximumIterations=1000000,
        nMeasurementsMin=6,
    ),
    outputDirRoot=outputDir,
)

addVertexFitting(
    s,
    field,
    vertexFinder=VertexFinder.AMVF,
    outputDirRoot=outputDir,
)

s.run()
