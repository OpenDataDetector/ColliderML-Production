#!/usr/bin/env python3
import argparse, pathlib, contextlib, acts, acts.examples
from acts.examples.simulation import (
    addParticleGun,
    MomentumConfig,
    EtaConfig,
    ParticleConfig,
    addPythia8,
    addFatras,
    addGeant4,
    ParticleSelectorConfig,
    addDigitization
)
from acts.examples.reconstruction import addSpacePointsMaking

from common import getOpenDataDetectorDirectory
from acts.examples.odd import getOpenDataDetector

parser = argparse.ArgumentParser(description="Full chain with the OpenDataDetector")

parser.add_argument("--events", "-n", help="Number of events", type=int, default=1000)
parser.add_argument("--pileup", "-p", help="Number of pileup events", type=int, default=0)

args = parser.parse_args()

u = acts.UnitConstants
geoDir = getOpenDataDetectorDirectory()
jobName = f"HSS_output_PU{args.pileup}"
outputDir = pathlib.Path.cwd() / jobName

oddMaterialMap = geoDir / "data/odd-material-maps.root"
#oddFieldMap = geoDir / "data/odd-field.root"
oddDigiConfig = geoDir / "config/odd-digi-smearing-config.json"
# oddDigiConfig = geoDir / "config/odd-digi-geometric-config.json"
oddSpacepointSel = geoDir / "config/odd-sp-config.json"
oddMaterialDeco = acts.IMaterialDecorator.fromFile(oddMaterialMap)

detector, trackingGeometry, decorators = getOpenDataDetector(
    geoDir, mdecorator=oddMaterialDeco
)
field = acts.ConstantBField(acts.Vector3(0.0, 0.0, 2.0 * u.T))
rnd = acts.examples.RandomNumbers(seed=411)

ttbar_pu200=False

ma = 55
m_ma=ma-0.5
p_ma=ma+0.5
ctau = 100
width = (1.9732699E-13)/ctau

# TODO Geant4 currently crashes with FPE monitoring
with acts.FpeMonitor():
    s = acts.examples.Sequencer(
        events=args.events,
        numThreads=8,
        logLevel=acts.logging.INFO,
        outputDir=str(outputDir))

    addPythia8(
        s,
        hardProcess=["Higgs:useBSM = on",
                     "HiggsBSM:gg2H2 = on",
                     "35:m0 = 125.0",
                     '35:mWidth = 0.00407',
                     '35:doForceWidth = on',
                     "35:onMode = off",
                     "35:onIfMatch = 36 36",

                     "36:oneChannel = 1 1.0 101 5 -5",
                     "36:m0=%.1f" % ma,
                     "36:mMin=%.1f" %m_ma,
                     "36:mMax = %.1f" %p_ma,
                     "36:mWidth= %.7g" % width,
                     "36:tau0 = %.1f" % ctau,
                     "ParticleDecays:limitTau0 = off",
                     "ParticleDecays:tau0Max = 100000.0",
                    ],
        npileup=args.pileup,
        vtxGen=acts.examples.GaussianVertexGenerator(
            stddev=acts.Vector4(
                0.0125 * u.mm, 0.0125 * u.mm, 55.5 * u.mm, 5.0 * u.ns
            ),
            mean=acts.Vector4(0, 0, 0, 0),
        ),
        rnd=rnd,
        outputDirCsv=outputDir,
        outputDirRoot=outputDir,
        printParticles=False,
    )
  
    addFatras(
        s,
        trackingGeometry,
        field,
        preSelectParticles=ParticleSelectorConfig(
#            eta=(-3.0, 3.0),
            pt=(1.0 * u.GeV, None),
            removeNeutral=True,
        ),
        outputDirCsv=outputDir,
        rnd=rnd,
    )

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=oddDigiConfig,
        outputDirCsv=outputDir,
        rnd=rnd,
    )

    # make spacepoints from the measurements
    addSpacePointsMaking(s, trackingGeometry, oddSpacepointSel)

    # add spacepoint writer
    s.addWriter(
        acts.examples.CsvSpacepointWriter(
            inputSpacepoints = "spacepoints",
            outputDir = str(outputDir),
            level = acts.logging.INFO
        )
    )

    s.run()
