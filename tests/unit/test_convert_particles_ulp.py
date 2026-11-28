"""D8: the Release 1 truth join (convert_particles.py) dropped a particle whose momentum component sits on a
float32 rounding tie. The second pass matches leftovers within 2 float32 ulp, one-to-one."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "postprocessing"))
cp = pytest.importorskip("convert_particles", reason="needs pyedm4hep, h5py")

COLS = ["vx", "vy", "vz", "px", "py", "pz"]


def _edm(n=5):
    rng = np.random.default_rng(1)
    d = {c: rng.normal(size=n).astype(np.float64) for c in COLS}
    d.update(PDG=np.arange(n) + 11, mass=np.ones(n), charge=np.zeros(n, np.float32), time=np.zeros(n, np.float32),
             created_in_simulation=np.zeros(n, bool))
    return pd.DataFrame(d)


def _digi(edm, bump=None):
    d = edm[COLS].astype(np.float32).copy()
    if bump is not None:
        i, c, k = bump
        d.loc[i, c] = np.float32(d.loc[i, c]) + k * np.spacing(np.float32(abs(d.loc[i, c])))
    d["event_id"] = 0
    d["vertex_primary"] = np.arange(len(d)) + 1
    d["perigee_d0"] = np.arange(len(d), dtype=np.float64) * 0.1
    d["perigee_z0"] = np.arange(len(d), dtype=np.float64) * 0.2
    return d


def _run(edm, digi):
    return cp.process_event_for_particles(0, 0, "unused", digi_particles_df=digi, preloaded_particles_df=edm)


def test_exact_rows_unchanged_and_tie_added():
    edm = _edm()
    ref = _run(edm, _digi(edm))                        # all exact
    assert ref["particle_id"].tolist() == [0, 1, 2, 3, 4]
    out = _run(edm, _digi(edm, bump=(2, "pz", 1)))      # particle 2: pz one float32 step away (the D8 tie)
    assert out["particle_id"].tolist() == [0, 1, 2, 3, 4]
    pd.testing.assert_frame_equal(out.reset_index(drop=True), ref.reset_index(drop=True))
    # the old exact join alone would have dropped it
    digi = _digi(edm, bump=(2, "pz", 1))
    e = edm.copy(); e["particle_id"] = np.arange(len(e), dtype="uint32")
    for c in COLS:
        e[c] = e[c].astype("float32")
    assert 2 not in pd.merge(e, digi, on=COLS, how="inner")["particle_id"].tolist()


def test_three_ulp_not_matched():
    edm = _edm()
    out = _run(edm, _digi(edm, bump=(2, "px", 3)))
    assert out["particle_id"].tolist() == [0, 1, 3, 4]


def test_ambiguous_not_matched():
    edm = _edm(3)
    digi = _digi(edm, bump=(1, "pz", 1))
    dup = digi.iloc[[1]].copy()
    dup["pz"] = np.float32(edm.loc[1, "pz"]) - np.spacing(np.float32(abs(edm.loc[1, "pz"])))
    digi = pd.concat([digi, dup], ignore_index=True)    # two ACTS candidates within 2 ulp of particle 1
    out = _run(edm, digi)
    assert out["particle_id"].tolist() == [0, 2]


def test_ordinal_across_zero():
    a = np.array([-np.float32(1e-45), 0.0, np.float32(1e-45)], dtype=np.float32)
    assert np.diff(cp._float32_ordinal(a)).tolist() == [1, 1]
