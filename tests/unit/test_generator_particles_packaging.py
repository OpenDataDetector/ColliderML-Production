"""generator_particles packaging wiring: published as truth/generator_particles, no id remap, no unit
conversion; particles_full can write the generator table alone."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "postprocessing"))
pytest.importorskip("scipy")
pkg = pytest.importorskip("package_native_parquet")
cpf = pytest.importorskip("convert_particles_full")
from utils.particle_id_map import ID_COLUMNS  # noqa: E402


def test_layout_and_no_conversion():
    assert pkg.OBJECT_LAYOUT["generator_particles"] == ("truth", "generator_particles")
    assert pkg.OBJECT_LAYOUT["particles"] == ("truth", "particles"), "the published particles table is unchanged"
    assert "generator_particles" not in ID_COLUMNS, "ids are already the MCParticle index"
    assert "generator_particles" not in pkg.TIME_COLUMNS, "times are already ns"


def _args(**kw):
    base = dict(config=None, events=None, sim_input_name=None, native_particles=None,
                generator_particles=None, particles_full_table=None)
    base.update(kw)
    return cpf.load_config(Namespace(**base))


def test_generator_only_option():
    assert _args().particles_full_table is True
    a = _args(generator_particles="hard_scatter", particles_full_table=False)
    assert a.particles_full_table is False and a.generator_particles == "hard_scatter"
    with pytest.raises(ValueError):
        _args(particles_full_table=False)            # nothing to write
