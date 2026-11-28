"""Codex round 7 finding 5: a kill at any point of a packaged-shard promotion is recovered to one complete
generation with its matching manifest (utils/promotion_journal.py)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "postprocessing"))
from utils import promotion_journal as pj  # noqa: E402


class Kill(Exception):
    pass


def _setup(tmp_path, previous: bool):
    out = tmp_path / "parquet"; out.mkdir()
    man = tmp_path / "chunk0_inputs.json"
    names = ["a.parquet", "b.parquet", "c.parquet"]
    if previous:
        for n in names:
            (out / n).write_text("old " + n)
        man.write_text(json.dumps({"gen": "old"}))
    moves = []
    for n in names:
        t = out / f".{n}.tok.tmp"
        t.write_text("new " + n)
        moves.append((t, out / n))
    return out, man, names, moves


def _state(out, man, names):
    gens = {(out / n).read_text().split()[0] if (out / n).exists() else None for n in names}
    m = json.loads(man.read_text())["gen"] if man.exists() else None
    return gens, m


@pytest.mark.parametrize("previous", [True, False])
def test_kill_after_each_step_recovers_one_generation(tmp_path, previous):
    n_steps = (6 if previous else 3) + 1      # renames + manifest
    for k in range(1, n_steps + 1):
        d = tmp_path / f"k{k}"; d.mkdir()
        out, man, names, moves = _setup(d, previous)
        jd = d / "attempt"

        def hook(step):
            if step == k:
                raise Kill()

        with pytest.raises(Kill):
            pj.promote(moves, man, lambda: man.write_text(json.dumps({"gen": "new"})), jd, _hook=hook)
        assert pj.read(jd)["state"] == "started"
        assert pj.recover(jd) == "rolled_back"
        gens, m = _state(out, man, names)
        if previous:
            assert gens == {"old"} and m == "old", (k, gens, m)
        else:
            assert gens == {None} and m is None, (k, gens, m)
        # the candidate bytes are kept under their temporary names
        assert all(t.read_text().startswith("new") for t, _ in moves)
        # a rerun of the same promotion from the restored state completes
        pj.promote(moves, man, lambda: man.write_text(json.dumps({"gen": "new"})), d / "attempt2")
        assert _state(out, man, names) == ({"new"}, "new")


def test_complete_promotion(tmp_path):
    out, man, names, moves = _setup(tmp_path, True)
    jd = tmp_path / "attempt"
    pj.promote(moves, man, lambda: man.write_text(json.dumps({"gen": "new"})), jd)
    assert pj.read(jd)["state"] == "complete" and pj.recover(jd) == "complete"
    assert _state(out, man, names) == ({"new"}, "new")
    assert sorted(p.name for p in (jd / "replaced").iterdir()) == sorted(names + [man.name])
    assert json.loads((jd / "replaced" / man.name).read_text())["gen"] == "old"
