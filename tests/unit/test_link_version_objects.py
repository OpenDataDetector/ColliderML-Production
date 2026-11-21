"""link_version_objects: parent in another base dir, terminal-file links, refuse to overwrite."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "scripts" / "cli" / "link_version_objects.py"


def _run(*a):
    return subprocess.run([sys.executable, str(TOOL), *map(str, a)], capture_output=True, text=True)


def test_parent_dir_and_terminal(tmp_path):
    pub = tmp_path / "pub" / "c" / "d"
    real = pub / "v1" / "runs" / "0" / "edm4hep.root"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"x")
    mid = pub / "v22" / "runs" / "0"
    mid.mkdir(parents=True)
    (mid / "edm4hep.root").symlink_to(real)
    st = tmp_path / "staging" / "c" / "d"
    args = ["--dataset-dir", st, "--parent-dataset-dir", pub, "--parent", "v22", "--version", "v23",
            "--runs", 0, 0, "--run-files", "edm4hep.root"]
    r = _run(*args)
    assert r.returncode == 0 and "would link" in r.stdout and not (st / "v23").exists(), "dry run by default"
    r = _run(*args, "--terminal", "--apply")
    assert r.returncode == 0, r.stderr
    link = st / "v23" / "runs" / "0" / "edm4hep.root"
    assert link.is_symlink() and Path(link.readlink()) == real.resolve()
    assert _run(*args, "--terminal", "--apply").returncode == 0, "identical link is accepted"
    r = _run(*args, "--apply")  # non-terminal target differs from the existing link
    assert r.returncode == 1 and "REFUSE" in r.stderr


def test_parquet_expect_files_and_contiguity(tmp_path):
    pub = tmp_path / "pub" / "c" / "d"
    obj = pub / "v1" / "parquet" / "truth" / "particles"
    obj.mkdir(parents=True)
    for s in (0, 100, 200):
        (obj / f"c.d.v1.truth.particles.events{s}-{s + 99}.parquet").write_bytes(b"x")
    st = tmp_path / "st" / "c" / "d"
    base = ["--dataset-dir", st, "--parent-dataset-dir", pub, "--parent", "v1", "--version", "v23",
            "--parquet", "truth/particles", "--events", 0, 299, "--terminal"]
    assert _run(*base, "--expect-files", 3).returncode == 0
    r = _run(*base, "--expect-files", 4)
    assert r.returncode != 0 and "expected 4" in r.stderr
