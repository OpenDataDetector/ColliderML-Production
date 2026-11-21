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
