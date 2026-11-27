"""Attempt isolation for the per-run production stages (v23, Codex round 5 finding 6).

An attempt of a per-run stage (digitization, pandora_reco, reco_tables) never writes into the run
directory while it runs. run_stage gives the stage a private run root

    <version>/.attempts/<stage>/<run>.a<K>/<run>/

that holds symlinks to the stage's declared inputs only (provenance.inputs, top-level names). The stage
writes its outputs there. Only after the task exits 0 are the outputs promoted into runs/<run>/, one
rename per top-level entry (same file system), with the attempt record set to status "promoting" and the
planned moves listed first, so an interrupted promotion is visible (reconcile marks it interrupted and the
unit is rerun). A failed attempt promotes nothing: its partial outputs stay in its own directory and the
accepted outputs of the run are never touched.

Collisions: a small UTF-8 file the stage wrote whose name already exists in the run directory and is not
one of the stage's declared outputs (e.g. timing_summary.txt, written by several stages) is not moved but
embedded in the attempt record. Any other existing entry of the same name (a rerun after an explicit
replacement) is first moved to <attempt>/replaced/ (kept for rollback, listed in the record).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

EMBED_MAX_BYTES = 1 << 20


def top_names(patterns) -> list[str]:
    """Top-level run-dir names of provenance patterns ('tracks/*.parquet' -> 'tracks', 'a|b' -> both)."""
    out = []
    for pat in patterns or []:
        for alt in str(pat).split("|"):
            name = alt.strip().split("/", 1)[0]
            if name and name not in out:
                out.append(name)
    return out


def attempt_root(version_dir: Path, stage: str, unit: int, attempt_no: int) -> Path:
    return Path(version_dir) / ".attempts" / stage / f"{unit}.a{attempt_no}"


def prepare(run_dir: Path, work: Path, input_names) -> list[str]:
    """Create the private run dir with symlinks to the declared inputs that exist (glob names allowed).
    Returns the linked names. The links point at the run-dir entries (whatever they resolve to)."""
    run_dir, work = Path(run_dir), Path(work)
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"attempt directory {work} is not empty")
    work.mkdir(parents=True, exist_ok=True)
    linked = []
    for name in input_names:
        matches = sorted(run_dir.glob(name)) if any(c in name for c in "*?[") else [run_dir / name]
        for src in matches:
            if os.path.lexists(src) and src.name not in linked:
                os.symlink(os.path.abspath(src), work / src.name)
                linked.append(src.name)
    return linked


def plan(work: Path, run_dir: Path, output_names) -> dict:
    """What a promotion would do: moves (name -> replaces existing?), embeds, kept (not moved)."""
    work, run_dir = Path(work), Path(run_dir)
    out_set = set(output_names)
    moves, embed, keep = [], [], []
    for e in sorted(os.scandir(work), key=lambda x: x.name):
        if e.is_symlink():
            continue
        dst = run_dir / e.name
        declared = e.name in out_set or any(Path(e.name).match(n) for n in out_set if any(c in n for c in "*?["))
        if not declared and os.path.lexists(dst) and e.is_file() and e.stat().st_size <= EMBED_MAX_BYTES \
                and _is_text(Path(e.path)):
            embed.append(e.name)       # e.g. timing_summary.txt, which several stages write
        else:
            moves.append({"name": e.name, "replaces": os.path.lexists(dst)})
    return {"moves": moves, "embed": embed, "keep": keep}


def _is_text(p: Path) -> bool:
    try:
        p.read_bytes().decode("utf-8")
        return True
    except (OSError, UnicodeDecodeError):
        return False


def promote(work: Path, run_dir: Path, the_plan: dict, replaced_dir: Path) -> dict:
    """Carry out a plan: rename each moved entry into the run dir (an existing declared output goes to
    replaced_dir first), embed the small colliding text files, leave the rest. Returns what was done."""
    work, run_dir, replaced_dir = Path(work), Path(run_dir), Path(replaced_dir)
    done = {"moved": [], "replaced": [], "embedded": {}, "kept": list(the_plan["keep"])}
    run_dir.mkdir(parents=True, exist_ok=True)
    for m in the_plan["moves"]:
        src, dst = work / m["name"], run_dir / m["name"]
        if os.path.lexists(dst):
            replaced_dir.mkdir(parents=True, exist_ok=True)
            os.rename(dst, replaced_dir / m["name"])
            done["replaced"].append(m["name"])
        os.rename(src, dst)
        done["moved"].append(m["name"])
    for name in the_plan["embed"]:
        p = work / name
        done["embedded"][name] = p.read_text()
        p.unlink()
    return done


def cleanup(attempt_dir: Path) -> bool:
    """Remove the input symlinks and every empty directory of an attempt; True if nothing is left."""
    attempt_dir = Path(attempt_dir)
    if not attempt_dir.exists():
        return True
    for dirpath, dirnames, filenames in os.walk(attempt_dir, topdown=False):
        for f in filenames + dirnames:
            p = Path(dirpath) / f
            if p.is_symlink():
                p.unlink()
        try:
            os.rmdir(dirpath)
        except OSError:
            pass
    return not attempt_dir.exists()


def purge(attempt_dir: Path) -> None:
    """Delete an attempt directory (failed or superseded attempts only; called by v23_control reconcile)."""
    if ".attempts" not in Path(attempt_dir).parts:
        raise ValueError(f"refusing to purge {attempt_dir}: not an attempt directory")
    shutil.rmtree(attempt_dir)
