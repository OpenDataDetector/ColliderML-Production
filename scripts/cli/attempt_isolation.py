"""Attempt isolation for the per-run production stages (v23, Codex round 5 finding 6).

An attempt of a per-run stage (digitization, pandora_reco, reco_tables) never writes into the run
directory while it runs. run_stage gives the stage a private run root

    <version>/.attempts/<stage>/<run>.a<K>/<run>/

that holds symlinks to the stage's declared inputs only (provenance.inputs, top-level names). The stage
writes its outputs there. Only after the task exits 0 are the outputs promoted into runs/<run>/, one
rename per top-level entry (same file system), with the attempt record set to status "promoting" and the
planned moves journaled first (<attempt>/promotion.json, state "started"), so an interrupted promotion is
rolled back by reconcile to the previous complete generation (rollback()) and the unit is rerun. Inputs are
pinned (hard links to the consumed inodes) and hashed before the stage runs. A failed attempt promotes nothing: its partial outputs stay in its own directory and the
accepted outputs of the run are never touched.

Collisions: a small UTF-8 file the stage wrote whose name already exists in the run directory and is not
one of the stage's declared outputs (e.g. timing_summary.txt, written by several stages) is not moved but
embedded in the attempt record. Any other existing entry of the same name (a rerun after an explicit
replacement) is first moved to <attempt>/replaced/ (kept for rollback, listed in the record).
"""
from __future__ import annotations

import json
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


def prepare(run_dir: Path, work: Path, input_names, pin: bool = True) -> list[str]:
    """Create the private run dir with the declared inputs that exist (glob names allowed). Returns the linked
    names.

    Generation pinning (Codex re-review finding 2): an input that is a produced entry of the run dir (a file or a
    directory, not a symlink) is pinned by hard links to its current inodes (a directory is recreated with hard
    links of its files), so an upstream replacement during the attempt (which renames the old entry into its own
    attempt's replaced/) cannot change what this attempt reads. A symlinked input (a parent link to a frozen,
    protected terminal) is linked to its resolved terminal. Where a hard link is refused (another file system) the
    input falls back to a symlink to the resolved path and the fallback is listed in <work>/../inputs_pinned.json.
    The origin realpath of every pinned file is recorded there; run_stage hashes the inputs from <work> BEFORE the
    stage runs and records those origins."""
    run_dir, work = Path(run_dir), Path(work)
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"attempt directory {work} is not empty")
    work.mkdir(parents=True, exist_ok=True)
    linked, pinned = [], {"files": {}, "fallback": [], "dirs": []}
    for name in input_names:
        matches = sorted(run_dir.glob(name)) if any(c in name for c in "*?[") else [run_dir / name]
        for src in matches:
            if not os.path.lexists(src) or src.name in linked:
                continue
            dst = work / src.name
            if not pin or src.is_symlink():
                os.symlink(os.path.realpath(src) if pin else os.path.abspath(src), dst)
            elif src.is_dir():
                for dirpath, dirnames, filenames in os.walk(src):
                    rel = Path(dirpath).relative_to(src)
                    (dst / rel).mkdir(parents=True, exist_ok=True)
                    pinned["dirs"].append(str(Path(src.name) / rel))
                    for fn in filenames:
                        _pin_file(Path(dirpath) / fn, dst / rel / fn, Path(src.name) / rel / fn, pinned)
            else:
                _pin_file(src, dst, Path(src.name), pinned)
            linked.append(src.name)
    if pin:
        (work.parent / "inputs_pinned.json").write_text(json.dumps(pinned, indent=1, sort_keys=True) + "\n")
    return linked


def _pin_file(src: Path, dst: Path, rel: Path, pinned: dict) -> None:
    real = os.path.realpath(src)
    try:
        os.link(real, dst)
    except OSError:
        os.symlink(real, dst)
        pinned["fallback"].append(str(rel))
    pinned["files"][str(rel)] = real
    pinned.setdefault("inodes", {})[str(rel)] = os.stat(real).st_ino


def pinned_inputs(attempt_dir: Path) -> dict:
    try:
        return json.loads((Path(attempt_dir) / "inputs_pinned.json").read_text())
    except (OSError, ValueError):
        return {"files": {}, "fallback": [], "dirs": []}


def plan(work: Path, run_dir: Path, output_names, inputs=()) -> dict:
    """What a promotion would do: moves (name -> replaces existing?), embeds, kept (not moved). Inputs (links and
    pinned hard links, top-level names) are never promoted."""
    work, run_dir = Path(work), Path(run_dir)
    out_set = set(output_names)
    moves, embed, keep = [], [], []
    for e in sorted(os.scandir(work), key=lambda x: x.name):
        if e.is_symlink() or e.name in set(inputs):
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


def promote(work: Path, run_dir: Path, the_plan: dict, replaced_dir: Path, journal_dir: Path | None = None,
            journal_extra: dict | None = None) -> dict:
    """Carry out a plan: rename each moved entry into the run dir (an existing declared output goes to
    replaced_dir first), embed the small colliding text files, leave the rest. Returns what was done.

    With journal_dir, <journal_dir>/promotion.json lists every move [source, final, backup or null] with state
    "started" BEFORE the first rename (fsync); the caller sets it "complete" after its record. A kill in between
    is rolled back by rollback() (v23_control reconcile): one complete generation is restored."""
    import run_provenance as rp
    work, run_dir, replaced_dir = Path(work), Path(run_dir), Path(replaced_dir)
    done = {"moved": [], "replaced": [], "embedded": {}, "kept": list(the_plan["keep"])}
    run_dir.mkdir(parents=True, exist_ok=True)
    moves = [[str(work / m["name"]), str(run_dir / m["name"]),
              str(replaced_dir / m["name"]) if os.path.lexists(run_dir / m["name"]) else None]
             for m in the_plan["moves"]]
    if journal_dir is not None:
        rp.write_journal(journal_dir, {"state": "started", "moves": moves, "manifest": None, "manifest_backup": None,
                                       **(journal_extra or {})})
    for src, dst, backup in moves:
        if backup is not None:
            Path(backup).parent.mkdir(parents=True, exist_ok=True)
            os.rename(dst, backup)
            done["replaced"].append(Path(dst).name)
        os.rename(src, dst)
        done["moved"].append(Path(dst).name)
    for name in the_plan["embed"]:
        p = work / name
        done["embedded"][name] = p.read_text()
        p.unlink()
    return done


def complete_journal(journal_dir: Path) -> None:
    import run_provenance as rp
    doc = rp.read_journal(journal_dir)
    if doc is not None:
        doc["state"] = "complete"
        rp.write_journal(journal_dir, doc)


def _packager_recover(journal_dir: Path) -> str:
    """The packager's own journal (utils/promotion_journal.py: shards back to their temporary names, the previous
    manifest restored)."""
    import importlib.util
    f = Path(__file__).resolve().parents[1] / "postprocessing" / "utils" / "promotion_journal.py"
    spec = importlib.util.spec_from_file_location("colliderml_promotion_journal", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.recover(journal_dir)


def rollback(journal_dir: Path) -> list[str]:
    """Undo an interrupted promotion ("started" journal), newest move first: a new entry that reached its final
    path goes back to its source, the kept previous entry goes back to the final path. Only renames; the journal ends
    "rolled_back". A packaging journal (it names the chunk manifest) is rolled back by the packager's own recover().
    Returns what was done."""
    import run_provenance as rp
    doc = rp.read_journal(journal_dir)
    if not doc or doc.get("state") != "started":
        return []
    if doc.get("manifest") is not None:
        return [f"packaging journal {_packager_recover(journal_dir)}"]
    did = []
    for src, dst, backup in reversed(list(doc.get("moves") or [])):
        dst_p = Path(dst)
        if backup is not None and os.path.lexists(backup):
            if os.path.lexists(dst_p):
                if src and not os.path.lexists(src):
                    Path(src).parent.mkdir(parents=True, exist_ok=True)
                    os.rename(dst_p, src)
                    did.append(f"new {dst_p.name} -> {src}")
                else:
                    aside = Path(journal_dir) / "rolled_back" / dst_p.name
                    aside.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(dst_p, aside)
                    did.append(f"new {dst_p.name} -> {aside}")
            os.rename(backup, dst_p)
            did.append(f"previous {dst_p.name} restored")
        elif backup is None and os.path.lexists(dst_p) and src and not os.path.lexists(src):
            Path(src).parent.mkdir(parents=True, exist_ok=True)
            os.rename(dst_p, src)
            did.append(f"new {dst_p.name} -> {src} (no previous generation)")
    doc["state"] = "rolled_back"
    doc["rolled_back"] = did
    rp.write_journal(journal_dir, doc)
    return did


def cleanup(attempt_dir: Path) -> bool:
    """Remove the input symlinks and every empty directory of an attempt; True if nothing is left."""
    attempt_dir = Path(attempt_dir)
    if not attempt_dir.exists():
        return True
    pins = pinned_inputs(attempt_dir)
    for unit_dir in [p for p in attempt_dir.iterdir() if p.is_dir() and p.name not in ("replaced", "rolled_back")]:
        for rel in pins.get("files") or {}:      # hard links of pinned inputs (the original entries are untouched)
            q = unit_dir / rel
            if q.exists() and not q.is_symlink() and os.stat(q).st_ino == (pins.get("inodes") or {}).get(rel):
                q.unlink()
    for f in ("inputs_pinned.json", "promotion.json"):
        doc = None
        if f == "promotion.json":
            try:
                doc = json.loads((attempt_dir / f).read_text())
            except (OSError, ValueError):
                doc = None
        if (attempt_dir / f).exists() and (f == "inputs_pinned.json" or (doc or {}).get("state") == "complete") \
                and not (attempt_dir / "replaced").exists():
            (attempt_dir / f).unlink()
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
