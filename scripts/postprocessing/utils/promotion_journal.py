"""Recoverable generation switch for packaged shards (v23, Codex round 7 finding 5).

A chunk's new shards are written under hidden temporary names and promoted by one rename each. Without a journal a
kill between two renames left a mixed generation. The journal ``<attempt dir>/promotion.json`` is written BEFORE
the first move:

    {"state": "started", "moves": [[tmp, final, backup_or_null], ...],
     "manifest": <shared chunk manifest>, "manifest_backup": <copy of the previous manifest or null>}

and set to "complete" after the new manifest is written. A replaced shard is moved to its backup path (under
``<attempt dir>/replaced/``), the previous manifest is copied there. ``recover`` of a "started" journal rolls back:
every promoted new shard goes back to its temporary name (bytes kept; reconcile owns temporary files), every backup
returns to its final name, the previous manifest is restored (or, for a first generation, the new manifest is moved
to ``<attempt dir>/manifest.rolled_back.json``). The result is the previous generation, complete, with its manifest.

The controller (v23_control.py reconcile/purge) uses ``recover`` and ``read``; the packager uses ``promote``.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

JOURNAL = "promotion.json"


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read(journal_dir) -> dict | None:
    p = Path(journal_dir) / JOURNAL
    return json.loads(p.read_text()) if p.exists() else None


def promote(moves: list[tuple[Path, Path]], manifest: Path | None, write_manifest, journal_dir: Path,
            _hook=None) -> dict:
    """Promote (tmp, final) pairs under a journal; write_manifest() writes the shared manifest after the moves.
    _hook(step) is called after every rename (tests inject a kill there)."""
    journal_dir = Path(journal_dir)
    backup_dir = journal_dir / "replaced"
    plan = [[str(t), str(f), str(backup_dir / f.name) if f.exists() else None] for t, f in moves]
    m_backup = str(backup_dir / manifest.name) if manifest is not None and manifest.exists() else None
    doc = {"state": "started", "moves": plan, "manifest": str(manifest) if manifest else None,
           "manifest_backup": m_backup}
    if any(b for _, _, b in plan) or m_backup:
        backup_dir.mkdir(parents=True, exist_ok=True)
    _write(journal_dir / JOURNAL, doc)
    if m_backup:
        shutil.copy2(manifest, m_backup)
    step = 0
    for t, f, b in plan:
        if b:
            os.replace(f, b)
            step += 1
            _hook and _hook(step)
        os.replace(t, f)
        step += 1
        _hook and _hook(step)
    write_manifest()
    step += 1
    _hook and _hook(step)
    doc["state"] = "complete"
    _write(journal_dir / JOURNAL, doc)
    return doc


def recover(journal_dir) -> str:
    """Roll back a "started" promotion; returns the journal state afterwards ("complete", "rolled_back" or
    "absent")."""
    journal_dir = Path(journal_dir)
    doc = read(journal_dir)
    if doc is None:
        return "absent"
    if doc["state"] != "started":
        return doc["state"]
    for t, f, b in reversed(doc["moves"]):
        t, f = Path(t), Path(f)
        if b:
            b = Path(b)
            if b.exists():                  # the previous shard was moved aside
                if f.exists():              # and the new one installed
                    os.replace(f, t)
                os.replace(b, f)
        elif f.exists() and not t.exists():  # first generation: the new shard was installed
            os.replace(f, t)
    if doc.get("manifest"):
        m = Path(doc["manifest"])
        if doc.get("manifest_backup"):
            tmp = m.with_name(f".{m.name}.restore.tmp")
            shutil.copy2(doc["manifest_backup"], tmp)
            os.replace(tmp, m)
        elif m.exists():
            os.replace(m, journal_dir / "manifest.rolled_back.json")
    doc["state"] = "rolled_back"
    _write(journal_dir / JOURNAL, doc)
    return "rolled_back"
