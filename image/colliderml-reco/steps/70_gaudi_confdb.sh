#!/bin/bash
# Gaudi's configurable databases (*.confdb2) are SQLite files in write-ahead-log mode.
# SQLite needs write access to the directory of such a file even to read it, so k4run fails
# for every user except root. Switch all of them, including those of the base image, to
# rollback-journal mode, which can be read from a read-only directory.
source /opt/colliderml-reco/steps/common.sh
source "$RECO_HOME/env/setup.sh"

python3 - <<'PY'
import os
import sqlite3

SQLITE_MAGIC = b"SQLite format 3\x00"
WAL, ROLLBACK = 2, 1   # file-format version bytes 18 and 19 of the SQLite header

converted = 0
for directory in dict.fromkeys(os.environ["LD_LIBRARY_PATH"].split(":")):
    if not os.path.isdir(directory):
        continue
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".confdb2"):
            continue
        path = os.path.join(directory, name)
        with open(path, "rb") as stream:
            header = stream.read(20)
        if not header.startswith(SQLITE_MAGIC) or header[18] != WAL:
            continue
        connection = sqlite3.connect(path)
        mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        connection.close()
        with open(path, "rb") as stream:
            header = stream.read(20)
        if mode != "delete" or header[18] != ROLLBACK:
            raise SystemExit(f"could not convert {path}")
        converted += 1
print(f"converted {converted} confdb2 files to rollback-journal mode")
if converted == 0:
    raise SystemExit("no write-ahead-log confdb2 file found: check this step against the base image")
PY
