"""Seed and transfer the player database between backends — account-agnostic.

Why this exists: the Neon database is set up on YOUR account for now and will be
handed to the client later. Nothing in the code is tied to an account — the
schema auto-creates on any empty database and the connection string is read from
an environment variable. This script is the only piece you need to move the
DATA, so the handover is just "point at the new database and copy the rows."

No secrets live in this file. Every connection string is read from an environment
variable:

    NEON_DATABASE_URL        the DESTINATION (the database being written to)
    NEON_SOURCE_URL          the SOURCE for a Neon -> Neon copy (handover)

Usage
-----
1) Create the schema on a fresh (e.g. client) Neon database:
       export NEON_DATABASE_URL='postgresql://...client db...'
       python neon_transfer.py init

2) Seed Neon from the local players.db you already built (no re-embedding):
       export NEON_DATABASE_URL='postgresql://...'
       python neon_transfer.py from-sqlite --db players.db

3) Hand over: copy everything from your Neon to the client's Neon:
       export NEON_SOURCE_URL='postgresql://...your db...'
       export NEON_DATABASE_URL='postgresql://...client db...'
       python neon_transfer.py copy

4) Check what's in the destination:
       python neon_transfer.py status

The copy is idempotent: players and embeddings upsert by primary key, so running
it twice is safe and a re-run only adds what's new.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

from face_engine.database import EMBEDDING_DIM
from face_engine.neon_database import NeonDatabase

DEST_ENV = "NEON_DATABASE_URL"
SOURCE_ENV = "NEON_SOURCE_URL"


def _require_env(name: str) -> str:
    dsn = os.environ.get(name)
    if not dsn:
        sys.exit(
            f"[ERROR] Environment variable {name} is not set.\n"
            f"        Set it to the connection string first, e.g.:\n"
            f"          export {name}='postgresql://user:pass@host/db'\n"
            f"        (Do not paste the connection string into any source file.)"
        )
    return dsn


# --- readers: pull raw rows from a source -------------------------------------


def _read_sqlite(db_path: Path):
    """Yield (players, embeddings) from a local SQLite players.db.

    players:    list of (player_id, name, alt_id, folder_path)
    embeddings: list of (player_id, photo_path, mtime, vector ndarray)
    """
    if not db_path.is_file():
        sys.exit(f"[ERROR] SQLite database not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        players = conn.execute(
            "SELECT player_id, name, alt_id, folder_path FROM players"
        ).fetchall()
        embeddings = [
            (
                pid,
                photo_path,
                float(mtime),
                np.frombuffer(blob, dtype=np.float32),
            )
            for pid, photo_path, mtime, blob in conn.execute(
                "SELECT player_id, photo_path, mtime, vector FROM embeddings"
            )
        ]
    finally:
        conn.close()
    return players, embeddings


def _read_neon(dsn: str):
    """Yield (players, embeddings) from a source Neon/Postgres database."""
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(dsn)
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT player_id, name, alt_id, folder_path FROM players")
            players = cur.fetchall()
            cur.execute("SELECT player_id, photo_path, mtime, vector FROM embeddings")
            embeddings = [
                (pid, photo_path, float(mtime), np.asarray(vec, dtype=np.float32))
                for pid, photo_path, mtime, vec in cur.fetchall()
            ]
    finally:
        conn.close()
    return players, embeddings


# --- writer: push rows into the destination Neon ------------------------------


def _write_dest(players, embeddings) -> None:
    dsn = _require_env(DEST_ENV)
    print(f"Writing to {DEST_ENV} -> {_safe_host(dsn)}")
    with NeonDatabase(dsn) as db:  # __init__ creates the schema if needed
        for pid, name, alt_id, folder_path in players:
            db.upsert_player(pid, name, folder_path, alt_id)
        print(f"  players upserted: {len(players)}")

        bad_dim = 0
        for i, (pid, photo_path, mtime, vec) in enumerate(embeddings, start=1):
            if vec.shape[0] != EMBEDDING_DIM:
                bad_dim += 1
                continue
            db.store_embedding(pid, photo_path, mtime, vec)
            if i % 200 == 0:
                print(f"  embeddings written: {i}/{len(embeddings)}")
        db.refresh_all_photo_counts()
        print(f"  embeddings written: {len(embeddings) - bad_dim} (skipped {bad_dim})")
        print(
            f"  destination now holds {db.player_count()} players / "
            f"{db.embedding_count()} embeddings."
        )


def _safe_host(dsn: str) -> str:
    """Show the host only, never the password, in logs."""
    tail = dsn.split("@")[-1]
    return tail.split("?")[0]


# --- subcommands --------------------------------------------------------------


def cmd_init(_args) -> int:
    dsn = _require_env(DEST_ENV)
    print(f"Ensuring schema on {_safe_host(dsn)} ...")
    with NeonDatabase(dsn) as db:
        print(
            f"Ready. Current contents: {db.player_count()} players / "
            f"{db.embedding_count()} embeddings."
        )
    return 0


def cmd_from_sqlite(args) -> int:
    players, embeddings = _read_sqlite(Path(args.db))
    print(f"Read {len(players)} players / {len(embeddings)} embeddings from {args.db}")
    _write_dest(players, embeddings)
    return 0


def cmd_copy(_args) -> int:
    src = _require_env(SOURCE_ENV)
    print(f"Reading from {SOURCE_ENV} -> {_safe_host(src)}")
    players, embeddings = _read_neon(src)
    print(f"Read {len(players)} players / {len(embeddings)} embeddings.")
    _write_dest(players, embeddings)
    return 0


def cmd_status(_args) -> int:
    dsn = _require_env(DEST_ENV)
    with NeonDatabase(dsn) as db:
        print(f"Destination: {_safe_host(dsn)}")
        print(f"  players:           {db.player_count()}")
        print(f"  players w/ photos: {db.players_with_photos()}")
        print(f"  embeddings:        {db.embedding_count()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Create the schema on NEON_DATABASE_URL.")

    p_sqlite = sub.add_parser(
        "from-sqlite", help="Seed NEON_DATABASE_URL from a local players.db."
    )
    p_sqlite.add_argument("--db", default="players.db", help="Local SQLite path.")

    sub.add_parser(
        "copy", help="Copy NEON_SOURCE_URL -> NEON_DATABASE_URL (account handover)."
    )
    sub.add_parser("status", help="Show row counts for NEON_DATABASE_URL.")

    args = parser.parse_args()
    handlers = {
        "init": cmd_init,
        "from-sqlite": cmd_from_sqlite,
        "copy": cmd_copy,
        "status": cmd_status,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
