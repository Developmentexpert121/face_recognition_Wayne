"""SQLite-backed player + embedding store with a per-photo cache.

Schema:
    players(player_id TEXT PK, name TEXT, folder_path TEXT, photo_count INT)
    embeddings(player_id TEXT, photo_path TEXT, mtime REAL, vector BLOB)

Embeddings are stored as raw float32 bytes (512 dims). The cache key for a photo
is (photo_path, mtime), so rebuilds only re-embed new or changed files.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EMBEDDING_DIM = 512


@dataclass
class LoadedEmbeddings:
    """All embeddings flattened for fast vectorized matching.

    matrix: (N, 512) float32, L2-normalized rows.
    player_ids: list of length N mapping each row to a player_id.
    names: dict player_id -> name.
    """

    matrix: np.ndarray
    player_ids: list[str]
    names: dict[str, str]

    @property
    def is_empty(self) -> bool:
        return self.matrix.shape[0] == 0


class PlayerDatabase:
    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS players (
                player_id   TEXT PRIMARY KEY,   -- tag id shown in the photo (last 6-digit)
                name        TEXT NOT NULL,
                alt_id      TEXT,               -- first id when the folder has two
                folder_path TEXT NOT NULL,
                photo_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                player_id  TEXT NOT NULL,
                photo_path TEXT NOT NULL,
                mtime      REAL NOT NULL,
                vector     BLOB NOT NULL,
                PRIMARY KEY (photo_path),
                FOREIGN KEY (player_id) REFERENCES players(player_id)
            );
            CREATE INDEX IF NOT EXISTS idx_emb_player ON embeddings(player_id);
            """
        )
        self._conn.commit()

    # --- player upserts -------------------------------------------------

    def upsert_player(
        self, player_id: str, name: str, folder_path: str, alt_id: str | None = None
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO players (player_id, name, alt_id, folder_path, photo_count)
            VALUES (?, ?, ?, ?, 0)
            ON CONFLICT(player_id) DO UPDATE SET name=excluded.name,
                                                 alt_id=excluded.alt_id,
                                                 folder_path=excluded.folder_path
            """,
            (player_id, name, alt_id, folder_path),
        )
        self._conn.commit()

    def refresh_photo_count(self, player_id: str) -> None:
        self._conn.execute(
            """
            UPDATE players SET photo_count =
                (SELECT COUNT(*) FROM embeddings WHERE player_id = ?)
            WHERE player_id = ?
            """,
            (player_id, player_id),
        )
        self._conn.commit()

    # --- embedding cache ------------------------------------------------

    def cached_mtime(self, photo_path: str) -> float | None:
        row = self._conn.execute(
            "SELECT mtime FROM embeddings WHERE photo_path = ?", (photo_path,)
        ).fetchone()
        return None if row is None else float(row[0])

    def store_embedding(
        self, player_id: str, photo_path: str, mtime: float, vector: np.ndarray
    ) -> None:
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        self._conn.execute(
            """
            INSERT INTO embeddings (player_id, photo_path, mtime, vector)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(photo_path) DO UPDATE SET player_id=excluded.player_id,
                                                  mtime=excluded.mtime,
                                                  vector=excluded.vector
            """,
            (player_id, photo_path, mtime, blob),
        )
        self._conn.commit()

    def delete_embeddings_not_in(self, player_id: str, keep_paths: set[str]) -> None:
        """Remove cached embeddings for a player whose source photo is gone."""
        rows = self._conn.execute(
            "SELECT photo_path FROM embeddings WHERE player_id = ?", (player_id,)
        ).fetchall()
        stale = [r[0] for r in rows if r[0] not in keep_paths]
        if stale:
            self._conn.executemany(
                "DELETE FROM embeddings WHERE photo_path = ?", [(p,) for p in stale]
            )
            self._conn.commit()

    def prune_missing_files(self) -> int:
        """Delete embeddings whose source photo no longer exists on disk.

        Used after a two-source build: a player can receive photos from a roster
        folder AND from bulk image folders, so per-folder cleanup isn't enough.
        Returns the number of stale rows removed.
        """
        rows = self._conn.execute(
            "SELECT photo_path FROM embeddings"
        ).fetchall()
        stale = [r[0] for r in rows if not Path(r[0]).exists()]
        if stale:
            self._conn.executemany(
                "DELETE FROM embeddings WHERE photo_path = ?", [(p,) for p in stale]
            )
            self._conn.commit()
        return len(stale)

    # --- id lookup (bulk-image enrollment) ------------------------------

    def load_id_lookup(self) -> dict[str, str]:
        """Map every known id (player_id and alt_id) -> canonical player_id.

        Bulk image filenames carry one of a player's ids; this lets us resolve a
        filename id to the player it belongs to regardless of which id appears.
        """
        lookup: dict[str, str] = {}
        for player_id, alt_id in self._conn.execute(
            "SELECT player_id, alt_id FROM players"
        ):
            lookup[str(player_id)] = player_id
            if alt_id:
                lookup[str(alt_id)] = player_id
        return lookup

    def refresh_all_photo_counts(self) -> None:
        """Recompute photo_count for every player from the embeddings table."""
        self._conn.execute(
            """
            UPDATE players SET photo_count = (
                SELECT COUNT(*) FROM embeddings
                WHERE embeddings.player_id = players.player_id
            )
            """
        )
        self._conn.commit()

    # --- bulk load for matching ----------------------------------------

    def load_all(self) -> LoadedEmbeddings:
        names = {
            row[0]: row[1]
            for row in self._conn.execute("SELECT player_id, name FROM players")
        }
        rows = self._conn.execute(
            "SELECT player_id, vector FROM embeddings"
        ).fetchall()
        if not rows:
            return LoadedEmbeddings(
                matrix=np.empty((0, EMBEDDING_DIM), dtype=np.float32),
                player_ids=[],
                names=names,
            )
        player_ids = [r[0] for r in rows]
        matrix = np.stack(
            [np.frombuffer(r[1], dtype=np.float32) for r in rows]
        ).astype(np.float32)
        return LoadedEmbeddings(matrix=matrix, player_ids=player_ids, names=names)

    def player_count(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        )

    def players_with_photos(self) -> int:
        return int(
            self._conn.execute(
                "SELECT COUNT(DISTINCT player_id) FROM embeddings"
            ).fetchone()[0]
        )

    def embedding_count(self) -> int:
        return int(
            self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PlayerDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_database(config):
    """Open the configured player store: shared Neon if set, else local SQLite.

    The builder and matcher call this instead of constructing a backend directly,
    so switching to the shared cloud database is purely a matter of setting the
    NEON_DATABASE_URL environment variable — no code change. The Neon module
    (and its psycopg/pgvector deps) is imported lazily, only when actually used,
    so SQLite-only installs need nothing extra.

    Args:
        config: An EngineConfig. ``config.resolved_neon_dsn()`` decides the backend.

    Returns:
        A NeonDatabase or PlayerDatabase — both expose the same interface.
    """
    dsn = config.resolved_neon_dsn()
    if dsn:
        from .neon_database import NeonDatabase

        return NeonDatabase(dsn)
    return PlayerDatabase(config.db_path)
