"""Neon (Postgres + pgvector) player + embedding store — the SHARED cloud backend.

This mirrors the SQLite ``PlayerDatabase`` interface exactly so the builder and
matcher can use either backend through the same calls. It is selected
automatically by ``open_database()`` when a Neon connection string is present
(see ``EngineConfig.resolved_neon_dsn``); otherwise the local SQLite store is
used and this module is never imported, so psycopg/pgvector stay optional.

Why a cloud store: several machines (possibly in different locations) can share
ONE player database instead of copying a ``players.db`` file around. The 512-d
ArcFace vectors are stored in a ``vector(512)`` column and matched the same way
(cosine similarity via a dot product on L2-normalized rows).

Credentials: the connection string contains a password and is read at runtime
from the environment (NEON_DATABASE_URL) — it is never hardcoded here or shipped
in the client zip.
"""

from __future__ import annotations

import numpy as np

from .database import EMBEDDING_DIM, LoadedEmbeddings


class NeonDatabase:
    """Postgres/pgvector-backed store implementing the PlayerDatabase interface."""

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency hint
            raise RuntimeError(
                "The Neon backend needs the 'psycopg[binary]' and 'pgvector' "
                "packages. Install them with: pip install -r requirements.txt"
            ) from exc

        # autocommit so each write lands immediately, matching the SQLite store
        # which commits after every operation.
        self._conn = psycopg.connect(dsn, autocommit=True)
        self._create_schema()
        register_vector(self._conn)

    def _create_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS players (
                    player_id   TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    alt_id      TEXT,
                    folder_path TEXT NOT NULL,
                    photo_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    photo_path TEXT PRIMARY KEY,
                    player_id  TEXT NOT NULL REFERENCES players(player_id),
                    mtime      DOUBLE PRECISION NOT NULL,
                    vector     vector({EMBEDDING_DIM}) NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_emb_player ON embeddings(player_id)"
            )

    # --- player upserts -------------------------------------------------

    def upsert_player(
        self, player_id: str, name: str, folder_path: str, alt_id: str | None = None
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO players (player_id, name, alt_id, folder_path, photo_count)
                VALUES (%s, %s, %s, %s, 0)
                ON CONFLICT (player_id) DO UPDATE SET name = EXCLUDED.name,
                                                      alt_id = EXCLUDED.alt_id,
                                                      folder_path = EXCLUDED.folder_path
                """,
                (player_id, name, alt_id, folder_path),
            )

    def refresh_photo_count(self, player_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE players SET photo_count =
                    (SELECT COUNT(*) FROM embeddings WHERE player_id = %s)
                WHERE player_id = %s
                """,
                (player_id, player_id),
            )

    # --- embedding cache ------------------------------------------------

    def cached_mtime(self, photo_path: str) -> float | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT mtime FROM embeddings WHERE photo_path = %s", (photo_path,)
            )
            row = cur.fetchone()
        return None if row is None else float(row[0])

    def store_embedding(
        self, player_id: str, photo_path: str, mtime: float, vector: np.ndarray
    ) -> None:
        vec = np.asarray(vector, dtype=np.float32)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO embeddings (player_id, photo_path, mtime, vector)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (photo_path) DO UPDATE SET player_id = EXCLUDED.player_id,
                                                       mtime = EXCLUDED.mtime,
                                                       vector = EXCLUDED.vector
                """,
                (player_id, photo_path, mtime, vec),
            )

    def delete_embeddings_not_in(self, player_id: str, keep_paths: set[str]) -> None:
        """Remove a player's cached embeddings whose source photo was dropped.

        Driven by an explicit keep-set (not a disk check), so it is safe on a
        shared database where photo paths belong to whichever machine enrolled
        them.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT photo_path FROM embeddings WHERE player_id = %s", (player_id,)
            )
            stale = [r[0] for r in cur.fetchall() if r[0] not in keep_paths]
            if stale:
                cur.execute(
                    "DELETE FROM embeddings WHERE photo_path = ANY(%s)", (stale,)
                )

    def prune_missing_files(self) -> int:
        """No-op on the shared backend.

        The SQLite store deletes embeddings whose source photo is missing from
        the LOCAL disk. On a shared cloud database that is dangerous: photos
        enrolled by another machine/location won't exist on the machine running
        the build, so a local disk check would wrongly delete everyone else's
        embeddings. We therefore never prune by local existence here. Stale
        cleanup, when needed, must be explicit (see ``delete_embeddings_not_in``).
        """
        return 0

    # --- id lookup (bulk-image enrollment) ------------------------------

    def load_id_lookup(self) -> dict[str, str]:
        lookup: dict[str, str] = {}
        with self._conn.cursor() as cur:
            cur.execute("SELECT player_id, alt_id FROM players")
            for player_id, alt_id in cur.fetchall():
                lookup[str(player_id)] = player_id
                if alt_id:
                    lookup[str(alt_id)] = player_id
        return lookup

    def refresh_all_photo_counts(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE players SET photo_count = (
                    SELECT COUNT(*) FROM embeddings
                    WHERE embeddings.player_id = players.player_id
                )
                """
            )

    # --- bulk load for matching ----------------------------------------

    def load_all(self) -> LoadedEmbeddings:
        with self._conn.cursor() as cur:
            cur.execute("SELECT player_id, name FROM players")
            names = {row[0]: row[1] for row in cur.fetchall()}
            cur.execute("SELECT player_id, vector FROM embeddings")
            rows = cur.fetchall()
        if not rows:
            return LoadedEmbeddings(
                matrix=np.empty((0, EMBEDDING_DIM), dtype=np.float32),
                player_ids=[],
                names=names,
            )
        player_ids = [r[0] for r in rows]
        # pgvector returns each vector as a numpy array already.
        matrix = np.stack(
            [np.asarray(r[1], dtype=np.float32) for r in rows]
        ).astype(np.float32)
        return LoadedEmbeddings(matrix=matrix, player_ids=player_ids, names=names)

    def player_count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM players")
            return int(cur.fetchone()[0])

    def players_with_photos(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(DISTINCT player_id) FROM embeddings")
            return int(cur.fetchone()[0])

    def embedding_count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embeddings")
            return int(cur.fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "NeonDatabase":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
