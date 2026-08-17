"""Central configuration for the face engine.

All tunables live here so the UI, CLI, and future apps share one source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Environment variable that holds the secret Neon/Postgres connection string.
# The connection string contains a password, so it is NEVER hardcoded here or
# shipped in the client zip — each machine sets it locally in its environment.
NEON_DSN_ENV = "NEON_DATABASE_URL"

# Image extensions we treat as photos when scanning folders.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


@dataclass
class EngineConfig:
    """Runtime configuration for the recognition engine.

    Attributes:
        model_name: InsightFace model pack. 'buffalo_l' = strong accuracy, ~326MB,
            downloaded once to ~/.insightface on first run.
        det_size: Detector input size. Larger finds smaller faces in big group
            photos at the cost of speed. 1024x1024 is a good default for
            high-resolution team photos.
        ctx_id: InsightFace device id. -1 = CPU, 0 = first GPU (needs
            onnxruntime-gpu). CPU is the Phase 1 default.
        match_threshold: Minimum cosine similarity to accept a match. Faces below
            this are reported as "Unidentified". ~0.45 is a sensible start for
            buffalo_l; tune against real data.
        dedupe: If True, a player can only be assigned to one face in the photo
            (highest-confidence match wins).
        row_gap_factor: Row clustering sensitivity. Two faces belong to the same
            row when their vertical centers differ by less than
            row_gap_factor * median_face_height. Lower = splits rows more eagerly.
        db_path: Location of the local SQLite database/embedding cache. Used when
            no Neon connection string is configured (the offline default).
        neon_dsn: Optional Postgres/Neon connection string for the SHARED cloud
            database (Postgres + pgvector). When set — directly or, preferably,
            via the NEON_DATABASE_URL environment variable — the cloud store is
            used instead of the local SQLite file, so several machines/locations
            share one player database. The secret string is read at runtime and
            never written into this file or the client zip.
    """

    model_name: str = "buffalo_l"
    det_size: tuple[int, int] = (1024, 1024)
    ctx_id: int = -1
    match_threshold: float = 0.45
    dedupe: bool = True
    row_gap_factor: float = 0.7
    db_path: Path = field(default_factory=lambda: Path("players.db"))
    neon_dsn: str | None = None

    def with_db_path(self, path: str | Path) -> "EngineConfig":
        """Return a copy pointing at a different database file."""
        from dataclasses import replace

        return replace(self, db_path=Path(path))

    def resolved_neon_dsn(self) -> str | None:
        """The active Neon connection string, if any.

        Prefers an explicitly set ``neon_dsn`` field; otherwise falls back to the
        NEON_DATABASE_URL environment variable. Returns None when neither is set,
        which keeps the local SQLite backend as the default.
        """
        if self.neon_dsn:
            return self.neon_dsn
        env = os.environ.get(NEON_DSN_ENV)
        return env or None

    @property
    def use_neon(self) -> bool:
        """True when a shared Neon database should be used instead of SQLite."""
        return self.resolved_neon_dsn() is not None
