"""Reusable, UI-independent facial recognition engine.

Phase 1 deliverable: build a player face database from per-player folders and
match faces detected in a team/group photo against it.
"""

from .config import EngineConfig
from .detector import FaceDetector
from .database import PlayerDatabase, open_database
from .builder import build_database, BuildReport
from .matcher import match_group_photo, MatchResult
from .rows import group_into_rows, row_labels, format_roster

__all__ = [
    "EngineConfig",
    "FaceDetector",
    "PlayerDatabase",
    "open_database",
    "build_database",
    "BuildReport",
    "match_group_photo",
    "MatchResult",
    "group_into_rows",
    "row_labels",
    "format_roster",
]
