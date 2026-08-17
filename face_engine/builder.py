"""Build a player face database from two enrollment sources.

Mode A — roster folders (the "source of truth" for names):
    Many subfolders named like 'Molly Brown # 100410 - 571172' or
    'Chase Bruton - 631197'. The folder name gives the player's name and ids.
    These folders are MOSTLY EMPTY (names only) but some may contain reference
    photos, which are embedded just like Phase 1.

Mode B — bulk image folders (~1000 loose images each):
    Individual player photos whose FILENAME contains the player's id (e.g.
    '..._100410_....jpg'). Each image is linked to a player via that id using the
    id->player map built from the roster folders, then embedded. Images whose
    filename has no matching id (e.g. stray group photos) are skipped.

The folder name remains the source of truth — no OCR required.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .config import IMAGE_EXTENSIONS, EngineConfig
from .database import PlayerDatabase, open_database
from .detector import FaceDetector

# Folder names come in a couple of real-world shapes, e.g.:
#   "Molly Brown # 100410 - 571172"   (name, id1, id2)
#   "Chase Bruton - 631197"           (name, single id)
# Rule that covers both: the name is the leading text; IDs are the exactly-6-digit
# numbers; the LAST 6-digit number is the tag printed in the photo and is the
# PRIMARY identifier (player_id). Folders with no 6-digit id (e.g. "ZZClass") are
# not players and are skipped.
SIX_DIGIT_ID = re.compile(r"(?<!\d)\d{6}(?!\d)")  # exactly 6 digits, not part of a longer run
NAME_SPLIT = re.compile(r"#|(?<!\d)\d{6}(?!\d)")  # name = text before first '#' or first id


@dataclass
class BuildReport:
    """Outcome of a database build, surfaced to the user."""

    players_added: int = 0  # players that got at least one embedding
    players_registered: int = 0  # named folders registered (incl. empty)
    photos_embedded: int = 0
    photos_cached: int = 0  # skipped because unchanged
    bulk_images_matched: int = 0  # bulk images linked to a player by filename id
    bulk_unmatched_id: int = 0  # bulk images with an id that matches no player
    bulk_no_id: int = 0  # bulk images with no 6-digit id (e.g. group photos)
    skipped_folders: list[str] = field(default_factory=list)  # bad folder names
    no_face_photos: list[str] = field(default_factory=list)  # no detectable face

    def summary(self) -> str:
        lines = [
            f"Players registered: {self.players_registered}",
            f"Players with photos: {self.players_added}",
            f"Photos embedded: {self.photos_embedded} "
            f"(cached/skipped: {self.photos_cached})",
        ]
        if self.bulk_images_matched or self.bulk_unmatched_id or self.bulk_no_id:
            lines.append(
                f"Bulk images: matched {self.bulk_images_matched}, "
                f"unknown id {self.bulk_unmatched_id}, no id {self.bulk_no_id}"
            )
        if self.skipped_folders:
            lines.append(
                f"Image folders with unrecognized names ({len(self.skipped_folders)}): "
                + ", ".join(self.skipped_folders[:10])
                + ("..." if len(self.skipped_folders) > 10 else "")
            )
        if self.no_face_photos:
            lines.append(f"Photos with no detectable face: {len(self.no_face_photos)}")
        return "\n".join(lines)


def parse_folder_name(folder_name: str) -> tuple[str, str, str | None] | None:
    """Parse a player folder name into (player_id, name, alt_id).

    Handles both real formats:
        "Molly Brown # 100410 - 571172" -> ("571172", "Molly Brown", "100410")
        "Chase Bruton - 631197"         -> ("631197", "Chase Bruton", None)

    player_id is the LAST 6-digit number (the tag shown in the photo, the primary
    identifier). alt_id is the first id when two are present, else None. Returns
    None when the folder has no 6-digit id (e.g. "ZZClass") so it can be skipped.
    """
    ids = SIX_DIGIT_ID.findall(folder_name)
    if not ids:
        return None
    player_id = ids[-1]
    alt_id = ids[0] if len(ids) >= 2 else None
    name = NAME_SPLIT.split(folder_name, maxsplit=1)[0].strip(" -–—_#\t")
    if not name:
        return None
    return player_id, name, alt_id


def _iter_images(folder: Path) -> Iterable[Path]:
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
            yield entry


def _scan_named_folders(parent: Path, report: BuildReport) -> Iterable[Path]:
    """Recursively find every named player folder under `parent`.

    A directory is a player folder when its name parses (see parse_folder_name),
    regardless of whether it currently holds images — roster folders are mostly
    empty and exist to supply names. Image-holding directories whose name does
    NOT parse are reported as skipped so the user can spot mislabeled folders.
    """
    for dirpath, dirnames, filenames in os.walk(parent):
        dirnames.sort()  # deterministic traversal order
        folder = Path(dirpath)
        if folder == parent:
            continue  # the parent itself is a container, not a player
        if parse_folder_name(folder.name) is not None:
            yield folder
            continue
        has_image = any(
            Path(f).suffix.lower() in IMAGE_EXTENSIONS for f in filenames
        )
        if has_image:
            report.skipped_folders.append(folder.name)


def build_database(
    roster_folders: Iterable[str | Path],
    config: EngineConfig,
    image_folders: Iterable[str | Path] = (),
    detector: FaceDetector | None = None,
    progress: Callable[[str], None] | None = None,
) -> BuildReport:
    """Enroll players from roster folders and (optionally) bulk image folders.

    Args:
        roster_folders: Parent folder(s) of named player subfolders
            ('Name # id1 - id2'). These define every player's name and ids, and
            any reference photos they contain are embedded (Mode A). Mostly empty.
        config: Engine configuration (model, db path, etc.).
        image_folders: Parent folder(s) of loose individual photos whose filename
            carries a player's id. Each is linked to a player by that id and
            embedded (Mode B). Filenames with no matching id are skipped.
        detector: Optional pre-loaded detector (reused across calls). Created if None.
        progress: Optional callback for status messages (used by the GUI).

    Returns:
        BuildReport summarizing what happened.
    """
    report = BuildReport()
    owns_detector = detector is None
    if detector is None:
        _emit(progress, "Loading recognition model...")
        detector = FaceDetector(config)

    db = open_database(config)
    try:
        # Mode A: register all named folders, embedding any photos they hold.
        seen_dirs: set[str] = set()
        for parent in roster_folders:
            parent_path = Path(parent)
            if not parent_path.is_dir():
                continue
            for player_dir in _scan_named_folders(parent_path, report):
                key = str(player_dir.resolve())
                if key in seen_dirs:  # avoid double-processing overlapping parents
                    continue
                seen_dirs.add(key)
                player_id, name, alt_id = parse_folder_name(player_dir.name)
                _process_player(
                    db, detector, player_id, name, alt_id, player_dir, report, progress
                )

        # Mode B: enroll loose bulk images, linked to players by filename id.
        image_folder_list = [Path(p) for p in image_folders if Path(p).is_dir()]
        if image_folder_list:
            id_lookup = db.load_id_lookup()
            for parent_path in image_folder_list:
                _process_bulk_folder(
                    db, detector, parent_path, id_lookup, report, progress
                )

        # Stale cleanup spans both sources: drop embeddings whose file is gone.
        db.prune_missing_files()
        db.refresh_all_photo_counts()
        report.players_added = db.players_with_photos()
    finally:
        db.close()
        # FaceAnalysis has no explicit close; let it be GC'd if we created it.
        if owns_detector:
            del detector

    return report


def _process_player(
    db: PlayerDatabase,
    detector: FaceDetector,
    player_id: str,
    name: str,
    alt_id: str | None,
    folder: Path,
    report: BuildReport,
    progress: Callable[[str], None] | None,
) -> None:
    db.upsert_player(player_id, name, str(folder), alt_id)
    report.players_registered += 1
    _emit(progress, f"Registering {player_id} - {name}")

    for image_path in _iter_images(folder):
        _embed_image(db, detector, player_id, image_path, report)


def _process_bulk_folder(
    db: PlayerDatabase,
    detector: FaceDetector,
    parent: Path,
    id_lookup: dict[str, str],
    report: BuildReport,
    progress: Callable[[str], None] | None,
) -> None:
    """Embed loose images, linking each to a player via an id in its filename."""
    _emit(progress, f"Scanning images in {parent.name}...")
    for dirpath, dirnames, filenames in os.walk(parent):
        dirnames.sort()
        for fname in sorted(filenames):
            if Path(fname).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_path = Path(dirpath) / fname
            ids = SIX_DIGIT_ID.findall(image_path.stem)
            if not ids:
                report.bulk_no_id += 1
                continue
            player_id = next((id_lookup[i] for i in ids if i in id_lookup), None)
            if player_id is None:
                report.bulk_unmatched_id += 1
                continue
            if _embed_image(db, detector, player_id, image_path, report):
                report.bulk_images_matched += 1


def _embed_image(
    db: PlayerDatabase,
    detector: FaceDetector,
    player_id: str,
    image_path: Path,
    report: BuildReport,
) -> bool:
    """Embed one image for a player, honoring the mtime cache. True if stored/cached."""
    path_str = str(image_path)
    mtime = image_path.stat().st_mtime
    if db.cached_mtime(path_str) == mtime:
        report.photos_cached += 1
        return True
    try:
        face = detector.embed_single(image_path)
    except ValueError:
        report.no_face_photos.append(path_str)
        return False
    if face is None:
        report.no_face_photos.append(path_str)
        return False
    db.store_embedding(player_id, path_str, mtime, face.embedding)
    report.photos_embedded += 1
    return True


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
