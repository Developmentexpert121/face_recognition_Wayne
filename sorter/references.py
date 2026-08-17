"""Build in-memory reference faces from the name-tag photos in the source.

A photo is a NAME-TAG photo when its filename contains a 6-digit student ID (the
same "last 6-digit number is the ID" rule the engine already uses elsewhere).
Those photos give us a face + the student's ID; every other photo is a regular
photo to be sorted.

The references live in memory only (one-shot job — no SQLite/Neon): a stacked
matrix of L2-normalized embeddings plus a parallel list of student IDs, matched
with a plain dot product.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from face_engine.builder import SIX_DIGIT_ID
from face_engine.config import IMAGE_EXTENSIONS
from face_engine.detector import FaceDetector

from .report import SortReport

EMBEDDING_DIM = 512


@dataclass
class References:
    """All name-tag faces stacked for fast matching."""

    matrix: np.ndarray  # (N, 512) float32, L2-normalized rows
    ids: list[str]  # length N, the student ID for each row

    @property
    def is_empty(self) -> bool:
        return self.matrix.shape[0] == 0

    def best_match(self, face_emb: np.ndarray) -> tuple[str, float, float]:
        """Best student for one face: (student_id, best_score, second_best_score).

        Reduces every reference photo to the best score PER student, then returns
        the top student and the runner-up's score so the caller can apply a margin
        tie-break. Assumes ``not is_empty``.
        """
        sims = self.matrix @ face_emb  # (N,)
        best_per_id: dict[str, float] = {}
        for sid, sim in zip(self.ids, sims):
            s = float(sim)
            if s > best_per_id.get(sid, -1.0):
                best_per_id[sid] = s
        ranked = sorted(best_per_id.items(), key=lambda kv: kv[1], reverse=True)
        best_id, best_score = ranked[0]
        second_best = ranked[1][1] if len(ranked) > 1 else -1.0
        return best_id, best_score, second_best


def scan_source(
    source_folders: list[str | Path],
    use_filename_ids: bool = False,
) -> tuple[list[tuple[Path, str]], list[Path]]:
    """Split every image under the source folder(s) into (name-tag, regular).

    use_filename_ids: if False (the default for school photos), filenames are
        IGNORED entirely — a 6-digit camera/export number in a filename must NOT
        be mistaken for a student ID. Tags are then found only by reading the
        actual tag in the image (OCR). Set True only when the filename really
        does carry the student's ID.

    Returns (tag_photos, regular_photos): tag_photos is a list of (path, id);
    regular_photos have no filename ID (and get OCR'd for a tag).
    """
    tag_photos: list[tuple[Path, str]] = []
    regular_photos: list[Path] = []
    for folder in source_folders:
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames.sort()
            for fname in sorted(filenames):
                path = Path(dirpath) / fname
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                ids = SIX_DIGIT_ID.findall(path.stem) if use_filename_ids else []
                if ids:
                    tag_photos.append((path, ids[-1]))  # last 6-digit = student ID
                else:
                    regular_photos.append(path)
    return tag_photos, regular_photos


def build_references(
    tag_photos: list[tuple[Path, str]],
    detector: FaceDetector,
    report: SortReport,
    progress: Callable[[str], None] | None = None,
) -> References:
    """Embed the face in each name-tag photo into an in-memory reference set."""
    embeddings: list[np.ndarray] = []
    ids: list[str] = []
    total = len(tag_photos)
    for i, (path, student_id) in enumerate(tag_photos, start=1):
        if progress is not None and i % 25 == 0:
            progress(f"Reading name tags {i}/{total}...")
        try:
            face = detector.embed_single(path)
        except ValueError:
            face = None  # unreadable/corrupt image
        if face is None:
            report.tag_no_face.append(str(path))
            continue
        embeddings.append(face.embedding)
        ids.append(student_id)
        report.reference_photos += 1

    if embeddings:
        matrix = np.stack(embeddings).astype(np.float32)
    else:
        matrix = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    return References(matrix=matrix, ids=ids)
