"""Match faces in a group photo against the player database.

Embeddings are L2-normalized, so cosine similarity is a plain dot product.
For each detected face we compute its best similarity to every stored photo,
reduce to the best score per player, and assign the highest-scoring player
(subject to the confidence threshold). Optional greedy de-dup prevents the
same player being assigned to two different faces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EngineConfig
from .database import LoadedEmbeddings, open_database
from .detector import DetectedFace, FaceDetector

UNIDENTIFIED = "Unidentified"


@dataclass
class MatchResult:
    """One detected face and its assigned identity."""

    bbox: tuple[int, int, int, int]
    player_id: str | None
    name: str
    confidence: float  # best cosine similarity in [-1, 1]
    det_score: float

    @property
    def identified(self) -> bool:
        return self.player_id is not None


def _best_per_player(
    face_emb: np.ndarray, db: LoadedEmbeddings
) -> dict[str, float]:
    """Best cosine similarity between one face and each player's photos."""
    sims = db.matrix @ face_emb  # (N,)
    best: dict[str, float] = {}
    for pid, sim in zip(db.player_ids, sims):
        s = float(sim)
        if s > best.get(pid, -1.0):
            best[pid] = s
    return best


def match_faces(
    faces: list[DetectedFace],
    db: LoadedEmbeddings,
    config: EngineConfig,
) -> list[MatchResult]:
    """Assign identities to already-detected faces."""
    if db.is_empty:
        return [
            MatchResult(f.bbox, None, UNIDENTIFIED, 0.0, f.det_score) for f in faces
        ]

    # Candidate (face_index, player_id, score) above threshold, best score first.
    candidates: list[tuple[int, str, float]] = []
    per_face_best: dict[int, tuple[str, float]] = {}
    for i, face in enumerate(faces):
        best = _best_per_player(face.embedding, db)
        if not best:
            continue
        top_pid = max(best, key=best.get)
        top_score = best[top_pid]
        per_face_best[i] = (top_pid, top_score)
        if top_score >= config.match_threshold:
            for pid, score in best.items():
                if score >= config.match_threshold:
                    candidates.append((i, pid, score))

    candidates.sort(key=lambda c: c[2], reverse=True)

    assigned_face: dict[int, tuple[str, float]] = {}
    used_players: set[str] = set()
    for face_idx, pid, score in candidates:
        if face_idx in assigned_face:
            continue
        if config.dedupe and pid in used_players:
            continue
        assigned_face[face_idx] = (pid, score)
        used_players.add(pid)

    results: list[MatchResult] = []
    for i, face in enumerate(faces):
        if i in assigned_face:
            pid, score = assigned_face[i]
            results.append(
                MatchResult(face.bbox, pid, db.names.get(pid, pid), score, face.det_score)
            )
        else:
            # Surface the unmatched top score for tuning visibility.
            conf = per_face_best.get(i, (None, 0.0))[1]
            results.append(
                MatchResult(face.bbox, None, UNIDENTIFIED, conf, face.det_score)
            )
    return results


def match_group_photo(
    group_photo_path: str,
    config: EngineConfig,
    detector: FaceDetector | None = None,
) -> list[MatchResult]:
    """Detect faces in a group photo and match them against the saved database."""
    if detector is None:
        detector = FaceDetector(config)
    faces = detector.detect_all(group_photo_path)
    db = open_database(config)
    try:
        loaded = db.load_all()
    finally:
        db.close()
    results = match_faces(faces, loaded, config)
    # Left-to-right by bbox x for stable, readable output (not row logic — Phase 2).
    results.sort(key=lambda r: r.bbox[0])
    return results
