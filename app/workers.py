"""QThread workers so model loading, DB building, and matching never freeze the UI.

A single FaceDetector is loaded once on a background thread and reused for both
building and matching.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal

from face_engine import EngineConfig
from face_engine.builder import BuildReport, build_database
from face_engine.detector import FaceDetector
from face_engine.matcher import MatchResult, match_group_photo


class ModelLoader(QThread):
    """Loads the InsightFace model off the UI thread."""

    loaded = Signal(object)  # FaceDetector
    failed = Signal(str)

    def __init__(self, config: EngineConfig) -> None:
        super().__init__()
        self._config = config

    def run(self) -> None:
        try:
            detector = FaceDetector(self._config)
        except Exception as exc:  # noqa: BLE001 - surface any load error to the UI
            self.failed.emit(str(exc))
            return
        self.loaded.emit(detector)


class BuildWorker(QThread):
    """Builds the player database from roster folders + bulk image folders."""

    progress = Signal(str)
    finished_ok = Signal(object)  # BuildReport
    failed = Signal(str)

    def __init__(
        self,
        roster_folders: list[str],
        config: EngineConfig,
        detector: FaceDetector,
        image_folders: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._roster_folders = roster_folders
        self._image_folders = image_folders or []
        self._config = config
        self._detector = detector

    def run(self) -> None:
        try:
            report: BuildReport = build_database(
                self._roster_folders,
                self._config,
                image_folders=self._image_folders,
                detector=self._detector,
                progress=self.progress.emit,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(report)


class MatchWorker(QThread):
    """Identifies faces in one or more group photos, one after another.

    Reuses the single loaded detector for every photo, so a batch of 4-5 photos
    only pays the model-load cost once. Emits per-photo progress and finishes
    with a list of (photo_path, results) pairs.
    """

    progress = Signal(str)
    finished_ok = Signal(object)  # list[tuple[str, list[MatchResult]]]
    failed = Signal(str)

    def __init__(
        self,
        group_photos: list[str],
        config: EngineConfig,
        detector: FaceDetector,
    ) -> None:
        super().__init__()
        self._group_photos = group_photos
        self._config = config
        self._detector = detector

    def run(self) -> None:
        pairs: list[tuple[str, list[MatchResult]]] = []
        total = len(self._group_photos)
        try:
            for idx, photo in enumerate(self._group_photos, start=1):
                from pathlib import Path

                self.progress.emit(
                    f"Identifying photo {idx} of {total}: {Path(photo).name}"
                )
                results: list[MatchResult] = match_group_photo(
                    photo, self._config, detector=self._detector
                )
                pairs.append((photo, results))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(pairs)


# Re-exported so the UI module can type-hint without importing the engine directly.
__all__ = ["ModelLoader", "BuildWorker", "MatchWorker", "QObject"]
