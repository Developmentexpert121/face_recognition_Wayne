"""QThread workers so model loading and sorting never freeze the GUI."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from face_engine import EngineConfig, FaceDetector
from sorter import sort_photos


class ModelLoader(QThread):
    """Loads the InsightFace model off the UI thread (done once at startup)."""

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


class OcrLoader(QThread):
    """Loads the OCR engine off the UI thread (used when filenames lack IDs)."""

    loaded = Signal(object)  # TagReader
    failed = Signal(str)

    def run(self) -> None:
        try:
            from sorter import TagReader

            reader = TagReader()
        except Exception as exc:  # noqa: BLE001 - OCR is optional; fall back to Tier 1
            self.failed.emit(str(exc))
            return
        self.loaded.emit(reader)


class SortWorker(QThread):
    """Runs one sort pass (dry or real), emitting progress and the SortReport."""

    progress = Signal(str)
    finished_ok = Signal(object)  # SortReport
    failed = Signal(str)

    def __init__(
        self,
        sources: list[str],
        proof: str,
        config: EngineConfig,
        detector: FaceDetector,
        threshold: float,
        margin: float,
        dry_run: bool,
        qr_reader=None,
        ocr_reader=None,
        fullsize_folders: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._sources = [Path(s) for s in sources]
        self._proof = Path(proof)
        self._config = config
        self._detector = detector
        self._threshold = threshold
        self._margin = margin
        self._dry_run = dry_run
        self._qr_reader = qr_reader
        self._ocr_reader = ocr_reader
        self._fullsize = [Path(f) for f in (fullsize_folders or [])]

    def run(self) -> None:
        try:
            report = sort_photos(
                self._sources,
                self._proof,
                self._config,
                detector=self._detector,
                threshold=self._threshold,
                margin=self._margin,
                dry_run=self._dry_run,
                qr_reader=self._qr_reader,
                ocr_reader=self._ocr_reader,
                fullsize_folders=self._fullsize or None,
                progress=self.progress.emit,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(report)
