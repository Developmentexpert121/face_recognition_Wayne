"""InsightFace wrapper: load the model once, detect faces, produce embeddings.

The model load is expensive, so a single FaceDetector instance should be created
and reused for an entire build/match run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import EngineConfig


@dataclass
class DetectedFace:
    """A single detected face with its location and normalized embedding."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    embedding: np.ndarray  # L2-normalized, shape (512,), float32
    det_score: float


class FaceDetector:
    """Thin wrapper around insightface.app.FaceAnalysis."""

    def __init__(self, config: EngineConfig) -> None:
        # Imported lazily so importing the package doesn't pull in onnxruntime
        # until an actual model is needed.
        from insightface.app import FaceAnalysis

        self._config = config
        # When a GPU is requested (ctx_id >= 0) ask for CUDA first, then CPU.
        # onnxruntime silently falls back to CPU if CUDA isn't available/installed,
        # so this is safe on a CPU-only machine (just prints a harmless warning).
        if config.ctx_id >= 0:
            # Prefer DirectML (works on any Windows GPU, no CUDA install), then
            # CUDA, then CPU. onnxruntime uses the first that's actually available
            # and ignores the rest, so this is safe with whichever build is
            # installed (onnxruntime-directml / -gpu / plain onnxruntime).
            providers = [
                "DmlExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        providers = self._available(providers)
        # Load ONLY the two models this app uses: find the face, embed the face.
        # buffalo_l also ships landmark_2d_106, landmark_3d_68 and genderage, and
        # FaceAnalysis runs every loaded model on every photo. Nothing here reads
        # landmarks or age, so that work was pure waste - dropping it measured
        # 1.99s -> 1.25s per photo (a 1247-photo shoot: 41 min -> 26 min).
        self._app = FaceAnalysis(
            name=config.model_name,
            providers=providers,
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=config.ctx_id, det_size=config.det_size)
        self.active_provider = self._detect_active_provider()

    @staticmethod
    def _available(preferred: list[str]) -> list[str]:
        """Keep only providers this onnxruntime build actually has (CPU always)."""
        try:
            import onnxruntime

            have = set(onnxruntime.get_available_providers())
        except Exception:  # noqa: BLE001
            return ["CPUExecutionProvider"]
        keep = [p for p in preferred if p in have]
        if "CPUExecutionProvider" not in keep:
            keep.append("CPUExecutionProvider")
        return keep

    def _detect_active_provider(self) -> str:
        """Report which execution provider actually loaded (GPU or CPU)."""
        try:
            for model in self._app.models.values():
                provs = model.session.get_providers()
                if provs:
                    return provs[0]
        except Exception:  # noqa: BLE001 - reporting only, never fatal
            pass
        return "CPUExecutionProvider"

    @staticmethod
    def _read_image(image_path: str | Path) -> np.ndarray:
        """Read an image as BGR. Uses imdecode so non-ASCII paths work on Windows."""
        path = Path(image_path)
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            raise ValueError(f"Empty or unreadable image: {path}")
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode image: {path}")
        return img

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.astype(np.float32)
        return (vec / norm).astype(np.float32)

    def _to_detected(self, face) -> DetectedFace:
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        return DetectedFace(
            bbox=(x1, y1, x2, y2),
            embedding=self._normalize(np.asarray(face.embedding, dtype=np.float32)),
            det_score=float(getattr(face, "det_score", 0.0)),
        )

    def detect_all(self, image_path: str | Path) -> list[DetectedFace]:
        """Detect every face in an image (e.g. a group photo)."""
        img = self._read_image(image_path)
        faces = self._app.get(img)
        return [self._to_detected(f) for f in faces]

    def embed_single(self, image_path: str | Path) -> DetectedFace | None:
        """Embed the dominant face in an individual player photo.

        If multiple faces are found (e.g. background people), the largest by
        bounding-box area is used. Returns None if no face is detected.
        """
        img = self._read_image(image_path)
        faces = self._app.get(img)
        if not faces:
            return None
        largest = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        return self._to_detected(largest)
