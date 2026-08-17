"""Read a student's ID from a QR code on the name tag — the fast path.

Finding and decoding a QR is far faster than reading text (OCR) and is exact
(no misreads, no angle/blur/upside-down problems), so this is the speed win: a
photo with a QR is identified almost instantly, and a photo without one is
dismissed just as fast.

DECODER: zxing-cpp, not OpenCV. This matters, and it was the bug. The name tag
is a small sticker — roughly 1.5% of frame width in a full-size portrait, about
100px of QR in a 24-megapixel frame. OpenCV's QRCodeDetector is unreliable at
that size: measured on the client's own photos it succeeded at 4000/3000/2000px
but FAILED at full resolution and at 5000/2600px on the same image, with no
consistent pattern, and missed some tags at every scale and both entry points.
The result was ZERO tags found across a real shoot — no references, so nothing
sorted. zxing-cpp read the same tags at every scale, ~10x faster, including one
photo OpenCV could not read at all.

OpenCV is kept as a fallback so a machine without the zxing wheel still runs
(degraded, not broken).

The decoded value is still validated against the real Proof-folder IDs (same
rule as the text reader), so a stray QR can never invent a wrong student.
"""

from __future__ import annotations

import cv2

from face_engine.detector import FaceDetector

from .tag_reader import _runs_of_six, _snap_to_valid

try:  # optional: the program still runs (less accurately) without it
    import zxingcpp

    _HAVE_ZXING = True
except ImportError:  # pragma: no cover - depends on the install
    zxingcpp = None
    _HAVE_ZXING = False


class QRReader:
    """Decode a name-tag QR into a validated student ID."""

    def __init__(self, max_dim: int | None = None) -> None:
        self._detector = cv2.QRCodeDetector()  # fallback only
        # Do NOT downscale by default. Measured end-to-end on the client's
        # photos: full resolution costs ~1.1s/photo (load + decode) and reads
        # every tag we know of, while halving the resolution drops tags and
        # saves only ~0.6s. Downscaling here trades recall for very little.
        self._max_dim = max_dim

    @property
    def engine(self) -> str:
        """Which decoder is actually in use — worth surfacing in the report."""
        return "zxing-cpp" if _HAVE_ZXING else "opencv (degraded)"

    def read_id(
        self,
        image_path,
        folder_ids: set[str],
        accept_prefixes: set[str] | None = None,
    ) -> tuple[str | None, str | None]:
        """Return (student_id, kind) from a QR code, or (None, None) if no usable
        QR is found. kind is 'exact' (matches a folder), 'fuzzy' (snapped to the
        nearest folder), or 'no_folder' (valid number, no folder yet)."""
        try:
            img = FaceDetector._read_image(image_path)
        except ValueError:
            return None, None

        if self._max_dim is not None:
            h, w = img.shape[:2]
            if max(h, w) > self._max_dim:
                scale = self._max_dim / max(h, w)
                img = cv2.resize(
                    img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )

        payloads = self._decode(img)
        if not payloads:
            return None, None

        candidates: list[str] = []
        for payload in payloads:
            candidates.extend(_runs_of_six(payload))

        for cand in candidates:
            if cand in folder_ids:
                return cand, "exact"
        for cand in candidates:
            snapped = _snap_to_valid(cand, folder_ids)
            if snapped is not None:
                return snapped, "fuzzy"
        if accept_prefixes:
            for cand in candidates:
                if len(cand) == 6 and cand[:3] in accept_prefixes:
                    return cand, "no_folder"
        return None, None

    def _decode(self, img) -> list[str]:
        """Every QR payload in the frame.

        When zxing is present its answer is FINAL — including "no QR here". Do
        not fall through to OpenCV on an empty result: most photos have no tag,
        so that fallback would fire on nearly every photo and cost ~14s each on
        a 24MP frame (measured), turning a 13-minute run into 6 hours. OpenCV is
        for the case where the zxing wheel is missing entirely.
        """
        if not _HAVE_ZXING:
            return self._decode_opencv(img)
        try:
            results = zxingcpp.read_barcodes(
                img,
                formats=zxingcpp.BarcodeFormat.QRCode,
                # try_rotate: tags are often worn upside down or sideways.
                try_rotate=True,
                # try_downscale: lets zxing hunt at several scales itself,
                # which is what makes it robust to the tag's size.
                try_downscale=True,
            )
        except Exception:  # noqa: BLE001 - never let a decode kill the run
            return []
        return [r.text for r in results if r.text]

    def _decode_opencv(self, img) -> list[str]:
        """Fallback decoder. Tries MULTI then SINGLE: the two OpenCV entry points
        genuinely disagree on tags this small, and neither dominates."""
        try:
            ok, decoded, _points, _straight = self._detector.detectAndDecodeMulti(img)
        except cv2.error:
            ok, decoded = False, []
        found = [d for d in decoded if d] if ok else []
        if found:
            return found
        try:
            data, _points, _straight = self._detector.detectAndDecode(img)
        except cv2.error:
            return []
        return [data] if data else []
