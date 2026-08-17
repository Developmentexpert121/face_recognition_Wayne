"""Read a student's 6-digit ID off the name-tag in a photo (OCR fallback).

Used only when the filename has no ID. The read is always validated against the
KNOWN set of student IDs (the Proof folder names): an exact hit is trusted, an
almost-hit is snapped to the single nearest real ID, and anything else is
rejected — so a bad OCR read can never invent a wrong student/folder.

OCR runs on RapidOCR (onnxruntime) — offline, no heavy framework, reuses the
runtime already installed. Imported lazily so Tier-1 (filename) installs need it
not at all.
"""

from __future__ import annotations

import re

import numpy as np

from face_engine.detector import FaceDetector

# Common OCR letter->digit confusions, applied only to mostly-numeric text lines
# (so a student's NAME isn't turned into digits).
_DIGIT_MAP = str.maketrans(
    {
        "O": "0", "o": "0", "Q": "0", "D": "0",
        "I": "1", "l": "1", "i": "1", "L": "1", "|": "1",
        "Z": "2", "z": "2",
        "A": "4",
        "S": "5", "s": "5",
        "G": "6", "b": "6",
        "T": "7",
        "B": "8",
        "g": "9", "q": "9",
    }
)


def _runs_of_six(text: str) -> list[str]:
    out: list[str] = []
    for run in re.findall(r"\d{6,}", text):
        if len(run) == 6:
            out.append(run)
        else:  # longer run (e.g. two numbers stuck together) — try both ends
            out.append(run[:6])
            out.append(run[-6:])
    return out


def _extract_candidate_ids(text: str) -> list[str]:
    """Pull 6-digit candidates from one OCR text line.

    Two passes, both validated against real folder IDs downstream so neither can
    invent a wrong student:
      1) raw digit runs from ANY line — catches clean reads embedded in a line
         with other text (e.g. "613504 Al Jammoor"), which the old strict
         numeric-only filter was dropping (the recall bug).
      2) on mostly-numeric lines, also apply the digit-confusion fixes
         (B->8, I->1, ...) so slightly-misread numbers still resolve.
    """
    out: list[str] = list(_runs_of_six(text))  # pass 1: safe raw digits, any line
    alnum = [ch for ch in text if ch.isalnum()]
    if alnum and sum(ch.isdigit() for ch in alnum) / len(alnum) >= 0.5:
        out.extend(_runs_of_six(text.translate(_DIGIT_MAP)))  # pass 2: numeric lines
    return out


def _levenshtein_le1(a: str, b: str) -> bool:
    """True if a and b are within edit distance 1 (sub/ins/del)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:  # at most one substitution
        return sum(x != y for x, y in zip(a, b)) == 1
    # one insertion/deletion: the shorter must be a subsequence missing one char
    shorter, longer = (a, b) if la < lb else (b, a)
    i = j = diff = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
        else:
            diff += 1
            j += 1
            if diff > 1:
                return False
    return True


def _snap_to_valid(candidate: str, valid_ids: set[str]) -> str | None:
    """Return the single valid ID within edit distance 1, else None (ambiguous)."""
    matches = [v for v in valid_ids if _levenshtein_le1(candidate, v)]
    return matches[0] if len(matches) == 1 else None


def _resize_max(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    import cv2

    scale = max_dim / longest
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


class TagReader:
    """Reads a tag ID from an image, validated against the known student IDs."""

    def __init__(self, max_dim: int = 2600) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency hint
            raise RuntimeError(
                "OCR needs the 'rapidocr-onnxruntime' package. Install it with: "
                "pip install -r requirements.txt"
            ) from exc
        self._ocr = RapidOCR()
        self._max_dim = max_dim

    def read_id(
        self,
        image_path,
        folder_ids: set[str],
        accept_prefixes: set[str] | None = None,
    ) -> tuple[str | None, str | None]:
        """Read the tag's student ID.

        Returns (student_id, kind):
          'exact'    - read matches a real folder ID (will be sorted).
          'fuzzy'    - snapped to the single nearest real folder ID (will be sorted).
          'no_folder'- a clean 6-digit number whose 3-digit prefix matches the
                       session's folders but has no folder yet (recognized, but
                       Wayne must create the folder; NOT sorted).
          (None,None)- nothing trustworthy found.
        """
        try:
            img = FaceDetector._read_image(image_path)
        except ValueError:
            return None, None
        img = _resize_max(img, self._max_dim)

        result, _ = self._ocr(img)
        if not result:
            return None, None

        candidates: list[str] = []
        for item in result:
            candidates.extend(_extract_candidate_ids(item[1]))

        for cand in candidates:  # 1) exact, trusted folder hit
            if cand in folder_ids:
                return cand, "exact"
        for cand in candidates:  # 2) snap a near-miss to the one real folder ID
            snapped = _snap_to_valid(cand, folder_ids)
            if snapped is not None:
                return snapped, "fuzzy"
        if accept_prefixes:  # 3) recognized student number, but no folder yet
            for cand in candidates:
                if len(cand) == 6 and cand[:3] in accept_prefixes:
                    return cand, "no_folder"
        return None, None
