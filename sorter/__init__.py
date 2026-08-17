"""School photo auto-sorter.

Reuses the face_engine recognition core to file unsorted individual student
photos into their per-student folders inside a Proof tree. Local, offline,
one-shot: each run is independent and keeps no database.
"""

from .references import References, build_references, scan_source
from .report import Assignment, SortReport
from .sorter import sort_photos
from .targets import TargetFolder, TargetIndex, build_target_index

__all__ = [
    "sort_photos",
    "SortReport",
    "Assignment",
    "References",
    "build_references",
    "scan_source",
    "TargetFolder",
    "TargetIndex",
    "build_target_index",
    "TagReader",
    "QRReader",
]


def __getattr__(name):
    # Lazy so importing `sorter` doesn't require the OCR dependency.
    if name == "TagReader":
        from .tag_reader import TagReader

        return TagReader
    if name == "QRReader":
        from .qr_reader import QRReader

        return QRReader
    raise AttributeError(name)
