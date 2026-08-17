"""Index the Proof folder tree so a student ID maps to their destination folder.

Expected layout (the client's):
    Proof/
      sorted/   <class>/   <Name - 631478>/ ...
      culled/   <class>/   <Name - 631478>/ ...

A student folder is any directory whose name contains a 6-digit ID; the ID is the
LAST 6-digit number (ignore an earlier school number), matching the engine's rule.
Sorted folders win over culled folders when the same ID appears in both.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from face_engine.builder import SIX_DIGIT_ID


@dataclass
class TargetFolder:
    """A student's destination folder in the Proof tree."""

    student_id: str
    path: Path
    class_name: str
    location: str  # 'sorted' or 'culled'


@dataclass
class TargetIndex:
    by_id: dict[str, TargetFolder]
    sorted_root: Path  # where promoted folders go (created on demand)
    culled_root: Path | None


def _find_child_dir(parent: Path, name: str) -> Path | None:
    """Case-insensitive lookup of an immediate child directory by name."""
    if not parent.is_dir():
        return None
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == name:
            return child
    return None


def _resolve_proof_root(proof: Path) -> Path:
    """Accept the folder CONTAINING the Proof folder as well as Proof itself.

    Picking the parent by mistake used to index zero students, and the failure
    was silent and total: with no valid IDs every tag read is rejected, so the
    run reports no tags and sorts nothing. Rather than fail that way, look one
    level down for a directory that does hold 'sorted'/'culled'.
    """
    if _find_child_dir(proof, "sorted") or _find_child_dir(proof, "culled"):
        return proof
    if not proof.is_dir():
        return proof
    for child in sorted(proof.iterdir()):
        if not child.is_dir():
            continue
        if _find_child_dir(child, "sorted") or _find_child_dir(child, "culled"):
            return child
    return proof


def build_target_index(proof_folder: Path) -> TargetIndex:
    """Map every student ID under Proof/sorted and Proof/culled to its folder."""
    proof = _resolve_proof_root(Path(proof_folder))
    sorted_root = _find_child_dir(proof, "sorted")
    culled_root = _find_child_dir(proof, "culled")

    by_id: dict[str, TargetFolder] = {}
    # Index sorted first so it takes precedence over a same-ID culled folder.
    for root, location in ((sorted_root, "sorted"), (culled_root, "culled")):
        if root is None:
            continue
        for dirpath, dirnames, _filenames in os.walk(root):
            dirnames.sort()
            directory = Path(dirpath)
            if directory == root:
                continue
            ids = SIX_DIGIT_ID.findall(directory.name)
            if not ids:
                continue  # class container, not a student folder
            student_id = ids[-1]
            existing = by_id.get(student_id)
            if existing is not None and existing.location == "sorted":
                continue  # keep the sorted folder
            by_id[student_id] = TargetFolder(
                student_id=student_id,
                path=directory,
                class_name=directory.parent.name,
                location=location,
            )

    # Promotions need a sorted root even if none existed yet.
    resolved_sorted_root = sorted_root if sorted_root is not None else proof / "Sorted"
    return TargetIndex(
        by_id=by_id, sorted_root=resolved_sorted_root, culled_root=culled_root
    )
