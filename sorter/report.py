"""Outcome of a sort run, surfaced to the user (and printed by the CLI).

Everything the operator needs to trust the run lives here: what was sorted, what
was deliberately left behind (and why), which student folders were promoted from
'culled' to 'sorted', and timing. Nothing is silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Assignment:
    """One regular photo assigned to a student folder."""

    photo: str
    student_id: str
    dest: str
    confidence: float  # best cosine similarity in [-1, 1]
    promoted: bool = False  # the target folder was moved culled -> sorted for this


@dataclass
class SortReport:
    """Full result of a sort run.

    Leftovers are split by reason so the operator can act on each: a low-face
    photo may just be a bad shot, a no-folder photo means a missing proof folder.
    """

    # references (name-tag photos)
    student_refs: int = 0  # distinct student IDs with at least one reference face
    reference_photos: int = 0  # name-tag photos successfully embedded
    tag_no_face: list[str] = field(default_factory=list)  # tag photo, no detectable face
    tag_no_folder: list[tuple[str, str]] = field(default_factory=list)  # (photo, id) no folder

    # tag scanning (photos with no filename ID, checked for a QR / text tag)
    ocr_attempted: int = 0  # photos scanned for a tag
    qr_reads: int = 0  # tags identified via a QR code (fast path)
    ocr_reads: int = 0  # tags identified via text reading (OCR fallback)
    ocr_exact: int = 0  # read matched a real folder ID exactly
    ocr_fuzzy: int = 0  # read snapped to the nearest real folder ID
    ocr_no_folder: int = 0  # read a valid student number that has no folder yet

    # full-size fallback recovery (photos the small proofs couldn't place)
    recovered_via_fullsize: int = 0  # photos sorted only after the full-size re-check
    fullsize_tag_recovered: int = 0  # name tags read only at full size

    # regular photos
    regular_total: int = 0
    assignments: list[Assignment] = field(default_factory=list)  # sorted (or planned)
    leftover_no_face: list[str] = field(default_factory=list)  # no detectable face
    leftover_low_conf: list[tuple[str, float]] = field(default_factory=list)  # (photo, best)
    leftover_no_folder: list[tuple[str, str]] = field(default_factory=list)  # (photo, id)

    promotions: list[str] = field(default_factory=list)  # "id (class)" moved culled->sorted
    move_incomplete: list[str] = field(default_factory=list)  # copied but source not removed
    dry_run: bool = False
    seconds: float = 0.0

    @property
    def sorted_count(self) -> int:
        return len(self.assignments)

    @property
    def leftover_count(self) -> int:
        return (
            len(self.leftover_no_face)
            + len(self.leftover_low_conf)
            + len(self.leftover_no_folder)
        )

    def summary(self) -> str:
        mode = "DRY RUN (nothing moved)" if self.dry_run else "DONE"
        lines = [
            f"=== Sort {mode} ===",
            f"Name-tag references: {self.student_refs} students "
            f"({self.reference_photos} tag photos embedded)",
        ]
        if self.ocr_attempted:
            lines.append(
                f"  tags read: {self.qr_reads} via QR, {self.ocr_reads} via text "
                f"(scanned {self.ocr_attempted})"
            )
        lines += [
            f"Photos to sort: {self.regular_total}",
            f"  Sorted into folders: {self.sorted_count}",
        ]
        if self.recovered_via_fullsize or self.fullsize_tag_recovered:
            lines.append(
                f"    (of those, {self.recovered_via_fullsize} recovered via full-size; "
                f"{self.fullsize_tag_recovered} tags read only at full size)"
            )
        lines += [
            f"  Left in source folder: {self.leftover_count}",
            f"    - recognized, folder not created yet: {len(self.leftover_no_folder)}",
            f"    - unidentified (no readable name tag): {len(self.leftover_low_conf)}",
            f"    - no clear face (group/blank shots): {len(self.leftover_no_face)}",
        ]
        # Actionable for Wayne: which recognized students still need a folder.
        per_student = self._no_folder_by_student()
        if per_student:
            lines.append(
                f"  >> Create folders for these recognized students "
                f"({len(per_student)}):"
            )
            for sid, n in per_student:
                lines.append(f"        {sid}: {n} photos")
        if self.promotions:
            lines.append(
                f"Folders moved culled -> sorted: {len(self.promotions)} "
                f"({', '.join(self.promotions[:8])}"
                + ("..." if len(self.promotions) > 8 else "")
                + ")"
            )
        if self.tag_no_face:
            lines.append(
                f"[!] Name-tag photos with NO detectable face: {len(self.tag_no_face)} "
                f"(these students have no reference; check focus/exposure)"
            )
        if self.move_incomplete:
            lines.append(
                f"[!] {len(self.move_incomplete)} photos were copied but the original "
                f"could NOT be deleted (likely read-only or OneDrive-locked). "
                f"Un-tick 'Read-only' on the source folder, or sort outside OneDrive."
            )
        lines.append(f"Time: {self.seconds:.1f}s")
        return "\n".join(lines)

    def _no_folder_by_student(self) -> list[tuple[str, int]]:
        """Recognized students with no folder yet -> (id, photo_count), biggest first.

        Combines their matched photos and their name-tag photo so the count is the
        student's full photo set that still needs a folder.
        """
        from collections import Counter

        counts: Counter[str] = Counter()
        for _photo, sid in self.leftover_no_folder:
            counts[sid] += 1
        for _photo, sid in self.tag_no_folder:
            counts[sid] += 1
        return counts.most_common()

    def write_csv(self, path) -> None:
        """Write a per-photo audit (one row per photo) for human verification.

        Columns: photo, status, student_id, confidence, detail. Lets the operator
        confirm every sorted photo went to the right student and review exactly
        which photos were left in the source (and why) before any real move.
        """
        import csv

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["photo", "status", "student_id", "confidence", "detail"])
            for a in self.assignments:
                writer.writerow(
                    [a.photo, "sorted", a.student_id, f"{a.confidence:.3f}", a.dest]
                )
            for photo, sid in self.leftover_no_folder:
                writer.writerow(
                    [photo, "recognized_no_folder", sid, "",
                     "face recognized; create folder for this ID then re-run"]
                )
            for photo, sid in self.tag_no_folder:
                writer.writerow(
                    [photo, "recognized_no_folder", sid, "",
                     "name-tag photo; create folder for this ID then re-run"]
                )
            for photo, best in self.leftover_low_conf:
                writer.writerow(
                    [photo, "unidentified", "", f"{best:.3f}", "no readable tag; kept in source"]
                )
            for photo in self.leftover_no_face:
                writer.writerow(
                    [photo, "no_face", "", "", "group/blank shot; kept in source"]
                )
