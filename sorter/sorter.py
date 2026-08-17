"""Orchestrate one sort run: reference faces -> match every photo -> move files.

Design priorities (the three reasons this failed years ago):
  * Speed   - one embed per photo on the fast (small) proofs; heavy full-size
              work happens ONLY for the few photos the proofs couldn't place.
  * Safety  - a photo is only filed on a confident, unambiguous match; anything
              uncertain is LEFT in the source (never guessed into a folder).
  * Recall  - two boosts so we don't miss a student who has a tag:
       1) Reference expansion: once a student is confidently identified, their
          matched photos become extra references, so a single dark/angled
          name-tag photo isn't the only anchor.
       2) Full-size fallback: for photos the small proofs can't read/match, we
          re-check the SAME photo's full-size version (better detail for tiny
          tags and small faces), then still move the *proof* file.

Nothing is deleted. 'Move' moves a regular photo into a student folder; tag
photos are COPIED in (the original stays behind as the safety net).
"""

from __future__ import annotations

import os
import shutil
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from face_engine import EngineConfig, FaceDetector
from face_engine.config import IMAGE_EXTENSIONS

from .references import References, build_references, scan_source
from .report import Assignment, SortReport
from .targets import TargetIndex, build_target_index

# How many leftovers to OCR before pausing to re-match. Small enough that one
# student's tag stops their remaining photos being read; large enough that the
# re-match loop isn't run for every single photo.
_OCR_BATCH = 12


def sort_photos(
    source_folders: list[str | Path],
    proof_folder: str | Path,
    config: EngineConfig,
    detector: FaceDetector | None = None,
    *,
    threshold: float = 0.5,
    margin: float = 0.05,
    dry_run: bool = False,
    qr_reader=None,
    ocr_reader=None,
    fullsize_folders: list[str | Path] | None = None,
    use_filename_ids: bool = False,
    progress: Callable[[str], None] | None = None,
) -> SortReport:
    """Sort every regular photo in ``source_folders`` into the Proof tree.

    fullsize_folders: optional folder(s) holding the FULL-SIZE version of each
        photo under the same filename. When given, any photo the proofs can't
        place is re-checked against its full-size version (the proof file is
        still what gets moved). This yields proof speed with full-size accuracy.
    """
    report = SortReport(dry_run=dry_run)
    t0 = time.perf_counter()

    if detector is None:
        _emit(progress, "Loading recognition model...")
        detector = FaceDetector(config)

    # Index targets FIRST: the set of valid IDs constrains/validates OCR reads.
    index = build_target_index(Path(proof_folder))
    valid_ids = set(index.by_id.keys())
    accept_prefixes = {i[:3] for i in valid_ids if len(i) >= 3}
    # No student folders means every tag read gets rejected for not matching a
    # real ID - the run finds nothing and sorts nothing, after doing all the
    # work. Stop now and say why, instead of an hour of silent failure.
    if not valid_ids:
        raise ValueError(
            f"No student folders found under '{proof_folder}'.\n\n"
            "Pick the folder that directly contains 'sorted' and/or 'culled' "
            "(often named 'Proof'), not the folder above it."
        )

    fullsize_index = _index_fullsize(fullsize_folders) if fullsize_folders else {}

    tag_photos, candidates = scan_source(
        [Path(s) for s in source_folders], use_filename_ids=use_filename_ids
    )
    # --- Tag pass 1: QR ONLY, over every photo --------------------------------
    # QR is ~70x faster than OCR (0.05s vs 3.45s per photo, measured), so it is
    # the only thing cheap enough to run on the whole shoot. OCR is NOT run here:
    # it would fire on every photo without a tag - nearly all of them - turning a
    # 1-minute scan into 70+. It runs later, over the far smaller leftover set.
    if qr_reader is not None and candidates:
        regular_photos: list[Path] = []
        total_c = len(candidates)
        for i, photo in enumerate(candidates, start=1):
            if progress is not None and i % 50 == 0:
                _emit(progress, f"Reading QR tags {i}/{total_c}...")
            student_id, kind = qr_reader.read_id(photo, valid_ids, accept_prefixes)
            if student_id is not None:
                tag_photos.append((photo, student_id))
                report.qr_reads += 1
                _count_kind(report, kind)
            else:
                regular_photos.append(photo)
    else:
        regular_photos = list(candidates)

    _emit(
        progress,
        f"Found {len(tag_photos)} name-tag photos and "
        f"{len(regular_photos)} photos to sort.",
    )

    refs = build_references(tag_photos, detector, report, progress=progress)
    report.student_refs = len(set(refs.ids))
    report.regular_total = len(regular_photos)

    # --- Pass 1: match the proofs; collect confident matches to enroll --------
    enroll: list[tuple[str, np.ndarray]] = []  # (id, embedding) of confident matches
    pending: list[tuple[Path, np.ndarray | None]] = []  # (photo, proof embedding)
    # Embed in parallel first, then file sequentially: the slow part scales
    # across cores while the file moves stay ordered and race-free.
    embeddings = _embed_many(detector, regular_photos, progress, "Reading faces")
    for photo, emb in zip(regular_photos, embeddings):
        status, sid, _best = _try_place(
            photo, emb, refs, index, report, threshold, margin, dry_run
        )
        if status == "sorted":
            enroll.append((sid, emb))
        elif status is None:
            pending.append((photo, emb))

    # --- Pass 2 (Fix: reference expansion): re-match pending with extra refs ---
    if enroll and pending:
        refs = _expand(refs, enroll)
        _emit(progress, f"Re-checking {len(pending)} photos with expanded references...")
        still: list[tuple[Path, np.ndarray | None]] = []
        for photo, emb in pending:
            status, _sid, _best = _try_place(
                photo, emb, refs, index, report, threshold, margin, dry_run
            )
            if status is None:
                still.append((photo, emb))
        pending = still

    # --- Tag pass 2: OCR, on the LEFTOVERS ONLY -------------------------------
    # The tag prints the ID as text beside the QR, and that text survives what
    # kills a QR: bend it on fabric, soften it slightly, and the QR's module grid
    # is unreadable while the digits still read. Measured on the client's tags:
    # QR 29/36, OCR 35/36 - and OCR got all 8 tags QR missed, every one an exact
    # folder match.
    #
    # Running it HERE, not in pass 1, is what makes it affordable. By now every
    # student QR found is sorted, so only the genuinely unidentified photos are
    # left: on the client's shoot 357 instead of 1247 - ~21 min instead of ~72.
    # Each new tag becomes a reference, so one OCR hit pulls in that student's
    # whole burst by face.
    #
    # It also stops early. One OCR hit identifies a whole student, and their
    # other ~40 photos then match by FACE - which is instant next to a 3.5s OCR.
    # So we read a small batch, immediately re-match everything still waiting,
    # and drop whatever just got placed out of the OCR queue. On a shoot whose
    # leftovers are a handful of students, that reads a fraction of them.
    if ocr_reader is not None and pending:
        _emit(progress, f"Reading text tags (OCR) on {len(pending)} leftovers...")
        queue = list(pending)
        unread: list[tuple[Path, np.ndarray | None]] = []
        read_count = 0
        while queue:
            batch, queue = queue[:_OCR_BATCH], queue[_OCR_BATCH:]
            recovered: list[tuple[str, np.ndarray]] = []
            for photo, emb in batch:
                report.ocr_attempted += 1
                read_count += 1
                student_id, kind = ocr_reader.read_id(
                    photo, valid_ids, accept_prefixes
                )
                if student_id is None:
                    unread.append((photo, emb))
                    continue
                tag_photos.append((photo, student_id))
                report.ocr_reads += 1
                _count_kind(report, kind)
                if emb is not None:
                    recovered.append((student_id, emb))
            if not recovered:
                # Nothing found in this batch. Still report progress: OCR is
                # ~3.5s/photo, so a silent stretch here reads as a hung app.
                _emit(
                    progress,
                    f"Reading printed IDs: {read_count} done, "
                    f"{len(queue)} to go ({report.ocr_reads} found)...",
                )
                continue
            # New references: re-match everything still queued or already read
            # but unplaced, so their photos never reach the OCR reader at all.
            refs = _expand(refs, recovered)
            report.student_refs = len(set(refs.ids))
            still_queued: list[tuple[Path, np.ndarray | None]] = []
            still_unread: list[tuple[Path, np.ndarray | None]] = []
            for bucket, dest in ((queue, still_queued), (unread, still_unread)):
                for photo, emb in bucket:
                    status, _sid, _best = _try_place(
                        photo, emb, refs, index, report, threshold, margin, dry_run
                    )
                    if status is None:
                        dest.append((photo, emb))
            queue, unread = still_queued, still_unread
            _emit(
                progress,
                f"Reading printed IDs: {read_count} done, {len(queue)} to go "
                f"({report.ocr_reads} found)...",
            )
        _emit(progress, f"OCR read {read_count} photos, found {report.ocr_reads} tags.")
        pending = unread

    # --- Full-size fallback (Fix: re-check just the misses at full res) --------
    if fullsize_index and pending:
        pending = _fullsize_fallback(
            pending, fullsize_index, refs, index, report, detector,
            qr_reader, ocr_reader, valid_ids, accept_prefixes,
            threshold, margin, dry_run, progress,
        )

    # --- Finalize whatever is still unplaced as leftovers ---------------------
    for photo, emb in pending:
        if emb is None:
            report.leftover_no_face.append(str(photo))
        else:
            best = 0.0 if refs.is_empty else refs.best_match(emb)[1]
            report.leftover_low_conf.append((str(photo), best))

    # --- Name-tag photos: copy into the folder, leave original as safety net ---
    for photo, student_id in tag_photos:
        target = index.by_id.get(student_id)
        if target is None:
            report.tag_no_folder.append((str(photo), student_id))
            continue
        dest_dir, _ = _ensure_sorted(target, index, report, dry_run)
        dest = _safe_dest(dest_dir, photo.name)
        if not dry_run:
            shutil.copy2(str(photo), str(dest))  # COPY: the original stays in source

    report.seconds = time.perf_counter() - t0
    return report


def _count_kind(report: SortReport, kind: str | None) -> None:
    """Tally how a tag read resolved: exact folder / snapped / no folder yet."""
    if kind == "exact":
        report.ocr_exact += 1
    elif kind == "fuzzy":
        report.ocr_fuzzy += 1
    else:
        report.ocr_no_folder += 1


def _read_tag(photo, qr_reader, ocr_reader, valid_ids, accept_prefixes):
    """Read a tag's ID, QR first (fast) then text (OCR). Returns (id, kind, via).

    Used by the full-size fallback, which re-checks a handful of photos and can
    afford both readers. The main scan runs the two in separate passes instead.
    """
    if qr_reader is not None:
        sid, kind = qr_reader.read_id(photo, valid_ids, accept_prefixes)
        if sid is not None:
            return sid, kind, "qr"
    if ocr_reader is not None:
        sid, kind = ocr_reader.read_id(photo, valid_ids, accept_prefixes)
        if sid is not None:
            return sid, kind, "ocr"
    return None, None, None


def _try_place(
    photo, embedding, refs, index, report, threshold, margin, dry_run
) -> tuple[str | None, str | None, float]:
    """Try to file ``photo`` using ``embedding``. Moves the file on a confident,
    unambiguous match. Returns (status, student_id, best_score) where status is
    'sorted', 'no_folder', or None (unplaced)."""
    if embedding is None or refs.is_empty:
        return None, None, 0.0
    best_id, best, second = refs.best_match(embedding)
    if best < threshold or (best - second) < margin:
        return None, None, best  # uncertain -> caller keeps it pending
    target = index.by_id.get(best_id)
    if target is None:
        report.leftover_no_folder.append((str(photo), best_id))
        return "no_folder", best_id, best
    dest_dir, promoted = _ensure_sorted(target, index, report, dry_run)
    dest = _safe_dest(dest_dir, Path(photo).name)
    if not dry_run:
        _move_file(photo, dest, report)
    report.assignments.append(Assignment(str(photo), best_id, str(dest), best, promoted))
    return "sorted", best_id, best


def _fullsize_fallback(
    pending, fullsize_index, refs, index, report, detector,
    qr_reader, ocr_reader, valid_ids, accept_prefixes,
    threshold, margin, dry_run, progress,
):
    """Re-check the pending (unplaced) photos against their full-size versions.

    Two steps: recover missed name tags at full size (new references), then
    re-embed the full-size face and re-match. The PROOF file is what gets moved.
    """
    _emit(progress, f"Full-size fallback on {len(pending)} unresolved photos...")

    # Step A: recover missed tags from full-size -> more references.
    if qr_reader is not None or ocr_reader is not None:
        recovered: list[tuple[str, np.ndarray]] = []
        for photo, _emb in pending:
            fs = fullsize_index.get(Path(photo).name)
            if fs is None:
                continue
            sid, _kind, _via = _read_tag(
                fs, qr_reader, ocr_reader, valid_ids, accept_prefixes
            )
            if sid is not None:
                fsface = _safe_embed(detector, fs)
                if fsface is not None:
                    recovered.append((sid, fsface.embedding))
                    report.fullsize_tag_recovered += 1
        if recovered:
            refs = _expand(refs, recovered)

    # Step B: re-embed full-size faces and re-match; move the proof file.
    still = []
    for photo, emb in pending:
        fs = fullsize_index.get(Path(photo).name)
        if fs is None:
            still.append((photo, emb))
            continue
        fsface = _safe_embed(detector, fs)
        if fsface is None:
            still.append((photo, emb))
            continue
        status, _sid, _best = _try_place(
            photo, fsface.embedding, refs, index, report, threshold, margin, dry_run
        )
        if status == "sorted":
            report.recovered_via_fullsize += 1
        elif status is None:
            still.append((photo, emb))
    return still


def _expand(refs: References, pairs: list[tuple[str, np.ndarray]]) -> References:
    """Return a new References with extra (id, embedding) rows appended."""
    pairs = [(i, e) for i, e in pairs if e is not None]
    if not pairs:
        return refs
    extra = np.stack([e for _, e in pairs]).astype(np.float32)
    ids = [i for i, _ in pairs]
    if refs.is_empty:
        return References(matrix=extra, ids=ids)
    return References(matrix=np.vstack([refs.matrix, extra]), ids=list(refs.ids) + ids)


def _index_fullsize(folders) -> dict[str, str]:
    """Map filename -> full path for every image under the full-size folder(s)."""
    idx: dict[str, str] = {}
    for folder in folders:
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames.sort()
            for name in filenames:
                if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                    idx.setdefault(name, str(Path(dirpath) / name))
    return idx


def _ensure_sorted(
    target, index: TargetIndex, report: SortReport, dry_run: bool
) -> tuple[Path, bool]:
    """Return the folder to file into, promoting culled -> sorted on first use."""
    if target.location == "sorted":
        return target.path, False

    dest_class_dir = index.sorted_root / target.class_name
    dest = dest_class_dir / target.path.name
    if not dry_run:
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target.path), str(dest))
    report.promotions.append(f"{target.student_id} ({target.class_name})")
    target.path = dest
    target.location = "sorted"
    return dest, True


def _clear_readonly(path) -> None:
    """Remove the read-only flag so the file can be deleted after copying."""
    try:
        os.chmod(str(path), stat.S_IWRITE)
    except OSError:
        pass


def _move_file(src, dst, report: SortReport) -> None:
    """Move src -> dst reliably, even across drives / OneDrive / read-only files.

    A plain move fails when the source is read-only (Windows won't delete it) or
    lives in a synced folder. We clear read-only first, then move; if that still
    fails, we copy and force-remove the original. Anything that ends up copied
    but not removed is recorded so the user is warned instead of left with silent
    duplicates.
    """
    _clear_readonly(src)
    try:
        shutil.move(str(src), str(dst))
        return
    except (OSError, shutil.Error):
        pass
    # Fallback: make sure the copy exists, then force-remove the source.
    try:
        if not Path(dst).exists():
            shutil.copy2(str(src), str(dst))
    except OSError:
        pass
    _clear_readonly(src)
    try:
        os.remove(str(src))
    except OSError:
        report.move_incomplete.append(str(src))  # copied but couldn't delete original


def _safe_embed(detector: FaceDetector, photo):
    """embed_single but never raises (corrupt/unreadable -> None)."""
    try:
        return detector.embed_single(photo)
    except ValueError:
        return None
    except Exception:  # noqa: BLE001 - one bad file must not kill a whole run
        return None


def _workers() -> int:
    """How many photos to process at once. Leaves cores for the OS and the UI."""
    return max(1, min(6, (os.cpu_count() or 2) - 2))


def _embed_many(
    detector: FaceDetector,
    photos: list[Path],
    progress: Callable[[str], None] | None = None,
    label: str = "Reading faces",
) -> list[np.ndarray | None]:
    """Embed a list of photos in parallel, preserving order.

    Face embedding is the slowest part of a run and each photo is independent,
    so it parallelises cleanly. onnxruntime releases the GIL during inference,
    so plain threads share the one loaded model (no extra 300MB per worker) and
    still scale: measured 1.11s -> 0.46s per photo on 6 threads.

    Only the embedding is parallel. Filing the photos stays strictly sequential
    in the caller, so folder creation and move/copy order never race.
    """
    total = len(photos)
    if total == 0:
        return []
    done = 0

    def one(photo):
        nonlocal done
        face = _safe_embed(detector, photo)
        done += 1
        if progress is not None and done % 50 == 0:
            _emit(progress, f"{label} {done}/{total}...")
        return None if face is None else face.embedding

    with ThreadPoolExecutor(max_workers=_workers()) as pool:
        return list(pool.map(one, photos))


def _safe_dest(dest_dir: Path, filename: str) -> Path:
    """A destination path that never overwrites an existing file."""
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem, suffix = Path(filename).stem, Path(filename).suffix
    n = 1
    while (dest_dir / f"{stem}_{n}{suffix}").exists():
        n += 1
    return dest_dir / f"{stem}_{n}{suffix}"


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
