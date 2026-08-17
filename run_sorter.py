"""CLI for the school photo auto-sorter.

Usage:
    python run_sorter.py --source PHOTOS_FOLDER [PHOTOS_FOLDER ...] --proof PROOF_FOLDER
        [--threshold 0.5] [--margin 0.05] [--det 640] [--gpu] [--dry-run]

Reads each student's 6-digit ID from the name-tag photo filenames, embeds those
faces, then moves every regular photo into the matching student's folder under
Proof/sorted (promoting the folder from Proof/culled when needed). A copy of each
name-tag photo is left in the source as a leftover safety net.

ALWAYS run with --dry-run first: it prints exactly what would happen and moves
nothing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from face_engine import EngineConfig, FaceDetector
from sorter import sort_photos


def main() -> int:
    parser = argparse.ArgumentParser(description="School photo auto-sorter")
    parser.add_argument(
        "--source", nargs="+", required=True,
        help="Folder(s) of unsorted photos (name-tag + regular mixed).",
    )
    parser.add_argument(
        "--proof", required=True,
        help="The Proof folder that contains 'sorted' and 'culled'.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Minimum face similarity to accept a match (higher = stricter).",
    )
    parser.add_argument(
        "--margin", type=float, default=0.05,
        help="Best match must beat the runner-up student by at least this much.",
    )
    parser.add_argument(
        "--det", type=int, default=640,
        help="Detector size (square). Portraits have large faces, so 640 is fast.",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU (needs onnxruntime-gpu).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Plan and print everything but move/copy nothing. Run this first.",
    )
    parser.add_argument(
        "--no-qr", action="store_true",
        help="Disable QR-code reading (the fast path).",
    )
    parser.add_argument(
        "--no-ocr", action="store_true",
        help="Disable text (OCR) tag reading. Combine with QR for maximum speed "
             "once all tags carry a QR code.",
    )
    parser.add_argument(
        "--fullsize", nargs="+", default=[],
        help="Optional folder(s) with the FULL-SIZE version of each photo (same "
             "filenames). Only photos the small proofs can't place are re-checked "
             "at full size — proof speed with full-size accuracy.",
    )
    parser.add_argument(
        "--use-filenames", action="store_true",
        help="Treat a 6-digit number in a photo's FILENAME as the student ID. "
             "Off by default — school photos never carry the ID in the filename, "
             "so a camera/export number must not be mistaken for a student ID.",
    )
    parser.add_argument(
        "--report", metavar="CSV",
        help="Write a per-photo audit CSV (photo, status, student, confidence). "
             "Use with --dry-run to verify accuracy before moving anything.",
    )
    args = parser.parse_args()

    proof = Path(args.proof)
    if not proof.is_dir():
        print(f"Proof folder not found: {proof}")
        return 1
    sources = [Path(s) for s in args.source]
    missing = [s for s in sources if not s.is_dir()]
    if missing:
        for s in missing:
            print(f"Source folder not found: {s}")
        return 1

    config = EngineConfig(
        det_size=(args.det, args.det),
        ctx_id=0 if args.gpu else -1,
        match_threshold=args.threshold,
    )

    if args.dry_run:
        print(">>> DRY RUN — no files will be moved or copied. <<<\n")

    print("Loading recognition model (first run downloads it)...")
    detector = FaceDetector(config)
    prov = getattr(detector, "active_provider", "")
    device = "GPU" if ("CUDA" in prov or "Dml" in prov) else "CPU"
    print(f"Running on: {device} ({getattr(detector, 'active_provider', '?')})")

    qr_reader = None
    if not args.no_qr:
        from sorter.qr_reader import QRReader

        qr_reader = QRReader()  # instant, no model/download

    ocr_reader = None
    if not args.no_ocr:
        try:
            from sorter.tag_reader import TagReader

            print("Loading OCR engine for name tags...")
            ocr_reader = TagReader()
        except Exception as exc:  # noqa: BLE001 - QR / filename still work
            print(f"[warn] OCR unavailable ({exc}); using QR/filename only.")

    fullsize = [Path(f) for f in args.fullsize if Path(f).is_dir()]

    report = sort_photos(
        sources,
        proof,
        config,
        detector=detector,
        threshold=args.threshold,
        margin=args.margin,
        dry_run=args.dry_run,
        qr_reader=qr_reader,
        ocr_reader=ocr_reader,
        fullsize_folders=fullsize or None,
        use_filename_ids=args.use_filenames,
        progress=print,
    )

    print()
    print(report.summary())
    if args.report:
        report.write_csv(args.report)
        print(f"\nPer-photo audit written to: {args.report}")
    if args.dry_run:
        print("\n(Dry run — re-run without --dry-run to actually move the photos.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
