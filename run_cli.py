"""CLI smoke test for the face engine.

Usage:
    python run_cli.py --roster PARENT_FOLDER [PARENT_FOLDER ...] --group GROUP.jpg
        [--images BULK_FOLDER ...] [--db players.db] [--threshold 0.45]
        [--det 1024] [--no-dedupe] [--gpu]

Enrolls players from named roster folders ('Name # id1 - id2'), optionally adds
bulk individual photos linked by an id in each filename, then identifies faces in
the group photo. Prints "Name (ID) - confidence%" and an Unidentified count, plus
timings — the core thing Phase 1 must validate.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from face_engine import EngineConfig
from face_engine.builder import build_database
from face_engine.detector import FaceDetector
from face_engine.matcher import match_group_photo
from face_engine.rows import format_roster


def main() -> int:
    parser = argparse.ArgumentParser(description="Team photo player identification (Phase 1)")
    parser.add_argument(
        "--roster", "--photos", nargs="+", dest="roster", required=True,
        help="One or more PARENT folders of named player subfolders "
             "('Name # id1 - id2'). Source of truth for names.",
    )
    parser.add_argument(
        "--images", nargs="+", default=[],
        help="Optional PARENT folders of loose individual photos whose filename "
             "carries a player's id.",
    )
    parser.add_argument(
        "--group", nargs="+", required=True,
        help="One or more group/team photos. Each gets its own <name>.txt roster.",
    )
    parser.add_argument("--db", default="players.db", help="SQLite database path.")
    parser.add_argument("--threshold", type=float, default=0.45, help="Match threshold.")
    parser.add_argument("--det", type=int, default=1024, help="Detector size (square).")
    parser.add_argument("--no-dedupe", action="store_true", help="Allow repeated players.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (needs onnxruntime-gpu).")
    args = parser.parse_args()

    config = EngineConfig(
        det_size=(args.det, args.det),
        ctx_id=0 if args.gpu else -1,
        match_threshold=args.threshold,
        dedupe=not args.no_dedupe,
        db_path=Path(args.db),
    )

    group_paths = [Path(g) for g in args.group]
    missing = [p for p in group_paths if not p.is_file()]
    if missing:
        for p in missing:
            print(f"Group photo not found: {p}")
        return 1

    print("Loading recognition model (first run downloads it)...")
    detector = FaceDetector(config)

    print("\n=== Building player database ===")
    t0 = time.perf_counter()
    report = build_database(
        args.roster, config, image_folders=args.images, detector=detector, progress=None
    )
    build_secs = time.perf_counter() - t0
    print(report.summary())
    print(f"Build time: {build_secs:.1f}s")

    for group_path in group_paths:
        print(f"\n=== Identifying group photo: {group_path.name} ===")
        t1 = time.perf_counter()
        results = match_group_photo(str(group_path), config, detector=detector)
        match_secs = time.perf_counter() - t1

        identified = [r for r in results if r.identified]
        unidentified = [r for r in results if not r.identified]

        print(f"Faces detected: {len(results)}")
        for r in identified:
            print(f"  {r.name} ({r.player_id}) - {r.confidence * 100:.1f}%")
        print(f"Unidentified: {len(unidentified)}")
        print(f"Match time: {match_secs:.2f}s")

        print("--- Roster (rows, left to right) ---")
        roster = format_roster(results, gap_factor=config.row_gap_factor)
        print(roster)

        # Save the roster next to the photo as <photo name>.txt
        out_path = group_path.with_suffix(".txt")
        try:
            out_path.write_text(roster + "\n", encoding="utf-8")
            print(f"Roster saved to: {out_path}")
        except OSError as exc:
            print(f"[WARN] Could not save roster to {out_path}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
