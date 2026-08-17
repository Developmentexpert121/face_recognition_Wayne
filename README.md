# Face Recognition — Photo Sorter & Team Photo ID

Two tools that share one face engine (`face_engine/`):

| Tool | Launcher | What it does |
|---|---|---|
| **Photo Sorter** | `run_sorter.bat` | Sorts a school shoot into per-student folders by reading each student's name tag, then matching their remaining photos by face. |
| **Team Photo ID** | `run_group_photo.bat` | Identifies players in a group/team photo against an enrolled roster. |

Both install from the same `setup.bat` and the same `requirements.txt`.

---

# Photo Sorter

Sorts school photo shoots into per-student folders by reading the name tag each
student wears, then matching every other photo of them by face.

## How it identifies a student

Two passes, in this order — they are not alternatives, they cover different
failures:

1. **QR code, every photo.** ~0.05s/photo. Catches most students in about a
   minute for a full shoot.
2. **Printed ID number (OCR), leftovers only.** ~3.45s/photo. Runs *after* the
   face matching from pass 1 has sorted everything it can, so it only sees the
   still-unidentified photos.

Measured on a real 36-tag shoot: QR read 29/36, OCR read 35/36 — and OCR read
**all 8** tags QR missed. A QR is all-or-nothing (bend it on fabric and it dies);
printed digits degrade gracefully. Running OCR on leftovers only is what keeps it
affordable: 240 photos instead of 1247, minutes instead of over an hour.

Every read is validated against the real Proof-folder IDs, so a bad read can
never invent a wrong student.

## Results

On a 1247-photo shoot: 43 students identified, 1125 photos sorted, 79 left,
~16 minutes.

## Requirements

Python 3.12 (newer versions have no prebuilt wheels for these libraries).
`setup.bat` creates a venv, installs everything, and verifies both tag readers
before reporting success.

## Running

Windows: double-click `setup.bat` once, then `run_sorter.bat`.
See `START HERE.txt` for the operator guide.

CLI:

```
python run_sorter.py --source PROOFS --proof PROOF_FOLDER [--fullsize RAW] --dry-run
```

Always `--dry-run` first — it plans everything and moves nothing.

## Folder layout it expects

```
Proof/
  sorted/  <class>/  <Name - 631478>/
  culled/  <class>/  <Name - 631478>/
```

The student ID is the last 6-digit number in the folder name. Point `--proof` at
the folder containing `sorted`/`culled` (it will also accept the folder above it).

## Notes

- Proofs must be full size (~2000x3000). A proof shrunk to 1000x666 makes the tag
  ~15px wide, which no decoder can read.
- The tag must be visible in at least one frame per student. On drape shoots the
  drape often covers it.
- Nothing is deleted. Tag photos are *copied* into the student folder; the
  original stays in the source as a safety net.

---

# Team Photo ID

Identifies players in a group/team photo. Enrol a roster (one folder of photos
per player, the folder name carrying their ID), then load a group shot and it
labels each detected face against that roster.

```
python run_cli.py --roster PARENT_FOLDER [PARENT_FOLDER ...] --group GROUP.jpg
```

Or double-click `run_group_photo.bat` for the GUI.

**Storage.** Defaults to a local SQLite file — fully offline, nothing to install.
Set the `NEON_DATABASE_URL` environment variable to use a shared Neon/Postgres
database instead, so several machines share one roster. The connection string is
read from the environment at runtime and is never hardcoded or committed.

`neon_transfer.py` seeds and moves the player database between the two backends.

---

## Repo notes

`face_engine/` is shared by both tools — detection, embedding, matching, and the
database backends. Only one copy of the sorter is kept here (the current one);
the older pre-zxing copy that lived alongside Team Photo ID was not carried over,
since its QR reader is the one that found no tags at all.
