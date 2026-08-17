"""Phase 2: group identified faces into rows and produce a left-to-right roster.

Team photos arrange people in horizontal rows (back/middle/front). A face's
vertical center tells us which row it's in; its horizontal center orders it
within the row. Output looks like:

    Back Row L to R: Susan Moore, Tim Collins, Frank Wright
    Middle Row L to R: Mike Turner, Leslie Simms
    Front Row L to R: Anna White, Rachel Green

If everyone stands in a single row, there is no Back/Front and we just emit:

    L to R: Susan Moore, Tim Collins, Frank Wright

Why this is not a simple "bucket by y" problem
----------------------------------------------
Two pitfalls show up on real team photos:

1. A single standing rank contains players of very different heights, so their
   head centers span more than a face height even though it is ONE row. Naive
   thresholding shatters it into spurious 1-2 person rows.
2. When players DO stand on risers, the back row is often curved/tilted and its
   end players dip into the middle row's height band, so a flat horizontal cut
   cannot separate them.

So we (a) split into coarse bands only at clear vertical GAPS, then (b) within a
band that is tall enough to possibly hold stacked rows, detrend the row's
curve (fit y as a function of x) and split the residuals only when they are
genuinely bimodal — i.e. there really is a second rank, not just tall kids.
"""

from __future__ import annotations

from statistics import median

import numpy as np

from .matcher import MatchResult

UNIDENTIFIED_LABEL = "Unidentified"

# --- tuning knobs (kept conservative to avoid inventing rows) ---------------
# Coarse bands break wherever consecutive vertical centers differ by more than
# COARSE_GAP_FACTOR * median_face_height.
COARSE_GAP_FACTOR = 0.7
# A band is only considered for a back/middle split if it is at least this many
# median face-heights tall (a single rank rarely exceeds ~1 face height).
SPLIT_SPAN_FACTOR = 1.2
# The two residual clusters must be separated by at least this fraction of a
# face height AND be clearly apart relative to their own spread, or we treat the
# band as one row.
SPLIT_MIN_GAP_FACTOR = 0.45
SPLIT_GAP_OVER_SPREAD = 1.5


def _vertical_center(r: MatchResult) -> float:
    return (r.bbox[1] + r.bbox[3]) / 2.0


def _horizontal_center(r: MatchResult) -> float:
    return (r.bbox[0] + r.bbox[2]) / 2.0


def _face_height(r: MatchResult) -> float:
    return max(1.0, float(r.bbox[3] - r.bbox[1]))


def _split_band(band: list[MatchResult], face_h: float) -> list[list[MatchResult]]:
    """Split one coarse band into stacked rows if (and only if) it holds them.

    Fits the band's vertical curve y=f(x) and clusters the residuals. A real
    second rank shows up as a clearly separated cluster of residuals; height
    variation within a single rank does not. Returns 1 or 2 rows.
    """
    if len(band) < 4:
        return [band]
    ys = np.array([_vertical_center(r) for r in band], dtype=float)
    if ys.max() - ys.min() < SPLIT_SPAN_FACTOR * face_h:
        return [band]

    xs = np.array([_horizontal_center(r) for r in band], dtype=float)
    deg = 2 if len(band) >= 6 else 1
    residual = ys - np.polyval(np.polyfit(xs, ys, deg), xs)

    order = np.argsort(residual)
    sorted_res = residual[order]
    # Best contiguous 2-way split of the sorted residuals (1-D k-means, k=2).
    best_cut, best_cost = 1, None
    for cut in range(1, len(sorted_res)):
        lo, hi = sorted_res[:cut], sorted_res[cut:]
        cost = float(((lo - lo.mean()) ** 2).sum() + ((hi - hi.mean()) ** 2).sum())
        if best_cost is None or cost < best_cost:
            best_cost, best_cut = cost, cut

    gap = sorted_res[best_cut] - sorted_res[best_cut - 1]
    spread_lo = sorted_res[:best_cut].std() if best_cut > 1 else 0.0
    spread_hi = sorted_res[best_cut:].std() if len(sorted_res) - best_cut > 1 else 0.0
    convincing = gap >= SPLIT_MIN_GAP_FACTOR * face_h and gap >= SPLIT_GAP_OVER_SPREAD * (
        spread_lo + spread_hi
    )
    if not convincing:
        return [band]

    higher = set(order[:best_cut].tolist())  # smaller residual = higher = back row
    back = [band[i] for i in range(len(band)) if i in higher]
    front = [band[i] for i in range(len(band)) if i not in higher]
    return [back, front]


def group_into_rows(
    results: list[MatchResult], gap_factor: float = COARSE_GAP_FACTOR
) -> list[list[MatchResult]]:
    """Cluster faces into rows (back→front), each row ordered left→right.

    Step 1 breaks the faces into coarse bands at clear vertical gaps (more than
    gap_factor * median_face_height between adjacent faces). Step 2 splits any
    band tall enough to hold a back+middle pair, but only when the head heights
    are genuinely bimodal. Rows are returned top-to-bottom (back row first).
    """
    if not results:
        return []
    if len(results) == 1:
        return [list(results)]

    face_h = median(_face_height(r) for r in results)
    by_y = sorted(results, key=_vertical_center)

    bands: list[list[MatchResult]] = [[by_y[0]]]
    for face in by_y[1:]:
        if _vertical_center(face) - _vertical_center(bands[-1][-1]) > gap_factor * face_h:
            bands.append([face])
        else:
            bands[-1].append(face)

    rows: list[list[MatchResult]] = []
    for band in bands:
        rows.extend(_split_band(band, face_h))

    rows.sort(key=lambda row: sum(_vertical_center(f) for f in row) / len(row))
    for row in rows:
        row.sort(key=_horizontal_center)
    return rows


def row_labels(num_rows: int) -> list[str]:
    """Human labels for `num_rows` rows, top (back) to bottom (front)."""
    if num_rows <= 1:
        return ["L to R"]
    if num_rows == 2:
        return ["Back Row", "Front Row"]
    if num_rows == 3:
        return ["Back Row", "Middle Row", "Front Row"]
    # More than 3 rows: keep Back/Front anchors, number the middle ones.
    labels = ["Back Row"]
    labels += [f"Row {i}" for i in range(2, num_rows)]
    labels.append("Front Row")
    return labels


def format_roster(
    results: list[MatchResult],
    gap_factor: float = COARSE_GAP_FACTOR,
    include_unidentified: bool = True,
) -> str:
    """Build the final roster text from match results."""
    rows = group_into_rows(results, gap_factor=gap_factor)
    if not rows:
        return "No faces detected."

    labels = row_labels(len(rows))
    lines: list[str] = []
    for label, row in zip(labels, rows):
        names = [
            (r.name if r.identified else UNIDENTIFIED_LABEL)
            for r in row
            if r.identified or include_unidentified
        ]
        if not names:
            continue
        prefix = "L to R" if label == "L to R" else f"{label} L to R"
        lines.append(f"{prefix}: " + ", ".join(names))
    return "\n".join(lines)
