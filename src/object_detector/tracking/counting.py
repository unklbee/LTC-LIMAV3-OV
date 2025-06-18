from typing import Dict, Set, Tuple, List
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# util
# ──────────────────────────────────────────────────────────────────────
def _side_of_line(pt: Tuple[int, int], line: Tuple[int, int, int, int]) -> int:
    x, y         = pt
    x1, y1, x2, y2 = line
    v = (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
    return 0 if v == 0 else (1 if v > 0 else -1)

# ──────────────────────────────────────────────────────────────────────
def update_count_tracks(
        tracks: np.ndarray,
        line  : Tuple[int, int, int, int],
        roi_poly,                              # tak dipakai di sini
        counts: Dict[int, int],
        track_sides: Dict[int, int],
        counted_tracks: Set[int],
        track_classes: Dict[int, int],
        names: List[str],
        _on_count                              # dibiarkan untuk kompatibilitas
) -> None:
    """Tambah counts bila centroid menyilang line."""
    if tracks.size == 0:
        return

    tid_idx = tracks.shape[1] - 1
    for tr in tracks:
        tid = int(tr[tid_idx])
        cx  = int((tr[0] + tr[2]) / 2)
        cy  = int((tr[1] + tr[3]) / 2)

        side = _side_of_line((cx, cy), line)
        if side == 0:
            continue

        prev = track_sides.get(tid)
        track_sides[tid] = side
        if prev is None or prev == side or tid in counted_tracks:
            continue

        cls_id = track_classes.get(tid)
        if cls_id is None:
            continue

        counts[cls_id] = counts.get(cls_id, 0) + 1
        counted_tracks.add(tid)
