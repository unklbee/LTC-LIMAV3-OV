# src/object_detector/tracking/history.py

from collections import deque
import numpy as np

def update_history(track_history: dict[int, deque], tracks: np.ndarray, max_len: int):
    """Append centroid to history deque per track_id."""
    if tracks.ndim == 2:
        tid_idx = tracks.shape[1] - 1
        for tr in tracks:
            tid = int(tr[tid_idx])
            cx, cy = int((tr[0] + tr[2]) / 2), int((tr[1] + tr[3]) / 2)
            hist = track_history.setdefault(tid, deque(maxlen=max_len))
            hist.append((cx, cy))
