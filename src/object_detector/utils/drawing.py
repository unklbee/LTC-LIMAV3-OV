# src/object_detector/utils/drawing.py
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List


def draw_roi_and_line(
    frame: np.ndarray,
    roi_poly: np.ndarray,
    line_pts: Tuple[Tuple[int,int], Tuple[int,int]]
) -> np.ndarray:
    """Overlay ROI polygon and counting line on frame."""
    if roi_poly is not None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [roi_poly], (128, 0, 128))
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    if line_pts is not None:
        (x1, y1), (x2, y2) = line_pts
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

def draw_detections(
    img: np.ndarray,
    dets: np.ndarray,
    ratio: float,
    dwdh: Tuple[float, float],
    names: list,
    colors: list
) -> np.ndarray:
    """Draw raw detection boxes (no tracking) on frame."""
    dw, dh = dwdh
    for *_, x0p, y0p, x1p, y1p, cls_id, conf in dets:
        x0 = int((x0p - dw) / ratio)
        y0 = int((y0p - dh) / ratio)
        x1 = int((x1p - dw) / ratio)
        y1 = int((y1p - dh) / ratio)
        cls = int(cls_id)
        color = colors[cls % len(colors)] if cls < len(colors) else (0,255,0)
        label = f"{names[cls]} {conf:.2f}" if cls < len(names) else f"ID{cls}"
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, label, (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def draw_tracks(
    img: np.ndarray,
    tracks: np.ndarray,
    history: Dict[int, List[Tuple[int,int]]],
    track_classes: Dict[int, int],
    names: List[str],
    colors: List[Tuple[int,int,int]]
) -> np.ndarray:
    """
    Draw tracked boxes with ID labels and tails on frame.

    Args:
      img:           BGR image as NumPy array (in-place drawing).
      tracks:        array of shape (N,5+) where last col is track ID.
      history:       mapping track_id -> list of (x,y) centroids.
      track_classes: mapping track_id -> class index.
      names:         list of class names.
      colors:        list of (B,G,R) tuples.

    Returns:
      img with overlays.
    """
    tid_idx = tracks.shape[1] - 1

    for tr in tracks:
        tid = int(tr[tid_idx])

        # Safely pick and coerce a single color tuple
        raw_color = colors[tid % len(colors)]
        try:
            # force ints and exactly 3 elements
            color = (int(raw_color[0]), int(raw_color[1]), int(raw_color[2]))
        except Exception:
            # fallback to white if malformed
            color = (255, 255, 255)

        # 1) Draw tail
        pts = history.get(tid, [])
        if len(pts) > 1:
            # convert to Nx1x2 int32 array for polylines
            arr = np.array(pts, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(img, [arr], False, color, 2)

        # 2) Draw bounding box
        x0, y0, x1, y1 = map(int, tr[:4])
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        # 3) Draw label
        cls = track_classes.get(tid)
        if cls is not None and 0 <= cls < len(names):
            label = f"{names[cls]}:{tid}"
        else:
            label = f"ID{tid}"
        # white text for readability
        cv2.putText(
            img, label,
            (x0, y0 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    return img