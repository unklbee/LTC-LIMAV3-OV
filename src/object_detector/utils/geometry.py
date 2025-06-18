from typing import Tuple

import cv2
import numpy as np

def box_iou(box, boxes):
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)

def denormalize_boxes(
    dets: np.ndarray,
    ratio: float,
    dwdh: Tuple[float, float],
    frame_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert padded model‐space coords → original image pixel boxes [x1,y1,x2,y2,score].
    - dets: Nx7 array [x0p,y0p,x1p,y1p,cls,conf]
    - ratio, dwdh: from letterbox
    - frame_shape: (height, width) of original frame
    """
    dw, dh = dwdh
    h, w = frame_shape
    out = []
    for *_, x0p, y0p, x1p, y1p, cls_id, conf in dets:
        # undo padding & scaling
        x0 = (x0p - dw) / ratio
        y0 = (y0p - dh) / ratio
        x1 = (x1p - dw) / ratio
        y1 = (y1p - dh) / ratio

        # clamp to image bounds
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w - 1, x1), min(h - 1, y1)

        out.append([x0, y0, x1, y1, float(conf)])

    return np.array(out, dtype=float)

def filter_by_roi(
    dets: np.ndarray,
    roi_poly: np.ndarray,
    ratio: float,
    dwdh: Tuple[float, float]
) -> np.ndarray:
    """
    Keep only detections whose centroids fall inside roi_poly.
    roi_poly must be an Nx1x2 int32 numpy array, or None.
    """
    if roi_poly is None or dets.size == 0:
        return dets

    dw, dh = dwdh
    contour = roi_poly.reshape(-1, 2)
    mask = []
    for *_, x0p, y0p, x1p, y1p, _, _ in dets:
        cx = ((x0p + x1p) / 2 - dw) / ratio
        cy = ((y0p + y1p) / 2 - dh) / ratio
        mask.append(cv2.pointPolygonTest(contour, (int(cx), int(cy)), False) >= 0)
    return dets[np.array(mask)]

def filter_classes(dets: np.ndarray, allowed_ids: set) -> np.ndarray:
    """Keep only detections whose class_id is in allowed_ids."""
    if dets.size == 0:
        return dets
    cls_ids = dets[:, 5].astype(int)
    mask = np.isin(cls_ids, list(allowed_ids))
    return dets[mask]