# Directory: src/object_detector/gui/coords.py
import numpy as np
from typing import Any, List
from PySide6.QtCore import QPointF


def serializable_to_pts(data: Any) -> List[QPointF]:
    """
    Convert serializable list of (x, y) pairs to list of QPointF.
    """
    pts: List[QPointF] = []
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        for x, y in arr:
            pts.append(QPointF(x, y))
    return pts