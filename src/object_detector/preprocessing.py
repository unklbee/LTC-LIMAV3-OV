import cv2
import numpy as np
from typing import Tuple

def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int],
    color: Tuple[int, int, int] = (114,114,114),
    auto: bool = True,
    scaleup: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    h0, w0 = img.shape[:2]
    r = min(new_shape[0]/h0, new_shape[1]/w0)
    if not scaleup:
        r = min(r, 1.0)
    new_unpadded = (int(round(w0*r)), int(round(h0*r)))
    dw, dh = new_shape[1]-new_unpadded[0], new_shape[0]-new_unpadded[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw, dh = dw/2, dh/2

    if new_unpadded != (w0, h0):
        img = cv2.resize(img, new_unpadded, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    # HWC → CHW, scale [0,1]
    img = img.transpose(2,0,1)[None].astype(np.float32) / 255.0
    return img, r, (dw, dh)
