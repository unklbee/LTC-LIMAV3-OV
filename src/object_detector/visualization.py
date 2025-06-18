import cv2
import numpy as np

def draw_detections(
    img: np.ndarray,
    detections: np.ndarray,
    ratio: float,
    dwdh: tuple[float, float],
    names: list[str],
    colors: list[tuple[int,int,int]],
    conf_thres: float
) -> np.ndarray:
    if detections.size == 0:
        return img

    dw, dh = dwdh
    out = img.copy()
    for det in detections:
        _, x0,y0,x1,y1, cls_f, score = det
        cls = int(cls_f)
        x0 = int(round((x0 - dw) / ratio))
        y0 = int(round((y0 - dh) / ratio))
        x1 = int(round((x1 - dw) / ratio))
        y1 = int(round((y1 - dh) / ratio))

        color = colors[cls]
        cv2.rectangle(out, (x0,y0), (x1,y1), color, 2)

        label = f"{names[cls]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(out, (x0, y0-th-4), (x0+tw, y0), color, -1)
        cv2.putText(
            out, label, (x0, y0-2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1
        )
    return out
