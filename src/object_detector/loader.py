import cv2
from pathlib import Path
from typing import Tuple, Optional

def load_source(source: str, img_exts: set) -> Tuple[bool, int, Optional, Optional]:
    if source == '0':
        src = 0
        is_image = False
    else:
        src = source
        ext = Path(source).suffix.lower().lstrip('.')
        is_image = ext in img_exts
    if is_image:
        frame = cv2.imread(source)
        return True, 0, frame, None
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {source}")
    wait = int(round(1000 / cap.get(cv2.CAP_PROP_FPS)))
    return False, wait, None, cap