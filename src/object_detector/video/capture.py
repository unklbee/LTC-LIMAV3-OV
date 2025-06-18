import cv2

def create_capture(source):
    cap = cv2.VideoCapture(source, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source!r}")
    return cap
