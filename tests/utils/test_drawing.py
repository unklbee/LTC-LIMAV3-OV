# src/object_detector/tests/utils/test_drawing.py
import numpy as np
from object_detector.utils.drawing import draw_detections, draw_tracks

def test_draw_detections_runs():
    img = np.zeros((200,200,3), np.uint8)
    dets = np.array([[10,10, 50,50, 1, 0.95]])
    out = draw_detections(img.copy(), dets, ratio=1, dwdh=(0,0), names=['a','b'], colors=[(255,0,0),(0,255,0)])
    assert out.shape == img.shape

# buat dummy tracks & history untuk draw_tracks...
