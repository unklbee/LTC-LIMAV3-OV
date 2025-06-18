# src/object_detector/tests/utils/test_geometry.py
import numpy as np
import cv2
from object_detector.utils.geometry import (
    filter_by_roi, denormalize_boxes, filter_classes
)

def test_filter_classes_empty():
    dets = np.zeros((0,7))
    out = filter_classes(dets, {1,2})
    assert out.shape[0] == 0

def test_filter_classes_basic():
    dets = np.array([[0,0,1,1, 1, 0, 0.9], [0,0,1,1, 3, 0, 0.8]])
    out = filter_classes(dets, {1})
    assert len(out) == 1 and out[0,4] == 1

def test_denormalize_boxes_clamp():
    # buat dummy dets
    dets = np.array([[10,10,110,110, 1,0,0.5]])
    ratio, dwdh = 0.5, (0,0)
    boxes = denormalize_boxes(dets, ratio, dwdh, frame_shape=(100,100))
    x0,y0,x1,y1,_ = boxes[0]
    assert x0 >= 0 and y0 >= 0 and x1 <= 99 and y1 <= 99

def test_filter_by_roi_inside_outside():
    # segitiga kecil ROI
    roi = np.array([[0,0],[100,0],[50,50]], dtype=np.int32).reshape(-1,1,2)
    # det centroid di dalam
    dets = np.array([[25,25,25,25, 0,0,0.9]])
    out = filter_by_roi(dets, roi, ratio=1, dwdh=(0,0))
    assert len(out) == 1
    # det centroid di luar
    dets2 = np.array([[150,150,150,150, 0,0,0.9]])
    out2 = filter_by_roi(dets2, roi, ratio=1, dwdh=(0,0))
    assert len(out2) == 0
